if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import psutil
import gc
import boto3
from botocore.exceptions import NoCredentialsError
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.franka_dataset import FrankaDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainFrankaDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # S3 configuration
        self.s3_bucket = 'pr-checkpoints'
        try:
            self.s3_client = boto3.client('s3')
            # Test S3 connection
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            print(f"✓ S3 connection established. Using bucket: {self.s3_bucket}")
        except Exception as e:
            print(f"⚠ S3 setup failed: {e}")
            print("Checkpoints will not be saved to S3")
            self.s3_client = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.lr_scheduler = None  # will be set in run()

    def s3_ckpt_path(self, epoch=None, latest=False):
        """Generate S3 checkpoint path"""
        if latest:
            return f"franka_diffusion_outputs/latest/checkpoint.pth"
        else:
            return f"franka_diffusion_outputs/epoch_{epoch}/checkpoint.pth"

    def save_checkpoint_s3(self, epoch, latest=False):
        """Save checkpoint to S3"""
        if self.s3_client is None:
            print("⚠ S3 client not available, skipping checkpoint save")
            return
            
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }
        
        # Save normalizer state_dict for robust inference
        if hasattr(self.model, 'normalizer') and self.model.normalizer is not None:
            checkpoint['normalizer_state_dict'] = self.model.normalizer.state_dict()
            
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        local_path = f'checkpoint_tmp.pth'
        torch.save(checkpoint, local_path)
        
        if latest:
            # Save only to latest path when latest=True
            s3_path = self.s3_ckpt_path(latest=True)
        else:
            # Save only to epoch-specific path when latest=False
            s3_path = self.s3_ckpt_path(epoch=epoch)
        
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_path)
            
            # Also save config as yaml for easy inspection/inference
            config_yaml_path = 'config_tmp.yaml'
            with open(config_yaml_path, 'w') as f:
                OmegaConf.save(self.cfg, f)
            s3_config_path = s3_path.replace('checkpoint.pth', 'config.yaml')
            self.s3_client.upload_file(config_yaml_path, self.s3_bucket, s3_config_path)
            
            os.remove(local_path)
            os.remove(config_yaml_path)
            print(f"✓ Checkpoint and config saved to S3: {s3_path}")
        except Exception as e:
            print(f"⚠ Failed to save checkpoint to S3: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            if os.path.exists('config_tmp.yaml'):
                os.remove('config_tmp.yaml')

    def load_latest_checkpoint_s3(self, device):
        """Load latest checkpoint from S3"""
        if self.s3_client is None:
            print("⚠ S3 client not available, skipping checkpoint load")
            return
            
        s3_latest = self.s3_ckpt_path(latest=True)
        local_path = 'checkpoint_latest_tmp.pth'
        try:
            self.s3_client.download_file(self.s3_bucket, s3_latest, local_path)
            ckpt = torch.load(local_path, map_location=device)
            
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            
            if self.lr_scheduler and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                
            if self.ema_model and 'ema_state_dict' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_state_dict'])
                
            # Restore normalizer state_dict if present
            if hasattr(self.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
                
            self.epoch = ckpt['epoch']
            self.global_step = ckpt['global_step']
            
            # Optionally restore config
            if 'config' in ckpt:
                self.cfg = OmegaConf.create(ckpt['config'])
                
            print(f"✓ Resumed from S3 checkpoint at epoch {self.epoch}, step {self.global_step}")
            os.remove(local_path)
            
        except Exception as e:
            print(f"No S3 checkpoint found or error loading checkpoint: {e}")

    def run(self):
        # print(f"Running workspace {self.__class__.__name__} with config:\n{OmegaConf.to_yaml(self.cfg)}")
        cfg = copy.deepcopy(self.cfg)

        # Resume from latest S3 checkpoint if requested
        device = torch.device(cfg.training.device)
        if cfg.training.resume:
            self.load_latest_checkpoint_s3(device)

        # configure dataset
        dataset: FrankaDataset
        print(f"Loading dataset from {cfg.task.dataset_path}")
        
        # Monitor memory before dataset loading
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage before dataset loading: {memory_before:.1f} MB")
        
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, FrankaDataset)
        
        # Monitor memory after dataset loading
        memory_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage after dataset loading: {memory_after:.1f} MB")
        print(f"Memory increase: {memory_after - memory_before:.1f} MB")
        
        print(f"Dataset {dataset.__class__.__name__} loaded with {len(dataset)} samples")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        print(f"Training dataloader created with {len(train_dataloader)} batches")
        
        # Force garbage collection
        gc.collect()
        
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure env
        # Note: For real robot applications, replace this with actual env runner
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure ema
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for validation
        val_batch = next(iter(train_dataloader))

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if cfg.training.max_train_steps is not None:
                            if self.global_step >= cfg.training.max_train_steps:
                                break

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema and ema is not None:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            # log of last batch is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and self.global_step >= cfg.training.max_train_steps:
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                    
                # validation
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if cfg.training.max_val_steps is not None:
                                if batch_idx >= cfg.training.max_val_steps:
                                    break
                            loss = policy.compute_loss(batch)
                            val_losses.append(loss.item())
                            
                    if len(val_losses) > 0:
                        val_loss = np.mean(val_losses)
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    policy.eval()
                    
                    # For demonstration purposes, create a dummy runner log
                    # Replace this with actual environment evaluation
                    runner_log = {
                        'test/mean_score': np.random.random(),  # Dummy score
                        'test/success_rate': np.random.random()  # Dummy success rate
                    }
                    
                    policy.train()
                    step_log.update(runner_log)

                # S3 Checkpoint saving strategy:
                # 1. Save latest checkpoint every 100 epochs (overwrites previous latest)  
                if (self.epoch) % 100 == 0:
                    self.save_checkpoint_s3(epoch=self.epoch + 1, latest=True)
                
                # 2. Save numbered checkpoint every 500 epochs
                if (self.epoch + 1) % 1 == 0:
                    self.save_checkpoint_s3(epoch=self.epoch + 1, latest=False)
                
                # log of last batch is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainFrankaDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
