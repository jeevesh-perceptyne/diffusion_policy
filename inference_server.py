#!/usr/bin/env python3

# Imports
import socket
import pickle
import threading
import torch
import numpy as np
import argparse
import os
import sys
import traceback
import boto3
import hydra
from omegaconf import OmegaConf
from termcolor import cprint
import pathlib

# Global frame index counter and lock
frame_idx_counter = 0
frame_idx_lock = threading.Lock()

# Add the project path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

# Import the diffusion policy components
from diffusion_policy.workspace.train_franka_diffusion_unet_hybrid_workspace import TrainFrankaDiffusionUnetHybridWorkspace

def save_debug_obs(obs_dict, prefix="debug_offline", frame_idx=0, out_dir="/tmp/diffusion_debug"):
    """Save observation tensors for debugging"""
    os.makedirs(out_dir, exist_ok=True)
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            np.save(os.path.join(out_dir, f"{prefix}_{key}_{frame_idx:04d}.npy"), value.cpu().numpy())
        elif isinstance(value, np.ndarray):
            np.save(os.path.join(out_dir, f"{prefix}_{key}_{frame_idx:04d}.npy"), value)

class DiffusionPolicyServer:
    def __init__(self, config_path, s3_bucket="pr-checkpoints", latest=True, epoch=None, 
                 device="cuda", use_ema=False, restore_checkpoint=True):
        self.device = device
        self.use_ema = use_ema
        
        # Load the model
        self.load_model(config_path, s3_bucket, latest, epoch, restore_checkpoint)
        print("Diffusion Policy Server initialized and ready for inference")
        
    def load_model(self, config_path, s3_bucket, latest, epoch, restore_checkpoint):
        """Load the Diffusion Policy model"""
        try:
            # Get checkpoint path
            if latest:
                checkpoint_name = "checkpoint_latest.pth"
            elif epoch is not None:
                checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            else:
                raise ValueError("Either latest must be True or epoch must be specified")
            
            # Load config using hydra's compose API
            config_path = os.path.abspath(config_path)
            config_dir = os.path.dirname(config_path)
            config_name = os.path.basename(config_path).replace('.yaml', '')
            
            # Make config_dir relative to current working directory
            current_dir = os.getcwd()
            config_dir_rel = os.path.relpath(config_dir, current_dir)
            
            # Initialize hydra with the relative config directory
            with hydra.initialize(config_path=config_dir_rel, version_base=None):
                self.cfg = hydra.compose(config_name=config_name)
            
            # Create workspace with model
            self.workspace = TrainFrankaDiffusionUnetHybridWorkspace(self.cfg)
            
            # Load checkpoint from S3 or local cache
            local_path = checkpoint_name
            
            if s3_bucket and s3_bucket.lower() != "null":
                cprint(f"Using S3 bucket: {s3_bucket}")
                try:
                    s3_client = boto3.client('s3')
                    if latest:
                        s3_path = "franka_diffusion_outputs/latest/checkpoint.pth"
                    else:
                        s3_path = f"franka_diffusion_outputs/epoch_{epoch}/checkpoint.pth"
                    
                    if restore_checkpoint and os.path.exists(local_path):
                        print(f"Loading checkpoint from local cache: {local_path}")
                    else:
                        print(f"Downloading checkpoint from S3: {s3_bucket}/{s3_path}")
                        s3_client.download_file(s3_bucket, s3_path, local_path)
                except Exception as e:
                    print(f"Failed to download from S3: {e}")
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(f"No checkpoint available locally or in S3")
            else:
                print(f"S3 bucket not provided, loading checkpoint locally from: {local_path}")
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Local checkpoint file not found: {local_path}")
            
            # Load checkpoint
            ckpt = torch.load(local_path, map_location=self.device)
            self.workspace.model.load_state_dict(ckpt['model_state_dict'])
            
            # Load EMA model if available
            if self.workspace.ema_model is not None and 'ema_state_dict' in ckpt:
                self.workspace.ema_model.load_state_dict(ckpt['ema_state_dict'])
            
            # Load normalizer state (essential for proper inference)
            if hasattr(self.workspace.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.workspace.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
            
            # Move models to device
            self.workspace.model.to(self.device)
            if self.workspace.ema_model is not None:
                self.workspace.ema_model.to(self.device)
                
            # Set model to eval mode
            self.workspace.model.eval()
            if self.workspace.ema_model is not None:
                self.workspace.ema_model.eval()
            
            print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
            
    def run_inference(self, obs_dict, frame_idx=None):
        """Run inference with the diffusion policy model - MATCH TRAINING EXACTLY"""
        try:
            # Print input statistics for debugging
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key} stats - shape: {value.shape}, mean: {np.mean(value):.4f}, std: {np.std(value):.4f}")
            
            # Convert numpy arrays to torch tensors if needed
            torch_obs = {}
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    torch_obs[key] = torch.from_numpy(value).float().to(self.device)
                else:
                    torch_obs[key] = value.to(self.device)
            
            # Select the appropriate model
            if self.use_ema and self.workspace.ema_model is not None:
                policy = self.workspace.ema_model
                print("Using EMA model for inference")
            else:
                policy = self.workspace.model
                print("Using main model for inference")
                
            # Run inference - pass obs_dict directly like in training
            with torch.no_grad():
                if frame_idx is not None:
                    save_debug_obs(torch_obs, prefix="debug_online", frame_idx=frame_idx)
                
                # The model expects the observation dict with proper keys
                result = policy.predict_action(torch_obs)
                pred_action = result['action_pred'] if isinstance(result, dict) else result
                
            return pred_action.cpu().numpy()
            
        except Exception as e:
            print(f"Inference failed: {e}")
            traceback.print_exc()
            return None

def handle_client(conn, server_instance):
    global frame_idx_counter
    try:
        # Receive data from client
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
            if data.endswith(b"<END>"):
                data = data[:-5]
                break

        # Deserialize observation
        obs_dict = pickle.loads(data)
        print(f"Received observation with keys: {list(obs_dict.keys())}")
        
        # Print shapes for debugging
        for key, value in obs_dict.items():
            print(f"  {key}: {value.shape}")

        # Validate input shapes - check for expected camera keys and joint_states
        expected_keys = ['left_camera', 'wrist_camera', 'right_camera', 'joint_states']
        missing_keys = [key for key in expected_keys if key not in obs_dict]
        if missing_keys:
            raise ValueError(f"Missing observation keys: {missing_keys}")

        # Validate shapes
        for cam_key in ['left_camera', 'wrist_camera', 'right_camera']:
            cam_shape = obs_dict[cam_key].shape
            if len(cam_shape) != 5:  # [B, T, C, H, W]
                raise ValueError(f"{cam_key} should have 5 dimensions [B,T,C,H,W], got {cam_shape}")
            if cam_shape[2] != 3:  # RGB channels
                raise ValueError(f"{cam_key} should have 3 channels, got {cam_shape[2]}")
            if cam_shape[3] != 480 or cam_shape[4] != 640:  # Image size
                print(f"Warning: {cam_key} has size {cam_shape[3]}x{cam_shape[4]}, expected 480x640")

        joint_shape = obs_dict['joint_states'].shape
        if len(joint_shape) != 3:  # [B, T, D]
            raise ValueError(f"joint_states should have 3 dimensions [B,T,D], got {joint_shape}")
        if joint_shape[2] != 8:  # 7 joints + 1 gripper
            raise ValueError(f"joint_states should have 8 dimensions, got {joint_shape[2]}")

        # Get and increment frame index safely
        with frame_idx_lock:
            frame_idx = frame_idx_counter
            frame_idx_counter += 1

        # Run inference with frame_idx
        actions = server_instance.run_inference(obs_dict, frame_idx=frame_idx)

        if actions is not None:
            print(f"Generated actions with shape: {actions.shape}")
            response = pickle.dumps(actions)
        else:
            print("Inference failed, sending None")
            response = pickle.dumps(None)

        # Send response back to client
        conn.sendall(response + b"<END>")

    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
        # Send error response
        try:
            response = pickle.dumps(None)
            conn.sendall(response + b"<END>")
        except:
            pass
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Inference Server")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the config file used for training")
    parser.add_argument("--s3_bucket", type=str, default="pr-checkpoints",
                        help="S3 bucket containing the checkpoint")
    parser.add_argument("--latest", action="store_true", default=True,
                        help="Use the latest checkpoint")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model for inference")
    parser.add_argument("--restore_checkpoint", action="store_true", default=True,
                        help="Restore from a local checkpoint if it exists")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host address")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port")
    
    args = parser.parse_args()
    
    # Initialize the server
    server_instance = DiffusionPolicyServer(
        config_path=args.config_path,
        s3_bucket=args.s3_bucket,
        latest=args.latest,
        epoch=args.epoch,
        device=args.device,
        use_ema=args.use_ema,
        restore_checkpoint=args.restore_checkpoint
    )
    
    # Start the socket server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)
    print(f"Server listening on {args.host}:{args.port}")
    
    try:
        while True:
            conn, addr = server.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=handle_client, args=(conn, server_instance)).start()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.close()

if __name__ == "__main__":
    main()
