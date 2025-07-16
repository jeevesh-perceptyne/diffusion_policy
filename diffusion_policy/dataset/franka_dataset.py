from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class FrankaDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
        ):
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta for caching
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            
            if dataset_path.endswith('.zarr') or dataset_path.endswith('.zarr.zip'):
                cache_zarr_path = dataset_path.replace('.zarr', f'_{shape_meta_hash}_processed.zarr')
                cache_zarr_path = cache_zarr_path.replace('.zip', '')
            else:
                cache_zarr_path = os.path.join(dataset_path, f'{shape_meta_hash}.zarr.zip')
            
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exist
                    try:
                        print('Cache does not exist. Creating!')
                        # For pre-processed zarr files, load and copy to memory
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            # Load directly from disk without caching to save memory
            print(f'Loading dataset directly from disk: {dataset_path}')
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=None  # Load from disk directly for memory efficiency
            )
        
        # Parse shape meta to identify RGB and lowdim keys
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data


def _get_replay_buffer(dataset_path, shape_meta, store):
    """Load replay buffer from zarr file."""
    
    if store is None:
        # Load directly from disk to save memory - for large datasets
        print(f"Loading dataset directly from disk without copying to memory")
        replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    else:
        # Load and copy to specified store (memory or other)
        if dataset_path.endswith('.zarr.zip'):
            # Load from zip store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, store=store)
        elif dataset_path.endswith('.zarr') or os.path.isdir(dataset_path):
            # Load from directory store
            with zarr.DirectoryStore(dataset_path) as dir_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=dir_store, store=store)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    return replay_buffer


def test():
    """Test function for debugging."""
    shape_meta = {
        'obs': {
            'left_camera': {
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'wrist_camera': {
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'right_camera': {
                'shape': [3, 240, 320],
                'type': 'rgb'
            },
            'joint_states': {
                'shape': [8],  # Adjust based on your robot's DOF
                'type': 'low_dim'
            }
        },
        'action': {
            'shape': [8]  # Adjust based on your action space
        }
    }
    
    dataset = FrankaDataset(
        shape_meta=shape_meta,
        dataset_path='data/Franka/processed.zarr.zip',
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        use_cache=True,
        val_ratio=0.1
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting an item
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Obs keys:", sample['obs'].keys())
    
    for key, value in sample['obs'].items():
        print(f"  {key}: {value.shape}")
    print(f"Action: {sample['action'].shape}")
    
    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("Normalizer keys:", list(normalizer.keys()))


if __name__ == '__main__':
    test()
