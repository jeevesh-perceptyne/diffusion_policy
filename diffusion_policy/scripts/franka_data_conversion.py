#!/usr/bin/env python3

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click #type:ignore
import pathlib
import zarr #type:ignore
import cv2 #type:ignore
import numpy as np #type:ignore
import multiprocessing
import concurrent.futures
from tqdm import tqdm #type:ignore
import av #type:ignore
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()


def Franka_data_to_replay_buffer(
        dataset_path: str, 
        out_store: zarr.storage.BaseStore = None,
        out_resolution: tuple = (480, 640),  # (height, width)
        n_encoding_threads: int = None,
        max_inflight_tasks: int = None,
        verify_read: bool = True
) -> ReplayBuffer:
    """
    Convert Franka dataset to ReplayBuffer format.
    
    Expected data structure:
    dataset_path/
        episode_001/
            episode_data.npz
            left_camera.mp4
            wrist_camera.mp4  
            right_camera.mp4
            left_depth_images/
            wrist_depth_images/
            right_depth_images/
        episode_002/
            ...
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_encoding_threads is None:
        n_encoding_threads = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_encoding_threads * 5
    
    dataset_path = pathlib.Path(dataset_path)
    episode_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {dataset_path}")
    
    print(f"Found {len(episode_dirs)} episodes")
    
    # Create zarr replay buffer directly (memory efficient)
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=out_store)
    
    # Set up compression for images
    from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k
    image_compressor = Jpeg2k(level=50)
    
    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        episode_data_path = episode_dir / "episode_data.npz"
        
        if not episode_data_path.exists():
            print(f"Warning: {episode_data_path} not found, skipping episode")
            continue
            
        # Load episode data
        episode_data = np.load(episode_data_path)
        joint_states = episode_data['joint_states'].astype(np.float32)
        gripper_states = episode_data['gripper_states'].astype(np.float32)
        obs = np.concatenate([joint_states, gripper_states], axis=-1)
        joint_actions = episode_data['gello_joint_states'].astype(np.float32)
        gripper_actions = episode_data['gello_gripper_percent'].astype(np.float32).reshape(-1, 1)  
        actions = np.concatenate([joint_actions, gripper_actions], axis=-1)
        
        episode_length = len(obs)
        print(f"Episode {episode_dir.name}: {episode_length} steps")
        
        # Prepare episode data dict
        episode_dict = {
            'joint_states': obs,
            'action': actions
        }
        
        # Process RGB videos from 3 cameras
        camera_names = ['left_camera', 'wrist_camera', 'right_camera']
        
        for camera_name in camera_names:
            video_path = episode_dir / f"{camera_name}.mp4"
            if video_path.exists():
                frames = extract_frames_from_video(video_path, episode_length, out_resolution)
                episode_dict[camera_name] = frames
            else:
                print(f"Warning: {video_path} not found")
        
        # Define compressors separately for each data type after all data is prepared
        compressors = {}
        for key, value in episode_dict.items():
            if key in camera_names and key in episode_dict:
                # Apply image compression only to camera data (4D arrays with RGB)
                if len(value.shape) == 4 and value.shape[-1] == 3:  # (T, H, W, 3) RGB videos
                    # Don't use Jpeg2k for now - it seems to have compatibility issues
                    compressors[key] = 'default'  # Use default compression instead
                    print(f"  Using default compression for {key}: shape {value.shape}")
                else:
                    compressors[key] = 'default'
                    print(f"  Using default compression for {key}: shape {value.shape}")
            else:
                # Use default compression for lowdim data (joint_states, action)
                compressors[key] = 'default'
                print(f"  Using default compression for {key}: shape {value.shape}")
        
        # Add episode directly to zarr with compression
        out_replay_buffer.add_episode(episode_dict, compressors=compressors)
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
    
    return out_replay_buffer


def extract_frames_from_video(video_path: pathlib.Path, target_length: int, out_resolution: tuple):
    """Extract frames from video and resize to target resolution (memory efficient)."""
    oh, ow = out_resolution
    
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        
        # Get video info
        vcc = stream.codec_context
        in_resolution = (vcc.height, vcc.width)
        
        # Create image transform
        image_tf = get_image_transform(
            input_res=(vcc.width, vcc.height),  # cv2 uses (width, height)
            output_res=(ow, oh),  # cv2 uses (width, height)
            bgr_to_rgb=False
        )
        
        # Pre-allocate output array for memory efficiency
        frames = np.zeros((target_length, oh, ow, 3), dtype=np.uint8)
        
        frame_count = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx >= target_length:
                break
                
            # Convert to numpy array
            img = frame.to_ndarray(format='rgb24')  # (H, W, 3)
            
            # Resize if needed
            if in_resolution != out_resolution:
                img = image_tf(img)
            
            frames[frame_count] = img
            frame_count += 1
        
        # Handle case where video is shorter than target_length
        if frame_count < target_length:
            # Duplicate last frame if video is shorter
            if frame_count > 0:
                last_frame = frames[frame_count - 1]
                for i in range(frame_count, target_length):
                    frames[i] = last_frame
            # If no frames were extracted, frames array is already zeros
        
        return frames


@click.command()
@click.option('--input', '-i', required=True, help='Input dataset directory')
@click.option('--output', '-o', required=True, help='Output zarr path')
@click.option('--resolution', '-r', default='480x640', help='Output resolution (HxW)')
@click.option('--n_encoding_threads', '-ne', default=-1, type=int, help='Number of encoding threads')
@click.option('--batch_size', '-b', default=10, type=int, help='Process episodes in batches to manage memory')
def main(input, output, resolution, n_encoding_threads, batch_size):
    """Convert Franka dataset to replay buffer format."""
    
    # Parse resolution
    h, w = tuple(int(x) for x in resolution.split('x'))
    out_resolution = (h, w)
    
    input_path = pathlib.Path(input).expanduser()
    output_path = pathlib.Path(output).expanduser()
    
    if not input_path.exists():
        raise ValueError(f"Input path {input_path} does not exist")
    
    
    # Set threading limits for better performance and memory usage
    cv2.setNumThreads(1)
    
    if n_encoding_threads <= 0:
        n_encoding_threads = min(4, multiprocessing.cpu_count())  # Limit threads to reduce memory
    
    print(f"Converting {input_path} to {output_path}")
    print(f"Target resolution: {out_resolution}")
    print(f"Encoding threads: {n_encoding_threads}")
    print(f"Batch size: {batch_size}")
    
    # Create output store
    if output_path.suffix == '.zip':
        store = zarr.ZipStore(output_path, mode='w')
    else:
        store = zarr.DirectoryStore(output_path)
    
    try:
        replay_buffer = Franka_data_to_replay_buffer(
            dataset_path=str(input_path),
            out_store=store,
            out_resolution=out_resolution,
            n_encoding_threads=n_encoding_threads
        )
        
        print(f"Successfully converted {replay_buffer.n_episodes} episodes")
        print(f"Total steps: {replay_buffer.n_steps}")
        
    finally:
        if hasattr(store, 'close'):
            store.close()


if __name__ == '__main__':
    main()
