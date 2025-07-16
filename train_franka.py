#!/usr/bin/env python3
"""
Franka Dataset Training Pipeline

This script provides a complete pipeline for training diffusion policy on the Franka dataset.

Usage:
1. First convert your raw data to replay buffer format:
   python diffusion_policy/scripts/Franka_data_conversion.py -i /path/to/your/raw/data -o data/Franka/processed.zarr.zip

2. Then train the model:
   python train_Franka.py

Make sure to adjust the configuration parameters in:
- diffusion_policy/config/task/Franka_multi_camera.yaml
- diffusion_policy/config/train_Franka_diffusion_unet_hybrid_workspace.yaml

Key parameters to adjust based on your setup:
- joint_dim: Number of joint dimensions for your robot
- action_dim: Number of action dimensions 
- image_shape: Resolution of your cameras [C, H, W]
- dataset_path: Path to your processed dataset
- Device settings (CUDA/CPU)
- Batch size based on your GPU memory
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import subprocess
import pathlib


def convert_data(input_path, output_path, resolution="240x320"):
    """Convert raw Franka data to replay buffer format."""
    print(f"Converting data from {input_path} to {output_path}")
    
    cmd = [
        "python", "diffusion_policy/scripts/Franka_data_conversion.py",
        "-i", str(input_path),
        "-o", str(output_path),
        "-r", resolution
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during data conversion: {result.stderr}")
        return False
    
    print("Data conversion completed successfully!")
    return True


def train_model(config_name="train_Franka_diffusion_unet_hybrid_workspace"):
    """Train the diffusion policy model."""
    print(f"Starting training with config: {config_name}")
    
    cmd = [
        "python", f"diffusion_policy/workspace/{config_name}.py"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Franka Dataset Training Pipeline")
    parser.add_argument("--mode", choices=["convert", "train", "all"], required=True,
                       help="Mode: convert data, train model, or do both")
    parser.add_argument("--input", type=str, 
                       help="Input raw data directory (required for convert mode)")
    parser.add_argument("--output", type=str, default="data/Franka/processed.zarr.zip",
                       help="Output processed data path")
    parser.add_argument("--resolution", type=str, default="240x320",
                       help="Output image resolution (HxW)")
    parser.add_argument("--config", type=str, default="train_Franka_diffusion_unet_hybrid_workspace",
                       help="Training config name")
    
    args = parser.parse_args()
    
    if args.mode in ["convert", "all"]:
        if not args.input:
            print("Error: --input is required for convert mode")
            return 1
        
        input_path = pathlib.Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            return 1
        
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = convert_data(input_path, output_path, args.resolution)
        if not success:
            return 1
    
    if args.mode in ["train", "all"]:
        output_path = pathlib.Path(args.output)
        if not output_path.exists():
            print(f"Error: Processed data {output_path} does not exist. Run conversion first.")
            return 1
        
        success = train_model(args.config)
        if not success:
            return 1
    
    print("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
