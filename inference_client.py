#!/usr/bin/env python3

import socket
import pickle
import cv2
import numpy as np
import threading
import time
import json
import argparse
import os
import sys
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
# Import robot components
from franky import *

import pathlib

# Add the project path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from intelrealsense import IntelRealSenseCamera, IntelRealSenseCameraConfig
    

FREQUENCY = 10.0  # Inference frequency (Hz)

class FrankaDiffusionClient:
    def __init__(self, 
                 cameras_config_path: str,
                 server_ip: str,
                 server_port: int = 5000,
                 robot_ip: str = "172.16.0.2",
                 max_history_length: int = 2):
        
        self.cameras_config_path = cameras_config_path
        self.server_ip = server_ip
        self.server_port = server_port
        self.robot_ip = robot_ip
        self.max_history_length = max_history_length
        
        # Robot state
        self.joint_state = None
        self.lock = threading.Lock()
        
        # Observation history for model input
        self.observation_history = []
        
        # Initialize Franka robot
        self.setup_robot()
        
        # Load camera extrinsics and setup cameras
        self.load_camera_config()
        self.setup_cameras()
        
        print("Franka Diffusion Policy client started.")
        
    def setup_robot(self):
        """Initialize connection to Franka robot using Franky"""
        try:
            self.robot = Robot(self.robot_ip)
            print(f"Connected to Franka robot at {self.robot_ip}")
            self.gripper = Gripper(self.robot)
            print("Gripper initialized")
        except Exception as e:
            print(f"Failed to connect to Franka robot: {e}")
            raise
            
    def get_robot_state(self):
        """Get current robot joint states"""
        try:
            # Get joint positions
            robot_state = self.robot.state
            joint_positions = robot_state.q  # Joint positions [7]
            
            gripper_width = self.gripper.width/2
            
            with self.lock:
                # Combine joint states and gripper state to match training format [8]
                self.joint_state = joint_positions.tolist() + [gripper_width]
                
            return True
            
        except Exception as e:
            print(f"Failed to read robot state: {e}")
            return False
        
    def load_camera_config(self):
        """Load camera configuration (intrinsics and extrinsics) from JSON file"""
        try:
            with open(self.cameras_config_path, 'r') as f:
                camera_data = json.load(f)
            
            self.extrinsics = {}
            self.camera_intrinsics = {}
            
            for camera_name, camera_config in camera_data.items():
                # Load extrinsics (transformation from camera to base frame)
                if 'extrinsics' in camera_config:
                    extrinsics = camera_config['extrinsics']
                    if 'SE3' in extrinsics:
                        # Use SE3 4x4 matrix directly
                        transform = np.array(extrinsics['SE3'])
                        self.extrinsics[camera_name] = transform
                    else:
                        # Fallback: try to load as [x, y, z, qx, qy, qz, qw]
                        extrinsic_data = extrinsics
                        if isinstance(extrinsic_data, list) and len(extrinsic_data) == 7:
                            translation = np.array(extrinsic_data[:3])
                            quaternion = np.array(extrinsic_data[3:7])  # [qx, qy, qz, qw]
                            rotation_matrix = R.from_quat(quaternion).as_matrix()
                            transform = np.eye(4)
                            transform[:3, :3] = rotation_matrix
                            transform[:3, 3] = translation
                            self.extrinsics[camera_name] = transform
                
                # Load intrinsics
                if 'intrinsics' in camera_config:
                    intrinsics = camera_config['intrinsics']
                    self.camera_intrinsics[camera_name] = {
                        'fx': intrinsics['fx'],
                        'fy': intrinsics['fy'],
                        'cx': intrinsics['cx'],
                        'cy': intrinsics['cy']
                    }
            
            print(f"Loaded camera config for {len(self.extrinsics)} cameras")
            
        except Exception as e:
            print(f"Failed to load camera config: {e}")
            # Use default extrinsics if config loading fails
            self.extrinsics = {}
            self.camera_intrinsics = {}
        
    def setup_cameras(self):
        """Setup Intel RealSense cameras - MATCH RECORDING SCRIPT EXACTLY"""
        self.camera_configs = {
            "left_camera": IntelRealSenseCameraConfig(
                serial_number="142422250807",
                fps=30, 
                width=640, 
                height=480,
                use_depth=False,  # Only RGB for diffusion policy
                mock=False
            ),
            "wrist_camera": IntelRealSenseCameraConfig(
                serial_number="213622250811",  # Update with your wrist camera serial
                fps=30, 
                width=640, 
                height=480,
                use_depth=False,
                mock=False
            ),
            "right_camera": IntelRealSenseCameraConfig(
                serial_number="025522060843",
                fps=30, 
                width=640, 
                height=480,
                use_depth=False,
                mock=False
            ),
        }
        
        self.cameras = {}
        for name, cfg in self.camera_configs.items():
            try:
                self.cameras[name] = IntelRealSenseCamera(cfg)
                self.cameras[name].connect()
                self.cameras[name].async_read()  # Start async reading
                print(f"Connected to camera {name} (serial: {cfg.serial_number})")
            except Exception as e:
                print(f"Failed to connect camera {name}: {e}")
                
    def get_current_images(self):
        """Get current images from all cameras - MATCH TRAINING FORMAT"""
        images = {}
        
        # Get images from all cameras (left, wrist, right)
        target_cameras = ["left_camera", "wrist_camera", "right_camera"]
        
        for name in target_cameras:
            if name in self.cameras:
                try:
                    # Read color image from camera
                    color_img = self.cameras[name].async_read()
                    
                    if color_img is not None:
                        # Convert BGR to RGB (OpenCV uses BGR by default)
                        if len(color_img.shape) == 3 and color_img.shape[2] == 3:
                            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize to match training format (480x640)
                        color_img = cv2.resize(color_img, (640, 480))
                        
                        # Normalize to [0, 1] range
                        color_img = color_img.astype(np.float32) / 255.0
                        
                        # Convert to CHW format for PyTorch (channels first)
                        color_img = np.transpose(color_img, (2, 0, 1))  # HWC -> CHW
                        
                        images[name] = color_img
                        
                except Exception as e:
                    print(f"Failed to read from camera {name}: {e}")
                    # Create dummy image if camera fails
                    dummy_img = np.zeros((3, 480, 640), dtype=np.float32)
                    images[name] = dummy_img
            else:
                print(f"Camera {name} not available, using dummy image")
                # Create dummy image if camera not available
                dummy_img = np.zeros((3, 480, 640), dtype=np.float32)
                images[name] = dummy_img
                
        return images
        
    def create_observation(self):
        """Create observation dictionary from current sensor data"""
        with self.lock:
            if self.joint_state is None:
                return None
                
            # Get current joint states [8] (7 joints + 1 gripper)
            joint_states = np.array(self.joint_state, dtype=np.float32)
            
        # Get images from all cameras
        images = self.get_current_images()
        
        # Ensure we have all required cameras
        required_cameras = ["left_camera", "wrist_camera", "right_camera"]
        for cam_name in required_cameras:
            if cam_name not in images:
                print(f"Missing camera {cam_name}, using dummy image")
                dummy_img = np.zeros((3, 480, 640), dtype=np.float32)
                images[cam_name] = dummy_img
        
        # Create observation dictionary matching training format
        obs = {
            'left_camera': images['left_camera'],    # [3, 480, 640]
            'wrist_camera': images['wrist_camera'],  # [3, 480, 640]
            'right_camera': images['right_camera'],  # [3, 480, 640]
            'joint_states': joint_states             # [8]
        }
        
        return obs
        
    def update_observation_history(self, obs):
        """Update observation history with new observation"""
        self.observation_history.append(obs)
        
        # Keep only the required number of observations
        if len(self.observation_history) > self.max_history_length:
            self.observation_history = self.observation_history[-self.max_history_length:]
            
    def create_model_input(self):
        """Create model input from observation history"""
        if len(self.observation_history) == 0:
            return None
            
        # Pad history if needed - use exactly the required history length
        required_length = self.max_history_length
        current_length = len(self.observation_history)
        
        if current_length < required_length:
            # Repeat the first observation to pad
            padding_needed = required_length - current_length
            first_obs = self.observation_history[0]
            padded_history = [first_obs] * padding_needed + self.observation_history
        else:
            padded_history = self.observation_history[-required_length:]
            
        # Stack observations for each modality
        left_camera_stack = []
        wrist_camera_stack = []
        right_camera_stack = []
        joint_states_stack = []
        
        for obs in padded_history:
            left_camera_stack.append(obs['left_camera'])
            wrist_camera_stack.append(obs['wrist_camera'])
            right_camera_stack.append(obs['right_camera'])
            joint_states_stack.append(obs['joint_states'])
            
        # Convert to numpy arrays - match training data format exactly
        left_camera_array = np.stack(left_camera_stack)      # [T, 3, 480, 640]
        wrist_camera_array = np.stack(wrist_camera_stack)    # [T, 3, 480, 640]
        right_camera_array = np.stack(right_camera_stack)    # [T, 3, 480, 640]
        joint_states_array = np.stack(joint_states_stack)    # [T, 8]
        
        # Add batch dimension to match training format
        left_camera_array = np.expand_dims(left_camera_array, axis=0)    # [1, T, 3, 480, 640]
        wrist_camera_array = np.expand_dims(wrist_camera_array, axis=0)  # [1, T, 3, 480, 640]
        right_camera_array = np.expand_dims(right_camera_array, axis=0)  # [1, T, 3, 480, 640]
        joint_states_array = np.expand_dims(joint_states_array, axis=0)  # [1, T, 8]
        
        print(f"Observation shapes - left_camera: {left_camera_array.shape}, "
              f"wrist_camera: {wrist_camera_array.shape}, "
              f"right_camera: {right_camera_array.shape}, "
              f"joint_states: {joint_states_array.shape}")
              
        obs_dict = {
            'left_camera': left_camera_array,
            'wrist_camera': wrist_camera_array,
            'right_camera': right_camera_array,
            'joint_states': joint_states_array
        }
        
        return obs_dict
        
    def send_to_server(self, obs_dict):
        """Send observation to server and receive actions"""
        try:
            # Serialize observation
            data = pickle.dumps(obs_dict) + b"<END>"
            
            # Send to server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10.0)  # 10 second timeout
            s.connect((self.server_ip, self.server_port))
            s.sendall(data)
            
            # Receive response
            response = b""
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                response += packet
                if response.endswith(b"<END>"):
                    response = response[:-5]
                    break
            
            actions = pickle.loads(response)
            s.close()
            
            return actions
            
        except Exception as e:
            print(f"Failed to communicate with server: {e}")
            return None
            
    def send_actions(self, actions):
        """Send predicted actions to Franka robot"""
        try:
            if actions is None:
                print("No actions to send")
                return
                
            # Take the first action from the predicted sequence
            action = actions[0, 0, :]  # [action_dim] - first batch, first timestep
            
            # Ensure we have the expected action dimension
            if len(action) < 8:
                print(f"Warning: action has {len(action)} dimensions, expected 8")
                return
                
            joint_actions = action[:7]  # First 7 elements are joint actions
            gripper_action = action[7] if len(action) > 7 else 0.0  # Last element is gripper action
            
            # Send joint commands to robot
            try:
                # Move to joint position with appropriate dynamics
                print(f"Moving robot to joint positions: {joint_actions.tolist()}")
                self.robot.move(JointMotion(joint_actions.tolist(), relative_dynamics_factor=0.05))
                
                # Control gripper (if available)
                try:
                    if hasattr(self.robot, 'gripper'):
                        # Scale gripper action to appropriate range (0-0.08m typically)
                        gripper_width = max(0.0, min(0.08, gripper_action))
                        self.robot.gripper.move(gripper_width)
                except Exception as e:
                    print(f"Failed to control gripper: {e}")
                
                print(f"Sent actions - Joints: {joint_actions}, Gripper: {gripper_action}")
                
            except Exception as e:
                print(f"Failed to send joint commands: {e}")
                
        except Exception as e:
            print(f"Failed to process actions: {e}")
            import traceback
            traceback.print_exc()
            
    def save_images_debug(self, images, filename_prefix=None):
        """Save images to file for debugging/visualization"""
        try:
            if filename_prefix is None:
                filename_prefix = f"images_{int(time.time())}"
            
            for cam_name, img in images.items():
                # Convert from CHW to HWC and denormalize
                img_hwc = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                img_uint8 = (img_hwc * 255).astype(np.uint8)
                
                filename = f"{filename_prefix}_{cam_name}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                print(f"Saved {cam_name} image to {filename}")
                
        except Exception as e:
            print(f"Failed to save images: {e}")
            
    def inference_step(self):
        """Main inference loop"""
        try:
            # Read current robot state
            if not self.get_robot_state():
                print("Failed to get robot state, skipping inference step")
                return
                
            # Create observation from current sensor data
            obs = self.create_observation()
            if obs is None:
                print("Failed to create observation, skipping inference step")
                return
                
            # Validate observation shapes
            print(f"Observation shapes - "
                  f"left_camera: {obs['left_camera'].shape}, "
                  f"wrist_camera: {obs['wrist_camera'].shape}, "
                  f"right_camera: {obs['right_camera'].shape}, "
                  f"joint_states: {obs['joint_states'].shape}")
                
            # Update observation history
            self.update_observation_history(obs)
            
            # Create model input
            obs_dict = self.create_model_input()
            if obs_dict is None:
                print("Failed to create model input")
                return
                
            # Send to server for inference
            actions = self.send_to_server(obs_dict)
            if actions is None:
                print("No actions received from server")
                return
            print(f"Received actions: {actions.shape}")
            
            # Send actions to robot
            self.send_actions(actions)
            
            # Save images for debugging (optional)
            # self.save_images_debug(obs)
            
        except Exception as e:
            print(f"Error in inference step: {e}")
            import traceback
            traceback.print_exc()
            
    def run(self):
        """Run the inference loop"""
        try:
            print("Starting inference loop...")
            
            while True:
                start_time = time.time()
                
                # Run inference step
                self.inference_step()
                
                # Maintain target frequency
                elapsed_time = time.time() - start_time
                print(f"Elapsed time for step: {elapsed_time:.3f} seconds")
                sleep_time = max(0, (1.0 / FREQUENCY) - elapsed_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("Stopping inference loop...")
        except Exception as e:
            print(f"Error in inference loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Disconnect cameras
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception as e:
                    print(f"Failed to disconnect camera: {e}")
                    
            # Stop robot (optional - robot will maintain last position)
            # self.robot.stop()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Franka Diffusion Policy Client")
    parser.add_argument("--cameras_config_path", type=str, required=True,
                        help="Path to cameras.json file with intrinsics and extrinsics")
    parser.add_argument("--server_ip", type=str, required=True,
                        help="IP address of the inference server")
    parser.add_argument("--server_port", type=int, default=5000,
                        help="Port of the inference server")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2",
                        help="IP address of the Franka robot")
    parser.add_argument("--max_history_length", type=int, default=2,
                        help="Maximum length of observation history")
    
    args = parser.parse_args()
    
    # Create and run the client
    client = FrankaDiffusionClient(
        cameras_config_path=args.cameras_config_path,
        server_ip=args.server_ip,
        server_port=args.server_port,
        robot_ip=args.robot_ip,
        max_history_length=args.max_history_length
    )
    
    # Run the inference loop
    client.run()


if __name__ == "__main__":
    main()
