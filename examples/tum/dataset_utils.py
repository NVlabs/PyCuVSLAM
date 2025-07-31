#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import os
from PIL import Image
import numpy as np

def load_frame(image_path: str) -> np.ndarray:
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path)
    frame = np.array(image)

    if image.mode == 'L':
        # mono8
        if len(frame.shape) != 2:
            raise ValueError("Expected mono8 image to have 2 dimensions [H W].")
    elif image.mode == 'RGB':
        # rgb8 - convert to BGR for cuvslam compatibility
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected rgb8 image to have 3 dimensions with 3 channels [H W C].")
        # Convert RGB to BGR by reversing the channel order and ensure contiguous
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    elif image.mode == 'I;16':
        # uint16 depth image
        if len(frame.shape) != 2:
            raise ValueError("Expected uint16 depth image to have 2 dimensions [H W].")
        frame = frame.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    # Ensure the array is contiguous in memory
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    
    return frame

def read_timestamp_file(file_path):
    """
    Read a TUM dataset timestamp file and extract timestamp-filename pairs.
    
    Args:
        file_path (str): Path to the timestamp file
    
    Returns:
        list: List of (timestamp, filename) tuples
    """
    result = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    result.append((float(parts[0]), parts[1]))
        return result
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading timestamp file {file_path}: {e}")
        return []

def find_matching_pairs(rgb_data, depth_data, max_time_diff=0.02):
    """
    Find matching RGB-depth pairs with timestamps less than max_time_diff apart.
    
    Args:
        rgb_data (list): List of (timestamp, filename) tuples for RGB images
        depth_data (list): List of (timestamp, filename) tuples for depth images
        max_time_diff (float): Maximum allowed time difference in seconds (default: 0.02)
    
    Returns:
        list: List of (rgb_time, rgb_file, depth_time, depth_file) tuples
    """
    i, j = 0, 0
    matched_pairs = []
    
    while i < len(rgb_data) and j < len(depth_data):
        rgb_time, rgb_file = rgb_data[i]
        depth_time, depth_file = depth_data[j]
        time_diff = abs(rgb_time - depth_time)
        
        if time_diff < max_time_diff:
            matched_pairs.append((rgb_time, rgb_file, depth_time, depth_file))
            i += 1
            j += 1
        elif rgb_time < depth_time:
            i += 1
        else:
            j += 1
    
    return matched_pairs

def has_time_gap(current_time, previous_time, max_gap=0.5):
    """
    Check if there's a significant time gap between frames.
    
    Args:
        current_time (float): Current frame timestamp
        previous_time (float): Previous frame timestamp
        max_gap (float): Maximum allowed gap in seconds (default: 0.5)
    
    Returns:
        bool: True if there's a significant gap, False otherwise
    """
    if previous_time is None:
        return False
    return (current_time - previous_time) > max_gap

def get_matched_rgbd_pairs(dataset_path, max_time_diff=0.02, max_gap=0.5):
    """
    Get all matched RGB-depth pairs from a TUM dataset.
    
    Args:
        dataset_path (str): Path to the TUM dataset
        max_time_diff (float): Maximum allowed time difference in seconds (default: 0.02)
        max_gap (float): Maximum allowed gap between consecutive frames (default: 0.5)
    
    Returns:
        list: List of (rgb_time, rgb_path, depth_path) tuples with time gaps filtered out
    """
    rgb_file_path = os.path.join(dataset_path, "rgb.txt")
    depth_file_path = os.path.join(dataset_path, "depth.txt")
    
    rgb_data = read_timestamp_file(rgb_file_path)
    depth_data = read_timestamp_file(depth_file_path)
    
    if not rgb_data or not depth_data:
        return []
    
    matched_pairs = find_matching_pairs(rgb_data, depth_data, max_time_diff)
    
    # Filter out large time gaps
    filtered_pairs = []
    prev_time = None
    
    for rgb_time, rgb_file, depth_time, depth_file in matched_pairs:
        if has_time_gap(rgb_time, prev_time, max_gap):
            print(f"Skipping time gap of {rgb_time - prev_time:.2f} seconds")
        else:
            rgb_path = os.path.join(dataset_path, rgb_file)
            depth_path = os.path.join(dataset_path, depth_file)
            filtered_pairs.append((rgb_time, rgb_path, depth_path))
        
        prev_time = rgb_time
    
    return filtered_pairs
