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
import numpy as np
import cuvslam as vslam
from typing import List, Tuple
from scipy.spatial.transform import Rotation
import json

def to_distortion_model(distortion: str) -> vslam.Distortion.Model:
    """Convert string distortion model name to vslam.Distortion.Model enum."""
    distortion_models = {
        'pinhole': vslam.Distortion.Model.Pinhole,
        'fisheye': vslam.Distortion.Model.Fisheye,
        'brown': vslam.Distortion.Model.Brown,
        'polynomial': vslam.Distortion.Model.Polynomial
    }
    if distortion not in distortion_models:
        raise ValueError(f"Unknown distortion model: {distortion}")
        
    return distortion_models[distortion]

def opengl_to_opencv_transform(rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from edex coordinate system to OpenCV coordinate system."""
    K = np.array([
        [1, 0,  0],
        [0, -1, 0],
        [0, 0, -1]])
    return (K @ rotation @ K.T), K @ translation

def transform_to_pose(transform_16: List[float]) -> vslam.Pose:
    """Convert a 4x4 transformation matrix to a vslam.Pose object."""
    transform = np.array(transform_16).reshape([-1, 4])
    rotation_opencv, translation_opencv = opengl_to_opencv_transform(transform[:3, :3], transform[:3, 3])
    rotation_quat = Rotation.from_matrix(rotation_opencv).as_quat()
    
    return vslam.Pose(rotation = rotation_quat, translation = translation_opencv)

def read_stereo_edex(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDEX file not found: {file_path}")
        
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    cameras = []
    for idx, cam_data in enumerate(data[0]['cameras']):
        config = {
            'camera_model': cam_data['intrinsics']['distortion_model'],
            'distortion_coefficients': cam_data['intrinsics']['distortion_params'],
            'intrinsics': cam_data['intrinsics']['focal'] + cam_data['intrinsics']['principal'],
            'resolution': cam_data['intrinsics']['size'],
            'extrinsics': cam_data['transform']
        }
        
        cam = vslam.Camera()
        cam.distortion = vslam.Distortion(
            to_distortion_model(config['camera_model']),
            config['distortion_coefficients']
        )
        cam.focal = config['intrinsics'][0:2]
        cam.principal = config['intrinsics'][2:4]
        cam.size = config['resolution']
        cam.rig_from_camera = transform_to_pose(config['extrinsics'])
        
        cameras.append(cam)
    
    return cameras
