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
from typing import Any, Optional, Tuple

import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Constants
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FPS = 30
DEFAULT_IMU_FREQUENCY = 200


def transform_to_pose(transform_matrix=None) -> vslam.Pose:
    """Convert a transformation matrix to a vslam.Pose object.
    
    Args:
        transform_matrix: Either a ZED transform object or a numpy array
                         representing the transformation matrix
                         
    Returns:
        vslam.Pose object
    """
    if transform_matrix is None:
        # Default identity transform
        rotation_matrix = np.eye(3)
        translation_vec = [0] * 3
        rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
        return vslam.Pose(rotation=rotation_quat, translation=translation_vec)
    
    # Handle ZED transform object
    rotation_matrix = transform_matrix.get_rotation_matrix().r
    translation_vec = transform_matrix.get_translation().get()
    
    # Convert to numpy arrays
    rotation_matrix = np.array(rotation_matrix).reshape([3, 3])
    translation_vec = np.array(translation_vec) / 1000

    # Convert rotation matrix to quaternion
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    
    return vslam.Pose(rotation=rotation_quat, translation=translation_vec)


def get_zed_camera(
    zed_intrinsics: sl.CameraParameters,
    transform_matrix: Optional[Any] = None,
    raw: bool = False
) -> vslam.Camera:
    """Create a Camera object from ZED intrinsics.
    
    Args:
        zed_intrinsics: ZED camera intrinsics object
        transform_matrix: Optional transformation matrix for camera pose
        raw: Whether to use raw (unrectified) images
        
    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    # ZED camera parameters have disto attribute
    # Distortion factor : [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
    # Radial (k1, k2, k3, k4, k5, k6), Tangential (p1,p2) and Prism (s1, s2, s3, s4) distortion
    # cuVSLAM Brown distortion model has the following order: [k_1, k_2, k_3, p_1, p_2]
    brown_disto = list(zed_intrinsics.disto[[0,1,4,2,3]])

    if raw:
        cam.distortion = vslam.Distortion(
            vslam.Distortion.Model.Brown,
            brown_disto
        )
    else:
        cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = zed_intrinsics.fx, zed_intrinsics.fy
    cam.principal = zed_intrinsics.cx, zed_intrinsics.cy
    
    # Get image size from the intrinsics
    # ZED camera parameters have image_size attribute
    cam.size = zed_intrinsics.image_size.width, zed_intrinsics.image_size.height

    if transform_matrix is not None:
        cam.rig_from_camera = transform_to_pose(transform_matrix)
    
    return cam


def get_zed_rgbd_rig(
    camera_info: sl.CameraInformation,
    run_stereo: bool = False
) -> vslam.Rig:
    """Create a RGBD Rig object from ZED camera information.
    
    Args:
        camera_info: ZED camera information object
        run_stereo: Whether to include stereo camera
        
    Returns:
        vslam.Rig object
    """
    rig = vslam.Rig()
    
    # Get calibration parameters
    calibration_params = camera_info.camera_configuration.calibration_parameters

    # Create left camera (reference camera)
    left_camera = get_zed_camera(calibration_params.left_cam, raw=False)

    print(f"Camera resolution: {left_camera.size}")
    
    # Create right camera with extrinsics
    if run_stereo:
        right_camera = get_zed_camera(
            calibration_params.right_cam,
            calibration_params.stereo_transform,
            raw=False
        )
        rig.cameras = [left_camera, right_camera]
    else:
        rig.cameras = [left_camera]
    
    return rig


def get_zed_stereo_rig(
    camera_info: sl.CameraInformation,
    raw: bool = False
) -> vslam.Rig:
    """Create a stereo Rig object from ZED camera information.
    
    Args:
        camera_info: ZED camera information object
        raw: Whether to use raw (unrectified) images
        
    Returns:
        vslam.Rig object
    """
    rig = vslam.Rig()
    
    # Get calibration parameters
    if raw:
        calibration_params = camera_info.camera_configuration.calibration_parameters_raw
    else:
        calibration_params = camera_info.camera_configuration.calibration_parameters

    # Create left camera (reference camera)
    left_camera = get_zed_camera(calibration_params.left_cam, raw=raw)
    
    # Create right camera with extrinsics
    right_camera = get_zed_camera(
        calibration_params.right_cam,
        calibration_params.stereo_transform,
        raw=raw
    )

    print(f"Camera resolution: {left_camera.size}")
    
    rig.cameras = [left_camera, right_camera]
    return rig


def setup_zed_camera(
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS,
    depth_mode = sl.DEPTH_MODE.NONE,
    depth_units = sl.UNIT.MILLIMETER
) -> Tuple[sl.Camera, sl.CameraInformation]:
    """Set up and configure a ZED camera.
    
    Args:
        resolution: Camera resolution as (width, height)
        fps: Frames per second
        depth_mode: Depth mode for the camera
        
    Returns:
        Tuple of (camera, camera_info)
    """
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_fps = fps
    init_params.depth_mode = depth_mode
    if depth_mode != sl.DEPTH_MODE.NONE:
        init_params.coordinate_units = depth_units

    # Set resolution
    if resolution == (640, 480):
        init_params.camera_resolution = sl.RESOLUTION.VGA
    elif resolution == (1280, 720):
        init_params.camera_resolution = sl.RESOLUTION.HD720
    
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {status}")
    
    # Get camera information
    camera_info = zed.get_camera_information()
    print(f"Camera FPS: {camera_info.camera_configuration.fps}")

    return zed, camera_info