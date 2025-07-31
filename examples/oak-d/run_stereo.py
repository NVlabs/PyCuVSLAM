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
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Add the realsense folder to the system path to import visualizer
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../realsense'))
)

from visualizer import RerunVisualizer

# Constants
RESOLUTION_720P = dai.MonoCameraProperties.SensorResolution.THE_720_P
FPS = 30
WARMUP_FRAMES = 60
SYNC_THRESHOLD_MS = 5 * 1e6  # 5ms in nanoseconds
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
QUEUE_SIZE = 8

# Camera border margins to exclude features near image edges
# This helps avoid using features from highly distorted regions in unrectified OAK-D images
# Features detected within these margins will not be processed
BORDER_TOP = 50
BORDER_BOTTOM = 0
BORDER_LEFT = 70
BORDER_RIGHT = 70

# Conversion factor from cm to meters
CM_TO_METERS = 100

def oak_transform_to_pose(oak_extrinsics: List[List[float]]) -> vslam.Pose:
    """Convert 4D transformation matrix to cuVSLAM pose.
    
    Args:
        oak_extrinsics: 4x4 transformation matrix from OAK calibration
        
    Returns:
        vslam.Pose object
    """
    extrinsics_array = np.array(oak_extrinsics)
    rotation_matrix = extrinsics_array[:3, :3]
    translation_vector = extrinsics_array[:3, 3] / CM_TO_METERS  # Convert to meters
    
    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vector)


def set_cuvslam_camera(oak_params: Dict[str, Any]) -> vslam.Camera:
    """Create a Camera object from OAK camera parameters.
    
    Args:
        oak_params: Dictionary containing camera parameters
        
    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(
        vslam.Distortion.Model.Polynomial, oak_params['distortion']
    )
    
    cam.focal = (
        oak_params['intrinsics'][0][0],
        oak_params['intrinsics'][1][1]
    )
    cam.principal = (
        oak_params['intrinsics'][0][2],
        oak_params['intrinsics'][1][2]
    )
    cam.size = oak_params['resolution']
    cam.rig_from_camera = oak_transform_to_pose(oak_params['extrinsics'])

    # Features within these outer frames will be ignored by cuVSLAM
    cam.border_top = BORDER_TOP
    cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT
    cam.border_right = BORDER_RIGHT
    
    return cam


def create_oak_pipeline() -> Tuple[dai.Pipeline, Tuple[int, int]]:
    """Create and configure the OAK-D pipeline.
    
    Returns:
        Tuple of (pipeline, resolution)
    """
    pipeline = dai.Pipeline()

    # Create stereo pair (left and right cameras)
    left_camera = pipeline.createMonoCamera()
    right_camera = pipeline.createMonoCamera()

    # Set camera properties
    left_camera.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right_camera.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Set resolution
    left_camera.setResolution(RESOLUTION_720P)
    right_camera.setResolution(RESOLUTION_720P)

    # Set FPS
    left_camera.setFps(FPS)
    right_camera.setFps(FPS)

    # Create output queues for both cameras
    left_out = pipeline.createXLinkOut()
    right_out = pipeline.createXLinkOut()

    left_out.setStreamName("left")
    right_out.setStreamName("right")

    # Link cameras to output
    left_camera.out.link(left_out.input)
    right_camera.out.link(right_out.input)

    resolution = left_camera.getResolutionSize()
    
    return pipeline, resolution


def get_stereo_calibration(
    device: dai.Device, resolution: Tuple[int, int]
) -> Dict[str, Dict[str, Any]]:
    """Get calibration data from the OAK-D device.
    
    Args:
        device: Connected OAK-D device
        resolution: Camera resolution as (width, height)
        
    Returns:
        Dictionary containing stereo calibration parameters
    """
    stereo_camera = {'left': {}, 'right': {}}
    
    # Set image size
    stereo_camera['left']['resolution'] = resolution
    stereo_camera['right']['resolution'] = resolution

    # Read the calibration data
    calib_data = device.readCalibration()

    # Get intrinsics for left and right cameras
    stereo_camera['left']['intrinsics'] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_B
    )
    stereo_camera['right']['intrinsics'] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C
    )

    # Get extrinsics (transformation of left and right cameras relative to center RGB camera)
    stereo_camera['left']['extrinsics'] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A
    )
    stereo_camera['right']['extrinsics'] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A
    )

    # Get distortion coefficients for left and right cameras (first 8 coefficients)
    stereo_camera['left']['distortion'] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_B
    )[:8]
    stereo_camera['right']['distortion'] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_C
    )[:8]

    
    return stereo_camera


def check_frame_synchronization(
    timestamp_left: int, timestamp_right: int
) -> bool:
    """Check if stereo frames are synchronized.
    
    Args:
        timestamp_left: Left frame timestamp in nanoseconds
        timestamp_right: Right frame timestamp in nanoseconds
        
    Returns:
        True if frames are synchronized, False otherwise
    """
    timestamp_diff = abs(timestamp_left - timestamp_right)
    if timestamp_diff > SYNC_THRESHOLD_MS:
        print(
            f"Warning: Stereo pair is not synchronized: timestamp difference: "
            f"{timestamp_diff/1e6:.2f} ms"
        )
        return False
    return True

# Convert OAK timestamp to nanoseconds
convert_oak_timestamp = lambda t: int(t.seconds * 1e9 + t.microseconds * 1e3)


def main() -> None:
    """Main function for OAK-D stereo tracking."""
    # Create and configure the pipeline
    pipeline, resolution = create_oak_pipeline()

    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:
        stereo_camera = get_stereo_calibration(device, resolution)

        cameras = [set_cuvslam_camera(stereo_camera['left']),
                   set_cuvslam_camera(stereo_camera['right'])
                  ]
        
        # Output synchronization
        left_queue = device.getOutputQueue("left", QUEUE_SIZE, False)
        right_queue = device.getOutputQueue("right", QUEUE_SIZE, False)

        # Create rig and tracker
        
        cfg = vslam.Tracker.OdometryConfig(
            async_sba=False,
            enable_final_landmarks_export=True,
            enable_observations_export=True,
            horizontal_stereo_camera=False
        )
        tracker = vslam.Tracker(vslam.Rig(cameras), cfg)

        # Initialize visualization and tracking variables
        visualizer = RerunVisualizer()
        frame_id = 0
        prev_timestamp: Optional[int] = None
        trajectory: List[np.ndarray] = []

        # Capture and process stereo frames
        while True:
            left_frame = left_queue.get()
            right_frame = right_queue.get()

            # Convert timestamps to nanoseconds
            timestamp_left = convert_oak_timestamp(left_frame.getTimestamp())
            timestamp_right = convert_oak_timestamp(right_frame.getTimestamp())

            # Check if frames are synchronized
            if not check_frame_synchronization(timestamp_left, timestamp_right):
                continue

            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp_left - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                    )

            frame_id += 1

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                left_img = left_frame.getCvFrame()
                right_img = right_frame.getCvFrame()

                # Track frame
                odom_pose_estimate, _ = tracker.track(timestamp_left, (left_img, right_img))
                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Visualize results
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_img],
                    pose=odom_pose,
                    observations_main_cam=[tracker.get_last_observations(0)],
                    trajectory=trajectory,
                    timestamp=timestamp_left
                )

            prev_timestamp = timestamp_left


if __name__ == "__main__":
    main()
