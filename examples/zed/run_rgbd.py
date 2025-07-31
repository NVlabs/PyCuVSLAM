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
from typing import List, Optional

import numpy as np
import pyzed.sl as sl

import cuvslam as vslam
from camera_utils import get_zed_rgbd_rig, setup_zed_camera

# Add path for visualizer import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'realsense'))
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 480)
FPS = 30
# Calculate jitter threshold based on FPS + 3ms buffer
FRAME_PERIOD_MS = 1000 / FPS  # Time between frames in milliseconds
IMAGE_JITTER_THRESHOLD_MS = (FRAME_PERIOD_MS + 2) * 1e6  # Convert to nanoseconds
RUN_STEREO = False


def main():
    """Main function for RGBD tracking with ZED camera."""
    # Initialize ZED camera
    zed, camera_info = setup_zed_camera(RESOLUTION, FPS, sl.DEPTH_MODE.PERFORMANCE, sl.UNIT.MILLIMETER)

    # Configure RGBD settings
    rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = 1000
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = RUN_STEREO

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_final_landmarks_export=True,
        odometry_mode=vslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings,
        horizontal_stereo_camera=True
    )

    # Create rig using utility function
    rig = get_zed_rgbd_rig(camera_info, RUN_STEREO)

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, cfg)
    visualizer = RerunVisualizer(num_viz_cameras=2+RUN_STEREO)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    
    # Create image containers
    image_left = sl.Mat()
    depth_left = sl.Mat()
    if RUN_STEREO:
        image_right = sl.Mat()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    print("Starting RGBD tracking with cuvslam...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get timestamp
                timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                
                # Check timestamp difference with previous frame
                if prev_timestamp is not None:
                    timestamp_diff = timestamp - prev_timestamp
                    if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                        print(
                            f"Warning: Camera stream message drop: timestamp gap "
                            f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                            f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
                        )

                frame_id += 1

                # Get images
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_measure(depth_left, sl.MEASURE.DEPTH)
                left_data = image_left.get_data()
                left_rgb = np.ascontiguousarray(left_data[:,:,[2,1,0]])
                depth_data = np.round(depth_left.get_data()).astype(np.uint16)

                if RUN_STEREO:
                    zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                    right_data = image_right.get_data()
                    right_rgb = np.ascontiguousarray(right_data[:,:,[2,1,0]])
                
                    # Track frame with both color and depth
                    odom_pose_estimate, _ = tracker.track(
                        timestamp, images=[left_rgb, right_rgb], depths=[depth_data]
                    )
                else:
                    odom_pose_estimate, _ = tracker.track(
                        timestamp, images=[left_rgb], depths=[depth_data]
                    )
                
                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue
                
                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Store current timestamp for next iteration
                prev_timestamp = timestamp

                # Visualize results for color and depth cameras
                # Same observations for both, since we only have one image
                observations = tracker.get_last_observations(0)
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_rgb, depth_data, right_rgb] if RUN_STEREO else [left_rgb, depth_data],
                    pose=odom_pose,
                    observations_main_cam=[observations]*(2+RUN_STEREO), # 2 for stereo, 1 for mono
                    trajectory=trajectory,
                    timestamp=timestamp
                )

    finally:
        # Clean up
        zed.close()


if __name__ == "__main__":
    main()
