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

"""
cuVSLAM Tracker - A wrapper combining Odometry and SLAM functionality.
"""

from typing import Optional, List, Callable, Any, Tuple
from .pycuvslam import Odometry, Slam, Rig, PoseEstimate, Pose, ImuMeasurement


class Tracker:
    """
    A wrapper that combines cuVSLAM Odometry and SLAM functionality.

    This class automatically manages both the Odometry and SLAM instances,
    providing a simplified interface for common use cases.
    """

    # Expose inner classes of Odometry & Slam
    OdometryMode = Odometry.OdometryMode
    MulticameraMode = Odometry.MulticameraMode
    OdometryConfig = Odometry.Config
    OdometryRGBDSettings = Odometry.RGBDSettings

    SlamConfig = Slam.Config
    SlamMetrics = Slam.Metrics
    SlamLocalizationSettings = Slam.LocalizationSettings

    def __init__(self, rig: Rig, odom_config: Optional[Odometry.Config] = OdometryConfig(), slam_config: Optional[Slam.Config] = None) -> None:
        """
        Initialize the cuVSLAM system.

        Args:
            rig: Camera rig configuration
            odom_config: Optional odometry configuration (uses defaults if omitted)
            slam_config: Optional SLAM configuration (disables SLAM if None)
        """
        # need to export observations/landmarks for Slam
        if slam_config is not None:
            odom_config.enable_observations_export = True
            odom_config.enable_landmarks_export = True
        # Create odometry
        self.odom = Odometry(rig, odom_config)
        # Create SLAM
        primary_cams = self.odom.get_primary_cameras()
        self.slam = Slam(rig, primary_cams, slam_config) if slam_config is not None else None

    def track(self, timestamp: int, images: List[Any], masks: Optional[List[Any]] = None, depths: Optional[List[Any]] = None) -> Tuple[PoseEstimate, Optional[Pose]]:
        """
        Track a rig pose using current image frame.

        This method combines tracking and SLAM processing in a single call.

        Args:
            timestamp: Images timestamp in nanoseconds
            images: List of numpy arrays containing the camera images
            masks: Optional list of numpy arrays containing masks for the images
            depths: Optional list of numpy arrays containing depth images

        Returns:
            PoseEstimate: The computed pose estimate from Odometry
            Pose: The computed pose estimate from SLAM
        """
        pose_estimate = self.odom.track(timestamp, images, masks, depths)

        slam_pose = None
        if self.slam:
            state = self.odom.get_state()
            slam_pose = self.slam.track(state)

        return pose_estimate, slam_pose

    def register_imu_measurement(self, sensor_index: int, imu_measurement: ImuMeasurement) -> None:
        """Register an IMU measurement with the tracker."""
        self.odom.register_imu_measurement(sensor_index, imu_measurement)

    def get_last_observations(self, camera_index: int) -> List[Any]:
        """Get observations from the last frame."""
        return self.odom.get_last_observations(camera_index)

    def get_last_landmarks(self) -> List[Any]:
        """Get landmarks from the last frame."""
        return self.odom.get_last_landmarks()

    def get_last_gravity(self) -> Optional[Any]:
        """Get gravity vector from the last frame."""
        return self.odom.get_last_gravity()

    def get_final_landmarks(self) -> dict[int, Any]:
        """Get all final landmarks from all frames."""
        return self.odom.get_final_landmarks()

    def get_all_slam_poses(self, max_poses_count: int = 0) -> Optional[List[Any]]:
        """Get all SLAM poses for each frame."""
        return self.slam.get_all_slam_poses(max_poses_count) if self.slam else None

    def set_slam_pose(self, pose: Pose) -> None:
        """Set the rig SLAM pose to a value provided by user."""
        if self.slam:
            self.slam.set_slam_pose(pose)

    def save_map(self, folder_name: str, callback: Callable[[bool], None]) -> None:
        """Save SLAM database (map) to folder asynchronously."""
        if self.slam:
            self.slam.save_map(folder_name, callback)

    def localize_in_map(self, folder_name: str, guess_pose: Pose, images: List[Any], settings: Any, callback: Callable[[Optional[Pose], str], None]) -> None:
        """Localize in the existing database (map) asynchronously."""
        if self.slam:
            self.slam.localize_in_map(folder_name, guess_pose, images, settings, callback)

    def get_pose_graph(self) -> Optional[Any]:
        """Get pose graph."""
        return self.slam.get_pose_graph() if self.slam else None

    def get_slam_metrics(self) -> Optional[Any]:
        """Get SLAM metrics."""
        return self.slam.get_slam_metrics() if self.slam else None

    def get_loop_closure_poses(self) -> Optional[List[Any]]:
        """Get list of last 10 loop closure poses with timestamps."""
        return self.slam.get_loop_closure_poses() if self.slam else None

    @staticmethod
    def merge_maps(rig: Rig, databases: List[str], output_folder: str) -> None:
        """Merge existing maps into one map."""
        Slam.merge_maps(rig, databases, output_folder)
