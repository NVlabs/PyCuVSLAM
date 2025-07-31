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
import queue
import threading
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam
from camera_utils import get_rs_vio_rig
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 360)
FPS = 30
IMU_FREQUENCY_ACCEL = 200
IMU_FREQUENCY_GYRO = 200
IMAGE_JITTER_THRESHOLD_MS = 35 * 1e6  # 35ms in nanoseconds
IMU_JITTER_THRESHOLD_MS = 6 * 1e6  # 6ms in nanoseconds


class ThreadWithTimestamp:
    """Helper class to manage timestamps between camera and IMU threads."""
    
    def __init__(
        self,
        low_rate_threshold_ns: int,
        high_rate_threshold_ns: int
    ) -> None:
        """Initialize timestamp tracker.
        
        Args:
            low_rate_threshold_ns: Threshold for low-rate (camera) stream
            high_rate_threshold_ns: Threshold for high-rate (IMU) stream
        """
        self.prev_low_rate_timestamp: Optional[int] = None
        self.prev_high_rate_timestamp: Optional[int] = None
        self.low_rate_threshold_ns = low_rate_threshold_ns
        self.high_rate_threshold_ns = high_rate_threshold_ns
        self.last_low_rate_timestamp: Optional[int] = None


def imu_thread(
    tracker: vslam.Tracker,
    q: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    motion_pipe: rs.pipeline
) -> None:
    """IMU processing thread.
    
    Args:
        tracker: cuVSLAM tracker instance
        q: Queue for communication with main thread
        thread_with_timestamp: Timestamp management object
        motion_pipe: RealSense motion pipeline
    """
    try:
        while True:
            imu_measurement = vslam.ImuMeasurement()
            imu_frames = motion_pipe.wait_for_frames()
            current_timestamp = int(imu_frames[0].timestamp * 1e6)

            # Check timestamp consistency with camera thread
            if (thread_with_timestamp.last_low_rate_timestamp is not None and
                    current_timestamp < thread_with_timestamp.last_low_rate_timestamp):
                print(
                    f"Warning: IMU stream timestamp is earlier than camera "
                    f"stream ({current_timestamp} < "
                    f"{thread_with_timestamp.last_low_rate_timestamp})"
                )
                continue

            # Check for timestamp gaps in IMU stream
            timestamp_diff = 0
            if thread_with_timestamp.prev_high_rate_timestamp is not None:
                timestamp_diff = (
                    current_timestamp - thread_with_timestamp.prev_high_rate_timestamp
                )
                if timestamp_diff > thread_with_timestamp.high_rate_threshold_ns:
                    print(
                        f"Warning: IMU stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{thread_with_timestamp.high_rate_threshold_ns/1e6:.2f} ms"
                    )
                elif timestamp_diff < 0:
                    print("Warning: IMU messages are not sequential")

            if timestamp_diff < 0:
                continue

            thread_with_timestamp.prev_high_rate_timestamp = deepcopy(
                current_timestamp
            )
            
            # Populate IMU measurement
            imu_measurement.timestamp_ns = current_timestamp
            accel_data = np.frombuffer(imu_frames[0].get_data(), dtype=np.float32)
            gyro_data = np.frombuffer(imu_frames[1].get_data(), dtype=np.float32)
            imu_measurement.linear_accelerations = accel_data[:3]
            imu_measurement.angular_velocities = gyro_data[:3]

            if timestamp_diff > 0:
                tracker.register_imu_measurement(0, imu_measurement)
    except Exception as e:
        print(f"IMU thread error: {e}")


def camera_thread(
    tracker: vslam.Tracker,
    q: queue.Queue,
    thread_with_timestamp: ThreadWithTimestamp,
    ir_pipe: rs.pipeline
) -> None:
    """Camera processing thread.
    
    Args:
        tracker: cuVSLAM tracker instance
        q: Queue for communication with main thread
        thread_with_timestamp: Timestamp management object
        ir_pipe: RealSense infrared pipeline
    """
    try:
        while True:
            ir_frames = ir_pipe.wait_for_frames()
            ir_left_frame = ir_frames.get_infrared_frame(1)
            ir_right_frame = ir_frames.get_infrared_frame(2)
            current_timestamp = int(ir_left_frame.timestamp * 1e6)

            # Check for timestamp gaps in camera stream
            if thread_with_timestamp.prev_low_rate_timestamp is not None:
                timestamp_diff = (
                    current_timestamp - thread_with_timestamp.prev_low_rate_timestamp
                )
                if timestamp_diff > thread_with_timestamp.low_rate_threshold_ns:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{thread_with_timestamp.low_rate_threshold_ns/1e6:.2f} ms"
                    )

            thread_with_timestamp.prev_low_rate_timestamp = deepcopy(
                current_timestamp
            )
            
            images = (
                np.asanyarray(ir_left_frame.get_data()),
                np.asanyarray(ir_right_frame.get_data())
            )

            odom_pose_estimate, _ = tracker.track(current_timestamp, images)
            odom_pose = odom_pose_estimate.world_from_rig.pose

            # Put result in queue for main thread
            q.put([current_timestamp, odom_pose, images])
            thread_with_timestamp.last_low_rate_timestamp = current_timestamp
    except Exception as e:
        print(f"Camera thread error: {e}")


def setup_camera_parameters() -> dict:
    """Set up camera parameters by starting pipeline briefly.
    
    Returns:
        Dictionary containing camera parameters
    """
    # Initialize RealSense configuration
    config = rs.config()
    pipeline = rs.pipeline()

    # Configure streams for initial setup
    config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO)

    # Start pipeline to get intrinsics and extrinsics
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    pipeline.stop()

    # Prepare camera parameters
    camera_params = {'left': {}, 'right': {}, 'imu': {}}

    # Get extrinsics and intrinsics
    camera_params['right']['extrinsics'] = frames[1].profile.get_extrinsics_to(
        frames[0].profile
    )
    camera_params['imu']['cam_from_imu'] = frames[2].profile.get_extrinsics_to(
        frames[0].profile
    )
    camera_params['left']['intrinsics'] = (
        frames[0].profile.as_video_stream_profile().intrinsics
    )
    camera_params['right']['intrinsics'] = (
        frames[1].profile.as_video_stream_profile().intrinsics
    )

    return camera_params


def main() -> None:
    """Main function for VIO tracking."""
    # Setup camera parameters
    camera_params = setup_camera_parameters()

    # Configure tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        debug_imu_mode=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial,
        horizontal_stereo_camera=True
    )

    # Create rig using utility function
    rig = get_rs_vio_rig(camera_params)

    # Initialize tracker
    tracker = vslam.Tracker(rig, cfg)

    # Set up IR pipeline
    ir_pipe = rs.pipeline()
    ir_config = rs.config()
    ir_config.enable_stream(
        rs.stream.infrared, 1, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )
    ir_config.enable_stream(
        rs.stream.infrared, 2, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FPS
    )

    # Configure device settings
    config_temp = rs.config()
    ir_wrapper = rs.pipeline_wrapper(ir_pipe)
    ir_profile = config_temp.resolve(ir_wrapper)
    device = ir_profile.get_device()

    # Disable IR emitter if supported
    depth_sensor = device.query_sensors()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Set up motion pipeline
    motion_pipe = rs.pipeline()
    motion_config = rs.config()
    motion_config.enable_stream(
        rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL
    )
    motion_config.enable_stream(
        rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO
    )

    # Start pipelines
    motion_pipe.start(motion_config)
    ir_pipe.start(ir_config)

    # Set up threading and visualization
    q = queue.Queue()
    visualizer = RerunVisualizer()
    thread_with_timestamp = ThreadWithTimestamp(
        IMAGE_JITTER_THRESHOLD_MS, IMU_JITTER_THRESHOLD_MS
    )

    # Start threads
    imu_thread_obj = threading.Thread(
        target=imu_thread,
        args=(tracker, q, thread_with_timestamp, motion_pipe),
        daemon=True
    )
    camera_thread_obj = threading.Thread(
        target=camera_thread,
        args=(tracker, q, thread_with_timestamp, ir_pipe),
        daemon=True
    )

    imu_thread_obj.start()
    camera_thread_obj.start()

    frame_id = 0
    trajectory: List[np.ndarray] = []

    try:
        while True:
            # Get the output from the queue with timeout
            try:
                timestamp, odom_pose, images = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if odom_pose is None:
                continue

            frame_id += 1
            trajectory.append(odom_pose.translation)

            gravity = None
            if cfg.odometry_mode == vslam.Tracker.OdometryMode.Inertial:
                # Gravity estimation requires sufficient keyframes with motion
                gravity = tracker.get_last_gravity()

            # Visualize results for left camera
            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[images[0]],
                pose=odom_pose,
                observations_main_cam=[tracker.get_last_observations(0)],
                trajectory=trajectory,
                timestamp=timestamp,
                gravity=gravity
            )

    except KeyboardInterrupt:
        print("Stopping VIO tracking...")
    finally:
        motion_pipe.stop()
        ir_pipe.stop()


if __name__ == "__main__":
    main()
