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
import yaml

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

# Local imports
from dataset_utils import load_frame, get_matched_rgbd_pairs

# Set up dataset path
tum_dataset_path = os.path.join(
    os.path.dirname(__file__),
    "dataset/rgbd_dataset_freiburg3_long_office_household"
)

def color_from_id(identifier):
    """Generate a color from an identifier."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]

# Constants
IMAGE_JITTER_THRESHOLD_MS = 40 * 1e6  # 40ms in nanoseconds

# Setup rerun visualizer
rr.init('tum_dataset', strict=True, spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Horizontal(contents=[
        rrb.Vertical(contents=[
            rrb.Spatial2DView(origin='world/camera/image', name='RGB Camera'),
            rrb.Spatial2DView(origin='world/camera/depth', name='Depth Camera')
        ]),
        rrb.Spatial3DView(
            name="3D",
            defaults=[rr.components.ImagePlaneDistance(0.5)]
        )
    ]),
), make_active=True)

# Setup coordinate basis for root
# cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Get matched RGB-D pairs from dataset
rgbd_pairs = get_matched_rgbd_pairs(
    tum_dataset_path, max_time_diff=0.02, max_gap=0.5
)
print(f"Found {len(rgbd_pairs)} matched RGB-D pairs")

# Load tum configuration from YAML
print("Loading tum configuration...")
config_path = os.path.join(tum_dataset_path, "freiburg3_rig.yaml")
with open(config_path, 'r') as file:
    config_data = yaml.safe_load(file)

# Set up camera parameters
camera = cuvslam.Camera()
camera.size = (
    config_data['rgb_camera']['image_width'],
    config_data['rgb_camera']['image_height']
)
camera.principal = config_data['rgb_camera']['principal_point']
camera.focal = config_data['rgb_camera']['focal_length']
camera.border_top = 20
camera.border_bottom = 20
camera.border_left = 10
camera.border_right = 50

# Set up RGBD settings
rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
rgbd_settings.depth_scale_factor = config_data['depth_camera']['scale']
rgbd_settings.depth_camera_id = 0
rgbd_settings.enable_depth_stereo_tracking = False

# Configure tracker
cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=True,
    enable_final_landmarks_export=True,
    odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
    rgbd_settings=rgbd_settings
)

# Initialize tracker
tracker = cuvslam.Tracker(cuvslam.Rig([camera]), cfg)

frame_id = 0
prev_timestamp = None
trajectory = []

# Process each matched RGB-D pair
for rgb_time, rgb_path, depth_path in rgbd_pairs:
    color_frame = load_frame(rgb_path)
    depth_frame = load_frame(depth_path)
    
    if color_frame is None or depth_frame is None:
        print(f"Warning: Failed to read image files: {rgb_path} or {depth_path}")
        continue
    
    # Convert timestamp to nanoseconds for tracker
    timestamp = int(rgb_time * 1e9)
    
    # Check timestamp difference with previous frame
    if prev_timestamp is not None:
        timestamp_diff = timestamp - prev_timestamp
        if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
            print(
                f"Warning: Camera stream message delayed: timestamp gap "
                f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                f"{IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms"
            )
    
    # Track frame
    odom_pose_estimate, _ = tracker.track(
        timestamp, images=[color_frame], depths=[depth_frame]
    )
    
    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose
    trajectory.append(odom_pose.translation)
    
    # Get observations
    observations = [tracker.get_last_observations(0)]
    
    # Extract observation points and colors
    obs_uv = [[o.u, o.v] for o in observations[0]]
    obs_colors = [color_from_id(o.id) for o in observations[0]]
    
    # Log visualization data
    rr.set_time_sequence('frame', frame_id)
    rr.log('trajectory', rr.LineStrips3D(trajectory), static=True)

    rr.log(
        "world/camera/rgb",
        rr.Transform3D(
            translation=odom_pose.translation,
            quaternion=odom_pose.rotation
        ),
        rr.Arrows3D(
            vectors=np.eye(3) * 0.2,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
        )
    )
    
    # Log RGB and depth images with observations
    rr.log('world/camera/image', rr.Image(color_frame).compress(jpeg_quality=80))
    rr.log(
        'world/camera/image/observations',
        rr.Points2D(obs_uv, radii=5, colors=obs_colors)
    )
    rr.log('world/camera/depth', rr.Image(depth_frame))
    rr.log(
        'world/camera/depth/observations',
        rr.Points2D(obs_uv, radii=5, colors=obs_colors)
    )
    
    # Update for next iteration
    frame_id += 1
    prev_timestamp = timestamp

