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
import cuvslam as vslam
import os
import json
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dataset_utils import read_stereo_edex
from PIL import Image

# generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier): return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

import tartanair as ta

# Initialize TartanAir.
tartanair_data_root = 'datasets/tartan_ground/'
ta.init(tartanair_data_root)

CAMERA_LIST = ['lcam_front', 'rcam_front', 'lcam_back', 'rcam_back', 'lcam_left', 'rcam_left', 'lcam_right',  'rcam_right', 'lcam_top', 'rcam_top', 'lcam_bottom', 'rcam_bottom']

# Create iterator.
ta_iterator = ta.iterator(env = ['OldTownFall'],
                        trajectory_id = ['Pose2000'],
                        difficulty = 'ground',
                        modality = 'image',
                        camera_name = CAMERA_LIST)


### setup rerun visualizer
rr.init('tartan_ground', strict=True, spawn=True)  # launch re-run instance
# setup rerun views
rr.send_blueprint(rrb.Blueprint(rrb.TimePanel(state="collapsed"),
                                rrb.Vertical(
                                    contents=[
                                        rrb.Horizontal(
                                            contents=[rrb.Spatial2DView(origin='car/cam0', name='front-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam1', name='front-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam2', name='back-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam3',  name='back-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam8', name='top-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam9', name='top-stereo_right'),]),
                                        rrb.Spatial3DView(name="3D", defaults=[rr.components.ImagePlaneDistance(0.5)]),
                                        rrb.Horizontal(
                                            contents=[rrb.Spatial2DView(origin='car/cam4', name='left-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam5', name='left-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam6', name='right-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam7',  name='right-stereo_right'),
                                                      rrb.Spatial2DView(origin='car/cam10',  name='bottom-stereo_left'),
                                                      rrb.Spatial2DView(origin='car/cam11',  name='bottom-stereo_right')])
                                        ]
                                    ), 
                                ),
                            make_active=True)
# setup coordinate basis for root, cuvslam uses right-hand system with  X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Load camera configuration from EDEX file
cameras = read_stereo_edex('tartan_ground.edex')

# Set up VSLAM rig and tracker
rig = vslam.Rig()
rig.cameras = cameras

cfg = vslam.Tracker.OdometryConfig(enable_final_landmarks_export = True, horizontal_stereo_camera=True)

tracker = vslam.Tracker(rig, cfg)

trajectory = []

# Process each frame
frame_id = 0
while True:
    try:
        v = next(ta_iterator)
    except StopIteration:
        print(f"Iterator exhausted at frame {frame_id}")
        break
    
    timestamp = frame_id
    images = [np.asarray(v[cnt]['image']) for cnt in CAMERA_LIST]
    # do multicamera visual tracking
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # get visualization data
    observations = [tracker.get_last_observations(i) for i in range(len(CAMERA_LIST))]
    landmarks = tracker.get_last_landmarks()
    final_landmarks = tracker.get_final_landmarks()
    # prepare visualization data
    observations_uv = [[[o.u, o.v] for o in obs_instance] for obs_instance in observations]
    observations_colors = [[color_from_id(o.id) for o in obs_instance] for obs_instance in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)
    # send results to rerun for visualization
    rr.set_time_sequence('frame', frame_id)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('car', rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('car/body', rr.Boxes3D(centers=[0, 0.3 / 2, 0], sizes=[[0.35, 0.3, 0.66]]))
    rr.log('car/landmarks_center', rr.Points3D(landmark_xyz, radii=0.02, colors=landmarks_colors))

    for i in range(len(cameras)):
        rr.log('car/cam%s/image' % i, rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log('car/cam%s/observations' % i, rr.Points2D(observations_uv[i], radii=5, colors=observations_colors[i]))

        # show only even cameras in 3D world
        if not i%2:
            rr.log('car/cam%s' % i, rr.Transform3D(translation=cameras[i].rig_from_camera.translation, 
                                                        rotation=rr.Quaternion(xyzw=cameras[i].rig_from_camera.rotation),
                                                        from_parent=False))
            rr.log('car/cam%s' % i, rr.Pinhole(image_plane_distance=1.,
                                            image_from_camera=np.array([[cameras[i].focal[0], 0, cameras[i].principal[0]],
                                                                        [0, cameras[i].focal[1], cameras[i].principal[1]],
                                                                        [0, 0, 1]]),
                                            width=cameras[i].size[0], height=cameras[i].size[1]))
    
    frame_id += 1
