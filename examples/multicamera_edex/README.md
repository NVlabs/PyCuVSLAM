# Tutorial: Running PyCuVSLAM Multicamera Visual Odometry

This tutorial demonstrates how to run PyCuVSLAM in multicamera mode using on different datasets

## Tartan Ground Dataset

cuVSLAM can be run in multicamera mode on setup with hardware-synchronized stereo cameras. This section describes how to run PyCuVSLAM on the [Tartan Ground dataset](https://tartanair.org/tartanground/)

### Prerequisites

1. Install the [tartanair](https://tartanair.org/index.html) package
2. Follow the [TartanGround download example](https://tartanair.org/examples.html#download-tartanground) to get image data

Your loader should look like this (please modify the path to the actual location of `pycuvslam/examples/multicamera_edex/datasets`):

```python
tartanground_data_root = '/home/admin/pycuvslam/examples/multicamera_edex/datasets/tartan_ground'
ta.init(tartanground_data_root)
ta.download_ground_multi_thread(env = ['OldTownFall'],
            version = ['v3_anymal'],
            modality = ['image'],
            camera_name = ['lcam_front', 'rcam_front', 
                           'lcam_left', 'rcam_left',
                           'lcam_right', 'rcam_right', 
                           'lcam_back', 'rcam_back',
                           'lcam_top', 'rcam_top', 
                           'lcam_bottom', 'rcam_bottom'],
            unzip = True)
```

> **Note**: Make sure you have sufficient disk space before downloading, as the extracted images for this environment can require up to 70GB
>  
> If you want to save space, you can select a smaller set of cameras. However, ensure that your selection includes valid stereo pairs, and update both the python script and the `tartan_ground.edex` file to match your chosen camera list

### Running 6-Stereo Camera Visual Tracking

```bash
python3 track_multicamera_tartan.py
```

After running the script, you should see a Rerun visualization window displaying the image streams from all the downloaded cameras:

![Tartan Visualization](../../assets/tutorial_multicamera_tartan.gif) 

## R2B Galileo Dataset

This example is based on the original tutorial provided for ROSbag in the [ISAAC ROS documentation](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_multi_hawk.html).

### Get Dataset

To obtain the multicamera datasets, you need to download [r2b datasets in ROSbag format](https://registry.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/r2bdataset2024/files) and convert them using the [edex extractor](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/isaac_ros_rosbag_utils/index.html#edex-extraction).

#### Step 1: Download and Verify ROSbag

1. Follow the original [ISAAC ROS Multi-camera Visual SLAM tutorial](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_multi_hawk.html) for execution on a rosbag
2. Ensure the rosbag file has been downloaded and verify you can play it within the docker container:

```bash
ros2 bag play ${ISAAC_ROS_WS}/isaac_ros_assets/rosbags/r2b_galileo
```

#### Step 2: Extract Data Using EDEX

Set up the [Isaac ROS rosbag utils](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/isaac_ros_rosbag_utils/index.html) package and run within the docker container:

```bash
ros2 run isaac_ros_rosbag_utils extract_edex \
  --config_path src/isaac_ros_common/isaac_ros_rosbag_utils/config/edex_extraction_nova.yaml \
  --rosbag_path ${ISAAC_ROS_WS}/isaac_ros_assets/rosbags/r2b_galileo \
  --edex_path ${ISAAC_ROS_WS}/isaac_ros_assets/r2b_galileo_edex
```

#### Expected Output Structure

The expected structure of the output folder:

```bash
r2b_galileo_edex
├── frame_metadata.jsonl
├── images
│   ├── back_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── front_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── left_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── raw_timestamps.csv
│   ├── right_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   └── synced_timestamps.csv
├── robot.urdf
└── stereo.edex
```

3. Exit from docker container and move `r2b_galileo_edex` folder to multicamera example folder:
```bash
mkdir -p ~/pycuvslam/examples/multicamera_edex/datasets
mv ~/workspaces/isaac/isaac_ros_assets/r2b_galileo_edex ~/pycuvslam/examples/multicamera_edex/datasets/r2b_galileo_edex
```

## Running 4-Stereo Camera Visual Tracking

Go to the `examples/multicamera_edex` folder and run

```bash
python3 track_multicamera_r2b.py
```

After running the script, you should see a Rerun visualization window displaying 8 unrectified image streams, corresponding to the left and right images from each of the 4 stereo cameras:

![R2B Visualization](../../assets/tutorial_multicamera_edex.gif) 