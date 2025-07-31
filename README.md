# PyCuVSLAM: CUDA-Accelerated Visual Odometry and Mapping

![Demo](pycuvslam.gif)

### [ArXiv paper](https://www.arxiv.org/abs/2506.04359) | [Python API](https://nvlabs.github.io/PyCuVSLAM/api.html) | [ROS2](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)

## Overview


PyCuVSLAM is the official Python wrapper for the NVIDIA cuVSLAM library, providing various Visual Tracking Camera modes and Simultaneous Localization and Mapping (SLAM) capabilities. Leveraging CUDA acceleration and a rich set of features, PyCuVSLAM delivers highly accurate, computationally efficient, real-time performance.

![Overview](assets/pycuvslam_overview.png)

## Table of Contents

- 🛠️ [System Requirements and Setup](#system-requirements-and-setup)
- [💻 Examples and Guides](#examples-and-guides)
- [🤖 ROS2 Support](#ros2-support)
- [📚 API Documentation and Technical Report](#api-documentation-and-technical-report)
- [⚙️ Performance and Troubleshooting](#performance-and-troubleshooting)
- [⚖️ License](#license)
- [🎓 Citation](#citation)

## System Requirements and Setup

PyCuVSLAM is supported on the following OS and platforms, with the system requirements and installation methods listed below:

| OS                                | Architecture | System Requirements                          | Supported Installation Method                   |
|-----------------------------------|--------------|----------------------------------------------|-------------------------------------------------|
| Ubuntu 22.04 (Desktop/Laptop)     | x86_64       | Python 3.10, Nvidia GPU with CUDA 12.6       | [Native][3], [Venv][4], [Conda][5], [Docker][6] |
| Ubuntu 24.04 (Desktop/Laptop)     | x86_64       | Nvidia GPU with CUDA 12.6                    | [Conda][5], [Docker][6]                         |
| Ubuntu 22.04 ([Nvidia Jetson][1]) | aarch64      | [Jetpack 6.1/6.2][2], Python 3.10, CUDA 12.6 | [Native][3], [Venv][4], [Conda][5], [Docker][6] |

[1]: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
[2]: https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/
[3]: #option-1-native-install
[4]: #option-2-using-venv
[5]: #option-3-using-conda
[6]: #option-4-using-docker


### CUDA Toolkit
Make sure you have the CUDA Toolkit installed, you can download the toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). If you install the CUDA toolkit for the first time, make sure to restart your computer.

### Environment setup
Depending on your OS and platform and the supported installation method, follow the instructions below to environment setup and PyCuVSLAM installation.

>**Note:**  to correctly clone PyCuVSLAM binaries, Git LFS is required before cloning the repository. Please install it by running:
>```bash
>sudo apt-get install git-lfs
>```

#### Option 1: Native Install
**Important**: This option is only available for Ubuntu 22.04 x86_64 and Jetpack 6.1/6.2 aarch64.

There are no special instructions for native install, proceed to [PyCuVSLAM installation](#pycuvslam-installation).

#### Option 2: Using Venv
**Important**: This option is only available for Ubuntu 22.04 x86_64 and Jetpack 6.1/6.2 aarch64.  

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
Proceed to [PyCuVSLAM installation](#pycuvslam-installation)  

#### Option 3: Using Conda

**Important**: This option has been tested on Ubuntu 22.04 x86_64 and Ubuntu 24.04 x86_64

Create a conda environment and install the required packages:

```bash
conda create -n pycuvslam python==3.10 pip
conda activate pycuvslam
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
>**Note**: for ubuntu 22.04 use `libstdcxx-ng=12.2.0` version

The `LD_LIBRARY_PATH` environment variable must be set every time you activate the
conda environment to ensure that the correct `libpython` library is loaded.

Proceed to [PyCuVSLAM installation](#pycuvslam-installation)

#### Option 4: Using Docker

PyCuVSLAM provides Docker support for both x86_64 and Jetson platforms with RealSense camera integration.

1. **Setup NGC (NVIDIA GPU Cloud):**
    ```bash
    docker login nvcr.io --username '$oauthtoken'
    ```
    For password, enter your NGC API key from: https://org.ngc.nvidia.com/setup/api-keys

2. **Clone the repository:**
    ```bash
    git clone https://github.com/NVlabs/pycuvslam.git
    cd pycuvslam
    ```

**For x86_64 (Desktop/Laptop):**

3. Build the x86 Docker image:
    ```bash
    docker build -f docker/Dockerfile.realsense-x86 -t pycuvslam:realsense-x86 .
    ```

4. Run the x86 container:
    ```bash
    bash docker/run_docker_x86.sh
    ```

**For Jetson (aarch64):**

3. Build the Jetson Docker image:
    ```bash
    docker build -f docker/Dockerfile.realsense-jetson -t pycuvslam:realsense-jetson .
    ```

4. Run the Jetson container:
    ```bash
    bash docker/run_docker_jetson.sh
    ```

**Features:**
- CUDA 12.6.1 support (Ubuntu 22.04)
- RealSense camera integration with librealsense
- X11 forwarding for GUI applications
- Automatic pycuvslam package installation
- USB device passthrough for camera access

### PyCuVSLAM Installation

1. Clone the PyCuVSLAM repository.
    ```bash
    git clone https://github.com/NVlabs/pycuvslam.git
    cd pycuvslam
    ```

2. Install the PyCuVSLAM package.
    ```bash
    pip install -e bin/x86_64
    ```
    For Jetson, use the following command:
    ```bash
    pip install -e bin/aarch64
    ```
3. Install PyCuVSLAM using one of the installation methods mentioned above, and then install the
   required packages for the examples:
    ```bash
    pip install -r examples/requirements.txt
    ```

## Examples and Guides

Explore various examples to quickly get started with PyCuVSLAM:

### Visual Tracking Mode Examples

- **Monocular Visual Odometry**
    - [EuRoC Dataset](examples/euroc/README.md)

- **Monocular-Depth Visual Odometry**
    - [TUM Dataset](examples/tum/README.md#running-monocular-depth-odometry)
    - [RealSense Live Camera](examples/realsense/README.md#running-monocular-depth-visual-odometry)
    - [ZED Live Camera](examples/zed/README.md#running-monocular-depth-visual-odometry)

- **Stereo Visual Odometry**
    - [KITTI Dataset](examples/kitti/README.md#running-pycuvslam-stereo-visual-odometry)
    - [RealSense Live Camera](examples/realsense/README.md#running-stereo-visual-odometry)
    - [ZED Live Camera](examples/zed/README.md#running-stereo-visual-odometry)
    - [OAK-D Live Camera](examples/oak-d/README.md#running-stereo-visual-odometry)

- **Stereo Visual-Inertial Odometry**
    - [EuRoC Dataset](examples/euroc/README.md#running-stereo-inertial-odometry)
    - [RealSense Live Camera](examples/realsense/README.md#running-stereo-inertial-odometry)

- **Multi-Camera Stereo Visual Odometry**
    - [Tartan Ground Dataset](examples/multicamera_edex/README.md#tartan-ground-dataset)
    - [R2B Galileo Dataset](examples/multicamera_edex/README.md#r2b-galileo-dataset)
    - [RealSense Live Camera](examples/realsense/README.md#running-multicamera-odometry)

### SLAM Examples

- [**Visual Mapping, Localization, and Map Saving/Loading**](examples/kitti/README.md#slam-mapping-collecting-storing-loading-and-localization)

### Advanced Features and Guides

- **Distorted Images**
    - [EuRoC Dataset](examples/euroc/README.md#distortion-models)
    - [OAK-D Live Camera](examples/oak-d/README.md#running-stereo-visual-odometry)
    - [ZED Live Camera](examples/zed/README.md#using-distorted-images)

- **Image Masking**
    - [Static Masks](examples/tum/README.md#masking-regions-to-prevent-feature-selection)
    - [Dynamic Masks](examples/kitti/README.md#dynamic-masks-with-pytorch-tensors)

- [**PyTorch GPU Tensor Handling**](examples/kitti/README.md#example-real-time-car-segmentation)

- [**RealSense Multi-Camera Assembly Guide**](examples/realsense/multicamera_hardware_assembly.md)

- [**Nvblox live 3D reconstruction**](https://nvidia-isaac.github.io/nvblox/pages/torch_examples_realsense.html#realsense-live-example)

## ROS2 Support

If you would like to use cuVSLAM in a ROS2 environment, please refer to the following links:
* [Isaac ROS cuVSLAM GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
* [Isaac ROS cuVSLAM Documentation](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html)
* [Isaac ROS cuVSLAM User Manual](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html)

## API Documentation and Technical Report

- For detailed API documentation, please visit the [PyCuVSLAM API Documentation](https://nvlabs.github.io/PyCuVSLAM/api.html)

- For technical details on the cuVSLAM algorithms, validation, and benchmarking results, refer to our [Technical Report](https://www.arxiv.org/abs/2506.04359)

## Performance and Troubleshooting

cuVSLAM is a highly optimized visual tracking library validated across numerous public datasets and popular robotic camera setups. For detailed benchmarking and validation results, please refer to our [technical report](https://arxiv.org/html/2506.04359v3#S3).

<img src="./assets/cuvslam_performance.png" alt="cuVSLAM performance" width="800" />

The accuracy and robustness of cuVSLAM can be influenced by several factors. If you experience performance issues, please check your system against these common causes:

- **Hardware Overload**: Hardware overload can negatively impact visual tracking, resulting in dropped frames or insufficient computational resources for cuVSLAM. Disable intensive visualization or image-saving operations to improve performance. For expected performance metrics on Jetson embedded platforms, see our [technical report](https://arxiv.org/html/2506.04359v3#A1.F13)

- **Intrinsic and Extrinsic Calibration**: Accurate camera calibration is crucial. Ensure your calibration parameters are precise. For more details, refer to our guide on [image undistortion](examples/euroc/README.md#distortion-models). If you're new to calibration, consider working with an [experienced vendors](https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/sensors/amr_extrinsic_calibration.html#extrinsic-calibration-of-sensors-in-custom-locations)

- **Synchronization and Timestamps**: Accurate synchronization significantly impacts cuVSLAM performance. Make sure multi-camera images are captured simultaneously—ideally through hardware synchronization—and verify correct relative timestamps across cameras. Refer to our [multi-camera hardware assembly guide](examples/realsense/multicamera_hardware_assembly.md) for building a rig with synchronized RealSense cameras

- **Frame Rate**: Frame rate significantly affects performance. The ideal frame rate depends on translational and rotational velocities. Typically, 30 FPS is suitable for most "human-speed" motions. Adjust accordingly for faster movements

- **Resolution**: Image resolution matters. VGA resolution or higher is recommended. cuVSLAM efficiently handles relatively high-resolution images due to CUDA acceleration

- **Image Quality**: Ensure good image quality by using suitable lenses, correct exposure, and proper white balance to avoid clipping large image regions. For significant distortion or external objects within the camera's field of view, please refer to our guide on [static masking](examples/kitti/README.md#static-masks)

- **Motion Blur**: Excessive motion blur can negatively impact tracking. Ensure that exposure times are short enough to minimize motion blur. If avoiding motion blur isn't feasible, consider increasing the frame rate or try the following [Mono-Depth](examples/realsense/README.md#running-monocular-depth-visual-odometry) or [Stereo Inertial](examples/realsense/README.md#running-stereo-inertial-odometry) tracking modes

### Troubleshooting FAQ

**Q**: When trying to run examples I get `ImportError: pycuvslam/cuvslam/x86/cuvslam/pycuvslam.so: invalid ELF header` 

**A**: You need Git LFS to correctly pull binary files: 
```bash
sudo apt-get install git-lfs
# in the repo directory:
git lfs install
git lfs pull
```

**Q**: Can I run PyCuVSLAM with Python 3.x?

**A**: We are working on supporting wider range of systems, but current version is only built for Python 3.10. We recommend using Docker or Conda for now.

### Reporting other issues
Are you having problems running PyCuVSLAM? Do you have any suggestions? We'd love to hear your feedback in the [issues](https://github.com/NVlabs/pycuvslam/issues) tab.

## License
This project is licensed under a non-commercial NVIDIA license, for details refer to the [LICENCE](LICENSE) file.

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{korovko2025cuvslam,
      title={cuVSLAM: CUDA accelerated visual odometry and mapping}, 
      author={Alexander Korovko and Dmitry Slepichev and Alexander Efitorov and Aigul Dzhumamuratova and Viktor Kuznetsov and Hesam Rabeti and Joydeep Biswas and Soha Pouya},
      year={2025},
      eprint={2506.04359},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.04359}, 
}
```