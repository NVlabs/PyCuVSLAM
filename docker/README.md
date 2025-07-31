# Docker Build for PyCuVSLAM with RealSense Support

This directory contains the Docker setup for building PyCuVSLAM with RealSense camera support for both x86 and Jetson platforms.

## Available Files

### x86 Platform
- **`Dockerfile.realsense-x86`** - Complete Docker image for x86_64 with CUDA 12.6.1 runtime and RealSense support
- **`run_docker_x86.sh`** - Run script for x86 platform

### Jetson Platform
- **`Dockerfile.realsense-jetson`** - Complete Docker image for ARM64 (Jetson) with CUDA 12.6.11-devel and RealSense support
- **`run_docker_jetson.sh`** - Run script for Jetson platform

## Building the Images

### Build for x86 platform:
```bash
docker build -f docker/Dockerfile.realsense-x86 -t pycuvslam:realsense-x86 .
```

### Build for Jetson platform:
```bash
docker build -f docker/Dockerfile.realsense-jetson -t pycuvslam:realsense-jetson .
```

## Running the Containers

### For x86 platform:
```bash
./run_docker_x86.sh
```

### For Jetson platform:
```bash
./run_docker_jetson.sh
```
## Key Features

### RealSense Integration
- Builds librealsense from source with Python bindings
- Includes udev rules for RealSense camera access
- Sets up proper Python path for `pyrealsense2` import
- Tests the installation by importing the module

### GUI Support
- Includes all necessary X11 and GUI libraries for rerun-sdk
- Proper X11 forwarding setup for GUI applications
- Support for RealSense camera visualization

### Automatic Package Installation
- Automatically installs pycuvslam package in editable mode
- Installs all Python dependencies from requirements.txt
- Ready to run examples immediately after container startup

### Platform-Specific Features
- **x86**: Uses CUDA 12.6.1 runtime, installs from `/pycuvslam/bin/x86_64`
- **Jetson**: Uses CUDA 12.6.11-devel, installs from `/pycuvslam/bin/aarch64`

## Dependencies Included
- Build tools (cmake, make, gcc)
- Python development headers
- USB and graphics libraries
- X11 and GUI libraries for rerun-sdk
- CUDA runtime environment (12.6.1 for x86, 12.6.11-devel for Jetson)

## Usage Example

After starting the container try to run basic stereo tracking with realsense camera:
```bash
python3 examples/realsense/run_stereo.py
```
