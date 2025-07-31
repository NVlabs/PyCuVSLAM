#!/bin/bash

# Ensure X11 forwarding is set up properly
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# Create xauth file if it doesn't exist
touch $XAUTH
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH

# Set up datasets path (create if doesn't exist)
DATASETS=$(realpath -s ~/datasets 2>/dev/null || echo ~/datasets)
mkdir -p "$DATASETS"

# Allow X11 connections from localhost
xhost +local:docker

echo "Starting PycuVSLAM Docker container with RealSense support for Jetson..."
echo "X11 forwarding enabled for GUI applications"
echo "Datasets directory: $DATASETS"

docker run -it \
  --runtime=nvidia \
  --rm \
  --gpus all \
  --privileged \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=$XAUTH \
  -e QT_X11_NO_MITSHM=1 \
  -e _X11_NO_MITSHM=1 \
  -e _MITSHM=0 \
  -v $XSOCK:$XSOCK \
  -v $XAUTH:$XAUTH \
  -v .:/pycuvslam \
  -v "$DATASETS":"$DATASETS" \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  --device=/dev/video2:/dev/video2 \
  -v /dev/bus/usb:/dev/bus/usb \
  pycuvslam:realsense-jetson \
  bash -c "
    # Install pycuvslam package in editable mode if it exists
    if [ -d '/pycuvslam/bin/aarch64' ]; then
        echo 'Installing pycuvslam package in editable mode for Jetson...'
        pip3 install -e /pycuvslam/bin/aarch64
        echo 'pycuvslam package installed successfully'
    else
        echo 'Warning: /pycuvslam/bin/aarch64 not found. Skipping pycuvslam installation.'
    fi
    
    # Start bash shell
    exec bash
  "

# Clean up X11 permissions
xhost -local:docker 