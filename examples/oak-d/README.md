# Tutorial: Running PyCuVSLAM Stereo Visual Odometry on OAK-D Stereo Camera

This tutorial demonstrates how to perform live PyCuVSLAM tracking using unrectified stereo images captured from an OAK-D stereo camera

> **Notes:**
> * The provided script has been developed and validated on the OAK-D W Pro stereo camera. Distortion models and order of cameras and its frames may differ for other OAK-D models. For more information about distortion models supported by cuVSLAM, see the [EuroC tutorial](../euroc/README.md#distortion-models).
> * **Global shutter** is a fundamental requirement for cuVSLAM. Please ensure your camera uses a global shutter sensor.

## Static Masks to Improve Visual Tracking

When using unrectified stereo images for visual tracking, residual geometric distortion may persist near the peripheral regions even after distortion correction. To mitigate this and improve tracking quality, we recommend applying static masks around the outer frame. These masks prevent PyCuVSLAM from selecting unreliable features near distorted image borders.

<img src="../../assets/tutorial_raw_images.png" alt="Static Masks Example" width="600" />

The figure above illustrates an example fisheye image from the [TUM-VI dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset). The original distorted fisheye image (left) and the corresponding undistorted image (right) are shown. The implemented static masks are indicated by red transparent borders. Each border (top, bottom, left, right) have independently specified thicknesses, allowing flexibility to mask out distorted regions appropriately.

To define these outer mask borders, specify pixel values independently for each camera instance as follows:

```
cam = vslam.Camera()
cam.border_top = 20
cam.border_bottom = 30
cam.border_left = 30
cam.border_right = 50
```

## Setting up the cuvslam environment
Refer to the [Installation Guide](../../README.md#pycuvslam-installation) for instructions on installing and configuring all required dependencies

## Setting up DepthAI
1. Install the [DepthAI Python library](https://github.com/luxonis/depthai-python) following the official documentation
2. Test your setup by running a basic [camera example](https://docs.luxonis.com/software/depthai/examples/rgb_preview/). Ensure it works correctly before proceeding

## Running Stereo Visual Odometry

```bash
python3 examples/oak-d/run_stereo.py
```

You should see the following interactive visualization in rerun: 
![Visualization Example](../../assets/tutorial_oakd_stereo.gif)

> **Note**: The PyCuVSLAM stereo tracker expects reliably synchronized stereo pairs with a stable FPS. If your camera pipeline is doing extensive on-device processing or AI inference, frame rates may drop, and image pairs may become unsynchronized. Watch for warnings about low FPS or mismatched stereo frames

If you experience low FPS even in the basic setup, you can investigate potential bottlenecks using the supplied [measurement tools](https://docs.luxonis.com/software/depthai/optimizing/) for OAK devices 