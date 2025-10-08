# Camera Models Python Module

This directory contains a complete Python implementation of camera models for lens distortion and undistortion.

## Files

- `camera.py` - Python implementation of camera models with distortion/undistortion capabilities
- `edex.py` - EDEX file format reader/writer with Intrinsics class
- `undistort.py` - Image undistortion tool for a single image
- `batch_undistort.py` - Batch processing script for undistorting multiple images
- `requirements.txt` - Python dependencies
- `input.edex` - Example EDEX file for testing
- `input.png` - Example image with distortion
- `output.png` - Example undistorted image

## Overview

The `camera.py` module provides camera models that handle the conversion between different coordinate spaces:

1. **Pixel coordinates** - Standard image coordinates (top-left is origin)
2. **UV coordinates** - Float image coordinates in pixels
3. **Normalized UV** - Coordinates after applying inverse calibration matrix
4. **XY coordinates** - Undistorted normalized coordinates (camera obscura model)

## Camera Models

### Base Class: `ICameraModel`

Abstract base class for all camera models. Provides:
- `normalize_point(uv)` - Convert pixel coordinates to undistorted normalized coordinates
- `denormalize_point(xy)` - Convert undistorted coordinates back to pixel coordinates
- `get_principal()` - Get principal point
- `get_focal()` - Get focal length
- `get_resolution()` - Get image resolution

### Implemented Models

1. **PinholeCameraModel** - Ideal pinhole camera (no distortion)

2. **FisheyeCameraModel** - Fisheye distortion with 4 coefficients (K1, K2, K3, K4)
   - Uses atan-based model
   - Newton-Raphson iterative undistortion

3. **Brown5KCameraModel** - Brown-Conrady distortion with 5 coefficients
   - 3 radial distortion coefficients (K1, K2, K3)
   - 2 tangential distortion coefficients (P1, P2)
   - Jacobian-based iterative undistortion

4. **PolynomialCameraModel** - Rational polynomial model
   - 6 radial distortion coefficients (K1-K6)
   - 2 tangential distortion coefficients (P1, P2)
   - Jacobian-based iterative undistortion

## EDEX Module

The `edex.py` module provides classes for working with EDEX files:

### Intrinsics Class

Represents camera intrinsics:
- `resolution` - Image resolution [width, height]
- `focal` - Focal length [fx, fy]
- `principal` - Principal point [cx, cy]
- `distortion_model` - Distortion model name
- `distortion_params` - Array of distortion parameters

Methods:
- `from_dict(data)` - Create from dictionary (parsed JSON)
- `to_dict()` - Convert to dictionary for JSON serialization

### EdexFile Class

EDEX file reader/writer:
- `version` - EDEX format version
- `cameras` - List of camera data
- `frame_start` - Start frame number
- `frame_end` - End frame number

Methods:
- `read(filename)` - Read EDEX file
- `write(filename)` - Write EDEX file

## Usage Examples

### Basic Usage

```python
import numpy as np
from camera import PinholeCameraModel, FisheyeCameraModel, create_camera_model

# Create a pinhole camera
resolution = np.array([1920, 1080], dtype=np.float32)
focal = np.array([1000.0, 1000.0], dtype=np.float32)
principal = np.array([960.0, 540.0], dtype=np.float32)

camera = PinholeCameraModel(resolution, focal, principal)

# Normalize a point (pixel to undistorted normalized coordinates)
uv = np.array([1000.0, 600.0], dtype=np.float32)
success, xy = camera.normalize_point(uv)
if success:
    print(f"Normalized coordinates: {xy}")

# Denormalize back (undistorted to pixel coordinates)
success, uv_back = camera.denormalize_point(xy)
if success:
    print(f"Back to pixel coordinates: {uv_back}")
```

### Using EDEX Files

```python
from edex import EdexFile, Intrinsics
from camera import create_camera_model

# Read EDEX file
edex = EdexFile()
if edex.read('camera.edex'):
    # Get intrinsics from first camera
    intr = edex.cameras[0]['intrinsics']
    
    # Create camera model
    camera = create_camera_model(
        intr.resolution, intr.focal, intr.principal,
        intr.distortion_model, intr.distortion_params
    )
    
    # Use camera model
    success, xy = camera.normalize_point(uv)
```

### Creating and Saving EDEX Files

```python
from edex import EdexFile, Intrinsics
import numpy as np

# Create intrinsics
intr = Intrinsics()
intr.resolution = np.array([1920, 1080], dtype=np.float32)
intr.focal = np.array([1000.0, 1000.0], dtype=np.float32)
intr.principal = np.array([960.0, 540.0], dtype=np.float32)
intr.distortion_model = "brown5k"
intr.distortion_params = np.array([0.1, -0.05, 0.01, 0.001, 0.002], dtype=np.float32)

# Create EDEX file
edex = EdexFile()
edex.version = "0.9"
edex.frame_start = 0
edex.frame_end = 100
edex.cameras.append({
    'intrinsics': intr,
    'transform': None,
    'sequence': []
})

# Save to file
edex.write('output.edex')
```

### Fisheye Camera

```python
from camera import FisheyeCameraModel

fisheye = FisheyeCameraModel(
    resolution, focal, principal,
    K1=-0.1, K2=0.01, K3=-0.001, K4=0.0001
)

success, xy = fisheye.normalize_point(uv)
```

### Brown-Conrady Model

```python
from camera import Brown5KCameraModel

brown = Brown5KCameraModel(
    resolution, focal, principal,
    K1=0.1, K2=-0.05, K3=0.01,  # radial
    P1=0.001, P2=0.002           # tangential
)

success, xy = brown.normalize_point(uv)
```

### Factory Function

```python
from camera import create_camera_model

# Create camera model from distortion model name and parameters
params = np.array([0.1, -0.05, 0.01, 0.001, 0.002], dtype=np.float32)
camera = create_camera_model(
    resolution, focal, principal,
    distortion_model="brown5k",
    parameters=params
)

if camera is not None:
    success, xy = camera.normalize_point(uv)
```

## Distortion Model Parameters

### Pinhole
- No parameters

### Fisheye/Fisheye4
- 4 parameters: [K1, K2, K3, K4]

### Brown5K
- 5 parameters: [K1, K2, K3, P1, P2]
- K1, K2, K3: radial distortion coefficients
- P1, P2: tangential distortion coefficients

### Polynomial
- 8 parameters: [K1, K2, P1, P2, K3, K4, K5, K6]
- Order matches OpenCV's `cv::initUndistortRectifyMap` and ROS messages
- K1-K6: radial distortion coefficients (rational polynomial)
- P1, P2: tangential distortion coefficients

## Dependencies

- NumPy

## Implementation Notes

1. **Coordinate System Convention**: The module follows the same coordinate system as cuVSLAM C APIs:
   - XY coordinates are double-flipped (flip x and flip y) from UV
   - This treats XY as an image from a camera obscura
   - Z points behind the camera, so moving forward makes Z negative

2. **Iterative Solvers**: Both fisheye and Jacobian-based models use Newton-Raphson iterative methods:
   - Maximum 10 iterations
   - Precision: ~1/100 of a pixel for F=1000px focal length
   - Early termination when convergence is achieved

3. **Numerical Stability**: 
   - Uses float32 (np.float32) for consistency with C++ version
   - Checks for invertibility of matrices
   - Validates input ranges to prevent numerical issues

## Undistort Tool

The `undistort.py` tool provides command-line image undistortion functionality for a single image.

### Usage

```bash
# Basic usage - input intrinsics and pinhole camera model are used for output
python3 undistort.py input.png input.edex output.png

# Undistort using custom output camera model
python3 undistort.py input.png input.edex output.png output.edex

# Use specific camera from input EDEX file
python3 undistort.py input.png input.edex output.png --camera 1
```

## Batch Undistort Tool

The `batch_undistort.py` tool processes multiple images in batch using the Python camera models directly.

### Usage

```bash
# Batch undistort all JPG images in a folder
python3 batch_undistort.py ~/raw "*.jpg" ~/stereo.edex 0 ~/undistorted

# Use specific camera from EDEX file
python3 batch_undistort.py ~/raw "cam0_right_*.jpg" ~/stereo.edex 1 ~/undistorted

# Process PNG images
python3 batch_undistort.py /path/to/images "frame_*.png" camera.edex 0 /path/to/output
```

### Arguments

- `in_folder` - Input folder containing images
- `in_images_mask` - Glob pattern for input images (e.g., "*.jpg", "frame_*.png")
- `in_edex` - Path to EDEX file with camera intrinsics
- `in_camera_id` - Camera index from EDEX file (starting from 0)
- `out_folder` - Output folder for undistorted images (created if doesn't exist)

### Features

- Automatically creates output folder
- Processes images to pinhole (undistorted) model
- Saves output as PNG files
- Shows progress with file counter
- Error handling for individual images (continues on failure)

### Undistort Tool Arguments

- `input_image` - Path to input distorted image
- `input_edex` - Path to EDEX file containing input camera intrinsics
- `output_image` - Path where undistorted image will be saved
- `output_edex` (optional) - Path to EDEX file with output camera model. If not provided, uses pinhole model with input intrinsics
- `--camera N` or `-c N` - Camera index from input EDEX file (default: 0)

### How It Works

1. Reads camera intrinsics from EDEX file(s)
2. Creates input and output camera models
3. For each pixel in the output image:
   - Converts output pixel coordinates to normalized undistorted coordinates using output model
   - Converts normalized coordinates to input pixel coordinates using input model
4. Uses OpenCV's remap to generate the final undistorted image

### EDEX File Format

The tool reads EDEX files which are JSON files with the following structure:

```json
[
  {
    "version": "0.9",
    "frame_start": 0,
    "frame_end": 100,
    "cameras": [
      {
        "intrinsics": {
          "size": [1920, 1080],
          "focal": [1000.0, 1000.0],
          "principal": [960.0, 540.0],
          "distortion_model": "brown5k",
          "distortion_params": [0.1, -0.05, 0.01, 0.001, 0.002]
        }
      }
    ]
  },
  { /* body */ }
]
```

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

