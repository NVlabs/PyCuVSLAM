#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights in and to this
software, related documentation, and any modifications thereto. Any use, reproduction, disclosure, or
distribution of this software and related documentation without an express license agreement from
NVIDIA CORPORATION is strictly prohibited.

Python implementation of image undistortion tool.
This is equivalent to main.cpp from the C++ codebase.
"""

import argparse
import sys
import numpy as np
import cv2
from typing import Optional, Tuple

from camera import ICameraModel, create_camera_model
from edex import Intrinsics, EdexFile


def create_camera_model_from_intrinsics(intrinsics: Intrinsics) -> Optional[ICameraModel]:
    """Create camera model from intrinsics."""
    return create_camera_model(
        intrinsics.resolution,
        intrinsics.focal,
        intrinsics.principal,
        intrinsics.distortion_model,
        intrinsics.distortion_params
    )


def undistort_image(input_model: ICameraModel, output_model: ICameraModel,
                   input_image: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """
    Undistort image using input and output camera models.
    
    Args:
        input_model: Camera model for input image
        output_model: Camera model for output image
        input_image: Input image to undistort
        output_shape: Output image shape (height, width)
        
    Returns:
        Undistorted output image
    """
    height, width = output_shape
    
    # Create remapping grids
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    # For each pixel in output image, find corresponding pixel in input image
    for y in range(height):
        for x in range(width):
            dst = np.array([x, y], dtype=np.float32)
            
            # Convert output pixel to normalized coordinates
            success, interim = output_model.normalize_point(dst)
            if not success:
                map_x[y, x] = -1
                map_y[y, x] = -1
                continue
            
            # Convert normalized coordinates to input pixel coordinates
            success, src = input_model.denormalize_point(interim)
            if not success:
                map_x[y, x] = -1
                map_y[y, x] = -1
                continue
            
            map_x[y, x] = src[0]
            map_y[y, x] = src[1]
    
    # Apply remapping
    output_image = cv2.remap(input_image, map_x, map_y, cv2.INTER_LINEAR)
    
    return output_image


def check_condition(condition: bool, message: str):
    """Check condition and exit with error message if false."""
    if not condition:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Undistort images using camera models from EDEX files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Undistort to pinhole (no output edex)
  %(prog)s input.png input.edex output.png

  # Undistort using custom output camera model
  %(prog)s input.png input.edex output.png output.edex

  # Use specific camera from input edex
  %(prog)s input.png input.edex output.png --camera 1
        '''
    )
    
    parser.add_argument('input_image', help='Input image file path')
    parser.add_argument('input_edex', help='Input EDEX file path with camera intrinsics')
    parser.add_argument('output_image', help='Output image file path')
    parser.add_argument('output_edex', nargs='?', default='',
                       help='Output EDEX file path (optional). If not set, input intrinsics and pinhole model are used.')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera number from input EDEX (default: 0)')
    
    args = parser.parse_args()
    
    # Read input EDEX file
    input_edex_file = EdexFile()
    check_condition(input_edex_file.read(args.input_edex), f"Can't read {args.input_edex}")
    check_condition(0 <= args.camera < len(input_edex_file.cameras),
                   f"No camera {args.camera} in {args.input_edex}")
    
    input_intr = input_edex_file.cameras[args.camera]['intrinsics']
    
    # Read output EDEX file or use input intrinsics with pinhole model
    if args.output_edex:
        output_edex_file = EdexFile()
        check_condition(output_edex_file.read(args.output_edex), f"Can't read {args.output_edex}")
        check_condition(len(output_edex_file.cameras) > 0, f"No camera in {args.output_edex}")
        output_intr = output_edex_file.cameras[0]['intrinsics']
    else:
        # Use input intrinsics but with pinhole model (no distortion)
        output_intr = Intrinsics()
        output_intr.resolution = input_intr.resolution.copy()
        output_intr.focal = input_intr.focal.copy()
        output_intr.principal = input_intr.principal.copy()
        output_intr.distortion_model = "pinhole"
        output_intr.distortion_params = np.array([], dtype=np.float32)
    
    # Create camera models
    input_model = create_camera_model_from_intrinsics(input_intr)
    check_condition(input_model is not None, "Cannot create input camera model")
    
    output_model = create_camera_model_from_intrinsics(output_intr)
    check_condition(output_model is not None, "Cannot create output camera model")
    
    # Load input image
    input_image = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    check_condition(input_image is not None, f"Cannot load input image: {args.input_image}")
    
    # Get output size
    output_height = int(output_intr.resolution[1])
    output_width = int(output_intr.resolution[0])
    
    # Undistort image
    print(f"Undistorting image...")
    print(f"  Input model: {input_intr.distortion_model}")
    print(f"  Output model: {output_intr.distortion_model}")
    print(f"  Input size: {input_image.shape[1]}x{input_image.shape[0]}")
    print(f"  Output size: {output_width}x{output_height}")
    
    output_image = undistort_image(input_model, output_model, input_image, (output_height, output_width))
    
    # Save output image
    success = cv2.imwrite(args.output_image, output_image)
    check_condition(success, f"Cannot save output image: {args.output_image}")
    
    print(f"Successfully saved undistorted image to {args.output_image}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

