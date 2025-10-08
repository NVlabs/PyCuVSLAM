#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights in and to this
software, related documentation, and any modifications thereto. Any use, reproduction, disclosure, or
distribution of this software and related documentation without an express license agreement from
NVIDIA CORPORATION is strictly prohibited.

Batch undistort images using Python camera models.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

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
                   input_image: np.ndarray, output_shape: tuple) -> np.ndarray:
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


def batch_undistort(in_folder, in_images_mask, in_edex, in_camera_id, out_folder):
    """
    Batch undistort images from a folder.
    
    Args:
        in_folder: Input folder containing images
        in_images_mask: Glob pattern for input images (e.g., "*.jpg")
        in_edex: Path to EDEX file with camera intrinsics
        in_camera_id: Camera ID to use from EDEX file (as string)
        out_folder: Output folder for undistorted images
    """
    # Read EDEX file
    edex_file = EdexFile()
    if not edex_file.read(in_edex):
        print(f"Error: Cannot read EDEX file: {in_edex}", file=sys.stderr)
        sys.exit(1)
    
    camera_id = int(in_camera_id)
    if camera_id < 0 or camera_id >= len(edex_file.cameras):
        print(f"Error: Camera {camera_id} not found in EDEX file (available: 0-{len(edex_file.cameras)-1})",
              file=sys.stderr)
        sys.exit(1)
    
    # Get input intrinsics
    input_intr = edex_file.cameras[camera_id]['intrinsics']
    
    # Create output intrinsics (pinhole model)
    output_intr = Intrinsics()
    output_intr.resolution = input_intr.resolution.copy()
    output_intr.focal = input_intr.focal.copy()
    output_intr.principal = input_intr.principal.copy()
    output_intr.distortion_model = "pinhole"
    output_intr.distortion_params = np.array([], dtype=np.float32)
    
    # Create camera models
    input_model = create_camera_model_from_intrinsics(input_intr)
    if input_model is None:
        print(f"Error: Cannot create input camera model", file=sys.stderr)
        sys.exit(1)
    
    output_model = create_camera_model_from_intrinsics(output_intr)
    if output_model is None:
        print(f"Error: Cannot create output camera model", file=sys.stderr)
        sys.exit(1)
    
    # Get output size
    output_height = int(output_intr.resolution[1])
    output_width = int(output_intr.resolution[0])
    
    print(f"Batch undistort configuration:")
    print(f"  Input model: {input_intr.distortion_model}")
    print(f"  Output model: {output_intr.distortion_model}")
    print(f"  Output size: {output_width}x{output_height}")
    print(f"  Camera ID: {camera_id}")
    print()
    
    # Create output folder
    out_folder_p = Path(out_folder).expanduser()
    out_folder_p.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    in_folder_p = Path(in_folder).expanduser()
    image_files = sorted(in_folder_p.glob(in_images_mask))
    
    if not image_files:
        print(f"Warning: No images found matching pattern '{in_images_mask}' in {in_folder}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for i, in_image_path in enumerate(image_files, 1):
        out_image_path = out_folder_p / in_image_path.with_suffix('.png').name
        
        try:
            # Load input image
            input_image = cv2.imread(str(in_image_path), cv2.IMREAD_UNCHANGED)
            if input_image is None:
                print(f"  [{i}/{len(image_files)}] ✗ Failed to load: {in_image_path.name}", file=sys.stderr)
                continue
            
            # Undistort image
            output_image = undistort_image(input_model, output_model, input_image, 
                                         (output_height, output_width))
            
            # Save output image
            success = cv2.imwrite(str(out_image_path), output_image)
            if not success:
                print(f"  [{i}/{len(image_files)}] ✗ Failed to save: {out_image_path.name}", file=sys.stderr)
                continue
            
            print(f"  [{i}/{len(image_files)}] ✓ {in_image_path.name} -> {out_image_path.name}")
            
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] ✗ Error processing {in_image_path.name}: {e}", file=sys.stderr)
            continue
    
    print(f"\nBatch processing complete!")


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("usage: batch_undistort in_folder in_images_mask in_edex in_camera_id out_folder")
        print("  in_camera_id - starting from 0")
        print("example: batch_undistort ~/raw 'cam0_right_*.jpg' ~/stereo.edex 1 ~/undistorted")
        exit(1)

    batch_undistort(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    exit(0)
