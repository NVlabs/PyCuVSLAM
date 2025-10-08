#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights in and to this
software, related documentation, and any modifications thereto. Any use, reproduction, disclosure, or
distribution of this software and related documentation without an express license agreement from
NVIDIA CORPORATION is strictly prohibited.

Python implementation of EDEX file format reader.
This provides a minimal subset of edex::EdexFile and edex::Intrinsics from the C++ codebase.
"""

import json
import sys
import numpy as np
from typing import Dict, Any


class Intrinsics:
    """
    Camera intrinsics data structure, equivalent to edex::Intrinsics.
    
    Attributes:
        resolution: Image resolution [width, height]
        focal: Focal length [fx, fy]
        principal: Principal point [cx, cy]
        distortion_model: Name of distortion model (e.g., "pinhole", "brown5k", "fisheye")
        distortion_params: Array of distortion parameters
    """
    
    def __init__(self):
        self.resolution = np.array([0.0, 0.0], dtype=np.float32)
        self.focal = np.array([0.0, 0.0], dtype=np.float32)
        self.principal = np.array([0.0, 0.0], dtype=np.float32)
        self.distortion_model = ""
        self.distortion_params = np.array([], dtype=np.float32)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Intrinsics':
        """
        Create Intrinsics from dictionary (parsed from JSON).
        
        Args:
            data: Dictionary containing intrinsics data from EDEX file
            
        Returns:
            Intrinsics object populated with data from dictionary
        """
        intr = Intrinsics()
        
        # Handle both 'size' and 'resolution' keys
        if 'size' in data:
            size = data['size']
            intr.resolution = np.array([size[0], size[1]], dtype=np.float32)
        elif 'resolution' in data:
            res = data['resolution']
            intr.resolution = np.array([res[0], res[1]], dtype=np.float32)
        
        focal = data['focal']
        intr.focal = np.array([focal[0], focal[1]], dtype=np.float32)
        
        principal = data['principal']
        intr.principal = np.array([principal[0], principal[1]], dtype=np.float32)
        
        intr.distortion_model = data.get('distortion_model', 'pinhole')
        
        distortion_params = data.get('distortion_params', [])
        intr.distortion_params = np.array(distortion_params, dtype=np.float32)
        
        return intr
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Intrinsics to dictionary for JSON serialization.
        
        Returns:
            Dictionary containing intrinsics data
        """
        return {
            'size': [float(self.resolution[0]), float(self.resolution[1])],
            'focal': [float(self.focal[0]), float(self.focal[1])],
            'principal': [float(self.principal[0]), float(self.principal[1])],
            'distortion_model': self.distortion_model,
            'distortion_params': self.distortion_params.tolist()
        }
    
    def __str__(self) -> str:
        """String representation of Intrinsics."""
        return (f"Intrinsics(resolution={self.resolution}, "
                f"focal={self.focal}, "
                f"principal={self.principal}, "
                f"model={self.distortion_model}, "
                f"params={self.distortion_params})")


class EdexFile:
    """
    Simple EDEX file reader/writer, equivalent to edex::EdexFile (read-only subset).
    
    EDEX is a JSON-based format for storing camera intrinsics, poses, and trajectories.
    The file consists of a two-element array: [header, body]
    
    Attributes:
        version: EDEX format version (e.g., "0.9")
        cameras: List of camera data dictionaries, each containing:
            - intrinsics: Intrinsics object
            - transform: Camera transform (optional)
            - sequence: Image sequence list (optional)
        frame_start: Start frame number
        frame_end: End frame number
    """
    
    def __init__(self):
        self.version = ""
        self.cameras = []  # List of camera data dictionaries
        self.frame_start = 0
        self.frame_end = 0
    
    def read(self, filename: str) -> bool:
        """
        Read EDEX file from disk.
        
        Args:
            filename: Path to EDEX file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                root = json.load(f)
            
            if not isinstance(root, list) or len(root) != 2:
                print(f"Error: EDEX file is not a JSON array [header, body]. File is {filename}", 
                      file=sys.stderr)
                return False
            
            header = root[0]
            body = root[1]
            
            return self._read_header(header) and self._read_body(body, filename)
        
        except Exception as e:
            print(f"Error reading EDEX file {filename}: {e}", file=sys.stderr)
            return False
    
    def write(self, filename: str) -> bool:
        """
        Write EDEX file to disk.
        
        Args:
            filename: Path to output EDEX file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            root = [
                self._write_header(),
                self._write_body()
            ]
            
            with open(filename, 'w') as f:
                json.dump(root, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error writing EDEX file {filename}: {e}", file=sys.stderr)
            return False
    
    def _read_header(self, header: Dict[str, Any]) -> bool:
        """
        Read EDEX header.
        
        Args:
            header: Header dictionary from JSON
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.version = header.get('version', '0.9')
            self.frame_start = header.get('frame_start', 0)
            self.frame_end = header.get('frame_end', 0)
            
            cameras = header.get('cameras', [])
            for cam_data in cameras:
                camera = {
                    'intrinsics': Intrinsics.from_dict(cam_data['intrinsics']),
                    'transform': cam_data.get('transform', None),
                    'sequence': cam_data.get('sequence', [])
                }
                self.cameras.append(camera)
            
            return True
        
        except Exception as e:
            print(f"Error reading EDEX header: {e}", file=sys.stderr)
            return False
    
    def _read_body(self, body: Dict[str, Any], filename: str) -> bool:
        """
        Read EDEX body (minimal implementation for undistort tool).
        
        Args:
            body: Body dictionary from JSON
            filename: Original filename (for error messages)
            
        Returns:
            True if successful, False otherwise
        """
        # For undistort tool, we only need header information
        # Body parsing can be extended if needed
        return True
    
    def _write_header(self) -> Dict[str, Any]:
        """
        Create header dictionary for JSON serialization.
        
        Returns:
            Header dictionary
        """
        header = {
            'version': self.version,
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'cameras': []
        }
        
        for camera in self.cameras:
            cam_dict = {
                'intrinsics': camera['intrinsics'].to_dict()
            }
            
            if camera.get('transform') is not None:
                cam_dict['transform'] = camera['transform']
            
            if camera.get('sequence'):
                cam_dict['sequence'] = camera['sequence']
            
            header['cameras'].append(cam_dict)
        
        return header
    
    def _write_body(self) -> Dict[str, Any]:
        """
        Create body dictionary for JSON serialization.
        
        Returns:
            Body dictionary (minimal implementation)
        """
        # Minimal body for basic functionality
        return {}
    
    def __str__(self) -> str:
        """String representation of EdexFile."""
        return (f"EdexFile(version={self.version}, "
                f"cameras={len(self.cameras)}, "
                f"frames={self.frame_start}-{self.frame_end})")


def main():
    """Simple test/example of EDEX file reading."""
    import os
    
    # Test with example file if it exists
    example_file = "example_camera.edex"
    if os.path.exists(example_file):
        print(f"Reading {example_file}...")
        edex = EdexFile()
        if edex.read(example_file):
            print(f"✓ Successfully read EDEX file")
            print(f"  {edex}")
            
            if edex.cameras:
                print(f"\nCamera 0:")
                intr = edex.cameras[0]['intrinsics']
                print(f"  {intr}")
        else:
            print(f"✗ Failed to read EDEX file")
    else:
        print(f"Example file {example_file} not found.")
        print("\nCreating a test EDEX file...")
        
        # Create a simple test EDEX file
        edex = EdexFile()
        edex.version = "0.9"
        edex.frame_start = 0
        edex.frame_end = 10
        
        intr = Intrinsics()
        intr.resolution = np.array([1920, 1080], dtype=np.float32)
        intr.focal = np.array([1000.0, 1000.0], dtype=np.float32)
        intr.principal = np.array([960.0, 540.0], dtype=np.float32)
        intr.distortion_model = "pinhole"
        intr.distortion_params = np.array([], dtype=np.float32)
        
        camera = {
            'intrinsics': intr,
            'transform': None,
            'sequence': []
        }
        edex.cameras.append(camera)
        
        print(f"Created: {edex}")
        print(f"Camera 0: {intr}")


if __name__ == "__main__":
    main()

