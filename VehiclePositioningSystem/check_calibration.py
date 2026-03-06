#!/usr/bin/env python3
"""
Diagnostic tool to analyze camera extrinsic calibration files.

Usage:
    python check_calibration.py camera_extrinsics.json
"""

import sys
import json
import numpy as np


def analyze_calibration(filename):
    """Analyze and display information about a calibration file."""
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return
    
    print("="*70)
    print(f"Camera Extrinsic Calibration Analysis: {filename}")
    print("="*70)
    
    # Check for metadata
    if "_metadata" in data:
        meta = data["_metadata"]
        print("\nMetadata:")
        print(f"  Transform type: {meta.get('transform_type', 'unknown')}")
        print(f"  World frame: {meta.get('world_frame', 'unknown')}")
        print(f"  Units: {meta.get('units', 'unknown')}")
    
    # Analyze each camera
    cameras = {k: v for k, v in data.items() if not k.startswith('_')}
    
    if len(cameras) == 0:
        print("\nNo camera calibrations found in file!")
        return
    
    print(f"\nFound {len(cameras)} camera(s):\n")
    
    for cam_id, cam_data in sorted(cameras.items()):
        print(f"Camera {cam_id}:")
        
        # Extract transform matrix
        if "transform" not in cam_data:
            print("  ERROR: No transform matrix found!")
            continue
        
        T = np.array(cam_data["transform"])
        if T.shape != (4, 4):
            print(f"  ERROR: Invalid transform shape {T.shape}, expected (4,4)")
            continue
        
        # Extract position (translation component)
        position = T[:3, 3]
        x, y, z = position
        distance = np.linalg.norm(position)
        
        print(f"  Position (relative to marker 95):")
        print(f"    X = {x*100:7.1f} cm  ({x:.3f} m)")
        print(f"    Y = {y*100:7.1f} cm  ({y:.3f} m)")
        print(f"    Z = {z*100:7.1f} cm  ({z:.3f} m)")
        print(f"    Distance from origin: {distance*100:.1f} cm ({distance:.2f} m)")
        
        # Check if values seem reasonable
        warnings = []
        if distance > 5.0:
            warnings.append(f"Camera very far from origin ({distance:.1f}m)")
        if distance < 0.3:
            warnings.append(f"Camera very close to origin ({distance:.2f}m)")
        if abs(z) > 5.0:
            warnings.append(f"Unusual Z coordinate ({z:.1f}m)")
        
        # Check standard deviations if available
        if "std_dev_cm" in cam_data:
            std = cam_data["std_dev_cm"]
            print(f"  Calibration stability (std dev):")
            print(f"    X = {std['x']:.2f} cm")
            print(f"    Y = {std['y']:.2f} cm")
            print(f"    Z = {std['z']:.2f} cm")
            
            max_std = max(std['x'], std['y'], std['z'])
            if max_std > 5.0:
                warnings.append(f"High position variability ({max_std:.1f}cm)")
            elif max_std < 0.5:
                print(f"  ✓ Excellent stability!")
            elif max_std < 2.0:
                print(f"  ✓ Good stability")
        
        # Check sample count
        if "samples" in cam_data:
            samples = cam_data["samples"]
            print(f"  Samples collected: {samples}")
            if samples < 20:
                warnings.append(f"Few samples collected ({samples})")
        
        # Display warnings
        if warnings:
            print(f"  ⚠ WARNINGS:")
            for warning in warnings:
                print(f"    • {warning}")
        
        print()
    
    # Overall assessment
    print("="*70)
    print("Interpretation Guide:")
    print("="*70)
    print("""
The position values show where each camera is located relative to marker 95.

Typical setups:
  • Markers on table/floor: Z ≈ camera height (100-300cm)
  • Cameras mounted above: X, Y ≈ horizontal offsets (50-300cm)
  • Distance from origin: Usually 1-3 meters for indoor setups

If you see unusual values (>5m distances, very large Z, etc.):
  1. The calibration may be incorrect
  2. Markers might not be detected properly
  3. Camera intrinsic calibration might be wrong
  4. Wrong marker size (should be 10cm)

Solution: Delete the calibration file and run calibration again:
  python calibrate_camera_extrinsics.py --samples 50
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_calibration.py <calibration_file.json>")
        print("\nExample:")
        print("  python check_calibration.py camera_extrinsics.json")
        sys.exit(1)
    
    analyze_calibration(sys.argv[1])
