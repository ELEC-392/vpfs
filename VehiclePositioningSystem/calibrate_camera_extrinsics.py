"""
Camera Extrinsic Calibration Script for Multi-Camera System

This script calibrates the extrinsic transformations between cameras using
the reference markers (95-99). It computes each camera's pose relative to
the world coordinate frame defined by the markers.

Usage:
    python calibrate_camera_extrinsics.py
    python calibrate_camera_extrinsics.py --samples 50  # Collect 50 samples per camera

Output:
    camera_extrinsics.json - Contains camera-to-world transforms for each camera

Process:
    1. Place all reference markers (95-99) visible to all cameras
    2. Marker 95 is at origin, marker 96 defines X-axis
    3. Script captures multiple frames from each camera
    4. Computes average camera pose for stability
    5. Saves transforms to JSON file
"""

import sys
import cv2
import numpy as np
import time
import json
import os
from collections import defaultdict

from utils import (
    Defaults,
    ArucoDetection,
    resolve_camera_intrinsics,
    det_to_transform_mat
)

# Pre-compute marker coordinate system
OBJ_POINTS = np.array([[-Defaults.TAG_SIZE/2,  Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,   Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,  -Defaults.TAG_SIZE/2, 0],
                       [-Defaults.TAG_SIZE/2, -Defaults.TAG_SIZE/2, 0]], dtype=np.float32)


def initialize_camera(camera_id, CAM_K, CAM_D):
    """Initialize a single camera."""
    camera_device = Defaults.CAMERA_SYMLINKS[camera_id]
    print(f"Initializing camera {camera_id} ({camera_device})...")
    
    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(camera_device, cv2.CAP_V4L2)
    
    if not cam.isOpened():
        print(f"  Cannot open camera {camera_id}")
        return None
    
    print(f"  Camera {camera_id} initialized")
    return cam


def detect_markers(frame, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, CAM_K, CAM_D):
    """Detect ArUco markers in a frame and return detections."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    
    detections = []
    if ids is not None and len(ids) > 0:
        for i, tag_id in enumerate(ids.flatten()):
            success, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corners[i], CAM_K, CAM_D, 
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                detections.append(
                    ArucoDetection(
                        tag_id=tag_id,
                        rvec=rvec.reshape(3),
                        tvec=tvec.reshape(3),
                        corners=corners[i]
                    )
                )
    
    return detections


def compute_camera_to_world_transform(detections):
    """
    Compute camera pose in world frame using reference markers.
    
    World frame:
    - Origin at marker 95
    - X-axis along direction from marker 95 to marker 96
    - Planar (all markers on same plane)
    """
    ORIGIN_MARKER_ID = 95
    X_AXIS_MARKER_ID = 96
    
    # Find origin marker
    origin_detection = None
    x_axis_detection = None
    
    for det in detections:
        if det.tag_id == ORIGIN_MARKER_ID:
            origin_detection = det
        if det.tag_id == X_AXIS_MARKER_ID:
            x_axis_detection = det
    
    if origin_detection is None:
        return None, "Origin marker 95 not detected"
    
    # Get camera->origin transform
    cam_to_origin = det_to_transform_mat(origin_detection)
    origin_to_cam = np.linalg.inv(cam_to_origin)
    
    # If we also have marker 96, align X-axis
    if x_axis_detection is not None:
        cam_to_x_marker = det_to_transform_mat(x_axis_detection)
        origin_to_x_marker = np.matmul(origin_to_cam, cam_to_x_marker)
        
        # Get direction from origin to marker 96 in world frame
        direction = origin_to_x_marker[:3, 3]
        
        # Project to XY plane and compute angle
        angle = np.arctan2(direction[1], direction[0])
        
        # Create rotation matrix to align with +X axis
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        R_align = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a,  cos_a, 0, 0],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ])
        
        # Apply alignment
        origin_to_cam = np.matmul(R_align, origin_to_cam)
    
    return origin_to_cam, None


def main():
    # Parse arguments
    num_samples = 50
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            num_samples = int(sys.argv[idx + 1])
    
    print("="*70)
    print("Camera Extrinsic Calibration")
    print("="*70)
    print(f"\nThis script will collect {num_samples} samples from each camera")
    print("to compute stable camera-to-world transformations.\n")
    print("Requirements:")
    print("  • All reference markers (95-99) must be visible to all cameras")
    print("  • Markers should be stationary")
    print("  • Marker 95 defines the origin")
    print("  • Marker 96 defines the X-axis direction\n")
    
    input("Press ENTER when markers are in position and ready to calibrate...")
    
    # ArUco setup
    aruco = cv2.aruco
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    ARUCO_PARAMS = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    DETECTOR = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS) if hasattr(aruco, "ArucoDetector") else None
    
    # Initialize cameras
    CAMERA_IDS = [0, 1, 2]
    cameras = []
    
    for cam_id in CAMERA_IDS:
        try:
            (in_fx, in_fy, in_cx, in_cy), CAM_D = resolve_camera_intrinsics(argv=sys.argv[1:], camera_id=cam_id)
            CAM_K = np.array([[in_fx, 0, in_cx], [0, in_fy, in_cy], [0, 0, 1]], dtype=np.float64)
            
            cam = initialize_camera(cam_id, CAM_K, CAM_D)
            if cam is not None:
                cameras.append({
                    "id": cam_id,
                    "cap": cam,
                    "K": CAM_K,
                    "D": CAM_D
                })
        except Exception as e:
            print(f"Failed to initialize camera {cam_id}: {e}")
    
    if len(cameras) == 0:
        print("\nERROR: No cameras initialized. Exiting.")
        return
    
    print(f"\nCalibrating {len(cameras)} cameras...")
    print(f"Collecting {num_samples} samples per camera...\n")
    
    # Warm up cameras (let auto-exposure settle, flush initial buffers)
    print("Warming up cameras (auto-exposure settling)...", end=" ", flush=True)
    for _ in range(10):
        for cam_info in cameras:
            cam_info["cap"].grab()
        time.sleep(0.1)
    print("✓\n")
    
    # Collect samples
    camera_transforms = {cam["id"]: [] for cam in cameras}
    sample_stats = {cam["id"]: {"success": 0, "failed": 0, "no_markers": 0} for cam in cameras}
    
    for sample_idx in range(num_samples):
        print(f"Sample {sample_idx + 1}/{num_samples}...", end=" ", flush=True)
        
        for cam_info in cameras:
            # CRITICAL: Flush old frames from buffer to get fresh capture
            # This prevents reading stale buffered frames
            for _ in range(2):  # Discard 2 old frames
                cam_info["cap"].grab()
            
            # Capture fresh frame
            ret = cam_info["cap"].grab()
            if ret:
                ret, frame = cam_info["cap"].retrieve()
            else:
                frame = None
            
            if not ret or frame is None:
                sample_stats[cam_info["id"]]["failed"] += 1
                continue
            
            # Detect markers
            detections = detect_markers(frame, DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
                                       cam_info["K"], cam_info["D"])
            
            if len(detections) == 0:
                sample_stats[cam_info["id"]]["no_markers"] += 1
                continue
            
            # Compute transform
            transform, error = compute_camera_to_world_transform(detections)
            
            if transform is not None:
                camera_transforms[cam_info["id"]].append(transform)
                sample_stats[cam_info["id"]]["success"] += 1
            else:
                # Error computing transform (missing marker 95 or 96)
                sample_stats[cam_info["id"]]["no_markers"] += 1
        
        print("✓")
        time.sleep(0.1)  # Delay between samples (allow cameras to capture fresh frames)
    
    # Print collection summary
    print("\n" + "="*70)
    print("Sample Collection Summary:")
    print("="*70)
    for cam_id in sorted(sample_stats.keys()):
        stats = sample_stats[cam_id]
        success_rate = (stats["success"] / num_samples) * 100 if num_samples > 0 else 0
        print(f"Camera {cam_id}:")
        print(f"  Successful: {stats['success']}/{num_samples} ({success_rate:.1f}%)")
        if stats["failed"] > 0:
            print(f"  Capture failed: {stats['failed']}")
        if stats["no_markers"] > 0:
            print(f"  No markers/missing ref markers: {stats['no_markers']}")
    print()
    
    # Cleanup cameras
    for cam_info in cameras:
        cam_info["cap"].release()
    
    # Compute average transforms
    print("\n" + "="*70)
    print("Computing average transforms...")
    print("="*70 + "\n")
    
    calibration_results = {}
    
    for cam_id in camera_transforms:
        transforms = camera_transforms[cam_id]
        
        if len(transforms) == 0:
            print(f"Camera {cam_id}: ERROR - No valid samples collected!")
            continue
        
        print(f"Camera {cam_id}: {len(transforms)}/{num_samples} valid samples")
        
        # Average rotation matrices using SVD
        rotations = [T[:3, :3] for T in transforms]
        M = np.mean(rotations, axis=0)
        U, _, Vt = np.linalg.svd(M)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1
            R_avg = U @ Vt
        
        # Average translations
        translations = [T[:3, 3] for T in transforms]
        t_avg = np.mean(translations, axis=0)
        
        # Build averaged transform
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg
        
        # Compute standard deviations for quality check
        t_std = np.std(translations, axis=0)
        
        print(f"  Position: X={t_avg[0]*100:.1f}cm Y={t_avg[1]*100:.1f}cm Z={t_avg[2]*100:.1f}cm")
        print(f"  Std Dev:  X={t_std[0]*100:.2f}cm Y={t_std[1]*100:.2f}cm Z={t_std[2]*100:.2f}cm")
        
        # Store result
        calibration_results[str(cam_id)] = {
            "transform": T_avg.tolist(),
            "samples": len(transforms),
            "std_dev_cm": {
                "x": float(t_std[0] * 100),
                "y": float(t_std[1] * 100),
                "z": float(t_std[2] * 100)
            }
        }
    
    if len(calibration_results) == 0:
        print("\nERROR: No cameras were successfully calibrated!")
        return
    
    # Save to JSON
    output_file = "camera_extrinsics.json"
    with open(output_file, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Calibration complete!")
    print(f"{'='*70}")
    print(f"\nCalibration saved to: {output_file}")
    print(f"\nTo use this calibration:")
    print(f"  python vehicle_position_system_multicam.py --extrinsics {output_file}")
    print()


if __name__ == "__main__":
    main()
