"""
Multi-Camera Vehicle Positioning System (VPS) runtime (ArUco-based).

- Opens three cameras concurrently (Jetson via GStreamer).
- Detects ArUco markers with OpenCV from all three cameras.
- Aggregates detections from all cameras for improved coverage.
- Sends tag pose updates to the VPFS backend via vpfs_connector.
- Shows live preview with marker overlays and FPS info for each camera.

Calibration:
- Attempts to load camera intrinsics from a JSON file (--calib path or
  camera_calibration.json next to this script). Falls back to hardcoded Brio 4K intrinsics.
"""

import sys
from ref_tags import ref_tags  # ensure tag registry is initialized
import vpfs_connector

import cv2
import numpy as np
import time
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from utils import (
    Defaults,
    ArucoDetection,
    find_camera,
    resolve_camera_intrinsics,
    draw_aruco_overlays,
    compute_camera_pos,
    compute_tag_poses
)

# Check for CUDA/GPU support
USE_GPU = True
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        USE_GPU = True
        print(f"GPU detected! CUDA-enabled devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        print(f"Using GPU acceleration for image processing")
except:
    print("GPU/CUDA not available, using CPU only")

# Pre-compute marker coordinate system (saves computation per frame)
OBJ_POINTS = np.array([[-Defaults.TAG_SIZE/2,  Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,   Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,  -Defaults.TAG_SIZE/2, 0],
                       [-Defaults.TAG_SIZE/2, -Defaults.TAG_SIZE/2, 0]], dtype=np.float32)


def initialize_camera(camera_id, CAM_K, CAM_D):
    """Initialize a single camera with proper settings."""
    camera_device = Defaults.CAMERA_SYMLINKS[camera_id]
    print(f"Initializing camera {camera_id} ({camera_device})...")
    
    # Reset to defaults first
    os.system(f"v4l2-ctl -d {camera_device} -c focus_automatic_continuous=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c focus_absolute=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c auto_exposure=1 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c exposure_time_absolute=200 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c brightness=128 2>/dev/null")

    # Create camera and set format before opening
    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(camera_device, cv2.CAP_V4L2)
    
    # Verify and reapply if needed
    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != Defaults.CAM_WIDTH or actual_h != Defaults.CAM_HEIGHT:
        print(f"First attempt: {actual_w}x{actual_h}, retrying with settings...")
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    
    # Camera control settings (applied via OpenCV as backup)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Disable autofocus
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure mode
    cam.set(cv2.CAP_PROP_EXPOSURE, 185)     # Set exposure

    max_fps = int(cam.get(cv2.CAP_PROP_FPS))

    # Log actual capture format
    frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera {camera_id}: {frameWidth}x{frameHeight} @ {max_fps} fps")
    
    # Log exposure settings
    exposure_value = cam.get(cv2.CAP_PROP_EXPOSURE)
    print(f"  Exposure: {exposure_value}")

    # Verify camera is available
    if not cam.isOpened():
        print(f"Cannot open camera {camera_id}")
        return None

    return cam


def process_camera_frame(frame, camera_id, camera_name, CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, gpu_frame=None, gpu_gray=None):
    """Process a single frame from one camera for ArUco detection."""
    # GPU-accelerated grayscale conversion if available
    if USE_GPU and gpu_frame is not None and gpu_gray is not None:
        gpu_frame.upload(frame)
        cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY, gpu_gray)
        gray = gpu_gray.download()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    rvecs, tvecs = None, None
    detections = []
    if ids is not None and len(ids) > 0:
        # Pose estimation for each marker using cv2.solvePnP
        # Use pre-computed objPoints to avoid re-allocation
        rvecs = []
        tvecs = []
        
        # Calculate pose for each marker
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corner, CAM_K, CAM_D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        # Build adapter objects to feed into utils
        for i, tag_id in enumerate(ids.flatten()):
            detections.append(
                ArucoDetection(
                    tag_id=tag_id,
                    rvec=rvecs[i].reshape(3),
                    tvec=tvecs[i].reshape(3),
                    corners=corners[i]
                )
            )

    # Overlay for visualization
    frame = draw_aruco_overlays(frame, corners, ids, CAM_K, CAM_D, Defaults.TAG_SIZE, rvecs, tvecs) if ids is not None else frame

    # Estimate camera pose from known reference tags (fused if multiple)
    cameraPos = compute_camera_pos(detections)
    
    # Add camera name and position overlay
    cv2.putText(frame, camera_name, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3, cv2.LINE_AA)
    if cameraPos is not None:
        pos_text = f"Pos: X{cameraPos[0]:.2f} Y{cameraPos[1]:.2f} Z{cameraPos[2]:.2f}"
        cv2.putText(frame, pos_text, (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frame, detections, cameraPos

def capture_and_process_camera(cam_info, CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, frame_time, gpu_frame=None, gpu_gray=None):
    """Worker function to capture and process a single camera frame in parallel."""
    ret, frame = cam_info["cap"].read()
    
    if not ret:
        frame = np.zeros((Defaults.CAM_HEIGHT, Defaults.CAM_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, f"{cam_info['name']} - NO SIGNAL", 
                  (50, Defaults.CAM_HEIGHT//2), cv2.FONT_HERSHEY_PLAIN, 
                  3, (0, 0, 255), 3, cv2.LINE_AA)
        return frame, [], None, 0.0
    
    # Process frame for ArUco detection
    frame, detections, cameraPos = process_camera_frame(
        frame, cam_info["id"], cam_info["name"], 
        CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
        gpu_frame, gpu_gray
    )
    
    # Calculate FPS per camera
    current_time = time.time()
    frameTime = current_time - frame_time
    fps = 1 / frameTime if frameTime > 0 else 0.0
    
    # Add FPS overlay
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{w}x{h} @ {fps:.1f}fps", 
               (10, h - 10), cv2.FONT_HERSHEY_PLAIN, 
               3, (255, 255, 255), 3, cv2.LINE_AA)
    
    return frame, detections, cameraPos, current_time


def main(argv=None):
    """Main loop for multi-camera visualization and processing."""
    # --- ArUco setup ---
    aruco = cv2.aruco
    # Pick a dictionary that matches your printed markers
    ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    ARUCO_PARAMS = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    DETECTOR     = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS) if hasattr(aruco, "ArucoDetector") else None

    # Camera configuration - process three cameras (IDs 0, 1, 2)
    CAMERA_IDS = [0, 1, 2]
    CAMERA_NAMES = ["Camera 1", "Camera 2", "Camera 3"]
    
    # Initialize all cameras with per-camera calibration
    cameras = []
    for cam_id, cam_name in zip(CAMERA_IDS, CAMERA_NAMES):
        try:
            # Load camera-specific intrinsics
            (in_fx, in_fy, in_cx, in_cy), CAM_D = resolve_camera_intrinsics(argv=argv, camera_id=cam_id)
            CAM_K = np.array([[in_fx, 0, in_cx], [0, in_fy, in_cy], [0, 0, 1]], dtype=np.float64)
            
            cam = initialize_camera(cam_id, CAM_K, CAM_D)
            if cam is not None:
                cameras.append({
                    "cap": cam,
                    "id": cam_id,
                    "name": cam_name,
                    "K": CAM_K,      # Store camera-specific intrinsics
                    "D": CAM_D       # Store camera-specific distortion
                })
        except Exception as e:
            print(f"Failed to initialize camera {cam_id}: {e}")
    
    if len(cameras) == 0:
        print("No cameras initialized. Exiting.")
        return
    
    print(f"\nInitialized {len(cameras)} cameras. Press ESC to quit.\n")
    print("Using multi-threaded processing for improved FPS...\n")
    
    # Pre-allocate GPU memory for each camera if GPU is available
    gpu_resources = []
    if USE_GPU:
        for _ in cameras:
            try:
                gpu_frame = cv2.cuda_GpuMat(Defaults.CAM_HEIGHT, Defaults.CAM_WIDTH, cv2.CV_8UC3)
                gpu_gray = cv2.cuda_GpuMat(Defaults.CAM_HEIGHT, Defaults.CAM_WIDTH, cv2.CV_8UC1)
                gpu_resources.append((gpu_frame, gpu_gray))
            except:
                gpu_resources.append((None, None))
    else:
        gpu_resources = [(None, None)] * len(cameras)
    
    # Main loop
    frame_times = [time.time()] * len(cameras)
    
    # Create thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        while True:
            # Submit all camera capture/process tasks in parallel
            futures = []
            for idx, cam_info in enumerate(cameras):
                gpu_frame, gpu_gray = gpu_resources[idx]
                future = executor.submit(
                    capture_and_process_camera,
                    cam_info, cam_info["K"], cam_info["D"], DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
                    frame_times[idx], gpu_frame, gpu_gray
                )
                futures.append((future, idx))
            
            # Collect results from all cameras
            frames = []
            all_detections = []
            all_tag_poses = {}
            
            for future, idx in futures:
                frame, detections, cameraPos, current_time = future.result()
                frames.append(frame)
                all_detections.extend(detections)
                frame_times[idx] = current_time
                
                # If we have a valid camera pose, transform all detected tags to map/world coords
                if cameraPos is not None:
                    tagPoses = compute_tag_poses(detections, cameraPos)
                    # Merge tag poses from this camera (later detections may override)
                    all_tag_poses.update(tagPoses)
            
            # Send aggregated tag poses to VPFS backend
            if all_tag_poses:
                vpfs_connector.send_update(all_tag_poses)
                print(f"Total tags detected: {len(all_tag_poses)}")
            
            # Display each camera in its own window
            for idx, (frame, cam_info) in enumerate(zip(frames, cameras)):
                # Resize to fit on screen (adjust scale as needed)
                # Use INTER_NEAREST for faster resize (less quality but much faster)
                scale = 0.25  # Adjust this to make windows larger/smaller
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                # Show in separate window for each camera
                cv2.imshow(cam_info["name"], resized)
            
            # Handle keyboard input
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    
    # Cleanup
    for cam_info in cameras:
        cam_info["cap"].release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # Parse command-line arguments using sys.argv for simplicity
    argv = sys.argv[1:]
    main(argv=argv)