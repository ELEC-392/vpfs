"""
Multi-camera Vehicle Positioning System
Displays three BRIO cameras simultaneously for visualization
"""

import cv2
import numpy as np
import time
import os
import sys
import platform
import subprocess
from dataclasses import dataclass

# Import from original file
from vehicle_position_system import (
    ArucoDetection,
    utils,
    vpfs_connector,
    CAM_K,
    CAM_D,
    tag_size,
    aruco_dict,
    aruco_params
)

# Camera configuration
CAMERAS = [
    {"device": "/dev/brio-camera1", "name": "Camera 1"},
    {"device": "/dev/brio-camera2", "name": "Camera 2"},
    {"device": "/dev/brio-camera3", "name": "Camera 3"}
]

camera_width = 4096
camera_height = 2160

def initialize_camera(device_path):
    """Initialize a single camera with proper settings."""
    print(f"Initializing {device_path}...")
    
    # Set camera controls via v4l2-ctl
    os.system(f"v4l2-ctl -d {device_path} -c focus_automatic_continuous=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {device_path} -c focus_absolute=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {device_path} -c auto_exposure=1 2>/dev/null")
    os.system(f"v4l2-ctl -d {device_path} -c exposure_time_absolute=200 2>/dev/null")
    os.system(f"v4l2-ctl -d {device_path} -c brightness=128 2>/dev/null")
    
    # Open camera
    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cam.open(device_path, cv2.CAP_V4L2)
    
    # Verify settings
    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cam.get(cv2.CAP_PROP_FPS))
    
    print(f"  {device_path}: {actual_w}x{actual_h} @ {actual_fps} fps")
    
    # Camera control settings
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cam.set(cv2.CAP_PROP_EXPOSURE, 85)
    
    return cam

def draw_aruco_overlays(img, corners, ids, rvecs=None, tvecs=None, camera_name=""):
    """Draw ArUco markers and detection info on frame."""
    font = cv2.FONT_HERSHEY_PLAIN
    
    if corners is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
        if rvecs is not None and tvecs is not None and len(rvecs) > 0:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(img, CAM_K, CAM_D, rvec, tvec, tag_size * 0.5)
        
        # Add text with relative position and distance
        for i, corner in enumerate(corners):
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            
            if tvecs is not None and i < len(tvecs):
                tvec = tvecs[i]
                distance = np.linalg.norm(tvec)
                text = f"ID{ids[i][0]}: {distance:.2f}m"
                cv2.putText(img, text, (center_x, center_y - 10), 
                          font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Add camera name
    cv2.putText(img, camera_name, (10, 30), font, 2, (255, 255, 0), 2, cv2.LINE_AA)
    
    return img

def process_frame(frame, camera_name):
    """Process a single frame for ArUco detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )
    
    detections = []
    rvecs = []
    tvecs = []
    
    if ids is not None:
        for i, corner in enumerate(corners):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corner], tag_size, CAM_K, CAM_D
            )
            if rvec is not None and tvec is not None:
                rvecs.append(rvec)
                tvecs.append(tvec)
                
                tag_id = ids[i][0]
                detections.append(
                    ArucoDetection(
                        tag_id=tag_id,
                        rvec=rvec.reshape(3),
                        tvec=tvec.reshape(3),
                        corners=corner
                    )
                )
    
    # Draw overlays
    frame = draw_aruco_overlays(frame, corners, ids, rvecs, tvecs, camera_name)
    
    # Estimate camera pose
    cameraPos = utils.compute_camera_pos(detections)
    
    # Display camera position
    if cameraPos is not None:
        pos_text = f"Pos: X{cameraPos[0]:.2f} Y{cameraPos[1]:.2f} Z{cameraPos[2]:.2f}"
        cv2.putText(frame, pos_text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 
                   1.5, (0, 255, 255), 2, cv2.LINE_AA)
    
    return frame, detections, cameraPos

def main():
    """Main loop for multi-camera visualization."""
    # Initialize all cameras
    cameras = []
    for cam_config in CAMERAS:
        try:
            cam = initialize_camera(cam_config["device"])
            cameras.append({
                "cap": cam,
                "name": cam_config["name"],
                "device": cam_config["device"]
            })
        except Exception as e:
            print(f"Failed to initialize {cam_config['device']}: {e}")
    
    if len(cameras) == 0:
        print("No cameras initialized. Exiting.")
        return
    
    print(f"\nInitialized {len(cameras)} cameras. Press 'q' to quit.\n")
    
    # Main loop
    frame_times = [time.time()] * len(cameras)
    
    while True:
        frames = []
        all_detections = []
        
        # Capture and process frames from all cameras
        for idx, cam_info in enumerate(cameras):
            ret, frame = cam_info["cap"].read()
            
            if not ret:
                print(f"Failed to read from {cam_info['name']}")
                frame = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
                cv2.putText(frame, f"{cam_info['name']} - NO SIGNAL", 
                          (50, camera_height//2), cv2.FONT_HERSHEY_PLAIN, 
                          3, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                frame, detections, cameraPos = process_frame(frame, cam_info["name"])
                all_detections.extend(detections)
            
            # Calculate FPS per camera
            current_time = time.time()
            frameTime = current_time - frame_times[idx]
            fps = 1 / frameTime if frameTime > 0 else 0.0
            frame_times[idx] = current_time
            
            h, w = frame.shape[:2]
            cv2.putText(frame, f"{w}x{h} @ {fps:.1f}fps", 
                       (10, h - 10), cv2.FONT_HERSHEY_PLAIN, 
                       1.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            frames.append(frame)
        
        # Display each camera in its own window
        for idx, (frame, cam_info) in enumerate(zip(frames, cameras)):
            # Resize to fit on screen (adjust as needed)
            scale = 0.4  # Adjust this to make windows larger/smaller
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Show in separate window for each camera
            cv2.imshow(cam_info["name"], resized)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    for cam_info in cameras:
        cam_info["cap"].release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()