"""
Vehicle Positioning System (VPS) runtime (ArUco-based).

- Opens a camera (Jetson via GStreamer or Windows via DirectShow).
- Detects ArUco markers with OpenCV.
- Estimates camera pose from known reference tags (utils.compute_camera_pos),
  then transforms detected tags into map/world coordinates.
- Sends tag pose updates to the VPFS backend via vpfs_connector.
- Shows a live preview with marker overlays and FPS info.

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

from utils import (
    Defaults,
    ArucoDetection,
    find_camera,
    resolve_camera_intrinsics,
    draw_aruco_overlays,
    compute_camera_pos,
    compute_tag_poses
)
import platform


def main(argv=None, camera_id=0):
    # --- Parse flags ---
    if argv is None:
        argv = []
    show_display = '--no-display' not in argv
    update_frequency_hz = 10  # default
    if '--hz' in argv:
        idx = argv.index('--hz')
        try:
            update_frequency_hz = float(argv[idx + 1])
            if update_frequency_hz <= 0 or update_frequency_hz > 60:
                print(f"Warning: Invalid frequency {update_frequency_hz}Hz, using default 10Hz")
                update_frequency_hz = 10
        except (IndexError, ValueError):
            update_frequency_hz = 10
    target_loop_time = 1.0 / update_frequency_hz
    print(f"Update Frequency: {update_frequency_hz} Hz (period: {target_loop_time*1000:.1f}ms)")
    print(f"Display: {'enabled' if show_display else 'disabled (headless)'}")

    # Connect to VPFS backend if --vpfs flag is provided
    # Usage: --vpfs                  (connects to http://localhost:5000)
    #        --vpfs http://host:5000  (connects to custom host)
    if '--vpfs' in argv:
        idx = argv.index('--vpfs')
        vpfs_url = "http://localhost:5000"
        if idx + 1 < len(argv) and not argv[idx + 1].startswith('--'):
            vpfs_url = argv[idx + 1]
        vpfs_connector.connect_to_server(vpfs_url)

    # Load camera intrinsics from JSON file or use defaults
    # Intrinsics used by the detector
    (in_fx, in_fy, in_cx, in_cy), CAM_D = resolve_camera_intrinsics(argv=argv, camera_id=camera_id)
    camera_intrinsics = (in_fx, in_fy, in_cx, in_cy)
    # Build OpenCV camera matrix from loaded intrinsics
    CAM_K = np.array([[in_fx, 0, in_cx], [0, in_fy, in_cy], [0, 0, 1]], dtype=np.float64)

    # --- ArUco setup ---
    aruco = cv2.aruco
    # Pick a dictionary that matches your printed markers
    ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    ARUCO_PARAMS = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    DETECTOR     = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS) if hasattr(aruco, "ArucoDetector") else None

    # Initializing camera
    camera_device = Defaults.CAMERA_SYMLINKS[camera_id] #if camera_id < len(Defaults.CAMERA_SYMLINKS) else 0
    print(f"Initializing camera {camera_device}...")
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
    # Keep V4L2 internal buffer at 1 frame so stale frames never accumulate.
    # Without this, if ArUco detection is slow the kernel buffer fills up,
    # triggering select() timeouts and driver crashes (same issue as multicam).
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    max_fps = int(cam.get(cv2.CAP_PROP_FPS))

    # Log actual capture format
    frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frameWidth, 'x', frameHeight, '@', max_fps)
    print(int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode())

    # Log exposure settings
    exposure_value = cam.get(cv2.CAP_PROP_EXPOSURE)
    print(f"Exposure: {exposure_value}")

    # Verify camera is available
    if not cam.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    # Running the main loop
    lastTime = time.time()
    last_timing_warning = 0
    TIMING_WARNING_INTERVAL = 10.0
    while True:
        loop_start = time.time()
        ret, frame = cam.read()
        if not ret:
            print("Failed to receive frame, exiting")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        if DETECTOR is not None:
            corners, ids, _ = DETECTOR.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        rvecs, tvecs = None, None
        detections = []
        if ids is not None and len(ids) > 0:
            # Pose estimation for each marker using cv2.solvePnP (OpenCV 4.x method)
            # Set up marker coordinate system (centered, Z pointing out)
            objPoints = np.array([[-Defaults.TAG_SIZE/2,  Defaults.TAG_SIZE/2, 0],
                                  [Defaults.TAG_SIZE/2,   Defaults.TAG_SIZE/2, 0],
                                  [Defaults.TAG_SIZE/2,  -Defaults.TAG_SIZE/2, 0],
                                  [-Defaults.TAG_SIZE/2, -Defaults.TAG_SIZE/2, 0]], dtype=np.float32)
            
            rvecs = []
            tvecs = []
            
            # Calculate pose for each marker
            for corner in corners:
                # Undistort corner points before solvePnP for better numerical
                # stability with IPPE_SQUARE (same approach as multicam script).
                pts = corner.reshape(4, 1, 2).astype(np.float32)
                pts_undistorted = cv2.undistortPoints(pts, CAM_K, CAM_D, P=CAM_K)
                success, rvec, tvec = cv2.solvePnP(
                    objPoints, pts_undistorted, CAM_K, None,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
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

        # If we have a valid camera pose, transform all detected tags to map/world coords
        REFERENCE_TAG_IDS = {95, 96, 97, 98, 99}
        tagPoses = {}
        if cameraPos is not None:
            tagPoses = compute_tag_poses(detections, cameraPos)

            # Print marker positions in the same format as the multicam script
            print(f"\n{'='*70}")
            print(f"MARKER POSITIONS (relative to marker 95 at origin):")
            print(f"{'='*70}")

            print("\n--- Reference Markers (Map Corners) ---")
            for tag_id in sorted([tid for tid in tagPoses if tid in REFERENCE_TAG_IDS]):
                x, y, z = tagPoses[tag_id]
                print(f"  Marker {tag_id:2d}: X={x*100:7.1f}cm  Y={y*100:7.1f}cm  (distance: {np.sqrt(x**2 + y**2)*100:.1f}cm)")

            mobile_markers = {tid: pos for tid, pos in tagPoses.items() if tid not in REFERENCE_TAG_IDS}
            if mobile_markers:
                print("\n--- Mobile Markers (Tracked Objects) ---")
                for tag_id in sorted(mobile_markers.keys()):
                    x, y, z = mobile_markers[tag_id]
                    print(f"  Marker {tag_id:2d}: X={x*100:7.1f}cm  Y={y*100:7.1f}cm  (distance: {np.sqrt(x**2 + y**2)*100:.1f}cm)")
                vpfs_connector.send_update(mobile_markers)
                print(f"\n✓ Sent {len(mobile_markers)} mobile marker(s) to VPFS")
            else:
                print("\n--- No mobile markers detected ---")

            print(f"{'='*70}\n")
        else:
            print("\nNo markers detected (need at least marker 95 visible)\n")

        # FPS overlay
        frameTime = time.time() - lastTime
        fps = 1 / frameTime if frameTime > 0 else 0.0
        lastTime = time.time()

        if show_display:
            cv2.putText(frame, f"{frameWidth}x{frameHeight} @ {fps:.2f} fps", (0, frameHeight - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.imshow('frame', cv2.resize(frame, (Defaults.CAM_WIDTH//4, Defaults.CAM_HEIGHT//4)))
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        # Maintain target update frequency
        loop_time = time.time() - loop_start
        if loop_time > target_loop_time:
            now = time.time()
            if now - last_timing_warning > TIMING_WARNING_INTERVAL:
                print(f"\n⚠ WARNING: Processing too slow for {update_frequency_hz}Hz ({loop_time*1000:.0f}ms/loop, target {target_loop_time*1000:.0f}ms)")
                last_timing_warning = now
        else:
            time.sleep(target_loop_time - loop_time)

    # Cleanup
    cam.release()
    if show_display:
        cv2.destroyAllWindows()   


if __name__ == "__main__":
    # Parse command-line arguments using sys.argv for simplicity
    argv = sys.argv[1:]

    # CAM id
    camera_id = 1

    main(argv=argv, camera_id=camera_id)