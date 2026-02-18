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

import utils
import platform

# ============================================================================
# Camera Configuration
# ============================================================================
# List of camera device paths to search (in priority order)
# These should match your udev symlinks or direct /dev/video* paths
# The script will use the first available camera from this list
CAMERA_DEVICES = [
    '/dev/brio-video',      # Logitech BRIO (udev symlink from 99-brio-camera.rules)
    # Add more cameras here as needed, e.g.:
    # '/dev/overhead-camera',
    # '/dev/side-camera',
    # '/dev/video2',        # Direct device path fallback
]

# Camera settings for Desktop mode
camera_id = 0  # Default fallback (Windows or if no device found)
camera_width = 4096
camera_height = 2160

def find_camera(device_list=None, search_model=None):
    """
    Find camera device on Linux.
    
    Args:
        device_list: List of device paths to check (default: CAMERA_DEVICES)
        search_model: Optional model name to search for if device_list fails (e.g., 'Logitech BRIO')
    
    Returns:
        Device path string (e.g., '/dev/brio-video' or '/dev/video2') or None
    """
    if device_list is None:
        device_list = CAMERA_DEVICES
    
    # First, check for devices in priority order
    for device in device_list:
        if os.path.exists(device):
            try:
                # Verify it's a valid video capture device
                result = os.popen(f'udevadm info {device} 2>/dev/null | grep "ID_V4L_CAPABILITIES"').read()
                if ':capture:' in result or result == '':  # Empty result means it might still work
                    print(f"Found camera at {device}")
                    return device
            except:
                continue
    
    # Fallback: search /dev/video* for specific model if specified
    if search_model:
        for i in range(20):  # Check up to video19
            device = f'/dev/video{i}'
            if not os.path.exists(device):
                continue
            try:
                result = os.popen(f'udevadm info {device} 2>/dev/null | grep -E "ID_V4L_PRODUCT|ID_V4L_CAPABILITIES"').read()
                if search_model in result and ':capture:' in result:
                    print(f"Found {search_model} at {device}")
                    return device
            except:
                continue
    
    return None

# Fallback intrinsics (Logitech Brio 4K): fx, fy, cx, cy
FALLBACK_INTRINSICS = (978.56, 973.73, 825.30, 467.65)

def _parse_calib_path(argv: list[str]) -> str | None:
    """Parse --calib <path> or --calib=path from argv; return None if not provided."""
    for i, arg in enumerate(argv):
        if arg.startswith("--calib="):
            return arg.split("=", 1)[1]
        if arg == "--calib" and i + 1 < len(argv):
            return argv[i + 1]
    return None

def _load_intrinsics_from_json(path: str):
    """
    Load intrinsics from a calibration JSON (camera_calib.py schema).
    Returns ((fx, fy, cx, cy), dist_coeffs_array) or None on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            calib = json.load(f)

        K = calib.get("camera_matrix")
        if not K or len(K) != 3 or any(len(row) != 3 for row in K):
            print(f"Invalid camera_matrix in {path}")
            return None

        fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])

        # dist_coeffs may be 5, 8, or more terms; accept any length
        dist_list = calib.get("dist_coeffs", [])
        if not isinstance(dist_list, list):
            dist_list = []
        dist = np.array(dist_list, dtype=np.float64).reshape(-1, 1) if dist_list else np.zeros((5, 1), dtype=np.float64)

        print(f"Loaded intrinsics from {path}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}, k={dist.flatten().tolist()}")
        return (fx, fy, cx, cy), dist
    except FileNotFoundError:
        print(f"Calibration file not found: {path}")
    except Exception as e:
        print(f"Failed to load calibration from {path}: {e}")
    return None

def _resolve_camera_intrinsics():
    """
    Resolve intrinsics and distortion in priority order:
    1) --calib path from CLI
    2) ./camera_calibration.json (next to this script)
    3) FALLBACK_INTRINSICS + zero distortion (Brio 4K)
    Returns ((fx, fy, cx, cy), dist_coeffs_array)
    """
    # CLI override
    cli_path = _parse_calib_path(sys.argv[1:])
    if cli_path:
        loaded = _load_intrinsics_from_json(cli_path)
        if loaded:
            return loaded

    # Default file next to this script
    default_path = os.path.join(os.path.dirname(__file__), "camera_calibration.json")
    loaded = _load_intrinsics_from_json(default_path)
    if loaded:
        return loaded

    print("Using fallback intrinsics (Logitech Brio 4K) and zero distortion.")
    return (FALLBACK_INTRINSICS), np.zeros((5, 1), dtype=np.float64)

# Intrinsics used by the detector
(in_fx, in_fy, in_cx, in_cy), CAM_D = _resolve_camera_intrinsics()
camera_intrinsics = (in_fx, in_fy, in_cx, in_cy)
# Build OpenCV camera matrix from loaded intrinsics
CAM_K = np.array([[in_fx, 0, in_cx], [0, in_fy, in_cy], [0, 0, 1]], dtype=np.float64)

# Tag physical size in meters (10 cm)
tag_size = 10 / 100

# --- ArUco setup ---
aruco = cv2.aruco
# Pick a dictionary that matches your printed markers
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
ARUCO_PARAMS = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS) if hasattr(aruco, "ArucoDetector") else None

# GStreamer pipeline to work with Jetson (uses hardware decode path)
pipeline = ' ! '.join([
    "v4l2src device=/dev/video0",
    "image/jpeg, fomat=MJPG, width=2560, height=1440, framerate=5/1",
    "nvv4l2decoder mjpeg=1",
    "nvvidconv",
    "videoconvert",
    "video/x-raw, format=(string)BGR",
    "appsink drop=true sync=false"
])
max_fps = "5/1"

# Toggle Jetson mode by including "jetson" in CLI args; otherwise use platform-appropriate backend
jetson = "jetson" in sys.argv
is_windows = platform.system() == "Windows"

if jetson:
    print(pipeline)
    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0")
    os.system("v4l2-ctl -d /dev/video0 -c focus_absolute=0")
    os.system("v4l2-ctl -d /dev/video0 -C focus_auto")
    os.system("v4l2-ctl -d /dev/video0 -C focus_absolute")
    cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
else:
    # Auto-detect camera on Linux
    camera_device = camera_id
    if not is_windows:
        detected = find_camera(search_model='Logitech BRIO')  # Fallback model search
        if detected:
            camera_device = detected
        else:
            print(f"No configured camera found, using default camera_id={camera_id}")
    
    if is_windows:
        cam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cam.open(camera_id + cv2.CAP_MSMF)
    else:
        # On Linux, reset camera to known state using v4l2-ctl before opening
        if not is_windows and isinstance(camera_device, str) and camera_device.startswith('/dev/'):
            print(f"Initializing camera {camera_device}...")
            # Reset to defaults first
            os.system(f"v4l2-ctl -d {camera_device} -c focus_automatic_continuous=0 2>/dev/null")
            os.system(f"v4l2-ctl -d {camera_device} -c focus_absolute=0 2>/dev/null")
            os.system(f"v4l2-ctl -d {camera_device} -c auto_exposure=1 2>/dev/null")
            os.system(f"v4l2-ctl -d {camera_device} -c exposure_time_absolute=200 2>/dev/null")
            os.system(f"v4l2-ctl -d {camera_device} -c brightness=128 2>/dev/null")
        
        # On Linux, create camera and set format before opening
        cam = cv2.VideoCapture()
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cam.open(camera_device, cv2.CAP_V4L2)
        
        # Verify and reapply if needed
        actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != camera_width or actual_h != camera_height:
            print(f"First attempt: {actual_w}x{actual_h}, retrying with settings...")
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        
        # Camera control settings (applied via OpenCV as backup)
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure mode
        cam.set(cv2.CAP_PROP_EXPOSURE, 85)  # Set exposure
    
    max_fps = int(cam.get(cv2.CAP_PROP_FPS))

# Log actual capture format
frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frameWidth, 'x', frameHeight, '@', max_fps)
print(int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode())

# Log exposure settings
exposure_value = cam.get(cv2.CAP_PROP_EXPOSURE)
print(f"Exposure: {exposure_value}")

# Overlay helpers
font = cv2.FONT_HERSHEY_PLAIN
def draw_aruco_overlays(img, corners, ids, rvecs=None, tvecs=None):
    if corners is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        if rvecs is not None and tvecs is not None and len(rvecs) > 0:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(img, CAM_K, CAM_D, rvec, tvec, tag_size * 0.5)
        # Add text with relative position and between parenthesis the euclidean distance
        for i, corner in enumerate(corners):
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            if rvecs is not None and tvecs is not None and i < len(tvecs):
                tvec = tvecs[i]
                text = f"X:{tvec[0][0]*100:.1f}cm Y:{tvec[1][0]*100:.1f}cm Z:{tvec[2][0]*100:.1f}cm ({np.linalg.norm(tvec)*100:.1f}cm)"
                cv2.putText(img, text, (center_x - 100, center_y - 40), font, 3, (255, 255, 0), 3, cv2.LINE_AA)
    return img

# Adapter to match utils.compute_camera_pos expected detection interface
class ArucoDetection:
    def __init__(self, tag_id: int, rvec: np.ndarray, tvec: np.ndarray, corners: np.ndarray):
        # OpenCV gives tag->camera (object->camera) pose. Convert to camera->tag.
        R_tc, _ = cv2.Rodrigues(rvec.reshape(3, 1))   # (3x3)
        t_tc = tvec.reshape(3, 1)                     # (3x1)
        R_ct = R_tc.T
        t_ct = -R_tc.T @ t_tc

        self.tag_id = int(tag_id)
        self.pose_R = R_ct
        self.pose_t = t_ct
        self.corners = corners.reshape(-1, 2)
        self.center = self.corners.mean(axis=0)

# Verify camera is available
if not cam.isOpened():
    print("Cannot open camera")
    sys.exit(1)

lastTime = time.time()

while True:
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
        objPoints = np.array([[-tag_size/2, tag_size/2, 0],
                               [tag_size/2, tag_size/2, 0],
                               [tag_size/2, -tag_size/2, 0],
                               [-tag_size/2, -tag_size/2, 0]], dtype=np.float32)
        
        rvecs = []
        tvecs = []
        
        # Calculate pose for each marker
        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(objPoints, corner, CAM_K, CAM_D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
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
    frame = draw_aruco_overlays(frame, corners, ids, rvecs, tvecs) if ids is not None else frame

    # Estimate camera pose from known reference tags (fused if multiple)
    cameraPos = utils.compute_camera_pos(detections)
    print(f"Camera Position: {cameraPos}" if cameraPos is not None else "Camera Position: Unknown")

    # If we have a valid camera pose, transform all detected tags to map/world coords
    tagPoses = {}
    if cameraPos is not None:
        tagPoses = utils.compute_tag_poses(detections, cameraPos)
        vpfs_connector.send_update(tagPoses)

    # FPS overlay
    frameTime = time.time() - lastTime
    fps = 1 / frameTime if frameTime > 0 else 0.0
    lastTime = time.time()
    cv2.putText(frame, f"{frameWidth}x{frameHeight} @ {fps:.2f} fps", (0, frameHeight - 10), font, 10, (255, 255, 255), 5, cv2.LINE_AA)

    # Display per-tag map-frame coordinates
    i = -100
    for tag in tagPoses:
        cv2.putText(frame, f"{tag}: X{tagPoses[tag][0]:.2f} Y{tagPoses[tag][1]:.2f} Z{tagPoses[tag][2]:.2f}", (0, frameHeight + i), font, 3, (255, 0, 255), 2, cv2.LINE_AA)
        i -= 50

    cv2.imshow('frame', cv2.resize(frame, (1080, 720)))
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
