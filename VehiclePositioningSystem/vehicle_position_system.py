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

# Camera settings for Desktop mode (Windows/DirectShow path)
camera_id = 0
camera_width = 4096
camera_height = 2160

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
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
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

# Toggle Jetson mode by including "jetson" in CLI args; otherwise use Windows desktop path
jetson = "jetson" in sys.argv

if jetson:
    print(pipeline)
    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0")
    os.system("v4l2-ctl -d /dev/video0 -c focus_absolute=0")
    os.system("v4l2-ctl -d /dev/video0 -C focus_auto")
    os.system("v4l2-ctl -d /dev/video0 -C focus_absolute")
    cam = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
else:
    cam = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cam.open(camera_id + cv2.CAP_MSMF)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    max_fps = int(cam.get(cv2.CAP_PROP_FPS))
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Avoid AF hunting

# Log actual capture format
frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frameWidth, 'x', frameHeight, '@', max_fps)
print(int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode())

# Overlay helpers
font = cv2.FONT_HERSHEY_PLAIN
def draw_aruco_overlays(img, corners, ids, rvecs=None, tvecs=None):
    if corners is not None and len(corners) > 0:
        aruco.drawDetectedMarkers(img, corners, ids)
        if rvecs is not None and tvecs is not None:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(img, CAM_K, CAM_D, rvec, tvec, tag_size)
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
        # Pose estimation for each marker
        # corners shape: (N,1,4,2). estimatePoseSingleMarkers returns (N,1,3) arrays.
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, tag_size, CAM_K, CAM_D)
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

    # If we have a valid camera pose, transform all detected tags to map/world coords
    tagPoses = {}
    if cameraPos is not None:
        tagPoses = utils.compute_tag_poses(detections, cameraPos)
        vpfs_connector.send_update(tagPoses)

    # FPS overlay
    frameTime = time.time() - lastTime
    fps = 1 / frameTime if frameTime > 0 else 0.0
    lastTime = time.time()
    cv2.putText(frame, f"{frameWidth}x{frameHeight} @ {fps:.2f} fps", (0, frameHeight - 10), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

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
