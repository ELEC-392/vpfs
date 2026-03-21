"""
Utilities for camera and AprilTag pose composition.

Overview:
- refTags: known world/map/frame poses for reference tags (world-to-tag transforms).
- det_to_transform_mat: convert a single detection's pose (R, t) into a 4x4 homogeneous matrix.
- compute_camera_pos: estimate the map-to-camera transform from detected reference tags.
- compute_tag_poses: transform detected tag poses into the map frame using a known camera pose.

Conventions:
- Homogeneous transforms are 4x4 matrices (R|t; 0 0 0 1).
- refTags[<id>].mat is a 4x4 world-to-tag transform.
- Detection provides:
  - det.tag_id: int ID of the tag.
  - det.pose_R: 3x3 rotation (camera-to-tag).
  - det.pose_t: 3x1 translation (camera-to-tag), in centimetres.
"""
import os
import sys
import json
import logging
import logging.handlers
import shutil
import subprocess
import threading
import time
import numpy as np
import ref_tags
import cv2

from pathlib import Path
from numpy.typing import *
from typing import Dict, Tuple

# World-to-tag transformation matrices (authoritative map of known tag poses).
# Rotations are defined in ref_tags; here we just reference them.
tags = ref_tags.ref_tags


class Defaults:
    # Fallback intrinsics (Logitech Brio 4K): fx, fy, cx, cy
    FALLBACK_INTRINSICS = (978.56, 973.73, 825.30, 467.65)
    CAM_WIDTH  = 3840
    CAM_HEIGHT = 2160
    CAMERA_SYMLINKS = [
        "/dev/brio-camera1",
        "/dev/brio-camera2",
        "/dev/brio-camera3"
    ]
    TAG_SIZE = 10  # 10 cm  (all coordinates in this codebase are in centimetres)


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


def parse_calib_path(argv: list[str]) -> str | None:
    """Parse --calib <path> or --calib=path from argv; return None if not provided."""
    for i, arg in enumerate(argv):
        if arg.startswith("--calib="):
            return arg.split("=", 1)[1]
        if arg == "--calib" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def load_intrinsics_from_json(path: str):
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


def resolve_camera_intrinsics(argv: list[str], camera_id: int = 0) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
    """
    Resolve intrinsics and distortion in priority order:
    1) --calib path from CLI
    2) ./camera{id}_calibration.json (next to this script, based on camera_id)
    3) ./camera_calibration.json (legacy fallback)
    4) FALLBACK_INTRINSICS + zero distortion (Brio 4K)
    
    Args:
        argv: Command-line arguments to parse for --calib override
        camera_id: Camera ID (0, 1, 2) to load corresponding calibration file
        
    Returns ((fx, fy, cx, cy), dist_coeffs_array)
    """
    # CLI override
    cli_path = parse_calib_path(argv)
    if cli_path:
        loaded = load_intrinsics_from_json(cli_path)
        if loaded:
            return loaded

    # Camera-specific calibration file: camera{id}_calibration.json
    camera_specific_path = os.path.join(os.path.dirname(__file__), f"camera{camera_id}_calibration.json")
    loaded = load_intrinsics_from_json(camera_specific_path)
    if loaded:
        return loaded

    # Legacy fallback: camera_calibration.json (for backward compatibility)
    default_path = os.path.join(os.path.dirname(__file__), "camera_calibration.json")
    loaded = load_intrinsics_from_json(default_path)
    if loaded:
        return loaded

    print(f"Using fallback intrinsics for camera {camera_id} (Logitech Brio 4K) and zero distortion.")
    return (Defaults.FALLBACK_INTRINSICS), np.zeros((5, 1), dtype=np.float64)


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
        device_list = Defaults.CAMERA_SYMLINKS
    
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


def draw_aruco_overlays(img, corners, ids, CAM_K, CAM_D, TAG_SIZE, rvecs=None, tvecs=None, world_positions=None):
    """
    Draw ArUco marker overlays on image.
    
    Args:
        world_positions: Optional dict mapping tag_id -> (x, y, z) in world/map coordinates.
                        If provided, displays world coords instead of camera coords.
    """
    if corners is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        if rvecs is not None and tvecs is not None and len(rvecs) > 0:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(img, CAM_K, CAM_D, rvec, tvec, TAG_SIZE * 0.5)
        # Add text with marker position
        for i, corner in enumerate(corners):
            c = corner[0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            
            # Get tag ID for this marker
            tag_id = int(ids[i][0]) if ids is not None and i < len(ids) else None
            
            # Display world coordinates if available, otherwise camera coordinates
            if world_positions is not None and tag_id is not None and tag_id in world_positions:
                # World/map coordinates (relative to marker 95)
                x, y, z, heading = world_positions[tag_id]
                text = f"ID:{tag_id} X:{x:.1f}cm Y:{y:.1f}cm H:{np.degrees(heading):.0f}\u00b0"
                cv2.putText(img, text, (center_x - 100, center_y - 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
            elif rvecs is not None and tvecs is not None and i < len(tvecs):
                # Camera coordinates (fallback)
                tvec = tvecs[i]
                dist = np.linalg.norm(tvec)
                text = f"ID:{tag_id if tag_id else '?'} cam: {dist:.1f}cm"
                cv2.putText(img, text, (center_x - 100, center_y - 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3, cv2.LINE_AA)
    return img


def det_to_transform_mat(detection) -> ArrayLike:
    """
    Convert a single tag detection (R, t) into a 4x4 homogeneous transform.

    Args:
        detection: An object with fields:
            - pose_t: (3x1) translation vector from camera to tag (camera frame).
            - pose_R: (3x3) rotation matrix from camera to tag (camera frame).

    Returns:
        4x4 numpy array for the camera-to-tag transform:
            [ R | t ]
            [ 0 0 0 1 ]
    """
    trans = detection.pose_t            # Expected shape: (3,1)
    rot = detection.pose_R              # Expected shape: (3,3)

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans.flatten()
    return T
    # # Stack rotation and translation into a 3x4, then append homogeneous row
    # mat = np.concatenate((rot, trans), axis=1)         # (3x4)
    # mat = np.concatenate((mat, [[0, 0, 0, 1]]), axis=0)  # (4x4)
    # return mat


def _average_rotations(rotations: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Compute weighted average rotation via SVD (nearest orthonormal matrix).
    rotations: list of 3x3 rotation matrices.
    weights: same length, non-negative.
    """
    M = np.zeros((3, 3), dtype=float)
    wsum = 0.0
    for R, w in zip(rotations, weights):
        M += float(w) * R
        wsum += float(w)
    if wsum <= 0:
        # Fallback: return identity if all weights are zero
        return np.eye(3)
    M /= wsum
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure a proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _average_translations(translations: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Weighted average of 3D translations.
    translations: list of shape (3,), weights: list of floats.
    """
    acc = np.zeros(3, dtype=float)
    wsum = 0.0
    for t, w in zip(translations, weights):
        acc += float(w) * t
        wsum += float(w)
    if wsum <= 0:
        return acc  # zeros
    return acc / wsum


def _fuse_map_to_cam(transforms: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Fuse multiple 4x4 map->camera transforms into a single transform
    by averaging rotation and translation separately.
    """
    Rs = [T[:3, :3] for T in transforms]
    ts = [T[:3, 3] for T in transforms]
    R_avg = _average_rotations(Rs, weights)
    t_avg = _average_translations(ts, weights)
    T = np.eye(4)
    T[:3, :3] = R_avg
    T[:3, 3] = t_avg
    return T


def compute_camera_pos(detections) -> ArrayLike | None:
    """
    Estimate the camera pose in map coordinates by fusing all matching detections.

    For each detection whose tag_id exists in tags, compute:
        cam_to_tag = det_to_transform_mat(det)
        map_to_cam_i = (map_to_tag) * inv(cam_to_tag)

    Then fuse all map_to_cam_i with a weighted average:
    - Rotation: SVD-based averaging (nearest orthonormal matrix).
    - Translation: weighted mean.
    - Weights: inverse of tag distance (||pose_t||) by default.

    Args:
        detections: Iterable of detections with tag_id, pose_R, pose_t.

    Returns:
        4x4 numpy array (map-to-camera transform), or None if no reference tag is found.
    """
    candidates: list[np.ndarray] = []
    weights: list[float] = []

    for det in detections:
        if det.tag_id in tags:
            # camera->tag (from the detector)
            cam_to_tag = det_to_transform_mat(det)
            # map->tag (from known field layout)
            map_to_tag = tags[det.tag_id].mat
            # Correct chain: p_tag = cam_to_tag @ map_to_cam @ p_world
            #   → map_to_cam = inv(cam_to_tag) @ map_to_tag
            map_to_cam_i = np.linalg.inv(cam_to_tag) @ map_to_tag
            candidates.append(map_to_cam_i)

            # Weight closer tags higher (you can swap to uniform weights = 1.0)
            dist = float(np.linalg.norm(np.asarray(det.pose_t).reshape(-1)))
            w = 1.0 / max(dist, 1e-3)
            weights.append(w)

    if not candidates:
        return None

    return _fuse_map_to_cam(candidates, weights)


def compute_tag_poses(detections, cam_pos: ArrayLike) -> Dict[int, Tuple[int, int, int]]:
    """
    Compute tag positions in the map frame given the camera pose.

    Correct chain:  p_tag = cam_to_tag @ map_to_cam @ p_world
      → map_to_tag = cam_to_tag @ map_to_cam

    Tag origin in world coords = inv(map_to_tag) @ [0,0,0,1]^T
                                = -R^T @ t
    (where R, t are the rotation/translation sub-blocks of map_to_tag)

    Args:
        detections: Iterable of detections with tag_id, pose_R, pose_t.
        cam_pos: 4x4 map-to-camera transform (from compute_camera_pos).

    Returns:
        Dict mapping tag_id -> (x, y, z) in map/world coordinates (centimetres).
    """
    tag_poses: Dict[int, Tuple[int, int, int]] = {}

    for det in detections:
        # camera->tag from the detector
        cam_to_tag = det_to_transform_mat(det)
        # map->tag: correct order is cam_to_tag @ map_to_cam
        map_to_tag = cam_to_tag @ cam_pos
        # Tag origin in world = -R^T @ t
        R = map_to_tag[:3, :3]
        t = map_to_tag[:3, 3]
        pos = -R.T @ t
        # Heading: angle of tag X axis w.r.t. world X axis.
        # R transforms world->tag, so tag X in world = first row of R.
        heading = float(np.arctan2(R[0, 1], R[0, 0]))
        tag_poses[det.tag_id] = (float(pos[0]), float(pos[1]), float(pos[2]), heading)

    return tag_poses


# ---------------------------------------------------------------------------
# Shared logging setup
# ---------------------------------------------------------------------------

def setup_vps_logging(log_file: str = "vps_debug.log") -> logging.Logger:
    """
    Create (or retrieve) the shared 'vps' logger with a rotating file handler
    (INFO+) and a stderr console handler (WARNING+).  Safe to call from both
    the single-camera and multi-camera scripts — handlers are added only once.
    """
    logger = logging.getLogger("vps")
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.DEBUG)

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# USB / kernel diagnostics
# ---------------------------------------------------------------------------

# Resolve v4l2-ctl once at import time — lives in /usr/sbin on most distros
# but pyenv virtualenvs only have /usr/bin in PATH.
_V4L2_CTL = shutil.which("v4l2-ctl") or "/usr/sbin/v4l2-ctl"


def dump_dmesg_usb(label: str = "") -> None:
    """
    Snapshot recent kernel USB/UVC messages and write them to the shared log.
    Tries journalctl -k (readable without root on systemd systems) first, then
    falls back to dmesg.  Call on the first camera read error and before USB
    recovery attempts to capture kernel errors while they are still in the ring
    buffer.
    """
    _log = logging.getLogger("vps")
    tag  = f" ({label})" if label else ""
    raw = ""
    source = "?"
    try:
        # 1. journalctl -k
        try:
            r = subprocess.run(
                ["journalctl", "-k", "--since", "-120s", "--no-pager", "-o", "short"],
                capture_output=True, text=True, timeout=6)
            real_lines = [l for l in r.stdout.splitlines() if not l.startswith("-- ")]
            if r.returncode == 0 and real_lines:
                raw = r.stdout
                source = "journalctl -k"
        except Exception:
            pass

        # 2. dmesg --since (util-linux >= 2.23)
        if not raw:
            try:
                r = subprocess.run(
                    ["dmesg", "--since", "-120s"],
                    capture_output=True, text=True, timeout=4)
                if r.stdout.strip():
                    raw = r.stdout
                    source = "dmesg --since"
            except Exception:
                pass

        # 3. Plain dmesg tail
        if not raw:
            try:
                r = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=4)
                raw = r.stdout
                source = "dmesg"
            except Exception:
                pass

        if not raw.strip():
            _log.warning(f"dmesg snapshot{tag}: all sources returned empty "
                         f"(kernel.dmesg_restrict may be 1 — "
                         f"run: sudo sysctl kernel.dmesg_restrict=0)")
            return

        all_lines = raw.splitlines()
        keywords  = ("usb", "uvc", "xhci", "ehci", "video4linux", "v4l2",
                     "error", "warn", "reset", "disconnect",
                     "overflow", "timeout", "failed", "unable",
                     "suspend", "resume", "power", "autosuspend")
        usb_lines = [l for l in all_lines if any(k in l.lower() for k in keywords)]
        if usb_lines:
            _log.warning(f"dmesg snapshot{tag} [{source}] — {len(usb_lines)} USB/UVC hits:\n"
                         + "\n".join(usb_lines[-40:]))
        else:
            tail = all_lines[-60:]
            _log.warning(f"dmesg snapshot{tag} [{source}]: no keyword matches — "
                         f"raw last {len(tail)} lines:\n" + "\n".join(tail))
    except Exception as exc:
        logging.getLogger("vps").warning(f"dmesg probe failed{tag}: {exc}")


def log_v4l2_state(device: str, label: str = "") -> None:
    """Query the actual V4L2 driver state via v4l2-ctl and write it to the log."""
    _log = logging.getLogger("vps")
    tag  = f" ({label})" if label else ""
    if not Path(_V4L2_CTL).exists():
        _log.debug(f"v4l2-ctl not found at {_V4L2_CTL} — cannot query {device}{tag}")
        return
    try:
        result = subprocess.run(
            [_V4L2_CTL, "-d", device,
             "--get-fmt-video", "--get-parm",
             "--get-ctrl",
             "brightness,exposure_time_absolute,focus_absolute,focus_automatic_continuous"],
            capture_output=True, text=True, timeout=3)
        state = (result.stdout or result.stderr).strip()
        _log.info(f"V4L2 state {device}{tag}:\n{state}")
    except Exception as exc:
        _log.warning(f"v4l2-ctl query failed for {device}{tag}: {exc}")


def solve_pnp_ippe(obj_pts: np.ndarray, img_pts: np.ndarray, cam_k: np.ndarray):
    """
    Solve PnP using IPPE_SQUARE and resolve the two-solution ambiguity by
    enforcing that the tag must be in front of the camera (tvec[2] > 0).

    Background
    ----------
    IPPE_SQUARE always produces exactly two geometrically valid solutions.
    When noise is high or the tag is far away both solutions have similar
    reprojection errors, so the standard solvePnP call oscillates between
    them frame-to-frame, producing the Z-axis flip observed in practice.

    The disambiguation rule used here is physical: in any overhead-camera
    setup the marker plane faces the camera, so the translation vector's Z
    component (distance along the optical axis) must be positive.  This
    single constraint uniquely selects the correct solution.

    After disambiguation the selected solution is refined with
    Levenberg-Marquardt to reduce remaining reprojection error.

    Args:
        obj_pts : (4, 3) float32 — 3-D object points (OBJ_POINTS).
        img_pts : (4, 1, 2) float32 — undistorted image corner points.
        cam_k   : (3, 3) float64 — camera intrinsic matrix.

    Returns:
        success (bool), rvec (3,1), tvec (3,1)
        Returns (False, None, None) if solvePnPGeneric finds no solution.
    """
    n, rvecs_all, tvecs_all, _ = cv2.solvePnPGeneric(
        obj_pts, img_pts, cam_k, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)

    if n == 0:
        return False, None, None

    # Pick the solution where the tag is in front of the camera (tvec[2] > 0).
    # solvePnPGeneric orders solutions by reprojection error (best first), so
    # index 0 is the fallback when the physical constraint cannot distinguish.
    chosen = 0
    if n >= 2:
        z0 = float(tvecs_all[0][2])
        z1 = float(tvecs_all[1][2])
        if z0 <= 0.0 and z1 > 0.0:
            chosen = 1

    rvec = rvecs_all[chosen]
    tvec = tvecs_all[chosen]

    # Refine with Levenberg-Marquardt starting from the disambiguated solution.
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, cam_k, None, rvec, tvec)
    return True, rvec, tvec


# Pre-computed marker corner geometry (shared by both VPS scripts).
OBJ_POINTS = np.array([
    [-Defaults.TAG_SIZE / 2,  Defaults.TAG_SIZE / 2, 0],
    [ Defaults.TAG_SIZE / 2,  Defaults.TAG_SIZE / 2, 0],
    [ Defaults.TAG_SIZE / 2, -Defaults.TAG_SIZE / 2, 0],
    [-Defaults.TAG_SIZE / 2, -Defaults.TAG_SIZE / 2, 0],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# ArUco detection helper
# ---------------------------------------------------------------------------

def detect_aruco(
    frame: np.ndarray,
    CAM_K: np.ndarray,
    CAM_D: np.ndarray,
    DETECTOR,
    ARUCO_DICT,
    ARUCO_PARAMS,
):
    """
    Detect ArUco markers in *frame* and compute their 6-DoF poses.

    Corner undistortion strategy (same as multicam script):
    - Detect on the original (distorted) grayscale image so corner accuracy
      benefits from the full pixel grid.
    - Undistort only the detected corner points with cv2.undistortPoints before
      solvePnP.  This avoids the focal-length shrinkage that full-image
      undistortion with alpha=1 introduces and is significantly cheaper.

    Returns:
        detections : list[ArucoDetection]
        rvecs      : list of (3,1) rotation vectors (original OpenCV convention)
        tvecs      : list of (3,1) translation vectors
        corners    : raw corner arrays from detectMarkers (for draw_aruco_overlays)
        ids        : id array from detectMarkers (may be None)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    rvecs: list      = []
    tvecs: list      = []
    detections: list = []

    if ids is not None and len(ids) > 0:
        for i, corner in enumerate(corners):
            pts = corner.reshape(-1, 1, 2).astype(np.float32)
            pts_u = cv2.undistortPoints(pts, CAM_K, CAM_D, P=CAM_K)
            success, rvec, tvec = solve_pnp_ippe(OBJ_POINTS, pts_u, CAM_K)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
                detections.append(ArucoDetection(
                    tag_id=int(ids[i][0]),
                    rvec=rvec.reshape(3),
                    tvec=tvec.reshape(3),
                    corners=corner))

    return detections, rvecs, tvecs, corners, ids


# ---------------------------------------------------------------------------
# Minimal background capture buffer
# ---------------------------------------------------------------------------

class CameraFrameBuffer:
    """
    Background thread that continuously drains the V4L2 kernel buffer and
    always makes the latest frame available via get().

    Without this, if the main loop runs slower than the camera FPS the
    V4L2 kernel buffer fills up, the driver triggers select() timeouts,
    and the camera appears to freeze or crash.  A background reader thread
    draining at camera speed keeps the buffer empty regardless of main-loop
    frequency.

    Usage::

        buf = CameraFrameBuffer(cap)
        ok, frame = buf.get()   # latest frame, non-blocking
        buf.stop()              # join the thread before exiting
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap     = cap
        self._lock    = threading.Lock()
        self._frame   = None
        self._ok      = False
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="cam-buf")
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                if ok and frame is not None:
                    self._frame = frame

    def get(self):
        """Return (ok, frame) — the most recently captured frame."""
        with self._lock:
            ok  = self._ok
            ref = self._frame
        return ok, (ref.copy() if ref is not None else None)

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._running = False
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# In-place terminal dashboard
# ---------------------------------------------------------------------------

class TerminalDashboard:
    """
    Redraws a fixed block of text in-place on stdout using ANSI escape codes.
    Works for both single-camera and multi-camera scripts.  Log warnings/errors
    are routed to stderr and appear above the dashboard without disrupting it.
    """
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _CYAN   = "\033[36m"
    _WHITE  = "\033[37m"

    def __init__(self):
        self._line_count = 0

    def render(self, lines: list) -> None:
        """Overwrite the previously rendered block with new content."""
        if self._line_count > 0:
            # Move cursor up by the number of previously written lines, then
            # erase everything from the cursor to the end of the screen.
            sys.stdout.write(f"\033[{self._line_count}A\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._line_count = len(lines)

    def build(
        self,
        update_count:         int,
        start_time:           float,
        loop_ms:              float,
        target_hz:            float,
        actual_hz:            float,
        camera_stats:         list,   # list of dicts — see below
        all_marker_positions: dict,
        mobile_markers:       dict,
        vpfs_sent:            bool,
        title:                str = "POSITIONING SYSTEM",
    ) -> list:
        """
        Build the dashboard line list.

        ``camera_stats`` is a list of dicts with keys:
            name              str
            frames_ok         int
            frames_err        int
            last_ok_ts        float | None
            is_alive          bool
            consecutive_errors int
        """
        REFERENCE_TAG_IDS = {95, 96, 97, 98, 99}
        W = 72
        now      = time.time()
        uptime_s = int(now - start_time)
        h = uptime_s // 3600
        m = (uptime_s % 3600) // 60
        s = uptime_s % 60
        ts = time.strftime("%H:%M:%S")

        lines = []
        B, R, G, Y, C = self._BOLD, self._RESET, self._GREEN, self._YELLOW, self._CYAN

        lines.append(f"{B}{'\u2500'*W}{R}")
        lines.append(f"{B}  {title}{R}  "
                     f"{C}{ts}{R}  uptime {h:02d}:{m:02d}:{s:02d}  updates {update_count}")
        lines.append(f"{B}{'\u2500'*W}{R}")

        fps_color = (G if actual_hz >= target_hz * 0.85 else
                     Y if actual_hz >= target_hz * 0.50 else self._RED)
        lines.append(f"  {B}Rate :{R} {fps_color}{actual_hz:5.2f} Hz{R}  "
                     f"target {target_hz:.0f} Hz   loop {loop_ms:5.1f} ms")

        lines.append(f"  {B}Cams :{R}")
        for cs in camera_stats:
            last   = cs["last_ok_ts"]
            age    = f"{now - last:.1f}s" if last else "never"
            errs   = cs["frames_err"]
            streak = cs["consecutive_errors"]
            if not cs["is_alive"]:
                status = f"{self._RED}DEAD{R}"
            elif streak > 0:
                status = f"{self._RED}ERR\u00d7{streak}{R}"
            elif errs == 0:
                status = f"{G}OK{R}"
            else:
                status = f"{Y}OK (errs={errs}){R}"
            lines.append(
                f"    {cs['name']:12s}  "
                f"ok={cs['frames_ok']:<7d}  err={errs:<5d}  "
                f"last_ok={age:>7s}  {status}")

        lines.append(f"{B}{'\u2500'*W}{R}")

        lines.append(f"  {B}Reference markers:{R}")
        ref_ids = sorted(tid for tid in all_marker_positions if tid in REFERENCE_TAG_IDS)
        if ref_ids:
            for tid in ref_ids:
                x, y, z, _ = all_marker_positions[tid]
                lines.append(f"    #{tid:2d}  X={x:7.1f} cm   Y={y:7.1f} cm   "
                             f"dist={np.sqrt(x**2+y**2):6.1f} cm")
        else:
            lines.append(f"    {Y}(none visible \u2014 need at least one of 95-99){R}")

        lines.append(f"  {B}Mobile markers:{R}")
        if mobile_markers:
            for tid in sorted(mobile_markers):
                x, y, z, heading = mobile_markers[tid]
                sent = f"  {G}\u2191 sent{R}" if vpfs_sent else ""
                lines.append(
                    f"    #{tid:2d}  X={x:7.1f} cm   Y={y:7.1f} cm   "
                    f"hdg={np.degrees(heading):6.1f}\u00b0   "
                    f"dist={np.sqrt(x**2+y**2):6.1f} cm{sent}")
        else:
            lines.append(f"    {Y}(none detected){R}")

        lines.append(f"{B}{'\u2500'*W}{R}")
        lines.append(f"  {self._WHITE}Ctrl+C to quit   warnings \u2192 stderr   "
                     f"full log \u2192 vps_debug.log{R}")

        return lines

