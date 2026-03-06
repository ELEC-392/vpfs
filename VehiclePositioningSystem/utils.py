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
  - det.pose_t: 3x1 translation (camera-to-tag), in meters.
"""
import os
import json
import numpy as np
import ref_tags 
import cv2

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
    TAG_SIZE = 10 / 100  # 10 cm in meters


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
                x, y, z = world_positions[tag_id]
                text = f"ID:{tag_id} X:{x*100:.1f}cm Y:{y*100:.1f}cm Z:{z*100:.1f}cm"
                cv2.putText(img, text, (center_x - 100, center_y - 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
            elif rvecs is not None and tvecs is not None and i < len(tvecs):
                # Camera coordinates (fallback)
                tvec = tvecs[i]
                dist = np.linalg.norm(tvec)
                text = f"ID:{tag_id if tag_id else '?'} cam: {dist*100:.1f}cm"
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
        Dict mapping tag_id -> (x, y, z) in map/world coordinates (meters).
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
        tag_poses[det.tag_id] = (float(pos[0]), float(pos[1]), float(pos[2]))

    return tag_poses


