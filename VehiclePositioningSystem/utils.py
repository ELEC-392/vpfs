"""
Utilities for camera and ArUco tag pose composition.

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

import numpy as np
from numpy.typing import *
from typing import Dict, Tuple
import ref_tags 

# World-to-tag transformation matrices (authoritative map of known tag poses).
# Rotations are defined in ref_tags; here we just reference them.
tags = ref_tags.ref_tags


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
    T[:3, 3] = trans
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
            # map->camera for this detection
            map_to_cam_i = np.matmul(map_to_tag, np.linalg.inv(cam_to_tag))
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

    Performs: map_to_tag = (map_to_cam) * (cam_to_tag)

    Args:
        detections: Iterable of detections with tag_id, pose_R, pose_t.
        cam_pos: 4x4 map-to-camera transform (from compute_camera_pos).

    Returns:
        Dict mapping tag_id -> (x, y, z) in map/world coordinates (meters).
        Values are extracted from the translation column of the map_to_tag matrix.
    """
    tag_poses: Dict[int, Tuple[int, int, int]] = {}

    for det in detections:
        # camera->tag from the detector
        cam_to_tag = det_to_transform_mat(det)
        # map->tag = (map->cam) * (cam->tag)
        map_to_tag = np.matmul(cam_pos, cam_to_tag)

        # Extract translation (last column of the 4x4)
        tag_poses[det.tag_id] = (
            map_to_tag[0][3],
            map_to_tag[1][3],
            map_to_tag[2][3],
        )

    return tag_poses
