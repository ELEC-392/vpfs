"""
Camera Extrinsic Calibration - Stereo-style approach.

Camera 0 is the REFERENCE camera. This script computes the rigid transform
from each secondary camera (1, 2, ...) INTO Camera 0's coordinate frame,
exactly like stereo camera calibration.

How it works:
    For each sample, all cameras capture the same scene (markers stationary).
    For every marker visible in both Camera 0 and Camera N, we compute:

        T_camN_to_cam0 = inv(cam0_to_marker) * camN_to_marker

    This is the transform that takes a 3D point expressed in Camera N's frame
    and expresses it in Camera 0's frame.

At runtime:
    - Camera 0 detections are used as-is in Camera 0's frame.
    - Camera 1/2 detections are first transformed into Camera 0's frame using
      the stored T_camN_to_cam0 transforms.
    - Then the unified set of Camera-0-frame observations is used to establish
      the world frame (marker 95 = corner reference, marker 96 defines +X axis).

Usage:
    python calibrate_camera_extrinsics.py
    python calibrate_camera_extrinsics.py --samples 100
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
    det_to_transform_mat,
    _average_rotations,
)

# Marker object points (4 corners in marker local frame, Z=0 plane)
OBJ_POINTS = np.array([
    [-Defaults.TAG_SIZE / 2,  Defaults.TAG_SIZE / 2, 0],
    [ Defaults.TAG_SIZE / 2,  Defaults.TAG_SIZE / 2, 0],
    [ Defaults.TAG_SIZE / 2, -Defaults.TAG_SIZE / 2, 0],
    [-Defaults.TAG_SIZE / 2, -Defaults.TAG_SIZE / 2, 0],
], dtype=np.float32)

REFERENCE_CAMERA_ID = 0


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def initialize_camera(camera_id, CAM_K, CAM_D):
    """Open a single camera and configure it."""
    camera_device = Defaults.CAMERA_SYMLINKS[camera_id]
    print(f"  Initializing camera {camera_id} ({camera_device})...")

    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(camera_device, cv2.CAP_V4L2)

    if not cam.isOpened():
        print(f"    Cannot open camera {camera_id}")
        return None

    print(f"    Camera {camera_id} ready")
    return cam


def grab_fresh_frame(cam):
    """Flush stale buffered frames and return the freshest one."""
    for _ in range(2):
        cam.grab()
    ret = cam.grab()
    if not ret:
        return None
    ret, frame = cam.retrieve()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def detect_markers(frame, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, CAM_K, CAM_D):
    """
    Detect ArUco markers and return pose-estimated ArucoDetection objects.

    Corner distortion is removed with cv2.undistortPoints (preserves CAM_K,
    avoiding the focal-length shrinkage of getOptimalNewCameraMatrix).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    detections = {}  # tag_id -> ArucoDetection
    if ids is None or len(ids) == 0:
        return detections

    for i, tag_id in enumerate(ids.flatten()):
        pts = corners[i].reshape(-1, 1, 2).astype(np.float32)
        pts_u = cv2.undistortPoints(pts, CAM_K, CAM_D, P=CAM_K)
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS, pts_u, CAM_K, None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if ok:
            detections[int(tag_id)] = ArucoDetection(
                tag_id=int(tag_id),
                rvec=rvec.reshape(3),
                tvec=tvec.reshape(3),
                corners=corners[i],
            )

    return detections  # {tag_id: ArucoDetection}


# ---------------------------------------------------------------------------
# Core calibration math
# ---------------------------------------------------------------------------

def compute_camN_to_cam0(dets_cam0, dets_camN):
    """
    Compute T_{camN -> cam0} from one pair of simultaneous frames.

    For each marker visible in both cameras:
        T = inv(cam0_to_marker) * camN_to_marker

    Returns a list of 4x4 transform estimates (one per common marker).
    """
    estimates = []
    for tag_id, det_cam0 in dets_cam0.items():
        if tag_id not in dets_camN:
            continue
        det_camN = dets_camN[tag_id]

        cam0_to_marker = det_to_transform_mat(det_cam0)   # T_{cam0 -> marker}
        camN_to_marker = det_to_transform_mat(det_camN)   # T_{camN -> marker}

        # T_{camN -> cam0}: take point in camN, express in cam0
        T = np.linalg.inv(cam0_to_marker) @ camN_to_marker
        estimates.append(T)

    return estimates


def average_transforms(transforms):
    """
    Average a list of 4x4 rigid transforms (R|t).
    Rotations are averaged via SVD; translations via arithmetic mean.
    """
    rotations    = [T[:3, :3] for T in transforms]
    translations = [T[:3,  3] for T in transforms]

    weights = [1.0] * len(rotations)
    R_avg = _average_rotations(rotations, weights)
    t_avg = np.mean(translations, axis=0)

    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3,  3] = t_avg
    return T_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_samples = 50
    if '--samples' in sys.argv:
        idx = sys.argv.index('--samples')
        if idx + 1 < len(sys.argv):
            num_samples = int(sys.argv[idx + 1])

    print("=" * 70)
    print("Multi-Camera Extrinsic Calibration  (stereo-style)")
    print("=" * 70)
    print(f"\nCamera 0 is the REFERENCE camera.")
    print(f"This script computes  T_{{camN -> cam0}}  for every secondary camera.")
    print(f"\nRequirements:")
    print(f"  * At least one ArUco marker visible in BOTH Camera 0 and each")
    print(f"    secondary camera simultaneously (reference markers 95-99 are ideal).")
    print(f"  * Markers must be completely stationary during capture.")
    print(f"  * Number of samples: {num_samples}\n")

    input("Press ENTER when the scene is ready...")

    # -----------------------------------------------------------------------
    # ArUco detector
    # -----------------------------------------------------------------------
    aruco = cv2.aruco
    ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    ARUCO_PARAMS = (aruco.DetectorParameters()
                    if hasattr(aruco, "DetectorParameters")
                    else aruco.DetectorParameters_create())
    DETECTOR     = (aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
                    if hasattr(aruco, "ArucoDetector") else None)

    # -----------------------------------------------------------------------
    # Camera initialisation
    # -----------------------------------------------------------------------
    print("\nInitialising cameras...")
    CAMERA_IDS = [0, 1, 2]
    cameras = []
    for cam_id in CAMERA_IDS:
        try:
            (fx, fy, cx, cy), D = resolve_camera_intrinsics(argv=sys.argv[1:], camera_id=cam_id)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            cap = initialize_camera(cam_id, K, D)
            if cap is not None:
                cameras.append({"id": cam_id, "cap": cap, "K": K, "D": D})
        except Exception as e:
            print(f"  Failed to initialise camera {cam_id}: {e}")

    if not any(c["id"] == REFERENCE_CAMERA_ID for c in cameras):
        print("\nERROR: Reference Camera 0 could not be initialised. Exiting.")
        return

    secondary_cameras = [c for c in cameras if c["id"] != REFERENCE_CAMERA_ID]
    cam0_info         = next(c for c in cameras if c["id"] == REFERENCE_CAMERA_ID)

    print(f"\nReference camera: Camera {REFERENCE_CAMERA_ID}")
    print(f"Secondary cameras: {[c['id'] for c in secondary_cameras]}")

    # -----------------------------------------------------------------------
    # Warm-up: flush buffers and let auto-exposure settle
    # -----------------------------------------------------------------------
    print("\nWarming up cameras...", end=" ", flush=True)
    for _ in range(15):
        for c in cameras:
            c["cap"].grab()
        time.sleep(0.1)
    print("done\n")

    # -----------------------------------------------------------------------
    # Sample collection
    # -----------------------------------------------------------------------
    # For each secondary camera: list of T_{camN->cam0} estimates
    all_transforms = {c["id"]: [] for c in secondary_cameras}
    stats = {c["id"]: {"used": 0, "skipped_no_common": 0} for c in secondary_cameras}

    print(f"Collecting {num_samples} samples...\n")

    for sample_idx in range(num_samples):
        print(f"  Sample {sample_idx + 1:3d}/{num_samples} ", end="", flush=True)

        # Grab fresh frames from all cameras
        frames = {}
        for c in cameras:
            frame = grab_fresh_frame(c["cap"])
            if frame is not None:
                frames[c["id"]] = frame

        if REFERENCE_CAMERA_ID not in frames:
            print("  [SKIP - Camera 0 failed]")
            for c in secondary_cameras:
                stats[c["id"]]["skipped_no_common"] += 1
            time.sleep(0.1)
            continue

        # Detect markers in Camera 0
        dets_cam0 = detect_markers(
            frames[REFERENCE_CAMERA_ID], DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
            cam0_info["K"], cam0_info["D"]
        )

        summary_parts = [f"cam0 sees {len(dets_cam0)} marker(s)"]

        for sec in secondary_cameras:
            cam_id = sec["id"]
            if cam_id not in frames:
                stats[cam_id]["skipped_no_common"] += 1
                summary_parts.append(f"cam{cam_id}: no frame")
                continue

            dets_camN = detect_markers(
                frames[cam_id], DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
                sec["K"], sec["D"]
            )

            estimates = compute_camN_to_cam0(dets_cam0, dets_camN)
            if estimates:
                all_transforms[cam_id].extend(estimates)
                stats[cam_id]["used"] += len(estimates)
                summary_parts.append(f"cam{cam_id}: +{len(estimates)}")
            else:
                stats[cam_id]["skipped_no_common"] += 1
                summary_parts.append(f"cam{cam_id}: no common markers "
                                     f"(cam0={list(dets_cam0.keys())}, "
                                     f"camN={list(dets_camN.keys())})")

        print("  |  " + "  ".join(summary_parts))
        time.sleep(0.1)

    # -----------------------------------------------------------------------
    # Release cameras
    # -----------------------------------------------------------------------
    for c in cameras:
        c["cap"].release()

    # -----------------------------------------------------------------------
    # Compute and report average transforms
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    calibration_data = {}

    for sec in secondary_cameras:
        cam_id = sec["id"]
        transforms = all_transforms[cam_id]

        print(f"\nCamera {cam_id}:")
        print(f"  Samples used   : {stats[cam_id]['used']}")
        print(f"  Samples skipped: {stats[cam_id]['skipped_no_common']}")

        if len(transforms) < 5:
            print(f"  [FAIL] Not enough samples (need >= 5). Camera {cam_id} skipped.")
            continue

        T_avg = average_transforms(transforms)

        # Translation: position of Camera N's origin in Camera 0's frame
        t = T_avg[:3, 3]
        dist = np.linalg.norm(t)

        # Standard deviation of translation estimates
        t_std = np.std([T[:3, 3] for T in transforms], axis=0)

        print(f"  Camera {cam_id} origin in Camera 0 frame:")
        print(f"    X = {t[0]*100:7.1f} cm   Y = {t[1]*100:7.1f} cm   Z = {t[2]*100:7.1f} cm")
        print(f"    Distance between cameras: {dist*100:.1f} cm")
        print(f"  Std dev (translation):")
        print(f"    X = {t_std[0]*100:.2f} cm   Y = {t_std[1]*100:.2f} cm   Z = {t_std[2]*100:.2f} cm")

        max_std = np.max(t_std) * 100
        if max_std < 0.5:
            print(f"  [PASS] Excellent calibration quality!")
        elif max_std < 2.0:
            print(f"  [PASS] Good calibration quality")
        elif max_std < 5.0:
            print(f"  [WARN] Moderate variability - consider more samples or better lighting")
        else:
            print(f"  [FAIL] High variability ({max_std:.1f} cm) - calibration may be unreliable")

        calibration_data[str(cam_id)] = {
            "transform": T_avg.tolist(),   # T_{camN -> cam0}
            "samples_used": len(transforms),
            "std_dev_cm": {
                "x": float(t_std[0] * 100),
                "y": float(t_std[1] * 100),
                "z": float(t_std[2] * 100),
            }
        }

    if not calibration_data:
        print("\n[FAIL] No cameras were successfully calibrated. Exiting without saving.")
        return

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "_metadata": {
            "description": "Inter-camera extrinsic calibration (stereo-style)",
            "transform_type": "camN_to_cam0",
            "reference_camera": REFERENCE_CAMERA_ID,
            "explanation": (
                "Each 'transform' is a 4x4 matrix T_{camN->cam0}. "
                "Apply it to a 3-D point expressed in Camera N's frame to get "
                "the same point in Camera 0's frame. "
                "Camera 0 detections need no transform. "
                "World frame (origin=marker95, X-axis=marker96) is derived "
                "at runtime from Camera 0's view of the reference markers."
            ),
            "units": "meters",
        }
    }
    output.update(calibration_data)

    output_file = "camera_extrinsics.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[DONE] Calibration saved to: {output_file}")
    print(f"\nCalibrated cameras: {list(calibration_data.keys())}")
    print(f"\nTo use:")
    print(f"  python vehicle_position_system_multicam.py --extrinsics {output_file}")
    print()


if __name__ == "__main__":
    main()
