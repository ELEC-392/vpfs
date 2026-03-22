"""
Single-Camera Vehicle Positioning System (VPS) runtime (ArUco-based).

- Opens one camera (V4L2 / Logitech Brio 4K).
- Detects ArUco markers (DICT_6X6_100) in a background-drained thread so the
  V4L2 kernel buffer never fills up regardless of main-loop frequency.
- Estimates camera pose from known reference tags (IDs 95-99, world positions
  in ref_tags.py) using utils.compute_camera_pos.
- Transforms detected mobile tags into world/map coordinates.
- Applies temporal smoothing to reduce position jitter.
- Sends tag pose updates to the VPFS backend via vpfs_connector.
- Renders a live terminal dashboard (no X11 display required by default).
- Optionally shows an OpenCV preview window (--display flag).

Usage::

    python vehicle_position_system.py                        # headless, 5 Hz
    python vehicle_position_system.py --display              # OpenCV window
    python vehicle_position_system.py --hz 10               # 10 Hz update rate
    python vehicle_position_system.py --cam 0               # camera index
    python vehicle_position_system.py --cam-fps 5           # camera capture FPS (default 5)
    python vehicle_position_system.py --calib calib.json    # custom intrinsics
    python vehicle_position_system.py --vpfs                # publish to localhost
    python vehicle_position_system.py --vpfs http://host:5000
    python vehicle_position_system.py --auto-exposure       # use camera autoexposure

Reference Markers (from ref_tags.py):
    95 - corner marker (any world coordinate; NOT required to be origin)
    96 - +X axis reference
    97 - far-right corner
    98 - far-left corner
    99 - centre-line marker

Set all coordinates as measured physical positions from your chosen origin.
The more reference markers visible, the more stable the pose estimate.
"""

import os
import sys
import signal
import time

import cv2
import numpy as np

import vpfs_connector
from ref_tags import ref_tags  # ensure tag registry is initialised on import

from utils import (
    Defaults,
    ArucoDetection,
    resolve_camera_intrinsics,
    draw_aruco_overlays,
    compute_camera_pos,
    compute_tag_poses,
    setup_vps_logging,
    dump_dmesg_usb,
    log_v4l2_state,
    _V4L2_CTL,
    OBJ_POINTS,
    detect_aruco,
    solve_pnp_ippe,
    CameraFrameBuffer,
    TerminalDashboard,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_TAG_IDS = set(ref_tags.keys())  # derived from ref_tags.py — no manual update needed
SMOOTHING_ALPHA   = 0.4   # EMA weight for the current measurement (0=frozen, 1=raw)


# ---------------------------------------------------------------------------
# Camera initialisation
# ---------------------------------------------------------------------------

def initialize_camera(camera_id: int, fps: int = 5, auto_exposure: bool = False) -> cv2.VideoCapture | None:
    """
    Open camera *camera_id* with the correct V4L2 settings for an ArUco pipeline.

    Key choices:
    - MJPEG at 4K: maximises resolution for long-range tag detection.
    - BUFFERSIZE=1 BEFORE open(): keeps the kernel queue to one frame so the
      background thread never accumulates stale frames.
    - FPS cap (default 5): reduces CPU/USB load; the background thread drains at
      camera speed anyway so the main loop runs at its own frequency.
    - Manual focus + manual exposure: prevents hunting that blurs tags.
    """
    device = Defaults.CAMERA_SYMLINKS[camera_id]
    log = setup_vps_logging()
    log.info(f"[cam{camera_id}] Initialising {device}  (fps={fps})")

    # Reset controls to known defaults.
    v4l = _V4L2_CTL
    os.system(f"{v4l} -d {device} -c focus_automatic_continuous=0 2>/dev/null")
    os.system(f"{v4l} -d {device} -c focus_absolute=0            2>/dev/null")
    if auto_exposure:
        # auto_exposure=3 → Aperture Priority (autoexposure) on UVC cameras
        os.system(f"{v4l} -d {device} -c auto_exposure=3             2>/dev/null")
    else:
        # auto_exposure=1 → Manual mode
        os.system(f"{v4l} -d {device} -c auto_exposure=1             2>/dev/null")
        os.system(f"{v4l} -d {device} -c exposure_time_absolute=200  2>/dev/null")
    os.system(f"{v4l} -d {device} -c brightness=128              2>/dev/null")

    # CAP_PROP_BUFFERSIZE must come BEFORE open() so the kernel REQBUFS ioctl
    # uses it.  Setting it after open() is belt-and-suspenders only.
    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  Defaults.CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # belt-and-suspenders after open

    log_v4l2_state(device, label=f"cam{camera_id} after open")

    # Re-apply resolution if the driver silently downgraded it.
    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != Defaults.CAM_WIDTH or actual_h != Defaults.CAM_HEIGHT:
        log.warning(f"[cam{camera_id}] Driver returned {actual_w}x{actual_h}, "
                    f"re-requesting {Defaults.CAM_WIDTH}x{Defaults.CAM_HEIGHT}")
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  Defaults.CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)

    cam.set(cv2.CAP_PROP_FPS,           fps)
    cam.set(cv2.CAP_PROP_AUTOFOCUS,     0)
    if auto_exposure:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # Aperture Priority (autoexposure)
    else:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # Manual exposure mode
        cam.set(cv2.CAP_PROP_EXPOSURE,      185)

    log_v4l2_state(device, label=f"cam{camera_id} fully configured")

    w          = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_actual = int(cam.get(cv2.CAP_PROP_FPS))
    buf        = int(cam.get(cv2.CAP_PROP_BUFFERSIZE))
    log.info(f"[cam{camera_id}] {w}x{h} @ {fps_actual} fps  "
             f"buffer={buf}  device={device}")

    if not cam.isOpened():
        log.error(f"[cam{camera_id}] VideoCapture.isOpened() returned False")
        return None

    return cam


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # ------------------------------------------------------------------ args
    show_display = "--display" in argv

    update_frequency_hz = 5.0
    if "--hz" in argv:
        idx = argv.index("--hz")
        try:
            update_frequency_hz = float(argv[idx + 1])
            if not (0 < update_frequency_hz <= 60):
                update_frequency_hz = 5.0
        except (IndexError, ValueError):
            pass

    camera_id = 0
    if "--cam" in argv:
        idx = argv.index("--cam")
        try:
            camera_id = int(argv[idx + 1])
        except (IndexError, ValueError):
            pass

    cam_fps = 5
    if "--cam-fps" in argv:
        idx = argv.index("--cam-fps")
        try:
            cam_fps = int(argv[idx + 1])
            if not (1 <= cam_fps <= 30):
                cam_fps = 5
        except (IndexError, ValueError):
            pass

    auto_exposure: bool = "--auto-exposure" in argv

    vpfs_url: str | None = None
    if "--vpfs" in argv:
        idx = argv.index("--vpfs")
        vpfs_url = "http://localhost:5000"
        if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            vpfs_url = argv[idx + 1]

    # ---------------------------------------------------------------- logging
    log = setup_vps_logging()
    log.info("=== SINGLE-CAMERA VPS STARTING ===")
    log.info(f"  camera_id={camera_id}  hz={update_frequency_hz}  cam_fps={cam_fps}"
             f"  display={show_display}  vpfs={vpfs_url}  auto_exposure={auto_exposure}")

    dump_dmesg_usb(label="startup")

    # ----------------------------------------------------------- VPFS connect
    if vpfs_url:
        vpfs_connector.connect_to_server(vpfs_url)
        log.info(f"VPFS connected to {vpfs_url}")

    # --------------------------------------------------------- intrinsics
    (in_fx, in_fy, in_cx, in_cy), CAM_D = resolve_camera_intrinsics(
        argv=argv, camera_id=camera_id)
    CAM_K = np.array([[in_fx, 0, in_cx],
                      [0, in_fy, in_cy],
                      [0,    0,      1]], dtype=np.float64)
    log.info(f"Intrinsics: fx={in_fx:.1f} fy={in_fy:.1f} "
             f"cx={in_cx:.1f} cy={in_cy:.1f}")

    # ------------------------------------------------------- ArUco detector
    aruco        = cv2.aruco
    ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    ARUCO_PARAMS = (aruco.DetectorParameters()
                   if hasattr(aruco, "DetectorParameters")
                   else aruco.DetectorParameters_create())
    # Subpixel corner refinement — critical for accurate pose at distance
    ARUCO_PARAMS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    ARUCO_PARAMS.cornerRefinementWinSize = 5
    ARUCO_PARAMS.cornerRefinementMaxIterations = 30
    ARUCO_PARAMS.cornerRefinementMinAccuracy = 0.01
    # Wider adaptive threshold window handles mixed/challenging lighting
    ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
    ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
    ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
    DETECTOR     = (aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
                   if hasattr(aruco, "ArucoDetector") else None)

    # --------------------------------------------------------- camera init
    cam = initialize_camera(camera_id, fps=cam_fps)
    if cam is None or not cam.isOpened():
        log.error("Failed to open camera — aborting")
        sys.exit(1)

    buf = CameraFrameBuffer(cam)
    log.info("CameraFrameBuffer started")

    # --------------------------------------------------- signal handling
    _shutdown = {"requested": False}

    def _on_signal(signum, frame):  # noqa: ARG001
        _shutdown["requested"] = True
        log.info(f"Received signal {signum}, shutting down")

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ---------------------------------------------------- dashboard + state
    dashboard     = TerminalDashboard()
    start_time    = time.time()
    update_count  = 0
    loop_ms       = 0.0
    actual_hz     = 0.0

    smooth_positions: dict[int, np.ndarray] = {}   # tag_id -> [x, y, z, heading]

    # Camera stat counters (mirrors the CameraCapture counters used in multicam)
    frames_ok          = 0
    frames_err         = 0
    consecutive_errors = 0
    last_ok_ts: float | None = None

    target_period = 1.0 / update_frequency_hz

    all_marker_positions: dict = {}
    mobile_markers:       dict = {}
    vpfs_sent              = False

    # ---------------------------------------------------------- main loop
    while not _shutdown["requested"]:
        loop_start = time.time()

        ok, frame = buf.get()

        if ok and frame is not None:
            frames_ok         += 1
            last_ok_ts         = time.time()
            consecutive_errors = 0

            # --- detect ArUco markers ------------------------------------
            detections, rvecs, tvecs, corners, ids = detect_aruco(
                frame, CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS)

            # --- camera pose from reference tags -------------------------
            camera_pos = compute_camera_pos(detections, CAM_K)

            # --- mobile tag world positions (with temporal smoothing) ----
            all_marker_positions = {}
            mobile_markers       = {}
            vpfs_sent            = False

            if camera_pos is not None:
                tag_poses = compute_tag_poses(detections, camera_pos)
                all_marker_positions = tag_poses

                for tid, pose in tag_poses.items():
                    p = np.array(pose, dtype=float)
                    if tid in smooth_positions:
                        smooth_positions[tid] = (
                            SMOOTHING_ALPHA * p
                            + (1.0 - SMOOTHING_ALPHA) * smooth_positions[tid])
                    else:
                        smooth_positions[tid] = p.copy()

                mobile_markers = {
                    tid: tuple(smooth_positions[tid])
                    for tid in smooth_positions
                    if tid not in REFERENCE_TAG_IDS
                }

                if mobile_markers and vpfs_url:
                    try:
                        vpfs_connector.send_update(mobile_markers)
                        vpfs_sent = True
                    except Exception as exc:
                        log.warning(f"vpfs_connector.send_update failed: {exc}")

            # --- optional OpenCV preview window --------------------------
            if show_display:
                display_frame = cv2.undistort(frame, CAM_K, CAM_D)
                display_frame = (
                    draw_aruco_overlays(
                        display_frame, corners, ids, CAM_K, None,
                        Defaults.TAG_SIZE, rvecs, tvecs)
                    if ids is not None else display_frame
                )
                h_px, w_px = display_frame.shape[:2]
                cv2.putText(display_frame,
                            f"{w_px}x{h_px}  {actual_hz:.1f} Hz",
                            (10, h_px - 10),
                            cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow("VPS - single camera",
                           cv2.resize(display_frame,
                                      (Defaults.CAM_WIDTH  // 4,
                                       Defaults.CAM_HEIGHT // 4)))
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    break

        else:
            frames_err         += 1
            consecutive_errors += 1
            if consecutive_errors == 1:
                log.warning(f"[cam{camera_id}] First read error  "
                            f"ok={frames_ok} err={frames_err}  "
                            f"cap.isOpened={cam.isOpened()}")
            elif consecutive_errors == 3:
                dump_dmesg_usb(label=f"cam{camera_id} error onset")
            elif consecutive_errors % 10 == 0:
                log.warning(f"[cam{camera_id}] {consecutive_errors} consecutive errors  "
                            f"ok={frames_ok} err={frames_err}")

        # --- terminal dashboard ------------------------------------------
        update_count  += 1
        current_time   = time.time()
        loop_ms        = (current_time - loop_start) * 1000.0  # processing time excl. sleep
        elapsed        = current_time - start_time
        actual_hz      = update_count / elapsed if elapsed > 0 else 0.0

        cam_stats = [{
            "name":               f"cam{camera_id}",
            "frames_ok":          frames_ok,
            "frames_err":         frames_err,
            "last_ok_ts":         last_ok_ts,
            "is_alive":           True,
            "consecutive_errors": consecutive_errors,
        }]

        dashboard.render(dashboard.build(
            update_count=update_count,
            start_time=start_time,
            loop_ms=loop_ms,
            target_hz=update_frequency_hz,
            actual_hz=actual_hz,
            camera_stats=cam_stats,
            title="SINGLE-CAMERA POSITIONING SYSTEM",
            all_marker_positions=all_marker_positions,
            mobile_markers=mobile_markers,
            vpfs_sent=vpfs_sent,
        ))

        # --- frequency throttle ------------------------------------------
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_period:
            time.sleep(target_period - loop_elapsed)

    # ------------------------------------------------------------ cleanup
    log.info("Shutting down VPS")
    buf.stop()
    cam.release()
    if show_display:
        cv2.destroyAllWindows()
    log.info("=== SINGLE-CAMERA VPS STOPPED ===")


if __name__ == "__main__":
    main(sys.argv[1:])
