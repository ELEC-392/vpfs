"""
Multi-Camera Vehicle Positioning System (VPS) runtime (ArUco-based).

- Opens three cameras.
- Detects ArUco markers (DICT_6X6_100) from all cameras.
- Each camera independently computes its world pose from any reference markers
  it sees (IDs 95-99), whose world positions are hardcoded in ref_tags.py.
- Mobile-tag world positions from all cameras are averaged together.
- Applies temporal smoothing to reduce position jitter.
- Sends tag pose updates to the VPFS backend via vpfs_connector.
- Shows live preview with marker overlays and FPS info for each camera.

Usage:
    python vehicle_position_system_multicam.py                    # Normal mode
    python vehicle_position_system_multicam.py --no-display       # Headless (faster)
    python vehicle_position_system_multicam.py --hz 5             # 5 Hz update rate
    python vehicle_position_system_multicam.py --calib <path>     # Custom intrinsics

Reference Markers (known world positions in ref_tags.py):
    95 - origin (0, 0)
    96 - (60 cm, 0)
    97 - (60 cm, 70 cm)
    98 - (0, 70 cm)
    99 - centre

Place all reference markers flat, oriented the same way as marker 95.
The more reference markers visible to a camera, the more stable its pose estimate.

Performance:
- Terminal shows FPS and marker positions in centimetres.
- Temporal smoothing reduces jitter (adjust SMOOTHING_ALPHA in code).

Camera Recovery:
- Automatic recovery from camera failures using USB device reset.
- System continues with remaining cameras if one fails.
"""

import sys
from ref_tags import ref_tags  # optional - for camera position overlay only
import vpfs_connector

import cv2
import numpy as np
import time
import os
import json
import re
import signal
import threading
import logging
import logging.handlers
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

from utils import (
    Defaults,
    ArucoDetection,
    resolve_camera_intrinsics,
    draw_aruco_overlays,
    compute_camera_pos,
    compute_tag_poses,
    det_to_transform_mat
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

# ---------------------------------------------------------------------------
# Logging — writes to the terminal AND a rotating file (vps_debug.log).
# The file persists across crashes / terminal clears for post-mortem analysis.
# ---------------------------------------------------------------------------
_log_file_handler = logging.handlers.RotatingFileHandler(
    "vps_debug.log", maxBytes=5 * 1024 * 1024, backupCount=2)
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_log_console_handler = logging.StreamHandler(sys.stderr)  # stderr keeps log warnings out of the dashboard
_log_console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
log = logging.getLogger("vps")
log.setLevel(logging.DEBUG)
log.addHandler(_log_file_handler)
log.addHandler(_log_console_handler)


def dump_dmesg_usb(label: str = "") -> None:
    """
    Snapshot recent kernel USB/UVC messages from dmesg and write them to the
    log.  Called automatically on the first camera read error and at recovery
    so that USB isochronous transfer errors, xHCI resets, and UVC timeouts
    are captured before the system fully breaks.
    """
    tag = f" ({label})" if label else ""
    try:
        # Try --since first (needs util-linux >= 2.23); fall back to tail.
        try:
            result = subprocess.run(
                ["dmesg", "--since", "-120s"],
                capture_output=True, text=True, timeout=4)
            raw = result.stdout
        except Exception:
            result = subprocess.run(
                ["dmesg"],
                capture_output=True, text=True, timeout=4)
            raw = result.stdout
        all_lines = raw.splitlines()
        keywords = ("usb", "uvc", "xhci", "ehci", "video4linux", "v4l2",
                    "error", "warn", "reset", "disconnect",
                    "overflow", "timeout", "failed", "unable", "suspend",
                    "resume", "power", "autosuspend")
        usb_lines = [l for l in all_lines
                     if any(k in l.lower() for k in keywords)]
        if usb_lines:
            log.warning(f"dmesg snapshot{tag} — filtered USB/UVC kernel messages "
                        f"({len(usb_lines)} hits):\n" + "\n".join(usb_lines[-40:]))
        else:
            # No keyword matches — dump the raw tail so we don't miss anything
            tail = all_lines[-60:]
            log.warning(f"dmesg snapshot{tag}: no keyword matches — "
                        f"raw last {len(tail)} lines:\n" + "\n".join(tail))
    except Exception as exc:
        log.warning(f"dmesg probe failed{tag}: {exc}")


# Resolve v4l2-ctl once at startup — it lives in /usr/sbin on most distros
# but pyenv virtualenvs only have /usr/bin in PATH.
_V4L2_CTL = shutil.which("v4l2-ctl") or "/usr/sbin/v4l2-ctl"


def log_v4l2_state(device: str, label: str = "") -> None:
    """Query the actual V4L2 driver state via v4l2-ctl and write it to the log.
    Use this at camera init and at the start of an error streak to verify that
    the format/FPS/buffer-count the driver negotiated matches what we requested.
    """
    tag = f" ({label})" if label else ""
    if not Path(_V4L2_CTL).exists():
        log.warning(f"v4l2-ctl not found at {_V4L2_CTL} — cannot query {device}{tag}")
        return
    try:
        result = subprocess.run(
            [_V4L2_CTL, "-d", device,
             "--get-fmt-video", "--get-parm",
             "--get-ctrl",
             "brightness,exposure_time_absolute,focus_absolute,focus_automatic_continuous"],
            capture_output=True, text=True, timeout=3)
        state = (result.stdout or result.stderr).strip()
        log.info(f"V4L2 state {device}{tag}:\n{state}")
    except Exception as exc:
        log.warning(f"v4l2-ctl query failed for {device}{tag}: {exc}")


def disable_usb_autosuspend(device: str) -> None:
    """
    Disable USB autosuspend for the USB device backing a /dev/videoN symlink.

    Linux USB autosuspend can silently power-down a camera after a period of
    perceived inactivity (default 2 s for many kernels/hubs), causing all
    cameras on the same controller to drop simultaneously — with no dmesg
    error, just a clean disappearance.  Writing 'on' to power/control and
    '-1' to power/autosuspend_delay_ms prevents this entirely.

    The sysfs path is resolved via udevadm so it works with symlinks like
    /dev/brio-camera0.
    """
    try:
        # Resolve the sysfs device path for this /dev node
        r = subprocess.run(
            ["udevadm", "info", "-q", "path", "-n", device],
            capture_output=True, text=True, timeout=3)
        if r.returncode != 0 or not r.stdout.strip():
            log.warning(f"disable_usb_autosuspend: udevadm could not resolve {device}")
            return

        sysfs = Path("/sys") / r.stdout.strip().lstrip("/")

        # Walk up the sysfs tree to find the USB device directory
        # (identified by the presence of idVendor)
        p = sysfs
        usb_dev = None
        while p != p.parent:
            if (p / "idVendor").exists():
                usb_dev = p
                break
            p = p.parent

        if usb_dev is None:
            log.warning(f"disable_usb_autosuspend: no USB device node found for {device}")
            return

        vendor = (usb_dev / "idVendor").read_text().strip()
        product = (usb_dev / "idProduct").read_text().strip()

        power_control  = usb_dev / "power" / "control"
        autosuspend_ms = usb_dev / "power" / "autosuspend_delay_ms"

        power_control.write_text("on")
        autosuspend_ms.write_text("-1")

        log.info(f"USB autosuspend disabled for {device} "
                 f"(vendor={vendor} product={product} sysfs={usb_dev})")

    except PermissionError:
        log.warning(f"disable_usb_autosuspend: permission denied writing to sysfs for {device}. "
                    f"Run as root or add a udev rule: "
                    f'ACTION=="add", SUBSYSTEM=="usb", ATTR{{power/control}}="on"')
    except Exception as exc:
        log.warning(f"disable_usb_autosuspend failed for {device}: {exc}")


# ---------------------------------------------------------------------------
# In-place terminal dashboard
# ---------------------------------------------------------------------------
class TerminalDashboard:
    """
    Redraws a fixed block of text in-place on stdout using ANSI escape codes.
    Log warnings/errors are routed to stderr and appear above the dashboard.
    """
    # ANSI helpers
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _CYAN   = "\033[36m"
    _WHITE  = "\033[37m"

    def __init__(self):
        self._anchored = False  # True once we've saved the cursor anchor

    # ------------------------------------------------------------------
    def render(self, lines: list) -> None:
        """Overwrite the previously rendered block with new content."""
        if not self._anchored:
            # ESC 7  →  DEC Save Cursor: record exactly where the dashboard starts
            sys.stdout.write("\0337")
            self._anchored = True
        else:
            # ESC 8  →  DEC Restore Cursor: jump back to the saved position
            # \033[J →  Erase from cursor to end of screen
            sys.stdout.write("\0338\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    def build(
        self,
        update_count:      int,
        start_time:        float,
        loop_ms:           float,
        target_hz:         float,
        actual_hz:         float,
        captures:          list,
        all_marker_positions: dict,
        mobile_markers:    dict,
        vpfs_sent:         bool,
    ) -> list:
        """Return a list of terminal lines representing the current state."""
        W = 72  # dashboard width
        now      = time.time()
        uptime_s = int(now - start_time)
        h, m, s  = uptime_s // 3600, (uptime_s % 3600) // 60, uptime_s % 60
        ts       = time.strftime("%H:%M:%S")

        REFERENCE_TAG_IDS = {95, 96, 97, 98, 99}

        lines = []
        B, R, G, Y, C = self._BOLD, self._RESET, self._GREEN, self._YELLOW, self._CYAN

        lines.append(f"{B}{'─'*W}{R}")
        lines.append(f"{B}  MULTI-CAMERA POSITIONING SYSTEM{R}  "
                     f"{C}{ts}{R}  uptime {h:02d}:{m:02d}:{s:02d}  updates {update_count}")
        lines.append(f"{B}{'─'*W}{R}")

        # --- FPS / timing panel ---
        fps_color = G if actual_hz >= target_hz * 0.85 else Y if actual_hz >= target_hz * 0.5 else self._RED
        lines.append(f"  {B}Rate :{R} {fps_color}{actual_hz:5.2f} Hz{R}  "
                     f"target {target_hz:.0f} Hz   "
                     f"loop {loop_ms:5.1f} ms")

        # --- Camera health panel ---
        lines.append(f"  {B}Cams :{R}")
        for c in captures:
            last   = c._last_ok_ts
            age    = f"{now - last:.1f}s" if last else "never"
            errs   = c._frames_err
            streak = c._consecutive_errors
            if not c.is_alive:
                status = f"{self._RED}DEAD{R}"
            elif streak > 0:
                status = f"{self._RED}ERR×{streak}{R}"
            elif errs == 0:
                status = f"{G}OK{R}"
            else:
                status = f"{Y}OK (errs={errs}){R}"
            lines.append(
                f"    {c._cam_info['name']:10s}  "
                f"ok={c._frames_ok:<7d}  err={errs:<5d}  "
                f"last_ok={age:>7s}  {status}")

        lines.append(f"{B}{'─'*W}{R}")

        # --- Markers panel ---
        lines.append(f"  {B}Reference markers:{R}")
        ref_ids = sorted([tid for tid in all_marker_positions if tid in REFERENCE_TAG_IDS])
        if ref_ids:
            for tid in ref_ids:
                x, y, z, _ = all_marker_positions[tid]
                lines.append(f"    #{tid:2d}  X={x*100:7.1f} cm   Y={y*100:7.1f} cm   "
                             f"dist={np.sqrt(x**2+y**2)*100:6.1f} cm")
        else:
            lines.append(f"    {Y}(none visible — need at least one of 95-99){R}")

        lines.append(f"  {B}Mobile markers:{R}")
        if mobile_markers:
            for tid in sorted(mobile_markers):
                x, y, z, heading = mobile_markers[tid]
                sent = f"  {G}↑ sent{R}" if vpfs_sent else ""
                lines.append(
                    f"    #{tid:2d}  X={x*100:7.1f} cm   Y={y*100:7.1f} cm   "
                    f"hdg={np.degrees(heading):6.1f}°   "
                    f"dist={np.sqrt(x**2+y**2)*100:6.1f} cm{sent}")
        else:
            lines.append(f"    {Y}(none detected){R}")

        lines.append(f"{B}{'─'*W}{R}")
        lines.append(f"  {self._WHITE}Ctrl+C to quit   warnings → stderr   full log → vps_debug.log{R}")

        return lines


# Pre-compute marker coordinate system (saves computation per frame)
OBJ_POINTS = np.array([[-Defaults.TAG_SIZE/2,  Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,   Defaults.TAG_SIZE/2, 0],
                       [Defaults.TAG_SIZE/2,  -Defaults.TAG_SIZE/2, 0],
                       [-Defaults.TAG_SIZE/2, -Defaults.TAG_SIZE/2, 0]], dtype=np.float32)


class CameraCapture:
    """
    Continuous background capture thread for a single camera.

    The Brio 4K produces frames at ~30fps regardless of how fast the main
    loop reads them.  If the main loop is slower (e.g. 1–5 Hz) the V4L2
    kernel buffer fills up, the driver throws select() timeouts, and the
    camera appears to crash.

    This class runs cap.read() in a daemon thread at full camera speed,
    always discarding old frames and keeping only the most recent one.
    The main loop calls get_frame() whenever it needs a fresh image; it
    never touches the VideoCapture object directly.

    Result: buffer is _always_ drained, crashes disappear, and the main
    loop can run at any frequency without affecting camera stability.
    """

    def __init__(self, cam_info: dict):
        self._cam_info = cam_info
        self._cap: cv2.VideoCapture = cam_info["cap"]
        self._lock  = threading.Lock()
        self._frame = None          # most recent decoded frame
        self._ok    = False         # whether the last read succeeded
        self._running = False
        self._thread: threading.Thread | None = None
        self._consecutive_errors = 0
        self._MAX_ERRORS = 30       # ~1 second of failures at 30fps before recovery
        # Diagnostic counters — read by the watchdog thread without a lock
        # (occasional torn reads are fine; these are for logging only)
        self._frames_ok:   int         = 0
        self._frames_err:  int         = 0
        self._first_err_ts: float|None = None  # wall time of first read error in current streak
        self._last_ok_ts:   float|None = None  # wall time of last successful read
        self._dmesg_done:  bool        = False  # dump dmesg only once per error streak

    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True,
                                        name=f"capture-{self._cam_info['name']}")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def get_frame(self):
        """Return (ok, frame) — the latest frame captured by the background thread."""
        # Minimise lock hold time: grab a reference under the lock, copy outside.
        # Holding the lock during a 6-12 MB memcopy blocks the capture thread from
        # draining the V4L2 buffer, causing back-pressure on the USB driver.
        with self._lock:
            ok = self._ok
            ref = self._frame  # reference only, no copy
        return ok, (ref.copy() if ref is not None else None)

    @property
    def is_alive(self):
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _capture_loop(self):
        name   = self._cam_info["name"]
        cam_id = self._cam_info["id"]
        device = Defaults.CAMERA_SYMLINKS[cam_id]
        exit_reason = "unknown"

        try:
            while self._running:
                ret, frame = self._cap.read()

                if ret and frame is not None:
                    self._frames_ok += 1
                    self._last_ok_ts = time.time()
                    with self._lock:
                        self._frame = frame
                        self._ok    = True
                    # Log recovery if we had an error streak
                    if self._consecutive_errors > 0:
                        log.info(f"[{name}] Read recovered after {self._consecutive_errors} errors "
                                 f"(total ok={self._frames_ok} err={self._frames_err})")
                    self._consecutive_errors = 0
                    self._first_err_ts = None
                    self._dmesg_done   = False

                else:
                    now = time.time()
                    self._frames_err        += 1
                    self._consecutive_errors += 1
                    with self._lock:
                        self._ok = False

                    # Record when this streak started
                    if self._first_err_ts is None:
                        self._first_err_ts = now
                        log.warning(f"[{name}] First read error — "
                                    f"ok={self._frames_ok} err={self._frames_err} "
                                    f"cap.isOpened={self._cap.isOpened()}")

                    # On 3rd consecutive error: dump V4L2 state + dmesg
                    if not self._dmesg_done and self._consecutive_errors == 3:
                        self._dmesg_done = True
                        log_v4l2_state(device, label=f"{name} at error onset")
                        dump_dmesg_usb(label=f"{name} error onset")

                    # Log every 5 errors to show progression
                    elif self._consecutive_errors % 5 == 0:
                        streak_s = now - self._first_err_ts
                        log.warning(f"[{name}] {self._consecutive_errors} consecutive errors "
                                    f"(streak={streak_s:.1f}s ok={self._frames_ok} err={self._frames_err})")

                    # Throttle retries to avoid hammering the USB bus with rapid V4L2
                    # ioctls — a tight spin with no sleep is the primary cause of hard
                    # camera crashes that require a computer reset.
                    time.sleep(1.0 / 30.0)

                    if self._consecutive_errors >= self._MAX_ERRORS:
                        log.error(f"[{name}] {self._consecutive_errors} consecutive failures — "
                                  f"triggering recovery "
                                  f"(ok={self._frames_ok} err={self._frames_err})")
                        dump_dmesg_usb(label=f"{name} pre-recovery")
                        self._try_recover()
                        self._consecutive_errors = 0

            exit_reason = "_running flag cleared (normal stop)"

        except Exception as exc:
            import traceback
            exit_reason = f"UNHANDLED EXCEPTION: {exc}"
            log.error(f"[{name}] Capture thread crashed — {exit_reason}\n"
                      f"{traceback.format_exc()}")
            dump_dmesg_usb(label=f"{name} thread crash")
            with self._lock:
                self._ok = False
            self._running = False

        finally:
            log.warning(f"[{name}] Capture thread exiting. Reason: {exit_reason} "
                        f"(ok={self._frames_ok} err={self._frames_err})")  

    def _try_recover(self):
        name   = self._cam_info["name"]
        cam_id = self._cam_info["id"]
        device = Defaults.CAMERA_SYMLINKS[cam_id]

        log.warning(f"[{name}] Starting recovery sequence...")
        try:
            force_release_camera(self._cap, device, cam_id)
            time.sleep(1.0)
            reset_usb_device(device)
            time.sleep(1.0)

            new_cap = initialize_camera(cam_id, self._cam_info["K"], self._cam_info["D"])
            if new_cap is not None and new_cap.isOpened():
                self._cap = new_cap
                self._cam_info["cap"] = new_cap
                log.info(f"[{name}] Recovery successful.")
            else:
                log.error(f"[{name}] Recovery failed — camera disabled.")
                self._running = False
        except Exception as exc:
            import traceback
            log.error(f"[{name}] Recovery raised an exception — disabling camera.\n"
                      f"{traceback.format_exc()}")
            self._running = False


def force_release_camera(cam, camera_device, camera_id):
    """
    Aggressively release camera with multiple attempts and USB reset.
    """
    if cam is None:
        return
    
    print(f"  Releasing camera {camera_id}...")
    
    # Try normal release multiple times
    for attempt in range(3):
        try:
            cam.release()
            time.sleep(0.2)
            break
        except Exception as e:
            print(f"    Release attempt {attempt+1} failed: {e}")
            time.sleep(0.1)
    
    # Brief pause to let the release propagate.
    time.sleep(0.3)
    
    # NOTE: Do NOT use 'fuser -k' here — if cam.release() leaves any fd open
    # (a known OpenCV/V4L2 race) that command would kill this process itself.
    
    print(f"  Camera {camera_id} released")


def reset_usb_device(camera_device):
    """
    Attempt to reset USB device using udevadm.
    This can help recover from hard crashes without rebooting.
    """
    try:
        print(f"  Attempting USB device reset for {camera_device}...")
        
        # Trigger udev action to re-enumerate device
        result = os.system(f"udevadm trigger --action=change {camera_device} 2>/dev/null")
        time.sleep(1.0)
        
        if result == 0:
            print(f"  USB device reset successful")
            return True
        else:
            print(f"  USB device reset failed (code {result})")
            return False
    except Exception as e:
        print(f"  USB reset exception: {e}")
        return False


def initialize_camera(camera_id, CAM_K, CAM_D):
    """Initialize a single camera with proper settings."""
    camera_device = Defaults.CAMERA_SYMLINKS[camera_id]
    log.info(f"Initializing camera {camera_id} ({camera_device})...")

    # Disable USB autosuspend before opening the device.
    # This is the most common cause of simultaneous multi-camera drops on Linux:
    # the USB power manager silently suspends the device after a few seconds of
    # perceived inactivity, requiring a computer reset to recover.
    disable_usb_autosuspend(camera_device)

    # Reset camera controls to known defaults (use resolved v4l2-ctl path)
    v4l = _V4L2_CTL
    os.system(f"{v4l} -d {camera_device} -c focus_automatic_continuous=0 2>/dev/null")
    os.system(f"{v4l} -d {camera_device} -c focus_absolute=0 2>/dev/null")
    os.system(f"{v4l} -d {camera_device} -c auto_exposure=1 2>/dev/null")
    os.system(f"{v4l} -d {camera_device} -c exposure_time_absolute=200 2>/dev/null")
    os.system(f"{v4l} -d {camera_device} -c brightness=128 2>/dev/null")

    # Create camera and set format.
    # IMPORTANT: CAP_PROP_BUFFERSIZE must be set BEFORE open() on V4L2 — that is
    # where OpenCV passes it to the kernel REQBUFS ioctl that allocates the queue.
    cam = cv2.VideoCapture()
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # <- must come before open()
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(camera_device, cv2.CAP_V4L2)

    # Re-apply buffer size after open as a belt-and-suspenders measure
    # (some OpenCV builds apply it only after open).
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Log what the V4L2 driver actually negotiated right after open.
    # This tells us whether BUFFERSIZE=1 and FPS=15 were honoured.
    log_v4l2_state(camera_device, label=f"cam{camera_id} after open")
    
    # Verify and reapply resolution if needed
    actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != Defaults.CAM_WIDTH or actual_h != Defaults.CAM_HEIGHT:
        print(f"First attempt: {actual_w}x{actual_h}, retrying with settings...")
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)

    # Cap FPS explicitly — three Brio 4K cameras at the default ~30fps MJPEG
    # can saturate a single USB 3.0 controller, causing isochronous transfer
    # errors that progressively corrupt the V4L2 state and require a reboot.
    # 15fps halves per-camera USB bandwidth while still being faster than the
    # fastest main-loop update rate (10Hz default).
    cam.set(cv2.CAP_PROP_FPS, 15)

    # Camera control settings (applied via OpenCV as backup)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Disable autofocus
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure mode
    cam.set(cv2.CAP_PROP_EXPOSURE, 185)     # Set exposure

    max_fps = int(cam.get(cv2.CAP_PROP_FPS))

    # Log actual capture format
    frameWidth  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf_size    = int(cam.get(cv2.CAP_PROP_BUFFERSIZE))
    exposure_value = cam.get(cv2.CAP_PROP_EXPOSURE)
    log.info(f"  Camera {camera_id}: {frameWidth}x{frameHeight} @ {max_fps} fps  "
             f"buffer={buf_size}  exposure={exposure_value}")

    # Final V4L2 state query — confirms FPS cap and buffer request were applied
    log_v4l2_state(camera_device, label=f"cam{camera_id} fully configured")

    # Verify camera is available
    if not cam.isOpened():
        print(f"Cannot open camera {camera_id}")
        return None

    return cam


def process_camera_frame(frame, camera_id, camera_name, CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, gpu_frame=None, gpu_gray=None, show_display=True):
    """
    Process a single frame from one camera for ArUco detection.
    
    Distortion handling strategy:
    - Detect ArUco corners on the ORIGINAL (distorted) grayscale image.
    - Undistort only the detected corner points using cv2.undistortPoints.
      This preserves the original camera matrix (CAM_K) for solvePnP and
      avoids the focal-length shrinkage that getOptimalNewCameraMatrix(alpha=1)
      introduces for lenses with large distortion coefficients.
    - Full-image undistortion (alpha=0) is only done for final display.
    """
    # --- Detection on original image ---
    if USE_GPU and gpu_frame is not None and gpu_gray is not None:
        gpu_frame.upload(frame)
        cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY, gpu_gray)
        gray = gpu_gray.download()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if DETECTOR is not None:
        corners, ids, _ = DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    rvecs, tvecs = None, None
    detections = []
    if ids is not None and len(ids) > 0:
        rvecs = []
        tvecs = []
        for corner in corners:
            # Undistort corner points only (shape: 1x4x2 -> 4x1x2 for undistortPoints)
            pts = corner.reshape(-1, 1, 2).astype(np.float32)
            pts_undistorted = cv2.undistortPoints(pts, CAM_K, CAM_D, P=CAM_K)
            # solvePnP with original CAM_K and no distortion (points already corrected)
            success, rvec, tvec = cv2.solvePnP(
                OBJ_POINTS, pts_undistorted, CAM_K, None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)

        for i, tag_id in enumerate(ids.flatten()):
            detections.append(
                ArucoDetection(
                    tag_id=tag_id,
                    rvec=rvecs[i].reshape(3),
                    tvec=tvecs[i].reshape(3),
                    corners=corners[i]
                )
            )

    # Estimate camera pose from reference tags
    cameraPos = None
    try:
        cameraPos = compute_camera_pos(detections)
    except:
        pass

    # --- Display work: skip entirely in headless mode ---
    # cv2.undistort on a full 1080p frame costs ~50-80ms per camera.
    # In headless mode this work is pure waste, so we return None.
    if not show_display:
        return None, detections, cameraPos

    # --- World coordinates for overlay (display only) ---
    world_positions = None
    if detections:
        temp_dict = {camera_id: detections}
        try:
            world_positions = compute_world_positions(temp_dict)
        except:
            pass

    # --- Undistort full image for display only (alpha=0: no black borders) ---
    display_frame = cv2.undistort(frame, CAM_K, CAM_D)
    display_frame = draw_aruco_overlays(
        display_frame, corners, ids, CAM_K, None,
        Defaults.TAG_SIZE, rvecs, tvecs, world_positions
    ) if ids is not None else display_frame

    # Add camera name overlay
    cv2.putText(display_frame, camera_name, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3, cv2.LINE_AA)
    if cameraPos is not None:
        cameraTranslation = cameraPos[0:3, 3].flatten()
        pos_text = f"Pos: X{cameraTranslation[0]:.2f} Y{cameraTranslation[1]:.2f} Z{cameraTranslation[2]:.2f}"
        cv2.putText(display_frame, pos_text, (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)

    return display_frame, detections, cameraPos


def process_frame_only(frame, cam_info, CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS, frame_time, gpu_frame=None, gpu_gray=None, show_display=True):
    """Worker function to process a captured frame (detection only, no capture)."""
    if frame is None:
        if show_display:
            blank_frame = np.zeros((Defaults.CAM_HEIGHT, Defaults.CAM_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"{cam_info['name']} - NO SIGNAL",
                        (50, Defaults.CAM_HEIGHT // 2), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 0, 255), 3, cv2.LINE_AA)
            return blank_frame, [], None, time.time()
        return None, [], None, time.time()

    # Process frame for ArUco detection
    processed_frame, detections, cameraPos = process_camera_frame(
        frame, cam_info["id"], cam_info["name"],
        CAM_K, CAM_D, DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
        gpu_frame, gpu_gray, show_display=show_display
    )

    current_time = time.time()

    if show_display and processed_frame is not None:
        # Add FPS overlay
        frameTime = current_time - frame_time
        fps = 1 / frameTime if frameTime > 0 else 0.0
        h, w = processed_frame.shape[:2]
        cv2.putText(processed_frame, f"{w}x{h} @ {fps:.1f}fps",
                    (10, h - 10), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 255, 255), 3, cv2.LINE_AA)

    return processed_frame, detections, cameraPos, current_time


def fuse_tag_poses_from_cameras(detections_by_camera):
    """
    Fuse tag detections from multiple cameras to compute world/map poses.
    
    Workflow:
    1. For each camera, compute camera pose using reference tags (95-99)
    2. Transform all detected tags (reference + mobile) to world coordinates
    3. Fuse multiple observations of the same tag across cameras by weighted averaging
    
    This function processes tag detections from multiple cameras, computes each camera's
    pose in the world coordinate system, transforms all tag detections to world coordinates,
    and fuses multiple observations of the same tag by averaging their positions and orientations.
    
    Args:
        detections_by_camera: Dict mapping camera_id (str or int) -> list of ArucoDetection objects
            Each ArucoDetection should have: tag_id, pose_R (3x3 rotation), pose_t (3x1 translation)
    
    Returns:
        Dict mapping tag_id (int) -> 4x4 transformation matrix in world/map coordinates
        Returns empty dict if no valid detections are found.
    
    Example:
        detections_by_camera = {
            'camera0': [det1, det2, det3],
            'camera1': [det4, det5],
            'camera2': [det6]
        }
        tag_poses = fuse_tag_poses_from_cameras(detections_by_camera)
        # tag_poses = {tag_id: 4x4_transform_matrix, ...}
    """
    
    # Step 1: Compute camera poses and transform all detections to world coordinates
    tag_observations = {}  # tag_id -> list of (4x4 transform matrix, weight)
    
    for camera_id, detections in detections_by_camera.items():
        if not detections:
            continue
            
        # Compute camera pose in world/map coordinates
        camera_pose = compute_camera_pos(detections)
        
        if camera_pose is None:
            # Cannot determine camera pose (no reference tags detected)
            continue
        
        # Transform each detected tag to world coordinates
        for det in detections:
            # Get camera->tag transform
            cam_to_tag = det_to_transform_mat(det)
            
            # Compute map->tag = (map->cam) * (cam->tag)
            map_to_tag = np.matmul(camera_pose, cam_to_tag)
            
            # Weight based on distance (closer = higher weight)
            dist = float(np.linalg.norm(det.pose_t.flatten()))
            weight = 1.0 / max(dist, 1e-3)
            
            # Store observation for this tag
            if det.tag_id not in tag_observations:
                tag_observations[det.tag_id] = []
            tag_observations[det.tag_id].append((map_to_tag, weight))
    
    # Step 2: Fuse multiple observations of each tag
    fused_tag_poses = {}
    
    for tag_id, observations in tag_observations.items():
        if len(observations) == 1:
            # Only one observation, use it directly
            fused_tag_poses[tag_id] = observations[0][0]
        else:
            # Multiple observations - fuse by averaging rotation and translation
            transforms = [obs[0] for obs in observations]
            weights = [obs[1] for obs in observations]
            
            # Extract rotations and translations
            rotations = [T[:3, :3] for T in transforms]
            translations = [T[:3, 3] for T in transforms]
            
            # Average rotation matrices using SVD (from utils._average_rotations logic)
            M = np.zeros((3, 3), dtype=float)
            wsum = 0.0
            for R, w in zip(rotations, weights):
                M += float(w) * R
                wsum += float(w)
            M /= wsum
            U, _, Vt = np.linalg.svd(M)
            R_avg = U @ Vt
            if np.linalg.det(R_avg) < 0:
                U[:, -1] *= -1
                R_avg = U @ Vt
            
            # Average translations
            t_avg = np.zeros(3, dtype=float)
            for t, w in zip(translations, weights):
                t_avg += float(w) * t
            t_avg /= wsum
            
            # Build fused transform matrix
            T_fused = np.eye(4)
            T_fused[:3, :3] = R_avg
            T_fused[:3, 3] = t_avg
            
            fused_tag_poses[tag_id] = T_fused
    
    return fused_tag_poses


def compute_world_positions(detections_by_camera):
    """
    Compute world-frame marker positions using hardcoded reference tag coordinates.

    For each camera:
      1. Call compute_camera_pos() to estimate map->camera transform from any
         reference markers (IDs 95-99) visible in that camera's frame.
      2. Call compute_tag_poses() to project ALL detected markers into world space.
    Then average world positions for tags seen by multiple cameras.

    No external calibration file required. Works as long as at least one
    reference marker is visible to each camera each frame.

    Args:
        detections_by_camera: dict  camera_id -> list[ArucoDetection]

    Returns:
        dict  tag_id -> (x_m, y_m, z_m)  in world coordinates (metres)
    """
    all_observations = {}  # tag_id -> list of (x, y, z, heading)

    for camera_id, detections in detections_by_camera.items():
        if not detections:
            continue

        # Estimate this camera's pose from visible reference markers
        # map_to_cam: transforms a point in map frame into camera frame
        map_to_cam = compute_camera_pos(detections)
        if map_to_cam is None:
            continue  # no reference markers visible on this camera this frame

        for det in detections:
            # cam_to_tag: transforms a point in camera frame into tag frame
            cam_to_tag = det_to_transform_mat(det)
            # Correct chain: p_tag = cam_to_tag @ map_to_cam @ p_world
            #   so map_to_tag = cam_to_tag @ map_to_cam
            # Tag origin in world = inv(map_to_tag)[:3,3] = -R.T @ t
            map_to_tag = cam_to_tag @ map_to_cam
            R = map_to_tag[:3, :3]
            t = map_to_tag[:3, 3]
            tag_pos_world = -R.T @ t   # tag origin in map/world coordinates
            # Heading: angle of tag X axis projected to world XY plane.
            # R transforms world->tag, so tag X in world = first row of R.
            heading = float(np.arctan2(R[0, 1], R[0, 0]))

            all_observations.setdefault(int(det.tag_id), []).append(
                (tag_pos_world[0], tag_pos_world[1], tag_pos_world[2], heading)
            )

    # Average across cameras for tags seen by more than one.
    # Position: arithmetic mean. Heading: circular mean (handles wraparound).
    result = {}
    for tag_id, obs in all_observations.items():
        arr = np.array(obs)  # shape (N, 4): x, y, z, heading
        mean_xyz = arr[:, :3].mean(axis=0)
        headings = arr[:, 3]
        mean_heading = float(np.arctan2(np.sin(headings).mean(), np.cos(headings).mean()))
        result[tag_id] = (float(mean_xyz[0]), float(mean_xyz[1]), float(mean_xyz[2]), mean_heading)

    return result


def compute_relative_marker_positions(detections_by_camera):
    """
    Compute marker positions relative to marker 95 (origin) without pre-defined world positions.
    
    This function establishes a coordinate system directly from the detected markers:
    - Marker 95 is placed at the origin (0, 0, 0)
    - Marker 96 defines the positive X-axis direction
    - The coordinate frame is built from the physical arrangement
    - All markers are expressed in this relative coordinate system
    
    Workflow:
    1. For each camera, detect all markers in camera coordinates
    2. Select a reference camera that sees marker 95
    3. In the reference camera frame, set marker 95 as origin
    4. Optionally align coordinate frame so marker 96 is along +X axis
    5. For other cameras, compute their pose relative to this coordinate system
    6. Transform all detections from all cameras into the unified frame
    7. Fuse multiple observations of the same marker
    
    Args:
        detections_by_camera: Dict mapping camera_id -> list of ArucoDetection objects
        
    Returns:
        Dict mapping tag_id -> (x, y, z) in meters relative to marker 95
        Returns empty dict if marker 95 is not detected
    """
    ORIGIN_MARKER_ID = 95
    X_AXIS_MARKER_ID = 96
    
    # Step 1: Find a reference camera that sees the origin marker (95)
    reference_camera_id = None
    reference_detections = None
    origin_detection = None
    
    for camera_id, detections in detections_by_camera.items():
        for det in detections:
            if det.tag_id == ORIGIN_MARKER_ID:
                reference_camera_id = camera_id
                reference_detections = detections
                origin_detection = det
                break
        if origin_detection:
            break
    
    if origin_detection is None:
        print(f"Warning: Origin marker {ORIGIN_MARKER_ID} not detected in any camera")
        return {}
    
    print(f"\nUsing camera {reference_camera_id} as reference (sees origin marker {ORIGIN_MARKER_ID})")
    
    # Step 2: Build coordinate transform - origin at marker 95
    # Get the camera->origin transform
    cam_to_origin = det_to_transform_mat(origin_detection)
    
    # To make marker 95 the origin, we need origin->camera transform
    # Then all markers will be expressed as: origin->marker = origin->camera * camera->marker
    origin_to_cam = np.linalg.inv(cam_to_origin)
    
    # Step 3: Collect all marker positions in the origin-centered frame
    marker_positions_raw = {}  # tag_id -> list of ((x,y,z), weight)
    
    for det in reference_detections:
        cam_to_marker = det_to_transform_mat(det)
        origin_to_marker = np.matmul(origin_to_cam, cam_to_marker)
        
        translation = origin_to_marker[:3, 3]
        dist = float(np.linalg.norm(det.pose_t.flatten()))
        weight = 1.0 / max(dist, 1e-3)
        
        if det.tag_id not in marker_positions_raw:
            marker_positions_raw[det.tag_id] = []
        marker_positions_raw[det.tag_id].append((translation, weight))
    
    # Step 4: Optionally align coordinate frame with marker 96 along X-axis
    # Find marker 96 position
    x_axis_marker_pos = None
    if X_AXIS_MARKER_ID in marker_positions_raw:
        # Average if multiple observations
        positions = [obs[0] for obs in marker_positions_raw[X_AXIS_MARKER_ID]]
        weights = [obs[1] for obs in marker_positions_raw[X_AXIS_MARKER_ID]]
        wsum = sum(weights)
        x_axis_marker_pos = sum(w * p for w, p in zip(weights, positions)) / wsum
        
        # Compute rotation to align marker 96 with +X axis
        # Project onto XY plane
        direction_xy = x_axis_marker_pos[:2]
        direction_xy_norm = np.linalg.norm(direction_xy)
        
        if direction_xy_norm > 1e-3:
            # Angle to rotate around Z-axis to align with +X
            angle = np.arctan2(direction_xy[1], direction_xy[0])
            
            # Rotation matrix around Z-axis
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            R_align = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a,  cos_a, 0, 0],
                [0,      0,     1, 0],
                [0,      0,     0, 1]
            ])
            
            print(f"Aligning coordinate frame: rotating {np.degrees(angle):.1f}° to place marker {X_AXIS_MARKER_ID} along +X axis")
            
            # Apply rotation to all markers
            aligned_positions = {}
            for tag_id, observations in marker_positions_raw.items():
                aligned_obs = []
                for pos, weight in observations:
                    pos_homogeneous = np.array([pos[0], pos[1], pos[2], 1])
                    aligned_pos = (R_align @ pos_homogeneous)[:3]
                    aligned_obs.append((aligned_pos, weight))
                aligned_positions[tag_id] = aligned_obs
            
            marker_positions_raw = aligned_positions
    
    # Step 5: Process other cameras using the established coordinate frame
    # For each other camera, find common markers to compute its pose
    for camera_id, detections in detections_by_camera.items():
        if camera_id == reference_camera_id or not detections:
            continue
        
        # Find markers common with the reference camera
        common_markers = {}
        for det in detections:
            if det.tag_id in marker_positions_raw:
                common_markers[det.tag_id] = det
        
        if len(common_markers) < 2:
            print(f"Camera {camera_id}: Not enough common markers ({len(common_markers)}), skipping")
            continue
        
        # Estimate this camera's pose using common markers (similar to PnP)
        # For simplicity, use marker 95 if visible, else use any common marker
        if ORIGIN_MARKER_ID in common_markers:
            # Direct transform through origin marker
            det_origin = common_markers[ORIGIN_MARKER_ID]
            cam_to_origin = det_to_transform_mat(det_origin)
            origin_to_cam = np.linalg.inv(cam_to_origin)
            
            # Add all detections from this camera
            for det in detections:
                cam_to_marker = det_to_transform_mat(det)
                origin_to_marker = np.matmul(origin_to_cam, cam_to_marker)
                
                translation = origin_to_marker[:3, 3]
                dist = float(np.linalg.norm(det.pose_t.flatten()))
                weight = 1.0 / max(dist, 1e-3)
                
                if det.tag_id not in marker_positions_raw:
                    marker_positions_raw[det.tag_id] = []
                marker_positions_raw[det.tag_id].append((translation, weight))
        else:
            print(f"Camera {camera_id}: Origin marker not visible, using approximate alignment")
            # Could implement more sophisticated alignment here
    
    # Step 6: Fuse multi-camera observations by weighted averaging
    marker_positions = {}
    for tag_id, observations in marker_positions_raw.items():
        positions = [obs[0] for obs in observations]
        weights = [obs[1] for obs in observations]
        wsum = sum(weights)
        
        # Weighted average
        avg_pos = sum(w * p for w, p in zip(weights, positions)) / wsum
        marker_positions[tag_id] = (float(avg_pos[0]), float(avg_pos[1]), float(avg_pos[2]))
    
    return marker_positions


def main(argv=None):
    """
    Main loop for multi-camera visualization and processing.
    
    Camera Handling Strategy:
    - Frames are captured SEQUENTIALLY (one camera at a time) to prevent USB bandwidth saturation
    - Processing (ArUco detection) is done in PARALLEL for performance
    - Frame buffers are flushed before capture to prevent overflow and ensure fresh frames
    - Automatic camera recovery if consecutive failures are detected
    
    Update Frequency:
    - System operates at fixed frequency (default 10Hz, configurable with --hz)
    - Mimics real positioning systems (GPS, motion capture, etc.)
    - Ensures deterministic, predictable updates
    - Prevents CPU overload and camera crashes
    - Better for integration with control systems (fixed dt)
    
    Camera Recovery:
    - After 10 consecutive failures, attempts aggressive recovery:
      1. Force release camera handle (multiple attempts)
      2. Reset USB device at driver level (udevadm)
      3. Reinitialize camera
    - If recovery fails, camera is marked as permanently failed
    - System continues operating with remaining working cameras
    - If all cameras fail, system exits with error message
    
    Temporal Smoothing:
    - Exponential moving average applied to marker positions
    - SMOOTHING_ALPHA = 0.3 (lower = smoother but slower response, higher = faster but more jitter)
    - Reduces fluctuations from detection noise
    
    Command-line Arguments:
    - --hz <frequency> : Update frequency in Hz (default: 10, recommended: 1-20)
    - --no-display : Run in headless mode (no visual windows, faster performance)
    - --calib <path> : Path to camera intrinsics calibration file
    """
    # Parse command-line flags
    show_display = True
    update_frequency_hz = 10  # Default 10Hz update rate
    
    if argv:
        if '--no-display' in argv:
            show_display = False
            print("Running in headless mode (no visual display)")
        
        # Parse update frequency
        if '--hz' in argv:
            idx = argv.index('--hz')
            if idx + 1 < len(argv):
                try:
                    update_frequency_hz = float(argv[idx + 1])
                    if update_frequency_hz <= 0 or update_frequency_hz > 60:
                        print(f"Warning: Invalid frequency {update_frequency_hz}Hz, using default 10Hz")
                        update_frequency_hz = 10
                except ValueError:
                    print(f"Warning: Invalid frequency value, using default 10Hz")
                    update_frequency_hz = 10

        # Connect to VPFS backend if --vpfs flag is provided
        # Usage: --vpfs                  (connects to http://localhost:5000)
        #        --vpfs http://host:5000  (connects to custom host)
        if '--vpfs' in argv:
            idx = argv.index('--vpfs')
            vpfs_url = "http://localhost:5000"
            if idx + 1 < len(argv) and not argv[idx + 1].startswith('--'):
                vpfs_url = argv[idx + 1]
            vpfs_connector.connect_to_server(vpfs_url)

    # Display system configuration
    target_loop_time = 1.0 / update_frequency_hz
    print(f"\n{'='*70}")
    print(f"MULTI-CAMERA POSITIONING SYSTEM")
    print(f"{'='*70}")
    print(f"Update Frequency: {update_frequency_hz} Hz (period: {target_loop_time*1000:.1f}ms)")
    print(f"Display Mode: {'Visual' if show_display else 'Headless (faster)'}")
    print(f"Reference Tags: hardcoded world positions from ref_tags.py")
    print(f"{'='*70}\n")
    
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
    
    # --- Start background capture threads (one per camera) ---
    # Each thread reads at the camera's full native rate (~30fps) so the V4L2
    # kernel buffer never fills up, regardless of how slow the main loop is.
    captures = []
    for cam_info in cameras:
        c = CameraCapture(cam_info)
        c.start()
        captures.append(c)
    log.info(f"Started {len(captures)} background capture thread(s).")

    # --- Watchdog thread ---
    # Every 15 s it logs per-camera frame counters, last-ok age, and thread
    # liveness to vps_debug.log.  This gives a clear time-series so we can
    # see exactly when a camera stopped delivering frames before the crash.
    def _watchdog(captures_ref, interval: float = 15.0):
        while any(c._running for c in captures_ref):
            time.sleep(interval)
            now = time.time()
            lines = ["=== CAMERA HEALTH ==="]
            for c in captures_ref:
                last = c._last_ok_ts
                age  = f"{now - last:.1f}s ago" if last else "never"
                lines.append(
                    f"  {c._cam_info['name']}: "
                    f"frames_ok={c._frames_ok}  frames_err={c._frames_err}  "
                    f"last_ok={age}  alive={c.is_alive}  "
                    f"consecutive_errors={c._consecutive_errors}")
            lines.append("=====================")
            log.info("\n".join(lines))

    _watchdog_thread = threading.Thread(
        target=_watchdog, args=(captures,),
        daemon=True, name="vps-watchdog")
    _watchdog_thread.start()
    log.info("Watchdog thread started (15 s interval) — output in vps_debug.log")

    # Main loop
    frame_times = [time.time()] * len(cameras)

    # Update frequency tracking
    update_count = 0
    start_time   = time.time()
    loop_ms      = 0.0   # processing time of the most recent loop iteration

    # In-place terminal dashboard
    dashboard = TerminalDashboard()

    # Temporal smoothing for marker positions (reduces jitter)
    # Using exponential moving average: smoothed = alpha * new + (1-alpha) * old
    smoothed_positions = {}  # tag_id -> (x, y, z)
    SMOOTHING_ALPHA = 0.3  # 0.3 = more smoothing, 0.7+ = more responsive

    # Timing diagnostics
    last_timing_warning = 0
    TIMING_WARNING_INTERVAL = 10.0  # Warn every 10 seconds if processing is slow

    # Wait briefly for background threads to fill their first frame
    time.sleep(0.5)

    print("\nStarting main processing loop...")
    print("Press ESC (with display) or Ctrl+C (headless) to exit\n")

    while True:
        loop_start = time.time()

        # Check if all background capture threads have died
        if not any(c.is_alive for c in captures):
            print("\nERROR: All camera capture threads have stopped. Exiting.")
            break

        # Step 1: Snapshot the latest frame from every camera simultaneously.
        # Background capture threads keep draining the V4L2 buffer at full
        # speed regardless of how long the processing below takes, so there
        # is no back-pressure on the USB driver.
        # The snapshot itself is instant (no V4L2 ioctls in this thread).
        captured_frames = []
        for cap in captures:
            ok, frame = cap.get_frame()
            captured_frames.append(frame if ok else None)

        # Step 2: Process each camera's frame SEQUENTIALLY.
        #
        # Rationale for dropping the ThreadPoolExecutor:
        # - Pool workers doing heavy CPU work (undistort + ArUco on full 1080p
        #   frames) saturated all cores simultaneously, starving the capture
        #   daemon threads of their scheduling slots.  When a capture thread
        #   missed its cap.read() window the V4L2 kernel timeout fired and the
        #   camera appeared to crash.
        # - The cameras are not hardware-synced, so parallel processing gave
        #   no temporal-alignment benefit over sequential.
        # - At 1–10 Hz update rates, sequential processing of 3 cameras is
        #   well within the time budget even on a single core.
        frames = []
        all_detections = []
        all_detections_by_camera = defaultdict(list)

        for idx, cam_info in enumerate(cameras):
            gpu_frame, gpu_gray = gpu_resources[idx]
            try:
                frame, detections, cameraPos, current_time = process_frame_only(
                    captured_frames[idx], cam_info, cam_info["K"], cam_info["D"],
                    DETECTOR, ARUCO_DICT, ARUCO_PARAMS,
                    frame_times[idx], gpu_frame, gpu_gray,
                    show_display=show_display
                )
                frames.append(frame)
                all_detections.extend(detections)
                all_detections_by_camera[cam_info["id"]].extend(detections)
                frame_times[idx] = current_time
            except Exception as e:
                log.warning(f"Processing error for {cam_info['name']}: {e}")
                blank = np.zeros((Defaults.CAM_HEIGHT, Defaults.CAM_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, f"{cam_info['name']} - PROCESSING ERROR",
                            (50, Defaults.CAM_HEIGHT // 2), cv2.FONT_HERSHEY_PLAIN,
                            3, (0, 0, 255), 3, cv2.LINE_AA)
                frames.append(blank)

        # Update statistics
        update_count += 1
        current_time  = time.time()
        loop_ms       = (current_time - loop_start) * 1000
        elapsed       = current_time - start_time
        actual_hz     = update_count / elapsed if elapsed > 0 else 0.0

        # Periodic stats to log file only (not terminal)
        if update_count % max(1, int(update_frequency_hz * 5)) == 0:
            log.info(f"[Stats] updates={update_count} actual={actual_hz:.2f}Hz "
                     f"target={update_frequency_hz}Hz loop={loop_ms:.0f}ms")

        # Compute world-frame marker positions using ref_tags.py coordinates
        raw_marker_positions = compute_world_positions(all_detections_by_camera)

        # Apply temporal smoothing to reduce jitter/fluctuations
        REFERENCE_TAG_IDS = {95, 96, 97, 98, 99}
        all_marker_positions = {}
        for tag_id, (x, y, z, heading) in raw_marker_positions.items():
            if tag_id in smoothed_positions:
                old_x, old_y, old_z, old_h = smoothed_positions[tag_id]
                smoothed_x = SMOOTHING_ALPHA * x + (1 - SMOOTHING_ALPHA) * old_x
                smoothed_y = SMOOTHING_ALPHA * y + (1 - SMOOTHING_ALPHA) * old_y
                smoothed_z = SMOOTHING_ALPHA * z + (1 - SMOOTHING_ALPHA) * old_z
                dh = np.arctan2(np.sin(heading - old_h), np.cos(heading - old_h))
                smoothed_h = float(np.arctan2(
                    np.sin(old_h + SMOOTHING_ALPHA * dh),
                    np.cos(old_h + SMOOTHING_ALPHA * dh)
                ))
                smoothed_positions[tag_id] = (smoothed_x, smoothed_y, smoothed_z, smoothed_h)
            else:
                smoothed_positions[tag_id] = (x, y, z, heading)
            all_marker_positions[tag_id] = smoothed_positions[tag_id]

        # Send mobile markers to VPFS backend
        mobile_markers = {tid: pos for tid, pos in all_marker_positions.items()
                          if tid not in REFERENCE_TAG_IDS}
        vpfs_sent = False
        if mobile_markers:
            vpfs_connector.send_update(mobile_markers)
            vpfs_sent = True

        # Render in-place terminal dashboard (no scrolling)
        dashboard.render(dashboard.build(
            update_count=update_count,
            start_time=start_time,
            loop_ms=loop_ms,
            target_hz=update_frequency_hz,
            actual_hz=actual_hz,
            captures=captures,
            all_marker_positions=all_marker_positions,
            mobile_markers=mobile_markers,
            vpfs_sent=vpfs_sent,
        ))

        # Display each camera in its own window (only if display is enabled)
        if show_display:
            for idx, (frame, cam_info) in enumerate(zip(frames, cameras)):
                try:
                    scale = 0.25
                    new_width  = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    resized = cv2.resize(frame, (new_width, new_height),
                                        interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(cam_info["name"], resized)
                except Exception as e:
                    print(f"Display error for {cam_info['name']}: {e}")

            # Handle keyboard input (ESC to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
        else:
            # In headless mode, minimal wait for keyboard interrupt check
            try:
                time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nReceived interrupt signal, shutting down...")
                break

        # Maintain fixed update frequency
        loop_time = time.time() - loop_start

        # Warn if processing is exceeding target loop time (can't maintain frequency)
        if loop_time > target_loop_time:
            if current_time - last_timing_warning > TIMING_WARNING_INTERVAL:
                print(f"\n⚠ WARNING: Processing too slow for {update_frequency_hz}Hz "
                      f"({loop_time*1000:.0f}ms/loop, target {target_loop_time*1000:.0f}ms)")
                print(f"  Consider: Lower --hz frequency, use --no-display, or reduce camera resolution")
                last_timing_warning = current_time
        else:
            # Sleep to maintain exact frequency
            sleep_time = target_loop_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Cleanup
    print("\nShutting down cameras...")
    # Stop background capture threads (stop() joins the thread internally)
    for cap in captures:
        cap.stop()
        print(f"  ✓ {cap._cam_info['name']} capture thread stopped")
    
    if show_display:
        cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    if total_time > 0 and update_count > 0:
        actual_hz = update_count / total_time
        print(f"\n{'='*70}")
        print(f"Session Summary:")
        print(f"  Runtime: {total_time:.1f}s")
        print(f"  Updates: {update_count}")
        print(f"  Target Frequency: {update_frequency_hz} Hz")
        print(f"  Actual Frequency: {actual_hz:.2f} Hz")
        print(f"  Accuracy: {(actual_hz/update_frequency_hz)*100:.1f}%")
        print(f"{'='*70}")
    
    print("Shutdown complete.")
    

if __name__ == "__main__":
    # Parse command-line arguments using sys.argv for simplicity
    argv = sys.argv[1:]
    main(argv=argv)