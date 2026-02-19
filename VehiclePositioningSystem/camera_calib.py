"""
Camera intrinsics calibration using a 14x9 chessboard pattern.

Workflow with helper:
- Camera is fixed in position, user holds chessboard in different poses.
- System continuously detects and displays chessboard pattern in real-time.
- Helper presses 'C' to capture frame when pattern is well-positioned.
- All captured images are saved to output directory for later recalibration.
- Press 'E' to end collection and run calibration.
- Uses rational polynomial model (8 coefficients) for wide-angle lens distortion.

Usage:
  python camera_calib.py --camera_id 0 --output calibration_images/
  python camera_calib.py --camera_id 1 --output calib_cam1/
  python camera_calib.py --camera_id 2

Arguments:
  --camera_id: Camera to calibrate (0, 1, or 2). Default: 0
  --output: Directory to save calibration images. Default: calib_images_cam{id}/

Controls:
  C = Capture frame (when pattern detected)
  E = End collection and calibrate
  ESC = Quit without saving

Notes:
- Uses camera symlinks and resolution from Defaults class (utils.py).
- Chessboard pattern: 14x9 inner corners.
- Move/tilt the chessboard to cover different areas and angles.
- Collect at least 10-15 good samples for accurate calibration.
"""

import numpy as np
import cv2 as cv
import os
import json
import sys
import shutil
import argparse
from pathlib import Path

from utils import Defaults

# Chessboard pattern configuration
PATTERN_SIZE = (14, 9)  # (width, height) in inner corners
PATTERN_WIDTH, PATTERN_HEIGHT = PATTERN_SIZE

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the chessboard
objp = np.zeros((PATTERN_HEIGHT * PATTERN_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_WIDTH, 0:PATTERN_HEIGHT].T.reshape(-1, 2)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Camera intrinsics calibration using chessboard pattern.")
parser.add_argument("--camera_id", type=int, default=0, choices=[0, 1, 2],
                    help="Camera ID to calibrate (0, 1, or 2). Default: 0")
parser.add_argument("--output", type=str, default=None,
                    help="Output directory for calibration images. Default: calib_images_cam{id}/")
args = parser.parse_args()

camera_id = args.camera_id
camera_device = Defaults.CAMERA_SYMLINKS[camera_id]

# Set output directory
if args.output:
    output_dir = Path(args.output)
else:
    output_dir = Path(__file__).parent / f"calib_images_cam{camera_id}"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n=== Camera Calibration ===")
print(f"Camera ID: {camera_id}")
print(f"Camera Device: {camera_device}")
print(f"Resolution: {Defaults.CAM_WIDTH}x{Defaults.CAM_HEIGHT}")
print(f"Pattern: {PATTERN_WIDTH}x{PATTERN_HEIGHT} inner corners")
print(f"Output Directory: {output_dir}")
print()

is_windows = sys.platform.startswith("win")
has_v4l2ctl = shutil.which("v4l2-ctl") is not None

# Optional: configure camera focus (Linux/Jetson only)
if not is_windows and has_v4l2ctl:
    print(f"Configuring camera settings for {camera_device}...")
    os.system(f"v4l2-ctl -d {camera_device} -c focus_automatic_continuous=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c focus_absolute=0 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c auto_exposure=1 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c exposure_time_absolute=200 2>/dev/null")
    os.system(f"v4l2-ctl -d {camera_device} -c brightness=128 2>/dev/null")

# Open the camera using the appropriate backend
if is_windows:
    # Use DirectShow on Windows
    cam = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    # Disable autofocus if supported
    cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv.CAP_PROP_FOCUS, 0)  # may be ignored by some drivers
else:
    # Use V4L2 backend on Linux/Jetson (simpler than GStreamer)
    cam = cv.VideoCapture()
    cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    cam.open(camera_device, cv.CAP_V4L2)
    
    # Verify and reapply if needed
    actual_w = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    if actual_w != Defaults.CAM_WIDTH or actual_h != Defaults.CAM_HEIGHT:
        print(f"First attempt: {actual_w}x{actual_h}, retrying with settings...")
        cam.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cam.set(cv.CAP_PROP_FRAME_WIDTH, Defaults.CAM_WIDTH)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, Defaults.CAM_HEIGHT)
    
    # Camera control settings
    cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    cam.set(cv.CAP_PROP_EXPOSURE, 185)

# Verify camera opened successfully
if not cam.isOpened():
    raise RuntimeError(f"Failed to open camera: {camera_device}")

# Log actual capture format
frameWidth = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
max_fps = int(cam.get(cv.CAP_PROP_FPS))
print(f"Actual capture: {frameWidth}x{frameHeight} @ {max_fps} fps")
print(f"\nHold chessboard in different positions/angles.")
print(f"Helper presses 'C' to capture when pattern is well-positioned.\n")

# Create preview window
cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
cv.resizeWindow("Calibration", 1280, 720)

# State variables
samples = 0

while True:
    # Grab a frame
    ret, img = cam.read()
    if not ret or img is None:
        continue

    # Convert to grayscale and detect chessboard
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)
    
    corners2 = None
    if found:
        # Subpixel refinement of detected corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Visualize detected corners on the image
        cv.drawChessboardCorners(img, PATTERN_SIZE, corners2, found)
        status_msg = "Pattern FOUND - press C to capture"
        status_color = (0, 255, 0)
    else:
        status_msg = "Searching for pattern..."
        status_color = (100, 100, 100)
    
    # HUD overlay
    cv.putText(img, f"Samples: {samples}", (10, 40), 
              cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(img, status_msg, (10, 90), 
              cv.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv.LINE_AA)
    cv.putText(img, "C=Capture  E=End & Calibrate  ESC=Quit", (10, 140), 
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv.LINE_AA)
    
    # Show preview
    cv.imshow("Calibration", img)
    key = cv.waitKey(1) & 0xFF
    
    # Handle keyboard input
    if key == ord('c') and found and corners2 is not None:
        # Capture and save
        image_filename = output_dir / f"calib_{samples:03d}.png"
        cv.imwrite(str(image_filename), img)
        print(f"✓ Saved sample {samples + 1}: {image_filename.name}")
        samples += 1
    
    elif key == ord('e'):
        # End collection and calibrate
        print(f"\nEnding collection. Total samples: {samples}")
        break
    
    elif key == 27:  # ESC
        print("\nCalibration cancelled.")
        cam.release()
        cv.destroyAllWindows()
        sys.exit(0)

# Cleanup
cam.release()
cv.destroyAllWindows()

# Load and process saved images for calibration
print(f"\n=== Processing Calibration Images ===")
image_files = sorted(output_dir.glob("calib_*.png"))

if len(image_files) < 3:
    print(f"ERROR: Need at least 3 images for calibration, found {len(image_files)}")
    sys.exit(1)

print(f"Found {len(image_files)} images in {output_dir}")

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image
image_size = None

for img_file in image_files:
    img = cv.imread(str(img_file))
    if img is None:
        print(f"  ✗ Failed to load: {img_file.name}")
        continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)
    
    # Find chessboard corners
    found, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)
    
    if found:
        # Refine corner positions
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        print(f"  ✓ Processed: {img_file.name}")
    else:
        print(f"  ✗ Pattern not found in: {img_file.name}")

if len(objpoints) < 3:
    print(f"\nERROR: Successfully processed only {len(objpoints)} images. Need at least 3.")
    sys.exit(1)

print(f"\n=== Running Calibration ===")
print(f"Using {len(objpoints)} images")
print(f"Image size: {image_size[0]}x{image_size[1]}")

# Calibrate using rational polynomial model for wide-angle lens
# This extends distortion coefficients from 5 to 8: (k1,k2,p1,p2,k3,k4,k5,k6)
# Rational model: radial = (1+k1*r²+k2*r⁴+k3*r⁶)/(1+k4*r²+k5*r⁴+k6*r⁶)
flags = cv.CALIB_RATIONAL_MODEL
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, image_size, None, None, flags=flags
)

print(f"\n=== Calibration Complete ===")
print(f"RMS reprojection error: {ret:.4f} pixels")
print(f"Distortion coefficients: {len(dist.ravel())} (rational polynomial model)")

# Save calibration to JSON (NumPy arrays converted to lists for JSON compatibility)
calib = {
    "camera_id": camera_id,
    "camera_device": camera_device,
    "image_width": image_size[0],
    "image_height": image_size[1],
    "camera_matrix": mtx.tolist(),                 # 3x3
    "dist_coeffs": dist.ravel().tolist(),          # k1,k2,p1,p2,k3,k4,k5,k6 (rational model)
    "distortion_model": "rational_polynomial",     # indicates 8-coefficient rational model
    "rms_reprojection_error": float(ret),
    "num_images": len(objpoints),
    "pattern_size": list(PATTERN_SIZE),
    "rvecs": [rv.ravel().tolist() for rv in rvecs],
    "tvecs": [tv.ravel().tolist() for tv in tvecs],
}
# Save with camera ID in filename for multi-camera setups
calib_filename = f"camera{camera_id}_calibration.json" if camera_id > 0 else "camera_calibration.json"
calib_path = os.path.join(os.path.dirname(__file__), calib_filename)
with open(calib_path, "w", encoding="utf-8") as f:
    json.dump(calib, f, indent=2)

print(f"\n=== Results ===")
print(f"Calibration file: {calib_path}")
print(f"Calibration images: {output_dir}")
print(f"Images used: {len(objpoints)}")

# Print camera matrix and derived intrinsics
print("\nCamera Matrix:")
print(mtx)
print("\nIntrinsic Parameters:")
print(f"  fx: {mtx[0][0]:.2f}")
print(f"  fy: {mtx[1][1]:.2f}")
print(f"  cx: {mtx[0][2]:.2f}")
print(f"  cy: {mtx[1][2]:.2f}")
print("\nDistortion Coefficients (Rational Polynomial Model):")
dist_names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
for i, (name, val) in enumerate(zip(dist_names, dist.ravel())):
    print(f"  {name}: {val:.6f}")
    if i >= len(dist.ravel()) - 1:
        break

print(f"\nCalibration complete! You can recalibrate later using the saved images in {output_dir}")
print(f"To use this calibration, pass: --calib {calib_path}")

