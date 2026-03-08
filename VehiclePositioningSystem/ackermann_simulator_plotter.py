"""
Live position plotter for VPFS.

Polls the /whereami/<team> REST endpoint at a configurable rate and plots the
vehicle's position and heading as an arrow on a matplotlib figure.  The axes
limits are derived from the spawn-point bounds in Config/spawn_points.yaml
(with the same fallback defaults used by fare_gen.py / ackermann_simulator.py).

Usage:
    python position_plotter.py [--vpfs URL] [--team INT] [--hz FLOAT]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import yaml


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_VPFS    = "http://localhost:5000"
DEFAULT_TEAM    = 0
DEFAULT_HZ      = 1.0
ARROW_LENGTH    = None   # set proportionally to arena size after loading bounds
TRAIL_LEN       = 20     # number of past positions kept in the trail


def load_spawn_bounds() -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) from Config/spawn_points.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "Config" / "spawn_points.yaml"
    xs: list[float] = []
    ys: list[float] = []

    if config_path.exists():
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
        for sp in data.get("spawn_points", []):
            if sp.get("active", True):
                coords = sp.get("coordinates", {})
                xs.append(float(coords.get("x", 0.0)))
                ys.append(float(coords.get("y", 0.0)))

    if len(xs) < 2:
        for i in range(1, 9):
            xs.append(float(i * 50))
            ys.append(float(i * 50))

    return min(xs), max(xs), min(ys), max(ys)


def fetch_position(base_url: str, team: int) -> dict | None:
    """
    Call GET /whereami/<team>?auth=<team> and return the parsed JSON,
    or None on any error.
    """
    url = f"{base_url}/whereami/{team}"
    try:
        resp = requests.get(url, params={"auth": str(team)}, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("position")   # may be None if team not in match
    except Exception as e:
        print(f"[!] Request failed: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Live VPFS position plotter")
    parser.add_argument("--vpfs",  default=DEFAULT_VPFS,  help="VPFS server URL")
    parser.add_argument("--team",  default=DEFAULT_TEAM,  type=int,   help="Team number")
    parser.add_argument("--hz",    default=DEFAULT_HZ,    type=float, help="Poll rate in Hz")
    args = parser.parse_args()

    x_min, x_max, y_min, y_max = load_spawn_bounds()
    padding = min(x_max - x_min, y_max - y_min) * 0.05
    arrow_len = min(x_max - x_min, y_max - y_min) * 0.06

    # ── Set up figure ──────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(f"VPFS – Team {args.team} position")

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Team {args.team} – live position  ({args.hz:.1f} Hz)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Map boundary rectangle
    rect = mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.5, edgecolor="steelblue", facecolor="none",
        linestyle="--", label="Map bounds",
    )
    ax.add_patch(rect)

    # Trail (faded line of past positions)
    trail_x: list[float] = []
    trail_y: list[float] = []
    (trail_line,) = ax.plot([], [], color="orange", alpha=0.4, linewidth=2, label="Trail")

    # Current position marker
    (pos_marker,) = ax.plot([], [], "o", color="red", markersize=8, label="Position")

    # Heading arrow
    heading_arrow = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )

    # Status text in top-left corner
    status_text = ax.text(
        0.02, 0.98, "Waiting for data…",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", color="gray",
    )

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    interval = 1.0 / args.hz
    print(f"Polling {args.vpfs}/whereami/{args.team} at {args.hz} Hz")

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()

            pos = fetch_position(args.vpfs, args.team)

            if pos is not None:
                x = pos["x"]
                y = pos["y"]
                heading = pos.get("heading", 0.0)   # radians

                # Update trail
                trail_x.append(x)
                trail_y.append(y)
                if len(trail_x) > TRAIL_LEN:
                    trail_x.pop(0)
                    trail_y.pop(0)
                trail_line.set_data(trail_x, trail_y)

                # Update position marker
                pos_marker.set_data([x], [y])

                # Update heading arrow
                dx = arrow_len * math.cos(heading)
                dy = arrow_len * math.sin(heading)
                heading_arrow.set_position((x, y))
                heading_arrow.xy = (x + dx, y + dy)

                status_text.set_text(
                    f"x={x:.3f}  y={y:.3f}  "
                    f"hdg={math.degrees(heading):+.1f}°"
                )
                status_text.set_color("black")
            else:
                status_text.set_text("No position data")
                status_text.set_color("gray")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
