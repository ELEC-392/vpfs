"""
Fare-guided Ackermann simulator for VPFS.

Polls /fares/current/<team> at 1 Hz. When a fare is claimed, the simulated
vehicle automatically drives from the pickup (src) to the dropoff (dest) using
a proportional heading controller.

The matplotlib figure shows:
  - Vehicle trail, position, and heading arrow (red)
  - Pickup waypoint (blue dot)
  - Dropoff waypoint (green dot)
  - Fare status text and a prominent "IN POSITION" banner when the server
    reports the vehicle is at the expected waypoint.

Usage:
    python fare_simulator.py [--vpfs URL] [--team INT]
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vpfs_connector import connect_to_server, send_update

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_VPFS = "http://localhost:5000"
DEFAULT_TEAM = 0
SIM_HZ       = 20        # kinematic steps per second
PUBLISH_HZ   = 1         # position publishes per second
WHEELBASE    = 15.0
SPEED        = 20.0      # units / second while driving
MAX_STEER    = math.radians(35)
TRAIL_LEN    = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_spawn_bounds() -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, 9):
        xs.append(float(i * 50))
        ys.append(float(i * 50))
    return min(xs), max(xs), min(ys), max(ys)


def fetch_current_fare(base_url: str, team: int) -> dict | None:
    """Return the fare dict from /fares/current/<team>, or None."""
    try:
        resp = requests.get(f"{base_url}/fares/current/{team}",
                            params={"auth": str(team)}, timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("fare")
    except Exception as e:
        print(f"[!] Fare request failed: {e}")
        return None


# ── Kinematic model ───────────────────────────────────────────────────────────

class AckermannSim:
    def __init__(self, x: float, y: float, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.steering = 0.0

    def step_toward(self, target_x: float, target_y: float, dt: float) -> None:
        """Single kinematic step steering proportionally toward (target_x, target_y)."""
        desired = math.atan2(target_y - self.y, target_x - self.x)
        diff = (desired - self.heading + math.pi) % (2 * math.pi) - math.pi
        steer_cmd = max(-MAX_STEER, min(MAX_STEER, diff * 1.5))
        # Low-pass filter steering to simulate finite steering rate
        self.steering += (steer_cmd - self.steering) * min(1.0, dt * 5.0)
        self.steering = max(-MAX_STEER, min(MAX_STEER, self.steering))

        self.heading += (SPEED / WHEELBASE) * math.tan(self.steering) * dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.x += SPEED * math.cos(self.heading) * dt
        self.y += SPEED * math.sin(self.heading) * dt

    def dist_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fare-guided Ackermann simulator")
    parser.add_argument("--vpfs", default=DEFAULT_VPFS, help="VPFS server URL")
    parser.add_argument("--team", default=DEFAULT_TEAM, type=int, help="Team number")
    args = parser.parse_args()

    x_min, x_max, y_min, y_max = load_spawn_bounds()
    padding   = min(x_max - x_min, y_max - y_min) * 0.05
    arrow_len = min(x_max - x_min, y_max - y_min) * 0.06

    # Start the vehicle in the centre of the arena
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    sim = AckermannSim(cx, cy)

    connect_to_server(args.vpfs)

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(f"VPFS Fare Simulator – Team {args.team}")

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Team {args.team} – fare-guided simulation")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.5, edgecolor="steelblue", facecolor="none",
        linestyle="--", label="Map bounds",
    ))

    # Trail and vehicle
    trail_x: list[float] = []
    trail_y: list[float] = []
    (trail_line,) = ax.plot([], [], color="orange", alpha=0.5, linewidth=2, label="Trail")
    (pos_marker,) = ax.plot([], [], "o", color="red", markersize=9, label="Vehicle", zorder=6)
    heading_arrow = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
        zorder=7,
    )

    # Fare waypoints
    (src_marker,)  = ax.plot([], [], "o", color="royalblue",  markersize=13,
                               label="Pickup",  zorder=5)
    (dest_marker,) = ax.plot([], [], "o", color="limegreen",  markersize=13,
                               label="Dropoff", zorder=5)
    src_label  = ax.text(0, 0, "Pickup",  color="royalblue", fontsize=8,
                          ha="center", va="bottom", visible=False)
    dest_label = ax.text(0, 0, "Dropoff", color="limegreen", fontsize=8,
                          ha="center", va="bottom", visible=False)

    # Status / IN POSITION texts
    fare_status_text = ax.text(
        0.02, 0.98, "Waiting for fare…",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", color="gray", fontweight="bold",
    )
    in_position_text = ax.text(
        0.5, 0.5, "IN POSITION",
        transform=ax.transAxes, fontsize=18,
        ha="center", va="center",
        color="green", fontweight="bold", visible=False,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="green", alpha=0.85),
        zorder=10,
    )

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    # ── Loop state ────────────────────────────────────────────────────────────
    dt_sim        = 1.0 / SIM_HZ
    publish_every = SIM_HZ // PUBLISH_HZ
    poll_every    = SIM_HZ          # poll API once per second
    step_count    = 0

    current_fare:    dict | None = None
    last_fare_id:    int  | None = None   # unique_id of the last known fare
    has_active_fare: bool        = False

    print(f"Team {args.team} fare simulator running. Waiting for a fare to be claimed…")

    try:
        while plt.fignum_exists(fig.number):
            t0 = time.time()

            # ── Poll fare API once per second ──────────────────────────────
            if step_count % poll_every == 0:
                fare = fetch_current_fare(args.vpfs, args.team)
                current_fare = fare

                fare_id = fare.get("unique_id") if fare else None

                if fare_id != last_fare_id:
                    last_fare_id = fare_id
                    if fare is not None:
                        sx = fare["src"]["x"]
                        sy = fare["src"]["y"]
                        dx = fare["dest"]["x"]
                        dy = fare["dest"]["y"]
                        src_marker.set_data([sx], [sy])
                        dest_marker.set_data([dx], [dy])
                        src_label.set_position((sx, sy + arrow_len * 0.6))
                        dest_label.set_position((dx, dy + arrow_len * 0.6))
                        src_label.set_visible(True)
                        dest_label.set_visible(True)
                        has_active_fare = True
                        print(f"New fare: pickup=({sx:.1f},{sy:.1f})  "
                              f"dropoff=({dx:.1f},{dy:.1f})")
                    else:
                        src_marker.set_data([], [])
                        dest_marker.set_data([], [])
                        src_label.set_visible(False)
                        dest_label.set_visible(False)
                        has_active_fare = False

                # Update status texts on every poll
                if current_fare is None:
                    fare_status_text.set_text("No active fare – waiting…")
                    fare_status_text.set_color("gray")
                    in_position_text.set_visible(False)
                else:
                    picked_up  = current_fare.get("pickedUp",  False)
                    completed  = current_fare.get("completed", False)
                    in_pos     = current_fare.get("inPosition", False)

                    if completed:
                        fare_status_text.set_text("Fare COMPLETED ✓")
                        fare_status_text.set_color("green")
                    elif picked_up:
                        fare_status_text.set_text("Picked up – heading to DROPOFF →")
                        fare_status_text.set_color("limegreen")
                    else:
                        fare_status_text.set_text("Heading to PICKUP →")
                        fare_status_text.set_color("royalblue")

                    in_position_text.set_visible(in_pos)

            # ── Determine navigation target and step ───────────────────────
            if has_active_fare and current_fare is not None:
                picked_up  = current_fare.get("pickedUp",  False)
                completed  = current_fare.get("completed", False)
                in_pos     = current_fare.get("inPosition", False)

                if completed:
                    pass  # fare done – stand still, wait for next fare
                elif in_pos:
                    pass  # at waypoint – hold position while server timer runs
                elif not picked_up:
                    sim.step_toward(current_fare["src"]["x"],
                                    current_fare["src"]["y"], dt_sim)
                else:
                    sim.step_toward(current_fare["dest"]["x"],
                                    current_fare["dest"]["y"], dt_sim)
            # No fare → vehicle stays still

            step_count += 1

            # ── Publish position ───────────────────────────────────────────
            if step_count % publish_every == 0:
                send_update({args.team: (sim.x, sim.y, 0.0, sim.heading)})

            # ── Update plot ────────────────────────────────────────────────
            trail_x.append(sim.x)
            trail_y.append(sim.y)
            if len(trail_x) > TRAIL_LEN:
                trail_x.pop(0)
                trail_y.pop(0)
            trail_line.set_data(trail_x, trail_y)

            pos_marker.set_data([sim.x], [sim.y])

            hdx = arrow_len * math.cos(sim.heading)
            hdy = arrow_len * math.sin(sim.heading)
            heading_arrow.set_position((sim.x, sim.y))
            heading_arrow.xy = (sim.x + hdx, sim.y + hdy)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - t0
            sleep_time = dt_sim - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
