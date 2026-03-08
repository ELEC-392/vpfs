"""
Ackermann vehicle trajectory simulator for VPFS.

Simulates Team 0's vehicle moving within the area bounded by the configured spawn
points, and publishes its position to the VPFS webserver at 1 Hz via Socket.IO.

The random seed is derived from the wall-clock time so each execution produces a
different trajectory.

Usage:
    python ackermann_simulator.py [--vpfs URL]
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

import yaml

# Allow importing vpfs_connector from this same folder
sys.path.insert(0, str(Path(__file__).parent))
from vpfs_connector import connect_to_server, send_update

# ── Team that this simulator drives ──────────────────────────────────────────
TEAM_ID = 0

# ── Simulation / publish rates ────────────────────────────────────────────────
SIM_HZ = 20       # internal kinematics steps per second
PUBLISH_HZ = 1    # position updates sent to the server per second

# ── Ackermann vehicle parameters ─────────────────────────────────────────────
WHEELBASE = 15.0           # distance between axles (same units as spawn coords)
SPEED_MIN = 15.0           # units / second
SPEED_MAX = 30.0           # units / second
MAX_STEER = math.radians(35)  # maximum steering angle


def load_spawn_bounds() -> tuple[float, float, float, float]:
    """
    Return (x_min, x_max, y_min, y_max) from Config/spawn_points.yaml.
    Falls back to the same 8-point defaults used by fare_gen.py.
    """
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
        # Mirror fare_gen.py defaults (8 evenly-spaced points)
        for i in range(1, 9):
            xs.append(float(i * 50))
            ys.append(float(i * 50))

    return min(xs), max(xs), min(ys), max(ys)


class AckermannSimulator:
    """
    Kinematic bicycle (Ackermann) model with:
    - A slow random-walk on the steering angle to produce varied trajectories.
    - Proportional boundary avoidance that biases steering toward the centre
      when the vehicle enters the perimeter margin.
    - A hard-clamp + heading reflection as a last resort if it reaches the wall.
    """

    def __init__(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        seed: float | None = None,
    ):
        rng = random.Random(seed)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        # Start well inside the boundary
        margin = min(x_max - x_min, y_max - y_min) * 0.1
        self.x = rng.uniform(x_min + margin, x_max - margin)
        self.y = rng.uniform(y_min + margin, y_max - margin)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.speed = rng.uniform(SPEED_MIN, SPEED_MAX)
        self.steering = 0.0

        self._rng = rng
        self._steer_noise = 0.0  # slow random walk component

    def step(self, dt: float) -> None:
        """Advance the vehicle state by dt seconds."""
        # Random-walk on steering noise; decay slowly so it stays bounded
        self._steer_noise += self._rng.gauss(0, math.radians(12)) * dt
        self._steer_noise *= 0.97

        # Boundary avoidance: steer toward centre when inside the margin zone
        margin = min(self.x_max - self.x_min, self.y_max - self.y_min) * 0.20
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0

        near_boundary = (
            self.x < self.x_min + margin or self.x > self.x_max - margin or
            self.y < self.y_min + margin or self.y > self.y_max - margin
        )

        if near_boundary:
            target_heading = math.atan2(cy - self.y, cx - self.x)
            diff = (target_heading - self.heading + math.pi) % (2 * math.pi) - math.pi
            avoid_steer = max(-MAX_STEER, min(MAX_STEER, diff * 0.8))
        else:
            avoid_steer = 0.0

        desired_steer = avoid_steer + self._steer_noise
        # Low-pass filter actual steering to simulate a finite steering rate
        self.steering += (desired_steer - self.steering) * min(1.0, dt * 3.0)
        self.steering = max(-MAX_STEER, min(MAX_STEER, self.steering))

        # Kinematic bicycle model
        self.heading += (self.speed / WHEELBASE) * math.tan(self.steering) * dt
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # Hard clamp + reflect heading if the vehicle escapes the boundary
        if self.x < self.x_min or self.x > self.x_max:
            self.heading = math.atan2(math.sin(self.heading), -math.cos(self.heading))
            self.x = max(self.x_min, min(self.x_max, self.x))
        if self.y < self.y_min or self.y > self.y_max:
            self.heading = math.atan2(-math.sin(self.heading), math.cos(self.heading))
            self.y = max(self.y_min, min(self.y_max, self.y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ackermann vehicle simulator for VPFS")
    parser.add_argument(
        "--vpfs", default="http://localhost:5000",
        help="VPFS server URL (default: http://localhost:5000)",
    )
    args = parser.parse_args()

    x_min, x_max, y_min, y_max = load_spawn_bounds()
    print(f"Spawn bounds: x=[{x_min}, {x_max}]  y=[{y_min}, {y_max}]")

    seed = time.time()
    sim = AckermannSimulator(x_min, x_max, y_min, y_max, seed=seed)
    print(
        f"Seed: {seed:.3f}  |  "
        f"Start: ({sim.x:.2f}, {sim.y:.2f}), "
        f"heading={math.degrees(sim.heading):.1f}°, "
        f"speed={sim.speed:.1f} u/s"
    )

    connect_to_server(args.vpfs)

    dt_sim = 1.0 / SIM_HZ
    publish_every = SIM_HZ // PUBLISH_HZ  # sim steps between each publish
    step_count = 0

    print(f"Simulating Team {TEAM_ID} at {SIM_HZ} Hz, publishing at {PUBLISH_HZ} Hz ...")
    while True:
        t0 = time.time()
        sim.step(dt_sim)
        step_count += 1

        if step_count % publish_every == 0:
            # Tuple layout expected by send_update: (x, y, _, heading)
            send_update({TEAM_ID: (sim.x, sim.y, 0.0, sim.heading)})
            print(
                f"[{time.strftime('%H:%M:%S')}] Team {TEAM_ID}: "
                f"x={sim.x:8.3f}  y={sim.y:8.3f}  "
                f"hdg={math.degrees(sim.heading):+7.1f}°  "
                f"steer={math.degrees(sim.steering):+6.1f}°"
            )

        elapsed = time.time() - t0
        sleep_time = dt_sim - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
