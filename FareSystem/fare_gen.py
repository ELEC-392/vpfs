"""
Spawn-point–biased fare generator.

Purpose:
- Pick two distinct spawn points and create a Fare between them.
- Enforce distance bounds and avoid reusing endpoints already used by active fares.
- Bias the fare type using per-point probabilities.
- Nudge probabilities toward a target distribution so the active list stays balanced.

Returns:
- A new Fare when a valid pair is found within 10 tries.
- None if no valid pairing can be found in those attempts.
"""

from attr import dataclass  # Uses the 'attrs' library's dataclass. stdlib alternative: from dataclasses import dataclass

from utils import Point
import random

from fare import Fare
from fare_prob import FareProbability
from fare_types import FareType, get_target_distribution

from functools import lru_cache
from pathlib import Path
from typing import Dict
import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "Config" / "spawn_points.yaml"

@dataclass
class SpawnPoint:
    """
    A predefined location on the map and its fare-type bias.
    - point: the (x, y) spawn coordinate.
    - biases: a FareProbability describing likelihoods for each FareType.
    """
    point: Point
    biases: FareProbability


def _fare_type_by_name(name: str) -> FareType:
    return next((ft for ft in FareType if ft.name == name), next(iter(FareType)))


# Convenience handles for the two documented fare types
STANDARD = _fare_type_by_name("STANDARD")
SPECIAL = _fare_type_by_name("SPECIAL")

@lru_cache(maxsize=1)
def load_spawn_points_config() -> Dict[str, dict]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing spawn points config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    spawn_points = data.get("spawn_points")
    if not isinstance(spawn_points, list) or not spawn_points:
        raise ValueError("spawn_points.yaml must define a non-empty 'spawn_points' dict")

    # Read the spawn points and their biases
    normalized: Dict[str, dict] = {}
    for spawn_point in spawn_points:
        is_active = spawn_point.get("active", True)
        if not is_active:
            continue

        name = spawn_point.get("name", "unnamed")
        coordinates = spawn_point.get("coordinates", {})
        fare_type_biases = spawn_point.get("fare_type_biases", {})

        x = float(coordinates.get("x", 0.0))
        y = float(coordinates.get("y", 0.0))

        biases: Dict[FareType, float] = {}
        for fare_type_name, weight in fare_type_biases.items():
            fare_type = _fare_type_by_name(fare_type_name.upper())
            biases[fare_type] = float(weight)

        normalized[f"{name}"] = {
            "point": Point(x, y),
            "biases": FareProbability(biases),
        }

    return normalized

spawn_points_config = load_spawn_points_config()

# Candidate spawn points and their type biases based on config file
points: [SpawnPoint] = []
for cfg in spawn_points_config.values():
    points.append(SpawnPoint(cfg["point"], cfg["biases"]))

# Minimum and maximum allowed distance between spawn points
DIST_MIN = 0.5
DIST_MAX = 999

# TODO: Find a way to link this to the one in FMS.py without a circular import
TARGET_FARES = 5

targetProbabilities = get_target_distribution()


def generate_fare(existingFares: [Fare]) -> Fare or None:
    """
    Try to generate a new fare that:
    - Uses two distinct spawn points not currently occupied by active fares.
    - Falls within distance bounds [DIST_MIN, DIST_MAX].
    - Selects a FareType by merging endpoint biases and reweighting toward targets.

    :param existingFares: Current fare list (active and inactive).
    :return: A new Fare or None if generation failed after several attempts.
    """

    # May be bound later to guarantee a range of distances
    min_dist = DIST_MIN
    max_dist = DIST_MAX

    # Collect endpoints from currently active fares to avoid reuse, and
    # track live counts of each fare type to compute balancing multipliers.
    existing = []
    quantities: dict[FareType, float] = {fare_type: 0 for fare_type in FareType}
    active_fares: int = 0
    for fare in existingFares:
        if fare.isActive:
            active_fares += 1
            existing.append(fare.src)
            existing.append(fare.dest)
            quantities[fare.type] += 1

    # Try up to 10 random pairings to find a legal (distinct + in-range + unused) route.
    ovf = 0
    success = False
    while ovf < 10 and not success:
        ovf += 1
        # Pick two random points and validate the pairing
        p1: SpawnPoint = random.choice(points)
        p2: SpawnPoint = random.choice(points)

        # Distance between endpoints (instance-method call style)
        dist = Point.dist(p1.point, p2.point)

        # Need two unique points, within distance bounds, and not already in use
        if (
            p1.point == p2.point
            or dist > max_dist
            or dist < min_dist
            or p1.point in existing
            or p2.point in existing
        ):
            continue
        success = True

    # Compute per-type probability multipliers to pull the mix toward targets.
    # If a type is underrepresented, it gets >1x multiplier; overrepresented gets <1x.
    prob_mul: dict[FareType, float] = {}
    for key, value in quantities.items():
        # Current share of active fares for this type (avoid div-by-zero)
        curr_ratio = value / max(active_fares, 1)
        if curr_ratio == 0:
            curr_ratio = 0.01  # small epsilon to avoid singularity
        # Bound multiplier between 0.25x and 10x
        prob_mul[key] = min(max(targetProbabilities[key] / curr_ratio, 1 / 4), 10)

    if success:
        # Merge endpoint biases into a base probability distribution
        prob = FareProbability.merge(p1.biases, p2.biases)

        # Reweight by multipliers to approach the target composition.
        # NOTE: If 'prob' is a dict-like, this should be prob.items():
        # for key, value in prob.items():
        for key, value in prob:
            prob[key] *= prob_mul.get(key, 1)

        # Sample a FareType from the reweighted distribution
        fare_type = prob.roll()

        # Build the new Fare
        return Fare(p1.point, p2.point, fare_type)

    # No valid pairing found in allotted attempts
    return None