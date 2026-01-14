"""
Fare type configuration loader.

- Reads fare types from Config/fare_types.yaml.
- Exposes the FareType enum plus helpers to retrieve per-type attributes.
"""
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "Config" / "fare_types.yaml"


@lru_cache(maxsize=1)
def load_fare_type_config() -> Dict[str, dict]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing fare type config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    fare_types = data.get("fare_types")
    if not isinstance(fare_types, dict) or not fare_types:
        raise ValueError("fare_types.yaml must define a non-empty 'fare_types' mapping")

    normalized: Dict[str, dict] = {}
    for name, cfg in fare_types.items():
        upper = str(name).upper()
        normalized[upper] = {
            "base_fare": float(cfg.get("base_fare", 0.0)),
            "distance_fare": float(cfg.get("distance_fare", 0.0)),
            "reputation": float(cfg.get("reputation", 0.0)),
            "load_time_multiplier": float(cfg.get("load_time_multiplier", 1.0)),
            "target_ratio": float(cfg.get("target_ratio", 0.0)),
        }

    return normalized


fare_type_config = load_fare_type_config()

FareType = Enum("FareType", {name: idx for idx, name in enumerate(fare_type_config.keys())})


def get_base_fare(fare_type: FareType) -> float:
    return fare_type_config[fare_type.name]["base_fare"]


def get_distance_fare(fare_type: FareType) -> float:
    return fare_type_config[fare_type.name]["distance_fare"]


def get_reputation(fare_type: FareType) -> float:
    return fare_type_config[fare_type.name]["reputation"]


def get_load_time_multiplier(fare_type: FareType) -> float:
    return fare_type_config[fare_type.name].get("load_time_multiplier", 1.0)


def get_target_distribution() -> dict[FareType, float]:
    raw = {FareType[name]: cfg.get("target_ratio", 0.0) for name, cfg in fare_type_config.items()}
    total = sum(raw.values())
    if total <= 0:
        # Fallback to uniform if config does not specify ratios
        return {ft: 1.0 / len(FareType) for ft in FareType}
    return {ft: val / total for ft, val in raw.items()}
