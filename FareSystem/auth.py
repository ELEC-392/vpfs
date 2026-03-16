"""Authentication helpers for VPFS.

Resolves a team number from a provided code based on operating mode:
- match: validate the code against an internal mapping (_authCodes).
- lab/home: interpret the input as a team number string.
Returns -1 on failure.
"""

from pathlib import Path
import yaml

from params import OperatingMode
from typing import Union

# _authCodes: dict[str : int] = {
#     "asdf" : 7
# }

def _load_auth_codes() -> dict[str, int]:
    config_path = Path(__file__).resolve().parents[1] / "Config" / "teams.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return {info["auth"]: team_id for team_id, info in data["teams"].items()}

_authCodes: dict[str, int] = _load_auth_codes()

def authenticate(code: str, mode: Union[OperatingMode, str]) -> int:
    """
    Get the team corresponding to a given authentication code
    :param code: Provided authentication code
    :param mode: Server operating mode
    :returns: Corresponding team number, or -1 if not found
    """
    is_match = (
            isinstance(mode, OperatingMode) and mode is OperatingMode.MATCH
        ) or (isinstance(mode, str) and mode.lower() == "match")

    # For match mode, check against auth code dict
    if is_match:
        return _authCodes.get(code, -1)
    else:
        try:
            return int(code)
        except ValueError as e:
            print(f"Expected team number, not code {code}")
            return -1
