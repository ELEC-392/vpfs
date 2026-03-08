"""Authentication helpers for VPFS.

Resolves a team number from a provided code based on operating mode:
- match: validate the code against an internal mapping (_authCodes).
- lab/home: interpret the input as a team number string.
Returns -1 on failure.
"""

from params import OperatingMode
from typing import Union

_authCodes: dict[str : int] = {
    "asdf" : 0
}

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
