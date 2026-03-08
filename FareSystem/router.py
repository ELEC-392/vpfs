"""
HTTP and WebSocket API for VPFS.

- Exposes REST endpoints for status, teams, fares, and position queries.
- Accepts real-time position updates via Socket.IO.
- Uses a shared match state in `fms` guarded by `fms.mutex`.
- Authenticates teams based on operating mode via `auth.authenticate`.
"""

import time
from pathlib import Path
import yaml

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from jsonschema.exceptions import ValidationError

from utils import Point
import fms
from jsonschema import validate
from threading import Thread
from auth import authenticate
from params import MODE, OperatingMode
from team import Team

# Create Flask app and Socket.IO wrapper
app = Flask(__name__)
sock = SocketIO(app)

# Optionally initialize lab-specific modules
app.app_context().push()
if MODE == OperatingMode.LAB:
    # Lab-only support code (import side effects, config, etc.)
    import lab_tms
    lab_tms.IDK = ""

@app.route("/")
def serve_root():
    """Health check endpoint."""
    return "VPFS is alive!\n"


@app.route("/match")
def serve_status():
    """
    Returns match/server status for the authenticated team.
    Query params:
      - auth: team code or team number depending on mode
    """
    auth_code = request.args.get("auth")  # None if missing
    team = authenticate(auth_code, MODE) if auth_code else -1

    # Update last poll time if team exists
    if team in fms.teams:
        fms.teams[team].lastStatus = time.time()

    # Return snapshot of match state (with lock)
    with fms.mutex:
        return jsonify({
            "mode": MODE.value,
            "match": fms.matchNum,
            "matchStart": fms.matchRunning,
            "timeRemain": fms.matchEndTime - time.time(),
            "inMatch": team in fms.teams,
            "team": team,
        })


def serve_fares(extended: bool, include_expired: bool):
    """
    Helper to serialize fares.
    - extended: include internal fields for dashboard or current-fare views.
    - include_expired: include fares that are no longer active.
    """
    data = []
    with fms.mutex:
        for idx, fare in enumerate(fms.fares):
            if fare.isActive or include_expired:
                data.append(fare.to_json_dict(idx, extended))
        return jsonify(data)


@app.route("/fares")
def serve_fares_normal():
    """
    Client-visible fare list.
    Query params:
      - all=true|false to include expired fares.
    """
    return serve_fares(
        False,
        request.args.get("all", default=False, type=lambda st: st.lower() == "true"),
    )


@app.route("/fares/claim/<int:idx>")
def claim_fare(idx: int):
    """
    Claims a fare for the authenticated team.
    - Path: idx is the fare index.
    - Query: auth carries code/team depending on mode.
    """
    team = authenticate(request.args.get("auth", default=""), MODE)
    with fms.mutex:
        success = False
        message = f"Team {team} has successfully claimed fare {idx}"
        if team == -1:
            message = "Authentication failed"
        elif team in fms.teams.keys():
            if idx < len(fms.fares):
                err = fms.fares[idx].claim_fare(idx, fms.teams[team])
                if err is None:
                    success = True
                else:
                    message = err
            else:
                message = f"Could not find fare with ID {idx}"
        else:
            message = f"Team {team} not in this match"

        return jsonify({
            "success": success,
            "message": message
        })


@app.route("/fares/drop/<int:idx>")
def drop_fare(idx: int):
    """
    Drops a previously claimed fare, returning it to the pool.
    - Path: idx is the fare index.
    - Query: auth carries code/team depending on mode.
    Only the team that claimed the fare may drop it, and only before pickup.
    """
    team = authenticate(request.args.get("auth", default=""), MODE)
    with fms.mutex:
        success = False
        message = f"Team {team} has successfully dropped fare {idx}"
        if team == -1:
            message = "Authentication failed"
        elif team in fms.teams.keys():
            if idx < len(fms.fares):
                err = fms.fares[idx].drop_fare(idx, fms.teams[team])
                if err is None:
                    success = True
                else:
                    message = err
            else:
                message = f"Could not find fare with ID {idx}"
        else:
            message = f"Team {team} not in this match"

        return jsonify({
            "success": success,
            "message": message
        })


@app.route("/fares/current/<int:team>")
def current_fare(team: int):
    """
    Returns the currently assigned fare (with extended info) for a team number.
    Requires authentication - teams can only query their own current fare.
    Query params:
      - auth: team code or team number depending on mode
    """
    authenticated_team = authenticate(request.args.get("auth", default=""), MODE)
    if authenticated_team == -1:
        return jsonify({"fare": None, "message": "Authentication failed"}), 401
    if authenticated_team != team:
        return jsonify({"fare": None, "message": f"Access denied: Team {authenticated_team} cannot view Team {team}'s fare"}), 403

    with fms.mutex:
        fare_dict = None
        message = f"Team {team} has successfully retrieved fare information"
        if team in fms.teams.keys():
            fare_idx = fms.teams[team].currentFare
            if fare_idx is None:
                message = f"Team {team} does not have an active fare."
            else:
                fare = fms.fares[fare_idx]
                fare_dict = fare.to_json_dict(fare_idx, True)
        else:
            message = f"Team {team} not in this match."

        return jsonify({
            "fare": fare_dict,
            "message": message
        })


@app.route("/teams/status/<int:team>")
def team_status(team: int):
    """
    Returns the current status (money and reputation) for the authenticated team.
    Query params:
      - auth: team code or team number depending on mode
    """
    authenticated_team = authenticate(request.args.get("auth", default=""), MODE)
    if authenticated_team == -1:
        return jsonify({"status": None, "message": "Authentication failed"}), 401
    if authenticated_team != team:
        return jsonify({"status": None, "message": f"Access denied: Team {authenticated_team} cannot view Team {team}'s status"}), 403

    with fms.mutex:
        if team not in fms.teams:
            return jsonify({"status": None, "message": f"Team {team} not in this match"}), 404
        t = fms.teams[team]
        return jsonify({
            "status": {
                "team": team,
                "money": t.money,
                "reputation": t.karma,
            },
            "message": ""
        })


@app.route("/whereami/<int:team>")
def whereami_get(team: int):
    """
    Gets the last known position for the authenticated team.
    Returns x/y coordinates and the last update timestamp.
    Requires authentication - teams can only query their own position.
    Query params:
      - auth: team code or team number depending on mode
    """
    # Authenticate the request
    auth_code = request.args.get("auth", default="")
    authenticated_team = authenticate(auth_code, MODE)
    
    # Check if authentication succeeded
    if authenticated_team == -1:
        return jsonify({
            "position": None,
            "last_update": 0,
            "message": "Authentication failed"
        }), 401
    
    # Check if authenticated team matches requested team
    if authenticated_team != team:
        return jsonify({
            "position": None,
            "last_update": 0,
            "message": f"Access denied: Team {authenticated_team} cannot view Team {team}'s position"
        }), 403
    
    point = None
    last_update: int = 0
    message = ""
    if team in fms.teams.keys():
        team_obj = fms.teams[team]
        point = {
            "x": team_obj.pos.x,
            "y": team_obj.pos.y,
            "heading": team_obj.heading,
        }
        last_update = team_obj.lastPosUpdate
    else:
        message = f"Team {team} not in this match"

    return jsonify({
        "position": point,
        "last_update": last_update,
        "message": message
    })


# Socket.IO endpoints
@sock.on("connect")
def sock_connect(auth):
    """Logs when a Socket.IO client connects."""
    print("Connected")
    
    # Send initial state to map monitor clients
    duck_colors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png"]
    
    # Gather data under mutex, then emit outside
    with fms.mutex:
        teams_data = []
        positions = {}
        for idx, team in enumerate(sorted(fms.teams.values(), key=lambda t: t.number)):
            teams_data.append({
                "id": team.number,
                "name": getattr(team, 'name', f"Team {team.number}"),
                "duck": duck_colors[idx % len(duck_colors)]
            })
            positions[team.number] = {
                "x": team.pos.x,
                "y": team.pos.y
            }
    
    # Emit outside the mutex to avoid blocking
    emit('initial_state', {
        'teams': teams_data,
        'positions': positions
    })


@sock.on("disconnect")
def sock_disconnect():
    """Logs when a Socket.IO client disconnects."""
    print("Disconnected")


# JSON schema for batched position updates: [{team:int, x:float, y:float}, ...]
whereami_update_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "team": {"type": "number"},
            "x": {"type": "number"},
            "y": {"type": "number"},
            "heading": {"type": "number"},
        },
        "required": ["team", "x", "y"],
    },
}


@sock.on("whereami_update")
def whereami_update(json):
    """
    Receives batched position updates over Socket.IO.
    Payload: list of {team, x, y, heading?} objects. Validated by JSON Schema.
    """
    # Log sending address (consider whitelisting in production)
    print(f"Recv whereami update from {request.remote_addr}")

    try:
        validate(json, schema=whereami_update_schema)
        for entry in json:
            team = entry['team']
            x = entry['x']
            y = entry['y']
            heading = entry.get('heading', 0.0)
            with fms.mutex:
                # Auto-register team on first position update
                if team not in fms.teams:
                    fms.teams[team] = Team(team)
                    print(f"Auto-registered team {team}")
                fms.teams[team].update_position(Point(x, y), heading)
            # Broadcast position update to map monitor clients
            sock.emit('position_update', {
                'team_id': team,
                'x': x,
                'y': y,
                'heading': heading,
            })
    except ValidationError as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    # Start background periodic task that advances match/fare state
    Thread(target=fms.periodic, daemon=True).start()
    # Start HTTP + Socket.IO server; bind to all interfaces
    sock.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)
