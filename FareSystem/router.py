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

from flask import Flask, jsonify, request, render_template, send_from_directory
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
# Configure template and static folders for map monitor
app = Flask(__name__,
            template_folder='../Dashboard/templates',
            static_folder='../Dashboard/static')
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

@app.route("/dashboard")
def serve_map_monitor():
    """Serve the map monitor interface."""
    return render_template('dashboard.html')

@app.route("/admin")
def serve_admin():
    """Serve the admin panel interface."""
    return render_template('admin.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets (map and duck images)."""
    assets_path = Path(__file__).parent.parent / 'Assets'
    return send_from_directory(assets_path, filename)

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

@app.route("/dashboard/teams")
def serve_teams():
    """
    Returns a list of teams with money, rep, current fare, and last update times.
    Intended for dashboard/monitoring use.
    """
    data = []
    with fms.mutex:
        for team in fms.teams.values():
            data.append({
                "number": team.number,
                "money": team.money,
                "rep": team.karma,
                "currentFare": team.currentFare,
                "position": {
                    "x": team.pos.x,
                    "y": team.pos.y
                },
                "lastPosUpdate": team.lastPosUpdate,
                "lastStatus": team.lastStatus
            })
    return jsonify(data)

@app.route("/api/map/teams")
def serve_map_teams():
    """
    Returns team configuration for map monitor.
    Maps team numbers to duck colors.
    """
    # Duck color mapping (7 teams max)
    duck_colors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png"]
    
    teams_data = []
    with fms.mutex:
        for idx, team in enumerate(sorted(fms.teams.values(), key=lambda t: t.number)):
            teams_data.append({
                "id": team.number,
                "name": getattr(team, 'name', f"Team {team.number}"),
                "duck": duck_colors[idx % len(duck_colors)]
            })
    return jsonify(teams_data)

@app.route("/api/map/positions")
def serve_map_positions():
    """
    Returns current normalized positions (0-1 range) for all teams.
    """
    positions = {}
    with fms.mutex:
        for team in fms.teams.values():
            # Positions are already normalized in the system
            positions[team.number] = {
                "x": team.pos.x,
                "y": team.pos.y
            }
    return jsonify(positions)

@app.route("/api/admin/team-names")
def serve_team_names():
    """
    Returns available team names from YAML file for autocomplete.
    """
    try:
        yaml_path = Path(__file__).parent.parent / 'Config' / 'team_names.yaml'
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return jsonify(data.get('teams', []))
    except Exception as e:
        print(f"Error loading team names: {e}")
        return jsonify([])

@app.route("/api/admin/current-teams")
def serve_current_teams():
    """
    Returns current teams with their names for admin panel.
    """
    teams_data = []
    with fms.mutex:
        for team in sorted(fms.teams.values(), key=lambda t: t.number):
            teams_data.append({
                "number": team.number,
                "name": getattr(team, 'name', f"Team {team.number}"),
                "money": team.money,
                "rep": team.karma
            })
    return jsonify(teams_data)

@app.route("/api/admin/configure-teams", methods=["POST"])
def configure_teams():
    """
    Configure teams for the match.
    Expects JSON: {"teams": [{"number": int, "name": str}, ...]}
    """
    if MODE != OperatingMode.LAB:
        return jsonify({"success": False, "message": "Team configuration only allowed in LAB mode"}), 403
    
    try:
        data = request.get_json()
        teams = data.get('teams', [])
        
        if not teams:
            return jsonify({"success": False, "message": "No teams provided"}), 400
        
        with fms.mutex:
            # Clear existing teams
            fms.teams.clear()
            
            # Add new teams
            for team_data in teams:
                team_number = team_data['number']
                team_name = team_data['name']
                
                team = Team(team_number)
                team.name = team_name  # Add name attribute
                fms.teams[team_number] = team
        
        # Broadcast update to map monitor clients
        duck_colors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png"]
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
        
        sock.emit('initial_state', {
            'teams': teams_data,
            'positions': positions
        })
        
        return jsonify({
            "success": True,
            "message": f"Successfully configured {len(teams)} team(s)",
            "teams": len(teams)
        })
    
    except Exception as e:
        print(f"Error configuring teams: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/clear-teams", methods=["POST"])
def clear_teams():
    """
    Clear all teams from the match.
    """
    if MODE != OperatingMode.LAB:
        return jsonify({"success": False, "message": "Team clearing only allowed in LAB mode"}), 403
    
    try:
        with fms.mutex:
            fms.teams.clear()
        
        # Broadcast update to map monitor
        sock.emit('initial_state', {
            'teams': [],
            'positions': {}
        })
        
        return jsonify({"success": True, "message": "All teams cleared"})
    except Exception as e:
        print(f"Error clearing teams: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/remove-team/<int:team_number>", methods=["DELETE"])
def remove_team(team_number):
    """
    Remove a specific team from the match.
    """
    if MODE != OperatingMode.LAB:
        return jsonify({"success": False, "message": "Team removal only allowed in LAB mode"}), 403
    
    try:
        with fms.mutex:
            if team_number not in fms.teams:
                return jsonify({"success": False, "message": f"Team {team_number} not found"}), 404
            
            del fms.teams[team_number]
        
        # Broadcast update to map monitor
        duck_colors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png"]
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
        
        sock.emit('initial_state', {
            'teams': teams_data,
            'positions': positions
        })
        
        return jsonify({"success": True, "message": f"Team {team_number} removed"})
    except Exception as e:
        print(f"Error removing team: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

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

@app.route("/dashboard/fares")
def serve_fares_dashboard():
    """Dashboard-oriented fare list (extended info, includes expired)."""
    return serve_fares(True, True)

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
        message = ""
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
                # NOTE: likely intended to use idx, not id (built-in). Left unchanged.
                message = f"Could not find fare with ID {id}"
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
    """
    with fms.mutex:
        fare_dict = None
        message = ""
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
