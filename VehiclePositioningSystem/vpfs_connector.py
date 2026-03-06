from typing import Dict, Tuple
import socketio

sock = socketio.Client()

connected = False

@sock.event
def connect():
    global connected
    connected = True
    print("Connected to VPFS")

@sock.event
def connect_error(data):
    global connected
    connected = False
    print("The connection to VPFS failed!")

@sock.event
def disconnect():
    global connected
    connected = False
    print("Disconnected from VPFS")


def connect_to_server(url: str = "http://localhost:5000"):
    """
    Connect to the VPFS backend.
    Call this explicitly from the runtime script after parsing --vpfs.
    """
    global connected
    if connected:
        return
    try:
        print(f"Connecting to VPFS at {url} ...")
        sock.connect(url)
    except Exception as e:
        print(f"Could not connect to VPFS: {e}")
        connected = False


def send_update(tagPoses: Dict[int, Tuple]):
    if not connected:
        return

    data = []
    for tag, pose in tagPoses.items():
        data.append({
            'team': tag,
            'x': pose[0],
            'y': pose[1],
            'heading': pose[3] if len(pose) > 3 else 0.0,
        })
    sock.emit("whereami_update", data)