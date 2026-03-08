# Vehicle Positioning and Fare System (VPFS) — Playground Branch

This is the **playground** branch of the VPFS, intended for students to explore and test the fare system and vehicle positioning interface. It includes a simulated Ackermann vehicle and a live position plotter so you can experiment with the full pipeline without any physical hardware.

---

## 1. Clone the Repository

Clone the `playground` branch directly:

```bash
git clone --branch playground git@github.com:ELEC-392/vpfs.git
cd vpfs
```

> The command above assumes your account has the SSH key properly set.
---

## 2. Set Up a Python Virtual Environment

This project requires **Python 3.12**. However, during the competition you will be accessing the Web API and `curl` commands.

### Installing Python 3.12

#### Windows
Download the installer from [python.org/downloads](https://www.python.org/downloads/) and run it. Make sure to check **"Add Python to PATH"** during installation.

#### macOS
Using [Homebrew](https://brew.sh):
```bash
brew install python@3.12
```

Or download the macOS installer from [python.org/downloads](https://www.python.org/downloads/).

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

#### Verify your installation
```bash
python3.12 --version   # Linux/macOS
py -3.12 --version     # Windows
```

---

### Creating the Virtual Environment

#### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (cmd)

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

When the environment is active you will see `(.venv)` at the start of your prompt.

---

## 3. Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

---

## 4. Start the VPFS Server

Open a terminal, activate the venv, and run:

```bash
python FareSystem/router.py
```

To verify the server is running, open a second terminal and run:

```bash
curl localhost:5000
```

You should see:

```
VPFS is alive!
```

The server pre-registers **Team 0** and immediately starts generating fares. It runs until you press `Ctrl+C`.

---


## 5. Simulating and Debugging Vehicle Position

Two scripts in `VehiclePositioningSystem/` let you test position reporting without a physical vehicle.

### Ackermann Simulator

Simulates a vehicle following a random Ackermann trajectory and publishes its position to the server at 1 Hz via Socket.IO:

```bash
python VehiclePositioningSystem/ackermann_simulator.py [--vpfs http://localhost:5000]
```

Each run produces a different trajectory (random seed from the system clock).

### Position Plotter

Polls the `/whereami/<team>` endpoint and plots the vehicle's live position and heading as an arrow in a matplotlib window:

```bash
python VehiclePositioningSystem/ackermann_simulator_plotter.py [--vpfs http://localhost:5000] [--team 0] [--hz 1]
```

Run both scripts at the same time to watch the simulated vehicle move in real time.

---

## 6. Simulating the Full Fare Pipeline

The fare simulator drives Team 0 autonomously through a complete fare: it polls `/fares/current/0` and navigates from the pickup point to the dropoff point automatically.

```bash
python VehiclePositioningSystem/fare_simulator.py [--vpfs http://localhost:5000] [--team 0]
```

**Workflow:**

1. Start the server (`router.py`) in one terminal.
2. Start the fare simulator in a second terminal.
3. Use the API (e.g. `curl`) or your own vehicle code to **claim a fare**:
   ```bash
   curl "localhost:5000/fares/claim/0?auth=0"
   ```
4. The simulator detects the claimed fare, plots the pickup (blue dot) and dropoff (green dot), and drives toward the pickup point.
5. When it arrives, the plot shows an **IN POSITION** banner. The server starts a brief timer.
6. Once picked up, the vehicle drives to the dropoff. **IN POSITION** appears again while the dropoff timer runs.
7. When the fare is complete, the vehicle stops and waits for the next fare to be claimed.
8. Check your balance and reputation:
   ```bash
   curl "localhost:5000/teams/status/0?auth=0"
   ```

---

## API Reference (summary)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/match?auth=` | Match and server status |
| GET | `/fares` | List of active fares |
| GET | `/fares/claim/<idx>?auth=` | Claim a fare |
| GET | `/fares/drop/<idx>?auth=` | Drop a claimed fare (before pickup) |
| GET | `/fares/current/<team>?auth=` | Current fare for a team |
| GET | `/whereami/<team>?auth=` | Last known position for a team |
| GET | `/teams/status/<team>?auth=` | Team money and reputation |

For all authenticated endpoints, pass `auth=<team_number>` as the query parameter (e.g. `auth=0` for Team 0 in LAB mode. Individual authentication codes will be provided before the competition).
