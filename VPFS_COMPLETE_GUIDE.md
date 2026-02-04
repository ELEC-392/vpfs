# VPFS System - Complete Guide

## 🎯 System Overview

The Vehicle Positioning and Fare System (VPFS) now includes:
1. **Map Monitor** - Real-time visualization of team positions
2. **Admin Panel** - Team configuration interface
3. **FareSystem** - Core game logic and APIs

## 🌐 Available Interfaces

### 1. Home Page
**URL:** http://localhost:5000/
- Health check endpoint
- System status

### 2. Map Monitor
**URL:** http://localhost:5000/map
- Real-time team position visualization
- Interactive duck avatars with team names
- Legend with coordinates
- Auto-updates when teams move

### 3. Admin Panel
**URL:** http://localhost:5000/admin
- Configure teams (1-7 teams)
- Assign custom team names with autocomplete
- Manage active teams
- View system status

## 📋 Quick Start

### Step 1: Start the Server
```bash
cd FareSystem
python router.py
```

### Step 2: Configure Teams
1. Open http://localhost:5000/admin
2. Click the number of teams (1-7)
3. Enter team names (autocomplete available)
4. Click "💾 Save Configuration"

### Step 3: View Map
1. Open http://localhost:5000/map
2. Teams appear with their duck avatars
3. Watch real-time position updates

### Step 4: Send Position Updates
Use the VehiclePositioningSystem or test script:
```bash
python test_positions.py
```

## 🦆 Team Assignment

Teams are automatically assigned:
- **Team Numbers:** 3, 5, 7, 9, 11, 13, 15
- **Duck Colors:** Blue, Red, Green, Yellow, Purple, Brown, Grey

## 📝 Team Names Database

Edit `Config/team_names.yaml` to customize autocomplete options:

```yaml
teams:
  - Team Alpha
  - Speed Demons
  - Your Custom Name
  # Add more here
```

## 🔧 API Reference

### Map Monitor APIs
- `GET /map` - Map interface
- `GET /api/map/teams` - Team configuration
- `GET /api/map/positions` - Current positions

### Admin APIs
- `GET /admin` - Admin interface
- `GET /api/admin/team-names` - Autocomplete names
- `GET /api/admin/current-teams` - Active teams
- `POST /api/admin/configure-teams` - Save teams
- `POST /api/admin/clear-teams` - Remove all
- `DELETE /api/admin/remove-team/<num>` - Remove one

### Position Updates (Socket.IO)
- Event: `whereami_update`
- Payload: `[{team: int, x: float, y: float}, ...]`

## 🎨 Features

### Map Monitor
- ✨ Real-time position updates via WebSocket
- 🦆 Animated duck movements (0.8s smooth transitions)
- 📍 Normalized coordinates (0-1 range)
- 🖱️ Hover to see team names
- 👆 Click to highlight teams
- 📊 Live statistics and connection status

### Admin Panel
- 🔢 Select 1-7 teams
- 🎯 Auto-assigned team numbers and ducks
- 💬 Autocomplete team names from YAML
- 💾 Instant save and broadcast to map
- 🗑️ Individual or bulk team removal
- 📋 Active teams sidebar

## 🔄 Workflow Example

```bash
# 1. Start server
cd FareSystem && python router.py

# 2. Configure teams (in browser)
# Open: http://localhost:5000/admin
# Select: 4 teams
# Enter names: Team Alpha, Speed Demons, etc.
# Click: Save Configuration

# 3. View map (in another tab)
# Open: http://localhost:5000/map
# See: 4 ducks appear on map

# 4. Send positions
python test_positions.py
# Watch: Ducks move on map in real-time
```

## 🎮 Integration Points

### From VehiclePositioningSystem
Send positions via Socket.IO:
```python
import socketio
sio = socketio.Client()
sio.connect('http://localhost:5000')
sio.emit('whereami_update', [
    {"team": 3, "x": 0.5, "y": 0.5}
])
```

### From External Systems
HTTP endpoints available for:
- Team configuration
- Position queries
- Status checks

## 🛡️ Security Notes

- Admin functions require LAB mode
- Position updates validated with JSON schema
- Team numbers automatically managed
- Concurrent access protected with mutexes

## 📁 File Structure

```
FareSystem/
├── router.py              # Main server (integrated)
├── fms.py                 # Field Management System
├── team.py                # Team class
└── ...

VehiclePositioningSystem/
├── templates/
│   ├── map_monitor.html   # Map interface
│   └── admin.html         # Admin interface
├── static/
│   ├── map_monitor.css
│   ├── map_monitor.js
│   ├── admin.css
│   └── admin.js
└── test_positions.py      # Test script

Config/
├── team_names.yaml        # Team name database
└── ...

Assets/
├── Map_Clean.png          # Map image
└── ducks/                 # Duck avatars
    ├── Blue.png
    ├── Red.png
    └── ...
```

## 🚀 Advanced Usage

### Custom Team Numbers
Teams use odd numbers starting from 3. To change:
```javascript
// In admin.js, modify generateTeamCards():
const teamNumber = (i + 1) * 2 + 1; // Current: 3,5,7,9,11,13,15
```

### Add More Duck Colors
1. Add PNG files to `Assets/ducks/`
2. Update color arrays in:
   - `router.py` (duck_colors)
   - `admin.js` (duckColors)

### Custom Map
Replace `Assets/Map_Clean.png` with your map image.
Positions automatically scale to any image size.

## 🐛 Troubleshooting

### Teams not appearing on map?
1. Check admin panel - are teams configured?
2. Verify positions sent via Socket.IO
3. Check browser console for errors

### Autocomplete not working?
1. Verify `Config/team_names.yaml` exists
2. Check `/api/admin/team-names` returns data
3. Clear browser cache

### Port already in use?
```bash
lsof -ti:5000 | xargs kill -9
```

## 📚 Additional Documentation

- `INTEGRATION_SUMMARY.md` - Initial map monitor setup
- `ADMIN_GUIDE.md` - Admin panel details
- `MAP_MONITOR_README.md` - Map monitor specifics
