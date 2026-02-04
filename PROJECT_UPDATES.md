# VPFS Project Updates Summary

## Session Date: January 26, 2026

---

## 1. Dashboard Reorganization

### New Folder Structure
- **Created**: `Dashboard/` folder at project root
- **Moved from** `VehiclePositioningSystem/`:
  - `static/` → `Dashboard/static/`
  - `templates/` → `Dashboard/templates/`

### Files Renamed
- `map_monitor.html` → `dashboard.html`
- `map_monitor.css` → `dashboard.css`
- `map_monitor.js` → `dashboard.js`

### Updated Path References
- **`FareSystem/router.py`**: Updated Flask template and static folder paths
  - Changed from `../VehiclePositioningSystem/templates` to `../Dashboard/templates`
  - Changed from `../VehiclePositioningSystem/static` to `../Dashboard/static`
- **`VehiclePositioningSystem/map_monitor.py`**: Updated to point to Dashboard folder
- **Route changed**: `/map` → `/dashboard`

---

## 2. Dependencies Management

### Updated `requirements.txt`
- **Merged** packages from `VehiclePositioningSystem/requirements.txt`
- **Added packages**:
  - `flask-cors==5.0.0`
  - `pupil-apriltags==1.0.4.post11`
- **Reorganized**: Alphabetically sorted all packages

---

## 3. Match Reproducibility & Fare Uniqueness

### Random Seed Implementation
**Modified**: `FareSystem/fms.py`
- Added `random.seed(matchNum)` in `start_match()` function
- Ensures all teams playing the same match number get identical fares
- Enables fair comparison across different time slots

### Unique Fare ID System
**Modified**: `FareSystem/fare.py`
- Added `unique_id` field to Fare class
- Formula: `unique_id = match_num * 1000 + sequence`
- Allows up to 999 fares per match with globally unique IDs
- Added `match_num` and `sequence` parameters to `__init__`

**Modified**: `FareSystem/fms.py`
- Added `fareSequence` global counter
- Resets to 0 when starting a new match
- Increments with each generated fare

**Modified**: `FareSystem/fare_gen.py`
- Updated `generate_fare()` to accept `match_num` and `sequence` parameters
- Passes these to Fare constructor

**Modified**: `FareSystem/fare.py` - `to_json_dict()`
- Added `unique_id` field to JSON response
- Kept `id` for backwards compatibility

---

## 4. Team Position Security

### Authentication for Position Queries
**Modified**: `FareSystem/router.py` - `/whereami/<team>` endpoint
- **Added**: Required authentication via `?auth=<code>` query parameter
- **Validation**: Teams can only query their own positions
- **HTTP Status Codes**: 
  - 401 for authentication failure
  - 403 for unauthorized access to other team's position

### How Authentication Works
- **LAB/HOME Mode**: Auth code = team number as string (e.g., `auth=1`)
- **MATCH Mode**: Auth code = secret string mapped in `FareSystem/auth.py`
- Uses existing `authenticate()` function for consistency

---

## 5. Enhanced Dashboard UI

### New Features Added

#### Open Fares Panel
- **Displays**: All active fares (claimed and unclaimed)
- **Information shown**:
  - Fare unique ID
  - Fare type (STANDARD/SPECIAL) with color coding
  - Distance in units
  - Countdown timer to expiry (MM:SS format)
- **Visual indicators**:
  - Yellow background for STANDARD fares
  - Blue background for SPECIAL fares
  - Gray dimmed appearance for claimed fares (with 🔒 icon)
  - Blinking red timer when < 30 seconds remain

#### Team Active Fare Display
- **Added**: Active fare indicator below team position
- **Shows**: "Active Fare: #[unique_id]" when team has claimed a fare
- **Auto-updates**: Every 2 seconds

#### Statistics Panel
- **Enhanced**: Added "Active Fares" counter
- **Tracks**: Total number of active fares in the system

### Layout Improvements

#### Reorganized Dashboard Layout
- **Left Column** (flexible width):
  - Map display (top)
  - Teams panel (bottom, horizontal grid layout)
- **Right Sidebar** (340px fixed):
  - Open Fares panel (scrollable)
  - Statistics panel

#### Teams Panel
- Changed from vertical list to **horizontal grid**
- Optimized for displaying up to 6 teams
- More compact with smaller fonts
- Responsive grid: `grid-template-columns: repeat(auto-fit, minmax(180px, 1fr))`

#### Styling Updates
- Added custom scrollbar styling for fares list
- Improved spacing and padding throughout
- Better visual hierarchy with color-coded sections
- Responsive design for screens < 1200px width

### Technical Improvements

#### JavaScript Enhancements (`dashboard.js`)
- **Auto-refresh**: Fares and team data update every 2 seconds
- **Countdown timers**: Update every second for real-time accuracy
- **Fare type mapping**: Properly handles enum values (0=STANDARD, 1=SPECIAL)
- **Error handling**: Added try-catch blocks and null checks for robustness
- **Debug logging**: Comprehensive console logging for troubleshooting

#### Bug Fixes
- Fixed fare type display (was showing "0" or "1", now shows "STANDARD" or "SPECIAL")
- Fixed TypeError when `modifiers` field is a number instead of string
- Fixed distance calculation with null/undefined checks
- Improved expiry time calculation

---

## 6. API Endpoints Summary

### Client-Facing Endpoints
- **GET** `/` - Health check
- **GET** `/dashboard` - Dashboard web interface
- **GET** `/admin` - Admin panel interface
- **GET** `/fares` - List of available fares (active only)
- **GET** `/fares?all=true` - Include expired fares
- **GET** `/whereami/<team>?auth=<code>` - Get team position (authenticated)
- **GET** `/fares/claim/<idx>?auth=<code>` - Claim a fare
- **GET** `/match?auth=<code>` - Get match status

### Dashboard/Admin Endpoints
- **GET** `/dashboard/teams` - Team data with positions and active fares
- **GET** `/dashboard/fares` - All fares (extended info, includes expired)
- **GET** `/api/map/teams` - Team configuration for map display
- **GET** `/api/map/positions` - Current team positions
- **GET** `/api/admin/team-names` - Available team names from YAML
- **GET** `/api/admin/current-teams` - Current teams with details
- **POST** `/api/admin/configure-teams` - Configure teams for match
- **POST** `/api/admin/clear-teams` - Clear all teams
- **DELETE** `/api/admin/remove-team/<team_number>` - Remove specific team

### WebSocket Events
- **connect** - Client connects, receives initial state
- **disconnect** - Client disconnects
- **whereami_update** - Receive position updates (batch)
- **position_update** - Broadcast position changes to clients
- **initial_state** - Send teams and positions to new clients

---

## 7. Database Considerations

### Unique Identifiers for Storage
- **Fare unique_id**: Globally unique across all matches
- **Match number**: Part of fare ID formula
- **Team numbers**: Used as primary identifiers

### Suggested Database Schema
```sql
CREATE TABLE fare_completions (
    fare_unique_id INTEGER,
    match_number INTEGER,
    team_number INTEGER,
    completion_time TIMESTAMP,
    money_earned REAL,
    reputation_earned REAL,
    PRIMARY KEY (fare_unique_id, team_number)
);
```

### Reproducibility Guarantees
- Same match number = identical fares in same order
- Unique IDs enable tracking specific fare completions
- Can compare team performance across different sessions

---

## 8. File Structure Summary

### New Files/Folders
```
Dashboard/
├── static/
│   ├── dashboard.css
│   └── dashboard.js
└── templates/
    └── dashboard.html
```

### Modified Files
```
FareSystem/
├── auth.py (documented usage)
├── fare.py (added unique_id)
├── fare_gen.py (added match_num, sequence params)
├── fms.py (added seeding, fareSequence)
└── router.py (updated paths, added auth to whereami)

VehiclePositioningSystem/
└── map_monitor.py (updated Dashboard paths)

requirements.txt (merged and alphabetized)
```

### Removed (moved to Dashboard/)
```
VehiclePositioningSystem/
├── static/ (moved)
└── templates/ (moved)
```

---

## 9. Configuration Files

### Existing Configurations Used
- **Config/fare_types.yaml** - Fare type definitions (STANDARD, SPECIAL)
- **Config/spawn_points.yaml** - Spawn point locations and biases
- **Config/team_names.yaml** - Available team names for autocomplete

### Operating Modes
- **LAB**: Development mode, simple team number authentication
- **HOME**: Similar to LAB mode
- **MATCH**: Competition mode, uses secret authentication codes

---

## 10. Key Improvements & Features

### For Competition Fairness
✅ Reproducible fares using match-based random seeding  
✅ Unique fare IDs for database tracking  
✅ Team position privacy (authentication required)  
✅ Consistent fare generation across sessions  

### For User Experience
✅ Real-time dashboard with live updates  
✅ Visual fare countdown timers  
✅ Team active fare indicators  
✅ Horizontal team layout for better space usage  
✅ Color-coded fare types  
✅ Responsive design  

### For System Administration
✅ Admin panel for team configuration  
✅ Comprehensive API endpoints  
✅ WebSocket support for real-time updates  
✅ Debug logging throughout frontend  

---

## 11. Future Considerations

### Potential Enhancements
- Database integration for match results storage
- Historical match data visualization
- Team performance analytics
- Leaderboard system
- Match replay functionality using seeded random state

### Security Improvements for MATCH Mode
- Generate unique authentication codes for each team
- Store codes securely (environment variables or encrypted config)
- Implement rate limiting on API endpoints
- Add IP whitelisting for admin endpoints

---

## Technologies Used
- **Backend**: Flask, Flask-SocketIO, Python 3.12
- **Frontend**: Vanilla JavaScript, Socket.IO client
- **Data**: YAML configuration files
- **Real-time**: WebSocket (Socket.IO)
- **Styling**: Custom CSS with modern design patterns

---

## Notes
- All changes maintain backwards compatibility where possible
- Debug logging can be removed for production deployment
- System tested with up to 6 teams and 8+ concurrent fares
- Dashboard optimized for 1920x1080 resolution, responsive down to 1200px

---

**Last Updated**: January 26, 2026  
**Project**: VPFS (Vehicle Positioning and Fare System)  
**Session Focus**: Dashboard reorganization, fare reproducibility, UI enhancements
