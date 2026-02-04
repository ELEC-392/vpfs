# Admin Interface - Quick Guide

## Overview
The admin interface at **http://localhost:5000/admin** allows you to configure teams for the VPFS match.

## Features

### Team Configuration
1. **Select Number of Teams** - Click one of the buttons (1-7) to choose how many teams will participate
2. **Team Cards Display** - Each team gets:
   - A unique duck avatar (different color)
   - A team number (3, 5, 7, 9, 11, 13, 15)
   - A name input field with autocomplete

### Autocomplete Team Names
- Start typing in any team name field
- Suggestions will appear from the team names database (Config/team_names.yaml)
- Select from suggestions or type a custom name

### Actions
- **💾 Save Configuration** - Saves the configured teams to the system and broadcasts to the map monitor
- **🗑️ Clear All Teams** - Removes all teams from the system
- **× Remove Individual Team** - Click the × button next to any team in the "Active Teams" sidebar

### Active Teams Sidebar
Shows all currently configured teams with:
- Team duck avatar
- Team name
- Team number
- Quick remove button

## Workflow

1. **Open Admin Panel**: http://localhost:5000/admin
2. **Select Number of Teams**: Click a number button (1-7)
3. **Enter Team Names**: Type names in the text fields (autocomplete available)
4. **Save Configuration**: Click "💾 Save Configuration"
5. **View on Map**: Navigate to http://localhost:5000/map to see teams

## Team Name Database

Team names are loaded from: `Config/team_names.yaml`

You can edit this file to add more team names for autocomplete:

```yaml
teams:
  - Team Alpha
  - Team Bravo
  - Your Custom Team Name
  # Add more team names here
```

## API Endpoints

### Admin Endpoints
- `GET /admin` - Admin panel interface
- `GET /api/admin/team-names` - Get autocomplete team names
- `GET /api/admin/current-teams` - Get currently configured teams
- `POST /api/admin/configure-teams` - Save team configuration
- `POST /api/admin/clear-teams` - Remove all teams
- `DELETE /api/admin/remove-team/<number>` - Remove specific team

## Integration with Map Monitor

When you save teams in the admin panel:
1. Teams are instantly added to the system
2. Map monitor at http://localhost:5000/map updates automatically
3. Duck avatars appear on the map with team names
4. Position updates work immediately

## Notes

- Team numbers are automatically assigned: 3, 5, 7, 9, 11, 13, 15
- Duck colors rotate through: Blue, Red, Green, Yellow, Purple, Brown, Grey
- Admin functions only work in LAB mode
- Changes broadcast to all connected map monitor clients in real-time
