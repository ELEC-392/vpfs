#!/usr/bin/env python3
"""
Quick test script to send position updates to teams
"""
import socketio
import time
import random

# Create Socket.IO client
sio = socketio.Client()

# Connect to server
sio.connect('http://localhost:5000')

print("Connected! Sending position updates...")

# Get current teams from the server
import requests
response = requests.get('http://localhost:5000/api/admin/current-teams')
teams_data = response.json()
teams = [t['number'] for t in teams_data]

if not teams:
    print("No teams configured! Please configure teams in the admin panel first.")
    sio.disconnect()
    exit(1)

print(f"Found {len(teams)} teams: {teams}")

positions = [(0.2 + i*0.1, 0.3 + i*0.1) for i in range(len(teams))]

# Send initial positions
update_data = []
for team, (x, y) in zip(teams, positions):
    update_data.append({"team": team, "x": x, "y": y})

sio.emit('whereami_update', update_data)
print(f"Sent initial positions for {len(teams)} teams")

# Simulate some movement
for i in range(10):
    time.sleep(2)
    update_data = []
    # Move one random team
    team = random.choice(teams)
    idx = teams.index(team)
    x = max(0, min(1, positions[idx][0] + random.uniform(-0.1, 0.1)))
    y = max(0, min(1, positions[idx][1] + random.uniform(-0.1, 0.1)))
    positions[idx] = (x, y)
    
    update_data.append({"team": team, "x": x, "y": y})
    sio.emit('whereami_update', update_data)
    print(f"Moved team {team} to ({x:.2f}, {y:.2f})")

sio.disconnect()
print("Done!")
