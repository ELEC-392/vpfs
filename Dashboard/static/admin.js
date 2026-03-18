/**
 * Admin Panel - Team Configuration
 */

class AdminPanel {
    constructor() {
        this.teamCount = 0;
        this.knownTeams = [];  // [{number, name}] from teams.yaml
        this.duckColors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png", "Violet.png", "Cyan.png"];
        this.activeTeams = {};
        this.autoRegisterEnabled = false;
        
        this.init();
    }

    async init() {
        // Load known teams from teams.yaml via server
        await this.loadKnownTeams();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load current teams
        await this.loadCurrentTeams();
        
        // Load system info
        await this.loadSystemInfo();

        // Start polling match state
        setInterval(() => this.loadSystemInfo(), 3000);
    }

    async loadKnownTeams() {
        try {
            const response = await fetch('/api/admin/known-teams');
            if (response.ok) {
                this.knownTeams = await response.json();
            }
        } catch (error) {
            console.error('Error loading known teams:', error);
        }
    }

    setupEventListeners() {
        // Number buttons
        document.querySelectorAll('.num-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const count = parseInt(e.target.dataset.count);
                this.setTeamCount(count);
            });
        });

        // Save button
        document.getElementById('save-btn').addEventListener('click', () => {
            this.saveConfiguration();
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearAllTeams();
        });

        // Configure match button
        document.getElementById('configure-match-btn').addEventListener('click', () => {
            this.configureMatch();
        });

        // Start match button
        document.getElementById('start-match-btn').addEventListener('click', () => {
            this.startMatch();
        });

        // Stop match button
        document.getElementById('stop-match-btn').addEventListener('click', () => {
            this.stopMatch();
        });

        // Reset match button
        document.getElementById('reset-match-btn').addEventListener('click', () => {
            this.resetMatch();
        });

        // Auto-register toggle
        document.getElementById('auto-register-toggle').addEventListener('change', (e) => {
            this.toggleAutoRegister(e.target.checked);
        });

        // Fare lines toggle
        document.getElementById('fare-lines-toggle').addEventListener('change', (e) => {
            this.setFareLinesVisible(e.target.checked);
        });
    }

    setTeamCount(count) {
        this.teamCount = count;
        
        // Update button states
        document.querySelectorAll('.num-btn').forEach(btn => {
            btn.classList.remove('active');
            if (parseInt(btn.dataset.count) === count) {
                btn.classList.add('active');
            }
        });

        // Generate team cards
        this.generateTeamCards();
    }

    generateTeamCards() {
        const grid = document.getElementById('teams-grid');
        grid.innerHTML = '';

        // Build the option list once
        const options = this.knownTeams
            .map(t => `<option value="${t.number}">${t.name} (Kit #${t.number})</option>`)
            .join('');

        // Pre-fill slots with currently active teams in kit-number order
        const activeList = Object.values(this.activeTeams)
            .sort((a, b) => a.number - b.number);

        for (let i = 0; i < this.teamCount; i++) {
            const slotNumber = i + 1;
            const duckColor = this.duckColors[i % this.duckColors.length];

            const card = document.createElement('div');
            card.className = 'team-card';
            card.innerHTML = `
                <div class="team-header">
                    <div class="duck-preview">
                        <img src="/assets/ducks/${duckColor}" alt="Slot ${slotNumber} Duck">
                    </div>
                    <div class="team-info">
                        <div class="team-label">Slot ${slotNumber}</div>
                    </div>
                </div>
                <div class="team-input-group">
                    <label for="team-slot-${slotNumber}">Select Team</label>
                    <select id="team-slot-${slotNumber}" class="team-select" data-slot="${slotNumber}">
                        <option value="">— choose a team —</option>
                        ${options}
                    </select>
                </div>
            `;

            grid.appendChild(card);

            // Pre-fill from currently active teams
            if (activeList[i]) {
                card.querySelector('.team-select').value = activeList[i].number;
            }
        }
    }

    async saveConfiguration() {
        const teams = [];
        const seen = new Set();

        document.querySelectorAll('.team-select').forEach(select => {
            if (!select.value) return;
            const teamNumber = parseInt(select.value);
            if (seen.has(teamNumber)) return;  // skip duplicate selections
            seen.add(teamNumber);
            const known = this.knownTeams.find(t => t.number === teamNumber);
            teams.push({
                number: teamNumber,
                name: known ? known.name : `Team ${teamNumber}`
            });
        });

        if (teams.length === 0) {
            this.showStatus('Please select at least one team', 'error');
            return;
        }

        try {
            const response = await fetch('/api/admin/configure-teams', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ teams })
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showStatus(`Successfully configured ${teams.length} team(s)!`, 'success');
                await this.loadCurrentTeams();
                // Reset selects to reflect active set
                this.generateTeamCards();
            } else {
                this.showStatus(result.message || 'Error saving configuration', 'error');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showStatus('Network error while saving', 'error');
        }
    }

    async clearAllTeams() {
        if (!confirm('Are you sure you want to remove all teams?')) {
            return;
        }

        try {
            const response = await fetch('/api/admin/clear-teams', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showStatus('All teams cleared successfully', 'success');
                await this.loadCurrentTeams();
                
                // Clear select fields
                document.querySelectorAll('.team-select').forEach(select => {
                    select.value = '';
                });
            } else {
                this.showStatus(result.message || 'Error clearing teams', 'error');
            }
        } catch (error) {
            console.error('Error clearing teams:', error);
            this.showStatus('Network error while clearing teams', 'error');
        }
    }

    async loadCurrentTeams() {
        try {
            const response = await fetch('/api/admin/current-teams');
            if (response.ok) {
                const teams = await response.json();
                this.activeTeams = {};
                
                teams.forEach(team => {
                    this.activeTeams[team.number] = team;
                });
                
                this.displayActiveTeams(teams);
                this.updateTeamsCount(teams.length);
            }
        } catch (error) {
            console.error('Error loading current teams:', error);
        }
    }

    displayActiveTeams(teams) {
        const container = document.getElementById('active-teams-list');
        
        if (teams.length === 0) {
            container.innerHTML = '<p class="empty-message">No teams configured</p>';
            return;
        }

        container.innerHTML = '';
        teams.forEach((team, idx) => {
            const duckColor = this.duckColors[idx % this.duckColors.length];
            
            const item = document.createElement('div');
            item.className = 'active-team-item';
            item.innerHTML = `
                <div class="active-team-duck">
                    <img src="/assets/ducks/${duckColor}" alt="${team.name}">
                </div>
                <div class="active-team-info">
                    <div class="active-team-name">${team.name}</div>
                    <div class="active-team-number">Team #${team.number}</div>
                </div>
                <button class="remove-team-btn" data-team-number="${team.number}">×</button>
            `;

            // Add remove button listener
            const removeBtn = item.querySelector('.remove-team-btn');
            removeBtn.addEventListener('click', () => {
                this.removeTeam(team.number);
            });

            container.appendChild(item);
        });
    }

    async removeTeam(teamNumber) {
        try {
            const response = await fetch(`/api/admin/remove-team/${teamNumber}`, {
                method: 'DELETE'
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showStatus(`Team #${teamNumber} removed`, 'success');
                await this.loadCurrentTeams();
            } else {
                this.showStatus(result.message || 'Error removing team', 'error');
            }
        } catch (error) {
            console.error('Error removing team:', error);
            this.showStatus('Network error while removing team', 'error');
        }
    }

    async loadSystemInfo() {
        try {
            const response = await fetch('/match');
            if (response.ok) {
                const data = await response.json();
                document.getElementById('system-mode').textContent = data.mode;
                // Update match state display
                const matchNum = document.getElementById('current-match-num');
                const matchStatus = document.getElementById('current-match-status');
                const timeRemain = document.getElementById('current-time-remain');
                if (matchNum)  matchNum.textContent  = data.match;
                if (matchStatus) {
                    if (data.matchStart)       matchStatus.textContent = '🟢 Running';
                    else if (data.matchPaused) matchStatus.textContent = '⏸️ Paused';
                    else                       matchStatus.textContent = '⏹️ Stopped';
                }
                if (timeRemain) {
                    const secs = Math.max(0, Math.round(data.timeRemain));
                    const m = Math.floor(secs / 60).toString().padStart(2, '0');
                    const s = (secs % 60).toString().padStart(2, '0');
                    timeRemain.textContent = data.matchStart ? `${m}:${s}` : '—';
                }
            }
        } catch (error) {
            console.error('Error loading system info:', error);
        }
    }

    async configureMatch() {
        const matchNumber   = parseInt(document.getElementById('match-number').value);
        const durationMins  = parseFloat(document.getElementById('match-duration').value);
        const seedInput     = document.getElementById('match-seed').value.trim();
        const seed          = seedInput !== '' ? parseInt(seedInput) : null;

        if (!matchNumber || matchNumber < 1 || !durationMins || durationMins <= 0) {
            this.showMatchStatus('Please enter a valid match number and duration', 'error');
            return;
        }

        try {
            const body = { match_number: matchNumber, duration_minutes: durationMins };
            if (seed !== null) body.seed = seed;
            const response = await fetch('/api/admin/match/configure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await response.json();
            if (response.ok) {
                this.showMatchStatus(result.message, 'success');
                await this.loadSystemInfo();
            } else {
                this.showMatchStatus(result.message || 'Error configuring match', 'error');
            }
        } catch (error) {
            this.showMatchStatus('Network error', 'error');
        }
    }

    async startMatch() {
        if (!confirm('Start the match now?')) return;
        try {
            const response = await fetch('/api/admin/match/start', { method: 'POST' });
            const result = await response.json();
            if (response.ok) {
                this.showMatchStatus(result.message, 'success');
                await this.loadSystemInfo();
            } else {
                this.showMatchStatus(result.message || 'Error starting match', 'error');
            }
        } catch (error) {
            this.showMatchStatus('Network error', 'error');
        }
    }

    async stopMatch() {
        if (!confirm('Stop the match now?')) return;
        try {
            const response = await fetch('/api/admin/match/stop', { method: 'POST' });
            const result = await response.json();
            this.showMatchStatus(result.message, response.ok ? 'success' : 'error');
            await this.loadSystemInfo();
        } catch (error) {
            this.showMatchStatus('Network error', 'error');
        }
    }

    async resetMatch() {
        if (!confirm('Reset the match timer? This will stop the current match and restore the same configuration.')) return;
        try {
            const response = await fetch('/api/admin/match/reset', { method: 'POST' });
            const result = await response.json();
            this.showMatchStatus(result.message, response.ok ? 'success' : 'error');
            await this.loadSystemInfo();
        } catch (error) {
            this.showMatchStatus('Network error', 'error');
        }
    }

    showMatchStatus(message, type) {
        const el = document.getElementById('match-status-message');
        el.textContent = message;
        el.className = `status-message ${type}`;
        setTimeout(() => { el.className = 'status-message'; }, 4000);
    }

    updateTeamsCount(count) {
        document.getElementById('teams-registered').textContent = count;
    }

    async setFareLinesVisible(visible) {
        try {
            const response = await fetch('/api/admin/fare-lines', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ visible })
            });
            if (!response.ok) {
                this.showMatchStatus('Error updating fare lines setting', 'error');
                // Revert toggle
                document.getElementById('fare-lines-toggle').checked = !visible;
            }
        } catch (error) {
            console.error('Error setting fare lines visibility:', error);
            this.showMatchStatus('Network error', 'error');
            document.getElementById('fare-lines-toggle').checked = !visible;
        }
    }

    async toggleAutoRegister(enabled) {
        this.autoRegisterEnabled = enabled;
        const manualConfig = document.getElementById('manual-team-config');

        if (enabled) {
            manualConfig.classList.add('disabled');
            try {
                const response = await fetch('/api/admin/auto-register-teams', { method: 'POST' });
                const result = await response.json();
                if (response.ok) {
                    this.showStatus(`Auto-registered ${result.teams} team(s) from teams.yaml`, 'success');
                    await this.loadCurrentTeams();
                } else {
                    this.showStatus(result.message || 'Error during auto-registration', 'error');
                    document.getElementById('auto-register-toggle').checked = false;
                    manualConfig.classList.remove('disabled');
                    this.autoRegisterEnabled = false;
                }
            } catch (error) {
                console.error('Error auto-registering teams:', error);
                this.showStatus('Network error during auto-registration', 'error');
                document.getElementById('auto-register-toggle').checked = false;
                manualConfig.classList.remove('disabled');
                this.autoRegisterEnabled = false;
            }
        } else {
            manualConfig.classList.remove('disabled');
            this.showStatus('Manual team configuration enabled', 'success');
        }
    }

    showStatus(message, type) {
        const statusEl = document.getElementById('status-message');
        statusEl.textContent = message;
        statusEl.className = `status-message ${type}`;
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            statusEl.className = 'status-message';
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Admin Panel...');
    const admin = new AdminPanel();
});
