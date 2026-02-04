/**
 * Admin Panel - Team Configuration
 */

class AdminPanel {
    constructor() {
        this.teamCount = 0;
        this.teamNames = [];
        this.duckColors = ["Blue.png", "Red.png", "Green.png", "Yellow.png", "Purple.png", "Brown.png", "Grey.png"];
        this.activeTeams = {};
        
        this.init();
    }

    async init() {
        // Load team names from server
        await this.loadTeamNames();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load current teams
        await this.loadCurrentTeams();
        
        // Load system info
        await this.loadSystemInfo();
    }

    async loadTeamNames() {
        try {
            const response = await fetch('/api/admin/team-names');
            if (response.ok) {
                this.teamNames = await response.json();
                this.populateDatalist();
            }
        } catch (error) {
            console.error('Error loading team names:', error);
        }
    }

    populateDatalist() {
        const datalist = document.getElementById('team-names-datalist');
        datalist.innerHTML = '';
        
        this.teamNames.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            datalist.appendChild(option);
        });
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

        for (let i = 0; i < this.teamCount; i++) {
            const teamNumber = (i + 1);  // Team numbers: 1, 2, 3, 4, 5, 6, 7
            const duckColor = this.duckColors[i % this.duckColors.length];
            
            const card = document.createElement('div');
            card.className = 'team-card';
            card.innerHTML = `
                <div class="team-header">
                    <div class="duck-preview">
                        <img src="/assets/ducks/${duckColor}" alt="Team ${teamNumber} Duck">
                    </div>
                    <div class="team-info">
                        <div class="team-label">Team Number</div>
                        <div class="team-number">#${teamNumber}</div>
                    </div>
                </div>
                <div class="team-input-group">
                    <label for="team-${teamNumber}-name">Team Name</label>
                    <input 
                        type="text" 
                        id="team-${teamNumber}-name" 
                        class="team-name-input"
                        placeholder="Enter team name..."
                        list="team-names-datalist"
                        data-team-number="${teamNumber}"
                    >
                </div>
            `;

            grid.appendChild(card);

            // Add input event listener
            const input = card.querySelector('.team-name-input');
            input.addEventListener('input', (e) => {
                if (e.target.value.trim()) {
                    e.target.classList.add('filled');
                } else {
                    e.target.classList.remove('filled');
                }
            });

            // Pre-fill if team already exists
            if (this.activeTeams[teamNumber]) {
                input.value = this.activeTeams[teamNumber].name;
                input.classList.add('filled');
            }
        }
    }

    async saveConfiguration() {
        const teams = [];
        const inputs = document.querySelectorAll('.team-name-input');
        
        inputs.forEach(input => {
            const teamNumber = parseInt(input.dataset.teamNumber);
            const teamName = input.value.trim();
            
            if (teamName) {
                teams.push({
                    number: teamNumber,
                    name: teamName
                });
            }
        });

        if (teams.length === 0) {
            this.showStatus('Please enter at least one team name', 'error');
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
                
                // Clear input fields
                document.querySelectorAll('.team-name-input').forEach(input => {
                    input.value = '';
                    input.classList.remove('filled');
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
            }
        } catch (error) {
            console.error('Error loading system info:', error);
        }
    }

    updateTeamsCount(count) {
        document.getElementById('teams-registered').textContent = count;
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
