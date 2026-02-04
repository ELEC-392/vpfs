/**
 * Map Monitor - Real-time Team Position Tracking
 * Uses WebSocket for asynchronous position updates
 */

class MapMonitor {
    constructor() {
        this.socket = null;
        this.teams = [];
        this.positions = {};
        this.duckElements = {};
        this.fares = [];
        this.teamData = {};
        this.faresUpdateInterval = null;
        this.countdownInterval = null;
        
        // Fare type mapping (enum value to name)
        this.fareTypeMap = {
            0: 'STANDARD',
            1: 'SPECIAL'
        };
        
        // Team color mapping (based on duck colors)
        this.teamColors = {
            'Blue.png': '#3b82f6',
            'Red.png': '#ef4444',
            'Green.png': '#10b981',
            'Yellow.png': '#f59e0b',
            'Purple.png': '#8b5cf6',
            'Brown.png': '#92400e',
            'Grey.png': '#6b7280'
        };
        
        this.init();
    }

    init() {
        // Connect to WebSocket
        this.connectWebSocket();
        
        // Setup UI event listeners
        this.setupEventListeners();
        
        // Start periodic fares update (every 2 seconds)
        this.faresUpdateInterval = setInterval(() => {
            this.fetchFares();
            this.fetchTeamData();
        }, 2000);
        
        // Start countdown timer update (every second)
        this.countdownInterval = setInterval(() => {
            this.updateFareCountdowns();
        }, 1000);
        
        // Initial fetch of fares and team data
        this.fetchFares();
        this.fetchTeamData();
    }

    connectWebSocket() {
        // Connect to Socket.IO server
        this.socket = io();

        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
            // Request initial state
            this.fetchInitialData();
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });

        // Receive initial state
        this.socket.on('initial_state', (data) => {
            console.log('Received initial state:', data);
            this.teams = data.teams;
            this.positions = data.positions;
            this.initializeUI();
        });

        // Receive position updates (asynchronous)
        this.socket.on('position_update', (data) => {
            console.log('Position update:', data);
            this.updatePosition(data.team_id, data.x, data.y);
        });
    }

    async fetchInitialData() {
        // Fallback: fetch initial data via REST API if not received via socket
        try {
            const [teamsResponse, positionsResponse] = await Promise.all([
                fetch('/api/map/teams'),
                fetch('/api/map/positions')
            ]);
            
            if (teamsResponse.ok && positionsResponse.ok) {
                this.teams = await teamsResponse.json();
                this.positions = await positionsResponse.json();
                this.initializeUI();
                
                // Also fetch fares and team data
                this.fetchFares();
                this.fetchTeamData();
            }
        } catch (error) {
            console.error('Error fetching initial data:', error);
        }
    }
    
    async fetchFares() {
        try {
            const response = await fetch('/fares');
            if (response.ok) {
                this.fares = await response.json();
                console.log('Fetched fares:', this.fares.length, this.fares);
                this.renderFares();
                this.renderFareRoutes();
                this.updateStats();
            }
        } catch (error) {
            console.error('Error fetching fares:', error);
        }
    }
    
    async fetchTeamData() {
        try {
            const response = await fetch('/dashboard/teams');
            if (response.ok) {
                const teams = await response.json();
                // Convert array to object keyed by team number
                this.teamData = {};
                teams.forEach(team => {
                    this.teamData[team.number] = team;
                });
                this.updateTeamFares();
                this.renderFareRoutes();
            }
        } catch (error) {
            console.error('Error fetching team data:', error);
        }
    }
    
    renderFareRoutes() {
        const svgLayer = document.getElementById('fare-routes-layer');
        if (!svgLayer) return;
        
        // Clear existing routes
        svgLayer.innerHTML = '';
        
        // Set viewBox to 0-100 range (matching percentage coordinates)
        svgLayer.setAttribute('viewBox', '0 0 100 100');
        svgLayer.setAttribute('preserveAspectRatio', 'none');
        
        console.log('Rendering fare routes, teamData:', this.teamData);
        
        // Map coordinate system is 500x500 pixels, need to normalize to 0-1 then to 0-100 for SVG
        const MAP_SIZE = 500;
        
        // Iterate through teams to find their claimed fares
        for (const [teamNum, teamData] of Object.entries(this.teamData)) {
            // Check if team has a current fare
            if (teamData.currentFare === null || teamData.currentFare === undefined) continue;
            
            // Get the fare using the index
            const fareIdx = teamData.currentFare;
            const fare = this.fares.find(f => f.id === fareIdx);
            
            if (!fare || !fare.src || !fare.dest) continue;
            
            console.log(`Team ${teamNum} has fare ${fareIdx}:`, fare);
            
            // Find team's duck color
            const team = this.teams.find(t => t.id === parseInt(teamNum));
            const teamColor = team ? (this.teamColors[team.duck] || '#6b7280') : '#6b7280';
            
            // Normalize pixel coordinates (0-500) to 0-1, then scale to 0-100 for SVG viewBox
            const srcX = (fare.src.x / MAP_SIZE) * 100;
            const srcY = (fare.src.y / MAP_SIZE) * 100;
            const destX = (fare.dest.x / MAP_SIZE) * 100;
            const destY = (fare.dest.y / MAP_SIZE) * 100;
            
            console.log(`Drawing route for team ${teamNum} with color ${teamColor} from (${srcX.toFixed(1)}, ${srcY.toFixed(1)}) to (${destX.toFixed(1)}, ${destY.toFixed(1)})`);
            
            // Create line element
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', srcX);
            line.setAttribute('y1', srcY);
            line.setAttribute('x2', destX);
            line.setAttribute('y2', destY);
            line.setAttribute('stroke', teamColor);
            line.setAttribute('stroke-width', '0.5');
            line.setAttribute('stroke-opacity', '0.7');
            line.setAttribute('stroke-dasharray', '2,1');
            svgLayer.appendChild(line);
            
            // Create start point (pickup)
            const startCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            startCircle.setAttribute('cx', srcX);
            startCircle.setAttribute('cy', srcY);
            startCircle.setAttribute('r', '1');
            startCircle.setAttribute('fill', teamColor);
            startCircle.setAttribute('fill-opacity', '0.8');
            startCircle.setAttribute('stroke', 'white');
            startCircle.setAttribute('stroke-width', '0.3');
            svgLayer.appendChild(startCircle);
            
            // Create end point (dropoff)
            const endCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            endCircle.setAttribute('cx', destX);
            endCircle.setAttribute('cy', destY);
            endCircle.setAttribute('r', '1.2');
            endCircle.setAttribute('fill', teamColor);
            endCircle.setAttribute('fill-opacity', '0.6');
            endCircle.setAttribute('stroke', 'white');
            endCircle.setAttribute('stroke-width', '0.3');
            endCircle.setAttribute('stroke-dasharray', '0.5,0.3');
            svgLayer.appendChild(endCircle);
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.textContent = 'Connected';
            statusElement.className = 'connected';
        } else {
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'disconnected';
        }
    }

    initializeUI() {
        // Check if we have teams
        if (!this.teams || this.teams.length === 0) {
            console.warn('No teams available. Add teams to see them on the map.');
            this.showNoTeamsMessage();
            return;
        }
        
        // Render legend
        this.renderLegend();
        
        // Create duck elements on map
        this.createDuckElements();
        
        // Position all ducks at their initial positions
        this.positionAllDucks();
        
        // Update statistics
        this.updateStats();
    }

    showNoTeamsMessage() {
        const legendContainer = document.getElementById('legend-items');
        legendContainer.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #6b7280;">
                <p style="margin-bottom: 10px;">No teams active</p>
                <p style="font-size: 12px;">Teams will appear here when they join the match.</p>
            </div>
        `;
        
        const activeTeamsElement = document.getElementById('active-teams');
        if (activeTeamsElement) {
            activeTeamsElement.textContent = '0';
        }
    }

    renderLegend() {
        const legendContainer = document.getElementById('legend-items');
        legendContainer.innerHTML = '';

        this.teams.forEach(team => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.dataset.teamId = team.id;

            const pos = this.positions[team.id] || { x: 0, y: 0 };

            legendItem.innerHTML = `
                <div class="legend-duck">
                    <img src="/assets/ducks/${team.duck}" alt="${team.name}">
                </div>
                <div class="legend-info">
                    <div class="legend-name">${team.name}</div>
                    <div class="legend-coords" id="coords-${team.id}">
                        x: ${(pos.x * 100).toFixed(1)}% y: ${(pos.y * 100).toFixed(1)}%
                    </div>
                    <div class="legend-fare" id="fare-${team.id}"></div>
                </div>
            `;

            // Add click handler to highlight team
            legendItem.addEventListener('click', () => {
                this.highlightTeam(team.id);
            });

            legendContainer.appendChild(legendItem);
        });
        
        // Update team fares
        this.updateTeamFares();
    }
    
    updateTeamFares() {
        this.teams.forEach(team => {
            const fareElement = document.getElementById(`fare-${team.id}`);
            if (fareElement) {
                const teamData = this.teamData[team.id];
                if (teamData && teamData.currentFare !== null && teamData.currentFare !== undefined) {
                    const fare = this.fares[teamData.currentFare];
                    if (fare) {
                        fareElement.textContent = `Active Fare: #${fare.unique_id || fare.id}`;
                        fareElement.style.display = 'block';
                    } else {
                        fareElement.textContent = `Active Fare: #${teamData.currentFare}`;
                        fareElement.style.display = 'block';
                    }
                } else {
                    fareElement.textContent = '';
                    fareElement.style.display = 'none';
                }
            }
        });
    }
    
    renderFares() {
        const faresContainer = document.getElementById('fares-list');
        if (!faresContainer) {
            console.warn('Fares container not found!');
            return;
        }
        
        // Filter out claimed fares that have expired
        const now = Date.now() / 1000;
        const activeFares = this.fares.filter(fare => {
            // If claimed and expired, don't show it
            if (fare.claimed && fare.expiry < now) {
                return false;
            }
            return true;
        });
        
        console.log('Rendering fares:', activeFares.length, activeFares);
        
        if (activeFares.length === 0) {
            faresContainer.innerHTML = `
                <div style="padding: 12px; text-align: center; color: #6b7280; font-size: 12px;">
                    No active fares available
                </div>
            `;
            return;
        }
        
        faresContainer.innerHTML = '';
        let renderedCount = 0;
        
        activeFares.forEach((fare, index) => {
            try {
                const fareElement = document.createElement('div');
                
                // Handle modifiers being either a string or number (enum value)
                let fareTypeName = 'STANDARD';
                if (typeof fare.modifiers === 'number') {
                    fareTypeName = this.fareTypeMap[fare.modifiers] || 'STANDARD';
                } else if (typeof fare.modifiers === 'string') {
                    fareTypeName = fare.modifiers.toUpperCase();
                }
                
                const isSpecial = fareTypeName === 'SPECIAL';
                const isClaimed = fare.claimed;
                
                fareElement.className = `fare-item ${isSpecial ? 'special' : ''} ${isClaimed ? 'claimed' : ''}`;
                fareElement.dataset.fareId = fare.id;
                fareElement.dataset.expiry = fare.expiry;
                
                // Calculate distance with null checks
                let distance = 0;
                if (fare.src && fare.dest) {
                    distance = this.calculateDistance(fare.src, fare.dest);
                } else {
                    console.warn(`Fare ${fare.id} missing src or dest:`, fare);
                    distance = 0;
                }
                
                // Calculate time left - fare.expiry is in seconds (epoch time)
                const now = Date.now() / 1000;
                const timeLeft = fare.expiry - now;
                console.log(`Fare ${fare.id}: expiry=${fare.expiry}, now=${now}, timeLeft=${timeLeft}`);
                
                fareElement.innerHTML = `
                    <div class="fare-header">
                        <span class="fare-id">Fare #${fare.unique_id || fare.id}${isClaimed ? ' 🔒' : ''}</span>
                        <span class="fare-type ${isSpecial ? 'special' : ''}">${fareTypeName}</span>
                    </div>
                    <div class="fare-details">
                        <span class="fare-distance">📏 ${distance.toFixed(2)} units</span>
                        <span class="fare-timer" data-expiry="${fare.expiry}">⏱ ${this.formatTime(Math.max(0, timeLeft))}</span>
                    </div>
                `;
                
                faresContainer.appendChild(fareElement);
                renderedCount++;
                console.log(`✓ Rendered fare ${index}: ID=${fare.id}, unique_id=${fare.unique_id}`);
            } catch (error) {
                console.error(`✗ Error rendering fare ${index}:`, error, fare);
            }
        });
        
        console.log(`Total fare elements rendered: ${renderedCount} out of ${activeFares.length}`);
        console.log('Fares container children:', faresContainer.children.length);
    }
    
    calculateDistance(src, dest) {
        const dx = dest.x - src.x;
        const dy = dest.y - src.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    formatTime(seconds) {
        if (seconds <= 0) return 'EXPIRED';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    updateFareCountdowns() {
        const now = Date.now() / 1000;
        document.querySelectorAll('.fare-timer').forEach(timer => {
            const expiry = parseFloat(timer.dataset.expiry);
            const timeLeft = Math.max(0, expiry - now);
            timer.textContent = `⏱ ${this.formatTime(timeLeft)}`;
            
            // Add blinking animation if expiring soon (< 30 seconds)
            if (timeLeft < 30 && timeLeft > 0) {
                timer.classList.add('expiring-soon');
            } else {
                timer.classList.remove('expiring-soon');
            }
        });
    }

    createDuckElements() {
        const ducksLayer = document.getElementById('ducks-layer');
        ducksLayer.innerHTML = '';

        this.teams.forEach(team => {
            const duckElement = document.createElement('div');
            duckElement.className = 'duck';
            duckElement.dataset.teamId = team.id;
            duckElement.id = `duck-${team.id}`;

            duckElement.innerHTML = `
                <img src="/assets/ducks/${team.duck}" alt="${team.name}">
                <div class="duck-label">${team.name}</div>
            `;

            // Add click handler
            duckElement.addEventListener('click', () => {
                this.highlightTeam(team.id);
            });

            ducksLayer.appendChild(duckElement);
            this.duckElements[team.id] = duckElement;
        });
    }

    positionAllDucks() {
        this.teams.forEach(team => {
            const pos = this.positions[team.id];
            if (pos) {
                this.positionDuck(team.id, pos.x, pos.y, false);
            }
        });
    }

    positionDuck(teamId, normalizedX, normalizedY, animate = true) {
        const duckElement = this.duckElements[teamId];
        if (!duckElement) return;

        // Convert normalized coordinates (0-1) to percentage
        const xPercent = normalizedX * 100;
        const yPercent = normalizedY * 100;

        // Apply position with CSS transform
        duckElement.style.left = `${xPercent}%`;
        duckElement.style.top = `${yPercent}%`;

        // Add animation class if animating
        if (animate) {
            duckElement.classList.add('updating');
            setTimeout(() => {
                duckElement.classList.remove('updating');
            }, 500);
        }
    }

    updatePosition(teamId, x, y) {
        // Update internal state
        this.positions[teamId] = { x, y };

        // Update duck position on map (with animation)
        this.positionDuck(teamId, x, y, true);

        // Update legend coordinates
        this.updateLegendCoordinates(teamId, x, y);

        // Update last update time
        this.updateLastUpdateTime();
    }

    updateLegendCoordinates(teamId, x, y) {
        const coordsElement = document.getElementById(`coords-${teamId}`);
        if (coordsElement) {
            coordsElement.textContent = `x: ${(x * 100).toFixed(1)}% y: ${(y * 100).toFixed(1)}%`;
        }
    }

    highlightTeam(teamId) {
        // Remove active class from all legend items
        document.querySelectorAll('.legend-item').forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to selected team
        const legendItem = document.querySelector(`.legend-item[data-team-id="${teamId}"]`);
        if (legendItem) {
            legendItem.classList.add('active');
        }

        // Briefly highlight the duck
        const duckElement = this.duckElements[teamId];
        if (duckElement) {
            duckElement.style.transform = 'translate(-50%, -50%) scale(1.5)';
            duckElement.style.zIndex = '1000';
            
            setTimeout(() => {
                duckElement.style.transform = '';
                duckElement.style.zIndex = '';
            }, 800);
        }
    }

    updateStats() {
        const activeTeamsElement = document.getElementById('active-teams');
        if (activeTeamsElement) {
            activeTeamsElement.textContent = this.teams.length;
        }
        
        const activeFaresElement = document.getElementById('active-fares');
        if (activeFaresElement) {
            // Count fares excluding claimed+expired ones
            const now = Date.now() / 1000;
            const visibleFares = this.fares.filter(fare => {
                if (fare.claimed && fare.expiry < now) {
                    return false;
                }
                return true;
            });
            activeFaresElement.textContent = visibleFares.length;
        }
    }

    updateLastUpdateTime() {
        const lastUpdateElement = document.getElementById('last-update');
        if (lastUpdateElement) {
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            lastUpdateElement.textContent = timeString;
        }
    }

    setupEventListeners() {
        // Handle window resize to maintain duck positions
        window.addEventListener('resize', () => {
            this.positionAllDucks();
        });
    }
}

// Initialize the map monitor when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Map Monitor...');
    const monitor = new MapMonitor();
});
