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
        
        // Physical map dimensions in centimetres (tune as needed)
        this.MAP_WIDTH_CM  = 609.0;
        this.MAP_HEIGHT_CM = 488.0;

        // Rendering offset in centimetres applied to duck and fare-line positions.
        // Increase X_SHIFT_CM to shift right, Y_SHIFT_CM to shift up.
        this.X_SHIFT_CM = 0.0;
        this.Y_SHIFT_CM = 11.0;

        this.matchEndTime  = 0;
        this.matchDuration = 0;
        this.matchRunning  = false;

        // Staleness tracking: teamId -> ms timestamp of last position_update received.
        // Ducks not updated within STALE_MS are moved to origin and greyed out.
        this.lastSeen  = {};
        this.STALE_MS  = 3000;

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
            'Grey.png': '#6b7280',
            'Violet.png': '#7c3aed',
            'Cyan.png': '#06b6d4'
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
            this.fetchMatchState();
            // Re-apply duck positions from cached data so they always reflect
            // the current MAP_WIDTH_CM / MAP_HEIGHT_CM / shift constants.
            this.positionAllDucks();
        }, 2000);
        
        // Start countdown timer update (every second)
        this.countdownInterval = setInterval(() => {
            this.updateFareCountdowns();
            this.updateMatchCountdown();
            this.checkStaleDucks();
        }, 1000);
        
        // Initial fetch of fares and team data
        this.fetchFares();
        this.fetchTeamData();
        this.fetchMatchState();
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
            const [faresResp, settingResp, spawnResp] = await Promise.all([
                fetch('/fares'),
                fetch('/api/settings/fare-lines'),
                fetch('/api/settings/spawn-points')
            ]);
            if (faresResp.ok) {
                this.fares = await faresResp.json();
                console.log('Fetched fares:', this.fares.length, this.fares);
                this.renderFares();
                if (settingResp.ok) {
                    const setting = await settingResp.json();
                    this.fareRoutesVisible = setting.visible;
                }
                if (this.fareRoutesVisible !== false) {
                    this.renderFareRoutes();
                } else {
                    this.clearFareRoutes();
                }
                this.updateStats();
            }
            if (spawnResp.ok) {
                const spawnData = await spawnResp.json();
                if (spawnData.visible && spawnData.spawnPoints && spawnData.spawnPoints.length) {
                    this.renderSpawnPoints(spawnData.spawnPoints);
                } else {
                    this.clearSpawnPoints();
                }
            }
        } catch (error) {
            console.error('Error fetching fares:', error);
        }
    }
    
    async fetchMatchState() {
        try {
            const response = await fetch('/match');
            if (response.ok) {
                const data = await response.json();
                this.matchRunning  = data.matchStart;
                this.matchPaused   = data.matchPaused;
                this.matchEndTime  = data.matchStart ? (Date.now() / 1000 + data.timeRemain) : 0;
                this.matchPausedRemain = (!data.matchStart && data.timeRemain > 0) ? data.timeRemain : 0;
                // Record the wall-clock time when the pause was first detected so timers freeze
                if (data.matchPaused && !this._pausedAtNow) {
                    this._pausedAtNow = Date.now() / 1000;
                } else if (!data.matchPaused) {
                    this._pausedAtNow = null;
                }
                // Track total match duration for the progress bar
                if (data.matchStart && data.timeRemain > 0) {
                    this._matchTotalSecs = this._matchTotalSecs || data.timeRemain;
                }
                if (!data.matchStart && !data.matchPaused) this._matchTotalSecs = null;
                this.updateMatchCountdown();
            }
        } catch (e) {
            console.error('Error fetching match state:', e);
        }
    }

    updateMatchCountdown() {
        const timeEl = document.getElementById('countdown-time');
        const barEl  = document.getElementById('countdown-bar');
        if (!timeEl || !barEl) return;

        if (!this.matchRunning || !this.matchEndTime) {
            // Paused: show frozen remaining time
            if (this.matchPaused && this.matchPausedRemain > 0) {
                const secsLeft = this.matchPausedRemain;
                const total    = this._matchTotalSecs || secsLeft || 1;
                const pct      = (secsLeft / total) * 100;
                const m = Math.floor(secsLeft / 60).toString().padStart(2, '0');
                const s = Math.floor(secsLeft % 60).toString().padStart(2, '0');
                timeEl.textContent = `${m}:${s}`;
                timeEl.className   = 'countdown-time';
                barEl.style.width  = `${pct}%`;
                barEl.className    = 'countdown-bar';
            } else {
                timeEl.textContent = '--:--';
                timeEl.className   = 'countdown-time';
                barEl.style.width  = '100%';
                barEl.className    = 'countdown-bar';
            }
            return;
        }

        const secsLeft = Math.max(0, this.matchEndTime - Date.now() / 1000);

        // Match just expired client-side — freeze fare timers immediately without
        // waiting for the next server poll (which may be up to 2s away).
        if (secsLeft === 0 && !this._pausedAtNow) {
            this._pausedAtNow = Date.now() / 1000;
        }

        const total    = this._matchTotalSecs || 1;
        const pct      = (secsLeft / total) * 100;
        const m = Math.floor(secsLeft / 60).toString().padStart(2, '0');
        const s = Math.floor(secsLeft % 60).toString().padStart(2, '0');

        timeEl.textContent = `${m}:${s}`;
        barEl.style.width  = `${pct}%`;

        const expiring = secsLeft <= 60;
        timeEl.className = 'countdown-time' + (secsLeft === 0 ? ' finished' : expiring ? ' expiring' : '');
        barEl.className  = 'countdown-bar'  + (expiring ? ' expiring' : '');
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
                this.renderLegend();   // re-render to update CS scores and ordering
                if (this.fareRoutesVisible !== false) {
                    this.renderFareRoutes();
                } else {
                    this.clearFareRoutes();
                }
            }
        } catch (error) {
            console.error('Error fetching team data:', error);
        }
    }

    clearFareRoutes() {
        const svgLayer = document.getElementById('fare-routes-layer');
        if (svgLayer) {
            // Clear everything except the spawn-points group
            Array.from(svgLayer.childNodes).forEach(n => {
                if (n.id !== 'spawn-points-layer') n.remove();
            });
        }
    }

    clearSpawnPoints() {
        const g = document.getElementById('spawn-points-layer');
        if (g) g.innerHTML = '';
    }

    renderSpawnPoints(points) {
        const svgLayer = document.getElementById('fare-routes-layer');
        const g        = document.getElementById('spawn-points-layer');
        if (!svgLayer || !g) return;

        // Make sure the SVG viewBox is set (it may not be if no fares are active)
        if (!svgLayer.getAttribute('viewBox')) {
            svgLayer.setAttribute('viewBox', `0 0 ${this.MAP_WIDTH_CM} ${this.MAP_HEIGHT_CM}`);
            svgLayer.setAttribute('preserveAspectRatio', 'none');
        }

        g.innerHTML = '';

        const NS = 'http://www.w3.org/2000/svg';

        for (const pt of points) {
            // Apply the same Y-flip and shift as fare routes
            const cx = pt.x + this.X_SHIFT_CM;
            const cy = this.MAP_HEIGHT_CM - (pt.y + this.Y_SHIFT_CM);

            const color  = pt.active ? '#16a34a' : '#94a3b8';
            const radius = 5;

            // Outer ring (subtle halo)
            const halo = document.createElementNS(NS, 'circle');
            halo.setAttribute('cx', cx);
            halo.setAttribute('cy', cy);
            halo.setAttribute('r', radius + 4);
            halo.setAttribute('fill', color);
            halo.setAttribute('fill-opacity', '0.18');
            g.appendChild(halo);

            // Filled dot
            const dot = document.createElementNS(NS, 'circle');
            dot.setAttribute('cx', cx);
            dot.setAttribute('cy', cy);
            dot.setAttribute('r', radius);
            dot.setAttribute('fill', color);
            dot.setAttribute('stroke', 'white');
            dot.setAttribute('stroke-width', '1.5');
            g.appendChild(dot);

            // Label
            const label = document.createElementNS(NS, 'text');
            label.setAttribute('x', cx);
            label.setAttribute('y', cy - radius - 3);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '9');
            label.setAttribute('font-weight', 'bold');
            label.setAttribute('fill', color);
            label.setAttribute('stroke', 'white');
            label.setAttribute('stroke-width', '2.5');
            label.setAttribute('paint-order', 'stroke');
            label.textContent = pt.name;
            g.appendChild(label);
        }
    }
    
    renderFareRoutes() {
        const svgLayer = document.getElementById('fare-routes-layer');
        if (!svgLayer) return;
        
        // Clear existing routes but preserve the spawn-points group
        Array.from(svgLayer.childNodes).forEach(n => {
            if (n.id !== 'spawn-points-layer') n.remove();
        });
        
        // Set viewBox to match physical map dimensions for direct cm coordinates
        svgLayer.setAttribute('viewBox', `0 0 ${this.MAP_WIDTH_CM} ${this.MAP_HEIGHT_CM}`);
        svgLayer.setAttribute('preserveAspectRatio', 'none');
        
        console.log('Rendering fare routes, teamData:', this.teamData);
        
        // Iterate through teams to find their claimed fares
        for (const [teamNum, teamData] of Object.entries(this.teamData)) {
            // Check if team has a current fare
            if (teamData.currentFare === null || teamData.currentFare === undefined) continue;
            
            // Get the fare using the index
            const fareIdx = teamData.currentFare;
            const fare = this.fares.find(f => f.id === fareIdx);
            
            if (!fare || !fare.src || !fare.dest) continue;

            // Hide the route line when the duck has left the map AND the fare has expired
            const _teamId = parseInt(teamNum);
            const _duckEl = this.duckElements[_teamId];
            const _duckOffline = _duckEl && _duckEl.classList.contains('duck-offline');
            const _fareExpired = fare.expiry && fare.expiry < (Date.now() / 1000);
            if (_duckOffline && _fareExpired) continue;
            
            console.log(`Team ${teamNum} has fare ${fareIdx}:`, fare);
            
            // Find team's duck color
            const team = this.teams.find(t => t.id === parseInt(teamNum));
            const teamColor = team ? (this.teamColors[team.duck] || '#6b7280') : '#6b7280';
            
            // Fare coordinates are in cm — apply offset then use directly in the cm viewBox (Y flipped: origin = bottom-left)
            const srcX  = fare.src.x  + this.X_SHIFT_CM;
            const srcY  = this.MAP_HEIGHT_CM - (fare.src.y  + this.Y_SHIFT_CM);
            const destX = fare.dest.x + this.X_SHIFT_CM;
            const destY = this.MAP_HEIGHT_CM - (fare.dest.y + this.Y_SHIFT_CM);
            
            console.log(`Drawing route for team ${teamNum} with color ${teamColor} from (${srcX.toFixed(1)}, ${srcY.toFixed(1)}) to (${destX.toFixed(1)}, ${destY.toFixed(1)})`);
            
            // Create line element
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', srcX);
            line.setAttribute('y1', srcY);
            line.setAttribute('x2', destX);
            line.setAttribute('y2', destY);
            line.setAttribute('stroke', teamColor);
            line.setAttribute('stroke-width', '3');
            line.setAttribute('stroke-opacity', '0.7');
            line.setAttribute('stroke-dasharray', '2,1');
            svgLayer.appendChild(line);
            
            // Create start point (pickup) — white fill, team-color border
            const startCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            startCircle.setAttribute('cx', srcX);
            startCircle.setAttribute('cy', srcY);
            startCircle.setAttribute('r', '6');
            startCircle.setAttribute('fill', 'white');
            startCircle.setAttribute('stroke', teamColor);
            startCircle.setAttribute('stroke-width', '2');
            svgLayer.appendChild(startCircle);

            // Pickup tolerance zone — dashed circle, radius = POSITION_TOLERANCE (15 cm)
            const startTol = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            startTol.setAttribute('cx', srcX);
            startTol.setAttribute('cy', srcY);
            startTol.setAttribute('r', '15');
            startTol.setAttribute('fill', 'none');
            startTol.setAttribute('stroke', teamColor);
            startTol.setAttribute('stroke-width', '1');
            startTol.setAttribute('stroke-opacity', '0.6');
            startTol.setAttribute('stroke-dasharray', '3,3');
            svgLayer.appendChild(startTol);

            // Create end point (dropoff) — team-color fill, white border
            const endCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            endCircle.setAttribute('cx', destX);
            endCircle.setAttribute('cy', destY);
            endCircle.setAttribute('r', '7');
            endCircle.setAttribute('fill', teamColor);
            endCircle.setAttribute('stroke', 'white');
            endCircle.setAttribute('stroke-width', '2');
            svgLayer.appendChild(endCircle);

            // Dropoff tolerance zone — dashed circle, radius = POSITION_TOLERANCE (15 cm)
            const endTol = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            endTol.setAttribute('cx', destX);
            endTol.setAttribute('cy', destY);
            endTol.setAttribute('r', '15');
            endTol.setAttribute('fill', 'none');
            endTol.setAttribute('stroke', teamColor);
            endTol.setAttribute('stroke-width', '1');
            endTol.setAttribute('stroke-opacity', '0.6');
            endTol.setAttribute('stroke-dasharray', '3,3');
            svgLayer.appendChild(endTol);
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

        // Compute competition score CS = 0.5 * Ĉ + 0.5 * R̂
        // where Ĉ and R̂ are min-max normalised money and reputation across all teams.
        const moneyValues = this.teams.map(t => (this.teamData[t.id] || {}).money ?? 0);
        const repValues   = this.teams.map(t => (this.teamData[t.id] || {}).rep   ?? 0);
        const minM = Math.min(...moneyValues), maxM = Math.max(...moneyValues);
        const minR = Math.min(...repValues),   maxR = Math.max(...repValues);
        const norm = (v, lo, hi) => (hi === lo) ? 0.5 : (v - lo) / (hi - lo);

        const teamsWithCS = this.teams.map(team => {
            const td = this.teamData[team.id] || {};
            const cHat = norm(td.money ?? 0, minM, maxM);
            const rHat = norm(td.rep   ?? 0, minR, maxR);
            return { team, cs: 0.5 * cHat + 0.5 * rHat };
        }).sort((a, b) => b.cs - a.cs);

        teamsWithCS.forEach(({ team, cs }) => {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.dataset.teamId = team.id;

            const pos = this.positions[team.id] || { x: 0, y: 0 };

            const teamData = this.teamData[team.id] || {};
            const money = teamData.money !== undefined ? Math.round(teamData.money) : '—';
            const rep   = teamData.rep   !== undefined ? Math.round(teamData.rep)   : '—';
            const csDisp = (cs * 100).toFixed(0);

            legendItem.innerHTML = `
                <div class="legend-duck">
                    <img src="/assets/ducks/${team.duck}" alt="${team.name}">
                </div>
                <div class="legend-info">
                    <div class="legend-name">${team.name} <span class="legend-cs">CS: ${csDisp}%</span></div>
                    <div class="legend-coords" id="coords-${team.id}">x: ${Math.round(pos.x)} cm &nbsp; y: ${Math.round(pos.y)} cm</div>
                    <div class="legend-stats" id="stats-${team.id}">$${money} &nbsp;&nbsp; ⭐<span class="${rep < 0 ? 'rep-neg' : ''}">${rep}</span></div>
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
            // Refresh position row
            const coordsElement = document.getElementById(`coords-${team.id}`);
            if (coordsElement) {
                const pos = this.positions[team.id] || { x: 0, y: 0 };
                coordsElement.textContent = `x: ${Math.round(pos.x)} cm   y: ${Math.round(pos.y)} cm`;
            }

            // Refresh money / rep row
            const statsElement = document.getElementById(`stats-${team.id}`);
            if (statsElement) {
                const teamData = this.teamData[team.id] || {};
                const money = teamData.money !== undefined ? Math.round(teamData.money) : '—';
                const rep   = teamData.rep   !== undefined ? Math.round(teamData.rep)   : '—';
                statsElement.innerHTML = `$${money} &nbsp;&nbsp; ⭐<span class="${rep < 0 ? 'rep-neg' : ''}">${rep}</span>`;
            }

            // Active fare row
            const fareElement = document.getElementById(`fare-${team.id}`);
            if (fareElement) {
                const teamData = this.teamData[team.id];
                if (teamData && teamData.currentFare !== null && teamData.currentFare !== undefined) {
                    const fare = this.fares[teamData.currentFare];
                    const fareId = fare ? fare.id : teamData.currentFare;
                    const phaseMap = {
                        'to_pickup':  { icon: '🚕', label: 'to pickup',   cls: 'phase-to-pickup'  },
                        'at_pickup':  { icon: '📍', label: 'at pickup',   cls: 'phase-at-pickup'  },
                        'to_dropoff': { icon: '🚗', label: 'to dropoff',  cls: 'phase-to-dropoff' },
                        'at_dropoff': { icon: '🏁', label: 'at dropoff',  cls: 'phase-at-dropoff' },
                    };
                    const phase = phaseMap[teamData.fareState];
                    const phaseHtml = phase
                        ? ` <span class="fare-phase ${phase.cls}">${phase.icon} ${phase.label}</span>`
                        : '';
                    fareElement.innerHTML = `<span class="fare-id-tag">#${fareId}</span>${phaseHtml}`;
                    fareElement.style.display = 'block';
                } else {
                    fareElement.innerHTML = '';
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
                
                // Calculate time left - use frozen timestamp when match is paused/ended
                const now = this._pausedAtNow || Date.now() / 1000;
                const timeLeft = fare.expiry - now;
                
                fareElement.innerHTML = `
                    <div class="fare-header">
                        <span class="fare-id">#${fare.id}${isClaimed ? ' \uD83D\uDD12' : ''}</span>
                        <span class="fare-pay">$${fare.pay.toFixed(2)}</span>
                        <span class="fare-type ${isSpecial ? 'special' : ''}">${fareTypeName}</span>
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
        // Use the frozen timestamp when paused so fare timers don't advance
        const now = this._pausedAtNow || Date.now() / 1000;
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

            const _nameNums = team.name.match(/\d+/g);
            const _badgeNum = _nameNums ? _nameNums[_nameNums.length - 1] : team.id;

            duckElement.innerHTML = `
                <img src="/assets/ducks/${team.duck}" alt="${team.name}">
                <div class="duck-badge">${team.name.charAt(0).toUpperCase()}${_badgeNum}</div>
                <div class="duck-label">${team.name}</div>
            `;

            // Add click handler
            duckElement.addEventListener('click', () => {
                this.highlightTeam(team.id);
            });

            ducksLayer.appendChild(duckElement);
            this.duckElements[team.id] = duckElement;
            // Give each duck a fresh grace window so newly-created elements
            // don't get instantly marked stale before the first VPS frame arrives.
            this.lastSeen[team.id] = Date.now();
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

    positionDuck(teamId, x, y, animate = true) {
        const duckElement = this.duckElements[teamId];
        if (!duckElement) return;

        // Apply rendering offset then convert to percentage (Y flipped: origin = bottom-left)
        const xPercent = ((x + this.X_SHIFT_CM) / this.MAP_WIDTH_CM)  * 100;
        const yPercent = (1 - (y + this.Y_SHIFT_CM) / this.MAP_HEIGHT_CM) * 100;

        // Apply position with CSS transform
        duckElement.style.left = `${xPercent}%`;
        duckElement.style.top  = `${yPercent}%`;

        // Add animation class if animating
        if (animate) {
            duckElement.classList.add('updating');
            setTimeout(() => {
                duckElement.classList.remove('updating');
            }, 500);
        }
    }

    updatePosition(teamId, x, y) {
        // Mark this duck as actively seen
        this.lastSeen[teamId] = Date.now();

        // If the duck was previously marked offline, bring it back
        const duckEl = this.duckElements[teamId];
        if (duckEl && duckEl.classList.contains('duck-offline')) {
            duckEl.classList.remove('duck-offline');
        }

        // Update internal state
        this.positions[teamId] = { x, y };

        // Update duck position on map (with animation)
        this.positionDuck(teamId, x, y, true);

        // Update legend coordinates
        this.updateLegendCoordinates(teamId, x, y);

        // Update last update time
        this.updateLastUpdateTime();
    }

    resetDuck(teamId) {
        const duckElement = this.duckElements[teamId];
        if (!duckElement || duckElement.classList.contains('duck-offline')) return;

        // Teleport instantly to origin — no 0.8 s slide across the map
        duckElement.style.transition = 'none';
        const xOrigin = (this.X_SHIFT_CM / this.MAP_WIDTH_CM)  * 100;
        const yOrigin = (1 - this.Y_SHIFT_CM / this.MAP_HEIGHT_CM) * 100;
        duckElement.style.left = `${xOrigin.toFixed(3)}%`;
        duckElement.style.top  = `${yOrigin.toFixed(3)}%`;
        duckElement.classList.add('duck-offline');
        // Restore the normal position transition after the teleport so the next
        // real update will animate smoothly
        requestAnimationFrame(() => { duckElement.style.transition = ''; });

        // Update legend to show out-of-area
        const coordsElement = document.getElementById(`coords-${teamId}`);
        if (coordsElement) coordsElement.textContent = 'out of area';
    }

    checkStaleDucks() {
        const now = Date.now();
        this.teams.forEach(team => {
            const last = this.lastSeen[team.id] || 0;
            if (last > 0 && (now - last) > this.STALE_MS) {
                this.resetDuck(team.id);
            }
        });
    }

    updateLegendCoordinates(teamId, x, y) {
        const coordsElement = document.getElementById(`coords-${teamId}`);
        if (coordsElement) {
            coordsElement.textContent = `x: ${Math.round(x)} cm   y: ${Math.round(y)} cm`;
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
