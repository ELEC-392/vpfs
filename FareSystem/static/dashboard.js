var teamsDiv, activeFareDiv, pastFareDiv, pastFareDivider;

var opMode = "";
const LAB_OP = "Lab";
const HOME_OP = "Home";
const MATCH_OP = "Match";

// Tracks fare IDs already seen as paid — used to fire confetti only on new completions.
const _paidFares = new Set();
let _faresInitialized = false;

function setVisibility(element, visible){
    if(!visible)
        element.style.display = "none"
    else
        element.style.display = "unset";
}

function getTimeUntil(utcTimeSeconds){
    let currentTime = (new Date()).getTime() / 1000
    return (utcTimeSeconds - currentTime);
}

function generateFareElement(fare, id){
    let type = "Normal";
    switch(fare.modifiers){
        case 1: type = "Subsidized"; break;
        case 2: type = "Senior"; break;
    }
    const el = document.createElement("div");
    el.classList.add("fare");
    el.id = id;
    el.innerHTML = `
    <h3>${fare.id}:</h3>
    <span class="tofrom">${fare.src.x.toFixed(0)},${fare.src.y.toFixed(0)} -> ${fare.dest.x.toFixed(0)},${fare.dest.y.toFixed(0)}</span>
    <span id="fare-${fare.id}-pay" style="padding:0">\$${fare.pay.toFixed(0)} / ${fare.reputation}%</span>
    <span id="fare-${fare.id}-modifier" style="display: none" class="bg-neutral">${type}</span>
    <span id="fare-${fare.id}-claim" style="display: none" class="bg-neutral">Team</span>
    <span id="fare-${fare.id}-pickedUp" style="display:none" class="bg-ok">Picked Up</span>
    <span id="fare-${fare.id}-completed" style="display:none" class="bg-ok">Completed</span>
    <span id="fare-${fare.id}-paid" style="display:none" class="bg-ok">Paid</span>
    <span id="fare-${fare.id}-inPosition" style="display:none" class="bg-warn">In Position</span>
    <span id="fare-${fare.id}-expiry" style="display:none">Expires in </span>
    `;
    // Place into the correct list at creation time
    const isExpired = getTimeUntil(fare.expiry) <= 0;
    const isActive = (fare.active !== undefined) ? !!fare.active : (!isExpired && !fare.completed && !fare.paid);
    (isActive ? activeFareDiv : pastFareDiv).appendChild(el);
    return el;
}

async function updateFares(){
    const req = await fetch("/dashboard/fares", { cache: "no-store" });
    const payload = await req.json();

    // Support both shapes: array of fares, or { active:[], past:[] }
    const fares = Array.isArray(payload)
        ? payload
        : ([]).concat(payload.active || payload.current || [], payload.past || payload.finished || []);

    for (const fare of fares){
        const id = fare.id;
        let el = document.getElementById(`fare-${id}`);
        if (!el) el = generateFareElement(fare, `fare-${id}`);

        const isExpired = getTimeUntil(fare.expiry) <= 0;
        const isActive = (fare.active !== undefined) ? !!fare.active : (!isExpired && !fare.completed && !fare.paid);

        // Move between lists if needed
        if (isActive && el.parentElement !== activeFareDiv) {
            activeFareDiv.appendChild(el);
        } else if (!isActive && el.parentElement !== pastFareDiv) {
            // Keep claimed above divider, unclaimed below
            if (fare.claimed && pastFareDivider) {
                pastFareDiv.insertBefore(el, pastFareDivider);
            } else if (pastFareDivider) {
                pastFareDiv.insertBefore(el, pastFareDivider.nextSibling);
            } else {
                pastFareDiv.appendChild(el);
            }
        }

        // Update fields
        const claimEl = document.getElementById(`fare-${id}-claim`);
        const expiryEl = document.getElementById(`fare-${id}-expiry`);

        if (fare.claimed){
            claimEl.innerText = `Team ${fare.team}`;
            setVisibility(claimEl, true);
            setVisibility(expiryEl, false);
        } else {
            const delta = getTimeUntil(fare.expiry);
            if (delta < 0) {
                expiryEl.innerText = "Expired";
            } else {
                expiryEl.innerText = `Expires in ${delta.toFixed(0)}`;
            }
            setVisibility(claimEl, false);
            setVisibility(expiryEl, true);
        }

        setVisibility(document.getElementById(`fare-${id}-inPosition`), !!fare.inPosition);
        setVisibility(document.getElementById(`fare-${id}-pickedUp`), !!fare.pickedUp);
        setVisibility(document.getElementById(`fare-${id}-completed`), !!fare.completed);
        setVisibility(document.getElementById(`fare-${id}-paid`), !!fare.paid);
        setVisibility(document.getElementById(`fare-${id}-modifier`), (fare.modifiers || 0) !== 0);

        // Confetti: trigger only when a fare transitions to paid for the first time.
        if (fare.paid && !_paidFares.has(id)) {
            if (_faresInitialized && fare.team != null) {
                triggerConfetti(fare.team);
            }
            _paidFares.add(id);
        }
    }
    _faresInitialized = true;
}

function triggerConfetti(teamNumber) {
    const card = document.getElementById(`team-${teamNumber}`);
    if (!card) return;

    // Flash the card
    card.classList.remove('fare-completed-flash');
    // Force reflow so re-adding the class always restarts the animation
    void card.offsetWidth;
    card.classList.add('fare-completed-flash');
    card.addEventListener('animationend', () => card.classList.remove('fare-completed-flash'), { once: true });

    // Confetti burst
    const rect = card.getBoundingClientRect();
    const originX = rect.left + rect.width  / 2;
    const originY = rect.top  + rect.height / 2;

    const colors = ['#f59e0b', '#10b981', '#3b82f6', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
    const PIECES = 45;

    for (let i = 0; i < PIECES; i++) {
        const piece = document.createElement('div');
        piece.className = 'confetti-piece';

        // Random trajectory
        const angle    = Math.random() * 2 * Math.PI;
        const dist     = 70 + Math.random() * 130;
        const dx       = (Math.cos(angle) * dist).toFixed(1);
        const dy       = (Math.sin(angle) * dist - 60).toFixed(1);  // bias upward
        const rot      = ((Math.random() - 0.5) * 720).toFixed(0);
        const dur      = (0.7 + Math.random() * 0.6).toFixed(2);
        const delay    = (Math.random() * 0.15).toFixed(2);
        const size     = (6 + Math.random() * 6).toFixed(1);
        const color    = colors[Math.floor(Math.random() * colors.length)];
        const isCircle = Math.random() > 0.5;

        piece.style.cssText = [
            `left:${originX}px`,
            `top:${originY}px`,
            `width:${size}px`,
            `height:${isCircle ? size : (parseFloat(size) * 1.4).toFixed(1)}px`,
            `background-color:${color}`,
            `border-radius:${isCircle ? '50%' : '2px'}`,
            `--dx:${dx}px`,
            `--dy:${dy}px`,
            `--rot:${rot}deg`,
            `--dur:${dur}s`,
            `--delay:${delay}s`,
        ].join(';');

        document.body.appendChild(piece);
        piece.addEventListener('animationend', () => piece.remove(), { once: true });
    }
}

function generateTeamElement(team, id) {
    let element = document.createElement("div");
    element.classList.add("team");
    element.id = id;
    element.innerHTML = `
    <h3 style="grid-area: title;">Team ${team.number}</h3>
    <div style="grid-area: money;">
        Money: <span id="team-${team.number}-money">${team.money}</span><br/>
        Reputation: <span id="team-${team.number}-reputation">${team.karma}%</span><br/>
        Current Fare: <span id="team-${team.number}-fare">None</span>
    </div>
    <div style="grid-area: position;">
        X <span id="team-${team.number}-x">${team.position.x.toFixed(2)}</span><br/>
        Y <span id="team-${team.number}-y">${team.position.y.toFixed(2)}</span><br/>
        Last Update: <span id="team-${team.number}-postime">${team.lastPosUpdate}</span>
    </div>
    <div style="grid-area:status;text-align:right" id="team-${team.number}-status"></div>
    <div style="grid-area: buttons;" class="team-buttons">
        ${(opMode == LAB_OP? `<button onclick="removeTeam(${team.number})">Remove Team</button>` : "")} 
    </div>
    `

    teamsDiv.appendChild(element);
    return element;
}

async function updateTeams(){
    const req = await fetch("/dashboard/teams"); // relative URL
    const data = await req.json();

    // Sort teams numerically by team.number
    const sorted = Array.isArray(data)
        ? data.slice().sort((a, b) => Number(a.number) - Number(b.number))
        : [];

    const orderIds = [];

    for (var team of sorted){
        let num = team.number;
        let element = document.getElementById(`team-${num}`);
        if(element == null){
            element = generateTeamElement(team, `team-${num}`)
        }
        orderIds.push(element.id);

        document.getElementById(`team-${team.number}-money`).innerText = `\$${team.money.toFixed(0)}`;
        document.getElementById(`team-${team.number}-reputation`).innerText = `${team.karma}%`;
        document.getElementById(`team-${team.number}-fare`).innerText = team.currentFare;
        document.getElementById(`team-${team.number}-x`).innerText = team.position.x.toFixed(2);
        document.getElementById(`team-${team.number}-y`).innerText = team.position.y.toFixed(2);

        // PRESERVED: last position update timing and color
        let posttime = document.getElementById(`team-${team.number}-postime`);
        timeDelta = -getTimeUntil(team.lastPosUpdate) * 1000;

        if(timeDelta > 10000000)
            posttime.innerHTML = "Never"
        else if (timeDelta > 10000)
            posttime.innerText = `${(timeDelta/1000).toFixed(0)}s`;
        else
            posttime.innerText = `${timeDelta.toFixed(0)}ms`;

        if(timeDelta > 5000)
            posttime.style.color = "red";
        else
            posttime.style.color = "unset";

        // PRESERVED: last status ping timing and color
        let status = document.getElementById(`team-${team.number}-status`)
        timeDelta = -getTimeUntil(team.lastStatus) * 1000;

        if(timeDelta > 10000000)
            status.innerHTML = "Not Connected"
        else if (timeDelta > 10000)
            status.innerText = `Last Ping: ${(timeDelta/1000).toFixed(0)}s`;
        else
            status.innerText = `Last Ping: ${timeDelta.toFixed(0)}ms`;

        if(timeDelta > 5000)
            status.style.color = "red";
        else
            status.style.color = "unset";
    }

    // Reorder DOM to match sorted order (preserves existing nodes/content)
    const container = teamsDiv || document.getElementById("teams");
    if (container) {
        const header = container.querySelector(".team.header");
        const frag = document.createDocumentFragment();
        for (const id of orderIds) {
            const node = document.getElementById(id);
            if (node) frag.appendChild(node);
        }
        container.textContent = "";
        if (header) container.appendChild(header);
        container.appendChild(frag);
    }

    // Prune any stale team nodes not in the latest data
    const host = teamsDiv || document.getElementById("teams");
    for (const child of Array.from(host.children)) {
        if (child.id && child.id.startsWith("team-") && !orderIds.includes(child.id)) {
            child.remove();
        }
    }
}

async function refreshMode() {
  const labControls = document.getElementById('lab-team-buttons');
  try {
    const res = await fetch('/match');
    const data = await res.json();
    const mode = String((data && data.mode) || '').toLowerCase();
    if (labControls) labControls.style.display = (mode === 'lab') ? '' : 'none';
  } catch (e) {
    if (labControls) labControls.style.display = 'none';
    console.warn('Failed to fetch /match', e);
  }
}

async function addTeam() {
  const input = document.getElementById('add-team-number');
  if (!input) return;
  const num = parseInt(input.value, 10);
  if (!Number.isInteger(num) || num <= 0) {
    alert('Enter a valid positive team number.');
    return;
  }
  try {
    const res = await fetch(`/Lab/AddTeam/${num}`, { method: 'GET' });
    const text = await res.text();
    if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
    input.value = '';
    await updateTeams(); // repopulate table
  } catch (e) {
    console.error(e);
    alert(`Failed to add team: ${e.message}`);
  }
}

// Ensure initial load populates panes and toggles LAB controls
document.addEventListener('DOMContentLoaded', () => {
  if (typeof updateFares === 'function') updateFares();
  if (typeof updateTeams === 'function') updateTeams();
  if (typeof refreshMode === 'function') {
    refreshMode();
    setInterval(refreshMode, 5000);
  }
});

function configureMatch(){
    fetch("/Lab/ConfigMatch", {
        method: "post",
        headers: new Headers({'content-type': 'application/json'}),
        body: JSON.stringify(
            {
                "number":parseInt(document.getElementById("match-num").value),
                "duration":parseInt(document.getElementById("match-duration").value),
            }
        )
    })
}
function startMatch(){
    fetch("/Lab/StartMatch", {
        method: "post"
    })
}

async function updateMatchInfo(){
    // Use special -2 team auth for the dashboard
    let res = await fetch("/match?auth=-2");
    let data = await res.json();

    opMode = data.mode;
    document.getElementById("title").innerText = `${opMode} Dashboard`;

    if(opMode == LAB_OP){
        // Show add team buttons
        document.getElementById("lab-team-buttons").style.display = "unset"
    }

    if(!data.matchStart){
        document.getElementById("match-status").innerText = `Match ${data.match}, Ready`
    } else {
        if(data.timeRemain < 0)
            document.getElementById("match-status").innerText = `Match ${data.match}, Finished`
        else
            document.getElementById("match-status").innerText = `Match ${data.match}, ${data.timeRemain.toFixed(0)}s`
    }
}

window.onload = () => {
  opMode = LAB_OP;
  activeFareDiv = document.getElementById("active-fares");
  pastFareDiv = document.getElementById("past-fares");
  pastFareDivider = document.getElementById("fare-divider");
  teamsDiv = document.getElementById("teams");

  setInterval(() => {
    updateFares();
    updateMatchInfo();
  }, 1000);

  // poll teams once per second
  setInterval(() => {
    updateTeams();
  }, 1000);

  // initial render
  updateTeams();
  refreshMode();
};

async function removeTeam(teamNumber) {
  try {
    const res = await fetch(`/Lab/RemoveTeam/${teamNumber}`, { method: 'GET' });
    const text = await res.text();
    if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
    await updateTeams();
  } catch (e) {
    console.error(e);
    alert(`Failed to remove team ${teamNumber}: ${e.message}`);
  }
}