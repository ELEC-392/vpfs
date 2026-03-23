/* ============================================================
   VPFS Referee Panel — JavaScript
   ============================================================ */

'use strict';

// ── State ───────────────────────────────────────────────────
let selectedTeamId   = null;   // currently watched team (number)
let selectedTeamName = "";
let pollTimer        = null;   // setInterval handle for right-panel refresh
let teamsData        = [];     // last team list from /api/referee/teams

// Fare-state human labels
const FARE_STATE_LABELS = {
    to_pickup:  "Going to pickup",
    at_pickup:  "At pickup",
    to_dropoff: "Going to dropoff",
    at_dropoff: "At dropoff",
};

// ── Boot ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    refreshTeamList();
    setInterval(refreshTeamList, 3000);   // refresh left panel every 3 s

    // Delegate counter button clicks
    document.getElementById("active-state").addEventListener("click", onCounterClick);

    // Unassign button
    document.getElementById("btn-unassign").addEventListener("click", onUnassign);

    // Mobile drawer toggle
    document.getElementById("btn-team-toggle").addEventListener("click", openDrawer);
    document.getElementById("btn-close-panel").addEventListener("click", closeDrawer);
    document.getElementById("ref-drawer-backdrop").addEventListener("click", closeDrawer);
});

// ── Drawer helpers (mobile) ──────────────────────────────────
function isMobile() {
    return window.matchMedia("(max-width: 700px)").matches;
}

function openDrawer() {
    document.getElementById("ref-team-list").classList.add("drawer-open");
    document.getElementById("ref-drawer-backdrop").classList.add("visible");
    document.getElementById("btn-team-toggle").setAttribute("aria-expanded", "true");
}

function closeDrawer() {
    document.getElementById("ref-team-list").classList.remove("drawer-open");
    document.getElementById("ref-drawer-backdrop").classList.remove("visible");
    document.getElementById("btn-team-toggle").setAttribute("aria-expanded", "false");
}

// ── Team list ───────────────────────────────────────────────
async function refreshTeamList() {
    try {
        const res  = await fetch("/api/referee/teams");
        if (!res.ok) return;
        const data = await res.json();
        teamsData  = data.teams || [];
        renderTeamList(teamsData);

        // Also quietly update right panel if a team is selected
        if (selectedTeamId !== null) {
            updateRightPanel();
        }
    } catch (e) {
        // Network error — silently ignore, keep stale list
    }
}

function renderTeamList(teams) {
    const container = document.getElementById("team-list-container");
    if (!teams.length) {
        container.innerHTML = '<p class="placeholder-msg">No teams registered yet.</p>';
        return;
    }

    container.innerHTML = teams.map(t => {
        const isSelected = (t.number === selectedTeamId);
        const isMe       = (t.assignedReferee === REFEREE_ID);
        let assignedHtml = "";
        if (t.assignedReferee) {
            const label = isMe ? "You" : (t.assignedRefereeName || t.assignedReferee);
            assignedHtml = `<span class="team-assigned-badge${isMe ? " is-me" : ""}">${escHtml(label)}</span>`;
        }

        const fareLabel = t.fareState ? (FARE_STATE_LABELS[t.fareState] || t.fareState) : "Idle";
        return `
            <div class="team-card${isSelected ? " selected" : ""}"
                 data-team-id="${t.number}"
                 data-team-name="${escHtml(t.name)}">
                <div>
                    <div class="team-card-name">${escHtml(t.name)}</div>
                    <div class="team-card-sub">${fareLabel}</div>
                </div>
                <div class="team-card-right">
                    ${assignedHtml}
                </div>
            </div>`;
    }).join("");

    // Bind click handlers
    container.querySelectorAll(".team-card").forEach(card => {
        card.addEventListener("click", () => selectTeam(
            parseInt(card.dataset.teamId, 10),
            card.dataset.teamName,
        ));
    });
}

// ── Team selection ──────────────────────────────────────────
async function selectTeam(teamId, teamName) {
    // Already watching this team — nothing to do
    if (selectedTeamId === teamId) return;

    // Prompt if we are already watching a different team
    if (selectedTeamId !== null) {
        if (!confirm(`You are currently watching "${selectedTeamName}".\nSwitch to "${teamName}"?\n\n"${selectedTeamName}" will become available for another referee.`)) {
            return;
        }
    }

    // Prompt if another referee is already watching the target team
    const targetTeam = teamsData.find(t => t.number === teamId);
    if (targetTeam && targetTeam.assignedReferee && targetTeam.assignedReferee !== REFEREE_ID) {
        const watcherName = targetTeam.assignedRefereeName || targetTeam.assignedReferee;
        if (!confirm(`"${teamName}" is currently being watched by ${watcherName}.\nTake over?`)) {
            return;
        }
    }

    selectedTeamId   = teamId;
    selectedTeamName = teamName;

    // Mark selected in left panel immediately
    document.querySelectorAll(".team-card").forEach(c => {
        c.classList.toggle("selected", parseInt(c.dataset.teamId, 10) === teamId);
    });

    // Assign self to this team on the server (clears old assignment automatically)
    try {
        await fetch("/api/referee/assign", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ teamId }),
        });
    } catch (e) { /* non-fatal */ }

    // Show right panel
    document.getElementById("empty-state").style.display  = "none";
    document.getElementById("active-state").style.display = "block";
    document.getElementById("panel-team-name").textContent = teamName;

    // On mobile: close the drawer so the judging panel is visible
    if (isMobile()) closeDrawer();

    // Start polling right panel
    clearInterval(pollTimer);
    updateRightPanel();
    pollTimer = setInterval(updateRightPanel, 2000);
}

async function onUnassign() {
    clearInterval(pollTimer);
    selectedTeamId   = null;
    selectedTeamName = "";
    // Persist unassign to server so team shows as free for other referees
    try {
        await fetch("/api/referee/unassign", { method: "POST" });
    } catch (e) { /* non-fatal */ }
    document.getElementById("empty-state").style.display  = "";
    document.getElementById("active-state").style.display = "none";
    // On mobile: open the drawer back up so the referee can pick a new team
    if (isMobile()) openDrawer();
    refreshTeamList();
}

// ── Right panel data ────────────────────────────────────────
async function updateRightPanel() {
    if (selectedTeamId === null) return;
    try {
        const res  = await fetch(`/api/referee/team/${selectedTeamId}/state`);
        if (!res.ok) return;
        const data = await res.json();
        applyRightPanelData(data);
    } catch (e) { /* ignore */ }
}

function applyRightPanelData(data) {
    // Lock / unlock entry controls based on match state
    const locked = !data.matchRunning;
    const notice = document.getElementById("match-locked-notice");
    if (notice) notice.style.display = locked ? "" : "none";
    document.querySelectorAll(
        "#active-state .btn-plus, #active-state .btn-minus"
    ).forEach(btn => {
        btn.disabled = locked;
    });

    // Fare status badge
    const badge = document.getElementById("panel-fare-badge");
    const state = data.fareState;
    badge.textContent = state ? (FARE_STATE_LABELS[state] || state) : "Idle";
    badge.className   = "fare-badge" + (state ? ` state-${state}` : " state-none");

    // Fare type badge
    const typeBadge = document.getElementById("panel-fare-type");
    if (data.fareType) {
        typeBadge.textContent = data.fareType;
        typeBadge.className   = `fare-type-badge type-${data.fareType}`;
    } else {
        typeBadge.textContent = "";
        typeBadge.className   = "fare-type-badge";
    }

    // Karma
    const karmaEl = document.getElementById("panel-karma");
    if (karmaEl) {
        const k = data.karma;
        karmaEl.textContent = (k !== undefined && k !== null) ? k.toFixed(1) : "—";
        karmaEl.className   = "karma-value" + (k >= 0 ? " karma-pos" : " karma-neg");
    }

    // Violations
    const v = data.violations || {};
    setCounter("STANDARD", v.STANDARD || 0);
    setCounter("SEVERE",    v.SEVERE   || 0);

    // Manual achievements (0 or 1 per fare)
    const a = data.achievements || {};
    setCounter("ZERO_DUCKS_GIVEN",  a.ZERO_DUCKS_GIVEN  || 0);
    setCounter("YOU_SPIN_ME_ROUND", a.YOU_SPIN_ME_ROUND || 0);

    // Auto achievements — show live on-track status
    const sfEl = document.getElementById("count-SAFETY_FIRST");
    if (sfEl) {
        if (data.fareState) {
            sfEl.textContent = data.safetyFirstOnTrack ? "✓ On track" : "✗ Lost";
            sfEl.className   = "auto-value" + (data.safetyFirstOnTrack ? " on-track" : " lost");
        } else {
            sfEl.textContent = "—";
            sfEl.className   = "auto-value";
        }
    }
    const hoEl = document.getElementById("count-HOLDING_OUT");
    if (hoEl) {
        if (data.fareState) {
            hoEl.textContent = data.holdingOutOnTrack ? "✓ On track" : "—";
            hoEl.className   = "auto-value" + (data.holdingOutOnTrack ? " on-track" : "");
        } else {
            hoEl.textContent = "—";
            hoEl.className   = "auto-value";
        }
    }
    const lmEl = document.getElementById("count-LOOK_MA_NO_HANDS");
    if (lmEl) {
        if (data.fareState) {
            lmEl.textContent = data.lookMaOnTrack ? "✓ On track" : "✗ Lost";
            lmEl.className   = "auto-value" + (data.lookMaOnTrack ? " on-track" : " lost");
        } else {
            lmEl.textContent = "—";
            lmEl.className   = "auto-value";
        }
    }
}

function setCounter(id, value) {
    const el = document.getElementById(`count-${id}`);
    if (el) el.textContent = value;

    // Disable the minus button when the counter reaches zero OR when match is locked
    const notice   = document.getElementById("match-locked-notice");
    const isLocked = notice && notice.style.display !== "none";
    const minusBtn = document.querySelector(`[data-action][data-type="${id}"].btn-minus`);
    if (minusBtn) minusBtn.disabled = isLocked || (value <= 0);
}

// ── Counter buttons ─────────────────────────────────────────
async function onCounterClick(e) {
    const btn = e.target.closest("[data-action]");
    if (!btn || selectedTeamId === null) return;

    const action = btn.dataset.action;  // "violation" or "achievement"
    const type   = btn.dataset.type;
    // Violations support delta ±1 (minus button present).
    // Achievements only support delta=1 (no minus button).
    const delta  = btn.classList.contains("btn-minus") ? -1 : 1;

    // Optimistic UI disable during request
    btn.disabled = true;

    try {
        const endpoint = `/api/referee/team/${selectedTeamId}/${action}`;
        const body = action === "violation"
            ? { type, delta }
            : { type };          // achievements don't send delta
        const res = await fetch(endpoint, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        });
        const data = await res.json();
        if (res.ok && data.success) {
            if (action === "violation") {
                const v = data.violations || {};
                setCounter("STANDARD", v.STANDARD || 0);
                setCounter("SEVERE",   v.SEVERE   || 0);
            } else {
                const a = data.achievements || {};
                setCounter("ZERO_DUCKS_GIVEN",  a.ZERO_DUCKS_GIVEN  || 0);
                setCounter("YOU_SPIN_ME_ROUND", a.YOU_SPIN_ME_ROUND || 0);
            }
        } else {
            showToast(data.message || "Error updating count", true);
        }
    } catch (e) {
        showToast("Network error", true);
    } finally {
        btn.disabled = false;
    }
}

// ── Toast ────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, isError = false) {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.className   = "toast show" + (isError ? " error" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.className = "toast"; }, 2800);
}

// ── Util ──────────────────────────────────────────────────────
function escHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
