"""Live web GUI for Wolf benchmark -- WebSocket event streaming + embedded HTML."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
from enum import Enum
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any

import websockets
from websockets.asyncio.server import serve as ws_serve

from wolf.engine.events import (
    EliminationEvent,
    GameEndEvent,
    GameEvent,
    NightResultEvent,
    PhaseChangeEvent,
    ReasoningEvent,
    SpeechEvent,
    VoteEvent,
    VoteResultEvent,
)

logger = logging.getLogger(__name__)


class WebEventListener:
    """Game event listener that serialises events to an asyncio queue.

    The queue is consumed by the WebSocket broadcaster.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._names: dict[str, str] = {}
        self._models: dict[str, str] = {}
        self._roles: dict[str, str] = {}
        self._teams: dict[str, str] = {}
        self._game_number: int = 0

    # Called by GameRunner before game starts
    def set_game_info(
        self,
        game_number: int,
        names: dict[str, str],
        models: dict[str, str],
        roles: dict[str, str],
        teams: dict[str, str],
    ) -> None:
        self._game_number = game_number
        self._names = dict(names)
        self._models = dict(models)
        self._roles = dict(roles)
        self._teams = dict(teams)

        # Emit a synthetic game_start event
        players = []
        for pid in names:
            players.append(
                {
                    "player_id": pid,
                    "name": names[pid],
                    "model": models.get(pid, ""),
                    "role": roles.get(pid, ""),
                    "team": teams.get(pid, ""),
                }
            )
        msg = json.dumps(
            {
                "type": "game_start",
                "game_number": game_number,
                "players": players,
            }
        )
        self.queue.put_nowait(msg)

    def __call__(self, event: GameEvent) -> None:
        d = self._event_to_dict(event)
        if d is not None:
            self.queue.put_nowait(json.dumps(d))

    def _event_to_dict(self, event: GameEvent) -> dict[str, Any] | None:
        """Convert a game event dataclass to a JSON-friendly dict."""
        d: dict[str, Any] = {"type": type(event).__name__}

        for f in dataclasses.fields(event):
            val = getattr(event, f.name)
            if isinstance(val, Enum):
                val = val.name
            d[f.name] = val

        # Enrich with player names
        if isinstance(event, ReasoningEvent):
            d["player_name"] = self._names.get(event.player_id, event.player_id)
            d["player_model"] = self._models.get(event.player_id, "")
        elif isinstance(event, SpeechEvent):
            d["player_name"] = self._names.get(event.player_id, event.player_id)
            d["player_model"] = self._models.get(event.player_id, "")
        elif isinstance(event, VoteEvent):
            d["voter_name"] = self._names.get(event.voter_id, event.voter_id)
            d["target_name"] = (
                self._names.get(event.target_id, event.target_id)
                if event.target_id
                else None
            )
        elif isinstance(event, VoteResultEvent):
            d["tally_named"] = {
                self._names.get(pid, pid): cnt
                for pid, cnt in event.tally.items()
            }
            d["eliminated_name"] = (
                self._names.get(event.eliminated_id, event.eliminated_id)
                if event.eliminated_id
                else None
            )
        elif isinstance(event, EliminationEvent):
            d["player_name"] = self._names.get(event.player_id, event.player_id)
        elif isinstance(event, NightResultEvent):
            d["kill_names"] = [self._names.get(k, k) for k in event.kills]
            d["saved_names"] = [self._names.get(s, s) for s in event.saved]
        elif isinstance(event, GameEndEvent):
            d["winner_names"] = [self._names.get(w, w) for w in event.winners]

        d["game_number"] = self._game_number
        return d


# ------------------------------------------------------------------
# Servers
# ------------------------------------------------------------------

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wolf Live</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px;padding:16px}
h1{color:#58a6ff;font-size:18px;margin-bottom:8px}
#status{color:#8b949e;font-size:11px;margin-bottom:12px}
.player-table{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:12px}
.player-table th{text-align:left;color:#8b949e;border-bottom:1px solid #21262d;padding:4px 8px}
.player-table td{padding:4px 8px;border-bottom:1px solid #161b22}
.team-werewolf{color:#f85149}
.team-village{color:#3fb950}
.phase{color:#d2a8ff;font-weight:bold;margin:12px 0 4px;font-size:14px;border-top:1px solid #21262d;padding-top:8px}
.speech{margin:2px 0;line-height:1.4}
.speech .name{color:#58a6ff;font-weight:bold}
.speech .model-tag{color:#8b949e;font-size:11px}
.speech .text{color:#c9d1d9}
.wolf-chat{margin:2px 0;line-height:1.4;border-left:3px solid #f85149;padding-left:8px;background:#1a0a0a}
.wolf-chat .tag{color:#f85149;font-size:10px;font-weight:bold}
.wolf-chat .name{color:#ff7b72;font-weight:bold}
.wolf-chat .text{color:#ffa198}
.reasoning{margin:2px 0;line-height:1.4;border-left:2px solid #30363d;padding-left:8px}
.reasoning .name{color:#8b949e;font-weight:bold;font-size:11px}
.reasoning .model-tag{color:#484f58;font-size:10px}
.reasoning .text{color:#8b949e;font-size:12px}
.reasoning-toggle{color:#484f58;cursor:pointer;font-size:11px;user-select:none}
.reasoning-body{display:none;white-space:pre-wrap;max-height:300px;overflow-y:auto}
.vote{color:#d29922;margin:1px 0}
.vote-result{color:#f0883e;margin:4px 0;font-weight:bold}
.elimination{color:#f85149;font-weight:bold;margin:4px 0}
.night-result{color:#a5d6ff;margin:4px 0}
.game-end{background:#161b22;border:2px solid #3fb950;border-radius:8px;padding:16px;margin:16px 0;text-align:center;font-size:16px}
.game-end.werewolf-win{border-color:#f85149}
.game-end .winner{font-size:24px;font-weight:bold}
#log{max-height:calc(100vh - 60px);overflow-y:auto}
</style>
</head>
<body>
<h1>Wolf Live</h1>
<div id="status">Connecting...</div>
<div id="log"></div>
<script>
const WS_PORT = window.__WS_PORT__ || (location.port ? parseInt(location.port)+1 : 8766);
const log = document.getElementById('log');
const status = document.getElementById('status');
let ws;
function connect() {
  ws = new WebSocket('ws://' + (location.hostname || 'localhost') + ':' + WS_PORT);
  ws.onopen = () => { status.textContent = 'Connected'; status.style.color = '#3fb950'; };
  ws.onclose = () => { status.textContent = 'Disconnected — reconnecting...'; status.style.color = '#f85149'; setTimeout(connect, 2000); };
  ws.onerror = () => { ws.close(); };
  ws.onmessage = (msg) => {
    const d = JSON.parse(msg.data);
    render(d);
    log.scrollTop = log.scrollHeight;
  };
}
function esc(s) { const t = document.createElement('span'); t.textContent = s; return t.innerHTML; }
function render(d) {
  const t = d.type;
  if (t === 'game_start') {
    let h = '<div class="phase">Game ' + d.game_number + '</div>';
    h += '<table class="player-table"><tr><th>Player</th><th>Model</th><th>Role</th><th>Team</th></tr>';
    d.players.forEach(p => {
      const tc = p.team === 'werewolf' ? 'team-werewolf' : 'team-village';
      h += '<tr><td>' + esc(p.name) + '</td><td>' + esc(p.model) + '</td><td class="'+tc+'">' + esc(p.role) + '</td><td class="'+tc+'">' + esc(p.team) + '</td></tr>';
    });
    h += '</table>';
    log.insertAdjacentHTML('beforeend', h);
  } else if (t === 'PhaseChangeEvent') {
    const labels = {NIGHT:'Night '+d.day, DAWN:'Dawn', DAY_DISCUSSION:'Day '+d.day+' Discussion', DAY_VOTE:'Day '+d.day+' Vote'};
    const label = labels[d.new_phase];
    if (label) log.insertAdjacentHTML('beforeend', '<div class="phase">' + esc(label) + '</div>');
  } else if (t === 'ReasoningEvent') {
    const short = d.player_model ? d.player_model.split(':')[0] : '';
    let text = d.reasoning || '';
    const preview = text;
    const id = 'r' + Math.random().toString(36).slice(2,9);
    log.insertAdjacentHTML('beforeend',
      '<div class="reasoning"><span class="name">' + esc(d.player_name) + '</span> <span class="model-tag">(' + esc(short) + ')</span> ' +
      '<span class="reasoning-toggle" onclick="var b=document.getElementById(\'' + id + '\');b.style.display=b.style.display===\'block\'?\'none\':\'block\'">[reasoning ▸]</span>' +
      '<div class="reasoning-body" id="' + id + '">' + esc(text) + '</div>' +
      '<div class="text">' + esc(preview) + '</div></div>');
  } else if (t === 'SpeechEvent') {
    const short = d.player_model ? d.player_model.split(':')[0] : '';
    let text = d.content || '';
    // Full text, no truncation
    if (d.channel === 'wolf') {
      log.insertAdjacentHTML('beforeend',
        '<div class="wolf-chat"><span class="tag">[WOLF] </span><span class="name">' + esc(d.player_name) + '</span> <span class="text">' + esc(text) + '</span></div>');
    } else {
      log.insertAdjacentHTML('beforeend',
        '<div class="speech"><span class="name">' + esc(d.player_name) + '</span> <span class="model-tag">(' + esc(short) + ')</span> <span class="text">' + esc(text) + '</span></div>');
    }
  } else if (t === 'VoteEvent') {
    const target = d.target_name ? d.target_name : 'abstain';
    log.insertAdjacentHTML('beforeend', '<div class="vote">' + esc(d.voter_name) + ' → ' + esc(target) + '</div>');
  } else if (t === 'VoteResultEvent') {
    const parts = Object.entries(d.tally_named || {}).sort((a,b) => b[1]-a[1]).map(([n,c]) => n+':'+c);
    let h = '<div class="vote-result">Tally: [' + esc(parts.join(', ')) + ']';
    if (d.tie) h += ' TIE';
    else if (d.eliminated_name) h += ' — ELIMINATED: ' + esc(d.eliminated_name);
    h += '</div>';
    log.insertAdjacentHTML('beforeend', h);
  } else if (t === 'EliminationEvent') {
    log.insertAdjacentHTML('beforeend',
      '<div class="elimination">KILL ' + esc(d.player_name) + ' (role=' + esc(d.role) + ', cause=' + esc(d.cause) + ')</div>');
  } else if (t === 'NightResultEvent') {
    let parts = [];
    if (d.kill_names && d.kill_names.length) parts.push('killed=[' + d.kill_names.map(esc).join(',') + ']');
    if (d.saved_names && d.saved_names.length) parts.push('saved=[' + d.saved_names.map(esc).join(',') + ']');
    if (parts.length) log.insertAdjacentHTML('beforeend', '<div class="night-result">Night: ' + parts.join(' ') + '</div>');
    else log.insertAdjacentHTML('beforeend', '<div class="night-result">Night: no kills</div>');
  } else if (t === 'GameEndEvent') {
    const cls = d.winning_team === 'werewolf' ? 'werewolf-win' : '';
    const color = d.winning_team === 'village' ? '#3fb950' : '#f85149';
    log.insertAdjacentHTML('beforeend',
      '<div class="game-end ' + cls + '"><div class="winner" style="color:'+color+'">' + esc(d.winning_team.toUpperCase()) + ' WINS</div><div>' + esc(d.reason) + '</div></div>');
  }
}
connect();
</script>
</body>
</html>"""


def _make_http_handler(html: str) -> type:
    """Create an HTTP request handler that serves the embedded HTML."""

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass  # suppress access logs

    return Handler


async def start_web_server(
    listener: WebEventListener,
    http_port: int = 8080,
    ws_port: int = 8765,
) -> None:
    """Start HTTP + WebSocket servers.  Runs forever (call as asyncio task)."""

    # --- HTTP server in a daemon thread ---
    # Inject the actual WS port into the page
    html = _HTML_PAGE.replace(
        "window.__WS_PORT__", str(ws_port)
    )
    handler_cls = _make_http_handler(html)
    httpd = HTTPServer(("0.0.0.0", http_port), handler_cls)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()
    logger.info("HTTP server on http://0.0.0.0:%d", http_port)

    # --- WebSocket broadcaster ---
    connections: set[Any] = set()

    async def ws_handler(websocket: Any) -> None:
        connections.add(websocket)
        try:
            async for _ in websocket:
                pass  # we only send, ignore incoming
        finally:
            connections.discard(websocket)

    async with ws_serve(ws_handler, "0.0.0.0", ws_port):
        logger.info("WebSocket server on ws://0.0.0.0:%d", ws_port)
        while True:
            msg = await listener.queue.get()
            dead: set[Any] = set()
            for ws in connections:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            connections -= dead
