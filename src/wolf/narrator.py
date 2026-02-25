"""Live benchmark output -- compact, data-dense console logging."""

from __future__ import annotations

import time

from wolf.engine.events import (
    EliminationEvent,
    GameEndEvent,
    GameEvent,
    NightResultEvent,
    PhaseChangeEvent,
    PrivateRevealEvent,
    SpeechEvent,
    VoteEvent,
    VoteResultEvent,
)
from wolf.engine.phase import Phase


# ANSI
_B = "\033[1m"
_D = "\033[2m"
_R = "\033[91m"
_G = "\033[92m"
_Y = "\033[93m"
_C = "\033[96m"
_M = "\033[95m"
_0 = "\033[0m"

# Player info (set externally)
_player_models: dict[str, str] = {}
_player_names: dict[str, str] = {}
_player_roles: dict[str, str] = {}
_game_start: float = 0.0


def set_player_info(
    names: dict[str, str],
    models: dict[str, str],
    roles: dict[str, str],
) -> None:
    """Set player display info."""
    global _game_start
    _player_names.update(names)
    _player_models.update(models)
    _player_roles.update(roles)
    _game_start = time.monotonic()


def _tag(player_id: str) -> str:
    """Compact player tag: Name (model)."""
    name = _player_names.get(player_id, player_id)
    model = _player_models.get(player_id, "")
    if model:
        short = model.split(":")[0]
        return f"{name}({short})"
    return name


def _elapsed() -> str:
    if _game_start:
        return f"{time.monotonic() - _game_start:.0f}s"
    return ""


def _out(msg: str) -> None:
    t = _elapsed()
    prefix = f"{_D}[{t:>5s}]{_0} " if t else ""
    print(f"{prefix}{msg}", flush=True)


class Narrator:
    """Event listener: benchmark-style compact output."""

    def __call__(self, event: GameEvent) -> None:
        if isinstance(event, PhaseChangeEvent):
            self._phase(event)
        elif isinstance(event, SpeechEvent):
            self._speech(event)
        elif isinstance(event, VoteEvent):
            self._vote(event)
        elif isinstance(event, VoteResultEvent):
            self._vote_result(event)
        elif isinstance(event, EliminationEvent):
            self._elimination(event)
        elif isinstance(event, NightResultEvent):
            self._night_result(event)
        elif isinstance(event, PrivateRevealEvent):
            self._reveal(event)
        elif isinstance(event, GameEndEvent):
            self._game_end(event)

    def _phase(self, e: PhaseChangeEvent) -> None:
        labels = {
            Phase.NIGHT: f"{_B}--- NIGHT {e.day} ---{_0}",
            Phase.DAWN: f"{_D}--- dawn ---{_0}",
            Phase.DAY_DISCUSSION: f"{_B}--- DAY {e.day} DISCUSSION ---{_0}",
            Phase.DAY_VOTE: f"{_B}--- DAY {e.day} VOTE ---{_0}",
        }
        label = labels.get(e.new_phase)
        if label:
            _out(label)

    def _speech(self, e: SpeechEvent) -> None:
        tag = _tag(e.player_id)
        text = e.content
        if len(text) > 300:
            text = text[:300] + "..."
        _out(f"  {_C}{tag}{_0}: {text}")

    def _vote(self, e: VoteEvent) -> None:
        voter = _tag(e.voter_id)
        if e.target_id:
            target = _tag(e.target_id)
            _out(f"  {voter} -> {_Y}{target}{_0}")
        else:
            _out(f"  {voter} -> {_D}abstain{_0}")

    def _vote_result(self, e: VoteResultEvent) -> None:
        tally_str = ", ".join(
            f"{_player_names.get(pid, pid)}:{cnt}"
            for pid, cnt in sorted(e.tally.items(), key=lambda x: -x[1])
        )
        if e.tie:
            _out(f"  tally=[{tally_str}] {_Y}TIE{_0}")
        elif e.eliminated_id:
            name = _player_names.get(e.eliminated_id, e.eliminated_id)
            _out(f"  tally=[{tally_str}] {_R}ELIMINATED: {name}{_0}")
        else:
            _out(f"  tally=[{tally_str}] no elimination")

    def _elimination(self, e: EliminationEvent) -> None:
        tag = _tag(e.player_id)
        _out(f"  {_R}KILL {tag} role={e.role} cause={e.cause}{_0}")

    def _night_result(self, e: NightResultEvent) -> None:
        parts = []
        if e.kills:
            kill_names = [_player_names.get(k, k) for k in e.kills]
            parts.append(f"killed=[{','.join(kill_names)}]")
        if e.saved:
            save_names = [_player_names.get(s, s) for s in e.saved]
            parts.append(f"saved=[{','.join(save_names)}]")
        if e.protected:
            parts.append(f"protected={len(e.protected)}")
        if parts:
            _out(f"  night: {' '.join(parts)}")
        else:
            _out(f"  night: no kills")

    def _reveal(self, e: PrivateRevealEvent) -> None:
        tag = _tag(e.player_id)
        _out(f"  {_D}reveal({tag}): {e.info}{_0}")

    def _game_end(self, e: GameEndEvent) -> None:
        winner = e.winning_team.upper()
        color = _G if e.winning_team == "village" else _R
        _out(f"\n{_B}{'='*50}{_0}")
        _out(f"{_B}{color}  RESULT: {winner} WINS{_0}")
        _out(f"  {e.reason}")
        # Role reveal table
        _out(f"  {'─'*46}")
        _out(f"  {'Player':<12} {'Model':<18} {'Role':<12} {'Status'}")
        _out(f"  {'─'*46}")
        for pid in _player_names:
            name = _player_names.get(pid, pid)
            model = _player_models.get(pid, "?").split(":")[0]
            role = _player_roles.get(pid, "?")
            # Check if player survived -- look at winners list
            survived = pid in e.winners if e.winners else "?"
            _out(f"  {name:<12} {model:<18} {role:<12} {'alive' if survived else 'dead'}")
        _out(f"{_B}{'='*50}{_0}")
