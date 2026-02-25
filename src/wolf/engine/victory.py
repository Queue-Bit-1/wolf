"""Victory condition checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolf.engine.events import GameEndEvent

if TYPE_CHECKING:
    from wolf.engine.state import GameState


def check_victory(state: GameState) -> GameEndEvent | None:
    """Return a ``GameEndEvent`` if the game is over, otherwise ``None``.

    Victory conditions:
    * **Village wins** -- all werewolf-team players are dead.
    * **Werewolf wins** -- alive werewolves >= alive villagers.
    """
    alive = state.get_alive_players()

    alive_wolves = [p for p in alive if p.team == "werewolf"]
    alive_villagers = [p for p in alive if p.team == "village"]

    if not alive_wolves:
        # All wolves dead -- village wins.
        village_members = state.get_players_by_team("village")
        return GameEndEvent(
            day=state.day,
            phase=state.phase,
            winning_team="village",
            winners=[p.player_id for p in village_members],
            reason="All werewolves have been eliminated.",
        )

    if len(alive_wolves) >= len(alive_villagers):
        # Wolves equal or outnumber villagers -- werewolf wins.
        wolf_members = state.get_players_by_team("werewolf")
        return GameEndEvent(
            day=state.day,
            phase=state.phase,
            winning_team="werewolf",
            winners=[p.player_id for p in wolf_members],
            reason="Werewolves equal or outnumber the villagers.",
        )

    return None
