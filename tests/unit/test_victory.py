"""Tests for wolf.engine.victory -- victory condition checks."""

from __future__ import annotations

import pytest

from wolf.engine.events import GameEndEvent
from wolf.engine.phase import Phase
from wolf.engine.state import GameState, PlayerSlot
from wolf.engine.victory import check_victory


# ======================================================================
# Helpers
# ======================================================================


def _make_state(
    alive_village: int,
    alive_wolves: int,
    dead_village: int = 0,
    dead_wolves: int = 0,
) -> GameState:
    """Build a GameState with the specified player counts."""
    players: list[PlayerSlot] = []
    idx = 0

    for _ in range(alive_village):
        players.append(
            PlayerSlot(
                player_id=f"v{idx}",
                name=f"Villager{idx}",
                role="villager",
                team="village",
                is_alive=True,
            )
        )
        idx += 1

    for _ in range(alive_wolves):
        players.append(
            PlayerSlot(
                player_id=f"w{idx}",
                name=f"Wolf{idx}",
                role="werewolf",
                team="werewolf",
                is_alive=True,
            )
        )
        idx += 1

    for _ in range(dead_village):
        players.append(
            PlayerSlot(
                player_id=f"dv{idx}",
                name=f"DeadVillager{idx}",
                role="villager",
                team="village",
                is_alive=False,
            )
        )
        idx += 1

    for _ in range(dead_wolves):
        players.append(
            PlayerSlot(
                player_id=f"dw{idx}",
                name=f"DeadWolf{idx}",
                role="werewolf",
                team="werewolf",
                is_alive=False,
            )
        )
        idx += 1

    return GameState(day=3, phase=Phase.DAWN, players=tuple(players))


# ======================================================================
# Tests
# ======================================================================


class TestVillageWins:
    """Village wins when all werewolves are dead."""

    def test_all_wolves_dead(self) -> None:
        state = _make_state(alive_village=3, alive_wolves=0, dead_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert isinstance(result, GameEndEvent)
        assert result.winning_team == "village"

    def test_village_wins_with_one_villager(self) -> None:
        state = _make_state(alive_village=1, alive_wolves=0, dead_wolves=1)
        result = check_victory(state)
        assert result is not None
        assert result.winning_team == "village"

    def test_village_winners_list(self) -> None:
        state = _make_state(alive_village=2, alive_wolves=0, dead_wolves=1, dead_village=1)
        result = check_victory(state)
        assert result is not None
        # Winners include all village-team players (alive + dead)
        village_players = state.get_players_by_team("village")
        assert set(result.winners) == {p.player_id for p in village_players}

    def test_village_wins_reason_text(self) -> None:
        state = _make_state(alive_village=3, alive_wolves=0, dead_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert "werewolves" in result.reason.lower() or "eliminated" in result.reason.lower()


class TestWerewolfWins:
    """Werewolf wins when alive wolves >= alive villagers."""

    def test_wolves_outnumber_villagers(self) -> None:
        state = _make_state(alive_village=1, alive_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert result.winning_team == "werewolf"

    def test_wolves_equal_villagers(self) -> None:
        """Exactly equal counts: wolves win (>= condition)."""
        state = _make_state(alive_village=2, alive_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert result.winning_team == "werewolf"

    def test_one_wolf_one_villager(self) -> None:
        state = _make_state(alive_village=1, alive_wolves=1)
        result = check_victory(state)
        assert result is not None
        assert result.winning_team == "werewolf"

    def test_werewolf_winners_list(self) -> None:
        state = _make_state(alive_village=1, alive_wolves=2, dead_wolves=0)
        result = check_victory(state)
        assert result is not None
        wolf_players = state.get_players_by_team("werewolf")
        assert set(result.winners) == {p.player_id for p in wolf_players}

    def test_werewolf_wins_reason_text(self) -> None:
        state = _make_state(alive_village=1, alive_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert "werewolves" in result.reason.lower() or "outnumber" in result.reason.lower()


class TestGameContinues:
    """Game continues when wolves are alive but outnumbered by villagers."""

    def test_more_villagers_than_wolves(self) -> None:
        state = _make_state(alive_village=3, alive_wolves=1)
        result = check_victory(state)
        assert result is None

    def test_many_more_villagers(self) -> None:
        state = _make_state(alive_village=5, alive_wolves=1)
        result = check_victory(state)
        assert result is None

    def test_two_villagers_one_wolf(self) -> None:
        state = _make_state(alive_village=2, alive_wolves=1)
        result = check_victory(state)
        assert result is None

    def test_continues_with_dead_players(self) -> None:
        state = _make_state(
            alive_village=3, alive_wolves=1, dead_village=2, dead_wolves=1
        )
        result = check_victory(state)
        assert result is None


class TestEdgeCases:
    """Edge cases and unusual compositions."""

    def test_no_players_at_all(self) -> None:
        """With no players, wolves list is empty -- village wins by default."""
        state = GameState(day=1, phase=Phase.DAWN, players=())
        result = check_victory(state)
        assert result is not None
        # No wolves = village wins
        assert result.winning_team == "village"

    def test_mixed_roles_on_village_team(self) -> None:
        """Seer, doctor, villager all count as village team."""
        players = (
            PlayerSlot(player_id="s1", name="Seer", role="seer", team="village", is_alive=True),
            PlayerSlot(player_id="d1", name="Doctor", role="doctor", team="village", is_alive=True),
            PlayerSlot(player_id="v1", name="Villager", role="villager", team="village", is_alive=True),
            PlayerSlot(player_id="w1", name="Wolf", role="werewolf", team="werewolf", is_alive=True),
        )
        state = GameState(day=2, phase=Phase.DAWN, players=players)
        # 3 village vs 1 wolf -- game continues
        result = check_victory(state)
        assert result is None

    def test_day_and_phase_propagated_in_result(self) -> None:
        state = _make_state(alive_village=3, alive_wolves=0, dead_wolves=2)
        result = check_victory(state)
        assert result is not None
        assert result.day == state.day
        assert result.phase == state.phase
