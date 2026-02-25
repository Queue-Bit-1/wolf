"""Tests for wolf.engine.state -- PlayerSlot, GameState, and GameStateView."""

from __future__ import annotations

import pytest

from wolf.engine.actions import UseAbilityAction
from wolf.engine.events import (
    GameEvent,
    PhaseChangeEvent,
    PrivateRevealEvent,
    SpeechEvent,
)
from wolf.engine.phase import Phase
from wolf.engine.state import GameState, GameStateView, PlayerSlot


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def sample_players() -> tuple[PlayerSlot, ...]:
    """A small set of players for testing."""
    return (
        PlayerSlot(player_id="p1", name="Alice", role="seer", team="village"),
        PlayerSlot(player_id="p2", name="Bob", role="werewolf", team="werewolf"),
        PlayerSlot(player_id="p3", name="Charlie", role="villager", team="village"),
        PlayerSlot(player_id="p4", name="Diana", role="doctor", team="village"),
    )


@pytest.fixture
def game_state(sample_players: tuple[PlayerSlot, ...]) -> GameState:
    """A game state populated with sample players."""
    return GameState(
        day=1,
        phase=Phase.DAY_DISCUSSION,
        players=sample_players,
    )


# ======================================================================
# PlayerSlot tests
# ======================================================================


class TestPlayerSlot:
    """Tests for the PlayerSlot frozen dataclass."""

    def test_creation_with_required_fields(self) -> None:
        slot = PlayerSlot(player_id="p1", name="Alice", role="seer", team="village")
        assert slot.player_id == "p1"
        assert slot.name == "Alice"
        assert slot.role == "seer"
        assert slot.team == "village"

    def test_defaults(self) -> None:
        slot = PlayerSlot(player_id="p1", name="Alice", role="seer", team="village")
        assert slot.is_alive is True
        assert slot.metadata == {}

    def test_frozen(self) -> None:
        slot = PlayerSlot(player_id="p1", name="Alice", role="seer", team="village")
        with pytest.raises(AttributeError):
            slot.name = "Bob"  # type: ignore[misc]

    def test_frozen_is_alive(self) -> None:
        slot = PlayerSlot(player_id="p1", name="Alice", role="seer", team="village")
        with pytest.raises(AttributeError):
            slot.is_alive = False  # type: ignore[misc]


# ======================================================================
# GameState query tests
# ======================================================================


class TestGameStateQueries:
    """Tests for query methods on GameState."""

    def test_get_player_found(self, game_state: GameState) -> None:
        player = game_state.get_player("p1")
        assert player is not None
        assert player.name == "Alice"
        assert player.role == "seer"

    def test_get_player_not_found(self, game_state: GameState) -> None:
        assert game_state.get_player("nonexistent") is None

    def test_get_alive_players_all_alive(self, game_state: GameState) -> None:
        alive = game_state.get_alive_players()
        assert len(alive) == 4
        assert all(p.is_alive for p in alive)

    def test_get_alive_players_after_kill(self, game_state: GameState) -> None:
        new_state = game_state.with_player_killed("p2")
        alive = new_state.get_alive_players()
        assert len(alive) == 3
        alive_ids = [p.player_id for p in alive]
        assert "p2" not in alive_ids

    def test_get_alive_player_ids(self, game_state: GameState) -> None:
        ids = game_state.get_alive_player_ids()
        assert ids == ["p1", "p2", "p3", "p4"]

    def test_get_alive_player_ids_after_kill(self, game_state: GameState) -> None:
        new_state = game_state.with_player_killed("p3")
        ids = new_state.get_alive_player_ids()
        assert "p3" not in ids
        assert len(ids) == 3

    def test_get_players_by_role(self, game_state: GameState) -> None:
        wolves = game_state.get_players_by_role("werewolf")
        assert len(wolves) == 1
        assert wolves[0].player_id == "p2"

    def test_get_players_by_role_not_found(self, game_state: GameState) -> None:
        result = game_state.get_players_by_role("witch")
        assert result == []

    def test_get_players_by_team_village(self, game_state: GameState) -> None:
        villagers = game_state.get_players_by_team("village")
        assert len(villagers) == 3
        assert all(p.team == "village" for p in villagers)

    def test_get_players_by_team_werewolf(self, game_state: GameState) -> None:
        wolves = game_state.get_players_by_team("werewolf")
        assert len(wolves) == 1
        assert wolves[0].player_id == "p2"


# ======================================================================
# GameState immutable update tests
# ======================================================================


class TestGameStateImmutableUpdates:
    """Tests for immutable state transitions on GameState."""

    def test_with_phase(self, game_state: GameState) -> None:
        new_state = game_state.with_phase(Phase.NIGHT)
        assert new_state.phase == Phase.NIGHT
        # Original unchanged
        assert game_state.phase == Phase.DAY_DISCUSSION

    def test_with_day(self, game_state: GameState) -> None:
        new_state = game_state.with_day(5)
        assert new_state.day == 5
        assert game_state.day == 1

    def test_with_player_killed(self, game_state: GameState) -> None:
        new_state = game_state.with_player_killed("p2")
        # New state: player is dead
        killed = new_state.get_player("p2")
        assert killed is not None
        assert killed.is_alive is False
        # Original: player still alive
        original = game_state.get_player("p2")
        assert original is not None
        assert original.is_alive is True

    def test_with_player_killed_preserves_others(self, game_state: GameState) -> None:
        new_state = game_state.with_player_killed("p2")
        for pid in ("p1", "p3", "p4"):
            player = new_state.get_player(pid)
            assert player is not None
            assert player.is_alive is True

    def test_with_night_action(self, game_state: GameState) -> None:
        action = UseAbilityAction(
            player_id="p1", ability_name="investigate", target_id="p2"
        )
        new_state = game_state.with_night_action("p1", action)
        assert "p1" in new_state.night_actions
        assert new_state.night_actions["p1"] is action
        # Original unchanged
        assert "p1" not in game_state.night_actions

    def test_with_event(self, game_state: GameState) -> None:
        event = SpeechEvent(day=1, phase=Phase.DAY_DISCUSSION, player_id="p1", content="hi")
        new_state = game_state.with_event(event)
        assert len(new_state.events) == 1
        assert new_state.events[0] is event
        # Original unchanged
        assert len(game_state.events) == 0

    def test_with_event_appends(self, game_state: GameState) -> None:
        event1 = SpeechEvent(day=1, phase=Phase.DAY_DISCUSSION, player_id="p1", content="first")
        event2 = SpeechEvent(day=1, phase=Phase.DAY_DISCUSSION, player_id="p2", content="second")
        state = game_state.with_event(event1).with_event(event2)
        assert len(state.events) == 2
        assert state.events[0].content == "first"
        assert state.events[1].content == "second"

    def test_clear_night_actions(self, game_state: GameState) -> None:
        action = UseAbilityAction(
            player_id="p1", ability_name="investigate", target_id="p2"
        )
        state_with_actions = game_state.with_night_action("p1", action)
        assert len(state_with_actions.night_actions) == 1

        cleared = state_with_actions.clear_night_actions()
        assert len(cleared.night_actions) == 0
        # Original with-actions unchanged
        assert len(state_with_actions.night_actions) == 1

    def test_original_state_not_modified_after_chain(self, game_state: GameState) -> None:
        """Verify that chaining multiple mutations does not modify the original."""
        original_day = game_state.day
        original_phase = game_state.phase
        original_alive_count = len(game_state.get_alive_players())
        original_event_count = len(game_state.events)

        _ = (
            game_state
            .with_day(99)
            .with_phase(Phase.GAME_OVER)
            .with_player_killed("p1")
            .with_event(SpeechEvent(day=99, phase=Phase.GAME_OVER, player_id="p1", content="x"))
        )

        assert game_state.day == original_day
        assert game_state.phase == original_phase
        assert len(game_state.get_alive_players()) == original_alive_count
        assert len(game_state.events) == original_event_count


# ======================================================================
# GameStateView tests
# ======================================================================


class TestGameStateView:
    """Tests for the filtered GameStateView."""

    def test_day_and_phase(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        assert view.day == 1
        assert view.phase == Phase.DAY_DISCUSSION

    def test_my_player_has_role(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        my = view.my_player
        assert my.player_id == "p1"
        assert my.role == "seer"
        assert my.team == "village"

    def test_my_player_not_found_raises(self) -> None:
        state = GameState(day=0, phase=Phase.SETUP, players=())
        view = GameStateView(state, "ghost")
        with pytest.raises(ValueError, match="ghost"):
            _ = view.my_player

    def test_alive_players_hides_other_roles(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        alive = view.alive_players
        for p in alive:
            if p.player_id == "p1":
                # Own player: role is visible
                assert p.role == "seer"
                assert p.team == "village"
            else:
                # Other players: role hidden
                assert p.role == "unknown"
                assert p.team == "unknown"
                assert p.metadata == {}

    def test_alive_players_count(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        assert len(view.alive_players) == 4

    def test_all_players_structure(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        all_p = view.all_players
        assert len(all_p) == 4
        for pid, name, is_alive in all_p:
            assert isinstance(pid, str)
            assert isinstance(name, str)
            assert isinstance(is_alive, bool)

    def test_all_players_contains_no_role_info(self, game_state: GameState) -> None:
        view = GameStateView(game_state, "p1")
        all_p = view.all_players
        # Tuples are (player_id, name, is_alive) -- no role field
        for entry in all_p:
            assert len(entry) == 3

    def test_events_filtering_private_reveal_visible_to_target(
        self, game_state: GameState
    ) -> None:
        """PrivateRevealEvent should only be visible to its target player."""
        private_event = PrivateRevealEvent(
            day=1, phase=Phase.DAWN, player_id="p1", info="p2 is werewolf"
        )
        public_event = SpeechEvent(
            day=1, phase=Phase.DAY_DISCUSSION, player_id="p3", content="hello"
        )
        state = game_state.with_event(private_event).with_event(public_event)

        # p1 should see both events
        view_p1 = GameStateView(state, "p1")
        assert len(view_p1.events) == 2

        # p2 should only see the public event
        view_p2 = GameStateView(state, "p2")
        assert len(view_p2.events) == 1
        assert isinstance(view_p2.events[0], SpeechEvent)

    def test_events_filtering_non_private_always_visible(
        self, game_state: GameState
    ) -> None:
        """Non-private events are visible to all players."""
        phase_event = PhaseChangeEvent(
            day=1,
            phase=Phase.DAY_DISCUSSION,
            old_phase=Phase.DAWN,
            new_phase=Phase.DAY_DISCUSSION,
        )
        state = game_state.with_event(phase_event)

        for pid in ("p1", "p2", "p3", "p4"):
            view = GameStateView(state, pid)
            assert len(view.events) == 1
