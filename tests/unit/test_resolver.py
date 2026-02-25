"""Tests for wolf.engine.resolver -- night-action resolution logic."""

from __future__ import annotations

import pytest

from wolf.engine.actions import NoAction, UseAbilityAction
from wolf.engine.events import (
    EliminationEvent,
    NightResultEvent,
    PrivateRevealEvent,
)
from wolf.engine.phase import Phase
from wolf.engine.resolver import resolve_night
from wolf.engine.state import GameState, PlayerSlot
from wolf.roles.registry import RoleRegistry


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def registry() -> RoleRegistry:
    """Return the role registry (auto-discovered on import)."""
    return RoleRegistry


@pytest.fixture
def five_player_state() -> GameState:
    """A 5-player game: 2 werewolves, 1 seer, 1 doctor, 1 villager."""
    return GameState(
        day=1,
        phase=Phase.NIGHT,
        players=(
            PlayerSlot(player_id="wolf1", name="Wolf1", role="werewolf", team="werewolf"),
            PlayerSlot(player_id="wolf2", name="Wolf2", role="werewolf", team="werewolf"),
            PlayerSlot(player_id="seer1", name="Seer", role="seer", team="village"),
            PlayerSlot(player_id="doc1", name="Doctor", role="doctor", team="village"),
            PlayerSlot(player_id="vil1", name="Villager", role="villager", team="village"),
        ),
    )


# ======================================================================
# Tests
# ======================================================================


class TestResolveNightEmpty:
    """When no night actions are submitted, nothing should happen."""

    def test_no_actions_produces_empty_night_result(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        new_state, events = resolve_night(five_player_state, registry)
        # Should still produce a NightResultEvent summary
        night_results = [e for e in events if isinstance(e, NightResultEvent)]
        assert len(night_results) == 1
        assert night_results[0].kills == []
        assert night_results[0].protected == []
        assert night_results[0].saved == []

    def test_no_actions_no_one_dies(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        new_state, events = resolve_night(five_player_state, registry)
        assert len(new_state.get_alive_players()) == 5


class TestWolfKill:
    """Werewolf kill action targeting a villager."""

    def test_single_wolf_kill(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        state = five_player_state.with_night_action("wolf1", kill_action)

        new_state, events = resolve_night(state, registry)

        # Villager should be dead
        target = new_state.get_player("vil1")
        assert target is not None
        assert target.is_alive is False

    def test_kill_produces_elimination_event(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        state = five_player_state.with_night_action("wolf1", kill_action)

        _, events = resolve_night(state, registry)

        elim_events = [e for e in events if isinstance(e, EliminationEvent)]
        assert len(elim_events) == 1
        assert elim_events[0].player_id == "vil1"
        assert elim_events[0].cause == "wolf_kill"
        assert elim_events[0].role == "villager"

    def test_kill_night_result_lists_kill(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        state = five_player_state.with_night_action("wolf1", kill_action)

        _, events = resolve_night(state, registry)

        night_result = [e for e in events if isinstance(e, NightResultEvent)][0]
        assert "vil1" in night_result.kills
        assert night_result.saved == []


class TestDoctorProtect:
    """Doctor protection interacting with wolf kills."""

    def test_doctor_protects_same_target_as_wolf(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        """Target is saved when doctor protects the wolf's kill target."""
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="seer1"
        )
        protect_action = UseAbilityAction(
            player_id="doc1", ability_name="protect", target_id="seer1"
        )
        state = (
            five_player_state
            .with_night_action("wolf1", kill_action)
            .with_night_action("doc1", protect_action)
        )

        new_state, events = resolve_night(state, registry)

        # Seer should survive
        seer = new_state.get_player("seer1")
        assert seer is not None
        assert seer.is_alive is True

        # Night result should show saved
        night_result = [e for e in events if isinstance(e, NightResultEvent)][0]
        assert "seer1" in night_result.saved
        assert "seer1" in night_result.protected
        assert night_result.kills == []

        # No elimination events
        elim_events = [e for e in events if isinstance(e, EliminationEvent)]
        assert len(elim_events) == 0

    def test_doctor_protects_different_target(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        """Kill succeeds when doctor protects a different player."""
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        protect_action = UseAbilityAction(
            player_id="doc1", ability_name="protect", target_id="seer1"
        )
        state = (
            five_player_state
            .with_night_action("wolf1", kill_action)
            .with_night_action("doc1", protect_action)
        )

        new_state, events = resolve_night(state, registry)

        # Villager should be dead
        villager = new_state.get_player("vil1")
        assert villager is not None
        assert villager.is_alive is False

        # Seer should survive
        seer = new_state.get_player("seer1")
        assert seer is not None
        assert seer.is_alive is True

        night_result = [e for e in events if isinstance(e, NightResultEvent)][0]
        assert "vil1" in night_result.kills
        assert "seer1" in night_result.protected
        assert night_result.saved == []


class TestSeerInvestigate:
    """Seer investigate action producing PrivateRevealEvent."""

    def test_investigate_produces_private_reveal(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        investigate_action = UseAbilityAction(
            player_id="seer1", ability_name="investigate", target_id="wolf1"
        )
        state = five_player_state.with_night_action("seer1", investigate_action)

        _, events = resolve_night(state, registry)

        reveals = [e for e in events if isinstance(e, PrivateRevealEvent)]
        assert len(reveals) == 1
        assert reveals[0].player_id == "seer1"
        assert "wolf1" in reveals[0].info or "Wolf1" in reveals[0].info
        assert "werewolf" in reveals[0].info


class TestFullNightScenario:
    """Test all three roles acting in the same night."""

    def test_doctor_protects_wolf_kills_seer_investigates(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        """Doctor protects seer, wolf kills villager, seer investigates wolf."""
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        protect_action = UseAbilityAction(
            player_id="doc1", ability_name="protect", target_id="seer1"
        )
        investigate_action = UseAbilityAction(
            player_id="seer1", ability_name="investigate", target_id="wolf1"
        )
        state = (
            five_player_state
            .with_night_action("wolf1", kill_action)
            .with_night_action("doc1", protect_action)
            .with_night_action("seer1", investigate_action)
        )

        new_state, events = resolve_night(state, registry)

        # Villager dead
        assert new_state.get_player("vil1").is_alive is False
        # Seer alive (was protected, but not targeted anyway)
        assert new_state.get_player("seer1").is_alive is True

        # Check NightResultEvent
        night_result = [e for e in events if isinstance(e, NightResultEvent)][0]
        assert "vil1" in night_result.kills
        assert "seer1" in night_result.protected

        # Check PrivateRevealEvent
        reveals = [e for e in events if isinstance(e, PrivateRevealEvent)]
        assert len(reveals) == 1
        assert reveals[0].player_id == "seer1"

        # Check EliminationEvent
        elim_events = [e for e in events if isinstance(e, EliminationEvent)]
        assert len(elim_events) == 1
        assert elim_events[0].player_id == "vil1"

    def test_events_stored_in_new_state(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        """Verify that all resolution events are appended to state.events."""
        kill_action = UseAbilityAction(
            player_id="wolf1", ability_name="kill", target_id="vil1"
        )
        state = five_player_state.with_night_action("wolf1", kill_action)

        new_state, events = resolve_night(state, registry)

        # Events in list should also be in state.events
        assert len(new_state.events) == len(events)
        for event in events:
            assert event in new_state.events

    def test_no_action_type_is_ignored(
        self, five_player_state: GameState, registry: RoleRegistry
    ) -> None:
        """NoAction entries should be skipped by the resolver."""
        no_action = NoAction(player_id="wolf1", reason="timeout")
        kill_action = UseAbilityAction(
            player_id="wolf2", ability_name="kill", target_id="vil1"
        )
        state = (
            five_player_state
            .with_night_action("wolf1", no_action)
            .with_night_action("wolf2", kill_action)
        )

        new_state, events = resolve_night(state, registry)

        # Only one kill from wolf2
        elim_events = [e for e in events if isinstance(e, EliminationEvent)]
        assert len(elim_events) == 1
        assert elim_events[0].player_id == "vil1"
