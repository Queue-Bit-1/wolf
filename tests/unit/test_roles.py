"""Tests for wolf.roles -- RoleRegistry, base classes, and all classic roles."""

from __future__ import annotations

import pytest

from wolf.engine.events import PrivateRevealEvent
from wolf.engine.phase import Phase
from wolf.engine.state import GameState, PlayerSlot
from wolf.roles.base import AbilityDefinition, RoleBase, Team
from wolf.roles.registry import RoleRegistry


# ======================================================================
# RoleRegistry tests
# ======================================================================


class TestRoleRegistry:
    """Tests for RoleRegistry plugin discovery and instance retrieval."""

    def test_discover_plugins_finds_four_classic_roles(self) -> None:
        # discover_plugins is called at import time; verify the expected set.
        all_roles = RoleRegistry.get_all()
        expected = {"villager", "werewolf", "seer", "doctor"}
        assert set(all_roles.keys()) == expected

    def test_get_returns_correct_instance(self) -> None:
        villager = RoleRegistry.get("villager")
        assert isinstance(villager, RoleBase)
        assert villager.name == "villager"

    def test_get_werewolf(self) -> None:
        wolf = RoleRegistry.get("werewolf")
        assert isinstance(wolf, RoleBase)
        assert wolf.name == "werewolf"

    def test_get_seer(self) -> None:
        seer = RoleRegistry.get("seer")
        assert isinstance(seer, RoleBase)
        assert seer.name == "seer"

    def test_get_doctor(self) -> None:
        doctor = RoleRegistry.get("doctor")
        assert isinstance(doctor, RoleBase)
        assert doctor.name == "doctor"

    def test_get_unknown_role_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown role"):
            RoleRegistry.get("witch")

    def test_get_returns_new_instances(self) -> None:
        a = RoleRegistry.get("villager")
        b = RoleRegistry.get("villager")
        assert a is not b  # fresh instance each time


# ======================================================================
# Role property tests
# ======================================================================


class TestVillager:
    """Tests for the Villager role."""

    def test_name(self) -> None:
        v = RoleRegistry.get("villager")
        assert v.name == "villager"

    def test_team(self) -> None:
        v = RoleRegistry.get("villager")
        assert v.team == Team.VILLAGE

    def test_description(self) -> None:
        v = RoleRegistry.get("villager")
        assert len(v.description) > 0

    def test_no_abilities(self) -> None:
        v = RoleRegistry.get("villager")
        assert v.abilities == []

    def test_resolve_ability_returns_empty(self) -> None:
        v = RoleRegistry.get("villager")
        result = v.resolve_ability("anything", "p1", "p2", None)
        assert result == []


class TestWerewolf:
    """Tests for the Werewolf role."""

    def test_name(self) -> None:
        w = RoleRegistry.get("werewolf")
        assert w.name == "werewolf"

    def test_team(self) -> None:
        w = RoleRegistry.get("werewolf")
        assert w.team == Team.WEREWOLF

    def test_description(self) -> None:
        w = RoleRegistry.get("werewolf")
        assert len(w.description) > 0

    def test_kill_ability(self) -> None:
        w = RoleRegistry.get("werewolf")
        assert len(w.abilities) == 1
        ability = w.abilities[0]
        assert ability.name == "kill"
        assert ability.phase == Phase.NIGHT
        assert ability.priority == 15

    def test_resolve_ability_returns_empty(self) -> None:
        """Wolf kills are handled by the resolver, not the role directly."""
        w = RoleRegistry.get("werewolf")
        result = w.resolve_ability("kill", "p_wolf", "p_target", None)
        assert result == []


class TestSeer:
    """Tests for the Seer role."""

    def test_name(self) -> None:
        s = RoleRegistry.get("seer")
        assert s.name == "seer"

    def test_team(self) -> None:
        s = RoleRegistry.get("seer")
        assert s.team == Team.VILLAGE

    def test_description(self) -> None:
        s = RoleRegistry.get("seer")
        assert len(s.description) > 0

    def test_investigate_ability(self) -> None:
        s = RoleRegistry.get("seer")
        assert len(s.abilities) == 1
        ability = s.abilities[0]
        assert ability.name == "investigate"
        assert ability.phase == Phase.NIGHT
        assert ability.priority == 20

    def test_resolve_ability_investigate(self) -> None:
        """Seer.resolve_ability returns a PrivateRevealEvent with the target's team."""
        from wolf.engine.events import PrivateRevealEvent

        s = RoleRegistry.get("seer")
        state = GameState(
            day=1,
            phase=Phase.NIGHT,
            players=(
                PlayerSlot(player_id="seer1", name="Seer", role="seer", team="village"),
                PlayerSlot(player_id="wolf1", name="Wolf", role="werewolf", team="werewolf"),
            ),
        )
        result = s.resolve_ability("investigate", "seer1", "wolf1", state)
        assert len(result) == 1
        assert isinstance(result[0], PrivateRevealEvent)
        assert "werewolf" in result[0].info

    def test_resolve_ability_non_investigate_returns_empty(self) -> None:
        s = RoleRegistry.get("seer")
        result = s.resolve_ability("other", "p1", "p2", None)
        assert result == []


class TestDoctor:
    """Tests for the Doctor role."""

    def test_name(self) -> None:
        d = RoleRegistry.get("doctor")
        assert d.name == "doctor"

    def test_team(self) -> None:
        d = RoleRegistry.get("doctor")
        assert d.team == Team.VILLAGE

    def test_description(self) -> None:
        d = RoleRegistry.get("doctor")
        assert len(d.description) > 0

    def test_protect_ability(self) -> None:
        d = RoleRegistry.get("doctor")
        assert len(d.abilities) == 1
        ability = d.abilities[0]
        assert ability.name == "protect"
        assert ability.phase == Phase.NIGHT
        assert ability.priority == 10

    def test_resolve_ability_returns_empty(self) -> None:
        """Doctor protection is handled by the resolver."""
        d = RoleRegistry.get("doctor")
        result = d.resolve_ability("protect", "doc", "target", None)
        assert result == []


# ======================================================================
# AbilityDefinition tests
# ======================================================================


class TestAbilityDefinition:
    """Tests for the AbilityDefinition dataclass."""

    def test_fields(self) -> None:
        ability = AbilityDefinition(
            name="test_ability",
            phase=Phase.NIGHT,
            priority=25,
            description="A test ability",
            targets="alive_others",
        )
        assert ability.name == "test_ability"
        assert ability.phase == Phase.NIGHT
        assert ability.priority == 25
        assert ability.description == "A test ability"
        assert ability.targets == "alive_others"

    def test_defaults(self) -> None:
        ability = AbilityDefinition(name="x", phase=Phase.NIGHT, priority=0)
        assert ability.description == ""
        assert ability.targets == "alive_others"

    def test_frozen(self) -> None:
        ability = AbilityDefinition(name="x", phase=Phase.NIGHT, priority=0)
        with pytest.raises(AttributeError):
            ability.name = "y"  # type: ignore[misc]
