"""Tests for wolf.agents.memory -- AgentMemory and PlayerModel."""

from __future__ import annotations

import pytest

from wolf.agents.memory import AgentMemory, Observation, PlayerModel


# ======================================================================
# PlayerModel tests
# ======================================================================


class TestPlayerModel:
    """Tests for the PlayerModel dataclass."""

    def test_creation_with_required_fields(self) -> None:
        pm = PlayerModel(player_id="p1", name="Alice")
        assert pm.player_id == "p1"
        assert pm.name == "Alice"

    def test_defaults(self) -> None:
        pm = PlayerModel(player_id="p1", name="Alice")
        assert pm.suspicion == 0.5
        assert pm.trust == 0.5
        assert pm.notes == []
        assert pm.voted_for == []
        assert pm.voted_by == []
        assert pm.claimed_role is None


# ======================================================================
# AgentMemory -- observations
# ======================================================================


class TestAgentMemoryObservations:
    """Tests for observation tracking."""

    def test_add_observation(self) -> None:
        memory = AgentMemory()
        obs = Observation(day=1, phase="NIGHT", content="Wolf killed someone")
        memory.add_observation(obs)
        assert len(memory.observations) == 1
        assert memory.observations[0] is obs

    def test_get_recent_observations_default(self) -> None:
        memory = AgentMemory()
        for i in range(15):
            memory.add_observation(
                Observation(day=1, phase="DAY", content=f"obs {i}")
            )
        recent = memory.get_recent_observations()
        assert len(recent) == 10
        # Should be the last 10
        assert recent[0].content == "obs 5"
        assert recent[-1].content == "obs 14"

    def test_get_recent_observations_custom_n(self) -> None:
        memory = AgentMemory()
        for i in range(5):
            memory.add_observation(
                Observation(day=1, phase="DAY", content=f"obs {i}")
            )
        recent = memory.get_recent_observations(n=3)
        assert len(recent) == 3
        assert recent[0].content == "obs 2"

    def test_get_recent_observations_fewer_than_n(self) -> None:
        memory = AgentMemory()
        memory.add_observation(Observation(day=1, phase="DAY", content="only one"))
        recent = memory.get_recent_observations(n=10)
        assert len(recent) == 1

    def test_get_important_observations(self) -> None:
        memory = AgentMemory()
        memory.add_observation(Observation(day=1, phase="DAY", content="low", importance=0.3))
        memory.add_observation(Observation(day=1, phase="DAY", content="medium", importance=0.7))
        memory.add_observation(Observation(day=1, phase="DAY", content="high", importance=0.9))

        important = memory.get_important_observations(threshold=0.7)
        assert len(important) == 2
        assert important[0].content == "medium"
        assert important[1].content == "high"

    def test_get_important_observations_default_threshold(self) -> None:
        memory = AgentMemory()
        memory.add_observation(Observation(day=1, phase="DAY", content="below", importance=0.6))
        memory.add_observation(Observation(day=1, phase="DAY", content="at", importance=0.7))
        memory.add_observation(Observation(day=1, phase="DAY", content="above", importance=0.8))

        important = memory.get_important_observations()
        assert len(important) == 2


# ======================================================================
# AgentMemory -- player models
# ======================================================================


class TestAgentMemoryPlayerModels:
    """Tests for player model management."""

    def test_get_player_model_creates_stub(self) -> None:
        memory = AgentMemory()
        pm = memory.get_player_model("p1")
        assert pm.player_id == "p1"
        assert pm.name == "p1"
        assert pm.suspicion == 0.5

    def test_get_player_model_returns_same_instance(self) -> None:
        memory = AgentMemory()
        pm1 = memory.get_player_model("p1")
        pm2 = memory.get_player_model("p1")
        assert pm1 is pm2

    def test_update_player_model_scalar_field(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", suspicion=0.9)
        pm = memory.get_player_model("p1")
        assert pm.suspicion == 0.9

    def test_update_player_model_trust(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", trust=0.2)
        pm = memory.get_player_model("p1")
        assert pm.trust == 0.2

    def test_update_player_model_list_field_appends(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", notes="suspicious behavior")
        memory.update_player_model("p1", notes="voted against Alice")
        pm = memory.get_player_model("p1")
        assert pm.notes == ["suspicious behavior", "voted against Alice"]

    def test_update_player_model_voted_for_appends(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", voted_for="p2")
        memory.update_player_model("p1", voted_for="p3")
        pm = memory.get_player_model("p1")
        assert pm.voted_for == ["p2", "p3"]

    def test_update_player_model_voted_by_appends(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", voted_by="p2")
        pm = memory.get_player_model("p1")
        assert pm.voted_by == ["p2"]

    def test_update_player_model_claimed_role(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", claimed_role="seer")
        pm = memory.get_player_model("p1")
        assert pm.claimed_role == "seer"

    def test_update_player_model_unknown_field_ignored(self) -> None:
        """Unknown fields should be silently skipped (warning logged)."""
        memory = AgentMemory()
        memory.update_player_model("p1", nonexistent_field="value")
        pm = memory.get_player_model("p1")
        assert not hasattr(pm, "nonexistent_field") or pm.suspicion == 0.5


# ======================================================================
# AgentMemory -- summarize_for_prompt
# ======================================================================


class TestAgentMemorySummarize:
    """Tests for summarize_for_prompt output."""

    def test_empty_memory_returns_no_memories_yet(self) -> None:
        memory = AgentMemory()
        result = memory.summarize_for_prompt()
        assert result == "(no memories yet)"

    def test_includes_key_observations_section(self) -> None:
        memory = AgentMemory()
        memory.add_observation(
            Observation(day=1, phase="NIGHT", content="Wolf killed Bob", importance=0.9)
        )
        result = memory.summarize_for_prompt()
        assert "=== Key Observations ===" in result
        assert "Wolf killed Bob" in result

    def test_includes_recent_observations_section(self) -> None:
        memory = AgentMemory()
        # Add a low-importance observation (recent but not important)
        memory.add_observation(
            Observation(day=1, phase="DAY", content="Charlie spoke", importance=0.3)
        )
        result = memory.summarize_for_prompt()
        assert "=== Recent Observations ===" in result
        assert "Charlie spoke" in result

    def test_includes_player_assessments_section(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", suspicion=0.8, claimed_role="seer")
        memory.get_player_model("p1").name = "Alice"

        # Need at least one observation to avoid "(no memories yet)"
        memory.add_observation(
            Observation(day=1, phase="DAY", content="filler", importance=0.1)
        )

        result = memory.summarize_for_prompt()
        assert "=== Player Assessments ===" in result
        assert "Alice" in result
        assert "suspicion=0.8" in result
        assert "claims to be seer" in result

    def test_includes_notes_in_assessment(self) -> None:
        memory = AgentMemory()
        memory.update_player_model("p1", notes="acted suspiciously")
        memory.add_observation(
            Observation(day=1, phase="DAY", content="filler", importance=0.1)
        )
        result = memory.summarize_for_prompt()
        assert "acted suspiciously" in result

    def test_day_and_phase_in_observation_lines(self) -> None:
        memory = AgentMemory()
        memory.add_observation(
            Observation(day=2, phase="DAWN", content="Bob was killed", importance=0.9)
        )
        result = memory.summarize_for_prompt()
        assert "[Day 2, DAWN]" in result
