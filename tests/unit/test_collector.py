"""Tests for wolf.metrics.collector -- MetricsCollector event processing."""

from __future__ import annotations

import pytest

from wolf.engine.events import (
    EliminationEvent,
    GameEndEvent,
    SpeechEvent,
    VoteEvent,
)
from wolf.engine.phase import Phase
from wolf.metrics.collector import MetricsCollector


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def collector() -> MetricsCollector:
    """A fresh MetricsCollector."""
    return MetricsCollector()


@pytest.fixture
def registered_collector() -> MetricsCollector:
    """A MetricsCollector with pre-registered players."""
    mc = MetricsCollector()
    mc.register_player("p1", "Alice", "seer", "village")
    mc.register_player("p2", "Bob", "werewolf", "werewolf")
    mc.register_player("p3", "Charlie", "villager", "village")
    return mc


# ======================================================================
# SpeechEvent processing
# ======================================================================


class TestSpeechEvent:
    """Tests for SpeechEvent handling."""

    def test_increments_speeches(self, registered_collector: MetricsCollector) -> None:
        event = SpeechEvent(
            day=1, phase=Phase.DAY_DISCUSSION, player_id="p1", content="Hello"
        )
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        p1_stats = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert p1_stats["speeches"] == 1

    def test_tracks_speech_content(self, registered_collector: MetricsCollector) -> None:
        event = SpeechEvent(
            day=1, phase=Phase.DAY_DISCUSSION, player_id="p1", content="I suspect Bob"
        )
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        p1_stats = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert "I suspect Bob" in p1_stats["speech_contents"]

    def test_multiple_speeches_accumulate(
        self, registered_collector: MetricsCollector
    ) -> None:
        for i in range(3):
            registered_collector(
                SpeechEvent(
                    day=1,
                    phase=Phase.DAY_DISCUSSION,
                    player_id="p1",
                    content=f"message {i}",
                )
            )
        summary = registered_collector.get_game_summary()
        p1_stats = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert p1_stats["speeches"] == 3


# ======================================================================
# VoteEvent processing
# ======================================================================


class TestVoteEvent:
    """Tests for VoteEvent handling."""

    def test_votes_cast_incremented(
        self, registered_collector: MetricsCollector
    ) -> None:
        event = VoteEvent(day=1, phase=Phase.DAY_VOTE, voter_id="p1", target_id="p2")
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        p1_stats = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert p1_stats["votes_cast"] == 1

    def test_votes_received_incremented(
        self, registered_collector: MetricsCollector
    ) -> None:
        event = VoteEvent(day=1, phase=Phase.DAY_VOTE, voter_id="p1", target_id="p2")
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        p2_stats = next(p for p in summary["players"] if p["player_id"] == "p2")
        assert p2_stats["votes_received"] == 1

    def test_vote_for_none_does_not_increment_received(
        self, registered_collector: MetricsCollector
    ) -> None:
        event = VoteEvent(day=1, phase=Phase.DAY_VOTE, voter_id="p1", target_id=None)
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        # No player should have votes_received incremented
        for p in summary["players"]:
            assert p["votes_received"] == 0

    def test_vote_targets_tracked(
        self, registered_collector: MetricsCollector
    ) -> None:
        registered_collector(
            VoteEvent(day=1, phase=Phase.DAY_VOTE, voter_id="p1", target_id="p2")
        )
        registered_collector(
            VoteEvent(day=2, phase=Phase.DAY_VOTE, voter_id="p1", target_id="p3")
        )
        summary = registered_collector.get_game_summary()
        p1_stats = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert p1_stats["vote_targets"] == ["p2", "p3"]


# ======================================================================
# EliminationEvent processing
# ======================================================================


class TestEliminationEvent:
    """Tests for EliminationEvent handling."""

    def test_marks_player_dead(
        self, registered_collector: MetricsCollector
    ) -> None:
        event = EliminationEvent(
            day=2, phase=Phase.DAWN, player_id="p2", role="werewolf", cause="vote"
        )
        registered_collector(event)
        summary = registered_collector.get_game_summary()
        p2_stats = next(p for p in summary["players"] if p["player_id"] == "p2")
        assert p2_stats["is_alive"] is False
        assert p2_stats["survived_until"] == 2
        assert p2_stats["elimination_cause"] == "vote"

    def test_backfills_role(self, collector: MetricsCollector) -> None:
        """If player wasn't registered, role is backfilled from event."""
        event = EliminationEvent(
            day=1, phase=Phase.DAWN, player_id="new_p", role="seer", cause="wolf_kill"
        )
        collector(event)
        summary = collector.get_game_summary()
        stats = next(p for p in summary["players"] if p["player_id"] == "new_p")
        assert stats["role"] == "seer"


# ======================================================================
# register_player
# ======================================================================


class TestRegisterPlayer:
    """Tests for register_player pre-population."""

    def test_pre_populates_stats(self, collector: MetricsCollector) -> None:
        collector.register_player("p1", "Alice", "seer", "village")
        summary = collector.get_game_summary()
        assert len(summary["players"]) == 1
        p1 = summary["players"][0]
        assert p1["player_id"] == "p1"
        assert p1["name"] == "Alice"
        assert p1["role"] == "seer"
        assert p1["team"] == "village"
        assert p1["speeches"] == 0
        assert p1["votes_cast"] == 0
        assert p1["is_alive"] is True


# ======================================================================
# get_game_summary
# ======================================================================


class TestGetGameSummary:
    """Tests for the game summary output structure."""

    def test_summary_structure(
        self, registered_collector: MetricsCollector
    ) -> None:
        summary = registered_collector.get_game_summary()
        assert "players" in summary
        assert "result" in summary
        assert "total_days" in summary
        assert "total_events" in summary

    def test_summary_contains_all_registered_players(
        self, registered_collector: MetricsCollector
    ) -> None:
        summary = registered_collector.get_game_summary()
        player_ids = {p["player_id"] for p in summary["players"]}
        assert player_ids == {"p1", "p2", "p3"}

    def test_summary_result_populated_after_game_end(
        self, registered_collector: MetricsCollector
    ) -> None:
        end_event = GameEndEvent(
            day=3,
            phase=Phase.GAME_OVER,
            winning_team="village",
            winners=["p1", "p3"],
            reason="All werewolves eliminated.",
        )
        registered_collector(end_event)
        summary = registered_collector.get_game_summary()
        assert summary["result"]["winning_team"] == "village"
        assert "p1" in summary["result"]["winners"]

    def test_summary_result_empty_when_no_end(
        self, registered_collector: MetricsCollector
    ) -> None:
        summary = registered_collector.get_game_summary()
        assert summary["result"] == {}

    def test_total_events_count(
        self, registered_collector: MetricsCollector
    ) -> None:
        registered_collector(
            SpeechEvent(day=1, phase=Phase.DAY_DISCUSSION, player_id="p1", content="hi")
        )
        registered_collector(
            VoteEvent(day=1, phase=Phase.DAY_VOTE, voter_id="p1", target_id="p2")
        )
        summary = registered_collector.get_game_summary()
        assert summary["total_events"] == 2

    def test_survived_until_finalized_for_alive_players(
        self, registered_collector: MetricsCollector
    ) -> None:
        """Alive players should have survived_until set to current_day."""
        registered_collector(
            SpeechEvent(day=3, phase=Phase.DAY_DISCUSSION, player_id="p1", content="hi")
        )
        summary = registered_collector.get_game_summary()
        p1 = next(p for p in summary["players"] if p["player_id"] == "p1")
        assert p1["survived_until"] == 3
        assert p1["is_alive"] is True
