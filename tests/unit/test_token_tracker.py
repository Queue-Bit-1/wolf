"""Tests for wolf.llm.token_tracker -- TokenTracker usage tracking."""

from __future__ import annotations

import pytest

from wolf.llm.token_tracker import TokenTracker


# ======================================================================
# Tests
# ======================================================================


class TestTokenTrackerRecord:
    """Tests for recording and querying token usage."""

    def test_record_single_call(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50, call_type="reasoning")
        usage = tracker.get_player_usage("p1")
        assert usage["total_input"] == 100
        assert usage["total_output"] == 50

    def test_record_multiple_calls_accumulate(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50, call_type="reasoning")
        tracker.record("p1", input_tokens=200, output_tokens=80, call_type="action")
        usage = tracker.get_player_usage("p1")
        assert usage["total_input"] == 300
        assert usage["total_output"] == 130

    def test_record_by_call_type_breakdown(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50, call_type="reasoning")
        tracker.record("p1", input_tokens=60, output_tokens=30, call_type="action")
        usage = tracker.get_player_usage("p1")
        assert usage["by_call_type"]["reasoning"] == {"input": 100, "output": 50}
        assert usage["by_call_type"]["action"] == {"input": 60, "output": 30}

    def test_record_same_call_type_accumulates(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50, call_type="reasoning")
        tracker.record("p1", input_tokens=150, output_tokens=70, call_type="reasoning")
        usage = tracker.get_player_usage("p1")
        assert usage["by_call_type"]["reasoning"] == {"input": 250, "output": 120}


class TestTokenTrackerGetPlayerUsage:
    """Tests for get_player_usage."""

    def test_unknown_player_returns_zeros(self) -> None:
        tracker = TokenTracker()
        usage = tracker.get_player_usage("unknown")
        assert usage["total_input"] == 0
        assert usage["total_output"] == 0
        assert usage["by_call_type"] == {}

    def test_known_player(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=10, output_tokens=5)
        usage = tracker.get_player_usage("p1")
        assert usage["total_input"] == 10
        assert usage["total_output"] == 5


class TestTokenTrackerGetTotalUsage:
    """Tests for get_total_usage across multiple players."""

    def test_total_across_multiple_players(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50)
        tracker.record("p2", input_tokens=200, output_tokens=80)
        tracker.record("p3", input_tokens=50, output_tokens=20)

        total = tracker.get_total_usage()
        assert total["total_input"] == 350
        assert total["total_output"] == 150

    def test_total_per_player_breakdown(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50)
        tracker.record("p2", input_tokens=200, output_tokens=80)

        total = tracker.get_total_usage()
        assert "p1" in total["per_player"]
        assert "p2" in total["per_player"]
        assert total["per_player"]["p1"]["total_input"] == 100
        assert total["per_player"]["p2"]["total_input"] == 200

    def test_total_empty_tracker(self) -> None:
        tracker = TokenTracker()
        total = tracker.get_total_usage()
        assert total["total_input"] == 0
        assert total["total_output"] == 0
        assert total["per_player"] == {}


class TestTokenTrackerReset:
    """Tests for the reset method."""

    def test_reset_clears_data(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50)
        tracker.record("p2", input_tokens=200, output_tokens=80)

        tracker.reset()

        assert tracker.get_player_usage("p1")["total_input"] == 0
        assert tracker.get_player_usage("p2")["total_input"] == 0
        total = tracker.get_total_usage()
        assert total["total_input"] == 0
        assert total["per_player"] == {}

    def test_can_record_after_reset(self) -> None:
        tracker = TokenTracker()
        tracker.record("p1", input_tokens=100, output_tokens=50)
        tracker.reset()
        tracker.record("p1", input_tokens=10, output_tokens=5)

        usage = tracker.get_player_usage("p1")
        assert usage["total_input"] == 10
        assert usage["total_output"] == 5
