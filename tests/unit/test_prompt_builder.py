"""Tests for wolf.agents.prompt_builder -- prompt construction and response parsing."""

from __future__ import annotations

import pytest

from wolf.agents.prompt_builder import PromptBuilder
from wolf.engine.actions import (
    NoAction,
    SpeakAction,
    UseAbilityAction,
    VoteAction,
)
from wolf.engine.phase import Phase
from wolf.engine.state import GameState, GameStateView, PlayerSlot


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def game_view() -> GameStateView:
    """A simple game state view for a seer player."""
    state = GameState(
        day=2,
        phase=Phase.DAY_DISCUSSION,
        players=(
            PlayerSlot(player_id="p1", name="Alice", role="seer", team="village"),
            PlayerSlot(player_id="p2", name="Bob", role="werewolf", team="werewolf"),
            PlayerSlot(player_id="p3", name="Charlie", role="villager", team="village"),
        ),
    )
    return GameStateView(state, "p1")


# ======================================================================
# build_system_prompt
# ======================================================================


class TestBuildSystemPrompt:
    """Tests for the system prompt builder."""

    def test_contains_role_info(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt(
            role_name="seer",
            role_description="You can investigate players.",
            role_instructions="Use your power wisely.",
            player_name="Alice",
        )
        assert "seer" in prompt
        assert "investigate" in prompt.lower() or "You can investigate" in prompt

    def test_contains_player_name(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt(
            role_name="villager",
            role_description="Regular villager.",
            role_instructions="Find werewolves.",
            player_name="Bob",
        )
        assert "Bob" in prompt

    def test_contains_behavioral_guidelines(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt(
            role_name="villager",
            role_description="Regular villager.",
            role_instructions="Find werewolves.",
            player_name="Bob",
        )
        assert "strategic" in prompt.lower() or "character" in prompt.lower()

    def test_contains_werewolf_game_context(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt(
            role_name="villager",
            role_description="Regular villager.",
            role_instructions="Find werewolves.",
            player_name="Bob",
        )
        assert "Werewolf" in prompt or "werewolf" in prompt


# ======================================================================
# build_reasoning_prompt
# ======================================================================


class TestBuildReasoningPrompt:
    """Tests for the reasoning prompt builder."""

    def test_night_ability(self, builder: PromptBuilder, game_view: GameStateView) -> None:
        prompt = builder.build_reasoning_prompt(game_view, "night_ability")
        assert "night" in prompt.lower()
        assert "ability" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_discussion(self, builder: PromptBuilder, game_view: GameStateView) -> None:
        prompt = builder.build_reasoning_prompt(game_view, "discussion")
        assert "discussion" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_vote(self, builder: PromptBuilder, game_view: GameStateView) -> None:
        prompt = builder.build_reasoning_prompt(game_view, "vote")
        assert "vote" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_unknown_action_type(self, builder: PromptBuilder, game_view: GameStateView) -> None:
        prompt = builder.build_reasoning_prompt(game_view, "unknown")
        assert len(prompt) > 0  # should still return something


# ======================================================================
# build_action_prompt
# ======================================================================


class TestBuildActionPrompt:
    """Tests for the action prompt builder."""

    def test_discussion_prompt(
        self, builder: PromptBuilder, game_view: GameStateView
    ) -> None:
        prompt = builder.build_action_prompt(game_view, "discussion", [])
        assert "SPEAK:" in prompt

    def test_vote_prompt_with_targets(
        self, builder: PromptBuilder, game_view: GameStateView
    ) -> None:
        prompt = builder.build_action_prompt(game_view, "vote", ["Alice", "Bob"])
        assert "VOTE:" in prompt
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "no_one" in prompt

    def test_vote_prompt_no_targets(
        self, builder: PromptBuilder, game_view: GameStateView
    ) -> None:
        prompt = builder.build_action_prompt(game_view, "vote", [])
        assert "VOTE:" in prompt
        assert "(none)" in prompt

    def test_night_ability_prompt(
        self, builder: PromptBuilder, game_view: GameStateView
    ) -> None:
        prompt = builder.build_action_prompt(
            game_view, "night_ability", ["Bob", "Charlie"]
        )
        assert "TARGET:" in prompt
        assert "Bob" in prompt
        assert "Charlie" in prompt

    def test_unknown_action_type(
        self, builder: PromptBuilder, game_view: GameStateView
    ) -> None:
        prompt = builder.build_action_prompt(game_view, "unknown", [])
        assert len(prompt) > 0


# ======================================================================
# parse_action_response
# ======================================================================


class TestParseActionResponse:
    """Tests for parsing raw LLM responses into typed Actions."""

    def test_speak_prefix(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "SPEAK: I think Bob is suspicious.",
            "discussion",
            "p1",
            [],
        )
        assert isinstance(action, SpeakAction)
        assert "Bob is suspicious" in action.content

    def test_speak_fallback_to_raw(self, builder: PromptBuilder) -> None:
        """If no SPEAK: prefix but text is long enough, treat as speech."""
        action = builder.parse_action_response(
            "I have something important to say about the game.",
            "discussion",
            "p1",
            [],
        )
        assert isinstance(action, SpeakAction)
        assert "important" in action.content

    def test_speak_too_short_returns_no_action(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response("ok", "discussion", "p1", [])
        assert isinstance(action, NoAction)

    def test_vote_valid_target(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "VOTE: Bob", "vote", "p1", ["Alice", "Bob", "Charlie"]
        )
        assert isinstance(action, VoteAction)
        assert action.target_id == "Bob"

    def test_vote_no_one(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "VOTE: no_one", "vote", "p1", ["Alice", "Bob"]
        )
        assert isinstance(action, VoteAction)
        assert action.target_id is None

    def test_vote_none_synonym(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "VOTE: abstain", "vote", "p1", ["Alice"]
        )
        assert isinstance(action, VoteAction)
        assert action.target_id is None

    def test_vote_case_insensitive(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "VOTE: bob", "vote", "p1", ["Alice", "Bob"]
        )
        assert isinstance(action, VoteAction)
        assert action.target_id == "Bob"

    def test_vote_invalid_target(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "VOTE: NonexistentPlayer", "vote", "p1", ["Alice", "Bob"]
        )
        assert isinstance(action, NoAction)

    def test_vote_no_prefix(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "I think Bob should go.", "vote", "p1", ["Bob"]
        )
        assert isinstance(action, NoAction)

    def test_target_valid(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "TARGET: Bob", "night_ability", "p1", ["Alice", "Bob"]
        )
        assert isinstance(action, UseAbilityAction)
        assert action.target_id == "Bob"

    def test_target_invalid(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "TARGET: Nobody", "night_ability", "p1", ["Alice", "Bob"]
        )
        assert isinstance(action, NoAction)

    def test_target_no_prefix(self, builder: PromptBuilder) -> None:
        action = builder.parse_action_response(
            "I want to investigate Bob", "night_ability", "p1", ["Bob"]
        )
        assert isinstance(action, NoAction)

    def test_malformed_input_falls_back_to_no_action(
        self, builder: PromptBuilder
    ) -> None:
        action = builder.parse_action_response("", "vote", "p1", ["Alice"])
        assert isinstance(action, NoAction)

    def test_unknown_action_type_returns_no_action(
        self, builder: PromptBuilder
    ) -> None:
        action = builder.parse_action_response(
            "some response", "unknown_type", "p1", []
        )
        assert isinstance(action, NoAction)
