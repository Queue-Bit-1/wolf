"""Integration test: run a full Werewolf game with RandomAgents."""

from __future__ import annotations

import pytest

from wolf.agents.random_agent import RandomAgent
from wolf.comms.manager import ChannelManager
from wolf.config.schema import (
    CommunicationConfig,
    GameConfig,
    RoleSlot,
    VotingConfig,
)
from wolf.engine.events import GameEndEvent
from wolf.engine.game import Game
from wolf.metrics.collector import MetricsCollector
from wolf.roles.registry import RoleRegistry


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def small_game_config() -> GameConfig:
    """A small 5-player game config for fast integration tests."""
    return GameConfig(
        game_name="integration_test",
        num_players=5,
        roles=[
            RoleSlot(role="werewolf", count=2),
            RoleSlot(role="seer", count=1),
            RoleSlot(role="doctor", count=1),
            RoleSlot(role="villager", count=1),
        ],
        max_days=5,
        voting=VotingConfig(
            method="plurality",
            allow_no_vote=False,
            reveal_votes=True,
            tie_breaker="random",
        ),
        communication=CommunicationConfig(
            allow_wolf_chat=True,
            allow_dms=False,
            discussion_rounds=1,
            max_speech_length=200,
        ),
    )


@pytest.fixture
def agents() -> dict[str, RandomAgent]:
    """Five RandomAgents keyed by player ID."""
    player_ids = ["p1", "p2", "p3", "p4", "p5"]
    return {pid: RandomAgent(player_id=pid, name=f"Player_{pid}") for pid in player_ids}


@pytest.fixture
def channel_manager(agents: dict[str, RandomAgent]) -> ChannelManager:
    """A ChannelManager configured for the integration test."""
    mgr = ChannelManager()
    config = CommunicationConfig(
        allow_wolf_chat=True,
        allow_dms=False,
        discussion_rounds=1,
    )
    player_ids = list(agents.keys())
    # Wolves are p1 and p2 based on role assignment order
    wolf_ids = ["p1", "p2"]
    mgr.create_channels(player_ids, wolf_ids, config)
    return mgr


@pytest.fixture
def collector() -> MetricsCollector:
    """A fresh MetricsCollector."""
    return MetricsCollector()


# ======================================================================
# Integration tests
# ======================================================================


class TestFullGameWithRandomAgents:
    """Run a complete game and verify it terminates with valid results."""

    @pytest.mark.asyncio
    async def test_game_completes_without_errors(
        self,
        small_game_config: GameConfig,
        agents: dict[str, RandomAgent],
        channel_manager: ChannelManager,
        collector: MetricsCollector,
    ) -> None:
        """A full game with RandomAgents runs to completion."""
        # Register players with the collector
        role_assignments = [
            ("p1", "werewolf", "werewolf"),
            ("p2", "werewolf", "werewolf"),
            ("p3", "seer", "village"),
            ("p4", "doctor", "village"),
            ("p5", "villager", "village"),
        ]
        for pid, role, team in role_assignments:
            collector.register_player(pid, f"Player_{pid}", role, team)

        game = Game(
            config=small_game_config,
            agents=agents,
            role_registry=RoleRegistry,
            channel_manager=channel_manager,
            event_listeners=[collector],
        )

        result = await game.run()

        # The game must produce a GameEndEvent
        assert isinstance(result, GameEndEvent)

    @pytest.mark.asyncio
    async def test_game_ends_with_valid_winning_team(
        self,
        small_game_config: GameConfig,
        agents: dict[str, RandomAgent],
        channel_manager: ChannelManager,
        collector: MetricsCollector,
    ) -> None:
        """The winning team must be either 'village' or 'werewolf'."""
        for pid, role, team in [
            ("p1", "werewolf", "werewolf"),
            ("p2", "werewolf", "werewolf"),
            ("p3", "seer", "village"),
            ("p4", "doctor", "village"),
            ("p5", "villager", "village"),
        ]:
            collector.register_player(pid, f"Player_{pid}", role, team)

        game = Game(
            config=small_game_config,
            agents=agents,
            role_registry=RoleRegistry,
            channel_manager=channel_manager,
            event_listeners=[collector],
        )

        result = await game.run()
        assert result.winning_team in ("village", "werewolf")

    @pytest.mark.asyncio
    async def test_game_summary_contains_all_players(
        self,
        small_game_config: GameConfig,
        agents: dict[str, RandomAgent],
        channel_manager: ChannelManager,
        collector: MetricsCollector,
    ) -> None:
        """The game summary must contain entries for all 5 players."""
        for pid, role, team in [
            ("p1", "werewolf", "werewolf"),
            ("p2", "werewolf", "werewolf"),
            ("p3", "seer", "village"),
            ("p4", "doctor", "village"),
            ("p5", "villager", "village"),
        ]:
            collector.register_player(pid, f"Player_{pid}", role, team)

        game = Game(
            config=small_game_config,
            agents=agents,
            role_registry=RoleRegistry,
            channel_manager=channel_manager,
            event_listeners=[collector],
        )

        await game.run()

        summary = collector.get_game_summary()
        player_ids = {p["player_id"] for p in summary["players"]}
        assert player_ids == {"p1", "p2", "p3", "p4", "p5"}

    @pytest.mark.asyncio
    async def test_game_produces_events(
        self,
        small_game_config: GameConfig,
        agents: dict[str, RandomAgent],
        channel_manager: ChannelManager,
        collector: MetricsCollector,
    ) -> None:
        """The game should produce a non-trivial number of events."""
        for pid, role, team in [
            ("p1", "werewolf", "werewolf"),
            ("p2", "werewolf", "werewolf"),
            ("p3", "seer", "village"),
            ("p4", "doctor", "village"),
            ("p5", "villager", "village"),
        ]:
            collector.register_player(pid, f"Player_{pid}", role, team)

        game = Game(
            config=small_game_config,
            agents=agents,
            role_registry=RoleRegistry,
            channel_manager=channel_manager,
            event_listeners=[collector],
        )

        await game.run()

        summary = collector.get_game_summary()
        # A game that runs at least one full cycle should have many events
        assert summary["total_events"] > 0

    @pytest.mark.asyncio
    async def test_game_respects_max_days(
        self,
        agents: dict[str, RandomAgent],
        channel_manager: ChannelManager,
    ) -> None:
        """If the game exceeds max_days it should still terminate."""
        config = GameConfig(
            game_name="max_day_test",
            num_players=5,
            roles=[
                RoleSlot(role="werewolf", count=2),
                RoleSlot(role="seer", count=1),
                RoleSlot(role="doctor", count=1),
                RoleSlot(role="villager", count=1),
            ],
            max_days=1,  # Very low to force early termination if no winner
            voting=VotingConfig(tie_breaker="no_elimination"),
            communication=CommunicationConfig(
                allow_wolf_chat=True,
                allow_dms=False,
                discussion_rounds=1,
            ),
        )

        game = Game(
            config=config,
            agents=agents,
            role_registry=RoleRegistry,
            channel_manager=channel_manager,
        )

        result = await game.run()
        assert isinstance(result, GameEndEvent)
        # Game must still produce a valid result even if it's a timeout
        assert result.winning_team in ("village", "werewolf")
