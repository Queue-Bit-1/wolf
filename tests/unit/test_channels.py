"""Tests for wolf.comms -- Channel types and ChannelManager."""

from __future__ import annotations

import pytest

from wolf.comms.channel import (
    DirectMessageChannel,
    PublicChannel,
    WolfChannel,
)
from wolf.comms.manager import ChannelManager
from wolf.comms.message import Message
from wolf.config.schema import CommunicationConfig
from wolf.engine.phase import Phase


# ======================================================================
# PublicChannel tests
# ======================================================================


class TestPublicChannel:
    """Tests for the PublicChannel."""

    def test_name(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.name == "public"

    def test_can_send_during_day_discussion(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.can_send("p1", Phase.DAY_DISCUSSION) is True

    def test_cannot_send_during_night(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.can_send("p1", Phase.NIGHT) is False

    def test_cannot_send_during_day_vote(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.can_send("p1", Phase.DAY_VOTE) is False

    def test_cannot_send_non_member(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.can_send("p3", Phase.DAY_DISCUSSION) is False

    def test_can_read_during_any_phase(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        for phase in Phase:
            assert ch.can_read("p1", phase) is True

    def test_cannot_read_non_member(self) -> None:
        ch = PublicChannel(["p1", "p2"])
        assert ch.can_read("p3", Phase.DAY_DISCUSSION) is False

    def test_members(self) -> None:
        ch = PublicChannel(["p1", "p2", "p3"])
        assert ch.members == frozenset({"p1", "p2", "p3"})


# ======================================================================
# WolfChannel tests
# ======================================================================


class TestWolfChannel:
    """Tests for the WolfChannel."""

    def test_name(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.name == "wolf"

    def test_can_send_during_night(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.can_send("w1", Phase.NIGHT) is True

    def test_cannot_send_during_day(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.can_send("w1", Phase.DAY_DISCUSSION) is False

    def test_non_wolf_cannot_send(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.can_send("villager1", Phase.NIGHT) is False

    def test_wolf_can_read(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.can_read("w1", Phase.NIGHT) is True
        assert ch.can_read("w1", Phase.DAY_DISCUSSION) is True

    def test_non_wolf_cannot_read(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.can_read("villager1", Phase.NIGHT) is False

    def test_members(self) -> None:
        ch = WolfChannel(["w1", "w2"])
        assert ch.members == frozenset({"w1", "w2"})


# ======================================================================
# DirectMessageChannel tests
# ======================================================================


class TestDirectMessageChannel:
    """Tests for the DirectMessageChannel."""

    def test_name_is_deterministic(self) -> None:
        ch1 = DirectMessageChannel("p1", "p2")
        ch2 = DirectMessageChannel("p2", "p1")
        assert ch1.name == ch2.name

    def test_name_format(self) -> None:
        ch = DirectMessageChannel("alpha", "beta")
        assert ch.name == "dm:alpha:beta"

    def test_only_members_can_send(self) -> None:
        ch = DirectMessageChannel("p1", "p2")
        assert ch.can_send("p1", Phase.DAY_DISCUSSION) is True
        assert ch.can_send("p2", Phase.DAY_DISCUSSION) is True
        assert ch.can_send("p3", Phase.DAY_DISCUSSION) is False

    def test_only_members_can_read(self) -> None:
        ch = DirectMessageChannel("p1", "p2")
        assert ch.can_read("p1", Phase.NIGHT) is True
        assert ch.can_read("p2", Phase.NIGHT) is True
        assert ch.can_read("p3", Phase.NIGHT) is False

    def test_default_allowed_phases(self) -> None:
        ch = DirectMessageChannel("p1", "p2")
        assert ch.can_send("p1", Phase.DAY_DISCUSSION) is True
        assert ch.can_send("p1", Phase.NIGHT) is False

    def test_custom_allowed_phases(self) -> None:
        ch = DirectMessageChannel(
            "p1", "p2", allowed_phases=[Phase.NIGHT, Phase.DAY_VOTE]
        )
        assert ch.can_send("p1", Phase.NIGHT) is True
        assert ch.can_send("p1", Phase.DAY_VOTE) is True
        assert ch.can_send("p1", Phase.DAY_DISCUSSION) is False

    def test_members(self) -> None:
        ch = DirectMessageChannel("p1", "p2")
        assert ch.members == frozenset({"p1", "p2"})


# ======================================================================
# ChannelManager tests
# ======================================================================


class TestChannelManager:
    """Tests for the ChannelManager."""

    @pytest.fixture
    def manager(self) -> ChannelManager:
        """A manager with a public channel and a wolf channel."""
        public = PublicChannel(["p1", "p2", "w1"])
        wolf = WolfChannel(["w1"])
        return ChannelManager(channels=[public, wolf])

    @pytest.mark.asyncio
    async def test_send_valid_message(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="p1",
            channel="public",
            content="Hello everyone",
            phase_name="DAY_DISCUSSION",
        )
        result = await manager.send(msg)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_denied_wrong_phase(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="p1",
            channel="public",
            content="Hello",
            phase_name="NIGHT",
        )
        result = await manager.send(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_denied_wrong_channel(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="p1",
            channel="wolf",
            content="Trying to sneak in",
            phase_name="NIGHT",
        )
        result = await manager.send(msg)
        assert result is False  # p1 is not a wolf

    @pytest.mark.asyncio
    async def test_send_wolf_channel_valid(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="w1",
            channel="wolf",
            content="Let's kill p1",
            phase_name="NIGHT",
        )
        result = await manager.send(msg)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_unknown_channel_rejected(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="p1",
            channel="nonexistent",
            content="test",
            phase_name="DAY_DISCUSSION",
        )
        result = await manager.send(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_empty_phase_name_rejected(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="p1",
            channel="public",
            content="test",
            phase_name="",
        )
        result = await manager.send(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_bypasses_permissions(self, manager: ChannelManager) -> None:
        msg = Message(
            sender_id="system",
            channel="public",
            content="Game announcement",
            phase_name="NIGHT",
        )
        # broadcast does not check permissions
        await manager.broadcast(msg)
        # Should be stored -- verify via get_visible_messages
        visible = manager.get_visible_messages("p1", Phase.DAY_DISCUSSION)
        assert any(m.content == "Game announcement" for m in visible)

    @pytest.mark.asyncio
    async def test_get_visible_messages_filters_by_channel(
        self, manager: ChannelManager
    ) -> None:
        public_msg = Message(
            sender_id="p1",
            channel="public",
            content="public message",
            phase_name="DAY_DISCUSSION",
        )
        wolf_msg = Message(
            sender_id="w1",
            channel="wolf",
            content="wolf secret",
            phase_name="NIGHT",
        )
        await manager.send(public_msg)
        await manager.send(wolf_msg)

        # p1 should see public but not wolf messages
        visible_p1 = manager.get_visible_messages("p1", Phase.DAY_DISCUSSION)
        assert any(m.content == "public message" for m in visible_p1)
        assert not any(m.content == "wolf secret" for m in visible_p1)

        # w1 should see both
        visible_w1 = manager.get_visible_messages("w1", Phase.DAY_DISCUSSION)
        assert any(m.content == "public message" for m in visible_w1)
        # Wolf channel read is allowed regardless of phase for members
        assert any(m.content == "wolf secret" for m in visible_w1)

    @pytest.mark.asyncio
    async def test_get_visible_messages_respects_visible_to(
        self, manager: ChannelManager
    ) -> None:
        msg = Message(
            sender_id="p1",
            channel="public",
            content="private-ish",
            phase_name="DAY_DISCUSSION",
            visible_to=frozenset({"p2"}),
        )
        await manager.send(msg)

        # p2 should see it
        visible_p2 = manager.get_visible_messages("p2", Phase.DAY_DISCUSSION)
        assert any(m.content == "private-ish" for m in visible_p2)

        # p1 should not see it (visible_to restricts)
        visible_p1 = manager.get_visible_messages("p1", Phase.DAY_DISCUSSION)
        assert not any(m.content == "private-ish" for m in visible_p1)


class TestChannelManagerCreateChannels:
    """Tests for the create_channels factory method."""

    def test_creates_public_channel(self) -> None:
        mgr = ChannelManager()
        config = CommunicationConfig(allow_wolf_chat=False, allow_dms=False)
        mgr.create_channels(["p1", "p2", "p3"], ["w1"], config)
        assert mgr.get_channel("public") is not None

    def test_creates_wolf_channel_when_enabled(self) -> None:
        mgr = ChannelManager()
        config = CommunicationConfig(allow_wolf_chat=True, allow_dms=False)
        mgr.create_channels(["p1", "p2", "w1"], ["w1"], config)
        assert mgr.get_channel("wolf") is not None

    def test_no_wolf_channel_when_disabled(self) -> None:
        mgr = ChannelManager()
        config = CommunicationConfig(allow_wolf_chat=False, allow_dms=False)
        mgr.create_channels(["p1", "p2", "w1"], ["w1"], config)
        assert mgr.get_channel("wolf") is None

    def test_creates_dm_channels_when_enabled(self) -> None:
        mgr = ChannelManager()
        config = CommunicationConfig(allow_wolf_chat=False, allow_dms=True)
        mgr.create_channels(["p1", "p2", "p3"], [], config)
        # 3 players -> C(3,2) = 3 DM channels + 1 public = 4
        assert mgr.get_channel("dm:p1:p2") is not None
        assert mgr.get_channel("dm:p1:p3") is not None
        assert mgr.get_channel("dm:p2:p3") is not None

    def test_no_dm_channels_when_disabled(self) -> None:
        mgr = ChannelManager()
        config = CommunicationConfig(allow_wolf_chat=False, allow_dms=False)
        mgr.create_channels(["p1", "p2", "p3"], [], config)
        assert mgr.get_channel("dm:p1:p2") is None

    def test_create_channels_replaces_previous(self) -> None:
        mgr = ChannelManager()
        config1 = CommunicationConfig(allow_wolf_chat=True, allow_dms=False)
        mgr.create_channels(["p1", "p2"], ["p1"], config1)
        assert mgr.get_channel("wolf") is not None

        config2 = CommunicationConfig(allow_wolf_chat=False, allow_dms=False)
        mgr.create_channels(["p1", "p2"], [], config2)
        assert mgr.get_channel("wolf") is None
