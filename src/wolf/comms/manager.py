"""Channel manager -- central hub for message routing and storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolf.comms.channel import (
    Channel,
    DirectMessageChannel,
    PublicChannel,
    WolfChannel,
)
from wolf.comms.message import Message
from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.config.schema import CommunicationConfig


class ChannelManager:
    """Manages communication channels and message storage."""

    def __init__(self, channels: list[Channel] | None = None) -> None:
        self._channels: dict[str, Channel] = {}
        self._messages: list[Message] = []
        if channels:
            for ch in channels:
                self._channels[ch.name] = ch

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, message: Message) -> bool:
        """Validate permissions and store a message.

        Returns ``True`` if the message was accepted, ``False`` otherwise.
        """
        channel = self._channels.get(message.channel)
        if channel is None:
            return False

        # Determine current phase from the message's phase_name.  If no
        # phase is given we still require the channel lookup to succeed.
        phase = _phase_from_name(message.phase_name)
        if phase is None:
            return False

        if not channel.can_send(message.sender_id, phase):
            return False

        self._messages.append(message)
        return True

    async def broadcast(self, message: Message) -> None:
        """Store a system message on the public channel without permission checks."""
        self._messages.append(message)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_visible_messages(
        self, player_id: str, phase: Phase
    ) -> list[Message]:
        """Return all messages this player is allowed to see given the current phase."""
        visible: list[Message] = []
        for msg in self._messages:
            channel = self._channels.get(msg.channel)
            if channel is None:
                # System broadcasts stored without a registered channel are
                # visible to everyone.
                if msg.channel == "public":
                    visible.append(msg)
                continue

            if not channel.can_read(player_id, phase):
                continue

            # If visible_to is set, restrict further.
            if msg.visible_to and player_id not in msg.visible_to:
                continue

            visible.append(msg)
        return visible

    # ------------------------------------------------------------------
    # Channel access
    # ------------------------------------------------------------------

    def get_channel(self, name: str) -> Channel | None:
        """Look up a channel by name."""
        return self._channels.get(name)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def create_channels(
        self,
        player_ids: list[str],
        wolf_ids: list[str],
        config: CommunicationConfig,
    ) -> None:
        """Create the standard set of game channels based on *config*.

        This replaces any channels previously registered.
        """
        self._channels.clear()

        # Public channel -- always present.
        public = PublicChannel(player_ids)
        self._channels[public.name] = public

        # Wolf channel.
        if config.allow_wolf_chat:
            wolf = WolfChannel(wolf_ids)
            self._channels[wolf.name] = wolf

        # Direct message channels between every pair of players.
        if config.allow_dms:
            for i, p1 in enumerate(player_ids):
                for p2 in player_ids[i + 1 :]:
                    dm = DirectMessageChannel(p1, p2)
                    self._channels[dm.name] = dm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _phase_from_name(phase_name: str) -> Phase | None:
    """Convert a phase name string to a Phase enum member, or None."""
    if not phase_name:
        return None
    try:
        return Phase[phase_name]
    except KeyError:
        # Try case-insensitive lookup.
        upper = phase_name.upper()
        try:
            return Phase[upper]
        except KeyError:
            return None
