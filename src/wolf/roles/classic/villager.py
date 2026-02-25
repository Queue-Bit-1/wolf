"""Villager role -- the basic village-team role with no abilities."""

from __future__ import annotations

from wolf.roles.base import RoleBase, Team
from wolf.roles.registry import RoleRegistry


@RoleRegistry.register
class Villager(RoleBase):
    """Plain villager with no special powers."""

    @property
    def name(self) -> str:
        return "villager"

    @property
    def team(self) -> str:
        return Team.VILLAGE

    @property
    def description(self) -> str:
        return (
            "A regular villager. You have no special abilities but can "
            "participate in discussion and voting. Use your reasoning to "
            "identify the werewolves."
        )
