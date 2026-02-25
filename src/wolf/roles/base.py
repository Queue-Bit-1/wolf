"""Base role definitions and ability framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wolf.engine.phase import Phase


@dataclass(frozen=True)
class AbilityDefinition:
    """Defines a role's usable ability."""

    name: str
    phase: Phase  # when this ability can be used
    priority: int  # lower = resolves first
    description: str = ""
    targets: str = "alive_others"  # "alive_others", "alive_any", "alive_wolves", etc.


class Team:
    """Team constants."""

    VILLAGE = "village"
    WEREWOLF = "werewolf"


class RoleBase(ABC):
    """Abstract base for all game roles."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def team(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def abilities(self) -> list[AbilityDefinition]:
        """Return the list of abilities this role has."""
        return []

    @property
    def prompt_instructions(self) -> str:
        """Role-specific instructions for the LLM agent prompt."""
        return self.description

    def resolve_ability(
        self, ability_name: str, user_id: str, target_id: str, game_state
    ) -> list:
        """Resolve an ability use. Returns list of GameEvents."""
        return []

    def on_death(self, player_id: str, game_state) -> list:
        """Called when a player with this role dies. Returns list of GameEvents."""
        return []
