"""Werewolf role -- the core antagonist role."""

from __future__ import annotations

from wolf.engine.phase import Phase
from wolf.roles.base import AbilityDefinition, RoleBase, Team
from wolf.roles.registry import RoleRegistry


@RoleRegistry.register
class Werewolf(RoleBase):
    """Werewolf that hunts villagers at night."""

    @property
    def name(self) -> str:
        return "werewolf"

    @property
    def team(self) -> str:
        return Team.WEREWOLF

    @property
    def description(self) -> str:
        return (
            "You are a werewolf. During the night, you and your fellow wolves "
            "choose a victim to kill. During the day, blend in with villagers "
            "and avoid suspicion. Coordinate with other wolves."
        )

    @property
    def abilities(self) -> list[AbilityDefinition]:
        return [
            AbilityDefinition(
                name="kill",
                phase=Phase.NIGHT,
                priority=15,
                description="Choose a player to kill",
                targets="alive_others",
            ),
        ]

    @property
    def prompt_instructions(self) -> str:
        return (
            "You are a werewolf. Your goal is to eliminate all villagers "
            "without being discovered.\n\n"
            "NIGHT PHASE:\n"
            "- Coordinate with your fellow wolves to choose a target.\n"
            "- Prioritize eliminating powerful village roles (seer, doctor) "
            "if you can identify them.\n\n"
            "DAY PHASE:\n"
            "- Blend in with the villagers. Act as though you are a regular "
            "villager or claim a village role if necessary.\n"
            "- Cast suspicion on others to divert attention from yourself "
            "and your fellow wolves.\n"
            "- Avoid defending other wolves too aggressively, as this can "
            "link you together.\n"
            "- Pay attention to voting patterns and use them to appear "
            "cooperative.\n\n"
            "GENERAL DECEPTION TIPS:\n"
            "- Be consistent with your claims across rounds.\n"
            "- Participate actively in discussion -- silence draws suspicion.\n"
            "- If accused, stay calm and provide logical counter-arguments.\n"
            "- Use the wolf chat at night to align your story with other wolves."
        )

    def resolve_ability(
        self, ability_name: str, user_id: str, target_id: str, game_state
    ) -> list:
        # Kills are handled by the resolver since multiple wolves vote on
        # the same target.  Individual resolve returns nothing.
        return []
