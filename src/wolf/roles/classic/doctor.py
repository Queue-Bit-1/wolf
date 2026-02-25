"""Doctor role -- the village protector."""

from __future__ import annotations

from wolf.engine.phase import Phase
from wolf.roles.base import AbilityDefinition, RoleBase, Team
from wolf.roles.registry import RoleRegistry


@RoleRegistry.register
class Doctor(RoleBase):
    """Village doctor who can protect one player from death each night."""

    @property
    def name(self) -> str:
        return "doctor"

    @property
    def team(self) -> str:
        return Team.VILLAGE

    @property
    def description(self) -> str:
        return (
            "You are the doctor. Each night you may choose one player to "
            "protect from being killed by the werewolves. Choose wisely to "
            "keep key village members alive."
        )

    @property
    def abilities(self) -> list[AbilityDefinition]:
        return [
            AbilityDefinition(
                name="protect",
                phase=Phase.NIGHT,
                priority=10,
                description="Protect a player from being killed tonight",
                targets="alive_any",
            ),
        ]

    @property
    def prompt_instructions(self) -> str:
        return (
            "You are the doctor, a vital protective role for the village.\n\n"
            "NIGHT PHASE:\n"
            "- Choose one player to protect. If the werewolves target that "
            "player, they will survive the night.\n"
            "- You may protect yourself if you believe you are a target.\n\n"
            "DAY PHASE:\n"
            "- Pay attention to discussion to figure out who might be "
            "targeted next.\n"
            "- Be cautious about revealing your role -- if the wolves know "
            "who the doctor is, they can work around your protection.\n"
            "- Consider protecting players who are contributing valuable "
            "information or who you suspect might be the seer.\n"
            "- If a player survives a night attack, avoid immediately "
            "claiming credit, as this confirms your identity to the wolves.\n"
            "- Vary your protection targets when possible to keep the "
            "werewolves guessing."
        )

    def resolve_ability(
        self, ability_name: str, user_id: str, target_id: str, game_state
    ) -> list:
        # Protection is handled by the night resolver which checks whether
        # the doctor's target matches the wolf kill target.
        return []
