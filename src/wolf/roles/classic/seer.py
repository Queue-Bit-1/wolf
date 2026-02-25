"""Seer role -- the village investigator."""

from __future__ import annotations

from wolf.engine.events import PrivateRevealEvent
from wolf.engine.phase import Phase
from wolf.roles.base import AbilityDefinition, RoleBase, Team
from wolf.roles.registry import RoleRegistry


@RoleRegistry.register
class Seer(RoleBase):
    """Village seer who can investigate one player each night."""

    @property
    def name(self) -> str:
        return "seer"

    @property
    def team(self) -> str:
        return Team.VILLAGE

    @property
    def description(self) -> str:
        return (
            "You are the seer. Each night you may investigate one player to "
            "learn which team they belong to. Use this information wisely to "
            "guide the village toward eliminating the werewolves."
        )

    @property
    def abilities(self) -> list[AbilityDefinition]:
        return [
            AbilityDefinition(
                name="investigate",
                phase=Phase.NIGHT,
                priority=20,
                description="Investigate a player to learn their team",
                targets="alive_others",
            ),
        ]

    @property
    def prompt_instructions(self) -> str:
        return (
            "You are the seer, the village's most powerful investigative role.\n\n"
            "NIGHT PHASE:\n"
            "- Choose one player to investigate. You will learn whether they "
            "are on the village team or the werewolf team.\n\n"
            "DAY PHASE:\n"
            "- Use the information you have gathered strategically.\n"
            "- Be cautious about revealing your role too early -- if the "
            "werewolves learn you are the seer, you become their primary "
            "target.\n"
            "- Consider sharing partial information or hinting at your "
            "findings without directly claiming to be the seer.\n"
            "- If you have confirmed a werewolf, find a way to guide the "
            "village vote toward that player without immediately exposing "
            "yourself.\n"
            "- If you are about to be eliminated, reveal your findings to "
            "give the village as much information as possible.\n"
            "- Keep track of all your investigations and their results to "
            "build a clear picture of the game state."
        )

    def resolve_ability(
        self, ability_name: str, user_id: str, target_id: str, game_state
    ) -> list:
        if ability_name != "investigate":
            return []

        # Look up the target's role to determine their team.
        target_player = game_state.get_player(target_id)
        if target_player is None:
            return []

        info = f"You investigated {target_id}: they are on the {target_player.team} team."
        return [
            PrivateRevealEvent(
                player_id=user_id,
                info=info,
                phase=Phase.NIGHT,
                day=game_state.day,
            ),
        ]
