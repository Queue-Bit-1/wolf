"""Night-action resolution logic."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wolf.engine.events import (
    EliminationEvent,
    NightResultEvent,
    PrivateRevealEvent,
)
from wolf.engine.phase import Phase

if TYPE_CHECKING:
    from wolf.engine.actions import UseAbilityAction
    from wolf.engine.events import GameEvent
    from wolf.engine.state import GameState


@dataclass
class NightAction:
    """A single resolved night action with its priority."""

    player_id: str
    ability: str
    target_id: str
    priority: int


def resolve_night(
    state: GameState,
    role_registry: object,
) -> tuple[GameState, list[GameEvent]]:
    """Resolve all night actions stored in *state* and return the updated
    state together with a list of events produced during resolution.

    Resolution order is determined by ability priority (lower = first):
        * Doctor protect  (priority 10)
        * Wolf kill        (priority 15)
        * Seer investigate (priority 20)

    If the doctor protects the same target the wolves chose to kill, the
    kill is cancelled ("saved").
    """

    # ------------------------------------------------------------------
    # 1. Build sorted NightAction list
    # ------------------------------------------------------------------
    night_actions: list[NightAction] = []
    for player_id, action in state.night_actions.items():
        # Only UseAbilityAction contributes to resolution.
        from wolf.engine.actions import UseAbilityAction

        if not isinstance(action, UseAbilityAction):
            continue

        # Look up priority from the role registry.
        player = state.get_player(player_id)
        if player is None:
            continue

        role = role_registry.get(player.role)  # type: ignore[union-attr]
        priority = _get_ability_priority(role, action.ability_name)

        night_actions.append(
            NightAction(
                player_id=player_id,
                ability=action.ability_name,
                target_id=action.target_id,
                priority=priority,
            )
        )

    night_actions.sort(key=lambda na: na.priority)

    # ------------------------------------------------------------------
    # 2. Walk through actions in priority order, tracking effects
    # ------------------------------------------------------------------
    protected: set[str] = set()
    kills: list[str] = []  # player_ids targeted for killing
    saved: list[str] = []
    events: list[GameEvent] = []

    wolf_kill_votes: list[str] = []  # all individual wolf target submissions

    for na in night_actions:
        if na.ability == "protect":
            protected.add(na.target_id)
        elif na.ability == "kill":
            wolf_kill_votes.append(na.target_id)
        elif na.ability == "investigate":
            # Seer gets a private reveal about the target's role.
            target_player = state.get_player(na.target_id)
            if target_player is not None:
                events.append(
                    PrivateRevealEvent(
                        day=state.day,
                        phase=Phase.DAWN,
                        player_id=na.player_id,
                        info=(
                            f"{target_player.name} ({na.target_id}) is a "
                            f"{target_player.role}."
                        ),
                    )
                )

    # ------------------------------------------------------------------
    # 3. Wolves coordinate: majority/unanimous vote picks ONE kill target
    #
    # Rules:
    #   - 1 wolf:  that wolf's pick is used (trivially unanimous).
    #   - 2 wolves: both must agree (unanimous) for a kill.
    #   - 3+ wolves: strict majority (>50%) must agree on one target.
    # If no consensus is reached, no kill happens that night.
    # ------------------------------------------------------------------
    if wolf_kill_votes:
        num_wolves = len(wolf_kill_votes)
        tally = Counter(wolf_kill_votes)
        max_votes = tally.most_common(1)[0][1]

        if num_wolves <= 2:
            # Unanimous required
            required = num_wolves
        else:
            # Strict majority
            required = (num_wolves // 2) + 1

        if max_votes >= required:
            top = [pid for pid, cnt in tally.items() if cnt == max_votes]
            chosen_target = random.choice(top)
            kills.append(chosen_target)
        # else: no consensus — no kill tonight

    # ------------------------------------------------------------------
    # 4. Determine actual kills (cancel if protected)
    # ------------------------------------------------------------------
    actual_kills: list[str] = []
    for target_id in kills:
        if target_id in protected:
            saved.append(target_id)
        else:
            if target_id not in actual_kills:
                actual_kills.append(target_id)

    # ------------------------------------------------------------------
    # 4. Apply kills to state and produce elimination events
    # ------------------------------------------------------------------
    new_state = state
    for target_id in actual_kills:
        target_player = new_state.get_player(target_id)
        if target_player is not None and target_player.is_alive:
            new_state = new_state.with_player_killed(target_id)
            events.append(
                EliminationEvent(
                    day=new_state.day,
                    phase=Phase.DAWN,
                    player_id=target_id,
                    role=target_player.role,
                    cause="wolf_kill",
                )
            )

    # ------------------------------------------------------------------
    # 5. Emit summary NightResultEvent
    # ------------------------------------------------------------------
    events.insert(
        0,
        NightResultEvent(
            day=state.day,
            phase=Phase.DAWN,
            kills=actual_kills,
            protected=list(protected),
            saved=saved,
        ),
    )

    # Append all events to the new state
    for event in events:
        new_state = new_state.with_event(event)

    return new_state, events


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_ability_priority(role: object, ability_name: str) -> int:
    """Extract the priority for *ability_name* from a role object.

    Falls back to ``50`` if the ability is not found.
    """
    for ability in getattr(role, "abilities", []):
        if ability.name == ability_name:
            return ability.priority
    return 50
