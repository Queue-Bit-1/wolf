"""Phase enum for game state machine."""

from enum import Enum, auto


class Phase(Enum):
    """Game phases in order of progression."""

    SETUP = auto()
    NIGHT = auto()
    DAWN = auto()
    DAY_DISCUSSION = auto()
    DAY_VOTE = auto()
    DAY_VOTE_RESULT = auto()
    GAME_OVER = auto()
