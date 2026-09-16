import random
from typing import Dict, Type

try:
    from src.player import Player
    from src.constants import Move
except ImportError:
    from player import Player
    from constants import Move


class AlwaysCooperatePlayer(Player):
    """Always cooperates."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Always Cooperate"

    def play(self) -> Move:
        return Move.COOPERATE


class AlwaysDefectPlayer(Player):
    """Always defects."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Always Defect"

    def play(self) -> Move:
        return Move.DEFECT


class TitForTatPlayer(Player):
    """Cooperates on round 1, then mimics the opponent's previous move."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Tit For Tat"

    def play(self) -> Move:
        if len(self.game_history) == 0:
            return Move.COOPERATE
        return self.game_history[-1][1]


class SuspiciousTitForTatPlayer(Player):
    """Defects on round 1, then mimics the opponent's previous move."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Suspicious Tit For Tat"

    def play(self) -> Move:
        if len(self.game_history) == 0:
            return Move.DEFECT
        return self.game_history[-1][1]


class TitForTwoTatsPlayer(Player):
    """Cooperates unless the opponent defected in BOTH of the last two rounds."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Tit For Two Tats"

    def play(self) -> Move:
        if len(self.game_history) < 2:
            return Move.COOPERATE
        if (
            self.game_history[-1][1] == Move.DEFECT
            and self.game_history[-2][1] == Move.DEFECT
        ):
            return Move.DEFECT
        return Move.COOPERATE


class TatForTitPlayer(Player):
    """Inverts what the opponent did in the last round."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Tat For Tit"

    def play(self) -> Move:
        if len(self.game_history) == 0:
            return Move.COOPERATE
        return (
            Move.DEFECT
            if self.game_history[-1][1] == Move.COOPERATE
            else Move.COOPERATE
        )


class RandomPlayer(Player):
    """Cooperates or defects with 50% probability."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Random"

    def play(self) -> Move:
        return random.choice([Move.COOPERATE, Move.DEFECT])


class DetectivePlayer(Player):
    """Probes with Cooperate, Defect, Cooperate, Cooperate. 
    If the opponent ever defects during rounds 1-4, plays Tit For Tat.
    If the opponent never defects, permanently defects to exploit them."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Detective"

    def play(self) -> Move:
        round_idx = len(self.game_history)
        if round_idx == 0:
            return Move.COOPERATE
        elif round_idx == 1:
            return Move.DEFECT
        elif round_idx == 2:
            return Move.COOPERATE
        elif round_idx == 3:
            return Move.COOPERATE
        else:
            # Check if opponent ever defected in the first 4 rounds
            opp_defected = any(h[1] == Move.DEFECT for h in self.game_history[:4])
            if opp_defected:
                return self.game_history[-1][1]  # TFT
            else:
                return Move.DEFECT  # Exploit sucker


class GrimTriggerPlayer(Player):
    """Cooperates until the opponent defects even once, then defects forever."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Grim Trigger"

    def play(self) -> Move:
        if any(h[1] == Move.DEFECT for h in self.game_history):
            return Move.DEFECT
        return Move.COOPERATE


class PavlovPlayer(Player):
    """Win-Stay, Lose-Shift: Cooperates on round 1.
    If the previous round resulted in (C,C) or (D,D), repeats its own previous move.
    If the previous round resulted in (C,D) or (D,C), flips its own previous move."""

    def __init__(self, name=None):
        super().__init__(name)
        self.strategy_name = "Pavlov"

    def play(self) -> Move:
        if len(self.game_history) == 0:
            return Move.COOPERATE
        my_last, opp_last = self.game_history[-1]
        if my_last == opp_last:
            return my_last  # Win / Satisficing -> Stay
        else:
            return Move.DEFECT if my_last == Move.COOPERATE else Move.COOPERATE  # Lose -> Shift


STRATEGY_REGISTRY: Dict[str, Type[Player]] = {
    "AlwaysCooperate": AlwaysCooperatePlayer,
    "AlwaysDefect": AlwaysDefectPlayer,
    "TitForTat": TitForTatPlayer,
    "SuspiciousTitForTat": SuspiciousTitForTatPlayer,
    "TitForTwoTats": TitForTwoTatsPlayer,
    "TatForTit": TatForTitPlayer,
    "Random": RandomPlayer,
    "Detective": DetectivePlayer,
    "GrimTrigger": GrimTriggerPlayer,
    "Pavlov": PavlovPlayer,
}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "AlwaysCooperate": "Always plays COOPERATE unconditionally.",
    "AlwaysDefect": "Always plays DEFECT unconditionally.",
    "TitForTat": "Starts with COOPERATE, then copies whatever the opponent played in the previous round.",
    "SuspiciousTitForTat": "Starts with DEFECT, then copies whatever the opponent played in the previous round.",
    "TitForTwoTats": "Cooperates initially and only retaliates with DEFECT if the opponent defected in BOTH of the last two consecutive rounds.",
    "TatForTit": "Starts with COOPERATE, then plays the opposite of whatever the opponent played in the last round.",
    "Random": "Randomly chooses COOPERATE or DEFECT with equal probability (unpredictable).",
    "Detective": "Plays [C, D, C, C] as an opening probe sequence. If the opponent ever defected, it plays Tit For Tat; if the opponent never retaliated, it defects forever to exploit.",
    "GrimTrigger": "Starts with COOPERATE, but if the opponent defects even once, it permanently defects for the rest of the match.",
    "Pavlov": "Win-Stay, Lose-Shift. Repeats move if round was (C,C) or (D,D); shifts move if (C,D) or (D,C).",
}


def get_strategy_class(name: str) -> Type[Player]:
    normalized = name.replace(" ", "").replace("_", "").replace("-", "").lower()
    for key, cls in STRATEGY_REGISTRY.items():
        if key.lower() == normalized or cls.__name__.lower() == normalized:
            return cls
    raise ValueError(f"Unknown strategy name: '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
