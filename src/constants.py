from enum import Enum


class Move(Enum):
    COOPERATE = 0
    DEFECT = 1

    @classmethod
    def from_str(cls, val: str) -> "Move":
        normalized = val.strip().upper()
        if "COOP" in normalized:
            return cls.COOPERATE
        elif "DEF" in normalized:
            return cls.DEFECT
        raise ValueError(f"Unknown move string: {val}")

    def __str__(self) -> str:
        return "COOPERATE" if self == Move.COOPERATE else "DEFECT"


DEFAULT_PAYOFF_MATRIX = {
    Move.COOPERATE: {Move.COOPERATE: (3, 3), Move.DEFECT: (0, 5)},
    Move.DEFECT: {Move.COOPERATE: (5, 0), Move.DEFECT: (1, 1)},
}
