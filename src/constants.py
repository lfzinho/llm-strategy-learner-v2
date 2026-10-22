from enum import Enum


class Move(Enum):
    """Internal move type. Prompts show these as Alpha (cooperate) and Beta (defect)."""
    COOPERATE = 0
    DEFECT = 1
