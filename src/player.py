from abc import ABC, abstractmethod
from typing import List, Tuple

try:
    from src.constants import Move
except ImportError:
    from constants import Move


class Player(ABC):
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.run_score = 0
        self.score = 0
        self.strategy_name = "None"
        self.game_history: List[Tuple[Move, Move]] = []

    @abstractmethod
    def play(self) -> Move:
        """Returns either Move.COOPERATE or Move.DEFECT."""
        pass

    def update_score(self, score: int):
        self.score += score

    def update_history(self, moves: Tuple[Move, Move]):
        """Records moves: (my_move, opp_move)"""
        self.game_history.append(moves)

    def finish_game(self):
        """Called at the end of a match/game."""
        self.run_score += self.score
        self.score = 0
        self.game_history = []

    def reset(self):
        """Full reset of scores and history."""
        self.run_score = 0
        self.score = 0
        self.game_history = []

    def copy(self) -> "Player":
        return self.__class__(name=self.name)
