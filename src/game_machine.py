import random
from typing import Dict, Optional, Tuple

try:
    from src.constants import Move, DEFAULT_PAYOFF_MATRIX
except ImportError:
    from constants import Move, DEFAULT_PAYOFF_MATRIX


class GameMachine:
    """Pure, deterministic Iterated Prisoner's Dilemma game engine.
    
    Zero environmental noise by default, guaranteeing predictable payoffs.
    """

    def __init__(
        self,
        payoff_matrix: Optional[Dict[Move, Dict[Move, Tuple[int, int]]]] = None,
        error_rate: float = 0.0,
    ):
        self.payoff_matrix = payoff_matrix or DEFAULT_PAYOFF_MATRIX
        self.error_rate = error_rate

    def get_payoff(self, p1_action: Move, p2_action: Move) -> Tuple[int, int]:
        return self.payoff_matrix[p1_action][p2_action]

    def _apply_noise(self, action: Move) -> Move:
        if self.error_rate > 0.0 and random.random() < self.error_rate:
            return Move.DEFECT if action == Move.COOPERATE else Move.COOPERATE
        return action

    def play_game(self, player1, player2) -> Tuple[Tuple[Move, Move], Tuple[int, int]]:
        """Executes a single round between player1 and player2.
        
        Returns:
            ((p1_executed, p2_executed), (p1_payoff, p2_payoff))
        """
        p1_action = player1.play()
        p2_action = player2.play()

        p1_executed = self._apply_noise(p1_action) if self.error_rate > 0 else p1_action
        p2_executed = self._apply_noise(p2_action) if self.error_rate > 0 else p2_action

        p1_payoff, p2_payoff = self.get_payoff(p1_executed, p2_executed)

        # Update clean 2-tuple history (my_move, opp_move)
        player1.update_history((p1_executed, p2_executed))
        player2.update_history((p2_executed, p1_executed))

        player1.update_score(p1_payoff)
        player2.update_score(p2_payoff)

        return (p1_executed, p2_executed), (p1_payoff, p2_payoff)
