"""Single entry point for playing a match, built on `axelrod.Match`."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import axelrod as axl


@dataclass
class MatchResult:
    moves: List[Tuple[str, str]]  # per round, e.g. ("C", "D")
    score1: int
    score2: int

    @property
    def rounds(self) -> int:
        return len(self.moves)

    def cooperation_rate(self, player_index: int = 0) -> float:
        if not self.moves:
            return 0.0
        return sum(1 for m in self.moves if m[player_index] == "C") / len(self.moves)


def play_match(
    player1: axl.Player,
    player2: axl.Player,
    turns: int,
    seed: Optional[int] = None,
    noise: float = 0.0,
) -> MatchResult:
    """Plays `turns` rounds with the standard payoffs (R=3, S=0, T=5, P=1)."""
    match = axl.Match((player1, player2), turns=turns, noise=noise, seed=seed)
    outcome = match.play()
    s1, s2 = match.final_score()
    moves = [(str(a), str(b)) for a, b in outcome]
    return MatchResult(moves=moves, score1=int(s1), score2=int(s2))
