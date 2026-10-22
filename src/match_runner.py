"""Match playing (on `axelrod.Match`) and match-length sampling."""
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import axelrod as axl

from src.config import GeometricRoundsConfig, NormalRoundsConfig, RoundsPerMatchConfig


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


class RoundSampler:
    """Draws match lengths from the configured distribution with a seeded RNG."""

    def __init__(self, config: RoundsPerMatchConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()

    def sample(self) -> int:
        cfg = self.config
        if isinstance(cfg, NormalRoundsConfig):
            return max(1, round(self.rng.gauss(cfg.average, cfg.std_dev)))
        if isinstance(cfg, GeometricRoundsConfig):
            u = self.rng.random()
            return min(cfg.max_rounds, max(1, math.ceil(math.log(1 - u) / math.log(1 - cfg.prob_end))))
        return cfg

    @property
    def average(self) -> float:
        """Expected length, used by prompts that give an approximate horizon."""
        cfg = self.config
        if isinstance(cfg, NormalRoundsConfig):
            return cfg.average
        if isinstance(cfg, GeometricRoundsConfig):
            return round(1 / cfg.prob_end, 1)
        return float(cfg)


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
