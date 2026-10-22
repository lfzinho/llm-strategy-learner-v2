import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import axelrod as axl

from src.config import (
    NormalRoundsConfig, RoundsPerMatchConfig, TournamentScoringConfig, describe_rounds_per_match,
)
from src.match_runner import RoundSampler, play_match
from src.progress import ProgressTracker


class TournamentMatchResult:
    def __init__(
        self,
        player1_name: str,
        player2_name: str,
        match_index: int,
        player1_score: int,
        player2_score: int,
        rounds_played: int,
        moves: Optional[List[str]] = None,
        llm_turns: Optional[List[Dict[str, Any]]] = None,
    ):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.match_index = match_index
        self.player1_score = player1_score
        self.player2_score = player2_score
        self.rounds_played = rounds_played
        self.moves = moves or []
        self.llm_turns = llm_turns or []

        if player1_score > player2_score:
            self.winner = player1_name
        elif player2_score > player1_score:
            self.winner = player2_name
        else:
            self.winner = "Draw"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "player1": self.player1_name,
            "player2": self.player2_name,
            "match_index": self.match_index,
            "p1_score": self.player1_score,
            "p2_score": self.player2_score,
            "winner": self.winner,
            "rounds_played": self.rounds_played,
            "moves": self.moves,
        }
        if self.llm_turns:
            d["llm_turns"] = self.llm_turns
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TournamentMatchResult":
        return cls(d["player1"], d["player2"], d["match_index"], d["p1_score"], d["p2_score"],
                   d["rounds_played"], d.get("moves"), d.get("llm_turns"))


class RoundRobinTournament:
    """Round-robin championship for the iterated game.

    Every participant plays every other for N matches. Tracks chess-style match points
    (W=1.0, D=0.5, L=0.0) alongside total game points, a cross table and a ranked
    leaderboard. Progress is checkpointed after every match so a crash can be resumed.
    """

    def __init__(
        self,
        players: List[axl.Player],
        matches_per_pairing: int = 5,
        rounds_per_match: RoundsPerMatchConfig = NormalRoundsConfig(),
        error_rate: float = 0.0,
        scoring_config: Optional[TournamentScoringConfig] = None,
        seed: Optional[int] = None,
    ):
        self.players = players
        self.matches_per_pairing = matches_per_pairing
        self.rounds_per_match = rounds_per_match
        self.error_rate = error_rate
        self.scoring_config = scoring_config or TournamentScoringConfig()
        self.random = random.Random(seed)
        self.sampler = RoundSampler(rounds_per_match, self.random)

        self.match_results: List[TournamentMatchResult] = []
        self.progress = ProgressTracker()
        for player in self.players:
            if hasattr(player, "set_console"):
                player.set_console(self.progress.console)

    # ------------------------------------------------------------ checkpoint
    @staticmethod
    def _checkpoint_path(output_dir: str) -> str:
        return os.path.join(output_dir, "tournament_checkpoint.json")

    def _load_checkpoint(self, output_dir: str) -> List[TournamentMatchResult]:
        path = self._checkpoint_path(output_dir)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = [p.name for p in self.players]
            if data.get("player_names") != names or data.get("matches_per_pairing") != self.matches_per_pairing:
                raise ValueError("line-up differs from the saved run")
            return [TournamentMatchResult.from_dict(d) for d in data["matches"]]
        except Exception as e:
            print(f"WARNING: could not use tournament checkpoint {path} ({e}). Starting the tournament from scratch.")
            return []

    def _save_checkpoint(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(self._checkpoint_path(output_dir), "w", encoding="utf-8") as f:
            json.dump({
                "player_names": [p.name for p in self.players],
                "matches_per_pairing": self.matches_per_pairing,
                "matches": [m.to_dict() for m in self.match_results],
            }, f)

    # ------------------------------------------------------------------- run
    def run_tournament(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        n = len(self.players)
        pairings = list(itertools.combinations(self.players, 2))
        total_matches = len(pairings) * self.matches_per_pairing

        print("\n" + "=" * 60)
        print("PHASE 2: ROUND-ROBIN CHAMPIONSHIP TOURNAMENT")
        print(f"Participants: {n} | Pairings: {len(pairings)} | Matches/pair: {self.matches_per_pairing}")
        print("=" * 60)

        llm_players = [p for p in self.players if hasattr(p, "finish_match")]
        for p in llm_players:
            p.phase = "tournament"
            p.match_number = 0
        stats_before = {id(p): p.stats_snapshot() for p in llm_players}

        done: List[TournamentMatchResult] = self._load_checkpoint(output_dir) if output_dir else []
        if done:
            print(f"Resuming tournament from checkpoint: {len(done)}/{total_matches} matches already played.")
        self.match_results = []

        self.progress.start()
        self.progress.add_overall_task("Tournament progress", total=total_matches)
        self.progress.set_status("starting...")
        overall = 0
        try:
            for pair_idx, (p1, p2) in enumerate(pairings, 1):
                self.progress.console.print(
                    f"[Pairing {pair_idx}/{len(pairings)}] {p1.name} vs {p2.name} ({self.matches_per_pairing} matches)")
                for m_idx in range(self.matches_per_pairing):
                    overall += 1
                    rounds = self.sampler.sample()  # always drawn, so a resumed run keeps the same random stream
                    seed = self.random.randrange(2**31)
                    if overall <= len(done):
                        self.match_results.append(done[overall - 1])
                        self.progress.update_overall(overall)
                        continue
                    self.match_results.append(self._play_pair(p1, p2, m_idx, rounds, seed, overall))
                    if output_dir:
                        self._save_checkpoint(output_dir)
        finally:
            self.progress.stop()

        results = self._build_results()
        results["llm_stats"] = {
            p.name: p.stats_since(stats_before[id(p)]) for p in llm_players
        }
        results["llm_names"] = [p.name for p in llm_players]
        return results

    def _play_pair(self, p1, p2, m_idx: int, rounds: int, seed: int, overall: int) -> TournamentMatchResult:
        self.progress.reset_current(f"Current match: {p1.name} vs {p2.name}", rounds)
        llm_turns: List[Dict[str, Any]] = []
        for me, other in ((p1, p2), (p2, p1)):
            if hasattr(me, "begin_match"):
                me.begin_match(other.name, rounds, self.sampler.average)
            if hasattr(me, "turn_callback"):
                me.turn_callback = lambda r: self.progress.update_current(r - 1)
        try:
            result = play_match(p1, p2, rounds, seed=seed, noise=self.error_rate)
        finally:
            for p in (p1, p2):
                if hasattr(p, "turn_callback"):
                    p.turn_callback = None
        self.progress.update_current(rounds)
        self.progress.update_overall(overall)

        for me, other in ((p1, p2), (p2, p1)):
            if hasattr(me, "finish_match"):
                llm_turns = llm_turns or list(me.turn_log)
                me.finish_match(other)
        self.progress.set_status(
            f"{p1.name} {result.score1} - {result.score2} {p2.name} ({rounds} rounds)")
        return TournamentMatchResult(
            p1.name, p2.name, m_idx + 1, result.score1, result.score2, rounds,
            moves=["".join(m) for m in result.moves], llm_turns=llm_turns,
        )

    # ------------------------------------------------------------- reporting
    def _build_results(self) -> Dict[str, Any]:
        sc = self.scoring_config
        names = [p.name for p in self.players]
        stats: Dict[str, Dict[str, Any]] = {
            p.name: {
                "name": p.name,
                "strategy_name": getattr(p, "strategy_name", p.name),
                "matches_played": 0, "wins": 0, "draws": 0, "losses": 0,
                "tournament_points": 0.0, "total_ipd_score": 0, "total_rounds": 0,
            }
            for p in self.players
        }
        h2h: Dict[Tuple[str, str], Dict[str, Any]] = {
            (a, b): {"p1_total_score": 0, "p2_total_score": 0, "wins": 0, "draws": 0, "losses": 0, "matches_played": 0}
            for a in names for b in names if a != b
        }

        for m in self.match_results:
            a, b = m.player1_name, m.player2_name
            s1, s2 = m.player1_score, m.player2_score
            for who, score in ((a, s1), (b, s2)):
                stats[who]["matches_played"] += 1
                stats[who]["total_ipd_score"] += score
                stats[who]["total_rounds"] += m.rounds_played
            h2h[(a, b)]["p1_total_score"] += s1
            h2h[(a, b)]["p2_total_score"] += s2
            h2h[(b, a)]["p1_total_score"] += s2
            h2h[(b, a)]["p2_total_score"] += s1
            h2h[(a, b)]["matches_played"] += 1
            h2h[(b, a)]["matches_played"] += 1

            if s1 == s2:
                for who in (a, b):
                    stats[who]["draws"] += 1
                    stats[who]["tournament_points"] += sc.draw_match_points
                h2h[(a, b)]["draws"] += 1
                h2h[(b, a)]["draws"] += 1
            else:
                winner, loser = (a, b) if s1 > s2 else (b, a)
                stats[winner]["wins"] += 1
                stats[loser]["losses"] += 1
                stats[winner]["tournament_points"] += sc.win_match_points
                stats[loser]["tournament_points"] += sc.loss_match_points
                h2h[(winner, loser)]["wins"] += 1
                h2h[(loser, winner)]["losses"] += 1

        leaderboard = []
        for s in stats.values():
            mp, tr = s["matches_played"], s["total_rounds"]
            s["avg_score_per_match"] = round(s["total_ipd_score"] / mp, 2) if mp else 0.0
            s["avg_score_per_round"] = round(s["total_ipd_score"] / tr, 2) if tr else 0.0
            leaderboard.append(s)
        leaderboard.sort(key=lambda x: (x["tournament_points"], x["total_ipd_score"]), reverse=True)
        for rank, item in enumerate(leaderboard, 1):
            item["rank"] = rank

        return {
            "leaderboard": leaderboard,
            "player_names": names,
            "cross_table": {f"{k[0]}_vs_{k[1]}": v for k, v in h2h.items()},
            "all_matches": [m.to_dict() for m in self.match_results],
            "total_matches": len(self.match_results),
            "matches_per_pairing": self.matches_per_pairing,
            "rounds_per_match": describe_rounds_per_match(self.rounds_per_match),
        }
