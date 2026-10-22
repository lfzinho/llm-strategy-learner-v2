import itertools
import random
from typing import Any, Dict, List, Optional, Tuple
try:
    from src.player import Player
    from src.game_machine import GameMachine
    from src.config import NormalRoundsConfig, RoundsPerMatchConfig, TournamentScoringConfig, describe_rounds_per_match
    from src.progress import ProgressTracker
except ImportError:
    from player import Player
    from game_machine import GameMachine
    from config import NormalRoundsConfig, RoundsPerMatchConfig, TournamentScoringConfig, describe_rounds_per_match
    from progress import ProgressTracker


class TournamentMatchResult:
    def __init__(
        self,
        player1_name: str,
        player2_name: str,
        match_index: int,
        player1_score: int,
        player2_score: int,
        rounds_played: int,
    ):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.match_index = match_index
        self.player1_score = player1_score
        self.player2_score = player2_score
        self.rounds_played = rounds_played

        if player1_score > player2_score:
            self.winner = player1_name
        elif player2_score > player1_score:
            self.winner = player2_name
        else:
            self.winner = "Draw"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player1": self.player1_name,
            "player2": self.player2_name,
            "match_index": self.match_index,
            "p1_score": self.player1_score,
            "p2_score": self.player2_score,
            "winner": self.winner,
            "rounds_played": self.rounds_played,
        }


class RoundRobinTournament:
    """Round-Robin Chess Championship Tournament for Iterated Prisoner's Dilemma.
    
    Every participant plays against every other participant for N matches of R rounds.
    Tracks chess-style tournament points (W=1.0, D=0.5, L=0.0) alongside total IPD game payoffs,
    and produces an all-play-all cross-table matrix and ranked leaderboard.
    """

    def __init__(
        self,
        players: List[Player],
        matches_per_pairing: int = 5,
        rounds_per_match: RoundsPerMatchConfig = NormalRoundsConfig(),
        error_rate: float = 0.0,
        scoring_config: Optional[TournamentScoringConfig] = None,
    ):
        self.players = players
        self.matches_per_pairing = matches_per_pairing
        self.rounds_per_match = rounds_per_match
        self.game_machine = GameMachine(error_rate=error_rate)
        self.scoring_config = scoring_config or TournamentScoringConfig()

        self.match_results: List[TournamentMatchResult] = []
        # Key: (p1_name, p2_name) -> dict with scores and W-D-L from p1's perspective
        self.head_to_head: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.progress = ProgressTracker()
        self.random = random.Random()

        for player in self.players:
            if hasattr(player, "set_console"):
                player.set_console(self.progress.console)

        # Reset all player scores
        for p in self.players:
            p.reset()

    def run_tournament(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print(f"🏆 PHASE 2: ROUND-ROBIN CHAMPIONSHIP TOURNAMENT")
        print(f"Participants: {len(self.players)} | Pairings: {len(self.players) * (len(self.players) - 1) // 2} | Matches/Pair: {self.matches_per_pairing}")
        print("=" * 60)

        total_matches = len(self.players) * (len(self.players) - 1) // 2 * self.matches_per_pairing
        self.progress.start()
        self.progress.add_overall_task("Tournament progress", total=total_matches)

        # Initialize head to head table
        player_names = [p.name for p in self.players]
        for p1_name in player_names:
            for p2_name in player_names:
                if p1_name != p2_name:
                    self.head_to_head[(p1_name, p2_name)] = {
                        "p1_total_score": 0,
                        "p2_total_score": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "matches_played": 0,
                    }

        # Stats per player
        player_stats: Dict[str, Dict[str, Any]] = {
            p.name: {
                "name": p.name,
                "strategy_name": getattr(p, "strategy_name", p.name),
                "matches_played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "tournament_points": 0.0,
                "total_ipd_score": 0,
                "total_rounds": 0,
            }
            for p in self.players
        }

        # Generate all pairings (combinations of 2 players)
        pairings = list(itertools.combinations(self.players, 2))
        total_pairings = len(pairings)
        overall_match_count = 0

        for pair_idx, (p1, p2) in enumerate(pairings, 1):
            print(f"\n[Pairing {pair_idx}/{total_pairings}] {p1.name} vs {p2.name} ({self.matches_per_pairing} matches)...")

            for m_idx in range(self.matches_per_pairing):
                overall_match_count += 1
                rounds_this_match = self._sample_rounds()
                self.progress.reset_current(f"Current match: {p1.name} vs {p2.name}", rounds_this_match)

                # Inform players of opponent if supported
                if hasattr(p1, "last_opponent_name"):
                    p1.last_opponent_name = p2.name
                if hasattr(p2, "last_opponent_name"):
                    p2.last_opponent_name = p1.name

                for round_idx in range(rounds_this_match):
                    self.game_machine.play_game(p1, p2)
                    self.progress.update_current(round_idx + 1)

                self.progress.update_overall(overall_match_count)
                s1 = p1.score
                s2 = p2.score

                # Finish match for both players
                if hasattr(p1, "finish_game") and "opponent_score" in p1.finish_game.__code__.co_varnames:
                    p1.finish_game(opponent_score=s2)
                else:
                    p1.finish_game()

                if hasattr(p2, "finish_game") and "opponent_score" in p2.finish_game.__code__.co_varnames:
                    p2.finish_game(opponent_score=s1)
                else:
                    p2.finish_game()

                # Record match
                res = TournamentMatchResult(
                    player1_name=p1.name,
                    player2_name=p2.name,
                    match_index=m_idx + 1,
                    player1_score=s1,
                    player2_score=s2,
                    rounds_played=rounds_this_match,
                )
                self.match_results.append(res)

                # Update stats
                player_stats[p1.name]["matches_played"] += 1
                player_stats[p2.name]["matches_played"] += 1
                player_stats[p1.name]["total_ipd_score"] += s1
                player_stats[p2.name]["total_ipd_score"] += s2
                player_stats[p1.name]["total_rounds"] += rounds_this_match
                player_stats[p2.name]["total_rounds"] += rounds_this_match

                # Update Head-to-Head
                h2h_12 = self.head_to_head[(p1.name, p2.name)]
                h2h_21 = self.head_to_head[(p2.name, p1.name)]

                h2h_12["p1_total_score"] += s1
                h2h_12["p2_total_score"] += s2
                h2h_12["matches_played"] += 1

                h2h_21["p1_total_score"] += s2
                h2h_21["p2_total_score"] += s1
                h2h_21["matches_played"] += 1

                if s1 > s2:
                    player_stats[p1.name]["wins"] += 1
                    player_stats[p2.name]["losses"] += 1
                    player_stats[p1.name]["tournament_points"] += self.scoring_config.win_match_points
                    player_stats[p2.name]["tournament_points"] += self.scoring_config.loss_match_points
                    h2h_12["wins"] += 1
                    h2h_21["losses"] += 1
                elif s2 > s1:
                    player_stats[p2.name]["wins"] += 1
                    player_stats[p1.name]["losses"] += 1
                    player_stats[p2.name]["tournament_points"] += self.scoring_config.win_match_points
                    player_stats[p1.name]["tournament_points"] += self.scoring_config.loss_match_points
                    h2h_21["wins"] += 1
                    h2h_12["losses"] += 1
                else:
                    player_stats[p1.name]["draws"] += 1
                    player_stats[p2.name]["draws"] += 1
                    player_stats[p1.name]["tournament_points"] += self.scoring_config.draw_match_points
                    player_stats[p2.name]["tournament_points"] += self.scoring_config.draw_match_points
                    h2h_12["draws"] += 1
                    h2h_21["draws"] += 1

        self.progress.stop()

        # Calculate averages and sort leaderboard
        leaderboard = []
        for p_name, stats in player_stats.items():
            mp = stats["matches_played"]
            tr = stats["total_rounds"]
            stats["avg_score_per_match"] = round(stats["total_ipd_score"] / mp, 2) if mp > 0 else 0.0
            stats["avg_score_per_round"] = round(stats["total_ipd_score"] / tr, 2) if tr > 0 else 0.0
            leaderboard.append(stats)

        # Sort primarily by Tournament Points, secondarily by Total IPD Score
        leaderboard.sort(key=lambda x: (x["tournament_points"], x["total_ipd_score"]), reverse=True)

        for rank, item in enumerate(leaderboard, 1):
            item["rank"] = rank

        # Format cross table dict with string keys for JSON serialization
        cross_table_serializable = {
            f"{k[0]}_vs_{k[1]}": v for k, v in self.head_to_head.items()
        }

        return {
            "leaderboard": leaderboard,
            "player_names": player_names,
            "cross_table": cross_table_serializable,
            "all_matches": [m.to_dict() for m in self.match_results],
            "total_matches": len(self.match_results),
            "matches_per_pairing": self.matches_per_pairing,
            "rounds_per_match": describe_rounds_per_match(self.rounds_per_match),
        }

    def _sample_rounds(self) -> int:
        if isinstance(self.rounds_per_match, NormalRoundsConfig):
            return max(1, round(self.random.gauss(self.rounds_per_match.average, self.rounds_per_match.std_dev)))
        return self.rounds_per_match
