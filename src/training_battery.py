import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from src.config import NormalRoundsConfig, RoundsPerMatchConfig
from src.dreamer import Dreamer
from src.match_runner import RoundSampler, play_match
from src.opponents import make_opponent, resolve_key
from src.progress import ProgressTracker
from src.scorecard import build_scorecard, print_scorecard, records_from_training, write_scorecard

CHECKPOINT_VERSION = 2


class TrainingBattery:
    """Structured sequential block training battery.

    Plays N consecutive matches against strategy 1, then N against strategy 2, and so on,
    so the LLM can learn each opponent in depth. Afterwards the Dreamer consolidates the
    memory into a playbook. The run can be resumed from `training_checkpoint.json`.
    """

    def __init__(
        self,
        llm_player,
        strategies: Optional[List[str]] = None,
        matches_per_strategy: int = 5,
        rounds_per_match: RoundsPerMatchConfig = NormalRoundsConfig(),
        error_rate: float = 0.0,
        dreamer_enabled: bool = True,
        dreamer_temperature: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.llm_player = llm_player
        self.matches_per_strategy = matches_per_strategy
        self.error_rate = error_rate
        self.dreamer_enabled = dreamer_enabled
        self.dreamer_temperature = dreamer_temperature
        self.random = random.Random(seed)
        self.sampler = RoundSampler(rounds_per_match, self.random)

        self.strategy_names = strategies or [
            "AlwaysCooperate", "AlwaysDefect", "TitForTat", "SuspiciousTitForTat",
            "TitForTwoTats", "TatForTit", "Random", "Detective",
        ]
        for name in self.strategy_names:
            resolve_key(name)  # fail before spending any tokens on a typo

        self.history_log: List[Dict[str, Any]] = []
        self.progress = ProgressTracker()
        self.llm_player.set_console(self.progress.console)

    # ------------------------------------------------------------ checkpoint
    def _checkpoint_path(self, output_dir: str) -> str:
        return os.path.join(output_dir, "training_checkpoint.json")

    def _save_checkpoint(self, output_dir: str, next_strategy_index: int, next_match_in_block: int,
                         global_match_index: int, matches_done: bool = False):
        p = self.llm_player
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "strategy_names": self.strategy_names,
            "next_strategy_index": next_strategy_index,
            "next_match_in_block": next_match_in_block,
            "global_match_index": global_match_index,
            "matches_done": matches_done,
            "history_log": self.history_log,
            "memory": p.memory.to_list(),
            "memory_step": p.memory.step,
            "master_playbook": p.master_playbook,
            "freeze_notes": p.freeze_notes,
            "stats": dict(p.stats),
        }
        with open(self._checkpoint_path(output_dir), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self, output_dir: str) -> Optional[Dict[str, Any]]:
        path = self._checkpoint_path(output_dir)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != CHECKPOINT_VERSION:
                raise ValueError(f"checkpoint version {data.get('version')} is not {CHECKPOINT_VERSION}")
            p = self.llm_player
            self.strategy_names = data["strategy_names"]
            self.history_log = data["history_log"]
            p.memory.load_list(data["memory"], data.get("memory_step", 0))
            p.master_playbook = data.get("master_playbook")
            p.freeze_notes = data.get("freeze_notes", False)
            p.stats.update(data.get("stats", {}))
            return data
        except Exception as e:
            print(f"WARNING: could not use checkpoint {path} ({e}). Starting training from scratch.")
            return None

    def _clear_checkpoint(self, output_dir: str):
        path = self._checkpoint_path(output_dir)
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------- run
    def run_training(self, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        os.makedirs(output_dir, exist_ok=True)
        player = self.llm_player
        player.phase = "training"
        total_matches = len(self.strategy_names) * self.matches_per_strategy
        history_path = os.path.join(output_dir, "training_history.json")

        checkpoint = self._load_checkpoint(output_dir)
        resume_strategy_index, resume_match_index, match_global_index = 1, 0, 0
        matches_done = False
        if checkpoint:
            resume_strategy_index = int(checkpoint["next_strategy_index"])
            resume_match_index = int(checkpoint["next_match_in_block"])
            match_global_index = int(checkpoint["global_match_index"])
            matches_done = bool(checkpoint.get("matches_done"))
            player.match_number = match_global_index
            print(f"\nResuming training from checkpoint (match {match_global_index}/{total_matches}"
                  f"{', Dreamer pending' if matches_done else ''}).")
        else:
            self.history_log = []

        print("\n" + "=" * 60)
        print(f"PHASE 1: SEQUENTIAL BLOCK TRAINING ({len(self.strategy_names)} strategies, "
              f"{self.matches_per_strategy} matches each, tier {player.tier})")
        print("=" * 60)

        if not matches_done:
            self.progress.start()
            self.progress.add_overall_task("Training progress", total=total_matches)
            self.progress.update_overall(match_global_index)
            self.progress.set_status("starting...")
            try:
                for s_idx, strat_name in enumerate(self.strategy_names, 1):
                    if resume_strategy_index > s_idx:
                        continue
                    self.progress.console.rule(
                        f"[bold yellow]Strategy block {s_idx}/{len(self.strategy_names)}: "
                        f"{strat_name} ({self.matches_per_strategy} matches)[/bold yellow]")
                    start_match = resume_match_index if resume_strategy_index == s_idx else 0
                    resume_strategy_index = 1

                    for m_idx in range(start_match, self.matches_per_strategy):
                        match_global_index += 1
                        self._play_one(strat_name, s_idx, m_idx, match_global_index)
                        self.progress.update_overall(match_global_index)
                        self._update_status()

                        next_s, next_m = s_idx, m_idx + 1
                        if next_m >= self.matches_per_strategy:
                            next_s, next_m = s_idx + 1, 0
                        self._save_checkpoint(output_dir, next_s, next_m, match_global_index)
            finally:
                self.progress.stop()
                # Whatever happened (including an abort), keep what was learned so far.
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(self.history_log, f, indent=2)
            self._save_checkpoint(output_dir, len(self.strategy_names) + 1, 0, match_global_index, matches_done=True)
            print(f"\nAll strategy blocks completed. Training history saved to {history_path}")
        else:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history_log, f, indent=2)

        self._write_scorecard(output_dir)

        master_playbook = None
        if self.dreamer_enabled:
            master_playbook = Dreamer(player, temperature=self.dreamer_temperature).synthesize_master_playbook(
                self.history_log, output_dir=output_dir)
        player.save_memory(os.path.join(output_dir, "memory.json"))
        self._clear_checkpoint(output_dir)  # only after the playbook and memory are safely written
        return self.history_log, master_playbook

    def _play_one(self, strat_name: str, s_idx: int, m_idx: int, match_global_index: int):
        player = self.llm_player
        opponent = make_opponent(strat_name)
        rounds = self.sampler.sample()
        player.begin_match(opponent.name, rounds, self.sampler.average)
        player.match_number = match_global_index
        self.progress.reset_current(f"Current match vs {opponent.name}", rounds)

        stats_before = player.stats_snapshot()
        lessons_before = set(player.lessons)
        player.turn_callback = lambda r: self.progress.update_current(r - 1)
        try:
            result = play_match(player, opponent, rounds, seed=self.random.randrange(2**31), noise=self.error_rate)
        finally:
            player.turn_callback = None
        self.progress.update_current(rounds)

        turns = list(player.turn_log)
        player.finish_match(opponent)
        self.history_log.append({
            "global_match": match_global_index,
            "strategy_block": strat_name,
            "match_in_block": m_idx + 1,
            "opponent_strategy": opponent.name,
            "llm_score": result.score1,
            "opponent_score": result.score2,
            "rounds_played": rounds,
            "moves": ["".join(m) for m in result.moves],
            "turns": turns,
            "winner": "LLM" if result.score1 > result.score2 else ("Opponent" if result.score2 > result.score1 else "Draw"),
            "new_notes": [n for n in player.lessons if n not in lessons_before],
            "llm_stats": player.stats_since(stats_before),
        })

    def _update_status(self):
        records = records_from_training(self.history_log)
        if not records:
            return
        card = build_scorecard(records, dict(self.llm_player.stats))
        o = card["overall"]
        stats = self.llm_player.stats
        self.progress.set_status(
            f"Running: {o['points_per_round']} pts/round ({o['efficiency']:.0%} of best) | "
            f"Alpha {o['alpha_rate']:.0%} | fallbacks {stats.get('fallbacks', 0)} | parse errors {stats.get('parse_failures', 0)}")

    def _write_scorecard(self, output_dir: str):
        player = self.llm_player
        card = build_scorecard(
            records_from_training(self.history_log), dict(player.stats), tier=player.tier,
            phase="training", reveals_opponent=player.reveals_opponent)
        write_scorecard(card, output_dir, "scorecard_training")
        print_scorecard(card, self.progress.console)
