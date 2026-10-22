import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    from src.game_machine import GameMachine
    from src.player_classes import get_strategy_class
    from src.dreamer import Dreamer
    from src.progress import ProgressTracker
    from src.config import NormalRoundsConfig, RoundsPerMatchConfig
except ImportError:
    from game_machine import GameMachine
    from player_classes import get_strategy_class
    from dreamer import Dreamer
    from progress import ProgressTracker
    from config import NormalRoundsConfig, RoundsPerMatchConfig


class TrainingBattery:
    """Structured sequential block training battery.
    
    Plays N consecutive matches against Strategy 1, then N consecutive matches against Strategy 2,
    etc., allowing the LLM to learn and hypothesize about each archetype in depth.
    Finally invokes the Dreamer phase to consolidate all notes into a Master Playbook.
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
        dreamer_custom_template: Optional[str] = None,
    ):
        self.llm_player = llm_player
        self.matches_per_strategy = matches_per_strategy
        self.rounds_per_match = rounds_per_match
        self.game_machine = GameMachine(error_rate=error_rate)
        self.dreamer_enabled = dreamer_enabled
        self.dreamer_temperature = dreamer_temperature
        self.dreamer_custom_template = dreamer_custom_template

        self.strategy_names = strategies or [
            "AlwaysCooperate",
            "AlwaysDefect",
            "TitForTat",
            "SuspiciousTitForTat",
            "TitForTwoTats",
            "TatForTit",
            "Random",
            "Detective",
        ]
        self.history_log: List[Dict[str, Any]] = []
        self.progress = ProgressTracker()
        if hasattr(self.llm_player, "set_console"):
            self.llm_player.set_console(self.progress.console)
        self.random = random.Random()

    def _sample_rounds(self) -> int:
        if isinstance(self.rounds_per_match, NormalRoundsConfig):
            return max(1, round(self.random.gauss(self.rounds_per_match.average, self.rounds_per_match.std_dev)))
        return self.rounds_per_match

    def _checkpoint_path(self, output_dir: str) -> str:
        return os.path.join(output_dir, "training_checkpoint.json")

    def _save_checkpoint(self, output_dir: str, next_strategy_index: int, next_match_in_block: int, global_match_index: int):
        checkpoint = {
            "version": 1,
            "strategy_names": self.strategy_names,
            "next_strategy_index": next_strategy_index,
            "next_match_in_block": next_match_in_block,
            "global_match_index": global_match_index,
            "history_log": self.history_log,
            "long_lasting_notes": list(self.llm_player.long_lasting_notes),
            "short_lasting_notes": list(self.llm_player.short_lasting_notes),
            "master_playbook": self.llm_player.master_playbook,
            "freeze_notes": self.llm_player.freeze_notes,
            "last_opponent_name": self.llm_player.last_opponent_name,
        }
        with open(self._checkpoint_path(output_dir), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self, output_dir: str):
        path = self._checkpoint_path(output_dir)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            self.strategy_names = data.get("strategy_names", self.strategy_names)
            self.history_log = data.get("history_log", [])
            self.llm_player.long_lasting_notes = data.get("long_lasting_notes", self.llm_player.long_lasting_notes)
            self.llm_player.short_lasting_notes = data.get("short_lasting_notes", self.llm_player.short_lasting_notes)
            self.llm_player.master_playbook = data.get("master_playbook", self.llm_player.master_playbook)
            self.llm_player.freeze_notes = data.get("freeze_notes", self.llm_player.freeze_notes)
            self.llm_player.last_opponent_name = data.get("last_opponent_name", self.llm_player.last_opponent_name)
            return data
        except Exception:
            return None

    def _clear_checkpoint(self, output_dir: str):
        path = self._checkpoint_path(output_dir)
        if os.path.exists(path):
            os.remove(path)

    def run_training(self, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        os.makedirs(output_dir, exist_ok=True)
        total_matches = len(self.strategy_names) * self.matches_per_strategy

        checkpoint = self._load_checkpoint(output_dir)
        resume_strategy_index = 1
        resume_match_index = 0
        match_global_index = 0

        if checkpoint:
            resume_strategy_index = int(checkpoint.get("next_strategy_index", 1))
            resume_match_index = int(checkpoint.get("next_match_in_block", 0))
            match_global_index = int(checkpoint.get("global_match_index", 0))
            self.history_log = checkpoint.get("history_log", self.history_log)
            print(f"\nResuming training from saved checkpoint at strategy {resume_strategy_index}, match offset {resume_match_index}.")
        else:
            self.history_log = []

        print("\n" + "=" * 60)
        print(f"🎯 PHASE 1: SEQUENTIAL BLOCK TRAINING ({len(self.strategy_names)} strategies, {self.matches_per_strategy} matches each)")
        print("=" * 60)

        self.progress.start()
        self.progress.add_overall_task("Training progress", total=total_matches)
        self.progress.update_overall(match_global_index)

        for s_idx, strat_name in enumerate(self.strategy_names, 1):
            if resume_strategy_index > s_idx:
                continue

            strat_cls = get_strategy_class(strat_name)

            try:
                from rich.console import Console
                from rich.rule import Rule
                console = self.progress.console
                console.print(Rule(f"[bold yellow]Strategy Block {s_idx}/{len(self.strategy_names)}: {strat_name} ({self.matches_per_strategy} matches)[/bold yellow]"))
            except Exception:
                print(f"\n--- Strategy Block {s_idx}/{len(self.strategy_names)}: {strat_name} ---")

            start_match = 0
            if resume_strategy_index == s_idx:
                start_match = resume_match_index
            resume_strategy_index = 1

            for m_idx in range(start_match, self.matches_per_strategy):
                match_global_index += 1
                opponent = strat_cls()
                self.llm_player.last_opponent_name = opponent.strategy_name
                rounds_this_match = self._sample_rounds()

                self.progress.reset_current(f"Current match vs {opponent.strategy_name}", rounds_this_match)

                # Play rounds
                for round_idx in range(rounds_this_match):
                    self.game_machine.play_game(self.llm_player, opponent)
                    self.progress.update_current(round_idx + 1)

                self.progress.update_overall(match_global_index)
                llm_score = self.llm_player.score
                opp_score = opponent.score

                # Snapshot notes before reflection
                notes_before = set(self.llm_player.long_lasting_notes)

                # Finish match (triggers LLM reflection and updates notes)
                self.llm_player.finish_game(opponent_score=opp_score)
                opponent.finish_game()

                # Detect newly added notes
                new_notes = [n for n in self.llm_player.long_lasting_notes if n not in notes_before]

                log_entry = {
                    "global_match": match_global_index,
                    "strategy_block": strat_name,
                    "match_in_block": m_idx + 1,
                    "opponent_strategy": opponent.strategy_name,
                    "llm_score": llm_score,
                    "opponent_score": opp_score,
                    "rounds_played": rounds_this_match,
                    "winner": "LLM" if llm_score > opp_score else ("Opponent" if opp_score > llm_score else "Draw"),
                    "new_notes": new_notes,
                    "all_long_notes": list(self.llm_player.long_lasting_notes),
                }
                self.history_log.append(log_entry)

                next_strategy_index = s_idx
                next_match_in_block = m_idx + 1
                if next_match_in_block >= self.matches_per_strategy:
                    next_strategy_index = s_idx + 1
                    next_match_in_block = 0
                self._save_checkpoint(output_dir, next_strategy_index, next_match_in_block, match_global_index)

            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history_log, f, indent=2)

        self.progress.stop()
        self._clear_checkpoint(output_dir)
        print(f"\nAll strategy blocks completed! Training history saved to {history_path}")

        master_playbook = None
        if self.dreamer_enabled:
            dreamer = Dreamer(
                self.llm_player,
                temperature=self.dreamer_temperature,
                custom_template=self.dreamer_custom_template,
            )
            master_playbook = dreamer.synthesize_master_playbook(self.history_log, output_dir=output_dir)

        return self.history_log, master_playbook
