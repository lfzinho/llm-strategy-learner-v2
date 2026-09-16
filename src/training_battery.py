import json
import os
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    from src.game_machine import GameMachine
    from src.player_classes import get_strategy_class
    from src.dreamer import Dreamer
except ImportError:
    from game_machine import GameMachine
    from player_classes import get_strategy_class
    from dreamer import Dreamer


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
        rounds_per_match: int = 5,
        error_rate: float = 0.0,
        dreamer_enabled: bool = True,
        dreamer_temperature: float = 0.2,
    ):
        self.llm_player = llm_player
        self.matches_per_strategy = matches_per_strategy
        self.rounds_per_match = rounds_per_match
        self.game_machine = GameMachine(error_rate=error_rate)
        self.dreamer_enabled = dreamer_enabled
        self.dreamer_temperature = dreamer_temperature

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

    def run_training(self, output_dir: str = "output") -> Tuple[List[Dict[str, Any]], Optional[str]]:
        os.makedirs(output_dir, exist_ok=True)
        total_matches = len(self.strategy_names) * self.matches_per_strategy

        print("\n" + "=" * 60)
        print(f"🎯 PHASE 1: SEQUENTIAL BLOCK TRAINING ({len(self.strategy_names)} strategies, {self.matches_per_strategy} matches each)")
        print("=" * 60)

        match_global_index = 0
        for s_idx, strat_name in enumerate(self.strategy_names, 1):
            strat_cls = get_strategy_class(strat_name)

            try:
                from rich.console import Console
                from rich.rule import Rule
                console = Console()
                console.print(Rule(f"[bold yellow]Strategy Block {s_idx}/{len(self.strategy_names)}: {strat_name} ({self.matches_per_strategy} matches)[/bold yellow]"))
            except Exception:
                print(f"\n--- Strategy Block {s_idx}/{len(self.strategy_names)}: {strat_name} ---")

            for m_idx in range(self.matches_per_strategy):
                match_global_index += 1
                opponent = strat_cls()
                self.llm_player.last_opponent_name = opponent.strategy_name

                # Play rounds
                for round_idx in range(self.rounds_per_match):
                    self.game_machine.play_game(self.llm_player, opponent)

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
                    "winner": "LLM" if llm_score > opp_score else ("Opponent" if opp_score > llm_score else "Draw"),
                    "new_notes": new_notes,
                    "all_long_notes": list(self.llm_player.long_lasting_notes),
                }
                self.history_log.append(log_entry)

            # Incremental save after each strategy block
            history_path = os.path.join(output_dir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history_log, f, indent=2)

        print(f"\nAll strategy blocks completed! Training history saved to {history_path}")

        master_playbook = None
        if self.dreamer_enabled:
            dreamer = Dreamer(self.llm_player, temperature=self.dreamer_temperature)
            master_playbook = dreamer.synthesize_master_playbook(self.history_log, output_dir=output_dir)

        return self.history_log, master_playbook
