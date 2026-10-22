import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.opponents import OPPONENTS, key_for_display_name
from src.prompts import build_dreamer_prompt

_ALPHA_BETA = {"C": "A", "D": "B"}


def _sequence(moves: List[str], index: int, limit: int = 10) -> str:
    return "".join(_ALPHA_BETA[m[index]] for m in moves[:limit])


def summarize_training(training_history: List[Dict[str, Any]], show_names: bool) -> str:
    """Measured results per opponent: points per round, Alpha rate, best sequence.

    With `show_names=False` (tiers where the model is not told what opponents exist) the
    opponents are labelled "Opponent type 1, 2, ..." so the playbook cannot leak the pool.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in training_history:
        groups[entry.get("opponent_strategy", "Unknown")].append(entry)

    lines = []
    for i, (display, entries) in enumerate(groups.items(), 1):
        key = key_for_display_name(display)
        name = OPPONENTS[key].prompt_name if (show_names and key) else (display if show_names else f"Opponent type {i}")
        rounds = sum(e.get("rounds_played", 0) for e in entries) or 1
        mine = sum(e.get("llm_score", 0) for e in entries)
        theirs = sum(e.get("opponent_score", 0) for e in entries)
        with_moves = [e for e in entries if e.get("moves")]
        alpha = sum(m[0] == "C" for e in with_moves for m in e["moves"])
        total_moves = sum(len(e["moves"]) for e in with_moves) or 1
        best = max(entries, key=lambda e: e.get("llm_score", 0) / max(1, e.get("rounds_played", 1)))
        lines.append(
            f"- {name}: {len(entries)} matches | you {mine / rounds:.2f} points/round, they {theirs / rounds:.2f} | "
            f"you chose Alpha {100 * alpha / total_moves:.0f}% of the time"
        )
        if best.get("moves"):
            lines.append(
                f"    best match ({best['llm_score'] / max(1, best['rounds_played']):.2f} points/round): "
                f"you {_sequence(best['moves'], 0)} / they {_sequence(best['moves'], 1)} (A=Alpha, B=Beta, first 10 rounds)"
            )
    lines.append("Reference: Alpha/Alpha every round = 3.00 points/round, Beta/Beta = 1.00, exploiting an Alpha player = 5.00.")
    return "\n".join(lines)


class Dreamer:
    """Consolidates the memory stream plus measured training results into a playbook."""

    def __init__(self, llm_player, temperature: float = 0.2):
        self.llm_player = llm_player
        self.temperature = temperature

    def _names_allowed(self) -> bool:
        tier = self.llm_player.prompt_library.tier(self.llm_player.tier)
        return tier.reveals_opponent or "pool_catalog" in tier.blocks

    def synthesize_master_playbook(self, training_history: List[Dict[str, Any]], output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        player = self.llm_player
        lessons = player.lessons
        summary = summarize_training(training_history, self._names_allowed())
        prompt = build_dreamer_prompt(player.prompt_library, summary, lessons)

        print("\n" + "=" * 60)
        print("ENTERING DREAMER PHASE: consolidating memory into a playbook...")
        print("=" * 60)

        fallback = False
        try:
            playbook = player.complete_text(prompt, self.temperature, kind="dreamer")
        except Exception as e:  # LLMCallFailed or provider-specific errors
            print(f"Dreamer call failed ({e}). Writing a data-only playbook instead.")
            playbook = self._data_only_playbook(summary, lessons)
            fallback = True

        with open(os.path.join(output_dir, "master_strategy_playbook.md"), "w", encoding="utf-8") as f:
            f.write(playbook.rstrip() + "\n")
        with open(os.path.join(output_dir, "master_strategy_playbook.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model": player.model_name,
                "provider": player.provider,
                "tier": player.tier,
                "fallback": fallback,
                "lessons_count": len(lessons),
                "training_summary": summary,
                "playbook_markdown": playbook,
            }, f, indent=2)

        player.set_master_playbook(playbook)
        try:
            from rich.markdown import Markdown
            from rich.panel import Panel
            player.console.print(Panel(Markdown(playbook), title="Master Strategy Playbook", border_style="gold1"))
        except Exception:
            print("\n=== Master Strategy Playbook ===\n" + playbook)
        return playbook

    @staticmethod
    def _data_only_playbook(summary: str, lessons: List[str]) -> str:
        lesson_lines = "\n".join(f"- {n}" for n in lessons) or "- (none)"
        return (
            "# Playbook (automatic data summary: the Dreamer LLM call failed)\n\n"
            "## Measured training results\n"
            f"{summary}\n\n"
            "## Lessons written during training\n"
            f"{lesson_lines}\n\n"
            "Prefer the choices that scored the most points per round in the results above."
        )
