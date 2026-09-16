import json
import os
from typing import Any, Dict, List, Optional

try:
    from src.prompts import build_dreamer_prompt
except ImportError:
    from prompts import build_dreamer_prompt


class Dreamer:
    """Dreamer phase: Synthesizes and consolidates all training observations and reflection notes
    into a single high-density Master Strategy Playbook.
    """

    def __init__(self, llm_player, temperature: float = 0.2, custom_template: Optional[str] = None):
        self.llm_player = llm_player
        self.temperature = temperature
        self.custom_template = custom_template

    def synthesize_master_playbook(
        self,
        training_history: List[Dict[str, Any]],
        output_dir: str = "output",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        # Collect raw notes and summary of matches
        all_raw_notes = list(self.llm_player.long_lasting_notes)
        summary_lines = []

        strategies_faced = {}
        for entry in training_history:
            strat = entry.get("opponent_strategy", "Unknown")
            if strat not in strategies_faced:
                strategies_faced[strat] = {"matches": 0, "llm_total": 0, "opp_total": 0, "notes": []}
            strategies_faced[strat]["matches"] += 1
            strategies_faced[strat]["llm_total"] += entry.get("llm_score", 0)
            strategies_faced[strat]["opp_total"] += entry.get("opponent_score", 0)
            for note in entry.get("new_notes", []):
                if note and note not in all_raw_notes:
                    all_raw_notes.append(note)

        for strat, stats in strategies_faced.items():
            summary_lines.append(
                f"- Faced {strat}: {stats['matches']} matches | LLM Points: {stats['llm_total']} | Opponent Points: {stats['opp_total']}"
            )

        summary_text = "\n".join(summary_lines)
        prompt = build_dreamer_prompt(
            training_summary=summary_text,
            raw_notes=all_raw_notes,
            custom_template=self.custom_template,
        )

        print("\n" + "=" * 60)
        print("🌙 ENTERING DREAMER PHASE: Consolidating Memory into Master Playbook...")
        print("=" * 60)

        playbook_content = ""
        try:
            if self.llm_player.use_gemini:
                from google.genai import types
                self.llm_player.usage_tracker.wait_and_record()
                response = self.llm_player.client.models.generate_content(
                    model=self.llm_player.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                    ),
                )
                playbook_content = response.text
            else:
                self.llm_player.usage_tracker.wait_and_record()
                # Direct completion for free-form rich markdown synthesis
                chat_resp = self.llm_player.client.client.chat.completions.create(
                    model=self.llm_player.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                playbook_content = chat_resp.choices[0].message.content

        except Exception as e:
            print(f"Error during Dreamer synthesis: {e}. Generating heuristic fallback playbook.")
            playbook_content = self._generate_fallback_playbook(all_raw_notes)

        # Save to markdown and json
        md_path = os.path.join(output_dir, "master_strategy_playbook.md")
        json_path = os.path.join(output_dir, "master_strategy_playbook.json")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(playbook_content + "\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.llm_player.model_name,
                "provider": self.llm_player.provider,
                "raw_notes_count": len(all_raw_notes),
                "playbook_markdown": playbook_content,
            }, f, indent=2)

        # Freeze into LLM Player
        self.llm_player.set_master_playbook(playbook_content)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.markdown import Markdown
            console = Console()
            console.print(Panel(Markdown(playbook_content), title="[bold gold1]🌟 Synthesized Master Strategy Playbook[/bold gold1]", border_style="gold1"))
        except Exception:
            print("\n=== Master Strategy Playbook ===\n" + playbook_content)

        print(f"\nMaster Strategy Playbook saved to {md_path}")
        return playbook_content

    def _generate_fallback_playbook(self, notes: List[str]) -> str:
        notes_str = "\n".join(f"- {n}" for n in notes)
        return f"""# Master Strategy Playbook (Heuristic Synthesis)

## 1. Quick Archetype Recognition Guide
- **Round 1**: Start with **COOPERATE** to test for reciprocal or cooperative partners.
- **Round 2**:
  - If opponent played **COOPERATE** in Round 1: Likely *Always Cooperate*, *Tit For Tat*, *Tit For Two Tats*, or *Detective*.
  - If opponent played **DEFECT** in Round 1: Likely *Always Defect* or *Suspicious Tit For Tat*. Test by defecting or cooperating.
- **Rounds 3-4**:
  - Probe with a single **DEFECT** if mutual cooperation is suspected to test forgiveness (differentiating *Always Cooperate* vs *Tit For Tat* vs *Grim Trigger*).

## 2. Optimal Counter-Strategy Policy
- **Against Always Cooperate**: Always **DEFECT** for maximal 5 pts/round.
- **Against Always Defect**: Always **DEFECT** for 1 pt/round (prevent sucker 0 pt).
- **Against Tit For Tat**: Always **COOPERATE** for sustained 3 pts/round.
- **Against Detective**: Retaliate against their round 2 defect, then cooperate once they enter TFT mode.
- **Against Suspicious Tit For Tat**: Retaliate or initiate mutual cooperation if possible.

## 3. Accumulated Notes
{notes_str}
"""
