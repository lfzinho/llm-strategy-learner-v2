import os
import time
import warnings
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

import instructor
from openai import OpenAI

try:
    from src.player import Player
    from src.constants import Move
    from src.prompts import build_turn_prompt, build_reflection_prompt
    from src.usage_tracker import UsageTracker
except ImportError:
    from player import Player
    from constants import Move
    from prompts import build_turn_prompt, build_reflection_prompt
    from usage_tracker import UsageTracker

# Suppress benign Google GenAI SDK AFC warning
warnings.filterwarnings("ignore", message=".*automatic function calling.*")


class LLMDecision(BaseModel):
    decision: Literal["COOPERATE", "DEFECT"] = Field(
        description="Your next move: either COOPERATE or DEFECT."
    )
    opponent_hypothesis: str = Field(
        description="Your current hypothesis on what strategy or behavioral archetype the opponent is executing."
    )
    reasoning: str = Field(
        description="Tactical reasoning explaining why this move maximizes cumulative score given your hypothesis."
    )
    new_short_term_memory: Optional[str] = Field(
        default=None,
        description="Optional: A concise observation about this opponent's reactions to record for the remainder of this match."
    )


class LLMLongTermReflection(BaseModel):
    identified_archetype: str = Field(
        description="The opponent's strategy archetype diagnosed after observing the full match."
    )
    reasoning: str = Field(
        description="Reflection on the match dynamics, probing effectiveness, and payoff optimality."
    )
    new_long_term_memory: Optional[str] = Field(
        default=None,
        description="Optional: A high-density heuristic or counter-play rule to persist across future games."
    )


class LLMPlayer(Player):
    def __init__(
        self,
        name: str = "LLM Agent",
        provider: str = "openrouter",
        model_name: Optional[str] = None,
        prompt_mode: str = "explicit_strategies",
        known_strategies: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_rpm: int = 30,
        max_rpd: int = 1000,
    ):
        super().__init__(name)
        self.strategy_name = "LLM Strategy Learner V2"
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openrouter")).lower()
        self.prompt_mode = prompt_mode
        self.known_strategies = known_strategies
        self.temperature = temperature

        if self.provider == "gemini":
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.use_gemini = True
        elif self.provider == "openrouter":
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "stealth/ox-alpha")
            self.use_gemini = False
        else:  # ollama
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
            self.use_gemini = False

        self.long_lasting_notes: List[str] = [
            "I am a rational game-theoretic agent playing the Prisoner's Dilemma. I seek to maximize cumulative score."
        ]
        self.short_lasting_notes: List[str] = []
        self.master_playbook: Optional[str] = None
        self.freeze_notes: bool = False
        self.last_opponent_name: Optional[str] = None

        self.usage_tracker = UsageTracker(max_rpm=max_rpm, max_rpd=max_rpd)
        self._init_client()

    def _init_client(self):
        if self.use_gemini:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY not found in environment.")
            self.client = genai.Client(api_key=api_key)
        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY")
            if not api_key:
                print("Warning: OPENROUTER_API_KEY not found in environment.")
            base_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key or "missing_key",
                default_headers={
                    "HTTP-Referer": "https://github.com/lfzinho/llm-strategy-learner",
                    "X-Title": "LLM Strategy Learner V2",
                },
            )
            self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
        else:  # ollama
            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
            if not ollama_url.endswith("/v1"):
                ollama_url = ollama_url.rstrip("/") + "/v1"
            base_client = OpenAI(base_url=ollama_url, api_key="ollama")
            self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def play(self) -> Move:
        prompt = build_turn_prompt(
            game_history=self.game_history,
            my_score=self.score,
            opp_score=0,  # relative score within match
            long_lasting_notes=self.long_lasting_notes,
            short_lasting_notes=self.short_lasting_notes,
            master_playbook=self.master_playbook,
            mode=self.prompt_mode,
            known_strategies=self.known_strategies,
        )

        consecutive_429s = 0
        max_retries = 3

        for attempt in range(max_retries + 1):
            try:
                if self.use_gemini:
                    from google.genai import types
                    self.usage_tracker.wait_and_record()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=LLMDecision,
                            temperature=self.temperature,
                        ),
                    )
                    decision_obj = response.parsed
                    if decision_obj is None:
                        import json
                        decision_obj = LLMDecision.model_validate_json(response.text)
                else:
                    self.usage_tracker.wait_and_record()
                    decision_obj = self.client.chat.completions.create(
                        model=self.model_name,
                        response_model=LLMDecision,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_retries=2,
                    )

                decision = decision_obj.decision
                round_num = len(self.game_history) + 1

                # Visual rich terminal output
                try:
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.table import Table
                    console = Console()
                    move_style = "[bold green]COOPERATE 🤝[/bold green]" if decision == "COOPERATE" else "[bold red]DEFECT ⚔️[/bold red]"
                    table = Table(show_header=False, box=None, padding=(0, 1))
                    table.add_row("[bold cyan]Model:[/bold cyan]", f"[dim]{self.model_name}[/dim] ({self.provider})")
                    table.add_row("[bold cyan]Round:[/bold cyan]", f"#{round_num}")
                    table.add_row("[bold cyan]Hypothesis:[/bold cyan]", f"[italic yellow]{decision_obj.opponent_hypothesis}[/italic yellow]")
                    table.add_row("[bold cyan]Decision:[/bold cyan]", move_style)
                    table.add_row("[bold cyan]Reasoning:[/bold cyan]", f"[white]{decision_obj.reasoning}[/white]")
                    if decision_obj.new_short_term_memory:
                        table.add_row("[bold magenta]+ Short Memory:[/bold magenta]", f"[italic]{decision_obj.new_short_term_memory}[/italic]")

                    console.print(Panel(table, title=f"[bold blue]🤖 {self.name} Move (Round {round_num})[/bold blue]", border_style="bright_blue", expand=False))
                except Exception:
                    print(f"[{self.name}] Round {round_num} -> {decision} | Hypothesis: {decision_obj.opponent_hypothesis} | Reasoning: {decision_obj.reasoning}")

                # Record memory if not frozen
                if not self.freeze_notes and decision_obj.new_short_term_memory and decision_obj.new_short_term_memory.strip():
                    self.short_lasting_notes.append(decision_obj.new_short_term_memory.strip())

                return Move.COOPERATE if decision == "COOPERATE" else Move.DEFECT

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    consecutive_429s += 1
                    if consecutive_429s >= 3:
                        from usage_tracker import DailyLimitReachedException
                        print(f"\n[ABORT] Received {consecutive_429s} consecutive 429 Quota/Rate errors. Halting execution.")
                        raise DailyLimitReachedException("Aborted due to 3 consecutive 429 API errors.")
                    sleep_sec = 20 * consecutive_429s
                    print(f"[RateLimit 429] Sleeping for {sleep_sec}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(sleep_sec)
                else:
                    print(f"LLM API Error on turn: {e}. Defaulting to DEFECT.")
                    return Move.DEFECT

        return Move.DEFECT

    def reflect_on_match(self, opponent_score: int = 0):
        if self.freeze_notes or not self.game_history:
            return

        prompt = build_reflection_prompt(
            game_history=self.game_history,
            my_score=self.score,
            opp_score=opponent_score,
            short_lasting_notes=self.short_lasting_notes,
            long_lasting_notes=self.long_lasting_notes,
            actual_opponent_name=self.last_opponent_name,
        )

        try:
            if self.use_gemini:
                from google.genai import types
                self.usage_tracker.wait_and_record()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LLMLongTermReflection,
                        temperature=self.temperature,
                    ),
                )
                reflection_obj = response.parsed
                if reflection_obj is None:
                    import json
                    reflection_obj = LLMLongTermReflection.model_validate_json(response.text)
            else:
                self.usage_tracker.wait_and_record()
                reflection_obj = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=LLMLongTermReflection,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_retries=2,
                )

            if reflection_obj.new_long_term_memory and reflection_obj.new_long_term_memory.strip():
                self.long_lasting_notes.append(reflection_obj.new_long_term_memory.strip())

            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.table import Table
                console = Console()
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_row("[bold cyan]Match Score:[/bold cyan]", f"[bold yellow]{self.score}[/bold yellow]")
                table.add_row("[bold cyan]Diagnosed Archetype:[/bold cyan]", f"[bold magenta]{reflection_obj.identified_archetype}[/bold magenta]")
                table.add_row("[bold cyan]Reflection:[/bold cyan]", f"[white]{reflection_obj.reasoning}[/white]")
                if reflection_obj.new_long_term_memory:
                    table.add_row("[bold magenta]+ Long-term Rule:[/bold magenta]", f"[bold green]{reflection_obj.new_long_term_memory}[/bold green]")
                console.print(Panel(table, title="[bold magenta]🧠 Post-Match Reflection & Learning[/bold magenta]", border_style="magenta", expand=False))
            except Exception:
                print(f"[Reflection] Archetype: {reflection_obj.identified_archetype} | New Rule: {reflection_obj.new_long_term_memory}")

        except Exception as e:
            print(f"Error during post-match reflection: {e}")

    def finish_game(self, opponent_score: int = 0):
        self.reflect_on_match(opponent_score=opponent_score)
        super().finish_game()
        self.short_lasting_notes = []

    def set_master_playbook(self, playbook_text: str):
        """Injects the synthesized Dreamer playbook and freezes further memory modifications."""
        self.master_playbook = playbook_text
        self.freeze_notes = True

    def copy(self) -> "LLMPlayer":
        new_player = LLMPlayer(
            name=self.name,
            provider=self.provider,
            model_name=self.model_name,
            prompt_mode=self.prompt_mode,
            known_strategies=self.known_strategies,
            temperature=self.temperature,
        )
        new_player.long_lasting_notes = list(self.long_lasting_notes)
        new_player.master_playbook = self.master_playbook
        new_player.freeze_notes = self.freeze_notes
        return new_player
