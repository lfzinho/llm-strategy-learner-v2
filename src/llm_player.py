import os
import time
import warnings
from collections import Counter
from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar

import axelrod as axl
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.constants import Move
from src.memory import MemoryStream
from src.opponents import OPPONENTS, key_for_display_name, resolve_key
from src.prompts import (
    MatchContext, PromptLibrary, build_reflection_prompt, build_turn_prompt, label,
)
from src.trace import TraceLog
from src.usage_tracker import DailyLimitReachedException, UsageTracker

# Suppress benign Google GenAI SDK AFC warning
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

_GAME = axl.Game()
T = TypeVar("T", bound=BaseModel)


def _to_move(action: axl.Action) -> Move:
    return Move.COOPERATE if action == axl.Action.C else Move.DEFECT


class LLMDecision(BaseModel):
    decision: Literal["Alpha", "Beta"] = Field(description="Your choice for this round: Alpha or Beta.")
    opponent_hypothesis: str = Field(description="Your current hypothesis about the other player's rule.")
    reasoning: str = Field(description="Brief reasoning comparing the points expected from each choice.")
    new_short_term_memory: Optional[str] = Field(
        default=None, description="Optional: one short note about the other player to keep for this match."
    )

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: Any) -> Any:
        """Small models vary the casing or fall back to the classic labels."""
        if isinstance(v, str):
            key = v.strip().upper()
            return {"ALPHA": "Alpha", "BETA": "Beta", "COOPERATE": "Alpha", "DEFECT": "Beta"}.get(key, v)
        return v


class LLMLongTermReflection(BaseModel):
    identified_archetype: str = Field(description="The rule the other player followed.")
    reasoning: str = Field(description="What the history shows and whether a better sequence existed.")
    new_long_term_memory: Optional[str] = Field(
        default=None, description="Optional: one short, general lesson for future matches."
    )
    importance: int = Field(default=5, description="How useful the lesson is, 1 (minor) to 10 (major).")

    @field_validator("importance", mode="before")
    @classmethod
    def clamp_importance(cls, v: Any) -> int:
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            return 5


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM keeps failing, so the run stops instead of silently playing fallbacks."""


class LLMCallFailed(Exception):
    """One logical LLM call failed after all retries."""


class LLMPlayer(axl.Player):
    """An LLM-driven Axelrod player.

    Long-term memory (memory stream, playbook) lives on the instance and must survive
    between matches, so `reset()` only clears the per-match state.
    """

    classifier = {
        "memory_depth": float("inf"),
        "stochastic": True,
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(
        self,
        name: str = "LLM Agent",
        provider: str = "openrouter",
        model_name: Optional[str] = None,
        tier: str = "t3_pool_hint",
        prompt_library: Optional[PromptLibrary] = None,
        known_strategies: Optional[List[str]] = None,
        temperature: float = 0.2,
        max_rpm: int = 30,
        max_rpd: int = 1000,
        request_timeout: float = 60.0,
        console: Optional[Console] = None,
        max_retries: int = 3,
        max_consecutive_fallbacks: int = 5,
        memory_top_k: int = 8,
        retry_sleep: float = 1.0,
    ):
        super().__init__()
        self.name = name
        self.strategy_name = "LLM Strategy Learner V2"
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openrouter")).lower()
        self.tier = tier
        self.prompt_library = prompt_library or PromptLibrary.load()
        self.prompt_library.tier(tier)  # fail fast on a bad tier name
        self.known_strategies = known_strategies
        self.temperature = temperature
        self.console = console or Console()
        self.max_retries = max(1, max_retries)
        self.max_consecutive_fallbacks = max_consecutive_fallbacks
        self.memory_top_k = memory_top_k
        self.retry_sleep = retry_sleep

        if self.provider == "gemini":
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        elif self.provider == "openrouter":
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "stealth/ox-alpha")
        elif self.provider == "mock":
            self.model_name = model_name or "mock"
        else:  # ollama
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.use_gemini = self.provider == "gemini"

        # Long-term state (persists across matches)
        self.memory = MemoryStream()
        self.short_lasting_notes: List[str] = []
        self.master_playbook: Optional[str] = None
        self.freeze_notes: bool = False

        # Per-match state
        self.game_history: List[Tuple[Move, Move]] = []
        self.score = 0
        self.opp_score = 0
        self.turn_callback = None  # called with the 1-based round number after each decision
        self.context = MatchContext(known_strategies=known_strategies)
        self.turn_log: List[dict] = []
        self._last_hypothesis = ""

        # Diagnostics
        self.phase = "training"
        self.match_number = 0
        self.stats: Counter = Counter()
        self.consecutive_fallbacks = 0
        self.trace = TraceLog()
        self.show_panels = True

        self.request_timeout = 180.0 if self.provider == "ollama" and request_timeout == 60.0 else request_timeout
        self.usage_tracker = UsageTracker(
            max_rpm=max_rpm, max_rpd=max_rpd, console=self.console,
            enabled=self.provider not in ("ollama", "mock"),
        )
        self._init_client()

    # ------------------------------------------------------------------ setup
    def _init_client(self):
        self.client = None
        if self.provider == "mock":
            return
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
                timeout=self.request_timeout,
                default_headers={
                    "HTTP-Referer": "https://github.com/lfzinho/llm-strategy-learner",
                    "X-Title": "LLM Strategy Learner V2",
                },
            )
            self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
        else:  # ollama
            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            if "://" not in ollama_url:
                ollama_url = "http://" + ollama_url
            ollama_url = ollama_url.rstrip("/")
            if ollama_url.endswith("/v1"):
                ollama_url = ollama_url[:-3].rstrip("/")
            self.ollama_base_url = ollama_url
            base_client = OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama", timeout=self.request_timeout)
            self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def set_console(self, console: Console):
        self.console = console
        self.usage_tracker.set_console(console)

    def configure_output(self, output_dir: str):
        """Points the trace log and usage log at the run's output directory."""
        self.trace.set_path(os.path.join(output_dir, "trace.jsonl"))
        self.usage_tracker.set_log_file(os.path.join(output_dir, "api_usage.json"))

    @property
    def reveals_opponent(self) -> bool:
        return self.prompt_library.tier(self.tier).reveals_opponent

    @property
    def lessons(self) -> List[str]:
        return [e.text for e in self.memory.reflections()]

    # ------------------------------------------------------------ match state
    def begin_match(self, opponent_name: str, total_rounds: int, avg_rounds: Optional[float] = None):
        """Called by the runner before each match. `opponent_name` may be a display name or key."""
        key = key_for_display_name(opponent_name)
        if key is None:
            try:
                key = resolve_key(opponent_name)
            except ValueError:
                key = None
        self.context = MatchContext(
            opponent_key=key, total_rounds=total_rounds,
            avg_rounds=avg_rounds if avg_rounds is not None else float(total_rounds),
            known_strategies=self.known_strategies,
        )
        self.match_number += 1
        self.turn_log = []
        self._last_hypothesis = ""

    def reset(self):
        """Clears per-match state only; memory and playbook persist across matches."""
        self.history.reset()
        self.game_history = []
        self.score = 0
        self.opp_score = 0

    def _sync_match_state(self, opponent: axl.Player):
        pairs = list(zip(self.history, opponent.history))
        self.game_history = [(_to_move(a), _to_move(b)) for a, b in pairs]
        scores = [_GAME.score(p) for p in pairs]
        self.score = sum(s[0] for s in scores)
        self.opp_score = sum(s[1] for s in scores)

    def _memory_lines(self) -> List[str]:
        opponent = self.context.opponent_key if self.reveals_opponent else None
        entries = self.memory.retrieve(self._last_hypothesis, k=self.memory_top_k, opponent=opponent)
        return self.memory.render(entries)

    def stats_snapshot(self) -> Counter:
        return Counter(self.stats)

    def stats_since(self, snapshot: Counter) -> dict:
        return {k: self.stats[k] - snapshot.get(k, 0) for k in self.stats if self.stats[k] - snapshot.get(k, 0)}

    # ------------------------------------------------------------- LLM calls
    def _call(self, response_model: Optional[Type[T]], prompt: str, temperature: float) -> Tuple[Any, str, Tuple[int, int]]:
        """One raw provider call. Returns (parsed object or text, raw text, (prompt_tokens, completion_tokens))."""
        if self.provider == "mock":
            return self._mock_response(response_model)

        self.usage_tracker.wait_and_record()
        if self.use_gemini:
            from google.genai import types
            config_kwargs = {"temperature": temperature}
            if response_model is not None:
                config_kwargs.update(response_mime_type="application/json", response_schema=response_model)
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config=types.GenerateContentConfig(**config_kwargs),
            )
            usage = getattr(response, "usage_metadata", None)
            tokens = (getattr(usage, "prompt_token_count", 0) or 0, getattr(usage, "candidates_token_count", 0) or 0)
            raw = response.text or ""
            if response_model is None:
                return raw, raw, tokens
            parsed = response.parsed if response.parsed is not None else response_model.model_validate_json(raw)
            return parsed, raw, tokens

        messages = [{"role": "user", "content": prompt}]
        if response_model is None:
            completion = self.client.client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=temperature,
            )
            raw = completion.choices[0].message.content or ""
            usage = getattr(completion, "usage", None)
            return raw, raw, (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)

        parsed, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name, response_model=response_model, messages=messages,
            temperature=temperature, max_retries=1,
        )
        raw = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        return parsed, raw, (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)

    def _call_with_retries(self, kind: str, response_model: Optional[Type[T]], prompt: str,
                           temperature: Optional[float] = None, **trace_fields) -> Any:
        """Calls the LLM with retries; writes one trace record; raises LLMCallFailed if all attempts fail."""
        temperature = self.temperature if temperature is None else temperature
        errors: List[str] = []
        rate_limits = 0
        started = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                result, raw, tokens = self._call(response_model, prompt, temperature)
                self.usage_tracker.record_tokens(*tokens)
                self.trace.write(
                    kind, phase=self.phase, match=self.match_number, prompt=prompt, raw=raw, attempts=attempt,
                    errors=errors, latency=round(time.time() - started, 2), tokens=list(tokens),
                    parsed=result.model_dump() if isinstance(result, BaseModel) else None, **trace_fields,
                )
                self.stats["retries"] += attempt - 1
                return result
            except DailyLimitReachedException:
                raise
            except Exception as e:
                text = f"{type(e).__name__}: {e}"
                errors.append(text)
                if "429" in text or "RESOURCE_EXHAUSTED" in text:
                    rate_limits += 1
                    self.stats["rate_limited"] += 1
                    if rate_limits >= 3:
                        self.trace.write(kind, phase=self.phase, match=self.match_number, prompt=prompt,
                                         attempts=attempt, errors=errors, aborted="rate_limit", **trace_fields)
                        raise DailyLimitReachedException("Aborted after 3 rate-limit (429) errors.") from e
                    time.sleep(20 * rate_limits)
                    continue
                if "validation" in type(e).__name__.lower() or "retry" in type(e).__name__.lower() or "json" in type(e).__name__.lower():
                    self.stats["parse_failures"] += 1
                self.console.print(f"[yellow]LLM {kind} attempt {attempt}/{self.max_retries} failed: {text[:200]}[/yellow]")
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * attempt)
        self.stats["retries"] += self.max_retries - 1
        self.trace.write(kind, phase=self.phase, match=self.match_number, prompt=prompt,
                         attempts=self.max_retries, errors=errors, failed=True, **trace_fields)
        raise LLMCallFailed(errors[-1] if errors else "unknown error")

    def complete_text(self, prompt: str, temperature: float, kind: str = "dreamer") -> str:
        """Free-form completion (used by the Dreamer). Raises LLMCallFailed on failure."""
        text = self._call_with_retries(kind, None, prompt, temperature=temperature)
        if not text or not text.strip():
            raise LLMCallFailed("empty completion")
        return text

    # ------------------------------------------------------------------ play
    def strategy(self, opponent: axl.Player) -> axl.Action:
        self._sync_match_state(opponent)
        move = self._decide()
        if self.turn_callback:
            self.turn_callback(len(self.game_history) + 1)
        return axl.Action.C if move == Move.COOPERATE else axl.Action.D

    def _fallback_move(self) -> Move:
        """Neutral fallback: open with Alpha, then repeat the opponent's last move."""
        return self.game_history[-1][1] if self.game_history else Move.COOPERATE

    def _decide(self) -> Move:
        round_num = len(self.game_history) + 1
        prompt = build_turn_prompt(
            self.prompt_library, self.tier, self.context, self.game_history,
            self.score, self.opp_score, self._memory_lines(), self.short_lasting_notes,
            playbook=self.master_playbook,
        )
        self.stats["turns"] += 1
        fields = dict(round=round_num, opponent=self.context.opponent_key)
        try:
            decision_obj: LLMDecision = self._call_with_retries("turn", LLMDecision, prompt, **fields)
        except LLMCallFailed as e:
            move = self._fallback_move()
            self.stats["fallbacks"] += 1
            self.consecutive_fallbacks += 1
            self.turn_log.append({"round": round_num, "decision": label(move), "hypothesis": "", "fallback": True})
            self.trace.write("fallback", phase=self.phase, match=self.match_number, round=round_num,
                             move=label(move), error=str(e), consecutive=self.consecutive_fallbacks)
            self.console.print(f"[red]LLM failed on round {round_num}; fallback move {label(move)} "
                               f"({self.consecutive_fallbacks} in a row)[/red]")
            if self.consecutive_fallbacks >= self.max_consecutive_fallbacks:
                raise LLMUnavailableError(
                    f"{self.consecutive_fallbacks} consecutive LLM failures (last error: {e}). "
                    f"See {self.trace.path or 'trace log (disabled)'} for details."
                ) from e
            return move
        self.consecutive_fallbacks = 0

        move = Move.COOPERATE if decision_obj.decision == "Alpha" else Move.DEFECT
        self._last_hypothesis = decision_obj.opponent_hypothesis
        self.turn_log.append({"round": round_num, "decision": decision_obj.decision,
                              "hypothesis": decision_obj.opponent_hypothesis, "fallback": False})
        self._show_move(decision_obj, round_num)
        if not self.freeze_notes and decision_obj.new_short_term_memory and decision_obj.new_short_term_memory.strip():
            self.short_lasting_notes.append(decision_obj.new_short_term_memory.strip())
        return move

    def _show_move(self, d: LLMDecision, round_num: int):
        if not self.show_panels:
            return
        style = "[bold green]Alpha[/bold green]" if d.decision == "Alpha" else "[bold red]Beta[/bold red]"
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[bold cyan]Hypothesis:[/bold cyan]", f"[italic yellow]{d.opponent_hypothesis}[/italic yellow]")
        table.add_row("[bold cyan]Decision:[/bold cyan]", style)
        table.add_row("[bold cyan]Reasoning:[/bold cyan]", d.reasoning)
        if d.new_short_term_memory:
            table.add_row("[bold magenta]+ Short note:[/bold magenta]", f"[italic]{d.new_short_term_memory}[/italic]")
        self.console.print(Panel(table, title=f"[bold blue]{self.name} - round {round_num} ({self.model_name})[/bold blue]",
                                 border_style="bright_blue", expand=False))

    # ------------------------------------------------------------ reflection
    def _observation_text(self) -> str:
        rounds = len(self.game_history)
        mine = "".join("A" if m == Move.COOPERATE else "B" for m, _ in self.game_history)
        theirs = "".join("A" if o == Move.COOPERATE else "B" for _, o in self.game_history)
        who = OPPONENTS[self.context.opponent_key].prompt_name if (self.reveals_opponent and self.context.opponent_key) else "an opponent"
        per_round = self.score / rounds if rounds else 0.0
        return (f"Match {self.match_number} against {who}: {rounds} rounds, you {self.score} points "
                f"({per_round:.2f} per round), they {self.opp_score}. You played {mine}; they played {theirs}.")

    def reflect_on_match(self):
        if self.freeze_notes or not self.game_history:
            return
        tag = self.context.opponent_key
        observation = self.memory.add("observation", self._observation_text(), importance=3, opponent=tag)
        prompt = build_reflection_prompt(
            self.prompt_library, self.tier, self.context, self.game_history, self.score, self.opp_score,
            self.short_lasting_notes, self._memory_lines(),
        )
        try:
            reflection: LLMLongTermReflection = self._call_with_retries(
                "reflection", LLMLongTermReflection, prompt, opponent=tag)
        except LLMCallFailed as e:
            self.stats["reflection_failures"] += 1
            self.console.print(f"[red]Reflection failed after retries: {str(e)[:200]}[/red]")
            return

        added = None
        if reflection.new_long_term_memory:
            added = self.memory.add(
                "reflection", reflection.new_long_term_memory, importance=reflection.importance, opponent=tag,
                evidence=[observation.id] if observation else [],
            )
        if not self.show_panels:
            return
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[bold cyan]Match points:[/bold cyan]", f"[bold yellow]{self.score}[/bold yellow] vs {self.opp_score}")
        table.add_row("[bold cyan]Diagnosed rule:[/bold cyan]", f"[bold magenta]{reflection.identified_archetype}[/bold magenta]")
        table.add_row("[bold cyan]Reflection:[/bold cyan]", reflection.reasoning)
        if reflection.new_long_term_memory:
            note = "new" if added else "already known (importance raised)"
            table.add_row("[bold magenta]Lesson:[/bold magenta]", f"[green]{reflection.new_long_term_memory}[/green] [dim]({note})[/dim]")
        self.console.print(Panel(table, title="[bold magenta]Post-match reflection[/bold magenta]", border_style="magenta", expand=False))

    def finish_match(self, opponent: axl.Player):
        """Reflects on the completed match, then clears per-match state."""
        self._sync_match_state(opponent)
        self.reflect_on_match()
        if not self.freeze_notes:
            self.memory.advance()
        self.reset()
        self.short_lasting_notes = []

    def save_memory(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"step": self.memory.step, "entries": self.memory.to_list()}, f, indent=2)

    def load_memory(self, path: str) -> bool:
        import json
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.memory.load_list(data["entries"], data.get("step", 0))
        return True

    def set_master_playbook(self, playbook_text: str):
        """Injects the Dreamer playbook and freezes further memory changes."""
        self.master_playbook = playbook_text
        self.freeze_notes = True

    # ------------------------------------------------------------------ mock
    def _mock_response(self, response_model):
        """Offline stand-in for an LLM: plays Alpha, then mirrors the opponent (Alpha Mimic)."""
        if response_model is LLMDecision:
            move = self._fallback_move()
            obj = LLMDecision(decision=label(move), opponent_hypothesis="Alpha Mimic",
                              reasoning="mock: copy the other player", new_short_term_memory=None)
        elif response_model is LLMLongTermReflection:
            obj = LLMLongTermReflection(identified_archetype="Alpha Mimic", reasoning="mock reflection",
                                        new_long_term_memory="Mirror the other player's last choice.", importance=5)
        else:
            text = "# Mock playbook\n- Open with Alpha, then repeat the other player's previous choice."
            return text, text, (0, 0)
        return obj, obj.model_dump_json(), (0, 0)
