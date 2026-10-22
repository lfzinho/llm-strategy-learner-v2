"""Loads prompts.toml and renders the per-round, reflection and dreamer prompts."""
import re
import tomllib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.constants import Move
from src.opponents import OPPONENTS, resolve_key

PAYOFFS = {
    (Move.COOPERATE, Move.COOPERATE): (3, 3),
    (Move.COOPERATE, Move.DEFECT): (0, 5),
    (Move.DEFECT, Move.COOPERATE): (5, 0),
    (Move.DEFECT, Move.DEFECT): (1, 1),
}

BLOCK_PLACEHOLDERS = {
    "strategy_catalog", "opponent_name", "opponent_description", "opponent_counter",
    "total_rounds", "round", "rounds_left", "avg_rounds",
}
TEMPLATE_PLACEHOLDERS = {
    "turn": {"system", "round", "my_score", "opp_score", "history", "memory", "playbook", "short_notes"},
    "reflection": {"opponent_info", "total_rounds", "my_score", "opp_score", "score_per_round",
                   "history_text", "short_notes_text", "memory_text"},
    "dreamer": {"training_summary", "raw_notes_text"},
}
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def label(move: Move) -> str:
    return "Alpha" if move == Move.COOPERATE else "Beta"


@dataclass
class TierSpec:
    name: str
    description: str
    reveals_opponent: bool
    blocks: List[str]


@dataclass
class MatchContext:
    """What the prompt may say about the current match."""
    opponent_key: Optional[str] = None
    total_rounds: Optional[int] = None
    avg_rounds: Optional[float] = None
    known_strategies: Optional[List[str]] = None


@dataclass
class PromptLibrary:
    blocks: Dict[str, str]
    tiers: Dict[str, TierSpec]
    turn_template: str
    reflection_template: str
    dreamer_template: str
    path: str = "prompts.toml"
    _validated: bool = field(default=False, repr=False)

    @classmethod
    def load(cls, path: str = "prompts.toml") -> "PromptLibrary":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        tiers = {
            name: TierSpec(
                name=name,
                description=t.get("description", ""),
                reveals_opponent=bool(t.get("reveals_opponent", False)),
                blocks=list(t["blocks"]),
            )
            for name, t in raw.get("tiers", {}).items()
        }
        lib = cls(
            blocks=dict(raw.get("blocks", {})),
            tiers=tiers,
            turn_template=raw["turn"]["template"],
            reflection_template=raw["reflection"]["template"],
            dreamer_template=raw["dreamer"]["template"],
            path=path,
        )
        lib.validate()
        return lib

    def validate(self):
        """Fails fast on typos: missing blocks, unknown placeholders."""
        for tier in self.tiers.values():
            for block in tier.blocks:
                if block not in self.blocks:
                    raise ValueError(f"{self.path}: tier '{tier.name}' uses unknown block '{block}'")
        for name, text in self.blocks.items():
            _check_placeholders(text, BLOCK_PLACEHOLDERS, f"{self.path}: block '{name}'")
        for name, text in (("turn", self.turn_template), ("reflection", self.reflection_template),
                           ("dreamer", self.dreamer_template)):
            _check_placeholders(text, TEMPLATE_PLACEHOLDERS[name], f"{self.path}: [{name}] template")
        self._validated = True

    def tier(self, name: str) -> TierSpec:
        if name not in self.tiers:
            raise ValueError(f"Unknown prompt tier '{name}'. Available: {', '.join(self.tiers)}")
        return self.tiers[name]


def _check_placeholders(text: str, allowed: set, where: str):
    unknown = set(_PLACEHOLDER_RE.findall(text)) - allowed
    if unknown:
        raise ValueError(f"{where} uses unknown placeholder(s): {sorted(unknown)}. Allowed: {sorted(allowed)}")


def _fill(text: str, values: Dict[str, object], where: str) -> str:
    def sub(match):
        key = match.group(1)
        if key not in values or values[key] is None:
            raise ValueError(f"{where} needs '{{{key}}}' but the match context does not provide it")
        return str(values[key])
    return _PLACEHOLDER_RE.sub(sub, text)


def format_strategy_catalog(known_strategies: Optional[List[str]] = None) -> str:
    keys = [resolve_key(k) for k in known_strategies] if known_strategies else list(OPPONENTS)
    return "\n".join(f"  - {OPPONENTS[k].prompt_name}: {OPPONENTS[k].description}" for k in keys)


def format_history(game_history: List[Tuple[Move, Move]]) -> str:
    if not game_history:
        return "Nothing yet: this is the first round."
    lines = []
    for i, (mine, theirs) in enumerate(game_history, 1):
        pm, po = PAYOFFS[(mine, theirs)]
        lines.append(f"Round {i}: you chose {label(mine)}, they chose {label(theirs)} (you +{pm}, they +{po})")
    return "\n".join(lines)


def format_notes(notes: List[str]) -> str:
    return "\n".join(f"- {n}" for n in notes) if notes else "- (nothing yet)"


def render_system(lib: PromptLibrary, tier_name: str, ctx: MatchContext, round_number: int = 1) -> str:
    tier = lib.tier(tier_name)
    values: Dict[str, object] = {
        "strategy_catalog": format_strategy_catalog(ctx.known_strategies),
        "total_rounds": ctx.total_rounds,
        "avg_rounds": None if ctx.avg_rounds is None else f"{ctx.avg_rounds:g}",
        "round": round_number,
        "rounds_left": None if ctx.total_rounds is None else ctx.total_rounds - round_number + 1,
    }
    if ctx.opponent_key:
        spec = OPPONENTS[resolve_key(ctx.opponent_key)]
        values.update(opponent_name=spec.prompt_name, opponent_description=spec.description,
                      opponent_counter=spec.counter)
    parts = [_fill(lib.blocks[b], values, f"{lib.path}: block '{b}' (tier '{tier.name}')") for b in tier.blocks]
    return "\n\n".join(parts)


def build_turn_prompt(
    lib: PromptLibrary,
    tier_name: str,
    ctx: MatchContext,
    game_history: List[Tuple[Move, Move]],
    my_score: int,
    opp_score: int,
    memory_lines: List[str],
    short_notes: List[str],
    playbook: Optional[str] = None,
) -> str:
    round_number = len(game_history) + 1
    playbook_text = ""
    if playbook and playbook.strip():
        playbook_text = f"\nYOUR PLAYBOOK FROM TRAINING\n{playbook.strip()}\n"
    values = {
        "system": render_system(lib, tier_name, ctx, round_number),
        "round": round_number,
        "my_score": my_score,
        "opp_score": opp_score,
        "history": format_history(game_history),
        "memory": format_notes(memory_lines),
        "playbook": playbook_text,
        "short_notes": format_notes(short_notes),
    }
    return _fill(lib.turn_template, values, f"{lib.path}: [turn]")


def build_reflection_prompt(
    lib: PromptLibrary,
    tier_name: str,
    ctx: MatchContext,
    game_history: List[Tuple[Move, Move]],
    my_score: int,
    opp_score: int,
    short_notes: List[str],
    memory_lines: List[str],
) -> str:
    rounds = len(game_history)
    opponent_info = ""
    if lib.tier(tier_name).reveals_opponent and ctx.opponent_key:
        opponent_info = f'The other player used the strategy "{OPPONENTS[resolve_key(ctx.opponent_key)].prompt_name}".'
    values = {
        "opponent_info": opponent_info,
        "total_rounds": rounds,
        "my_score": my_score,
        "opp_score": opp_score,
        "score_per_round": f"{my_score / rounds:.2f}" if rounds else "0.00",
        "history_text": format_history(game_history),
        "short_notes_text": format_notes(short_notes),
        "memory_text": format_notes(memory_lines),
    }
    return _fill(lib.reflection_template, values, f"{lib.path}: [reflection]")


def build_dreamer_prompt(lib: PromptLibrary, training_summary: str, raw_notes: List[str]) -> str:
    values = {"training_summary": training_summary, "raw_notes_text": format_notes(raw_notes)}
    return _fill(lib.dreamer_template, values, f"{lib.path}: [dreamer]")
