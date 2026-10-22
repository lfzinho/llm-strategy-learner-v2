"""Opponent registry backed by the Axelrod library.

Each entry maps a config key to an Axelrod strategy plus:
- `display_name`: the conventional name, used in reports.
- `prompt_name`: a neutral name (Alpha/Beta vocabulary) used inside prompts so the
  model cannot lean on what it remembers about the textbook strategy names.
- `description` / `counter`: neutral prose used by the informed prompt tiers.
  Counters are verified against the real strategies in tests/test_opponents.py.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import axelrod as axl


@dataclass(frozen=True)
class OpponentSpec:
    key: str
    axl_class: Type[axl.Player]
    display_name: str
    prompt_name: str
    description: str
    counter: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)


_SPECS: List[OpponentSpec] = [
    OpponentSpec(
        "AlwaysCooperate", axl.Cooperator, "Always Cooperate", "Always Alpha",
        "Plays Alpha every round, no matter what you do.",
        "Play Beta every round. It never reacts, so you collect 5 points per round.",
    ),
    OpponentSpec(
        "AlwaysDefect", axl.Defector, "Always Defect", "Always Beta",
        "Plays Beta every round, no matter what you do.",
        "Play Beta every round. Alpha would only give it free points and earn you 0.",
    ),
    OpponentSpec(
        "TitForTat", axl.TitForTat, "Tit For Tat", "Alpha Mimic",
        "Plays Alpha in round 1, then repeats whatever you played in the previous round.",
        "Play Alpha every round. Mutual Alpha pays 3 per round; every Beta you play is answered with a Beta next round.",
    ),
    OpponentSpec(
        "SuspiciousTitForTat", axl.SuspiciousTitForTat, "Suspicious Tit For Tat", "Beta Mimic",
        "Plays Beta in round 1, then repeats whatever you played in the previous round.",
        "Play Alpha every round. You take one 0-point round at the start, after which it copies your Alpha and pays 3 per round; answering its opening Beta with Beta locks both of you at 1 per round.",
    ),
    OpponentSpec(
        "TitForTwoTats", axl.TitFor2Tats, "Tit For Two Tats", "Zen Alpha Mimic",
        "Plays Alpha, and only plays Beta after you played Beta in two consecutive rounds. A single Beta is forgiven.",
        "Alternate Beta and Alpha, starting with Beta. It never sees two Betas in a row, so it keeps playing Alpha and you average 4 points per round.",
    ),
    OpponentSpec(
        "TatForTit", axl.AntiTitForTat, "Tat For Tit", "Reverse Mimic",
        "Plays Alpha in round 1, then plays the opposite of what you played in the previous round (Alpha after your Beta, Beta after your Alpha).",
        "Play Beta every round. It answers every Beta with Alpha, so you collect 5 points per round. Playing Alpha only makes it play Beta against you.",
    ),
    OpponentSpec(
        "Random", axl.Random, "Random", "Random Choice",
        "Plays Alpha or Beta with equal probability, independent of what you do.",
        "Play Beta every round. Its moves ignore yours, and Beta pays more than Alpha for either move it makes (3.0 versus 1.5 points per round on average).",
        aliases=("random",),
    ),
    OpponentSpec(
        "Detective", axl.Detective, "Detective", "Prober",
        "Opens with Alpha, Beta, Alpha, Alpha as a probe. If you never played Beta during those four rounds it plays Beta for the rest of the match; otherwise it repeats your previous move each round.",
        "Play Beta in rounds 1 to 3 so the probe sees you retaliate, then play Alpha every round. If you stay Alpha during the probe it turns on you for good.",
        aliases=("probe", "probing"),
    ),
    OpponentSpec(
        "GrimTrigger", axl.Grudger, "Grim Trigger", "Beta Resentful",
        "Plays Alpha until you play Beta once, then plays Beta for the rest of the match.",
        "Play Alpha every round. Never play Beta: it cannot be forgiven, and a single Beta costs you every remaining round.",
        aliases=("grudger", "grim"),
    ),
    OpponentSpec(
        "Pavlov", axl.WinStayLoseShift, "Pavlov", "Stay-Shift",
        "Plays Alpha in round 1. After a round where both players made the same move it repeats its own move; after a round where the moves differed it switches to the other move.",
        "Play Alpha every round. Both of you settle into mutual Alpha at 3 points per round.",
        aliases=("win-stay", "win stay", "wsls"),
    ),
]

OPPONENTS: Dict[str, OpponentSpec] = {s.key: s for s in _SPECS}


def resolve_key(name: str) -> str:
    """Resolves a loose name ('tit for tat', 'Alpha Mimic', 'TitForTat') to a registry key."""
    normalized = name.replace(" ", "").replace("_", "").replace("-", "").lower()
    for spec in _SPECS:
        candidates = {spec.key, spec.display_name, spec.prompt_name}
        if normalized in {c.replace(" ", "").replace("-", "").lower() for c in candidates}:
            return spec.key
    raise ValueError(f"Unknown strategy name: '{name}'. Available: {list(OPPONENTS)}")


def make_opponent(name: str) -> axl.Player:
    """Builds a fresh Axelrod player whose `name` is the project display name."""
    spec = OPPONENTS[resolve_key(name)]
    player = spec.axl_class()
    player.name = spec.display_name
    return player


def key_for_display_name(display_name: str):
    """Maps a report display name back to a registry key (None for non-pool players)."""
    for spec in _SPECS:
        if spec.display_name == display_name:
            return spec.key
    return None


def identify_label(text: str):
    """Returns the registry key an LLM hypothesis names, or None.

    The longest matching name wins, so 'Zen Alpha Mimic' is not read as 'Alpha Mimic'.
    """
    haystack = text.lower().replace("-", " ")
    best_key, best_len = None, 0
    for spec in _SPECS:
        for name in {spec.display_name, spec.prompt_name, *spec.aliases}:
            needle = name.lower().replace("-", " ")
            if needle in haystack and len(needle) > best_len:
                best_key, best_len = spec.key, len(needle)
    return best_key
