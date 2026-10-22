"""Opponent registry backed by the Axelrod library.

Each entry maps a config key to an Axelrod strategy plus the display name and
description used in prompts and reports.
"""
from dataclasses import dataclass
from typing import Dict, List, Type

import axelrod as axl


@dataclass(frozen=True)
class OpponentSpec:
    key: str
    axl_class: Type[axl.Player]
    display_name: str
    description: str


_SPECS: List[OpponentSpec] = [
    OpponentSpec("AlwaysCooperate", axl.Cooperator, "Always Cooperate",
                 "Always plays COOPERATE unconditionally."),
    OpponentSpec("AlwaysDefect", axl.Defector, "Always Defect",
                 "Always plays DEFECT unconditionally."),
    OpponentSpec("TitForTat", axl.TitForTat, "Tit For Tat",
                 "Starts with COOPERATE, then copies whatever the opponent played in the previous round."),
    OpponentSpec("SuspiciousTitForTat", axl.SuspiciousTitForTat, "Suspicious Tit For Tat",
                 "Starts with DEFECT, then copies whatever the opponent played in the previous round."),
    OpponentSpec("TitForTwoTats", axl.TitFor2Tats, "Tit For Two Tats",
                 "Cooperates initially and only retaliates with DEFECT if the opponent defected in BOTH of the last two consecutive rounds."),
    OpponentSpec("TatForTit", axl.AntiTitForTat, "Tat For Tit",
                 "Starts with COOPERATE, then plays the opposite of whatever the opponent played in the last round."),
    OpponentSpec("Random", axl.Random, "Random",
                 "Randomly chooses COOPERATE or DEFECT with equal probability (unpredictable)."),
    OpponentSpec("Detective", axl.Detective, "Detective",
                 "Plays [C, D, C, C] as an opening probe sequence. If the opponent ever defected, it plays Tit For Tat; if the opponent never retaliated, it defects forever to exploit."),
    OpponentSpec("GrimTrigger", axl.Grudger, "Grim Trigger",
                 "Starts with COOPERATE, but if the opponent defects even once, it permanently defects for the rest of the match."),
    OpponentSpec("Pavlov", axl.WinStayLoseShift, "Pavlov",
                 "Win-Stay, Lose-Shift. Repeats move if round was (C,C) or (D,D); shifts move if (C,D) or (D,C)."),
]

OPPONENTS: Dict[str, OpponentSpec] = {s.key: s for s in _SPECS}

STRATEGY_DESCRIPTIONS: Dict[str, str] = {s.key: s.description for s in _SPECS}


def resolve_key(name: str) -> str:
    """Resolves a loose name ('tit for tat', 'Tit-For-Tat') to a registry key."""
    normalized = name.replace(" ", "").replace("_", "").replace("-", "").lower()
    for key in OPPONENTS:
        if key.lower() == normalized:
            return key
    raise ValueError(f"Unknown strategy name: '{name}'. Available: {list(OPPONENTS)}")


def make_opponent(name: str) -> axl.Player:
    """Builds a fresh Axelrod player whose `name` is the project display name."""
    spec = OPPONENTS[resolve_key(name)]
    player = spec.axl_class()
    player.name = spec.display_name
    return player
