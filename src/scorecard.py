"""Run scorecard: answers "is the model playing well, and if not, why?"

Per opponent it measures points per round against the best achievable score (found by
searching every fixed move sequence against that deterministic opponent), how often the
model cooperated, and (in tiers that hide the opponent) how quickly it named the rule.
It also reports how many calls failed, so API trouble is never mistaken for bad play.
"""
import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import axelrod as axl

from src.match_runner import play_match
from src.opponents import OPPONENTS, identify_label, key_for_display_name, make_opponent

EXHAUSTIVE_LIMIT = 10  # search every sequence up to this many rounds; longer matches use patterns
WARN_EFFICIENCY = 0.70
GOOD_EFFICIENCY = 0.90


@dataclass
class MatchRecord:
    opponent_key: str
    llm_moves: str  # "CDC..." (C = Alpha, D = Beta)
    opp_moves: str
    llm_score: int
    opp_score: int
    hypotheses: List[str] = field(default_factory=list)  # one per round, may be empty

    @property
    def rounds(self) -> int:
        return len(self.llm_moves)


def _to_actions(seq: str) -> List[axl.Action]:
    return [axl.Action.C if c == "C" else axl.Action.D for c in seq]


def _score_sequence(key: str, seq: str) -> int:
    me = axl.MockPlayer(actions=_to_actions(seq))
    return play_match(me, make_opponent(key), len(seq), seed=0).score1


def _patterns(turns: int) -> List[str]:
    """Candidate sequences for long matches (a lower bound on the true optimum)."""
    base = ["C", "D", "DC", "CD", "DDDC", "CD" + "C"]
    out = set()
    for b in base:
        out.add((b * turns)[:turns])
    out.add("C" * (turns - 1) + "D")
    out.add("D" * 3 + "C" * (turns - 3) if turns > 3 else "D" * turns)
    out.add("DC" * ((turns - 2) // 2) + "DD" if turns > 3 else "D" * turns)
    return list(out)


@lru_cache(maxsize=None)
def best_achievable(key: str, turns: int) -> float:
    """Highest score a fixed sequence of `turns` rounds can earn against the opponent."""
    if key == "Random":  # stochastic: Beta every round is optimal, 3.0 points per round on average
        return 3.0 * turns
    if turns <= EXHAUSTIVE_LIMIT:
        return float(max(_score_sequence(key, "".join(s)) for s in itertools.product("CD", repeat=turns)))
    return float(max(_score_sequence(key, p) for p in _patterns(turns)))


def build_scorecard(
    records: List[MatchRecord],
    llm_stats: Optional[Dict[str, int]] = None,
    tier: str = "",
    phase: str = "",
    reveals_opponent: bool = False,
) -> Dict[str, Any]:
    llm_stats = dict(llm_stats or {})
    groups: Dict[str, List[MatchRecord]] = defaultdict(list)
    for r in records:
        groups[r.opponent_key].append(r)

    per_opponent: Dict[str, Any] = {}
    total_score = total_best = total_rounds = total_alpha = total_mutual = 0
    for key, recs in groups.items():
        rounds = sum(r.rounds for r in recs)
        score = sum(r.llm_score for r in recs)
        best = sum(best_achievable(key, r.rounds) for r in recs)
        alpha = sum(r.llm_moves.count("C") for r in recs)
        mutual = sum(1 for r in recs for a, b in zip(r.llm_moves, r.opp_moves) if a == b == "C")
        entry: Dict[str, Any] = {
            "opponent": OPPONENTS[key].display_name,
            "prompt_name": OPPONENTS[key].prompt_name,
            "matches": len(recs),
            "rounds": rounds,
            "points_per_round": round(score / rounds, 2) if rounds else 0.0,
            "best_points_per_round": round(best / rounds, 2) if rounds else 0.0,
            "efficiency": round(score / best, 3) if best else 1.0,
            "regret_per_round": round((best - score) / rounds, 2) if rounds else 0.0,
            "alpha_rate": round(alpha / rounds, 3) if rounds else 0.0,
            "mutual_alpha_rate": round(mutual / rounds, 3) if rounds else 0.0,
        }
        if not reveals_opponent:
            ident = _identification(recs, key)
            if ident:
                entry.update(ident)
        per_opponent[key] = entry
        total_score += score
        total_best += best
        total_rounds += rounds
        total_alpha += alpha
        total_mutual += mutual

    overall = {
        "matches": len(records),
        "rounds": total_rounds,
        "points_per_round": round(total_score / total_rounds, 2) if total_rounds else 0.0,
        "efficiency": round(total_score / total_best, 3) if total_best else 1.0,
        "alpha_rate": round(total_alpha / total_rounds, 3) if total_rounds else 0.0,
        "mutual_alpha_rate": round(total_mutual / total_rounds, 3) if total_rounds else 0.0,
    }
    ident_values = [e["identification_accuracy"] for e in per_opponent.values() if "identification_accuracy" in e]
    if ident_values:
        overall["identification_accuracy"] = round(sum(ident_values) / len(ident_values), 3)

    card = {
        "tier": tier, "phase": phase, "reveals_opponent": reveals_opponent,
        "overall": overall, "per_opponent": per_opponent, "llm_stats": llm_stats,
    }
    card["verdicts"] = _verdicts(card)
    return card


def _identification(recs: List[MatchRecord], key: str) -> Optional[Dict[str, Any]]:
    """How often the model's stated hypothesis named the true opponent, and how fast."""
    with_hyp = [r for r in recs if r.hypotheses]
    if not with_hyp:
        return None
    correct = rounds = 0
    first_rounds = []
    for r in with_hyp:
        hit = [identify_label(h) == key for h in r.hypotheses if h]
        rounds += len(hit)
        correct += sum(hit)
        if any(hit):
            first_rounds.append(hit.index(True) + 1)
    if not rounds:
        return None
    return {
        "identification_accuracy": round(correct / rounds, 3),
        "rounds_to_identify": round(sum(first_rounds) / len(first_rounds), 1) if first_rounds else None,
        "identified_in_matches": f"{len(first_rounds)}/{len(with_hyp)}",
    }


def _verdicts(card: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    stats = card["llm_stats"]
    turns = stats.get("turns", 0)
    fallbacks = stats.get("fallbacks", 0)
    if fallbacks:
        share = 100 * fallbacks / turns if turns else 100
        level = "FAIL" if share > 20 else "WARN"
        out.append({"level": level, "message": f"{fallbacks} of {turns} rounds ({share:.0f}%) were played by the fallback rule "
                                               f"because the LLM call failed. Results mix model play with fallback play; see trace.jsonl."})
    if stats.get("parse_failures"):
        out.append({"level": "WARN", "message": f"{stats['parse_failures']} responses could not be parsed (model output not valid JSON for the schema)."})
    if stats.get("reflection_failures"):
        out.append({"level": "WARN", "message": f"{stats['reflection_failures']} post-match reflections failed, so those matches added no memory."})

    overall = card["overall"]
    if overall["rounds"] and overall["alpha_rate"] < 0.15 and overall["efficiency"] < WARN_EFFICIENCY:
        out.append({"level": "WARN", "message": f"The model chose Alpha only {overall['alpha_rate']:.0%} of the time and earned "
                                                f"{overall['efficiency']:.0%} of the best achievable points: it is stuck in Beta/Beta."})
    for entry in card["per_opponent"].values():
        eff = entry["efficiency"]
        text = (f"vs {entry['opponent']}: {entry['points_per_round']} points/round of a possible "
                f"{entry['best_points_per_round']} ({eff:.0%})")
        if eff < WARN_EFFICIENCY:
            out.append({"level": "WARN", "message": text + f"; Alpha rate {entry['alpha_rate']:.0%}."})
        elif eff >= GOOD_EFFICIENCY:
            out.append({"level": "PASS", "message": text})
        else:
            out.append({"level": "OK", "message": text})
    if not any(v["level"] in ("WARN", "FAIL") for v in out):
        out.insert(0, {"level": "PASS", "message": "No problems detected."})
    return out


# ------------------------------------------------------------------ output
def render_markdown(card: Dict[str, Any]) -> str:
    o = card["overall"]
    lines = [
        f"# Scorecard: {card['tier'] or 'run'} ({card['phase'] or 'all phases'})",
        "",
        f"- Matches: {o['matches']} | Rounds: {o['rounds']}",
        f"- Points per round: **{o['points_per_round']}** | Efficiency: **{o['efficiency']:.0%}** of the best achievable",
        f"- Alpha (cooperate) rate: {o['alpha_rate']:.0%} | Mutual Alpha: {o['mutual_alpha_rate']:.0%}",
    ]
    if "identification_accuracy" in o:
        lines.append(f"- Opponent identification accuracy: {o['identification_accuracy']:.0%}")
    lines += ["", "| Opponent | Matches | Points/round | Best possible | Efficiency | Alpha rate | Identified |",
              "|---|---|---|---|---|---|---|"]
    for e in card["per_opponent"].values():
        ident = f"{e['identification_accuracy']:.0%} (round {e['rounds_to_identify']:g})" if "identification_accuracy" in e and e.get("rounds_to_identify") else (
            f"{e['identification_accuracy']:.0%}" if "identification_accuracy" in e else "n/a")
        lines.append(f"| {e['opponent']} | {e['matches']} | {e['points_per_round']} | {e['best_points_per_round']} | "
                     f"{e['efficiency']:.0%} | {e['alpha_rate']:.0%} | {ident} |")
    s = card["llm_stats"]
    if s:
        lines += ["", "LLM health: " + ", ".join(f"{k}={v}" for k, v in sorted(s.items()))]
    lines += ["", "## Verdicts"] + [f"- **{v['level']}**: {v['message']}" for v in card["verdicts"]]
    return "\n".join(lines) + "\n"


def print_scorecard(card: Dict[str, Any], console=None):
    from rich.console import Console
    from rich.table import Table
    console = console or Console()
    o = card["overall"]
    table = Table(title=f"Scorecard - {card['tier']} ({card['phase']})  |  {o['points_per_round']} points/round, "
                        f"{o['efficiency']:.0%} of best achievable, Alpha {o['alpha_rate']:.0%}")
    for col in ("Opponent", "Matches", "Points/round", "Best", "Efficiency", "Alpha rate", "Identified"):
        table.add_column(col, justify="left" if col == "Opponent" else "right")
    for e in card["per_opponent"].values():
        eff = e["efficiency"]
        color = "green" if eff >= GOOD_EFFICIENCY else ("red" if eff < WARN_EFFICIENCY else "yellow")
        ident = f"{e['identification_accuracy']:.0%}" if "identification_accuracy" in e else "n/a"
        table.add_row(e["opponent"], str(e["matches"]), str(e["points_per_round"]), str(e["best_points_per_round"]),
                      f"[{color}]{eff:.0%}[/{color}]", f"{e['alpha_rate']:.0%}", ident)
    console.print(table)
    colors = {"PASS": "green", "OK": "white", "WARN": "yellow", "FAIL": "red"}
    for v in card["verdicts"]:
        if v["level"] in ("WARN", "FAIL") or v["message"] == "No problems detected.":
            console.print(f"[{colors[v['level']]}]{v['level']}[/{colors[v['level']]}] {v['message']}")


def write_scorecard(card: Dict[str, Any], output_dir: str, name: str = "scorecard") -> str:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    md_path = os.path.join(output_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(card))
    return md_path


def compare_tiers_markdown(cards: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# Tier comparison", "",
             "| Tier | Phase | Points/round | Efficiency | Alpha rate | Identification | Fallbacks |",
             "|---|---|---|---|---|---|---|"]
    for tier, card in cards.items():
        o = card["overall"]
        ident = f"{o['identification_accuracy']:.0%}" if "identification_accuracy" in o else "n/a"
        lines.append(f"| {tier} | {card['phase']} | {o['points_per_round']} | {o['efficiency']:.0%} | "
                     f"{o['alpha_rate']:.0%} | {ident} | {card['llm_stats'].get('fallbacks', 0)} |")
    lines += ["", "Efficiency = points earned / best points achievable against the same opponents "
                  "(1.00 means the model found the best fixed response every match)."]
    return "\n".join(lines) + "\n"


def records_from_training(history: List[Dict[str, Any]]) -> List[MatchRecord]:
    out = []
    for e in history:
        key = key_for_display_name(e.get("opponent_strategy", ""))
        if key is None or not e.get("moves"):
            continue
        out.append(MatchRecord(
            opponent_key=key,
            llm_moves="".join(m[0] for m in e["moves"]),
            opp_moves="".join(m[1] for m in e["moves"]),
            llm_score=e["llm_score"], opp_score=e["opponent_score"],
            hypotheses=[t.get("hypothesis", "") for t in e.get("turns", [])],
        ))
    return out


def records_from_tournament(matches: List[Dict[str, Any]], llm_name: str) -> List[MatchRecord]:
    out = []
    for m in matches:
        if llm_name not in (m["player1"], m["player2"]) or not m.get("moves"):
            continue
        first = m["player1"] == llm_name
        key = key_for_display_name(m["player2"] if first else m["player1"])
        if key is None:
            continue
        idx = 0 if first else 1
        out.append(MatchRecord(
            opponent_key=key,
            llm_moves="".join(x[idx] for x in m["moves"]),
            opp_moves="".join(x[1 - idx] for x in m["moves"]),
            llm_score=m["p1_score"] if first else m["p2_score"],
            opp_score=m["p2_score"] if first else m["p1_score"],
            hypotheses=[t.get("hypothesis", "") for t in m.get("llm_turns", [])],
        ))
    return out
