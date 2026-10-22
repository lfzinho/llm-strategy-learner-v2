import re

import pytest

from src.constants import Move
from src.opponents import OPPONENTS
from src.prompts import (
    MatchContext, PromptLibrary, build_dreamer_prompt, build_reflection_prompt, build_turn_prompt, render_system,
)

LIB = PromptLibrary.load("prompts.toml")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
HISTORY = [(Move.COOPERATE, Move.DEFECT), (Move.DEFECT, Move.COOPERATE)]


def ctx(**kw):
    base = dict(opponent_key="TitForTat", total_rounds=10, avg_rounds=10.0)
    base.update(kw)
    return MatchContext(**base)


def turn(tier, **kw):
    return build_turn_prompt(LIB, tier, ctx(**kw), HISTORY, 5, 5, ["- lesson"], [], None)


def test_five_tiers_exist_and_every_one_renders_without_placeholders():
    assert list(LIB.tiers) == ["t1_informed", "t2_named", "t3_pool_hint", "t4_blind", "t5_open_ended"]
    for tier in LIB.tiers:
        prompt = turn(tier)
        assert not PLACEHOLDER.findall(prompt), f"{tier}: {PLACEHOLDER.findall(prompt)}"
        assert "Round: 3" in prompt


def test_information_ladder():
    name, counter = OPPONENTS["TitForTat"].prompt_name, OPPONENTS["TitForTat"].counter
    t1, t2, t3, t4, t5 = (turn(t) for t in LIB.tiers)
    assert name in t1 and counter in t1
    assert name in t2 and counter not in t2
    # t3 lists the whole pool (so the name appears) but never singles out the opponent or gives a counter
    assert "WHO YOU MIGHT FACE" in t3 and "YOUR OPPONENT" not in t3 and counter not in t3
    assert "WHO YOU MIGHT FACE" not in t4 and name not in t4
    assert name not in t5


def test_blind_tiers_never_leak_the_opponent():
    for tier in ("t3_pool_hint", "t4_blind", "t5_open_ended"):
        prompt = build_reflection_prompt(LIB, tier, ctx(), HISTORY, 5, 5, [], [])
        assert "TitForTat" not in prompt and "Alpha Mimic" not in prompt and "Tit For Tat" not in prompt


def test_revealing_tier_names_opponent_in_reflection():
    prompt = build_reflection_prompt(LIB, "t1_informed", ctx(), HISTORY, 5, 5, [], [])
    assert "Alpha Mimic" in prompt


def test_horizon_wording_differs_by_tier():
    assert "exactly 10 rounds" in turn("t1_informed")
    assert "around 10 rounds" in turn("t4_blind")
    assert "do not know how long" in turn("t5_open_ended")
    assert "Normal(" not in turn("t4_blind")


def test_known_horizon_counts_down():
    late = build_turn_prompt(LIB, "t2_named", ctx(), HISTORY * 4, 0, 0, [], [], None)
    assert "so 2 rounds remain" in late  # round 9 of 10


def test_prompts_use_neutral_labels():
    for tier in LIB.tiers:
        prompt = turn(tier).lower()
        for word in ("cooperate", "defect", "prisoner", "nash", "tit for tat"):
            assert word not in prompt, f"{tier} mentions '{word}'"


def test_history_shows_points_per_round():
    prompt = turn("t4_blind")
    assert "Round 1: you chose Alpha, they chose Beta (you +0, they +5)" in prompt
    assert "Round 2: you chose Beta, they chose Alpha (you +5, they +0)" in prompt


def test_playbook_and_memory_sections_are_rendered():
    prompt = build_turn_prompt(LIB, "t4_blind", ctx(), [], 0, 0, ["- remembered thing"], ["short note"], "PLAY NICE")
    assert "PLAY NICE" in prompt and "remembered thing" in prompt and "short note" in prompt


def test_missing_context_is_a_clear_error():
    with pytest.raises(ValueError, match="total_rounds"):
        render_system(LIB, "t1_informed", MatchContext(opponent_key="TitForTat"))
    with pytest.raises(ValueError, match="Unknown prompt tier"):
        LIB.tier("nope")


def test_typos_in_prompts_file_fail_fast(tmp_path):
    bad = tmp_path / "bad.toml"
    text = open("prompts.toml", encoding="utf-8").read().replace('strategy "{opponent_name}"', 'strategy "{oponent_name}"', 1)
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown placeholder"):
        PromptLibrary.load(str(bad))

    bad.write_text(open("prompts.toml", encoding="utf-8").read().replace('"horizon_known", "opponent_named"]', '"horizon_knwon"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown block"):
        PromptLibrary.load(str(bad))


def test_dreamer_prompt_includes_results_and_lessons():
    prompt = build_dreamer_prompt(LIB, "- Alpha Mimic: 3.00 points/round", ["lesson one"])
    assert "3.00 points/round" in prompt and "lesson one" in prompt
    assert not PLACEHOLDER.findall(prompt)
