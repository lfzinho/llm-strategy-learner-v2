import pytest

from src.scorecard import (
    MatchRecord, best_achievable, build_scorecard, compare_tiers_markdown, records_from_tournament,
    records_from_training, render_markdown,
)


def record(key, mine, theirs, score, opp_score, hypotheses=None):
    return MatchRecord(key, mine, theirs, score, opp_score, hypotheses or [])


@pytest.mark.parametrize("key,expected", [
    ("AlwaysCooperate", 50), ("AlwaysDefect", 10), ("TitForTat", 32), ("TitForTwoTats", 42),
    ("TatForTit", 50), ("GrimTrigger", 32), ("Random", 30.0),
])
def test_best_achievable_matches_brute_force(key, expected):
    assert best_achievable(key, 10) == expected


def test_best_achievable_for_long_matches_is_a_sound_lower_bound():
    assert best_achievable("AlwaysCooperate", 14) == 70
    assert best_achievable("TitForTat", 14) >= 14 * 3
    assert best_achievable("TatForTit", 14) == 70


def test_good_play_is_efficient_and_passes():
    card = build_scorecard([record("TitForTat", "C" * 10, "C" * 10, 30, 30)], {"turns": 10})
    entry = card["per_opponent"]["TitForTat"]
    assert entry["efficiency"] == pytest.approx(30 / 32, abs=0.001)
    assert entry["alpha_rate"] == 1.0 and entry["mutual_alpha_rate"] == 1.0
    assert card["verdicts"][0]["level"] == "PASS"
    assert not any(v["level"] in ("WARN", "FAIL") for v in card["verdicts"])


def test_defect_collapse_is_flagged():
    # The failure seen in the old run: mutual Beta against a forgiving opponent.
    card = build_scorecard([record("TitForTwoTats", "D" * 10, "CD" + "D" * 8, 5 + 9, 0 + 9)], {})
    entry = card["per_opponent"]["TitForTwoTats"]
    assert entry["efficiency"] < 0.5
    messages = " ".join(v["message"] for v in card["verdicts"] if v["level"] == "WARN")
    assert "Tit For Two Tats" in messages and "stuck in Beta/Beta" in messages


def test_fallbacks_and_parse_failures_are_reported_separately_from_play_quality():
    card = build_scorecard([record("TitForTat", "C" * 10, "C" * 10, 30, 30)],
                           {"turns": 10, "fallbacks": 3, "parse_failures": 4, "reflection_failures": 1})
    levels = {v["level"]: v["message"] for v in card["verdicts"]}
    assert "3 of 10 rounds (30%)" in levels["FAIL"]
    assert "4 responses could not be parsed" in " ".join(v["message"] for v in card["verdicts"])
    assert "1 post-match reflections failed" in " ".join(v["message"] for v in card["verdicts"])

    mild = build_scorecard([record("TitForTat", "C" * 10, "C" * 10, 30, 30)], {"turns": 100, "fallbacks": 2})
    assert any(v["level"] == "WARN" and "2 of 100" in v["message"] for v in mild["verdicts"])


def test_identification_accuracy_and_speed():
    hyps = ["unsure", "Zen Alpha Mimic?", "Alpha Mimic", "Alpha Mimic", "alpha mimic"]
    card = build_scorecard([record("TitForTat", "C" * 5, "C" * 5, 15, 15, hyps)], {}, reveals_opponent=False)
    entry = card["per_opponent"]["TitForTat"]
    assert entry["identification_accuracy"] == pytest.approx(3 / 5)
    assert entry["rounds_to_identify"] == 3
    assert card["overall"]["identification_accuracy"] == pytest.approx(0.6)


def test_identification_is_skipped_when_the_tier_reveals_the_opponent():
    card = build_scorecard([record("TitForTat", "C" * 5, "C" * 5, 15, 15, ["Alpha Mimic"] * 5)], {}, reveals_opponent=True)
    assert "identification_accuracy" not in card["per_opponent"]["TitForTat"]


def test_records_from_training_and_tournament():
    history = [{"opponent_strategy": "Tit For Tat", "moves": ["CC", "CD"], "llm_score": 3, "opponent_score": 8,
                "turns": [{"hypothesis": "a"}, {"hypothesis": "b"}]}]
    rec = records_from_training(history)[0]
    assert (rec.opponent_key, rec.llm_moves, rec.opp_moves, rec.hypotheses) == ("TitForTat", "CC", "CD", ["a", "b"])

    matches = [
        {"player1": "LLM", "player2": "Random", "p1_score": 4, "p2_score": 1, "moves": ["DC", "DD"]},
        {"player1": "Random", "player2": "LLM", "p1_score": 1, "p2_score": 4, "moves": ["CD", "DD"]},
        {"player1": "Random", "player2": "Detective", "p1_score": 1, "p2_score": 1, "moves": ["CC"]},
    ]
    recs = records_from_tournament(matches, "LLM")
    assert len(recs) == 2
    assert all(r.llm_moves == "DD" and r.opp_moves in ("CD", "CD") for r in recs)
    assert recs[1].llm_score == 4 and recs[1].opp_score == 1


def test_markdown_and_comparison_render():
    card = build_scorecard([record("TitForTat", "C" * 10, "C" * 10, 30, 30)], {"turns": 10}, tier="t4_blind", phase="training")
    md = render_markdown(card)
    assert "Scorecard: t4_blind" in md and "Tit For Tat" in md and "## Verdicts" in md
    table = compare_tiers_markdown({"t4_blind": card})
    assert "| t4_blind | training |" in table
