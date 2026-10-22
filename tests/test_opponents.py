import pytest
import axelrod as axl

from src.match_runner import play_match
from src.opponents import OPPONENTS, make_opponent, resolve_key


def moves_vs(opponent_key, my_moves):
    """Plays a scripted sequence against an opponent; returns the opponent's moves."""
    script = axl.MockPlayer(actions=[axl.Action.C if m == "C" else axl.Action.D for m in my_moves])
    result = play_match(script, make_opponent(opponent_key), turns=len(my_moves), seed=0)
    return "".join(m[1] for m in result.moves)


def test_payoffs_are_standard():
    assert play_match(make_opponent("AlwaysCooperate"), make_opponent("AlwaysCooperate"), 4).score1 == 12
    r = play_match(make_opponent("AlwaysDefect"), make_opponent("AlwaysCooperate"), 4)
    assert (r.score1, r.score2) == (20, 0)
    r = play_match(make_opponent("AlwaysDefect"), make_opponent("AlwaysDefect"), 4)
    assert (r.score1, r.score2) == (4, 4)


def test_always_strategies():
    assert moves_vs("AlwaysCooperate", "DDDD") == "CCCC"
    assert moves_vs("AlwaysDefect", "CCCC") == "DDDD"


def test_tit_for_tat_variants():
    assert moves_vs("TitForTat", "CDDC") == "CCDD"
    assert moves_vs("SuspiciousTitForTat", "CDDC") == "DCDD"
    assert moves_vs("TitForTwoTats", "DCDDC") == "CCCCD"


def test_tat_for_tit_inverts_last_move():
    assert moves_vs("TatForTit", "CCDC") == "CDDC"


def test_detective_probes_then_exploits_cooperators():
    assert moves_vs("Detective", "CCCCCC") == "CDCCDD"


def test_detective_plays_tft_after_retaliation():
    # Opponent defects in the probe, so Detective copies from round 5 on.
    assert moves_vs("Detective", "DCCCDC") == "CDCCCD"


def test_grim_trigger_never_forgives():
    assert moves_vs("GrimTrigger", "CCDCCC") == "CCCDDD"


def test_pavlov_win_stay_lose_shift():
    # C,C -> stay C; (C,D) -> shift to D; (D,D) -> stay D... vs a defector from round 2
    assert moves_vs("Pavlov", "CDDD") == "CCDC"


def test_registry_names_resolve():
    assert resolve_key("tit for tat") == "TitForTat"
    assert resolve_key("Tit-For-Two-Tats") == "TitForTwoTats"
    with pytest.raises(ValueError):
        resolve_key("nope")


def test_display_names_are_stable():
    for key, spec in OPPONENTS.items():
        assert make_opponent(key).name == spec.display_name


# --- the counter-strategies written into the T1 prompt must actually work ---------------
COUNTER_SEQUENCES = {
    "AlwaysCooperate": lambda n: "D" * n,
    "AlwaysDefect": lambda n: "D" * n,
    "TitForTat": lambda n: "C" * n,
    "SuspiciousTitForTat": lambda n: "C" * n,
    "TitForTwoTats": lambda n: ("DC" * n)[:n],
    "TatForTit": lambda n: "D" * n,
    "Detective": lambda n: "DDD" + "C" * (n - 3),
    "GrimTrigger": lambda n: "C" * n,
    "Pavlov": lambda n: "C" * n,
}


@pytest.mark.parametrize("key", list(COUNTER_SEQUENCES))
def test_counter_strategy_scores_within_10_percent_of_the_optimum(key):
    from src.scorecard import best_achievable

    n = 10
    score = play_match(
        axl.MockPlayer(actions=[axl.Action.C if c == "C" else axl.Action.D for c in COUNTER_SEQUENCES[key](n)]),
        make_opponent(key), n, seed=0,
    ).score1
    assert score >= 0.9 * best_achievable(key, n), f"{key}: counter scores {score}, optimum {best_achievable(key, n)}"


def test_random_counter_beta_beats_alpha_on_average():
    n, total = 10, {"C": 0, "D": 0}
    for seed in range(60):
        for move in "CD":
            total[move] += play_match(axl.MockPlayer(actions=[axl.Action[move]]), make_opponent("Random"), n, seed=seed).score1
    assert total["D"] > 1.6 * total["C"]


def test_every_pool_opponent_has_a_neutral_description_and_counter():
    for spec in OPPONENTS.values():
        assert spec.description and spec.counter
        for word in ("cooperate", "defect", "tit for tat"):
            assert word not in (spec.description + spec.counter + spec.prompt_name).lower(), (spec.key, word)


def test_identify_label_prefers_the_longest_name():
    from src.opponents import identify_label
    assert identify_label("looks like a Zen Alpha Mimic to me") == "TitForTwoTats"
    assert identify_label("probably an alpha mimic") == "TitForTat"
    assert identify_label("Tit-for-Tat") == "TitForTat"
    assert identify_label("something unknown") is None
