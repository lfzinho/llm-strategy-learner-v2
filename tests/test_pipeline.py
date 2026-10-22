import json
import os
import sys

import pytest

import main as main_module
from src.llm_player import LLMDecision, LLMPlayer, LLMUnavailableError
from src.opponents import make_opponent
from src.prompts import PromptLibrary
from src.tournament import RoundRobinTournament
from src.training_battery import TrainingBattery

LIB = PromptLibrary.load("prompts.toml")
STRATEGIES = ["AlwaysCooperate", "TitForTat", "TatForTit"]


def mock_player(tier="t4_blind", **kw):
    player = LLMPlayer(provider="mock", tier=tier, prompt_library=LIB, **kw)
    player.show_panels = False
    return player


def battery(player, **kw):
    kw.setdefault("strategies", STRATEGIES)
    kw.setdefault("matches_per_strategy", 2)
    kw.setdefault("rounds_per_match", 4)
    kw.setdefault("seed", 1)
    return TrainingBattery(player, **kw)


def test_full_training_run_writes_every_artifact(tmp_path):
    player = mock_player()
    player.configure_output(str(tmp_path))
    history, playbook = battery(player).run_training(str(tmp_path))

    assert len(history) == 6 and playbook.startswith("# Mock playbook")
    for name in ("training_history.json", "master_strategy_playbook.md", "master_strategy_playbook.json",
                 "memory.json", "scorecard_training.json", "scorecard_training.md", "trace.jsonl"):
        assert (tmp_path / name).exists(), name
    assert not (tmp_path / "training_checkpoint.json").exists()  # cleared only at the very end

    entry = history[0]
    assert entry["moves"] and len(entry["moves"]) == entry["rounds_played"] == 4
    assert len(entry["turns"]) == 4 and "hypothesis" in entry["turns"][0]
    assert json.loads((tmp_path / "master_strategy_playbook.json").read_text())["fallback"] is False


def test_training_resumes_after_an_llm_outage_without_repeating_matches(tmp_path):
    class FlakyPlayer(LLMPlayer):
        fail_after = 14  # decisions before the "outage" (4 rounds per match -> mid third match)

        def _mock_response(self, response_model):
            if response_model is LLMDecision:
                FlakyPlayer.fail_after -= 1
                if FlakyPlayer.fail_after < 0:
                    raise ConnectionError("API down")
            return super()._mock_response(response_model)

    flaky = FlakyPlayer(provider="mock", tier="t4_blind", prompt_library=LIB, max_retries=1,
                        max_consecutive_fallbacks=2, retry_sleep=0)
    flaky.show_panels = False
    flaky.configure_output(str(tmp_path))
    with pytest.raises(LLMUnavailableError, match="2 consecutive"):
        battery(flaky).run_training(str(tmp_path))

    assert (tmp_path / "training_checkpoint.json").exists()
    saved = json.loads((tmp_path / "training_history.json").read_text())
    assert len(saved) == 3  # finished matches survive the crash

    player = mock_player()
    player.configure_output(str(tmp_path))
    history, _ = battery(player).run_training(str(tmp_path))
    assert [h["global_match"] for h in history] == [1, 2, 3, 4, 5, 6]
    assert [h["opponent_strategy"] for h in history].count("Tit For Tat") == 2
    assert not (tmp_path / "training_checkpoint.json").exists()

    # The trace shows why the first attempt died.
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert any(r["kind"] == "turn" and r.get("failed") and "API down" in r["errors"][0] for r in trace)
    assert sum(r["kind"] == "fallback" for r in trace) == 2


def test_corrupt_checkpoint_is_reported_and_training_restarts(tmp_path, capsys):
    (tmp_path / "training_checkpoint.json").write_text("{not json")
    history, _ = battery(mock_player()).run_training(str(tmp_path))
    assert len(history) == 6
    assert "could not use checkpoint" in capsys.readouterr().out


def test_unknown_strategy_fails_before_any_llm_call():
    with pytest.raises(ValueError, match="Unknown strategy"):
        battery(mock_player(), strategies=["TitForTat", "Nope"])


def test_dreamer_failure_writes_a_flagged_data_only_playbook(tmp_path):
    player = mock_player()

    def broken(prompt, temperature, kind="dreamer"):
        raise RuntimeError("no completion")

    player.complete_text = broken
    _, playbook = battery(player).run_training(str(tmp_path))

    assert "automatic data summary" in playbook and "points/round" in playbook
    assert json.loads((tmp_path / "master_strategy_playbook.json").read_text())["fallback"] is True
    assert player.master_playbook == playbook and player.freeze_notes


def test_dreamer_summary_hides_opponent_names_in_blind_tiers(tmp_path):
    blind = mock_player("t4_blind")
    battery(blind).run_training(str(tmp_path / "blind"))
    summary = json.loads((tmp_path / "blind" / "master_strategy_playbook.json").read_text())["training_summary"]
    assert "Opponent type 1" in summary and "Alpha Mimic" not in summary

    named = mock_player("t2_named")
    battery(named).run_training(str(tmp_path / "named"))
    summary = json.loads((tmp_path / "named" / "master_strategy_playbook.json").read_text())["training_summary"]
    assert "Alpha Mimic" in summary


def classical_tournament(seed=3, rounds=5):
    players = [make_opponent(k) for k in ("AlwaysDefect", "TitForTat", "Random", "Detective")]
    return RoundRobinTournament(players, matches_per_pairing=2, rounds_per_match=rounds, seed=seed)


def test_same_seed_gives_identical_tournaments():
    a = classical_tournament().run_tournament()
    b = classical_tournament().run_tournament()
    assert a["all_matches"] == b["all_matches"]
    assert classical_tournament(seed=4).run_tournament()["all_matches"] != a["all_matches"]


def test_tournament_resumes_from_checkpoint_with_identical_results(tmp_path):
    full = classical_tournament().run_tournament(output_dir=str(tmp_path))
    path = tmp_path / "tournament_checkpoint.json"
    data = json.loads(path.read_text())
    assert len(data["matches"]) == 12
    data["matches"] = data["matches"][:5]
    path.write_text(json.dumps(data))

    resumed = classical_tournament().run_tournament(output_dir=str(tmp_path))
    assert resumed["all_matches"] == full["all_matches"]
    assert resumed["leaderboard"] == full["leaderboard"]


def test_tournament_ignores_a_checkpoint_from_a_different_lineup(tmp_path, capsys):
    classical_tournament().run_tournament(output_dir=str(tmp_path))
    other = RoundRobinTournament([make_opponent("Random"), make_opponent("Pavlov")], matches_per_pairing=2,
                                 rounds_per_match=3, seed=3)
    results = other.run_tournament(output_dir=str(tmp_path))
    assert results["total_matches"] == 2
    assert "could not use tournament checkpoint" in capsys.readouterr().out


def test_tournament_with_llm_reports_hypotheses_and_stats():
    llm = mock_player("t3_pool_hint")
    llm.freeze_notes = True
    players = [llm, make_opponent("TitForTat"), make_opponent("AlwaysDefect")]
    results = RoundRobinTournament(players, matches_per_pairing=1, rounds_per_match=3, seed=1).run_tournament()
    llm_matches = [m for m in results["all_matches"] if llm.name in (m["player1"], m["player2"])]
    assert len(llm_matches) == 2 and all(len(m["llm_turns"]) == 3 for m in llm_matches)
    assert results["llm_stats"][llm.name]["turns"] == 6


FAST_CONFIG = """
[experiment]
seed = 5
[training]
matches_per_strategy = 1
rounds_per_match = 4
strategies = ["AlwaysCooperate", "TitForTat"]
[evaluation]
matches_per_pairing = 1
rounds_per_match = 4
participants = ["LLMPlayer", "AlwaysCooperate", "TitForTat"]
"""


def run_main(monkeypatch, tmp_path, *extra):
    config = tmp_path / "config.toml"
    config.write_text(FAST_CONFIG)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(config), "--provider", "mock",
                                      "--output-dir", str(tmp_path / "out"), *extra])
    main_module.main()


def test_main_runs_the_whole_pipeline(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path)
    out = tmp_path / "out"
    for name in ("scorecard_training.md", "scorecard_tournament.json", "tournament_leaderboard.json",
                 "tournament_report.html", "trace.jsonl", "memory.json"):
        assert (out / name).exists(), name
    assert not (out / "tournament_checkpoint.json").exists()


def test_main_tiers_runner_compares_every_tier(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, "--tiers", "all")
    out = tmp_path / "out"
    table = (out / "tier_comparison.md").read_text()
    for tier in LIB.tiers:
        assert f"| {tier} | tournament |" in table
        assert (out / tier / "scorecard_tournament.json").exists()
    assert set(json.loads((out / "tier_comparison.json").read_text())) == set(LIB.tiers)


def test_main_eval_only_reuses_saved_playbook_and_memory(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, "--train-only")
    assert not (tmp_path / "out" / "tournament_leaderboard.json").exists()
    run_main(monkeypatch, tmp_path, "--eval-only")
    out = capsys.readouterr().out
    assert "Loaded playbook" in out and "Loaded memory" in out
    assert (tmp_path / "out" / "scorecard_tournament.json").exists()


def test_main_quick_and_no_llm(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, "--no-llm", "--quick")
    assert (tmp_path / "out" / "tournament_leaderboard.json").exists()
    assert not (tmp_path / "out" / "scorecard_tournament.json").exists()


def test_main_stops_cleanly_when_the_llm_is_unavailable(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(LLMPlayer, "_mock_response", lambda self, model: (_ for _ in ()).throw(ConnectionError("down")))
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, tmp_path)
    assert exit_info.value.code == 2
    assert "RUN STOPPED" in capsys.readouterr().out
    assert (tmp_path / "out" / "training_checkpoint.json").exists()


def test_show_prompt_prints_without_calling_an_llm(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--show-prompt", "t1_informed", "--opponent", "TitForTwoTats"])
    main_module.main()
    out = capsys.readouterr().out
    assert "Zen Alpha Mimic" in out and "Alternate Beta and Alpha" in out and "POST-MATCH REFLECTION" in out

    monkeypatch.setattr(sys, "argv", ["main.py", "--show-prompt", "t4_blind"])
    main_module.main()
    assert "Alternate Beta and Alpha" not in capsys.readouterr().out


def test_unknown_tier_is_rejected_before_running(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown prompt tier"):
        run_main(monkeypatch, tmp_path, "--tier", "t9")
