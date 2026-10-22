from types import SimpleNamespace

import pytest

from src.llm_player import (
    LLMDecision, LLMLongTermReflection, LLMPlayer, LLMUnavailableError,
)
from src.match_runner import play_match
from src.opponents import make_opponent
from src.prompts import PromptLibrary
from src.trace import read_trace
from src.usage_tracker import DailyLimitReachedException

LIB = PromptLibrary.load("prompts.toml")


class StubClient:
    """Replays scripted outcomes: a move name ("Alpha"/"Beta"), or an Exception to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []
        self.calls = 0
        outer = self

        def create_with_completion(model, response_model, messages, **kwargs):
            outer.calls += 1
            outer.prompts.append(messages[0]["content"])
            if response_model is LLMDecision:
                step = outer.script.pop(0) if outer.script else "Alpha"
                if isinstance(step, Exception):
                    raise step
                obj = LLMDecision(decision=step, opponent_hypothesis="Alpha Mimic", reasoning="r",
                                  new_short_term_memory="short note")
            else:
                obj = LLMLongTermReflection(identified_archetype="a", reasoning="r",
                                            new_long_term_memory="long note", importance=7)
            completion = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=obj.model_dump_json()))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
            )
            return obj, completion

        self.chat = SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion))


def make_player(script, tier="t4_blind", **kw):
    player = LLMPlayer(provider="ollama", model_name="stub", tier=tier, prompt_library=LIB,
                       retry_sleep=0, max_retries=kw.pop("max_retries", 3), **kw)
    player.client = StubClient(script)
    player.show_panels = False
    return player


def run(player, opponent_key="AlwaysCooperate", turns=3):
    opponent = make_opponent(opponent_key)
    player.begin_match(opponent.name, turns, float(turns))
    result = play_match(player, opponent, turns=turns, seed=1)
    return opponent, result


def test_llm_player_plays_axelrod_match():
    player = make_player(["Alpha", "Beta", "Beta"])
    _, result = run(player)
    assert [m[0] for m in result.moves] == ["C", "D", "D"]
    assert (result.score1, result.score2) == (3 + 5 + 5, 3)


def test_decision_accepts_classic_labels_and_any_case():
    assert LLMDecision(decision="cooperate", opponent_hypothesis="", reasoning="").decision == "Alpha"
    assert LLMDecision(decision=" BETA ", opponent_hypothesis="", reasoning="").decision == "Beta"


def test_prompt_reports_real_opponent_score():
    player = make_player(["Alpha", "Alpha"])
    run(player, turns=2)
    prompts = player.client.prompts
    assert "Other player's points this match: 0" in prompts[0]
    assert "Your points this match: 3" in prompts[1]
    assert "Other player's points this match: 3" in prompts[1]


def test_notes_survive_reset_and_match_state_does_not():
    player = make_player(["Alpha"] * 6)
    opponent, _ = run(player, "TitForTat", turns=2)
    player.finish_match(opponent)

    assert "long note" in player.lessons
    assert player.short_lasting_notes == []
    assert player.game_history == []
    assert any(e.kind == "observation" for e in player.memory.entries)

    run(player, "TitForTat", turns=2)
    assert "long note" in player.lessons


def test_reflection_links_to_its_observation():
    player = make_player(["Alpha"] * 4)
    opponent, _ = run(player, turns=2)
    player.finish_match(opponent)
    reflection = player.memory.reflections()[0]
    observation = next(e for e in player.memory.entries if e.kind == "observation")
    assert reflection.evidence == [observation.id] and reflection.importance == 7


def test_blind_tier_observation_does_not_name_the_opponent():
    blind = make_player(["Alpha"] * 2, tier="t4_blind")
    opponent, _ = run(blind, "TitForTat", turns=2)
    blind.finish_match(opponent)
    assert "Alpha Mimic" not in blind.memory.entries[0].text

    informed = make_player(["Alpha"] * 2, tier="t2_named")
    opponent, _ = run(informed, "TitForTat", turns=2)
    informed.finish_match(opponent)
    assert "Alpha Mimic" in informed.memory.entries[0].text


def test_frozen_player_reads_memory_but_learns_nothing():
    player = make_player(["Alpha"] * 4)
    player.memory.add("reflection", "old lesson", importance=9)
    player.set_master_playbook("PLAYBOOK TEXT")
    opponent, _ = run(player, turns=2)
    player.finish_match(opponent)

    assert len(player.memory.entries) == 1
    assert "old lesson" in player.client.prompts[0] and "PLAYBOOK TEXT" in player.client.prompts[0]
    assert player.short_lasting_notes == []


def test_failed_call_is_retried_then_succeeds():
    player = make_player([RuntimeError("timeout"), RuntimeError("timeout"), "Beta"])
    _, result = run(player, turns=1)
    assert result.moves[0][0] == "D"
    assert player.stats["retries"] == 2 and player.stats["fallbacks"] == 0


def test_exhausted_retries_use_the_neutral_fallback_and_are_counted():
    player = make_player([RuntimeError("down")] * 3 + ["Alpha"] * 5, max_consecutive_fallbacks=5)
    _, result = run(player, "AlwaysDefect", turns=3)
    # Round 1 falls back to Alpha (not Beta); later rounds are answered normally.
    assert result.moves[0][0] == "C"
    assert player.stats["fallbacks"] == 1
    assert player.turn_log[0]["fallback"] is True


def test_fallback_repeats_the_opponents_last_move():
    player = make_player(["Beta"] + [RuntimeError("down")] * 3, max_consecutive_fallbacks=9)
    _, result = run(player, "AlwaysCooperate", turns=2)
    assert [m[0] for m in result.moves] == ["D", "C"]  # opponent played Alpha, so fallback is Alpha


def test_repeated_failures_abort_the_run_loudly():
    player = make_player([RuntimeError("down")] * 100, max_consecutive_fallbacks=3, max_retries=1)
    with pytest.raises(LLMUnavailableError, match="3 consecutive"):
        run(player, turns=10)
    assert player.stats["fallbacks"] == 3


def test_rate_limit_aborts_after_three_429s(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    player = make_player([RuntimeError("Error code: 429 quota")] * 5)
    with pytest.raises(DailyLimitReachedException):
        run(player, turns=1)
    assert player.stats["rate_limited"] == 3


def test_trace_records_prompt_response_errors_and_fallbacks(tmp_path):
    player = make_player([RuntimeError("boom"), "Beta", RuntimeError("x"), RuntimeError("x"), RuntimeError("x")],
                         max_consecutive_fallbacks=9)
    player.configure_output(str(tmp_path))
    run(player, turns=2)

    records = list(read_trace(str(tmp_path / "trace.jsonl")))
    turns = [r for r in records if r["kind"] == "turn"]
    assert turns[0]["attempts"] == 2 and "boom" in turns[0]["errors"][0]
    assert turns[0]["parsed"]["decision"] == "Beta" and turns[0]["tokens"] == [11, 5]
    assert "Round: 1" in turns[0]["prompt"] and turns[0]["opponent"] == "AlwaysCooperate"
    assert turns[1].get("failed") is True
    assert [r["kind"] for r in records].count("fallback") == 1


def test_token_usage_is_recorded(tmp_path):
    player = make_player(["Alpha", "Alpha"])
    player.usage_tracker.enabled = True
    player.configure_output(str(tmp_path))
    run(player, turns=2)
    assert player.usage_tracker.prompt_tokens == 22 and player.usage_tracker.completion_tokens == 10


def test_memory_roundtrip_on_disk(tmp_path):
    player = make_player(["Alpha"] * 2)
    opponent, _ = run(player, turns=2)
    player.finish_match(opponent)
    path = str(tmp_path / "memory.json")
    player.save_memory(path)

    fresh = make_player([])
    assert fresh.load_memory(path) is True
    assert fresh.lessons == player.lessons and fresh.memory.step == player.memory.step
    assert fresh.load_memory(str(tmp_path / "missing.json")) is False


def test_mock_provider_needs_no_network():
    player = LLMPlayer(provider="mock", tier="t4_blind", prompt_library=LIB)
    player.show_panels = False
    opponent, result = run(player, "TitForTat", turns=4)
    assert [m[0] for m in result.moves] == ["C"] * 4
