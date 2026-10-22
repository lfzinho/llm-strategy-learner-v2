from types import SimpleNamespace

from src.llm_player import LLMDecision, LLMLongTermReflection, LLMPlayer
from src.match_runner import play_match
from src.opponents import make_opponent


def make_player(moves):
    """LLMPlayer whose client replays `moves` ("COOPERATE"/"DEFECT") and records prompts."""
    player = LLMPlayer(provider="ollama", model_name="stub")
    player.prompts = []
    queue = list(moves)

    def create(model, response_model, messages, **kwargs):
        player.prompts.append(messages[0]["content"])
        if response_model is LLMDecision:
            return LLMDecision(
                decision=queue.pop(0), opponent_hypothesis="h", reasoning="r",
                new_short_term_memory="short note",
            )
        return LLMLongTermReflection(
            identified_archetype="a", reasoning="r", new_long_term_memory="long note",
        )

    player.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return player


def test_llm_player_plays_axelrod_match():
    player = make_player(["COOPERATE", "DEFECT", "DEFECT"])
    opponent = make_opponent("AlwaysCooperate")

    result = play_match(player, opponent, turns=3, seed=1)

    assert [m[0] for m in result.moves] == ["C", "D", "D"]
    assert (result.score1, result.score2) == (3 + 5 + 5, 3 + 0 + 0)


def test_prompt_reports_real_opponent_score():
    player = make_player(["COOPERATE", "COOPERATE"])
    play_match(player, make_opponent("AlwaysCooperate"), turns=2, seed=1)
    # Round 1 prompt: nobody has scored. Round 2 prompt: both earned 3 from mutual cooperation.
    assert "Opponent Current Match Score: 0" in player.prompts[0]
    assert "Your Current Match Score: 3" in player.prompts[1]
    assert "Opponent Current Match Score: 3" in player.prompts[1]


def test_notes_survive_reset_and_match_state_does_not():
    player = make_player(["COOPERATE"] * 4)
    opponent = make_opponent("TitForTat")
    play_match(player, opponent, turns=2, seed=1)
    player.finish_match(opponent)

    assert "long note" in player.long_lasting_notes
    assert player.short_lasting_notes == []
    assert player.game_history == []

    play_match(player, opponent, turns=2, seed=1)
    assert "long note" in player.long_lasting_notes
    assert len(player.history) == 2
