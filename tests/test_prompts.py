from src.prompts import build_system_prompt, build_turn_prompt


def test_revealed_strategy_prompt_includes_identity_and_match_length():
    prompt = build_system_prompt(
        mode="revealed_strategy",
        opponent_strategy="Always Cooperate",
    )

    assert "Always Cooperate" in prompt
    assert "Normal(10, 1)" in prompt


def test_all_builtin_turn_prompts_include_match_length():
    for mode in ("explicit_strategies", "generic_archetypes", "blind_optimization", "revealed_strategy"):
        prompt = build_turn_prompt(
            game_history=[],
            my_score=0,
            opp_score=0,
            long_lasting_notes=[],
            short_lasting_notes=[],
            mode=mode,
            opponent_strategy="Always Defect",
        )
        assert "Normal(10, 1)" in prompt