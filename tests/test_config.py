import os
import tempfile

import pytest
from pydantic import ValidationError

from src.config import (
    EnvironmentConfig, GeometricRoundsConfig, MasterConfig, NormalRoundsConfig, load_config,
)
from src.match_runner import RoundSampler
from src.prompts import PromptLibrary


def test_default_config():
    config = MasterConfig()
    assert config.environment.error_rate == 0.0
    assert config.prompts.tier == "t3_pool_hint"
    assert config.training.matches_per_strategy == 5
    assert config.training.rounds_per_match == NormalRoundsConfig(average=10, std_dev=1)
    assert config.evaluation.matches_per_pairing == 5


def test_provider_and_model_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

    config = MasterConfig()
    from src.llm_player import LLMPlayer

    player = LLMPlayer(provider=config.llm.provider)

    assert config.llm.provider == "ollama"
    assert player.model_name == "qwen2.5:7b"


def test_invalid_error_rate():
    with pytest.raises(ValueError):
        EnvironmentConfig(error_rate=1.5)
    with pytest.raises(ValueError):
        EnvironmentConfig(error_rate=-0.1)


def test_load_root_config_and_its_tier_exists_in_prompts_file():
    config = load_config("config.toml")
    lib = PromptLibrary.load(config.prompts.prompts_file)
    assert config.prompts.tier in lib.tiers


def test_unknown_config_key_is_an_error():
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
        f.write("[training]\nmatches_per_strategi = 3\n")
        path = f.name
    try:
        with pytest.raises(ValidationError):
            load_config(path)
    finally:
        os.remove(path)


def test_load_custom_toml_file():
    sample = """
[experiment]
name = "custom_test"

[prompts]
tier = "t1_informed"

[training]
matches_per_strategy = 3
rounds_per_match = { prob_end = 0.1 }
"""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
        f.write(sample)
        path = f.name
    try:
        loaded = load_config(path)
        assert loaded.experiment.name == "custom_test"
        assert loaded.prompts.tier == "t1_informed"
        assert loaded.training.matches_per_strategy == 3
        assert loaded.training.rounds_per_match == GeometricRoundsConfig(prob_end=0.1)
    finally:
        os.remove(path)


def test_round_samplers():
    import random
    assert RoundSampler(7).sample() == 7
    normal = RoundSampler(NormalRoundsConfig(average=10, std_dev=1), random.Random(1))
    assert all(5 <= normal.sample() <= 15 for _ in range(200))

    geo = RoundSampler(GeometricRoundsConfig(prob_end=0.1, max_rounds=50), random.Random(1))
    samples = [geo.sample() for _ in range(2000)]
    assert min(samples) >= 1 and max(samples) <= 50
    assert 7 < sum(samples) / len(samples) < 12  # mean is about 1 / prob_end
    assert geo.average == 10.0


def test_same_seed_gives_same_match_lengths():
    import random
    a = RoundSampler(NormalRoundsConfig(), random.Random(5))
    b = RoundSampler(NormalRoundsConfig(), random.Random(5))
    assert [a.sample() for _ in range(20)] == [b.sample() for _ in range(20)]
