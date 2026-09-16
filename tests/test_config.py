import os
import tempfile
import pytest
from src.config import load_config, MasterConfig, EnvironmentConfig


def test_default_config():
    config = MasterConfig()
    assert config.environment.error_rate == 0.0
    assert config.prompts.prompt_mode == "explicit_strategies"
    assert config.training.matches_per_strategy == 5
    assert config.training.rounds_per_match == 5
    assert config.evaluation.matches_per_pairing == 5


def test_invalid_error_rate():
    with pytest.raises(ValueError):
        EnvironmentConfig(error_rate=1.5)
    with pytest.raises(ValueError):
        EnvironmentConfig(error_rate=-0.1)


def test_load_root_toml_config():
    assert os.path.exists("config.toml")
    config = load_config("config.toml")
    assert config.environment.error_rate == 0.0
    assert config.prompts.prompt_mode == "explicit_strategies"
    assert "explicit_strategies" in config.prompts.system_prompts
    assert "generic_archetypes" in config.prompts.system_prompts
    assert "blind_optimization" in config.prompts.system_prompts
    assert "prompt_template" in config.prompts.reflection
    assert "prompt_template" in config.prompts.dreamer


def test_load_custom_toml_file():
    sample_toml = """
[experiment]
name = "custom_test"

[prompts]
prompt_mode = "generic_archetypes"

[prompts.system_prompts]
generic_archetypes = "Custom generic prompt template"

[training]
matches_per_strategy = 3
"""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
        f.write(sample_toml)
        temp_path = f.name

    try:
        loaded = load_config(temp_path)
        assert loaded.experiment.name == "custom_test"
        assert loaded.prompts.prompt_mode == "generic_archetypes"
        assert loaded.prompts.system_prompts.get("generic_archetypes") == "Custom generic prompt template"
        assert loaded.training.matches_per_strategy == 3
    finally:
        os.remove(temp_path)
