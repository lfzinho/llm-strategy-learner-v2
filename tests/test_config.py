import os
import tempfile
import json
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


def test_load_config_from_file():
    sample_data = {
        "prompts": {"prompt_mode": "generic_archetypes"},
        "environment": {"error_rate": 0.0},
        "training": {"matches_per_strategy": 3},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        loaded = load_config(temp_path)
        assert loaded.prompts.prompt_mode == "generic_archetypes"
        assert loaded.training.matches_per_strategy == 3
        assert loaded.environment.error_rate == 0.0
    finally:
        os.remove(temp_path)
