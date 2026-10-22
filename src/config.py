import os
import json
from typing import Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator

# Python 3.11+ built-in tomllib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


class ExperimentConfig(BaseModel):
    name: str = "v2_championship_tournament"
    seed: int = 42
    output_dir: str = "output"


class LLMRateLimitConfig(BaseModel):
    max_rpm: int = 30
    max_rpd: int = 1000


class LLMConfig(BaseModel):
    provider: Literal["openrouter", "gemini", "ollama"] = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter")
    )
    model_name: Optional[str] = Field(
        default=None  # resolved at runtime by LLMPlayer based on provider + env vars
    )
    temperature: float = 0.2
    max_retries: int = 3
    rate_limit: LLMRateLimitConfig = Field(default_factory=LLMRateLimitConfig)



class EnvironmentConfig(BaseModel):
    error_rate: float = 0.0
    payoff_matrix: Dict[str, Tuple[int, int]] = Field(
        default_factory=lambda: {
            "cooperate_cooperate": (3, 3),
            "defect_cooperate": (5, 0),
            "cooperate_defect": (0, 5),
            "defect_defect": (1, 1),
        }
    )

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("error_rate must be between 0.0 and 1.0")
        return v


PromptModeType = Literal["explicit_strategies", "generic_archetypes", "blind_optimization", "revealed_strategy", "custom"]


class NormalRoundsConfig(BaseModel):
    average: float = 10.0
    std_dev: float = 1.0

    @field_validator("average")
    @classmethod
    def validate_average(cls, v: float) -> float:
        if v < 1:
            raise ValueError("rounds_per_match average must be at least 1")
        return v

    @field_validator("std_dev")
    @classmethod
    def validate_std_dev(cls, v: float) -> float:
        if v < 0:
            raise ValueError("rounds_per_match std_dev must be non-negative")
        return v


RoundsPerMatchConfig = Union[int, NormalRoundsConfig]


def describe_rounds_per_match(rounds_per_match: RoundsPerMatchConfig) -> str:
    if isinstance(rounds_per_match, NormalRoundsConfig):
        return f"Normal({rounds_per_match.average:g}, {rounds_per_match.std_dev:g})"
    return str(rounds_per_match)


class PromptsConfig(BaseModel):
    prompt_mode: PromptModeType = "explicit_strategies"
    system_prompts: Dict[str, str] = Field(default_factory=dict)
    reflection: Dict[str, str] = Field(default_factory=dict)
    dreamer: Dict[str, str] = Field(default_factory=dict)


class DreamerConfig(BaseModel):
    enabled: bool = True
    model_name: Optional[str] = None
    temperature: float = 0.2


class TrainingConfig(BaseModel):
    enabled: bool = True
    matches_per_strategy: int = 5
    rounds_per_match: RoundsPerMatchConfig = Field(default_factory=NormalRoundsConfig)
    strategies: List[str] = Field(
        default_factory=lambda: [
            "AlwaysCooperate",
            "AlwaysDefect",
            "TitForTat",
            "SuspiciousTitForTat",
            "TitForTwoTats",
            "TatForTit",
            "Random",
            "Detective",
        ]
    )
    dreamer: DreamerConfig = Field(default_factory=DreamerConfig)


class TournamentScoringConfig(BaseModel):
    win_match_points: float = 1.0
    draw_match_points: float = 0.5
    loss_match_points: float = 0.0


class EvaluationConfig(BaseModel):
    enabled: bool = True
    matches_per_pairing: int = 5
    rounds_per_match: RoundsPerMatchConfig = Field(default_factory=NormalRoundsConfig)
    participants: List[str] = Field(
        default_factory=lambda: [
            "LLMPlayer",
            "AlwaysCooperate",
            "AlwaysDefect",
            "TitForTat",
            "SuspiciousTitForTat",
            "TitForTwoTats",
            "TatForTit",
            "Random",
            "Detective",
        ]
    )
    scoring: TournamentScoringConfig = Field(default_factory=TournamentScoringConfig)


class ReportingConfig(BaseModel):
    export_html: bool = True
    export_markdown: bool = True
    export_json: bool = True
    terminal_style: Literal["rich", "simple"] = "rich"


class MasterConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def load_config(config_path: Optional[str] = None) -> MasterConfig:
    """Loads configuration from config.toml (or config.json), validating with Pydantic."""
    if not config_path:
        # Default priority: config.toml > config.json
        if os.path.exists("config.toml"):
            config_path = "config.toml"
        elif os.path.exists("config.json"):
            config_path = "config.json"
        else:
            config_path = "config.toml"

    if os.path.exists(config_path):
        if config_path.endswith(".toml"):
            with open(config_path, "rb") as f:
                raw_data = tomllib.load(f)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        # Convert payoff matrix values from list to tuple if needed
        if "environment" in raw_data and "payoff_matrix" in raw_data["environment"]:
            pm = raw_data["environment"]["payoff_matrix"]
            for k, v in pm.items():
                if isinstance(v, list):
                    pm[k] = tuple(v)

        return MasterConfig.model_validate(raw_data)
    else:
        print(f"Warning: Configuration file '{config_path}' not found. Using default MasterConfig.")
        return MasterConfig()
