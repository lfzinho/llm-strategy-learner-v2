import os
import tomllib
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strict(BaseModel):
    """Rejects unknown keys so a typo in config.toml fails loudly instead of being ignored."""
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(Strict):
    name: str = "v2_championship_tournament"
    seed: int = 42
    output_dir: str = "output"


class LLMRateLimitConfig(Strict):
    max_rpm: int = 30
    max_rpd: int = 1000


class LLMConfig(Strict):
    provider: Literal["openrouter", "gemini", "ollama", "mock"] = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter")
    )
    model_name: Optional[str] = None  # resolved at runtime from provider + env vars
    temperature: float = 0.2
    max_retries: int = 3  # attempts per call before falling back
    max_consecutive_fallbacks: int = 5  # abort the run if the LLM keeps failing
    rate_limit: LLMRateLimitConfig = Field(default_factory=LLMRateLimitConfig)


class EnvironmentConfig(Strict):
    error_rate: float = 0.0  # probability that a move is flipped (noise)

    @field_validator("error_rate")
    @classmethod
    def validate_error_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("error_rate must be between 0.0 and 1.0")
        return v


class NormalRoundsConfig(Strict):
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


class GeometricRoundsConfig(Strict):
    """After every round the match ends with probability `prob_end` (unknown horizon)."""
    prob_end: float
    max_rounds: int = 200

    @field_validator("prob_end")
    @classmethod
    def validate_prob_end(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("prob_end must be between 0 and 1 (exclusive)")
        return v


RoundsPerMatchConfig = Union[int, NormalRoundsConfig, GeometricRoundsConfig]


def describe_rounds_per_match(rounds: RoundsPerMatchConfig) -> str:
    if isinstance(rounds, NormalRoundsConfig):
        return f"Normal({rounds.average:g}, {rounds.std_dev:g})"
    if isinstance(rounds, GeometricRoundsConfig):
        return f"Geometric(prob_end={rounds.prob_end:g})"
    return str(rounds)


class PromptsConfig(Strict):
    tier: str = "t3_pool_hint"  # a tier defined in prompts_file
    prompts_file: str = "prompts.toml"
    memory_top_k: int = 8  # how many remembered entries go into each round's prompt


class DreamerConfig(Strict):
    enabled: bool = True
    temperature: float = 0.2


class TrainingConfig(Strict):
    matches_per_strategy: int = 5
    rounds_per_match: RoundsPerMatchConfig = Field(default_factory=NormalRoundsConfig)
    strategies: List[str] = Field(
        default_factory=lambda: [
            "AlwaysCooperate", "AlwaysDefect", "TitForTat", "SuspiciousTitForTat",
            "TitForTwoTats", "TatForTit", "Random", "Detective",
        ]
    )
    dreamer: DreamerConfig = Field(default_factory=DreamerConfig)


class TournamentScoringConfig(Strict):
    win_match_points: float = 1.0
    draw_match_points: float = 0.5
    loss_match_points: float = 0.0


class EvaluationConfig(Strict):
    enabled: bool = True
    matches_per_pairing: int = 5
    rounds_per_match: RoundsPerMatchConfig = Field(default_factory=NormalRoundsConfig)
    participants: List[str] = Field(
        default_factory=lambda: [
            "LLMPlayer", "AlwaysCooperate", "AlwaysDefect", "TitForTat", "SuspiciousTitForTat",
            "TitForTwoTats", "TatForTit", "Random", "Detective",
        ]
    )
    scoring: TournamentScoringConfig = Field(default_factory=TournamentScoringConfig)


class ReportingConfig(Strict):
    export_html: bool = True
    export_markdown: bool = True
    export_json: bool = True


class MasterConfig(Strict):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def load_config(config_path: str = "config.toml") -> MasterConfig:
    """Loads config.toml, validating with Pydantic (unknown keys are errors)."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file '{config_path}' not found. Using default MasterConfig.")
        return MasterConfig()
    with open(config_path, "rb") as f:
        return MasterConfig.model_validate(tomllib.load(f))
