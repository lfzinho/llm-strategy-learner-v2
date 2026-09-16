import json
import os
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    name: str = "v2_championship_tournament"
    seed: int = 42
    output_dir: str = "output"


class LLMRateLimitConfig(BaseModel):
    max_rpm: int = 30
    max_rpd: int = 1000


class LLMConfig(BaseModel):
    provider: Literal["openrouter", "gemini", "ollama"] = "openrouter"
    model_name: Optional[str] = None
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


PromptModeType = Literal["explicit_strategies", "generic_archetypes", "blind_optimization"]


class PromptsConfig(BaseModel):
    prompt_mode: PromptModeType = "explicit_strategies"
    modes_info: Optional[Dict[str, str]] = None


class DreamerConfig(BaseModel):
    enabled: bool = True
    model_name: Optional[str] = None
    temperature: float = 0.2


class TrainingConfig(BaseModel):
    enabled: bool = True
    matches_per_strategy: int = 5
    rounds_per_match: int = 5
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
    rounds_per_match: int = 5
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


def load_config(config_path: str = "config.json") -> MasterConfig:
    """Loads and validates configuration from a JSON file, falling back to defaults if file is missing."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return MasterConfig.model_validate(raw_data)
    else:
        print(f"Warning: Configuration file '{config_path}' not found. Using default configuration.")
        return MasterConfig()
