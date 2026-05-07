from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelTier = Literal["small", "medium", "heavy"]


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    base_url: str
    env_key_var: str = "TOGETHER_API_KEY"
    max_retries: int = 3
    retry_backoff_base: float = 2.0


class ModelTierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    context_length: int | None = None
    supports_structured_output: bool | None = None
    input_cost_per_m: float | None = None
    output_cost_per_m: float | None = None
    temperature: float
    max_tokens: int
    notes: str | None = None


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    small: ModelTierConfig
    medium: ModelTierConfig
    heavy: ModelTierConfig


class PdfPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser: str = "pymupdf"
    extract_captions: bool = False
    extract_equations: bool = False
    min_section_chars: int
    max_section_chars: int


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf: PdfPipelineConfig
    extraction: dict[str, Any] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dsn_env: str = "PAPER_DECOMPOSER_DSN"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


class PaperDecomposerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    api: ApiConfig
    models: ModelsConfig
    pipeline: PipelineConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


class RuntimeModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    max_tokens: int


class RuntimePipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser: str
    extract_captions: bool
    extract_equations: bool
    min_section_chars: int
    max_section_chars: int
    extraction: dict[str, Any] = Field(default_factory=dict)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str
    api_key: str
    model_tiers: dict[ModelTier, RuntimeModelConfig]
    pipeline: RuntimePipelineConfig
    raw: PaperDecomposerConfig

    def tier(self, tier: ModelTier) -> RuntimeModelConfig:
        return self.model_tiers[tier]


class RhetoricalRole(str, Enum):
    abstract = "abstract"
    introduction = "introduction"
    background = "background"
    method = "method"
    theory = "theory"
    evaluation = "evaluation"
    discussion = "discussion"
    appendix = "appendix"
    other = "other"


class EvidenceArtifact(BaseModel):
    artifact_type: str
    artifact_id: str
    caption: str
    source_page: int


class Section(BaseModel):
    section_number: str | None = None
    title: str
    role: RhetoricalRole
    body_text: str
    artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    char_count: int


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None


class PaperDocument(BaseModel):
    metadata: PaperMetadata
    sections: list[Section] = Field(default_factory=list)
    all_artifacts: list[EvidenceArtifact] = Field(default_factory=list)


__all__ = [
    "ApiConfig",
    "AppSettings",
    "DatabaseConfig",
    "EvidenceArtifact",
    "ModelTier",
    "ModelTierConfig",
    "ModelsConfig",
    "PaperDecomposerConfig",
    "PaperDocument",
    "PaperMetadata",
    "PdfPipelineConfig",
    "PipelineConfig",
    "RhetoricalRole",
    "RuntimeModelConfig",
    "RuntimePipelineConfig",
    "Section",
]
