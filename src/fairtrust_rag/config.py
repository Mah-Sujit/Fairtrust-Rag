"""Configuration loading and validation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 500
    chunk_overlap: int = 80
    embedding_provider: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    top_k: int = 3
    minimum_retrieval_score: float = 0.08
    minimum_claim_support: float = 0.18
    maximum_answer_risk: float = 0.55
    risk_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "retrieval": 0.35,
            "unsupported_claims": 0.55,
            "citation": 0.10,
        }
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")
        if self.embedding_dimensions <= 0 or self.top_k <= 0:
            raise ValueError("embedding_dimensions and top_k must be positive")
        if self.embedding_provider not in {"hashing", "sentence_transformers"}:
            raise ValueError(
                "embedding_provider must be 'hashing' or 'sentence_transformers'"
            )
        if not 0 <= self.maximum_answer_risk <= 1:
            raise ValueError("maximum_answer_risk must be between 0 and 1")
        if abs(sum(self.risk_weights.values()) - 1.0) > 1e-6:
            raise ValueError("risk_weights must sum to 1")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Settings":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))
