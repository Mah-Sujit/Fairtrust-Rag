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
    verification_provider: str = "lexical"
    verification_model: str = "cross-encoder/nli-deberta-v3-small"
    minimum_nli_confidence: float = 0.50
    conflict_detection_enabled: bool = False
    minimum_conflict_confidence: float = 0.80
    minimum_conflict_overlap: float = 0.20
    maximum_conflict_pairs: int = 100
    retrieval_retry_enabled: bool = False
    retry_top_k: int = 8
    citation_verification_enabled: bool = True
    generation_provider: str = "extractive"
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 120
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
        if self.verification_provider not in {"lexical", "nli"}:
            raise ValueError("verification_provider must be 'lexical' or 'nli'")
        if not 0 <= self.minimum_nli_confidence <= 1:
            raise ValueError("minimum_nli_confidence must be between 0 and 1")
        if not 0 <= self.minimum_conflict_confidence <= 1:
            raise ValueError("minimum_conflict_confidence must be between 0 and 1")
        if not 0 <= self.minimum_conflict_overlap <= 1:
            raise ValueError("minimum_conflict_overlap must be between 0 and 1")
        if self.maximum_conflict_pairs <= 0:
            raise ValueError("maximum_conflict_pairs must be positive")
        if self.retry_top_k < self.top_k:
            raise ValueError("retry_top_k must be at least top_k")
        if self.generation_provider not in {"extractive", "ollama"}:
            raise ValueError("generation_provider must be 'extractive' or 'ollama'")
        if self.ollama_timeout_seconds <= 0:
            raise ValueError("ollama_timeout_seconds must be positive")
        if not 0 <= self.maximum_answer_risk <= 1:
            raise ValueError("maximum_answer_risk must be between 0 and 1")
        if abs(sum(self.risk_weights.values()) - 1.0) > 1e-6:
            raise ValueError("risk_weights must sum to 1")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Settings":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))
