"""Shared domain models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: List[str]


@dataclass(frozen=True)
class ClaimVerification:
    claim: str
    status: str
    support_score: float
    evidence_chunk_id: Optional[str] = None


@dataclass(frozen=True)
class TrustReport:
    question: str
    decision: str
    answer: Optional[str]
    risk_score: float
    reason: str
    citations: List[str]
    retrieved: List[SearchResult]
    claims: List[ClaimVerification]

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "decision": self.decision,
            "answer": self.answer,
            "risk_score": round(self.risk_score, 4),
            "reason": self.reason,
            "citations": self.citations,
            "retrieved": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "source": item.chunk.source,
                    "score": round(item.score, 4),
                    "text": item.chunk.text,
                }
                for item in self.retrieved
            ],
            "claims": [
                {
                    "claim": item.claim,
                    "status": item.status,
                    "support_score": round(item.support_score, 4),
                    "evidence_chunk_id": item.evidence_chunk_id,
                }
                for item in self.claims
            ],
        }

