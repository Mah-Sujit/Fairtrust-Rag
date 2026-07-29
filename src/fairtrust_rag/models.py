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
class EvidenceConflict:
    left_chunk_id: str
    right_chunk_id: str
    left_text: str
    right_text: str
    confidence: float


@dataclass(frozen=True)
class CitationVerification:
    claim: str
    citation_id: Optional[str]
    status: str
    support_score: float


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
    conflict_score: float
    conflicts: List[EvidenceConflict]
    retrieval_attempts: int = 1
    citation_verifications: List[CitationVerification] = field(default_factory=list)
    citation_precision: float = 0.0
    citation_coverage: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "decision": self.decision,
            "answer": self.answer,
            "risk_score": round(self.risk_score, 4),
            "reason": self.reason,
            "citations": self.citations,
            "retrieval_attempts": self.retrieval_attempts,
            "citation_precision": round(self.citation_precision, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "citation_verifications": [
                {
                    "claim": item.claim,
                    "citation_id": item.citation_id,
                    "status": item.status,
                    "support_score": round(item.support_score, 4),
                }
                for item in self.citation_verifications
            ],
            "conflict_score": round(self.conflict_score, 4),
            "conflicts": [
                {
                    "left_chunk_id": item.left_chunk_id,
                    "right_chunk_id": item.right_chunk_id,
                    "left_text": item.left_text,
                    "right_text": item.right_text,
                    "confidence": round(item.confidence, 4),
                }
                for item in self.conflicts
            ],
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
