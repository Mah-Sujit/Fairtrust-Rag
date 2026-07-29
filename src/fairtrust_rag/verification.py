"""Atomic claim extraction and evidence-verification baseline."""

import re
from typing import List, Sequence, Set

from .models import ClaimVerification, SearchResult


WORDS = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
}


def _terms(text: str) -> Set[str]:
    return {word for word in WORDS.findall(text.lower()) if word not in STOP_WORDS}


def extract_claims(answer: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|;\s+", answer.strip())
        if item.strip()
    ]


class LexicalEvidenceVerifier:
    """Transparent stub; replace with a trained NLI verifier for experiments."""

    def __init__(self, minimum_support: float = 0.18) -> None:
        self.minimum_support = minimum_support

    def verify(
        self, claims: Sequence[str], evidence: Sequence[SearchResult]
    ) -> List[ClaimVerification]:
        output = []
        for claim in claims:
            claim_terms = _terms(claim)
            candidates = []
            for result in evidence:
                overlap = len(claim_terms & _terms(result.chunk.text))
                score = overlap / max(1, len(claim_terms))
                candidates.append((score, result.chunk.chunk_id))
            best_score, best_chunk = max(candidates, default=(0.0, None))
            status = "supported" if best_score >= self.minimum_support else "insufficient_evidence"
            output.append(
                ClaimVerification(
                    claim=claim,
                    status=status,
                    support_score=best_score,
                    evidence_chunk_id=best_chunk if status == "supported" else None,
                )
            )
        return output

