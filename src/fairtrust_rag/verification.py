"""Atomic claim extraction and evidence verification."""

import math
import re
from typing import Any, List, Optional, Sequence, Set

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


class NLIEvidenceVerifier:
    """Three-way claim verification using a Natural Language Inference model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        minimum_confidence: float = 0.50,
        cross_encoder: Optional[Any] = None,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    'NLI verification requires: pip install -e ".[semantic]"'
                ) from exc
            cross_encoder = CrossEncoder(model_name)
        self.model_name = model_name
        self.minimum_confidence = minimum_confidence
        self.model = cross_encoder

    @staticmethod
    def _probabilities(logits: Sequence[float]) -> List[float]:
        values = [float(value) for value in logits]
        peak = max(values)
        exponentials = [math.exp(value - peak) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def verify(
        self, claims: Sequence[str], evidence: Sequence[SearchResult]
    ) -> List[ClaimVerification]:
        output = []
        for claim in claims:
            if not evidence:
                output.append(
                    ClaimVerification(
                        claim=claim,
                        status="insufficient_evidence",
                        support_score=0.0,
                    )
                )
                continue

            premises = []
            for result in evidence:
                sentences = [
                    sentence.strip()
                    for sentence in re.split(
                        r"(?<=[.!?])\s+", result.chunk.text
                    )
                    if sentence.strip()
                ]
                premises.extend(
                    (sentence, result.chunk.chunk_id) for sentence in sentences
                )
            pairs = [(sentence, claim) for sentence, _ in premises]
            logits = self.model.predict(pairs)
            candidates = []
            for (_, chunk_id), scores in zip(premises, logits):
                probabilities = self._probabilities(scores)
                candidates.append((probabilities, chunk_id))

            support_score, support_chunk = max(
                (probabilities[1], chunk_id)
                for probabilities, chunk_id in candidates
            )
            contradiction_score, contradiction_chunk = max(
                (probabilities[0], chunk_id)
                for probabilities, chunk_id in candidates
            )
            if support_score >= self.minimum_confidence:
                status = "supported"
                chunk_id = support_chunk
            elif contradiction_score >= self.minimum_confidence:
                status = "contradicted"
                chunk_id = contradiction_chunk
            else:
                status = "insufficient_evidence"
                chunk_id = None
            output.append(
                ClaimVerification(
                    claim=claim,
                    status=status,
                    support_score=support_score,
                    evidence_chunk_id=chunk_id,
                )
            )
        return output
