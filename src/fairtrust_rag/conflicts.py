"""Passage-to-passage evidence conflict detection."""

import math
import re
from typing import Any, List, Optional, Sequence, Tuple

from .models import EvidenceConflict, SearchResult

WORDS = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
}


class NoOpConflictDetector:
    def detect(
        self, evidence: Sequence[SearchResult]
    ) -> Tuple[float, List[EvidenceConflict]]:
        return 0.0, []


class NLIConflictDetector:
    """Detect contradictory sentence pairs from different retrieved chunks."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        minimum_confidence: float = 0.80,
        minimum_lexical_overlap: float = 0.20,
        max_pairs: int = 100,
        cross_encoder: Optional[Any] = None,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if not 0 <= minimum_lexical_overlap <= 1:
            raise ValueError("minimum_lexical_overlap must be between 0 and 1")
        if cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    'NLI conflict detection requires: pip install -e ".[semantic]"'
                ) from exc
            cross_encoder = CrossEncoder(model_name)
        self.model = cross_encoder
        self.minimum_confidence = minimum_confidence
        self.minimum_lexical_overlap = minimum_lexical_overlap
        self.max_pairs = max_pairs

    @staticmethod
    def _contradiction_probability(scores: Sequence[float]) -> float:
        values = [float(value) for value in scores]
        peak = max(values)
        exponentials = [math.exp(value - peak) for value in values]
        return exponentials[0] / sum(exponentials)

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]

    def _has_subject_overlap(self, left: str, right: str) -> bool:
        left_terms = {
            word for word in WORDS.findall(left.lower()) if word not in STOP_WORDS
        }
        right_terms = {
            word for word in WORDS.findall(right.lower()) if word not in STOP_WORDS
        }
        overlap = len(left_terms & right_terms) / max(
            1, min(len(left_terms), len(right_terms))
        )
        return overlap >= self.minimum_lexical_overlap

    def detect(
        self, evidence: Sequence[SearchResult]
    ) -> Tuple[float, List[EvidenceConflict]]:
        candidates = []
        for left_index, left in enumerate(evidence):
            for right in evidence[left_index + 1 :]:
                if left.chunk.chunk_id == right.chunk.chunk_id:
                    continue
                for left_sentence in self._sentences(left.chunk.text):
                    for right_sentence in self._sentences(right.chunk.text):
                        if not self._has_subject_overlap(
                            left_sentence, right_sentence
                        ):
                            continue
                        candidates.append(
                            (
                                left_sentence,
                                right_sentence,
                                left.chunk.chunk_id,
                                right.chunk.chunk_id,
                            )
                        )
                        if len(candidates) >= self.max_pairs:
                            break
                    if len(candidates) >= self.max_pairs:
                        break
                if len(candidates) >= self.max_pairs:
                    break
            if len(candidates) >= self.max_pairs:
                break

        if not candidates:
            return 0.0, []

        logits = self.model.predict(
            [(left, right) for left, right, _, _ in candidates]
        )
        conflicts = []
        maximum_score = 0.0
        for candidate, scores in zip(candidates, logits):
            confidence = self._contradiction_probability(scores)
            maximum_score = max(maximum_score, confidence)
            if confidence >= self.minimum_confidence:
                left, right, left_chunk, right_chunk = candidate
                conflicts.append(
                    EvidenceConflict(
                        left_chunk_id=left_chunk,
                        right_chunk_id=right_chunk,
                        left_text=left,
                        right_text=right,
                        confidence=confidence,
                    )
                )
        return maximum_score, conflicts
