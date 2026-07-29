"""Risk scoring and selective-answering policy."""

from typing import Mapping, Sequence, Tuple

from .models import ClaimVerification, GeneratedAnswer, SearchResult


def calculate_risk(
    retrieved: Sequence[SearchResult],
    claims: Sequence[ClaimVerification],
    answer: GeneratedAnswer,
    minimum_retrieval_score: float,
    weights: Mapping[str, float],
) -> float:
    best_score = retrieved[0].score if retrieved else 0.0
    retrieval_risk = 1.0 - min(1.0, best_score / max(minimum_retrieval_score, 1e-9))
    unsupported_risk = (
        sum(claim.status != "supported" for claim in claims) / len(claims)
        if claims else 1.0
    )
    supported = [claim for claim in claims if claim.status == "supported"]
    cited_supported = sum(
        claim.evidence_chunk_id in answer.citations for claim in supported
    )
    citation_risk = (
        1.0 - cited_supported / len(supported) if supported else 1.0
    )
    risk = (
        weights["retrieval"] * retrieval_risk
        + weights["unsupported_claims"] * unsupported_risk
        + weights["citation"] * citation_risk
    )
    return max(0.0, min(1.0, risk))


def apply_safety_gates(
    risk: float, evidence_sufficient: bool, conflict_score: float = 0.0
) -> float:
    """Apply mandatory evidence gates to the reported risk.

    The weighted score measures soft signals. Evidence relevance is a hard
    prerequisite: an answer cannot be reliable when no retrieved passage meets
    the configured minimum. Detected conflict establishes a risk floor equal
    to its confidence.
    """
    return max(risk, conflict_score) if evidence_sufficient else 1.0


def decide(
    risk: float,
    maximum_answer_risk: float,
    evidence_sufficient: bool = True,
    conflict_detected: bool = False,
) -> Tuple[str, str]:
    if not evidence_sufficient:
        return "abstain", "No retrieved passage met the minimum relevance threshold."
    if conflict_detected:
        return (
            "show_conflict",
            "Retrieved evidence contains a high-confidence contradiction; "
            "a definitive answer has been withheld.",
        )
    if risk <= maximum_answer_risk:
        return "answer", "Retrieved evidence supports the answer within the configured risk limit."
    return "abstain", "The available evidence does not support a sufficiently reliable answer."
