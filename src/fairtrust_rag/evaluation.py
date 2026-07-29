"""Evaluation and group-fairness metrics for FairTrust-RAG experiments."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from .pipeline import FairTrustRAG


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_decision: Optional[str] = None
    gold_answer: Optional[str] = None
    group: str = "all"
    source_dataset: Optional[str] = None
    evidence_condition: Optional[str] = None
    supporting_documents: List[str] = field(default_factory=list)
    candidate_documents: List[str] = field(default_factory=list)


def load_cases(path: Union[str, Path]) -> List[EvaluationCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cases.append(EvaluationCase(**item))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid evaluation case on line {line_number}"
                ) from exc
    return cases


def _answer_matches(answer: Optional[str], gold_answer: Optional[str]) -> Optional[bool]:
    if gold_answer is None:
        return None
    return gold_answer.lower().strip() in (answer or "").lower()


def _source_names(paths: Iterable[str]) -> set:
    return {Path(path).name for path in paths}


def _rate(values: Iterable[bool]) -> Optional[float]:
    items = list(values)
    return sum(items) / len(items) if items else None


def summarize_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    answered = [item["decision"] == "answer" for item in results]
    correct_answers = [
        bool(item["answer_correct"])
        for item in results
        if item["answer_correct"] is not None
    ]
    decision_matches = [
        bool(item["decision_correct"])
        for item in results
        if item["decision_correct"] is not None
    ]
    answered_with_gold = [
        item
        for item in results
        if item["decision"] == "answer" and item["answer_correct"] is not None
    ]
    hallucinations = [
        not bool(item["answer_correct"]) for item in answered_with_gold
    ]
    calibration_items = [
        abs(
            (1.0 - float(item["risk_score"]))
            - float(bool(item["answer_correct"]))
        )
        for item in results
        if item["answer_correct"] is not None
    ]
    retrieval_recalls = [
        float(item["supporting_document_recall"])
        for item in results
        if item.get("supporting_document_recall") is not None
    ]
    joint_retrieval = [
        bool(item["all_supporting_documents_retrieved"])
        for item in results
        if item.get("all_supporting_documents_retrieved") is not None
    ]
    return {
        "cases": len(results),
        "coverage": _rate(answered),
        "answer_accuracy": _rate(correct_answers),
        "decision_accuracy": _rate(decision_matches),
        "hallucination_rate": _rate(hallucinations),
        "abstention_rate": _rate(
            item["decision"] == "abstain" for item in results
        ),
        "conflict_rate": _rate(
            item["decision"] == "show_conflict" for item in results
        ),
        "average_risk": (
            sum(float(item["risk_score"]) for item in results) / len(results)
            if results else None
        ),
        "calibration_error": (
            sum(calibration_items) / len(calibration_items)
            if calibration_items else None
        ),
        "supporting_document_recall": (
            sum(retrieval_recalls) / len(retrieval_recalls)
            if retrieval_recalls else None
        ),
        "joint_supporting_document_recall": _rate(joint_retrieval),
    }


def fairness_summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in results:
        grouped[str(item["group"])].append(item)
    group_metrics = {
        group: summarize_results(items) for group, items in sorted(grouped.items())
    }

    def gap(metric: str) -> Optional[float]:
        values = [
            float(metrics[metric])
            for metrics in group_metrics.values()
            if metrics[metric] is not None
        ]
        return max(values) - min(values) if len(values) >= 2 else None

    return {
        "groups": group_metrics,
        "coverage_gap": gap("coverage"),
        "answer_accuracy_gap": gap("answer_accuracy"),
        "abstention_rate_gap": gap("abstention_rate"),
        "hallucination_rate_gap": gap("hallucination_rate"),
        "calibration_error_gap": gap("calibration_error"),
        "worst_group_coverage": min(
            (
                float(metrics["coverage"])
                for metrics in group_metrics.values()
                if metrics["coverage"] is not None
            ),
            default=None,
        ),
        "worst_group_answer_accuracy": min(
            (
                float(metrics["answer_accuracy"])
                for metrics in group_metrics.values()
                if metrics["answer_accuracy"] is not None
            ),
            default=None,
        ),
    }


def run_evaluation(
    pipeline: FairTrustRAG, cases: Iterable[EvaluationCase]
) -> Dict[str, object]:
    results = []
    for case in cases:
        if case.candidate_documents:
            report = pipeline.ask(
                case.question,
                allowed_sources=case.candidate_documents,
            )
        else:
            report = pipeline.ask(case.question)
        answer_correct = _answer_matches(report.answer, case.gold_answer)
        gold_sources = _source_names(case.supporting_documents)
        retrieved_sources = _source_names(
            item.chunk.source for item in report.retrieved
        )
        supporting_document_recall = (
            len(gold_sources & retrieved_sources) / len(gold_sources)
            if gold_sources else None
        )
        all_supporting_documents_retrieved = (
            gold_sources <= retrieved_sources if gold_sources else None
        )
        decision_correct = (
            report.decision == case.expected_decision
            if case.expected_decision is not None else None
        )
        results.append(
            {
                "case_id": case.case_id,
                "group": case.group,
                "question": case.question,
                "decision": report.decision,
                "expected_decision": case.expected_decision,
                "decision_correct": decision_correct,
                "answer_correct": answer_correct,
                "risk_score": report.risk_score,
                "conflict_score": report.conflict_score,
                "retrieval_attempts": report.retrieval_attempts,
                "source_dataset": case.source_dataset,
                "evidence_condition": case.evidence_condition,
                "supporting_documents": case.supporting_documents,
                "candidate_documents": case.candidate_documents,
                "retrieved_documents": sorted(retrieved_sources),
                "supporting_document_recall": supporting_document_recall,
                "all_supporting_documents_retrieved": (
                    all_supporting_documents_retrieved
                ),
            }
        )
    return {
        "summary": summarize_results(results),
        "fairness": fairness_summary(results),
        "results": results,
    }
