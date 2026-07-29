"""Counterfactual evaluation-case generation."""

from typing import Iterable, List, Optional

from .evaluation import EvaluationCase


def generate_counterfactual_cases(
    case_id_prefix: str,
    question_template: str,
    groups: Iterable[str],
    expected_decision: Optional[str] = None,
    gold_answer: Optional[str] = None,
) -> List[EvaluationCase]:
    """Generate matched questions by replacing the ``{group}`` placeholder."""
    if "{group}" not in question_template:
        raise ValueError("question_template must contain the {group} placeholder")
    cases = []
    for index, group in enumerate(groups, start=1):
        cases.append(
            EvaluationCase(
                case_id=f"{case_id_prefix}-{index:03d}",
                question=question_template.format(group=group),
                expected_decision=expected_decision,
                gold_answer=gold_answer,
                group=group,
            )
        )
    return cases
