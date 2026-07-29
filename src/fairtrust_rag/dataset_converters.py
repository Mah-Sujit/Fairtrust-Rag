"""Converters from public benchmark formats to FairTrust-RAG artifacts."""

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .evaluation import EvaluationCase


def convert_hotpotqa(
    input_path: Union[str, Path],
    documents_dir: Union[str, Path],
    cases_path: Union[str, Path],
    sample_size: int = 50,
    seed: int = 42,
) -> Tuple[int, int]:
    """Convert a deterministic HotpotQA sample into documents and JSONL cases."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    with Path(input_path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if sample_size > len(records):
        raise ValueError("sample_size exceeds the number of input records")

    selected = random.Random(seed).sample(records, sample_size)
    output_documents = Path(documents_dir)
    output_documents.mkdir(parents=True, exist_ok=True)
    output_cases = Path(cases_path)
    output_cases.parent.mkdir(parents=True, exist_ok=True)

    cases: List[EvaluationCase] = []
    document_count = 0
    for record in selected:
        case_id = str(record["_id"])
        title_to_files: Dict[str, List[str]] = {}
        for context_index, context in enumerate(record["context"]):
            title, sentences = context
            filename = f"{case_id}_{context_index:02d}.txt"
            relative_name = f"hotpotqa/{filename}"
            text = f"Title: {title}\n\n" + " ".join(sentences).strip()
            (output_documents / filename).write_text(text + "\n", encoding="utf-8")
            title_to_files.setdefault(title, []).append(relative_name)
            document_count += 1

        supporting_titles = {
            str(title) for title, _sentence_index in record["supporting_facts"]
        }
        supporting_documents = sorted(
            filename
            for title in supporting_titles
            for filename in title_to_files.get(title, [])
        )
        cases.append(
            EvaluationCase(
                case_id=f"hotpotqa-{case_id}",
                question=str(record["question"]),
                expected_decision="answer",
                gold_answer=str(record["answer"]),
                group="general",
                source_dataset="hotpotqa",
                evidence_condition="distractor",
                supporting_documents=supporting_documents,
            )
        )

    with output_cases.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    return len(cases), document_count
