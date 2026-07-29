"""Command-line experiment runner."""

import argparse
import json
from pathlib import Path

from .config import Settings
from .evaluation import load_cases, run_evaluation
from .pipeline import FairTrustRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FairTrust-RAG")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--dataset", required=True, help="JSONL evaluation cases")
    parser.add_argument("--config")
    parser.add_argument("--index-path")
    parser.add_argument("--output", help="Optional JSON results path")
    args = parser.parse_args()

    settings = Settings.from_json(args.config) if args.config else Settings()
    pipeline = FairTrustRAG(settings)
    index_path = Path(args.index_path) if args.index_path else None
    if index_path and index_path.exists():
        pipeline.load_index(index_path)
    else:
        pipeline.ingest(args.documents)
        if index_path:
            pipeline.save_index(index_path)

    evaluation = run_evaluation(pipeline, load_cases(args.dataset))
    serialized = json.dumps(evaluation, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
