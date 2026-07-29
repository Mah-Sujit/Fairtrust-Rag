"""Command-line interface."""

import argparse
import json
from pathlib import Path

from .config import Settings
from .pipeline import FairTrustRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FairTrust-RAG baseline")
    parser.add_argument("--documents", required=True, help="Text/Markdown file or directory")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--config", help="Optional JSON configuration path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_json(Path(args.config)) if args.config else Settings()
    pipeline = FairTrustRAG(settings)
    chunks = pipeline.ingest(args.documents)
    if not chunks:
        raise SystemExit("No non-empty .txt or .md documents were found.")
    print(json.dumps(pipeline.ask(args.question).to_dict(), indent=2))


if __name__ == "__main__":
    main()

