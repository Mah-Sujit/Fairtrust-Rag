"""HotpotQA conversion command."""

import argparse

from .dataset_converters import convert_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HotpotQA into FairTrust-RAG artifacts"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--documents-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cases, documents = convert_hotpotqa(
        args.input,
        args.documents_dir,
        args.cases,
        args.sample_size,
        args.seed,
    )
    print(f"Converted {cases} cases and {documents} evidence documents.")


if __name__ == "__main__":
    main()
