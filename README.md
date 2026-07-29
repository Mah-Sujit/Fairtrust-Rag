# FairTrust-RAG

FairTrust-RAG is a minimal, modular research baseline for risk-controlled
retrieval-augmented generation. It ingests local documents, retrieves relevant
evidence, produces an answer, verifies atomic claims, calculates risk, and
either answers or abstains with an explanation.

This first version is intentionally dependency-free and runs locally on macOS
with Python 3.9 or newer. Its hashing embedder, extractive answer generator, and
lexical evidence verifier are transparent baselines—not research-grade models.
Their interfaces are designed to be replaced later.

## Current pipeline

```text
Text/Markdown documents
        ↓
Cleaning and overlapping chunks
        ↓
Hashing embeddings + in-memory cosine retrieval
        ↓
Extractive answer generator
        ↓
Atomic claim extraction
        ↓
Lexical evidence verification
        ↓
Weighted risk score
        ↓
Answer or abstain + structured trust report
```

## Setup on macOS

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

No API key, model download, or external database is required.

## Run the example

```bash
fairtrust-rag \
  --documents data/documents \
  --question "When should a trustworthy system abstain?" \
  --config configs/default.json
```

The command returns JSON containing the decision, answer, risk score,
retrieved evidence, citations, and claim-verification results.

You can also run it without installation:

```bash
PYTHONPATH=src python -m fairtrust_rag.cli \
  --documents data/documents \
  --question "What does retrieval-augmented generation combine?"
```

## Run tests

The tests use Python's built-in test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Or install the development extra and use pytest:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Configuration

Edit `configs/default.json` to change chunking, retrieval, verification, and
decision thresholds. The risk weights must add up to `1.0`.

The baseline risk is:

```text
risk =
  retrieval_weight × retrieval_risk
  + unsupported_claim_weight × unsupported_claim_ratio
  + citation_weight × citation_risk
```

The controller answers only when risk is at or below
`maximum_answer_risk`. These values are initial engineering defaults and must
be calibrated on validation data before making research claims.

## Module map

- `ingestion.py`: loads `.txt` and `.md` files and creates overlapping chunks
- `embeddings.py`: embedding interface and deterministic hashing baseline
- `retrieval.py`: in-memory cosine vector search
- `generation.py`: generator interface and extractive baseline
- `verification.py`: claim extraction and transparent verification stub
- `trust.py`: risk calculation and answer/abstain policy
- `pipeline.py`: end-to-end orchestration
- `models.py`: typed inputs, outputs, and trust report
- `cli.py`: runnable command-line entry point

## Research roadmap

Replace one baseline component at a time and evaluate each change:

1. Add a Sentence Transformers embedder and persistent FAISS index.
2. Add an open-source LLM adapter (for example, Ollama or Transformers).
3. Replace lexical verification with a three-way NLI model: supported,
   contradicted, or insufficient evidence.
4. Add evidence-conflict and citation-entailment checks.
5. Learn and calibrate risk on labelled validation data.
6. Add retrieve-again and show-conflict controller actions.
7. Add group-level reliability, coverage, and abstention evaluation.

Keep the simple implementations as experimental baselines. A strong study
should compare every new component against them and include ablations.

## Important limitation

This repository is an initial research scaffold. Its scores are not calibrated
probabilities, the verifier cannot detect nuanced contradiction, and the
extractive generator is not an LLM. Do not use it for high-stakes decisions.
