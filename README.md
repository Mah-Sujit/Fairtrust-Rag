# FairTrust-RAG

FairTrust-RAG is a minimal, modular research baseline for risk-controlled
retrieval-augmented generation. It ingests local documents, retrieves relevant
evidence, produces an answer, verifies atomic claims, calculates risk, and
either answers or abstains with an explanation.

The core baseline is dependency-free. The default example configuration uses
Sentence Transformers for semantic retrieval, while automated tests retain the
fast deterministic hashing baseline. The extractive answer generator and
lexical evidence verifier remain transparent baselines—not research-grade
models.

## Current pipeline

```text
Text/Markdown documents
        ↓
Cleaning and overlapping chunks
        ↓
Semantic or hashing embeddings + in-memory cosine retrieval
        ↓
Passage-to-passage conflict detection
        ↓
Extractive answer generator
        ↓
Atomic claim extraction
        ↓
Three-way NLI or lexical evidence verification
        ↓
Weighted risk score
        ↓
Answer or abstain + structured trust report
```

## Setup on macOS

From this project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[semantic]"
```

No API key or external database is required. The semantic model is downloaded
once on first use and then loaded from the local model cache.

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
decision thresholds. Set `embedding_provider` to `sentence_transformers` for
semantic retrieval or `hashing` for the dependency-free baseline. The risk
weights must add up to `1.0`.

Set `verification_provider` to `nli` to classify every generated claim as
`supported`, `contradicted`, or `insufficient_evidence`. The default NLI model
is `cross-encoder/nli-deberta-v3-small`. Use `lexical` for the fast,
dependency-free test baseline.

When `conflict_detection_enabled` is true, retrieved sentences from different
chunks are compared before generation. The trust report includes the maximum
`conflict_score` and every sentence pair above
`minimum_conflict_confidence`. `maximum_conflict_pairs` bounds runtime.

The controller treats a detected contradiction as a separate outcome:
`decision` becomes `show_conflict`, the definitive answer and citations are
withheld, and the report exposes both conflicting sentences. Conflict
confidence also becomes the minimum reported risk for that response.

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

Evidence relevance is a mandatory safety gate. If no retrieved passage meets
`minimum_retrieval_score`, the framework abstains and reports risk `1.0`
regardless of the softer weighted signals.

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

1. Evaluate semantic retrieval against the hashing baseline and add a
   persistent FAISS index.
2. Add an open-source LLM adapter (for example, Ollama or Transformers).
3. Calibrate the three-way NLI verifier on domain-specific labelled claims.
4. Add passage-to-passage evidence-conflict and citation-entailment checks.
5. Learn and calibrate risk on labelled validation data.
6. Add retrieve-again and show-conflict controller actions.
7. Add group-level reliability, coverage, and abstention evaluation.

Keep the simple implementations as experimental baselines. A strong study
should compare every new component against them and include ablations.

## Important limitation

This repository is an initial research scaffold. Its scores are not calibrated
probabilities, the verifier cannot detect nuanced contradiction, and the
extractive generator is not an LLM. Do not use it for high-stakes decisions.
