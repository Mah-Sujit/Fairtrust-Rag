# FairTrust-RAG

FairTrust-RAG is a minimal, modular research baseline for risk-controlled
retrieval-augmented generation. It ingests local documents, retrieves relevant
evidence, produces an answer, verifies atomic claims, calculates risk, and
either answers or abstains with an explanation.

The core baseline is dependency-free. The default example configuration uses
Sentence Transformers for semantic retrieval, while automated tests retain the
fast deterministic hashing baseline. Generation can use either the extractive
baseline or a local Ollama model.

## Current pipeline

```text
Text/Markdown documents
        ↓
Cleaning and overlapping chunks
        ↓
Semantic or hashing embeddings + in-memory cosine retrieval
        ↓
Automatic expanded retrieval retry when evidence is weak
        ↓
Passage-to-passage conflict detection
        ↓
Extractive or local Ollama answer generator
        ↓
Atomic claim extraction
        ↓
Three-way NLI or lexical evidence verification
        ↓
Claim-to-citation verification
        ↓
Weighted risk score
        ↓
Answer / show conflict / abstain + structured trust report
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
retrieved evidence, retrieval attempts, citation metrics, conflicts, and
claim-verification results.

Reuse a persistent index between runs:

```bash
fairtrust-rag \
  --documents data/documents \
  --index-path data/indexes/documents.json \
  --question "What is Natural Language Processing?" \
  --config configs/default.json
```

The first run creates the index. Later runs load it without re-embedding the
documents. Rebuild it after changing the source documents or embedding model.

## Optional local Ollama generation

Install Ollama separately, download a model, and start its local service:

```bash
ollama pull llama3.2:3b
ollama serve
```

Use `configs/hotpotqa-ollama.json` for the multi-hop benchmark. It retrieves
eight candidate passages and uses a prompt that requires cross-document
reasoning, concise answers, and claim-level citations. The default remains
`extractive`, so Ollama is not required for tests or basic operation.

## Run benchmark and fairness evaluation

Evaluation data uses one JSON object per line:

```json
{"case_id":"q1","question":"What is NLP?","expected_decision":"answer","gold_answer":"artificial intelligence","group":"group_a"}
```

Run the included example:

```bash
fairtrust-evaluate \
  --documents data/documents \
  --dataset data/evaluation/example.jsonl \
  --config configs/default.json \
  --index-path data/indexes/documents.json \
  --output results/example.json
```

The report includes answer accuracy, decision accuracy, coverage,
hallucination rate, abstention rate, conflict rate, average risk, calibration
error, group-level metrics, fairness gaps, and worst-group performance.

## Convert HotpotQA

Keep original public datasets under the ignored `data/external/` directory.
Create a deterministic 50-case HotpotQA pilot with:

```bash
fairtrust-convert-hotpotqa \
  --input data/external/hotpotqa/hotpot_dev_distractor_v1.json \
  --documents-dir data/benchmarks/hotpotqa/documents \
  --cases data/benchmarks/hotpotqa/cases.jsonl \
  --sample-size 50 \
  --seed 42
```

The converter creates the ten distractor-setting evidence passages for every
sampled question and records the original ID, source dataset, evidence
condition, all ten candidate documents, and gold supporting-document
filenames. Evaluation restricts retrieval to each question's candidate set,
matching the HotpotQA distractor protocol. Do not tune on the same
sample later used for final testing. Original and generated benchmark data are
ignored by Git; regenerate them from the documented source and seed.

Run the multi-hop generation experiment after Ollama is running:

```bash
fairtrust-evaluate \
  --documents data/benchmarks/hotpotqa-v2/documents \
  --dataset data/benchmarks/hotpotqa-v2/cases.jsonl \
  --config configs/hotpotqa-ollama.json \
  --index-path data/indexes/hotpotqa-50-v3.json \
  --output results/hotpotqa-50-ollama.json
```

The result now reports supporting-document recall and joint supporting-document
recall. These separate retrieval failures from answer-generation failures.

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

Retrieval thresholds are embedding-model specific. The included `0.10`
semantic threshold is only a starter value derived from the example corpus;
calibrate it on a separate validation split before reporting results.

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

When `retrieval_retry_enabled` is true, insufficient first-pass evidence
triggers a second search using both the original question and a simplified
content query, merged up to `retry_top_k`. The final report records one or two
retrieval attempts. If the expanded search is still inadequate, the controller
abstains.

Citation verification restricts evidence verification to the chunks actually
cited by the generator. Reports include citation precision, citation coverage,
and a result for each factual claim.

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
- `retrieval.py`: cosine vector search and persistent JSON indexes
- `generation.py`: extractive and local Ollama generators
- `verification.py`: lexical and NLI claim verification
- `conflicts.py`: NLI passage-to-passage contradiction detection
- `trust.py`: risk calculation and answer/abstain policy
- `pipeline.py`: end-to-end orchestration
- `evaluation.py`: reliability, selective-answering, and group metrics
- `fairness.py`: matched demographic counterfactual case generation
- `models.py`: typed inputs, outputs, and trust report
- `cli.py` and `evaluation_cli.py`: application and experiment commands

## Research roadmap

The implementation framework is now complete. The remaining work is empirical
research rather than missing scaffolding:

1. Build and manually validate a larger labelled benchmark.
2. Calibrate retrieval, NLI, conflict, and risk thresholds on validation data.
3. Compare hashing, semantic, extractive, and Ollama baselines.
4. Run ablations for retry, NLI, citation, conflict, and safety-gate modules.
5. Construct justified demographic counterfactual groups and inspect them for
   validity before interpreting fairness gaps.
6. Analyse latency, memory, token use, errors, and limitations.

Keep the simple implementations as experimental baselines. A strong study
should compare every new component against them and include ablations.

## Important limitation

This repository is an initial research scaffold. Its scores are not calibrated
probabilities, the verifier cannot detect nuanced contradiction, and the
extractive generator is not an LLM. Do not use it for high-stakes decisions.
