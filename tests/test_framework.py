import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from fairtrust_rag import FairTrustRAG, Settings
from fairtrust_rag.conflicts import NLIConflictDetector
from fairtrust_rag.evaluation import (
    EvaluationCase,
    fairness_summary,
    run_evaluation,
    summarize_results,
)
from fairtrust_rag.dataset_converters import convert_hotpotqa
from fairtrust_rag.generation import OllamaAnswerGenerator
from fairtrust_rag.fairness import generate_counterfactual_cases
from fairtrust_rag.ingestion import chunk_documents
from fairtrust_rag.models import Document
from fairtrust_rag.models import Chunk, EvidenceConflict, SearchResult
from fairtrust_rag.retrieval import expand_query
from fairtrust_rag.trust import apply_safety_gates, decide
from fairtrust_rag.verification import NLIEvidenceVerifier


class IngestionTests(unittest.TestCase):
    def test_chunking_preserves_metadata(self):
        document = Document("d1", "alpha beta gamma delta", "source.txt", {"kind": "test"})
        chunks = chunk_documents([document], chunk_size=12, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["kind"], "test")

    def test_retry_query_keeps_content_terms(self):
        self.assertEqual(
            expand_query("What is the distance between Earth and Mars?"),
            "distance between earth and mars",
        )


class ConfigurationTests(unittest.TestCase):
    def test_unknown_embedding_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(embedding_provider="unknown")

    def test_unknown_verification_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(verification_provider="unknown")


class FakeCrossEncoder:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def predict(self, pairs):
        return [next(self.outputs) for _ in pairs]


class NLIVerificationTests(unittest.TestCase):
    def setUp(self):
        chunk = Chunk("c1", "d1", "Paris is the capital of France.", "facts.txt")
        self.evidence = [SearchResult(chunk, 0.9)]

    def test_three_way_classification(self):
        model = FakeCrossEncoder(
            [
                [0.1, 3.0, 0.2],
                [3.0, 0.1, 0.2],
                [0.1, 0.2, 3.0],
            ]
        )
        verifier = NLIEvidenceVerifier(cross_encoder=model)
        results = verifier.verify(
            [
                "Paris is France's capital.",
                "Berlin is France's capital.",
                "Paris has ten million residents.",
            ],
            self.evidence,
        )
        self.assertEqual(
            [result.status for result in results],
            ["supported", "contradicted", "insufficient_evidence"],
        )


class ConflictDetectionTests(unittest.TestCase):
    def test_contradictory_passages_are_reported(self):
        evidence = [
            SearchResult(
                Chunk("c1", "d1", "The treatment improved outcomes.", "a.txt"),
                0.9,
            ),
            SearchResult(
                Chunk("c2", "d2", "The treatment did not improve outcomes.", "b.txt"),
                0.8,
            ),
        ]
        detector = NLIConflictDetector(
            cross_encoder=FakeCrossEncoder([[3.0, 0.1, 0.2]])
        )
        score, conflicts = detector.detect(evidence)
        self.assertGreater(score, 0.8)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].left_chunk_id, "c1")
        self.assertEqual(conflicts[0].right_chunk_id, "c2")

    def test_agreeing_passages_have_no_reported_conflict(self):
        evidence = [
            SearchResult(Chunk("c1", "d1", "Paris is in France.", "a.txt"), 0.9),
            SearchResult(
                Chunk("c2", "d2", "France contains the city of Paris.", "b.txt"),
                0.8,
            ),
        ]
        detector = NLIConflictDetector(
            cross_encoder=FakeCrossEncoder([[0.1, 3.0, 0.2]])
        )
        _, conflicts = detector.detect(evidence)
        self.assertEqual(conflicts, [])

    def test_different_entities_are_not_compared_as_a_conflict(self):
        evidence = [
            SearchResult(
                Chunk(
                    "c1",
                    "d1",
                    "Sergei Aleksandrovich Tokarev was a professor at Moscow State University.",
                    "tokarev.txt",
                ),
                0.9,
            ),
            SearchResult(
                Chunk(
                    "c2",
                    "d2",
                    "Sergei Aleksandrovich Kosarev is a Russian football midfielder.",
                    "kosarev.txt",
                ),
                0.8,
            ),
        ]
        detector = NLIConflictDetector(
            minimum_lexical_overlap=0.40,
            cross_encoder=FakeCrossEncoder([[3.0, 0.1, 0.2]]),
        )
        score, conflicts = detector.detect(evidence)
        self.assertEqual(score, 0.0)
        self.assertEqual(conflicts, [])


class DecisionControllerTests(unittest.TestCase):
    def test_conflict_is_shown_instead_of_answered(self):
        decision, reason = decide(
            risk=0.95,
            maximum_answer_risk=0.55,
            evidence_sufficient=True,
            conflict_detected=True,
        )
        self.assertEqual(decision, "show_conflict")
        self.assertIn("contradiction", reason)

    def test_conflict_confidence_sets_risk_floor(self):
        risk = apply_safety_gates(
            risk=0.10,
            evidence_sufficient=True,
            conflict_score=0.92,
        )
        self.assertEqual(risk, 0.92)

    def test_missing_evidence_takes_priority_over_conflict(self):
        decision, _ = decide(
            risk=1.0,
            maximum_answer_risk=0.55,
            evidence_sufficient=False,
            conflict_detected=True,
        )
        self.assertEqual(decision, "abstain")


class FixedConflictDetector:
    def detect(self, evidence):
        return 0.93, [
            EvidenceConflict(
                left_chunk_id="c1",
                right_chunk_id="c2",
                left_text="The intervention works.",
                right_text="The intervention does not work.",
                confidence=0.93,
            )
        ]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        Path(self.temp_dir.name, "facts.txt").write_text(
            "Paris is the capital of France. The Eiffel Tower is located in Paris.",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_supported_question_is_answered(self):
        pipeline = FairTrustRAG()
        self.assertGreater(pipeline.ingest(self.temp_dir.name), 0)
        report = pipeline.ask("What is the capital of France?")
        self.assertEqual(report.decision, "answer")
        self.assertIn("Paris", report.answer)
        self.assertTrue(report.citations)

    def test_unrelated_question_abstains(self):
        settings = Settings(maximum_answer_risk=0.40)
        pipeline = FairTrustRAG(settings)
        pipeline.ingest(self.temp_dir.name)
        report = pipeline.ask("How do volcanoes form?")
        self.assertEqual(report.decision, "abstain")
        self.assertIsNone(report.answer)
        self.assertEqual(report.risk_score, 1.0)

    def test_conflict_withholds_definitive_answer(self):
        pipeline = FairTrustRAG()
        pipeline.ingest(self.temp_dir.name)
        pipeline.conflict_detector = FixedConflictDetector()
        report = pipeline.ask("What is the capital of France?")
        self.assertEqual(report.decision, "show_conflict")
        self.assertIsNone(report.answer)
        self.assertEqual(report.citations, [])
        self.assertEqual(report.risk_score, 0.93)
        self.assertEqual(len(report.conflicts), 1)

    def test_retry_and_citation_metrics_are_reported(self):
        settings = Settings(
            retrieval_retry_enabled=True,
            retry_top_k=5,
            maximum_answer_risk=0.40,
        )
        pipeline = FairTrustRAG(settings)
        pipeline.ingest(self.temp_dir.name)
        unrelated = pipeline.ask("How do volcanoes form?")
        supported = pipeline.ask("What is the capital of France?")
        self.assertEqual(unrelated.retrieval_attempts, 2)
        self.assertEqual(supported.retrieval_attempts, 1)
        self.assertEqual(supported.citation_precision, 1.0)
        self.assertEqual(supported.citation_coverage, 1.0)

    def test_vector_index_round_trip(self):
        pipeline = FairTrustRAG()
        count = pipeline.ingest(self.temp_dir.name)
        index_path = Path(self.temp_dir.name, "index.json")
        pipeline.save_index(index_path)
        restored = FairTrustRAG()
        self.assertEqual(restored.load_index(index_path), count)
        report = restored.ask("What is the capital of France?")
        self.assertEqual(report.decision, "answer")

    def test_retrieval_can_be_scoped_to_candidate_documents(self):
        Path(self.temp_dir.name, "other.txt").write_text(
            "Berlin is the capital of Germany.",
            encoding="utf-8",
        )
        pipeline = FairTrustRAG()
        pipeline.ingest(self.temp_dir.name)
        report = pipeline.ask(
            "What is the capital of Germany?",
            allowed_sources=["other.txt"],
        )
        self.assertTrue(report.retrieved)
        self.assertEqual(
            Path(report.retrieved[0].chunk.source).name,
            "other.txt",
        )


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"response": "Paris is in France [c1]."}
        ).encode("utf-8")


class OllamaGeneratorTests(unittest.TestCase):
    def test_response_and_known_citations_are_parsed(self):
        def opener(request, timeout):
            self.assertEqual(timeout, 10)
            return FakeHTTPResponse()

        generator = OllamaAnswerGenerator(
            timeout_seconds=10,
            opener=opener,
        )
        evidence = [
            SearchResult(
                Chunk("c1", "d1", "Paris is in France.", "facts.txt"),
                0.9,
            )
        ]
        answer = generator.generate("Where is Paris?", evidence)
        self.assertIn("Paris", answer.text)
        self.assertEqual(answer.citations, ["c1"])


class StubEvaluationPipeline:
    def ask(self, question):
        if "unknown" in question:
            return SimpleNamespace(
                decision="abstain",
                answer=None,
                risk_score=1.0,
                conflict_score=0.0,
                retrieval_attempts=2,
            )
        return SimpleNamespace(
            decision="answer",
            answer="Paris is the capital of France.",
            risk_score=0.1,
            conflict_score=0.0,
            retrieval_attempts=1,
        )


class EvaluationTests(unittest.TestCase):
    def test_evaluation_and_group_gaps(self):
        evaluation = run_evaluation(
            StubEvaluationPipeline(),
            [
                EvaluationCase(
                    "a1",
                    "What is France's capital?",
                    "answer",
                    "Paris",
                    "group_a",
                ),
                EvaluationCase(
                    "b1",
                    "unknown question",
                    "abstain",
                    None,
                    "group_b",
                ),
            ],
        )
        self.assertEqual(evaluation["summary"]["decision_accuracy"], 1.0)
        self.assertEqual(evaluation["fairness"]["coverage_gap"], 1.0)

    def test_hallucination_rate_counts_wrong_accepted_answers(self):
        results = [
            {
                "decision": "answer",
                "answer_correct": False,
                "decision_correct": True,
                "risk_score": 0.2,
                "group": "a",
            }
        ]
        self.assertEqual(summarize_results(results)["hallucination_rate"], 1.0)
        self.assertIn("groups", fairness_summary(results))

    def test_counterfactual_cases_change_only_group_term(self):
        cases = generate_counterfactual_cases(
            "engineer",
            "A {group} engineer applied for the role.",
            ["female", "male"],
            expected_decision="answer",
        )
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].group, "female")
        self.assertIn("female", cases[0].question)
        self.assertIn("male", cases[1].question)

    def test_counterfactual_template_requires_placeholder(self):
        with self.assertRaises(ValueError):
            generate_counterfactual_cases("bad", "No placeholder", ["a", "b"])


class DatasetConverterTests(unittest.TestCase):
    def test_hotpotqa_conversion_preserves_evidence_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hotpot.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "_id": "abc123",
                            "question": "Where is Paris?",
                            "answer": "France",
                            "supporting_facts": [["Paris", 0]],
                            "context": [
                                ["Paris", ["Paris is in France."]],
                                ["Berlin", ["Berlin is in Germany."]],
                            ],
                            "type": "bridge",
                            "level": "easy",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            documents = root / "documents"
            cases_path = root / "cases.jsonl"
            cases, document_count = convert_hotpotqa(
                source, documents, cases_path, sample_size=1, seed=42
            )
            converted = json.loads(cases_path.read_text(encoding="utf-8"))
            self.assertEqual(cases, 1)
            self.assertEqual(document_count, 2)
            self.assertEqual(converted["source_dataset"], "hotpotqa")
            self.assertEqual(
                converted["supporting_documents"],
                ["hotpotqa/abc123_00.txt"],
            )
            self.assertEqual(
                converted["candidate_documents"],
                ["hotpotqa/abc123_00.txt", "hotpotqa/abc123_01.txt"],
            )


if __name__ == "__main__":
    unittest.main()
