import tempfile
import unittest
from pathlib import Path

from fairtrust_rag import FairTrustRAG, Settings
from fairtrust_rag.conflicts import NLIConflictDetector
from fairtrust_rag.ingestion import chunk_documents
from fairtrust_rag.models import Document
from fairtrust_rag.models import Chunk, EvidenceConflict, SearchResult
from fairtrust_rag.trust import apply_safety_gates, decide
from fairtrust_rag.verification import NLIEvidenceVerifier


class IngestionTests(unittest.TestCase):
    def test_chunking_preserves_metadata(self):
        document = Document("d1", "alpha beta gamma delta", "source.txt", {"kind": "test"})
        chunks = chunk_documents([document], chunk_size=12, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["kind"], "test")


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


if __name__ == "__main__":
    unittest.main()
