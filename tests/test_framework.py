import tempfile
import unittest
from pathlib import Path

from fairtrust_rag import FairTrustRAG, Settings
from fairtrust_rag.ingestion import chunk_documents
from fairtrust_rag.models import Document


class IngestionTests(unittest.TestCase):
    def test_chunking_preserves_metadata(self):
        document = Document("d1", "alpha beta gamma delta", "source.txt", {"kind": "test"})
        chunks = chunk_documents([document], chunk_size=12, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["kind"], "test")


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


if __name__ == "__main__":
    unittest.main()

