"""End-to-end FairTrust-RAG orchestration."""

from pathlib import Path
from typing import Optional, Union

from .config import Settings
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .generation import AnswerGenerator, ExtractiveAnswerGenerator
from .ingestion import chunk_documents, load_documents
from .models import TrustReport
from .retrieval import InMemoryVectorStore
from .trust import apply_safety_gates, calculate_risk, decide
from .verification import LexicalEvidenceVerifier, extract_claims


class FairTrustRAG:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.settings = settings or Settings()
        if self.settings.embedding_provider == "sentence_transformers":
            embedder = SentenceTransformerEmbedder(self.settings.embedding_model)
        else:
            embedder = HashingEmbedder(self.settings.embedding_dimensions)
        self.store = InMemoryVectorStore(embedder)
        self.generator = generator or ExtractiveAnswerGenerator()
        self.verifier = LexicalEvidenceVerifier(self.settings.minimum_claim_support)

    def ingest(self, path: Union[str, Path]) -> int:
        documents = load_documents(path)
        chunks = chunk_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.store.add(chunks)
        return len(chunks)

    def ask(self, question: str) -> TrustReport:
        if not question.strip():
            raise ValueError("question cannot be empty")
        retrieved = self.store.search(question, self.settings.top_k)
        answer = self.generator.generate(question, retrieved)
        claims = self.verifier.verify(extract_claims(answer.text), retrieved)
        risk = calculate_risk(
            retrieved,
            claims,
            answer,
            self.settings.minimum_retrieval_score,
            self.settings.risk_weights,
        )
        evidence_sufficient = bool(
            retrieved
            and retrieved[0].score >= self.settings.minimum_retrieval_score
        )
        risk = apply_safety_gates(risk, evidence_sufficient)
        decision, reason = decide(
            risk, self.settings.maximum_answer_risk, evidence_sufficient
        )
        return TrustReport(
            question=question,
            decision=decision,
            answer=answer.text if decision == "answer" else None,
            risk_score=risk,
            reason=reason,
            citations=answer.citations if decision == "answer" else [],
            retrieved=retrieved,
            claims=claims,
        )
