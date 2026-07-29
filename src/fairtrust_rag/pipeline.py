"""End-to-end FairTrust-RAG orchestration."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .config import Settings
from .conflicts import NLIConflictDetector, NoOpConflictDetector
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .generation import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    OllamaAnswerGenerator,
)
from .ingestion import chunk_documents, load_documents
from .models import CitationVerification, ClaimVerification, TrustReport
from .retrieval import InMemoryVectorStore, expand_query
from .trust import apply_safety_gates, calculate_risk, decide
from .verification import LexicalEvidenceVerifier, NLIEvidenceVerifier, extract_claims


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
        if generator is not None:
            self.generator = generator
        elif self.settings.generation_provider == "ollama":
            self.generator = OllamaAnswerGenerator(
                self.settings.ollama_model,
                self.settings.ollama_url,
                self.settings.ollama_timeout_seconds,
            )
        else:
            self.generator = ExtractiveAnswerGenerator()
        if self.settings.verification_provider == "nli":
            self.verifier = NLIEvidenceVerifier(
                self.settings.verification_model,
                self.settings.minimum_nli_confidence,
            )
        else:
            self.verifier = LexicalEvidenceVerifier(
                self.settings.minimum_claim_support
            )
        if self.settings.conflict_detection_enabled:
            shared_model = getattr(self.verifier, "model", None)
            self.conflict_detector = NLIConflictDetector(
                self.settings.verification_model,
                self.settings.minimum_conflict_confidence,
                self.settings.minimum_conflict_overlap,
                self.settings.maximum_conflict_pairs,
                cross_encoder=shared_model,
            )
        else:
            self.conflict_detector = NoOpConflictDetector()

    def ingest(self, path: Union[str, Path]) -> int:
        documents = load_documents(path)
        chunks = chunk_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        self.store.add(chunks)
        return len(chunks)

    def save_index(self, path: Union[str, Path]) -> None:
        self.store.save(path)

    def load_index(self, path: Union[str, Path]) -> int:
        return self.store.load(path)

    def _verify_citations(
        self,
        claims: List[str],
        citations: List[str],
        retrieved: list,
    ) -> Tuple[List[CitationVerification], float, float]:
        if not self.settings.citation_verification_enabled or not claims:
            return [], 0.0, 0.0
        cited_evidence = [
            item for item in retrieved if item.chunk.chunk_id in citations
        ]
        verified: List[ClaimVerification] = self.verifier.verify(
            claims, cited_evidence
        )
        results = [
            CitationVerification(
                claim=item.claim,
                citation_id=item.evidence_chunk_id,
                status=item.status,
                support_score=item.support_score,
            )
            for item in verified
        ]
        supported = sum(item.status == "supported" for item in results)
        precision = supported / len(results) if results else 0.0
        coverage = supported / len(claims)
        return results, precision, coverage

    def ask(self, question: str) -> TrustReport:
        if not question.strip():
            raise ValueError("question cannot be empty")
        retrieved = self.store.search(question, self.settings.top_k)
        retrieval_attempts = 1
        evidence_sufficient = bool(
            retrieved
            and retrieved[0].score >= self.settings.minimum_retrieval_score
        )
        if self.settings.retrieval_retry_enabled and not evidence_sufficient:
            retrieved = self.store.search_many(
                [question, expand_query(question)],
                self.settings.retry_top_k,
            )
            retrieval_attempts = 2
            evidence_sufficient = bool(
                retrieved
                and retrieved[0].score >= self.settings.minimum_retrieval_score
            )
        conflict_score, conflicts = self.conflict_detector.detect(retrieved)
        answer = self.generator.generate(question, retrieved)
        extracted_claims = extract_claims(answer.text)
        claims = self.verifier.verify(extracted_claims, retrieved)
        citation_verifications, citation_precision, citation_coverage = (
            self._verify_citations(
                extracted_claims, answer.citations, retrieved
            )
        )
        risk = calculate_risk(
            retrieved,
            claims,
            answer,
            self.settings.minimum_retrieval_score,
            self.settings.risk_weights,
        )
        conflict_detected = bool(conflicts)
        risk = apply_safety_gates(
            risk,
            evidence_sufficient,
            conflict_score if conflict_detected else 0.0,
        )
        decision, reason = decide(
            risk,
            self.settings.maximum_answer_risk,
            evidence_sufficient,
            conflict_detected,
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
            conflict_score=conflict_score,
            conflicts=conflicts,
            retrieval_attempts=retrieval_attempts,
            citation_verifications=citation_verifications,
            citation_precision=citation_precision,
            citation_coverage=citation_coverage,
        )
