"""Answer-generation interfaces and a local extractive baseline."""

import re
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from .models import GeneratedAnswer, SearchResult


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(
        self, question: str, evidence: Sequence[SearchResult]
    ) -> GeneratedAnswer:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    """Returns the best evidence sentences; replace with an LLM adapter later."""

    def generate(
        self, question: str, evidence: Sequence[SearchResult]
    ) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(text="", citations=[])
        sentences = []
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        for result in evidence:
            for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.text):
                if sentence.strip():
                    terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                    sentences.append((len(question_terms & terms), sentence.strip()))
        sentences.sort(key=lambda item: item[0], reverse=True)
        answer = " ".join(sentence for _, sentence in sentences[:2])
        return GeneratedAnswer(text=answer, citations=[evidence[0].chunk.chunk_id])


class OllamaAnswerGenerator(AnswerGenerator):
    """Local Ollama adapter using its HTTP generation endpoint."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.timeout_seconds = timeout_seconds
        self.opener = opener or urllib.request.urlopen

    def generate(
        self, question: str, evidence: Sequence[SearchResult]
    ) -> GeneratedAnswer:
        context = "\n\n".join(
            f"[{item.chunk.chunk_id}] {item.chunk.text}" for item in evidence
        )
        prompt = (
            "You are an evidence-grounded question-answering system. Some "
            "questions require connecting facts from two or more passages. "
            "Work out that connection, then return only a concise direct answer. "
            "Use only the supplied evidence and cite every factual claim with "
            "its chunk identifier in square brackets. Never cite a chunk that "
            "does not support the claim. If the passages do not establish the "
            "answer, return exactly: INSUFFICIENT EVIDENCE.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )
        request = urllib.request.Request(
            self.url,
            data=json.dumps(
                {"model": self.model, "prompt": prompt, "stream": False}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Ollama is unavailable. Start Ollama and ensure the configured "
                f"model '{self.model}' is installed."
            ) from exc
        text = str(payload.get("response", "")).strip()
        known_ids = {item.chunk.chunk_id for item in evidence}
        citations = [
            citation
            for citation in re.findall(r"\[([^\]]+)\]", text)
            if citation in known_ids
        ]
        return GeneratedAnswer(text=text, citations=list(dict.fromkeys(citations)))
