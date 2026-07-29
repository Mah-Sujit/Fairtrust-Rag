"""Answer-generation interfaces and a local extractive baseline."""

import re
from abc import ABC, abstractmethod
from typing import Sequence

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
