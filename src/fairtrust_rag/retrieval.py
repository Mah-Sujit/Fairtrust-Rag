"""In-memory vector retrieval with JSON persistence."""

import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from .embeddings import Embedder
from .models import Chunk, SearchResult

QUESTION_WORDS = {
    "a", "an", "are", "do", "does", "how", "is", "should", "the", "what",
    "when", "where", "which", "who", "why",
}


def expand_query(query: str) -> str:
    """Create a deterministic content-focused retry query."""
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in QUESTION_WORDS
    ]
    return " ".join(terms) or query


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._records: List[Tuple[Chunk, List[float]]] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        self._records.extend((chunk, self.embedder.embed(chunk.text)) for chunk in chunks)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        query_vector = self.embedder.embed(query)
        ranked = sorted(
            (
                SearchResult(chunk=chunk, score=max(0.0, cosine_similarity(query_vector, vector)))
                for chunk, vector in self._records
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:top_k]

    def search_many(
        self, queries: Sequence[str], top_k: int = 3
    ) -> List[SearchResult]:
        best_by_chunk = {}
        for query in queries:
            for result in self.search(query, top_k):
                previous = best_by_chunk.get(result.chunk.chunk_id)
                if previous is None or result.score > previous.score:
                    best_by_chunk[result.chunk.chunk_id] = result
        return sorted(
            best_by_chunk.values(), key=lambda result: result.score, reverse=True
        )[:top_k]

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: Union[str, Path]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk": {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                },
                "vector": vector,
            }
            for chunk, vector in self._records
        ]
        destination.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: Union[str, Path]) -> int:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._records = [
            (
                Chunk(
                    chunk_id=item["chunk"]["chunk_id"],
                    document_id=item["chunk"]["document_id"],
                    text=item["chunk"]["text"],
                    source=item["chunk"]["source"],
                    metadata=item["chunk"].get("metadata", {}),
                ),
                [float(value) for value in item["vector"]],
            )
            for item in payload
        ]
        return len(self._records)
