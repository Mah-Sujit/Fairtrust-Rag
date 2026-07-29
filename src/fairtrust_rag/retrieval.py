"""In-memory vector retrieval."""

from typing import List, Sequence, Tuple

from .embeddings import Embedder
from .models import Chunk, SearchResult


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

    def __len__(self) -> int:
        return len(self._records)

