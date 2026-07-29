"""Dependency-free baseline embeddings."""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic signed feature hashing suitable for a local baseline."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector

