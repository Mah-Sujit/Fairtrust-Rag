"""FairTrust-RAG minimal research framework."""

from .config import Settings
from .pipeline import FairTrustRAG

__all__ = ["FairTrustRAG", "Settings"]
__version__ = "0.1.0"

