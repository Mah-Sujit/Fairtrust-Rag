"""Plain-text and Markdown document ingestion."""

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Union

from .models import Chunk, Document


SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(path: Union[str, Path]) -> List[Document]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        digest = hashlib.sha1(str(file_path.resolve()).encode()).hexdigest()[:12]
        documents.append(
            Document(
                document_id=digest,
                text=text,
                source=str(file_path),
                metadata={"filename": file_path.name},
            )
        )
    return documents


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        normalized = re.sub(r"\s+", " ", document.text).strip()
        start = 0
        index = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            text = normalized[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}_chunk_{index:04d}",
                        document_id=document.document_id,
                        text=text,
                        source=document.source,
                        metadata=document.metadata,
                    )
                )
            if end >= len(normalized):
                break
            start = max(end - overlap, start + step)
            index += 1
    return chunks

