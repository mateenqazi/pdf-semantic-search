from pathlib import Path
from threading import Lock
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray
from tokenizers import Tokenizer

from app.constants import (
    CHUNK_TOKENS,
    EMBEDDING_BATCH_SIZE,
    MODEL_NAME,
    MODEL_TOKEN_LIMIT,
    OVERLAP_TOKENS,
)


class Embedder(Protocol):
    def split_text(self, text: str) -> list[str]: ...
    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


def token_chunks(text: str, tokenizer: Tokenizer) -> list[str]:
    """Split text into overlapping chunks at token boundaries."""
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    chunks: list[str] = []
    for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS):
        end = min(start + CHUNK_TOKENS, len(offsets))
        start_character = offsets[start][0]
        end_character = offsets[end - 1][1]
        chunk = text[start_character:end_character].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(offsets):
            break
    return chunks


class LocalEmbedder:
    def __init__(self, cache_dir: Path, threads: int | None = None) -> None:
        self.model = TextEmbedding(MODEL_NAME, cache_dir=str(cache_dir), threads=threads)
        # Use the model's token limit instead of the tokenizer's default of 512.
        self.model.model.tokenizer.enable_truncation(max_length=MODEL_TOKEN_LIMIT)
        # Copy the tokenizer so chunking doesn't change the model's settings.
        self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.lock = Lock()

    def split_text(self, text: str) -> list[str]:
        return token_chunks(text, self.tokenizer)

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        # Release the model lock between batches so searches can run.
        vectors: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            with self.lock:
                # Embeddings are computed as the generator is consumed; keep the lock held.
                vectors.extend(self.model.passage_embed(batch, batch_size=EMBEDDING_BATCH_SIZE))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        with self.lock:
            return np.asarray(next(self.model.query_embed(text)), dtype=np.float32)
