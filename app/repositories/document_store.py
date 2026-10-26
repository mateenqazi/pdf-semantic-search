import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.constants import CHUNK_TOKENS, MODEL_NAME, OVERLAP_TOKENS
from app.pdf import Chunk
from app.schemas import Document, SearchHit


def normalize_vectors(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    vectors = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(lengths).all():
        raise ValueError("Embedding model returned a non-finite vector.")
    if (lengths == 0).any():
        raise ValueError("Embedding model returned a zero vector.")
    return vectors / lengths


class DocumentStore:
    """Store documents in SQLite and search their embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    chunk_id INTEGER NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (document_id, chunk_id)
                );
            """)
            signature = f"v1:{MODEL_NAME}:{CHUNK_TOKENS}:{OVERLAP_TOKENS}"
            connection.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES ('index_signature', ?)",
                (signature,),
            )
            stored_signature = connection.execute(
                "SELECT value FROM metadata WHERE key = 'index_signature'"
            ).fetchone()["value"]
            if stored_signature != signature:
                raise RuntimeError(
                    "Index configuration changed. Use a new DATA_DIR and re-upload PDFs."
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def get_document(self, document_id: str) -> Document | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return Document(**dict(row)) if row is not None else None

    def list_documents(self) -> list[Document]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY rowid").fetchall()
        return [Document(**dict(row)) for row in rows]

    def save_document(
        self, document: Document, chunks: list[Chunk], vectors: NDArray[np.float32]
    ) -> tuple[Document, bool]:
        """Return the document and True if it was newly saved."""
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match the document chunks.")
        vectors = normalize_vectors(vectors).astype("<f4")
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO documents (document_id, filename, pages, chunks)
                   VALUES (?, ?, ?, ?) ON CONFLICT(document_id) DO NOTHING""",
                (document.document_id, document.filename, document.pages, document.chunks),
            )
            if cursor.rowcount == 0:
                # Another upload may have saved this PDF while embeddings were being computed.
                row = connection.execute(
                    "SELECT * FROM documents WHERE document_id = ?", (document.document_id,)
                ).fetchone()
                return Document(**dict(row)), False

            chunk_rows = [
                (document.document_id, chunk_id, chunk.page, chunk.text, vector.tobytes())
                for chunk_id, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ]
            connection.executemany(
                """INSERT INTO chunks (document_id, chunk_id, page, text, vector)
                   VALUES (?, ?, ?, ?, ?)""",
                chunk_rows,
            )
        return document, True

    def search(
        self,
        query_vector: NDArray[np.float32],
        *,
        document_id: str | None,
        top_k: int,
        min_score: float,
    ) -> list[SearchHit]:
        sql = """SELECT chunks.*, documents.filename
                 FROM chunks JOIN documents USING(document_id)"""
        parameters: tuple[str, ...] = ()
        if document_id is not None:
            sql += " WHERE chunks.document_id = ?"
            parameters = (document_id,)
        sql += " ORDER BY chunks.document_id, chunks.chunk_id"
        with self.connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        if not rows:
            return []

        vectors = np.stack([np.frombuffer(row["vector"], dtype="<f4") for row in rows])
        scores = np.clip(vectors @ normalize_vectors(query_vector), -1, 1)
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        results: list[SearchHit] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            row = rows[index]
            results.append(
                SearchHit(
                    document_id=row["document_id"],
                    filename=row["filename"],
                    page=row["page"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=score,
                )
            )
        return results
