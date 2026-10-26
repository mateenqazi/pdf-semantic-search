import hashlib
import logging
import sqlite3

from app.answers import AnswerGenerator
from app.constants import NO_PASSAGES_ANSWER
from app.embeddings import Embedder
from app.pdf import InvalidPdfError, extract_chunks
from app.repositories.document_store import DocumentStore
from app.schemas import AnswerResponse, Document, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


class DocumentNotFoundError(LookupError):
    pass


class AnswersDisabledError(RuntimeError):
    pass


class ProcessingError(RuntimeError):
    """Document processing failed."""


class DocumentService:
    """Handle document uploads, searches, and answers."""

    def __init__(
        self, store: DocumentStore, embedder: Embedder, generator: AnswerGenerator | None = None
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator

    def list_documents(self) -> list[Document]:
        try:
            return self.store.list_documents()
        except sqlite3.Error as exc:
            logger.exception("Could not list documents")
            raise ProcessingError("Could not list documents. Please retry.") from exc

    def upload_pdf(self, data: bytes, filename: str) -> tuple[Document, bool]:
        """Return the document and True if it was newly saved."""
        document_id = hashlib.sha256(data).hexdigest()
        try:
            existing = self.store.get_document(document_id)
            if existing is not None:
                return existing, False

            page_count, chunks = extract_chunks(data, self.embedder.split_text)
            document = Document(
                document_id=document_id,
                filename=filename,
                pages=page_count,
                chunks=len(chunks),
            )
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            return self.store.save_document(document, chunks, vectors)
        except InvalidPdfError:
            raise
        except Exception as exc:
            logger.exception("Could not index document %s", document_id)
            raise ProcessingError(
                "Indexing failed. No partial document was saved; retry the upload."
            ) from exc

    def search(self, request: SearchRequest) -> SearchResponse:
        try:
            if request.document_id is not None:
                document = self.store.get_document(request.document_id)
                if document is None:
                    raise DocumentNotFoundError("Document not found.")

            vector = self.embedder.embed_query(request.query)
            results = self.store.search(
                vector,
                document_id=request.document_id,
                top_k=request.top_k,
                min_score=request.min_score,
            )
            return SearchResponse(query=request.query, results=results)
        except DocumentNotFoundError:
            raise
        except Exception as exc:
            logger.exception("Could not search documents")
            raise ProcessingError("Search failed. Please retry.") from exc

    def answer(self, request: SearchRequest) -> AnswerResponse:
        if self.generator is None:
            raise AnswersDisabledError("Answer generation is disabled. Set OPENAI_API_KEY.")

        passages = self.search(request).results
        if not passages:
            return AnswerResponse(query=request.query, answer=NO_PASSAGES_ANSWER, sources=[])

        try:
            text = self.generator.answer(request.query, passages)
        except Exception as exc:
            logger.exception("Could not generate an answer")
            raise ProcessingError("Answer generation failed. Please retry.") from exc
        return AnswerResponse(query=request.query, answer=text, sources=passages)
