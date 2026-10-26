import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from openai import OpenAI

from app.answers import AnswerGenerator, OpenAIAnswerGenerator
from app.constants import DEFAULT_MODEL
from app.embeddings import Embedder, LocalEmbedder
from app.repositories.document_store import DocumentStore
from app.routers.documents import router
from app.services.document_service import DocumentService


def default_generator() -> OpenAIAnswerGenerator | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAIAnswerGenerator(OpenAI(timeout=60), os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def create_app(
    data_dir: Path | None = None,
    embedder: Embedder | None = None,
    generator: AnswerGenerator | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database_dir = data_dir if data_dir is not None else Path(os.getenv("DATA_DIR", ".data"))
        model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", ".cache/models"))
        threads = int(os.environ["EMBEDDING_THREADS"]) if os.getenv("EMBEDDING_THREADS") else None
        store = DocumentStore(database_dir / "index.sqlite3")
        model = embedder if embedder is not None else LocalEmbedder(model_cache_dir, threads)
        default_answers = default_generator() if generator is None else None
        answers = generator if generator is not None else default_answers
        try:
            application.state.document_service = DocumentService(store, model, answers)
            yield
        finally:
            if default_answers is not None:
                default_answers.close()

    application = FastAPI(
        title="PDF Semantic Search",
        version="0.2.0",
        lifespan=lifespan,
        description="Upload text PDFs, search them, and get cited answers to questions.",
    )
    application.include_router(router)
    return application


app = create_app()
