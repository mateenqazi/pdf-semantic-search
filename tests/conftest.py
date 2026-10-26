from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen.canvas import Canvas

from app.main import create_app


def make_pdf(*pages: str) -> bytes:
    stream = BytesIO()
    canvas = Canvas(stream, invariant=1)
    for page in pages:
        text = canvas.beginText(40, 800)
        for line in page.splitlines():
            text.textLine(line)
        canvas.drawText(text)
        canvas.showPage()
    canvas.save()
    return stream.getvalue()


class FakeEmbedder:
    """Use fixed vectors to test the API without loading a model."""

    def split_text(self, text: str) -> list[str]:
        return [text]

    def embed_query(self, text: str) -> np.ndarray:
        if "car" in text.lower():
            return np.array([1, 0, 0], dtype=np.float32)
        if "banana" in text.lower():
            return np.array([0, 1, 0], dtype=np.float32)
        return np.array([0, 0, 1], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(text) for text in texts])


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def isolate_openai_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.fixture
def client(tmp_path, embedder):
    with TestClient(create_app(tmp_path, embedder)) as client:
        yield client
