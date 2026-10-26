import os
from pathlib import Path

import pytest
from conftest import make_pdf
from fastapi.testclient import TestClient

from app.constants import CHUNK_TOKENS
from app.embeddings import LocalEmbedder
from app.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run the real model"
    ),
]


@pytest.fixture(scope="module")
def real_embedder():
    return LocalEmbedder(Path(os.getenv("MODEL_CACHE_DIR", ".cache/models")))


def test_real_cross_language_retrieval(tmp_path, real_embedder):
    pdf = make_pdf(
        "Bananen wachsen in warmen tropischen Regionen.",
        "Bei einer Autopanne bekommt der Kunde einen Ersatzwagen.\n"
        "Die Kosten fuer einen Mietwagen werden erstattet.",
    )
    with TestClient(create_app(tmp_path, real_embedder)) as client:
        response = client.post("/documents", files={"file": ("demo.pdf", pdf, "application/pdf")})
        assert response.status_code == 201, response.text
        for query in [
            "Can I get a rental car if my vehicle breaks down?",
            "Wer bezahlt den Mietwagen bei einer Panne?",
        ]:
            response = client.post("/query", json={"query": query, "top_k": 1})
            assert response.status_code == 200
            assert response.json()["results"][0]["page"] == 2


def test_token_chunks_fit_model_and_preserve_text(real_embedder):
    text = " ".join(
        f"Abschnitt {i}: Gewährleistung und Ersatzfahrzeuge werden geprüft." for i in range(80)
    )
    chunks = real_embedder.split_text(text)
    assert len(chunks) > 1
    assert all(chunk in text for chunk in chunks)
    assert all(
        len(real_embedder.tokenizer.encode(c, add_special_tokens=False).ids) <= CHUNK_TOKENS + 1
        for c in chunks
    )
    assert text.startswith(chunks[0]) and text.endswith(chunks[-1])
    # Check that overlapping chunks leave no gaps in the original text.
    cursor = 0
    for chunk in chunks:
        start = text.find(chunk, max(0, cursor - len(chunk)))
        assert 0 <= start <= cursor
        cursor = max(cursor, start + len(chunk))
    assert cursor == len(text)
