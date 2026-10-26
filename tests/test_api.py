from io import BytesIO

import pytest
from conftest import make_pdf
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from app.main import create_app


def upload(client, data=None, filename="sample.pdf"):
    if data is None:
        data = make_pdf("A car warranty.")
    return client.post("/documents", files={"file": (filename, data, "application/pdf")})


def test_upload_search_and_page_citations(client):
    response = upload(
        client, make_pdf("", "Banana cultivation.", "The car warranty lasts two years.")
    )
    assert response.status_code == 201
    doc = response.json()
    assert doc["pages"] == 3 and doc["chunks"] == 2
    result = client.post("/query", json={"query": "car", "top_k": 1}).json()["results"][0]
    assert result["document_id"] == doc["document_id"]
    assert result["page"] == 3
    assert result["score"] == pytest.approx(1)
    assert "two years" in result["text"]
    assert client.get("/documents").json() == [doc]


def test_duplicate_is_idempotent(client):
    first = upload(client).json()
    second = upload(client, filename="renamed.pdf")
    assert second.status_code == 200
    assert second.json() == first
    assert len(client.get("/documents").json()) == 1


def test_document_filter_and_score_threshold(client):
    car = upload(client).json()
    upload(client, make_pdf("Banana farming."), "banana.pdf")
    response = client.post(
        "/query", json={"query": "banana", "document_id": car["document_id"], "min_score": 0.5}
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
    hits = client.post(
        "/query", json={"query": "banana", "document_id": car["document_id"]}
    ).json()["results"]
    assert {hit["document_id"] for hit in hits} == {car["document_id"]}


def test_restart_preserves_index(tmp_path, embedder):
    with TestClient(create_app(tmp_path, embedder)) as first:
        doc = upload(first).json()
    with TestClient(create_app(tmp_path, embedder)) as second:
        hit = second.post("/query", json={"query": "car"}).json()["results"][0]
        assert hit["document_id"] == doc["document_id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "x" * 501},
        {"query": "car", "top_k": 0},
        {"query": "car", "top_k": 21},
    ],
)
def test_query_validation(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_empty_index_and_missing_document(client):
    assert client.post("/query", json={"query": "car"}).json()["results"] == []
    assert client.post("/query", json={"query": "car", "document_id": "a" * 64}).status_code == 404


@pytest.mark.parametrize("data", [b"not a PDF", b"%PDF-1.7\nbroken", make_pdf("")])
def test_invalid_and_empty_pdf(client, data):
    response = upload(client, data)
    assert response.status_code == 422
    assert client.get("/documents").json() == []


def test_encrypted_pdf(client):
    writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf("secret"))))
    writer.encrypt("password")
    stream = BytesIO()
    writer.write(stream)
    response = upload(client, stream.getvalue())
    assert response.status_code == 422
    assert "Encrypted" in response.json()["detail"]


def test_oversized_upload(client, monkeypatch):
    monkeypatch.setattr("app.routers.documents.MAX_UPLOAD_BYTES", 10)
    assert upload(client).status_code == 413


def test_database_failure_rolls_back_document_and_allows_retry(client):
    store = client.app.state.document_service.store
    with store.connect() as db:
        db.execute("""CREATE TRIGGER reject_chunk BEFORE INSERT ON chunks
                      BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
    assert upload(client).status_code == 503
    assert client.get("/documents").json() == []
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        db.execute("DROP TRIGGER reject_chunk")
    assert upload(client).status_code == 201


def test_health_and_openapi(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert set(client.get("/openapi.json").json()["paths"]) == {
        "/health",
        "/documents",
        "/query",
        "/answer",
    }
