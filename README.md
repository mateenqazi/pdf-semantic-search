# PDF Semantic Search

A local Python API built with FastAPI and Pydantic for the PDF RAG assessment.
Upload a text PDF, then search its content using natural-language questions in
German or English. Upload and search work without an OpenAI API key.

## Tech stack

| Tool | Purpose |
| --- | --- |
| Python 3.12, FastAPI, Uvicorn | Application language, API framework, and web server |
| Pydantic | Request validation and response models |
| pypdf and Tokenizers | Extract PDF text and split it at token boundaries |
| FastEmbed, ONNX Runtime, Jina embeddings | Generate German/English text embeddings locally on the CPU |
| SQLite | Store document metadata, text chunks, and embedding vectors in one local database file |
| NumPy | Rank passages by cosine similarity |
| OpenAI SDK (optional) | Generate answers from retrieved passages with source citations |
| uv, pytest, Ruff | Install locked dependencies, run tests, and check code style |

## Setup

With Git and uv installed, run:

```bash
git clone https://github.com/mateenqazi/pdf-semantic-search.git
cd pdf-semantic-search
uv python install 3.12
uv sync --locked
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`uv sync --locked` creates the virtual environment and installs the locked dependencies.
SQLite is created automatically; no database server is needed.

The first start needs internet access to download the embedding model (about 640 MB).
Wait for `Application startup complete`, then open:

- [Health check](http://127.0.0.1:8000/health): should return `{"status":"ok"}`.
- [Swagger UI](http://127.0.0.1:8000/docs): upload a PDF and try the API.

Stop the server with Ctrl+C.

## Upload and search

In Swagger, expand an endpoint and select **Try it out**, then **Execute**:

1. Use `POST /documents` to upload a PDF in the required `file` field.
2. Copy the returned `document_id` if you want to search only that PDF.
3. Use `POST /query` with this minimal JSON body:

```json
{
  "query": "How long is the standard warranty for a new Volkswagen vehicle?"
}
```

The response contains matching passages with document ID, filename, page, chunk ID,
text, and similarity score. Pages start at 1; highest-scoring passages come first.

Both `POST /query` and `POST /answer` accept these JSON fields:

| Parameter | Required? | Default | Allowed values / purpose |
| --- | --- | --- | --- |
| `query` | **Yes** | None | Nonblank question, 1–500 characters after trimming whitespace |
| `top_k` | No | `5` | Integer from 1 to 20; maximum passages retrieved |
| `document_id` | No | `null` | ID returned by upload; omit or use `null` to search all documents |
| `min_score` | No | `-1.0` | Minimum cosine similarity, from -1 to 1 |

Only `document_id` accepts `null`. Invalid input returns 422.
Similarity scores are not confidence percentages.

| Endpoint | Purpose |
| --- | --- |
| `POST /documents` | Upload and index a PDF; 201 for a new document, 200 for a duplicate |
| `GET /documents` | List uploaded documents |
| `POST /query` | Search for matching passages |
| `POST /answer` | Generate an optional answer with source citations |
| `GET /health` | Return `{"status":"ok"}` after startup |

Other errors: 413 for oversized uploads, 404 for an unknown document ID during search,
and 503 for processing failures or unavailable answer generation.

## Optional OpenAI answers

To enable answers, stop the server and run in the same terminal (macOS/Linux):

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-5.4"
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Call `POST /answer` with the same body as `/query`. It returns `query`, `answer`, and
`sources`; citation `[1]` refers to the first source. Without a key, it returns 503.
Set the key in the terminal; `.env` files are not loaded automatically.

This sends the question and retrieved passages to OpenAI and incurs API charges.

## Approach

1. Extract text from each PDF page, keeping its page number.
2. Split the text into 256-token chunks with a 48-token overlap to retain context.
3. Embed the chunks locally with `jinaai/jina-embeddings-v2-base-de` and save them in SQLite.
4. Embed the question, rank stored vectors by cosine similarity, and return the top
   matching passages with page references.
5. For optional `/answer` requests, send those passages to OpenAI to generate a cited answer.

Content hashes prevent duplicate uploads. Database transactions prevent partial saves.

The database is `.data/index.sqlite3`; the model cache is `.cache/models`.
Both are reused after restarting the server.

## Project structure

- `app/main.py`: application setup and startup/shutdown.
- `app/routers/`: endpoints and HTTP errors.
- `app/services/`: upload, search, and answer workflows.
- `app/repositories/`: SQLite storage and vector search.
- `app/schemas.py` and `app/constants.py`: validation models and shared settings.
- `app/pdf.py`, `app/embeddings.py`, and `app/answers.py`: PDF, embedding, and OpenAI logic.
- `tests/`: API/storage checks and opt-in real-model tests.

## Tests

```bash
uv run pytest -q                         # 16 tests; 2 real-model tests skipped
RUN_INTEGRATION=1 uv run pytest -q       # All 18 tests, including the real model
uv run ruff check .
uv run ruff format --check .
```

Tests cover upload/search, validation, persistence, duplicates, and rollback.
Real-model tests check bilingual retrieval and chunk coverage. OpenAI answers are
checked manually; automated tests make no paid API calls.

## Limitations

- Text PDFs only; no OCR for scanned pages. Maximum upload: 20 MiB and 300 pages,
  with extracted-text and chunk limits defined in `app/constants.py`.
- Indexing is synchronous and can delay searches. Exact vector search suits small collections.
- Unrelated questions can still return weak matches; there is no calibrated relevance cutoff.
