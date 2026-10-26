import re
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from app.constants import MAX_CHARACTERS, MAX_CHUNKS, MAX_PAGES


class InvalidPdfError(ValueError):
    """The PDF cannot be read or contains no usable text."""


@dataclass(frozen=True)
class Chunk:
    page: int
    text: str


def extract_pages(data: bytes) -> list[str]:
    """Keep blank pages so page numbers match the PDF."""
    if b"%PDF-" not in data[:1024]:
        raise InvalidPdfError("The uploaded file is not a PDF.")

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPdfError("Encrypted PDFs are not supported; upload a decrypted copy.")
        if len(reader.pages) > MAX_PAGES:
            raise InvalidPdfError(f"PDF exceeds the {MAX_PAGES}-page limit.")

        pages: list[str] = []
        character_count = 0
        for page in reader.pages:
            # Keep table columns separated when extracting text.
            extracted_text = page.extract_text(extraction_mode="layout") or ""
            text = re.sub(r"\s+", " ", extracted_text).strip()
            character_count += len(text)
            if character_count > MAX_CHARACTERS:
                raise InvalidPdfError("PDF contains too much extracted text.")
            pages.append(text)
    except InvalidPdfError:
        raise
    except Exception as exc:
        raise InvalidPdfError("PDF could not be read; it may be malformed or unsupported.") from exc

    if not any(pages):
        raise InvalidPdfError("PDF contains no extractable text. Scanned PDFs require OCR first.")
    return pages


def extract_chunks(data: bytes, split_text: Callable[[str], list[str]]) -> tuple[int, list[Chunk]]:
    pages = extract_pages(data)
    chunks: list[Chunk] = []
    for page_number, text in enumerate(pages, start=1):
        if not text:
            continue
        chunks.extend(Chunk(page_number, part) for part in split_text(text))
        if len(chunks) > MAX_CHUNKS:
            raise InvalidPdfError(f"PDF exceeds the {MAX_CHUNKS}-chunk limit.")
    return len(pages), chunks
