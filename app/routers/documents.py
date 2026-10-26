from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status

from app.constants import MAX_UPLOAD_BYTES
from app.pdf import InvalidPdfError
from app.schemas import AnswerResponse, Document, SearchRequest, SearchResponse
from app.services.document_service import (
    AnswersDisabledError,
    DocumentNotFoundError,
    DocumentService,
    ProcessingError,
)

router = APIRouter()


def get_document_service(request: Request) -> DocumentService:
    return cast(DocumentService, request.app.state.document_service)


ServiceDependency = Annotated[DocumentService, Depends(get_document_service)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/documents")
def list_documents(service: ServiceDependency) -> list[Document]:
    try:
        return service.list_documents()
    except ProcessingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.post("/documents", status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile, response: Response, service: ServiceDependency) -> Document:
    with file.file:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, "PDF exceeds the 20 MiB upload limit."
        )

    # Store only the filename, without any directories sent by the client.
    filename = (file.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    filename = filename[:255] or "document.pdf"
    try:
        document, created = service.upload_pdf(data, filename)
    except InvalidPdfError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ProcessingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if not created:
        response.status_code = status.HTTP_200_OK
    return document


@router.post("/query")
def search_documents(request: SearchRequest, service: ServiceDependency) -> SearchResponse:
    try:
        return service.search(request)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ProcessingError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.post("/answer")
def answer_question(request: SearchRequest, service: ServiceDependency) -> AnswerResponse:
    try:
        return service.answer(request)
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (AnswersDisabledError, ProcessingError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
