import re
from typing import Protocol

from openai import OpenAI

from app.constants import (
    CITATION_PATTERN,
    DEFAULT_MODEL,
    INSTRUCTIONS,
    NO_ANSWER_TOKEN,
    NO_PASSAGES_ANSWER,
)
from app.schemas import SearchHit


class AnswerGenerator(Protocol):
    def answer(self, question: str, passages: list[SearchHit]) -> str: ...


def build_prompt(question: str, passages: list[SearchHit]) -> str:
    sources = "\n\n".join(
        f"[{number}] {hit.filename}, page {hit.page}\n{hit.text}"
        for number, hit in enumerate(passages, start=1)
    )
    return f"Passages:\n\n{sources}\n\nQuestion: {question}"


class OpenAIAnswerGenerator:
    def __init__(self, client: OpenAI, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    def answer(self, question: str, passages: list[SearchHit]) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=INSTRUCTIONS,
            input=build_prompt(question, passages),
        )
        if response.status != "completed":
            raise ValueError("The model did not complete its answer.")
        text = response.output_text.strip()
        if not text:
            raise ValueError("The model returned an empty answer.")
        if text == NO_ANSWER_TOKEN:
            return NO_PASSAGES_ANSWER

        citations = re.findall(CITATION_PATTERN, text)
        if not citations or any(
            not citation.isdecimal() or not 1 <= int(citation) <= len(passages)
            for citation in citations
        ):
            raise ValueError("The answer must cite valid source numbers.")
        return text

    def close(self) -> None:
        self.client.close()
