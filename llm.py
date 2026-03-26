"""LLM integration — single Google Gemini API call for answer generation.

Wraps the Google GenAI SDK to produce a grounded answer from
the retrieved filing excerpts in exactly one API request.
"""

import time
from dataclasses import dataclass

from google import genai
from google.genai import types

import config
from prompt_template import build_messages
from vector_store import ChunkRecord


@dataclass(frozen=True)
class LLMResponse:
    """Response from the LLM with metadata."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    contexts_used: int


def generate_answer(
    question: str,
    contexts: list[ChunkRecord],
    model: str | None = None,
) -> LLMResponse:
    """Generate a grounded answer using a single Gemini API call.

    Args:
        question: The user's business question.
        contexts: Retrieved and ranked filing excerpts.
        model: Optional model override (defaults to config.LLM_MODEL).

    Returns:
        LLMResponse with the answer and usage metadata.
    """
    model_name = model or config.LLM_MODEL
    system_msg, user_msg = build_messages(question, contexts)

    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    start = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=0.2,
        ),
    )
    latency = time.time() - start

    answer_text = response.text if response.text else ""

    # Extract token usage from response metadata
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    return LLMResponse(
        answer=answer_text,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=round(latency, 2),
        contexts_used=len(contexts),
    )
