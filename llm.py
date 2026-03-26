"""LLM integration — single Anthropic API call for answer generation.

Wraps the Anthropic messages API to produce a grounded answer from
the retrieved filing excerpts in exactly one API request.
"""

import time
from dataclasses import dataclass

import anthropic

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
    """Generate a grounded answer using a single Anthropic API call.

    Args:
        question: The user's business question.
        contexts: Retrieved and ranked filing excerpts.
        model: Optional model override (defaults to config.LLM_MODEL).

    Returns:
        LLMResponse with the answer and usage metadata.

    Raises:
        anthropic.APIError: If the API call fails.
    """
    model = model or config.LLM_MODEL
    system_msg, user_msg = build_messages(question, contexts)

    # Let the client discover the API key from env/config automatically.
    # Only pass explicitly if set in our config.
    kwargs = {}
    if config.ANTHROPIC_API_KEY:
        kwargs["api_key"] = config.ANTHROPIC_API_KEY
    client = anthropic.Anthropic(**kwargs)

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = time.time() - start

    answer_text = response.content[0].text if response.content else ""

    return LLMResponse(
        answer=answer_text,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_seconds=round(latency, 2),
        contexts_used=len(contexts),
    )
