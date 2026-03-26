"""Prompt template for SEC EDGAR RAG system.

Constructs the system and user messages for a single LLM call,
injecting retrieved filing excerpts with rich metadata headers.
"""

from vector_store import ChunkRecord

SYSTEM_PROMPT = """\
You are a financial analyst assistant specializing in SEC EDGAR filings \
(10-K annual reports and 10-Q quarterly reports). You provide accurate, \
well-structured answers grounded exclusively in the provided filing excerpts.

Rules:
1. Base your answer ONLY on the provided filing excerpts. If the excerpts \
don't contain enough information to fully answer, say so explicitly and \
answer with what is available.
2. Cite specific filings when making claims using the format: \
(Company, Filing Type, Filing Date).
3. For comparison questions, organize your answer with clear per-company \
sections or a comparison table.
4. Include specific numbers, percentages, dates, and direct references \
from the filings when available.
5. Distinguish between 10-K (annual) and 10-Q (quarterly) data when the \
distinction is relevant.
6. If excerpts from different time periods are available, highlight trends \
or changes over time.
7. Structure your response with markdown headings and bullet points for \
readability."""


def build_user_message(question: str, contexts: list[ChunkRecord]) -> str:
    """Build the user message with retrieved excerpts and the question.

    Args:
        question: The user's natural-language business question.
        contexts: List of ChunkRecord objects from retrieval.

    Returns:
        Formatted user message string.
    """
    parts = ["## Filing Excerpts\n"]

    for i, ctx in enumerate(contexts, 1):
        header = (
            f"### Excerpt {i} — {ctx.company} ({ctx.ticker}) | "
            f"{ctx.filing_type} | {ctx.filing_date}"
        )
        if ctx.section_name:
            header += f" | {ctx.section_name}"

        parts.append(header)
        parts.append(ctx.text.strip())
        parts.append("")  # blank line

    parts.append("---\n")
    parts.append("## Question")
    parts.append(question)
    parts.append(
        "\nProvide a comprehensive, well-structured answer based on "
        "the filing excerpts above."
    )

    return "\n".join(parts)


def build_messages(
    question: str, contexts: list[ChunkRecord]
) -> tuple[str, str]:
    """Build the full prompt (system message + user message).

    Args:
        question: The user's question.
        contexts: Retrieved chunks.

    Returns:
        Tuple of (system_message, user_message).
    """
    return SYSTEM_PROMPT, build_user_message(question, contexts)
