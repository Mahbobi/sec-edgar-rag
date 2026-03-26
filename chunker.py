"""SEC filing-aware document chunker.

Parses structured SEC 10-K/10-Q filings into semantically meaningful chunks,
respecting section boundaries (Item 1, 1A, 7, etc.) and using tiktoken for
accurate token measurement. Each chunk carries rich metadata for retrieval
filtering and citation.
"""

import re
from dataclasses import dataclass

import tiktoken

import config

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Regex to detect section headers in the body text of SEC filings.
# Matches patterns like "Item 1.", "Item 1A.", "Item 7A.", "ITEM 1.", etc.
# These appear both in the table of contents and as actual section headings.
_SECTION_PATTERN = re.compile(
    r"^(?:PART\s+(?:I{1,3}|IV),?\s*)?Item\s+(\d+[A-C]?)\.?\s*[|\.\s]",
    re.IGNORECASE,
)

# Map section IDs to human-readable names
SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity",
    "6": "Reserved",
    "7": "Management Discussion & Analysis",
    "7A": "Quantitative and Qualitative Disclosures",
    "8": "Financial Statements",
    "9": "Changes in Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Foreign Jurisdictions Disclosure",
    "10": "Directors and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Related Transactions",
    "14": "Principal Accountant Fees",
    "15": "Exhibits and Financial Statements",
    "16": "Form 10-K Summary",
}


@dataclass(frozen=True)
class Chunk:
    """A text chunk with its metadata."""

    text: str
    section_id: str
    section_name: str
    chunk_index: int
    token_count: int


@dataclass(frozen=True)
class FilingMetadata:
    """Metadata parsed from a filing's header."""

    company: str
    ticker: str
    filing_type: str
    filing_date: str
    source_file: str


def parse_filing_header(content: str, filename: str) -> tuple[FilingMetadata, str]:
    """Parse the structured header and return metadata + body text.

    The header format is:
        Company: ...
        Ticker: ...
        Filing Type: ...
        Filing Date: ...
        ...
        ============
        <body>

    Args:
        content: Full file content.
        filename: Original filename for fallback parsing.

    Returns:
        Tuple of (FilingMetadata, body_text).
    """
    header_fields = {}
    body_start = 0

    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("=" * 10):
            body_start = i + 1
            break

        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if key in ("company", "ticker", "filing type", "filing date", "cik"):
                header_fields[key] = value

    body = "\n".join(lines[body_start:])

    # Extract filing type - normalize to 10-K or 10-Q
    raw_type = header_fields.get("filing type", "")
    if "10-K" in raw_type or "10-K" in filename:
        filing_type = "10-K"
    elif "10-Q" in raw_type or "10-Q" in filename:
        filing_type = "10-Q"
    else:
        filing_type = raw_type

    # Parse ticker from filename as fallback
    ticker = header_fields.get("ticker", "")
    if not ticker:
        parts = filename.split("_")
        if parts:
            ticker = parts[0]

    # Parse date from filename as fallback
    filing_date = header_fields.get("filing date", "")
    if not filing_date:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if date_match:
            filing_date = date_match.group(1)

    return (
        FilingMetadata(
            company=header_fields.get("company", ticker),
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            source_file=filename,
        ),
        body,
    )


def _find_sections(body: str) -> list[tuple[str, int]]:
    """Find section boundaries in the filing body.

    Returns list of (section_id, line_index) tuples, sorted by line_index.
    Only keeps the LAST occurrence of each section ID (the actual content
    heading, not the table of contents entry).
    """
    lines = body.split("\n")
    section_positions: dict[str, int] = {}

    for i, line in enumerate(lines):
        match = _SECTION_PATTERN.match(line.strip())
        if match:
            section_id = match.group(1).upper()
            # Keep the last occurrence (body heading, not TOC)
            section_positions[section_id] = i

    # Sort by position
    return sorted(section_positions.items(), key=lambda x: x[1])


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_ENCODER.encode(text))


def _split_text_into_chunks(
    text: str,
    max_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into chunks of approximately max_tokens with overlap.

    Prefers splitting at paragraph boundaries, then sentence boundaries,
    then whitespace.
    """
    if _count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    # Split by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        candidate_tokens = _count_tokens(candidate)

        if candidate_tokens <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # If a single paragraph exceeds max_tokens, split it by sentences
            if _count_tokens(para) > max_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    sub_candidate = (
                        (sub_chunk + " " + sent).strip() if sub_chunk else sent
                    )
                    if _count_tokens(sub_candidate) <= max_tokens:
                        sub_chunk = sub_candidate
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        # If a single sentence exceeds max_tokens, just take it
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap between consecutive chunks
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = _ENCODER.encode(chunks[i - 1])
            overlap_text = _ENCODER.decode(prev_tokens[-overlap_tokens:])
            overlapped.append(overlap_text.strip() + "\n\n" + chunks[i])
        chunks = overlapped

    return chunks


def chunk_filing(content: str, filename: str) -> tuple[FilingMetadata, list[Chunk]]:
    """Parse and chunk a single SEC filing.

    Args:
        content: Full file content (header + body).
        filename: Original filename.

    Returns:
        Tuple of (metadata, list of Chunk objects).
    """
    metadata, body = parse_filing_header(content, filename)

    # Clean up XBRL noise at the start of the body
    # XBRL data appears as a long concatenated string without spaces
    cleaned_lines = []
    for line in body.split("\n"):
        # Skip XBRL-like lines (very long with no spaces, or contain typical XBRL patterns)
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if len(stripped) > 200 and " " not in stripped[:100]:
            continue
        if stripped.startswith("http://fasb.org") or stripped.startswith("xbrli:"):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # Find section boundaries
    sections = _find_sections(body)
    body_lines = body.split("\n")

    chunks: list[Chunk] = []

    if not sections:
        # No sections detected - chunk the whole body
        text_chunks = _split_text_into_chunks(body)
        for i, text in enumerate(text_chunks):
            tc = _count_tokens(text)
            if tc < 20:  # Skip tiny chunks
                continue
            chunks.append(Chunk(
                text=text,
                section_id="",
                section_name="Full Document",
                chunk_index=i,
                token_count=tc,
            ))
        return metadata, chunks

    # Process each section
    for idx, (section_id, start_line) in enumerate(sections):
        # Determine end of this section
        if idx + 1 < len(sections):
            end_line = sections[idx + 1][1]
        else:
            end_line = len(body_lines)

        section_text = "\n".join(body_lines[start_line:end_line]).strip()
        section_name = SECTION_NAMES.get(section_id, f"Item {section_id}")

        # Split section into chunks
        text_chunks = _split_text_into_chunks(section_text)

        for i, text in enumerate(text_chunks):
            tc = _count_tokens(text)
            if tc < 20:  # Skip tiny chunks
                continue
            chunks.append(Chunk(
                text=text,
                section_id=section_id,
                section_name=section_name,
                chunk_index=i,
                token_count=tc,
            ))

    return metadata, chunks
