"""Retrieval pipeline: query → embed → search → re-rank → return.

Handles company/section detection from the query, semantic search,
metadata boosting, and diversity re-ranking to ensure balanced
coverage across companies for comparison questions.
"""

import re
from collections import defaultdict

import tiktoken

import config
from embeddings import Embedder
from vector_store import VectorStore, ChunkRecord

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Section keyword mapping for query-based section boosting
_SECTION_KEYWORDS = {
    "risk": ["1A"],
    "cybersecurity": ["1C"],
    "revenue": ["7", "8"],
    "financial": ["7", "8"],
    "growth": ["7"],
    "profit": ["7", "8"],
    "earnings": ["7", "8"],
    "regulatory": ["1A", "3"],
    "legal": ["3"],
    "business": ["1"],
    "competition": ["1", "1A"],
    "strategy": ["1", "7"],
    "outlook": ["7"],
    "guidance": ["7"],
    "compensation": ["11"],
    "governance": ["10"],
}


class Retriever:
    """SEC filing retrieval pipeline with diversity re-ranking."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._store = store or VectorStore()
        self._ticker_map = self._store.get_ticker_company_map()
        self._all_tickers = set(self._ticker_map.keys())
        self._company_to_ticker = {
            v.lower(): k for k, v in self._ticker_map.items()
        }

    def _detect_companies(self, query: str) -> list[str]:
        """Detect company tickers mentioned in the query.

        Matches both tickers (AAPL, NVDA) and company names
        (Apple, NVIDIA, JPMorgan). Requires tickers >=3 chars to avoid
        false positives from short tickers like BA, MS, HD, etc.
        """
        query_lower = query.lower()
        detected = set()

        # Match tickers — only if they appear as uppercase words in original query
        # and are at least 3 chars (avoids false positives: BA, MS, HD, T, V, etc.)
        for ticker in self._all_tickers:
            if len(ticker) >= 3 and re.search(rf"\b{re.escape(ticker)}\b", query):
                detected.add(ticker)

        # Match company names (case-insensitive, word-boundary)
        # Build robust matching for common names
        _NAME_ALIASES: dict[str, list[str]] = {
            "JPM": ["jpmorgan", "jp morgan", "chase"],
            "BAC": ["bank of america"],
            "GS": ["goldman sachs", "goldman"],
            "MS": ["morgan stanley"],
            "BRK": ["berkshire"],
            "META": ["meta", "facebook"],
            "GOOG": ["google", "alphabet"],
            "AMZN": ["amazon"],
            "AAPL": ["apple"],
            "MSFT": ["microsoft"],
            "TSLA": ["tesla"],
            "NVDA": ["nvidia"],
            "NFLX": ["netflix"],
            "CRM": ["salesforce"],
            "ORCL": ["oracle"],
            "INTC": ["intel"],
            "AMD": ["amd"],
            "ADBE": ["adobe"],
            "IBM": ["ibm"],
            "CSCO": ["cisco"],
            "PFE": ["pfizer"],
            "JNJ": ["johnson & johnson", "johnson and johnson", "j&j"],
            "MRK": ["merck"],
            "LLY": ["eli lilly", "lilly"],
            "ABBV": ["abbvie"],
            "UNH": ["unitedhealth", "united health"],
            "TMO": ["thermo fisher"],
            "XOM": ["exxon"],
            "CVX": ["chevron"],
            "BA": ["boeing"],
            "LMT": ["lockheed"],
            "RTX": ["raytheon"],
            "GE": ["general electric"],
            "CAT": ["caterpillar"],
            "DE": ["deere", "john deere"],
            "HD": ["home depot"],
            "WMT": ["walmart"],
            "COST": ["costco"],
            "TGT": ["target"],
            "MCD": ["mcdonald"],
            "SBUX": ["starbucks"],
            "NKE": ["nike"],
            "KO": ["coca-cola", "coca cola", "coke"],
            "PEP": ["pepsi", "pepsico"],
            "PG": ["procter", "procter & gamble"],
            "DIS": ["disney", "walt disney"],
            "CMCSA": ["comcast"],
            "T": ["at&t"],
            "VZ": ["verizon"],
            "V": ["visa"],
            "MA": ["mastercard"],
            "AXP": ["american express", "amex"],
            "BLK": ["blackrock"],
            "UPS": ["ups"],
        }

        for ticker, aliases in _NAME_ALIASES.items():
            if ticker in self._all_tickers:
                for alias in aliases:
                    if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                        detected.add(ticker)
                        break

        return sorted(detected)

    def _detect_sections(self, query: str) -> list[str]:
        """Detect relevant SEC filing sections from query keywords."""
        query_lower = query.lower()
        sections = set()
        for keyword, section_ids in _SECTION_KEYWORDS.items():
            if keyword in query_lower:
                sections.update(section_ids)
        return sorted(sections)

    def _diversity_rerank(
        self,
        candidates: list[ChunkRecord],
        detected_companies: list[str],
        max_per_company: int = config.MAX_CHUNKS_PER_COMPANY,
        max_total: int = config.TOP_K_FINAL,
        token_budget: int = config.CONTEXT_TOKEN_BUDGET,
    ) -> list[ChunkRecord]:
        """Re-rank candidates for diversity across companies and sections.

        For multi-company queries, ensures balanced representation.
        Uses a greedy selection algorithm that picks the highest-scoring
        chunk that doesn't over-represent any company.
        """
        is_comparison = len(detected_companies) > 1

        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        selected: list[ChunkRecord] = []
        company_counts: dict[str, int] = defaultdict(int)
        section_per_company: dict[str, set[str]] = defaultdict(set)
        total_tokens = 0

        def _add_chunk(chunk: ChunkRecord) -> bool:
            """Try to add a chunk. Returns True if added."""
            nonlocal total_tokens
            chunk_tokens = len(_ENCODER.encode(chunk.text))
            if total_tokens + chunk_tokens > token_budget:
                return False
            selected.append(chunk)
            company_counts[chunk.ticker] += 1
            if chunk.section_id:
                section_per_company[chunk.ticker].add(chunk.section_id)
            total_tokens += chunk_tokens
            return True

        selected_ids: set[int] = set()

        if is_comparison:
            # Phase 1: Guarantee at least 2 chunks per detected company
            for ticker in detected_companies:
                company_chunks = sorted(
                    [c for c in sorted_candidates if c.ticker == ticker],
                    key=lambda c: c.score,
                    reverse=True,
                )
                added = 0
                for chunk in company_chunks:
                    if added >= 2:
                        break
                    if chunk.id not in selected_ids:
                        if _add_chunk(chunk):
                            selected_ids.add(chunk.id)
                            added += 1

            # Phase 2: Fill remaining slots with highest-scoring chunks
            for chunk in sorted_candidates:
                if len(selected) >= max_total:
                    break
                if chunk.id in selected_ids:
                    continue
                if company_counts[chunk.ticker] >= max_per_company:
                    continue
                if _add_chunk(chunk):
                    selected_ids.add(chunk.id)
        else:
            # Single-company or general query — simple top-k with diversity
            for chunk in sorted_candidates:
                if len(selected) >= max_total:
                    break
                if chunk.id in selected_ids:
                    continue

                chunk_tokens = len(_ENCODER.encode(chunk.text))
                if total_tokens + chunk_tokens > token_budget:
                    continue

                # Limit same section repetition
                if (
                    chunk.section_id
                    and chunk.section_id in section_per_company.get(chunk.ticker, set())
                    and company_counts[chunk.ticker] >= 3
                ):
                    continue

                if _add_chunk(chunk):
                    selected_ids.add(chunk.id)

        return selected

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[ChunkRecord], dict]:
        """Full retrieval pipeline for a business question.

        Args:
            query: Natural language business question.
            top_k: Override for final number of results.

        Returns:
            Tuple of (ranked_chunks, retrieval_info_dict).
        """
        top_k = top_k or config.TOP_K_FINAL

        # Step 1: Detect companies and sections from query
        detected_companies = self._detect_companies(query)
        detected_sections = self._detect_sections(query)

        # Step 2: Embed query
        query_embedding = self._embedder.embed_single(query)

        # Step 3: Initial search — broad search plus targeted per-company search
        candidates = self._store.search(
            query_embedding,
            top_k=config.TOP_K_CANDIDATES,
        )

        # For detected companies, also do targeted searches to ensure coverage
        if detected_companies:
            existing_ids = {c.id for c in candidates}
            for ticker in detected_companies:
                targeted = self._store.search(
                    query_embedding,
                    top_k=10,
                    ticker_filter=[ticker],
                )
                for chunk in targeted:
                    if chunk.id not in existing_ids:
                        candidates.append(chunk)
                        existing_ids.add(chunk.id)

        # Step 4: Metadata boosting
        boosted = []
        for chunk in candidates:
            score = chunk.score

            # Boost chunks from detected companies
            if detected_companies and chunk.ticker in detected_companies:
                score += 0.15

            # Boost chunks from relevant sections
            if detected_sections and chunk.section_id in detected_sections:
                score += 0.10

            boosted.append(ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                company=chunk.company,
                ticker=chunk.ticker,
                filing_type=chunk.filing_type,
                filing_date=chunk.filing_date,
                section_id=chunk.section_id,
                section_name=chunk.section_name,
                chunk_index=chunk.chunk_index,
                source_file=chunk.source_file,
                score=score,
            ))

        # Step 5: Diversity re-ranking
        results = self._diversity_rerank(
            boosted,
            detected_companies,
            max_total=top_k,
        )

        info = {
            "detected_companies": detected_companies,
            "detected_sections": detected_sections,
            "initial_candidates": len(candidates),
            "final_results": len(results),
            "companies_in_results": sorted(
                set(r.ticker for r in results)
            ),
        }

        return results, info
