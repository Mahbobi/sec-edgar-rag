"""SQLite-backed vector store with numpy cosine similarity search.

Designed for corpus sizes up to ~20k chunks where brute-force cosine
similarity is fast enough (<100ms). All embeddings are loaded into a
single numpy matrix at query time for vectorized search.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class ChunkRecord:
    """A stored chunk with its metadata and optional relevance score."""

    id: int
    text: str
    company: str
    ticker: str
    filing_type: str
    filing_date: str
    section_id: str
    section_name: str
    chunk_index: int
    source_file: str
    score: float = 0.0


class VectorStore:
    """SQLite + numpy vector store for document chunks."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or config.DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self._embeddings_matrix: np.ndarray | None = None
        self._row_ids: list[int] = []

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                company TEXT NOT NULL,
                ticker TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                section_id TEXT NOT NULL DEFAULT '',
                section_name TEXT NOT NULL DEFAULT '',
                chunk_index INTEGER NOT NULL DEFAULT 0,
                source_file TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker ON chunks(ticker)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_filing_type ON chunks(filing_type)"
        )
        self._conn.commit()

    def insert_chunks(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata_list: list[dict],
    ) -> None:
        """Insert a batch of chunks with their embeddings and metadata.

        Args:
            texts: Chunk text content.
            embeddings: numpy array of shape (N, dim).
            metadata_list: List of dicts with keys: company, ticker,
                filing_type, filing_date, section_id, section_name,
                chunk_index, source_file.
        """
        rows = []
        for text, emb, meta in zip(texts, embeddings, metadata_list):
            emb_blob = emb.astype(np.float32).tobytes()
            rows.append((
                text,
                emb_blob,
                meta.get("company", ""),
                meta.get("ticker", ""),
                meta.get("filing_type", ""),
                meta.get("filing_date", ""),
                meta.get("section_id", ""),
                meta.get("section_name", ""),
                meta.get("chunk_index", 0),
                meta.get("source_file", ""),
            ))

        self._conn.executemany(
            """INSERT INTO chunks
               (text, embedding, company, ticker, filing_type, filing_date,
                section_id, section_name, chunk_index, source_file)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        self._embeddings_matrix = None  # Invalidate cache

    def _load_embeddings(self) -> None:
        """Load all embeddings and ticker info into memory for fast search."""
        cursor = self._conn.execute(
            "SELECT id, embedding, ticker FROM chunks ORDER BY id"
        )
        rows = cursor.fetchall()

        if not rows:
            self._embeddings_matrix = np.zeros((0, config.EMBEDDING_DIM))
            self._row_ids = []
            self._row_tickers: list[str] = []
            return

        self._row_ids = [r["id"] for r in rows]
        self._row_tickers = [r["ticker"] for r in rows]
        emb_list = [
            np.frombuffer(r["embedding"], dtype=np.float32)
            for r in rows
        ]
        self._embeddings_matrix = np.vstack(emb_list)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 40,
        ticker_filter: list[str] | None = None,
    ) -> list[ChunkRecord]:
        """Search for the most similar chunks to the query embedding.

        When ticker_filter is provided, filters BEFORE ranking so that
        top-k results are all from the specified companies.

        Args:
            query_embedding: 1D numpy array of shape (dim,).
            top_k: Number of results to return.
            ticker_filter: Optional list of tickers to restrict results to.

        Returns:
            List of ChunkRecord sorted by descending similarity score.
        """
        if self._embeddings_matrix is None:
            self._load_embeddings()

        if len(self._row_ids) == 0:
            return []

        # Cosine similarity (embeddings are already L2-normalized)
        scores = self._embeddings_matrix @ query_embedding

        # Apply ticker filter BEFORE ranking (mask out non-matching scores)
        if ticker_filter:
            ticker_set = set(ticker_filter)
            mask = np.array(
                [t in ticker_set for t in self._row_tickers], dtype=bool
            )
            filtered_scores = np.where(mask, scores, -1.0)
        else:
            filtered_scores = scores

        # Get top-k indices
        num_results = min(top_k, int(np.sum(filtered_scores > -1.0)))
        if num_results == 0:
            return []

        if len(filtered_scores) <= top_k:
            top_indices = np.argsort(filtered_scores)[::-1][:num_results]
        else:
            top_indices = np.argpartition(filtered_scores, -num_results)[-num_results:]
            top_indices = top_indices[np.argsort(filtered_scores[top_indices])[::-1]]

        # Fetch full records for top results
        results = []
        for idx in top_indices:
            if filtered_scores[idx] <= -1.0:
                continue
            row_id = self._row_ids[idx]
            row = self._conn.execute(
                """SELECT id, text, company, ticker, filing_type, filing_date,
                          section_id, section_name, chunk_index, source_file
                   FROM chunks WHERE id = ?""",
                (row_id,),
            ).fetchone()

            if row is None:
                continue

            results.append(ChunkRecord(
                id=row["id"],
                text=row["text"],
                company=row["company"],
                ticker=row["ticker"],
                filing_type=row["filing_type"],
                filing_date=row["filing_date"],
                section_id=row["section_id"],
                section_name=row["section_name"],
                chunk_index=row["chunk_index"],
                source_file=row["source_file"],
                score=float(scores[idx]),
            ))

        return results

    def get_stats(self) -> dict:
        """Return index statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        companies = self._conn.execute(
            "SELECT DISTINCT ticker FROM chunks ORDER BY ticker"
        ).fetchall()
        filing_types = self._conn.execute(
            "SELECT filing_type, COUNT(*) FROM chunks GROUP BY filing_type"
        ).fetchall()

        return {
            "total_chunks": total,
            "companies": [r[0] for r in companies],
            "filing_types": {r[0]: r[1] for r in filing_types},
        }

    def get_all_tickers(self) -> list[str]:
        """Return all unique tickers in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT ticker FROM chunks ORDER BY ticker"
        ).fetchall()
        return [r[0] for r in rows]

    def get_ticker_company_map(self) -> dict[str, str]:
        """Return a mapping of ticker -> company name."""
        rows = self._conn.execute(
            "SELECT DISTINCT ticker, company FROM chunks"
        ).fetchall()
        return {r["ticker"]: r["company"] for r in rows}

    def clear(self) -> None:
        """Delete all chunks from the store."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self._embeddings_matrix = None
        self._row_ids = []

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
