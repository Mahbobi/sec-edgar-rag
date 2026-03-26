"""Indexing pipeline: load corpus → chunk → embed → store.

Reads the SEC EDGAR corpus from data/corpus/, parses each filing with
section-aware chunking, generates embeddings, and stores everything
in the SQLite vector store.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import config
from chunker import chunk_filing, FilingMetadata
from embeddings import Embedder
from vector_store import VectorStore


def load_manifest(corpus_dir: Path) -> dict:
    """Load manifest.json from the corpus directory."""
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {corpus_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_corpus(
    corpus_dir: Path | None = None,
    db_path: Path | None = None,
    force_reindex: bool = False,
) -> dict:
    """Build the vector index from the SEC filing corpus.

    Args:
        corpus_dir: Path to the corpus directory containing .txt files.
        db_path: Path for the SQLite database.
        force_reindex: If True, delete existing index before rebuilding.

    Returns:
        Dict with indexing statistics.
    """
    corpus_dir = corpus_dir or config.DATA_DIR
    db_path = db_path or config.DB_PATH

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Please extract edgar_corpus.zip into data/corpus/"
        )

    # Load manifest
    manifest = load_manifest(corpus_dir)
    file_list = manifest.get("files", [])

    print(f"Corpus: {manifest.get('corpus', 'Unknown')}")
    print(f"Files listed in manifest: {len(file_list)}")
    print(f"Filing types: {manifest.get('filing_types', {})}")

    # Initialize components
    embedder = Embedder()
    store = VectorStore(db_path)

    if force_reindex:
        print("Clearing existing index...")
        store.clear()

    # Check if already indexed
    existing_stats = store.get_stats()
    if existing_stats["total_chunks"] > 0 and not force_reindex:
        print(
            f"Index already contains {existing_stats['total_chunks']} chunks. "
            "Use --force to reindex."
        )
        return existing_stats

    start_time = time.time()
    total_chunks = 0
    total_files = 0
    company_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    errors: list[str] = []

    for i, filename in enumerate(file_list):
        filepath = corpus_dir / filename
        if not filepath.exists():
            errors.append(f"File not found: {filename}")
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            metadata, chunks = chunk_filing(content, filename)

            if not chunks:
                errors.append(f"No chunks produced from: {filename}")
                continue

            # Prepare data for batch insertion
            texts = [c.text for c in chunks]
            meta_list = [
                {
                    "company": metadata.company,
                    "ticker": metadata.ticker,
                    "filing_type": metadata.filing_type,
                    "filing_date": metadata.filing_date,
                    "section_id": c.section_id,
                    "section_name": c.section_name,
                    "chunk_index": c.chunk_index,
                    "source_file": filename,
                }
                for c in chunks
            ]

            # Batch embed
            embeddings = embedder.embed_batched(texts)

            # Store
            store.insert_chunks(texts, embeddings, meta_list)

            total_chunks += len(chunks)
            total_files += 1
            company_counts[metadata.ticker] = (
                company_counts.get(metadata.ticker, 0) + len(chunks)
            )
            for c in chunks:
                key = c.section_name or "Unknown"
                section_counts[key] = section_counts.get(key, 0) + 1

            # Update company mapping
            config.TICKER_TO_COMPANY[metadata.ticker] = metadata.company

            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(file_list):
                elapsed = time.time() - start_time
                print(
                    f"  [{i + 1}/{len(file_list)}] "
                    f"{total_chunks} chunks from {total_files} files "
                    f"({elapsed:.1f}s)"
                )

        except Exception as e:
            errors.append(f"Error processing {filename}: {e}")
            traceback.print_exc()

    elapsed = time.time() - start_time

    stats = {
        "total_chunks": total_chunks,
        "total_files": total_files,
        "elapsed_seconds": round(elapsed, 1),
        "company_counts": dict(sorted(company_counts.items())),
        "section_counts": dict(
            sorted(section_counts.items(), key=lambda x: -x[1])
        ),
        "errors": errors,
    }

    print(f"\nIndexing complete in {elapsed:.1f}s")
    print(f"  Files processed: {total_files}/{len(file_list)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Companies: {len(company_counts)}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"    - {err}")

    store.close()
    return stats


if __name__ == "__main__":
    force = "--force" in sys.argv
    index_corpus(force_reindex=force)
