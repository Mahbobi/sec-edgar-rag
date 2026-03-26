"""Pre-built example queries for testing and demo purposes.

Run this script to execute example queries through the full RAG pipeline
and print the results. Useful for quick testing and as a deliverable
showing the system in action.

Usage:
    python examples.py                  # Run all examples
    python examples.py --query "..."    # Run a custom query
"""

import sys
import time
import json

from embeddings import Embedder
from retriever import Retriever
from vector_store import VectorStore
from llm import generate_answer

EXAMPLE_QUERIES = [
    "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
    "How has NVIDIA's revenue and growth outlook changed over the last two years?",
    "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
    "Compare the business strategies of Microsoft, Google, and Meta in the AI space based on their recent filings.",
    "What are the key financial metrics and trends for Amazon across its most recent annual and quarterly reports?",
]


def run_query(question: str, retriever: Retriever, verbose: bool = True) -> dict:
    """Run a single query through the full pipeline."""
    print(f"\n{'='*80}")
    print(f"Q: {question}")
    print(f"{'='*80}")

    # Retrieve
    start = time.time()
    contexts, info = retriever.retrieve(question)
    retrieval_time = time.time() - start

    if verbose:
        print(f"\nRetrieval ({retrieval_time:.3f}s):")
        print(f"  Detected companies: {info['detected_companies']}")
        print(f"  Detected sections: {info['detected_sections']}")
        print(f"  Candidates → Final: {info['initial_candidates']} → {info['final_results']}")
        print(f"  Companies in results: {info['companies_in_results']}")

    if not contexts:
        print("\nNo relevant contexts found.")
        return {"question": question, "answer": None, "error": "No contexts"}

    # Generate answer
    response = generate_answer(question, contexts)

    print(f"\nAnswer ({response.latency_seconds}s, {response.input_tokens}+{response.output_tokens} tokens):")
    print(f"\n{response.answer}")

    if verbose:
        print(f"\nSources:")
        for i, ctx in enumerate(contexts, 1):
            print(
                f"  {i}. {ctx.company} ({ctx.ticker}) | "
                f"{ctx.filing_type} {ctx.filing_date} | "
                f"{ctx.section_name} | score={ctx.score:.4f}"
            )

    return {
        "question": question,
        "answer": response.answer,
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "llm_latency": response.latency_seconds,
        "retrieval_latency": round(retrieval_time, 3),
        "contexts_used": len(contexts),
        "retrieval_info": info,
    }


def main():
    print("Initializing SEC EDGAR RAG System...")
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder=embedder, store=store)

    stats = store.get_stats()
    print(f"Index: {stats['total_chunks']} chunks, {len(stats['companies'])} companies")

    # Check for custom query
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        if idx + 1 < len(sys.argv):
            query = sys.argv[idx + 1]
            run_query(query, retriever)
            store.close()
            return

    # Run all examples
    results = []
    for q in EXAMPLE_QUERIES:
        result = run_query(q, retriever)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        status = "OK" if r.get("answer") else "FAIL"
        tokens = f"{r.get('input_tokens', 0)}+{r.get('output_tokens', 0)}"
        latency = r.get("llm_latency", 0)
        print(f"  [{status}] {r['question'][:60]}... ({tokens} tokens, {latency}s)")

    store.close()


if __name__ == "__main__":
    main()
