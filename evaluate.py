"""Quality evaluation harness for the SEC EDGAR RAG system.

Measures retrieval quality (Recall@k, MRR) and answer quality
(groundedness, completeness) using predefined test cases.

Usage:
    python evaluate.py                  # Run retrieval evaluation
    python evaluate.py --full           # Run retrieval + LLM evaluation
"""

import json
import sys
import time
from dataclasses import dataclass

from embeddings import Embedder
from retriever import Retriever
from vector_store import VectorStore
from llm import generate_answer


@dataclass
class TestCase:
    """A test question with expected retrieval targets."""

    question: str
    expected_tickers: list[str]
    expected_sections: list[str]
    description: str


# Test cases with expected retrieval targets
TEST_CASES = [
    TestCase(
        question="What are Apple's primary risk factors?",
        expected_tickers=["AAPL"],
        expected_sections=["1A"],
        description="Single company, specific section",
    ),
    TestCase(
        question="How has NVIDIA's revenue changed recently?",
        expected_tickers=["NVDA"],
        expected_sections=["7", "8"],
        description="Single company, financial data",
    ),
    TestCase(
        question="Compare risk factors of Apple and Tesla",
        expected_tickers=["AAPL", "TSLA"],
        expected_sections=["1A"],
        description="Multi-company comparison, specific section",
    ),
    TestCase(
        question="What is JPMorgan's business strategy?",
        expected_tickers=["JPM"],
        expected_sections=["1", "7"],
        description="Single company, business overview",
    ),
    TestCase(
        question="What cybersecurity risks does Microsoft face?",
        expected_tickers=["MSFT"],
        expected_sections=["1A", "1C"],
        description="Single company, cybersecurity focus",
    ),
    TestCase(
        question="Compare the financial performance of Google and Meta",
        expected_tickers=["GOOG", "META"],
        expected_sections=["7", "8"],
        description="Multi-company financial comparison",
    ),
    TestCase(
        question="What legal proceedings are pending against pharmaceutical companies?",
        expected_tickers=["PFE", "JNJ", "MRK", "LLY", "ABBV"],
        expected_sections=["3", "1A"],
        description="Multi-company, legal focus",
    ),
    TestCase(
        question="What is Tesla's management discussion of operations?",
        expected_tickers=["TSLA"],
        expected_sections=["7"],
        description="Single company, MD&A",
    ),
    TestCase(
        question="How do the major banks compare in terms of risk exposure?",
        expected_tickers=["JPM", "BAC", "GS", "MS"],
        expected_sections=["1A", "7A"],
        description="Multi-company banking sector",
    ),
    TestCase(
        question="What is Amazon's competitive position in cloud computing?",
        expected_tickers=["AMZN"],
        expected_sections=["1", "1A"],
        description="Single company, competitive analysis",
    ),
]


def evaluate_retrieval(retriever: Retriever) -> dict:
    """Evaluate retrieval quality across test cases.

    Metrics:
    - Company Recall@k: Were expected companies in the results?
    - Section Recall@k: Were expected sections in the results?
    - Company Detection Accuracy: Did the query parser detect right companies?
    - MRR: Mean reciprocal rank of first relevant result.
    """
    results = []

    for tc in TEST_CASES:
        contexts, info = retriever.retrieve(tc.question)

        result_tickers = set(c.ticker for c in contexts)
        result_sections = set(c.section_id for c in contexts)

        # Company recall
        expected_in_results = set(tc.expected_tickers) & result_tickers
        company_recall = (
            len(expected_in_results) / len(tc.expected_tickers)
            if tc.expected_tickers
            else 1.0
        )

        # Section recall
        expected_sections_in_results = set(tc.expected_sections) & result_sections
        section_recall = (
            len(expected_sections_in_results) / len(tc.expected_sections)
            if tc.expected_sections
            else 1.0
        )

        # Company detection accuracy
        detected = set(info["detected_companies"])
        expected = set(tc.expected_tickers)
        detection_precision = (
            len(detected & expected) / len(detected) if detected else 0.0
        )
        detection_recall = (
            len(detected & expected) / len(expected) if expected else 1.0
        )

        # MRR (rank of first relevant result by company match)
        mrr = 0.0
        for i, ctx in enumerate(contexts):
            if ctx.ticker in tc.expected_tickers:
                mrr = 1.0 / (i + 1)
                break

        results.append({
            "question": tc.question,
            "description": tc.description,
            "company_recall": round(company_recall, 3),
            "section_recall": round(section_recall, 3),
            "detection_precision": round(detection_precision, 3),
            "detection_recall": round(detection_recall, 3),
            "mrr": round(mrr, 3),
            "result_tickers": sorted(result_tickers),
            "result_sections": sorted(result_sections),
            "expected_tickers": tc.expected_tickers,
            "detected_companies": info["detected_companies"],
            "num_results": len(contexts),
        })

    # Aggregate metrics
    avg_company_recall = sum(r["company_recall"] for r in results) / len(results)
    avg_section_recall = sum(r["section_recall"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_detection_recall = sum(r["detection_recall"] for r in results) / len(results)

    return {
        "test_cases": results,
        "aggregate": {
            "avg_company_recall": round(avg_company_recall, 3),
            "avg_section_recall": round(avg_section_recall, 3),
            "avg_mrr": round(avg_mrr, 3),
            "avg_detection_recall": round(avg_detection_recall, 3),
            "total_cases": len(results),
        },
    }


def main():
    full_eval = "--full" in sys.argv

    print("Initializing evaluation harness...")
    embedder = Embedder()
    store = VectorStore()
    retriever = Retriever(embedder=embedder, store=store)

    stats = store.get_stats()
    print(f"Index: {stats['total_chunks']} chunks, {len(stats['companies'])} companies\n")

    # Retrieval evaluation
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    start = time.time()
    eval_results = evaluate_retrieval(retriever)
    elapsed = time.time() - start

    for r in eval_results["test_cases"]:
        status = "PASS" if r["company_recall"] >= 0.5 and r["mrr"] > 0 else "FAIL"
        print(
            f"  [{status}] {r['description']}"
            f" | Co.Recall={r['company_recall']:.1%}"
            f" | Sec.Recall={r['section_recall']:.1%}"
            f" | MRR={r['mrr']:.3f}"
        )

    agg = eval_results["aggregate"]
    print(f"\nAggregate ({elapsed:.1f}s):")
    print(f"  Avg Company Recall: {agg['avg_company_recall']:.1%}")
    print(f"  Avg Section Recall: {agg['avg_section_recall']:.1%}")
    print(f"  Avg MRR: {agg['avg_mrr']:.3f}")
    print(f"  Avg Detection Recall: {agg['avg_detection_recall']:.1%}")

    if full_eval:
        print(f"\n{'='*60}")
        print("ANSWER QUALITY EVALUATION (LLM)")
        print(f"{'='*60}")

        for tc in TEST_CASES[:3]:  # Run LLM eval on first 3 only (cost)
            contexts, info = retriever.retrieve(tc.question)
            if not contexts:
                print(f"  SKIP: {tc.description} (no contexts)")
                continue

            response = generate_answer(tc.question, contexts)
            print(f"\n  Q: {tc.question}")
            print(f"  Tokens: {response.input_tokens}+{response.output_tokens}")
            print(f"  Latency: {response.latency_seconds}s")
            print(f"  Answer length: {len(response.answer)} chars")
            # Basic groundedness check: answer should mention expected companies
            mentions = sum(
                1 for t in tc.expected_tickers
                if t in response.answer or any(
                    c.company.split()[0] in response.answer
                    for c in contexts if c.ticker == t
                )
            )
            groundedness = mentions / len(tc.expected_tickers) if tc.expected_tickers else 1.0
            print(f"  Company mentions in answer: {mentions}/{len(tc.expected_tickers)} ({groundedness:.0%})")

    # Save results
    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    store.close()


if __name__ == "__main__":
    main()
