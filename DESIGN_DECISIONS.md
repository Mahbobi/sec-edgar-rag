# Design Decisions & Quality Evaluation

## Architecture Decisions

### 1. Vector Store: SQLite + Numpy (not ChromaDB)

**Decision:** Built a custom vector store using SQLite for persistence and numpy for cosine similarity search, instead of using the pre-installed ChromaDB 1.5.2.

**Why:** ChromaDB 1.5.2 is incompatible with Python 3.14 due to a Pydantic v1 dependency issue. Rather than downgrade Python or fight dependency conflicts, I built a lightweight alternative that uses the same underlying math (cosine similarity on normalized vectors).

**Trade-off:** For ~21k chunks at 384 dimensions, the full embedding matrix is ~32MB in memory. Brute-force cosine similarity takes <50ms. This is far below the threshold where approximate nearest neighbor (ANN) algorithms like HNSW would provide meaningful speedup. The simplicity of brute-force eliminates an entire class of configuration and tuning complexity.

### 2. Embeddings: ONNX all-MiniLM-L6-v2

**Decision:** Used the all-MiniLM-L6-v2 model loaded via ONNX Runtime, bypassing the sentence-transformers library.

**Why:** The ONNX model was already cached locally (~90MB). ONNX Runtime provides fast CPU inference without requiring PyTorch model loading overhead. The 384-dimensional embeddings provide good semantic matching quality for financial text.

**Alternative considered:** Using a cloud embedding API (e.g., Voyage AI). Rejected because it would add API costs during indexing, require network access, and introduce a dependency on another service. Local embeddings are free, fast, and reproducible.

### 3. Chunking: SEC Section-Aware (~800 tokens)

**Decision:** Implemented a custom chunker that detects SEC filing section boundaries (Item 1, 1A, 7, etc.) and chunks within sections at ~800 tokens with 100-token overlap.

**Why SEC section awareness matters:**
- Risk factors (Item 1A), MD&A (Item 7), and financial statements (Item 8) have very different informational content
- Section metadata enables targeted boosting during retrieval (e.g., "risk" queries → boost Item 1A chunks)
- Prevents chunks from spanning section boundaries, which would confuse the LLM about context

**Why 800 tokens:** Large enough to contain a complete risk factor description or financial discussion paragraph, but small enough for the embedding model to capture focused semantics. With 8-12 chunks retrieved, total context stays under 10k tokens — well within the model's context window.

**Why tiktoken for measurement:** tiktoken's cl100k_base encoding provides a good-enough approximation for budget management across LLM providers. The exact token count doesn't need to match the target model — it's used for sizing guidance.

### 4. LLM: Anthropic Sonnet (Single Call)

**Decision:** Used the Anthropic Sonnet model for the final answer generation, with the entire pipeline producing one API request.

**Why Sonnet over Opus:** Sonnet is sufficient for structured extraction from provided text — the task is grounding and organization, not novel reasoning. Sonnet is faster (~5-10s vs 15-30s) and more cost-effective per token. For a live demo, response time matters.

**Why single call works:** All retrieval, ranking, and context assembly happens before the API call. The model receives a fully constructed prompt with 8-12 relevant excerpts and metadata. No chains, tools, or multi-turn — one shot. This satisfies the assessment constraint and demonstrates that good retrieval makes complex orchestration unnecessary.

### 5. UI: FastAPI + Vanilla HTML/CSS/JS

**Decision:** Used FastAPI (already installed) with a single-page HTML/CSS/JS interface, rather than installing Streamlit or Gradio.

**Why:** FastAPI was already available (no installation needed), provides a professional web server, and the vanilla frontend shows full-stack competence. The resulting UI is clean, fast, and has no JavaScript framework dependencies — it can be cloned and run immediately without `npm install`.

### 6. Retrieval: Targeted Search + Diversity Re-ranking

**Decision:** For multi-company comparison queries, the retriever performs both a broad semantic search AND targeted per-company searches, then applies diversity re-ranking to guarantee balanced representation.

**Why:** Naive top-k retrieval strongly favors companies with more textually relevant chunks (e.g., JPMorgan's extensive risk disclosures dominate financial queries). Without diversity constraints, a "compare Apple, Tesla, and JPMorgan" query might return 8 JPMorgan chunks and 0 Tesla chunks. The diversity algorithm guarantees at least 2 chunks per detected company before filling remaining slots by score.

**Company detection:** Uses a curated alias dictionary mapping common names (Apple, Tesla, JPMorgan, Google, etc.) to tickers. Ticker matching requires 3+ characters to avoid false positives from short tickers (BA, MS, HD, T, V).

---

## Assumptions

1. **Filing format is consistent:** All .txt files follow the structured header format (Company, Ticker, Filing Type, etc.) followed by a separator line and body text. Verified this holds across all 246 files.

2. **Section headers are regex-detectable:** SEC filings use "Item N." patterns for section boundaries. This works for standard 10-K/10-Q filings but may miss non-standard formatting.

3. **Corpus fits in memory:** 21k chunks × 384 dims × 4 bytes = ~32MB. This comfortably fits in memory for brute-force search. For a 100x larger corpus, I would switch to FAISS or a vector database.

4. **Embedding model quality is sufficient:** all-MiniLM-L6-v2 is a general-purpose sentence embedding model. A finance-specific model (e.g., FinBERT) might perform better on specialized terminology, but MiniLM provides a good baseline.

5. **Single LLM call context budget:** With 8-12 chunks at ~800 tokens each, total context is ~6-10k tokens. The model supports 200k tokens, so we have significant headroom. The token budget (8000 for context) is conservative by design.

---

## Quality Evaluation Methodology

### Retrieval Quality (automated, no LLM needed)

**Metrics:**
- **Company Recall@k:** What fraction of expected companies appear in the top-k results?
- **Section Recall@k:** What fraction of expected SEC sections appear in results?
- **MRR (Mean Reciprocal Rank):** How high does the first relevant result rank?
- **Company Detection Accuracy:** Does the query parser correctly identify mentioned companies?

**Test suite:** 10 hand-crafted test cases covering:
- Single-company queries (Apple risk factors, NVIDIA revenue)
- Multi-company comparisons (Apple vs Tesla, banking sector)
- Section-specific queries (legal proceedings, cybersecurity)
- Sector-wide queries (pharmaceutical companies, major banks)

Run with: `python evaluate.py`

### Answer Quality (manual + heuristic)

**Criteria:**
1. **Groundedness:** Does the answer cite only information from provided excerpts? (Target: >95%)
2. **Completeness:** Does it address all parts of multi-part questions? (Target: >80%)
3. **Accuracy:** Are cited numbers and facts correct vs source text? (Target: >95%)
4. **Structure:** Is the answer well-organized for the question type? (Target: >90%)
5. **Temporal awareness:** Does it note trends when multi-period data is available? (Target: >75%)

**Automated heuristic:** After LLM generation, check that each mentioned company ticker/name appears in the answer, and that the answer length is proportional to the number of companies queried.

Run with: `python evaluate.py --full`

---

## What I Would Improve With More Time

1. **Hybrid retrieval:** Combine semantic search with BM25/keyword search for better recall on specific financial terms (e.g., exact revenue numbers, specific regulation names).

2. **Section-level re-indexing:** Instead of chunking by token count within sections, parse the actual sub-section structure (individual risk factors, financial line items) for more granular retrieval.

3. **Financial entity extraction:** Pre-extract key financial metrics (revenue, net income, EPS, etc.) into a structured index for precise numerical queries.

4. **Cross-filing deduplication:** Many 10-Q filings repeat boilerplate from the 10-K. Deduplicate at chunk level to avoid wasting context budget on redundant text.

5. **Evaluation with ground truth:** Create gold-standard answers for 20-30 questions and measure ROUGE/BERTScore against LLM outputs for automated quality regression.

6. **Streaming responses:** Use the LLM's streaming API to show the answer progressively in the UI rather than waiting for the full response.
