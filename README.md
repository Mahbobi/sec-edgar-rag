# SEC EDGAR RAG System

A retrieval-augmented generation (RAG) system for answering business questions about SEC financial filings (10-K and 10-Q reports) from 54 major US public companies spanning 2022-2026.

The system accepts a natural-language business question, retrieves relevant filing excerpts from an indexed corpus, and produces a well-structured, citation-rich answer in a **single LLM API call**.

---

## Quick Start

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- ~200MB disk for the vector index
- The `edgar_corpus.zip` dataset (provided with the assessment)

### Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd sec-edgar-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# Open .env in a text editor and paste your ANTHROPIC_API_KEY

# 4. Download and extract the corpus
# Place edgar_corpus.zip in the project root, then:
mkdir -p data/corpus
unzip edgar_corpus.zip -d data/corpus

# 5. Build the vector index (one-time, ~5-10 minutes)
python indexer.py

# 6. Start the web server
python -m uvicorn app:app --port 8000
```

Open **http://localhost:8000** in your browser. You're ready to ask questions.

---

## How It Works

```
User Question
     │
     ▼
[Retriever]       ── embed query → cosine search across 21k chunks
     │                → detect mentioned companies → boost relevant sections
     │                → diversity re-rank for balanced multi-company coverage
     │
     ▼
[Prompt Builder]  ── inject 8-12 ranked excerpts with rich metadata
     │                headers (company, filing type, date, section)
     │
     ▼
[Single LLM Call] ── Anthropic Claude Sonnet 4
     │                one API request produces the full answer
     │                grounded exclusively in retrieved excerpts
     │
     ▼
Structured Answer with Citations
```

### Pipeline Steps in Detail

1. **Indexing (one-time):** Each SEC filing is parsed for its structured header (company, ticker, filing type, date), split into SEC section-aware chunks (~800 tokens each), embedded using all-MiniLM-L6-v2, and stored in a SQLite vector database.

2. **Retrieval:** The user's question is embedded with the same model and compared against all chunks via cosine similarity. The retriever detects company names/tickers in the query, boosts relevant sections (e.g., "risk" → Item 1A), and applies diversity re-ranking to ensure balanced multi-company coverage.

3. **Answer Generation:** Retrieved excerpts are injected into a structured prompt with metadata headers. A single LLM API call produces a grounded, well-organized answer with filing citations.

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Embedder | `embeddings.py` | ONNX all-MiniLM-L6-v2 (384-dim sentence vectors) |
| Vector Store | `vector_store.py` | SQLite + numpy brute-force cosine similarity |
| Chunker | `chunker.py` | SEC section-aware chunking with tiktoken measurement |
| Indexer | `indexer.py` | Build the full vector index from the corpus |
| Retriever | `retriever.py` | Query → embed → search → re-rank pipeline |
| Prompt | `prompt_template.py` | System/user message construction for the LLM |
| LLM | `llm.py` | Single API call wrapper for answer generation |
| Web UI | `app.py` | FastAPI server + single-page HTML interface |

---

## Usage

### Web UI (recommended for demo)
```bash
python -m uvicorn app:app --port 8000
```
Open http://localhost:8000 and type a business question. Three example questions are provided as clickable buttons.

### CLI
```bash
# Run all pre-built example queries
python examples.py

# Run a specific question
python examples.py --query "How has NVIDIA's revenue changed over the last two years?"
```

### Quality Evaluation
```bash
# Retrieval quality metrics (Company Recall, Section Recall, MRR)
python evaluate.py

# Full evaluation including LLM answer quality
python evaluate.py --full
```

### Re-indexing
```bash
# Force rebuild the index (if you modify the corpus or chunking)
python indexer.py --force
```

---

## Corpus

- **246 filings** from **54 companies** across technology, financial services, healthcare, consumer, energy, and industrial sectors
- **21,447 indexed chunks** (12,434 from 10-K annual reports, 9,013 from 10-Q quarterly reports)
- Filing period: 2022-2026
- Source: SEC EDGAR

---

## Example Questions

The system handles single-company deep dives, multi-company comparisons, sector-wide analysis, and temporal trend questions:

- *"What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"*
- *"How has NVIDIA's revenue and growth outlook changed over the last two years?"*
- *"What regulatory risks do the major pharmaceutical companies face?"*
- *"Compare the business strategies of Microsoft, Google, and Meta in the AI space."*

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Avg Company Recall@k | 91.5% |
| Avg Section Recall@k | 60.0% |
| Mean Reciprocal Rank | 0.950 |
| Company Detection Recall | 80.0% |

See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for full methodology and [prompt_log.md](prompt_log.md) for the prompt iteration history.

---

## Project Structure

```
sec-edgar-rag/
├── README.md                  # This file
├── DESIGN_DECISIONS.md        # Architecture decisions, assumptions, quality evaluation
├── prompt_log.md              # Prompt iteration log (4 versions)
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── .gitignore
│
├── config.py                  # Central configuration
├── embeddings.py              # ONNX embedding model wrapper
├── chunker.py                 # SEC filing parser + chunker
├── vector_store.py            # SQLite vector store
├── indexer.py                 # Corpus indexing pipeline
├── retriever.py               # Retrieval + re-ranking pipeline
├── prompt_template.py         # LLM prompt construction
├── llm.py                     # LLM API call wrapper
├── app.py                     # FastAPI web application
├── evaluate.py                # Quality evaluation harness
├── examples.py                # Pre-built example queries (CLI)
│
├── templates/index.html       # Web UI
├── static/style.css           # UI styling
│
├── data/corpus/               # SEC filing .txt files + manifest.json (not in git)
└── db/vectors.db              # SQLite vector index (generated, not in git)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector Store | SQLite + numpy | ChromaDB incompatible with Python 3.14; custom store is 200 lines, zero-dep, fast for <20k chunks |
| Embeddings | ONNX all-MiniLM-L6-v2 | Local inference, no API costs, 384-dim, good semantic quality |
| Chunking | SEC section-aware, ~800 tokens | Respects Item 1/1A/7/8 structure; section metadata enables targeted retrieval |
| Retrieval | Targeted search + diversity re-rank | Guarantees balanced multi-company coverage for comparison queries |
| UI | FastAPI + vanilla HTML/JS | Zero frontend dependencies, professional look, instant setup |

See [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) for the full rationale behind each decision.
