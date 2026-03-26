"""FastAPI web application for the SEC EDGAR RAG system.

Serves a single-page UI and provides an API endpoint for querying
the RAG pipeline. The final answer is produced in a single LLM call.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from embeddings import Embedder
from retriever import Retriever
from vector_store import VectorStore
from llm import generate_answer

# Shared state initialized at startup
_retriever: Retriever | None = None
_store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retriever and vector store on startup."""
    global _retriever, _store

    print("Loading vector store and embedder...")
    start = time.time()
    _store = VectorStore()
    embedder = Embedder()
    _retriever = Retriever(embedder=embedder, store=_store)

    stats = _store.get_stats()
    print(
        f"Ready in {time.time() - start:.1f}s — "
        f"{stats['total_chunks']} chunks, "
        f"{len(stats['companies'])} companies"
    )

    yield

    if _store:
        _store.close()


app = FastAPI(
    title="SEC EDGAR RAG System",
    description="RAG-powered Q&A over SEC 10-K and 10-Q filings",
    lifespan=lifespan,
)

# Static files and templates
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def query(request: Request):
    """Process a business question through the RAG pipeline.

    Accepts JSON: {"question": "...", "top_k": 10}
    Returns JSON with the answer, sources, and metadata.
    """
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        return JSONResponse(
            {"error": "Please provide a question."},
            status_code=400,
        )

    if not _retriever:
        return JSONResponse(
            {"error": "System not initialized. Please run the indexer first."},
            status_code=503,
        )

    try:
        top_k = body.get("top_k", config.TOP_K_FINAL)

        # Step 1: Retrieve relevant chunks
        retrieval_start = time.time()
        contexts, retrieval_info = _retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start

        if not contexts:
            return JSONResponse({
                "answer": "No relevant filing excerpts found for your question. "
                "Please try rephrasing or asking about a specific company.",
                "sources": [],
                "metadata": {"retrieval_info": retrieval_info},
            })

        # Step 2: Generate answer via single LLM call
        llm_response = generate_answer(question, contexts)

        # Step 3: Build source citations
        sources = [
            {
                "company": ctx.company,
                "ticker": ctx.ticker,
                "filing_type": ctx.filing_type,
                "filing_date": ctx.filing_date,
                "section": ctx.section_name,
                "relevance_score": round(ctx.score, 4),
                "excerpt_preview": ctx.text[:200] + "..."
                if len(ctx.text) > 200
                else ctx.text,
            }
            for ctx in contexts
        ]

        return JSONResponse({
            "answer": llm_response.answer,
            "sources": sources,
            "metadata": {
                "model": llm_response.model,
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "llm_latency_seconds": llm_response.latency_seconds,
                "retrieval_latency_seconds": round(retrieval_time, 3),
                "contexts_used": llm_response.contexts_used,
                "retrieval_info": retrieval_info,
            },
        })

    except Exception as e:
        return JSONResponse(
            {"error": f"Error processing query: {str(e)}"},
            status_code=500,
        )


@app.get("/api/stats")
async def stats():
    """Return index statistics."""
    if not _store:
        return JSONResponse({"error": "Store not initialized"}, status_code=503)

    return JSONResponse(_store.get_stats())


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
