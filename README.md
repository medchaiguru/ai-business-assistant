# AI Customer Support Automation System

Production-ready RAG pipeline that automates customer support for a Muay Thai gym. Queries are answered from business data using retrieval-augmented generation with semantic caching, served through a FastAPI backend and containerized with Docker Compose.

## Architecture

```
                        ┌──────────────┐
                        │   Gradio UI  │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │   FastAPI    │
                        │  + Middleware │──── Metrics ──── Redis
                        └──────┬───────┘
                               │
                     ┌─────────▼──────────┐
                     │   LangGraph RAG    │
                     │     Workflow       │
                     └─────────┬──────────┘
                        ┌──────┴───────┐
                        │              │
                ┌───────▼──────┐ ┌─────▼──────┐
                │  Semantic    │ │  RAG Chain  │
                │  Cache       │ │  (Retriever │
                │  (ChromaDB)  │ │  + LLM)     │
                └──────────────┘ └─────┬───────┘
                                       │
                                ┌──────▼───────┐
                                │   ChromaDB   │
                                │ Vector Store │
                                └──────────────┘
```

## Key Engineering Decisions

**LangGraph workflow orchestration** — The RAG pipeline is modeled as a stateful graph (`check_cache → call_rag → END`), making it easy to add nodes without touching existing logic.

**Semantic caching with ChromaDB** — similar questions are served from a vector similarity cache (**threshold 0.92**), avoiding redundant LLM calls and reducing cost/latency.

**Async-first design** — All I/O (LLM calls, vector search, Redis, ChromaDB) uses `async/await` for high concurrency under FastAPI's event loop.

**Redis metrics pipeline** — Token usage, cost, and response times are tracked per-request via middleware and stored in Redis using **pipelined** writes for minimal overhead.

**Runtime re-indexing** — The `/admin/index` endpoint triggers background **re-indexing** without downtime, hot-swapping the retriever in app state.

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Orchestration | LangGraph / LangChain |
| Vector Store | ChromaDB (HTTP client) |
| Caching | Semantic cache (ChromaDB) |
| Metrics Store | Redis |
| API | FastAPI + Uvicorn |
| Frontend | Gradio |
| Observability | LangSmith tracing, rotating file logs |
| Containerization | Docker Compose |
| Code Quality | Ruff, Pylint, Mypy (strict), Pytest |

## Project Structure

```
app/
├── main.py                  # FastAPI app, lifespan, middleware
├── config.py                # Pydantic settings (env-driven)
├── logger.py                # Rotating file + console logging
├── api/
│   ├── startup.py           # App state initialization
│   ├── deps.py              # FastAPI dependency injection
│   └── routers/
│       ├── query.py         # POST /query — RAG inference
│       ├── health.py        # GET  /health
│       ├── metrics.py       # GET  /metrics — token/cost stats
│       └── admin.py         # POST /admin/index — re-index data
├── components/
│   ├── workflow.py          # LangGraph RAG state machine
│   ├── rag_chain.py         # Retriever → Prompt → LLM chain
│   ├── semantic_cache.py    # Vector similarity cache
│   ├── vector_store.py      # ChromaDB client + retriever factory
│   ├── embedding.py         # OpenAI embeddings config
│   ├── llm.py               # OpenAI LLM config + cost calc
│   ├── prompt.py            # Prompt templates
│   ├── splitter.py          # Markdown-aware text chunking
│   ├── data_source.py       # JSON → LangChain Document loader
│   ├── models.py            # Pydantic request/response schemas
│   └── metrics.py           # Redis-backed metrics manager
├── web_ui/
│   └── chat.py              # Gradio chat interface
└── test/                    # Pytest suite (API + component tests)
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Submit a question, get a RAG-generated answer with sources |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Token usage, request count, avg response time, total cost |
| `POST` | `/admin/index` | Trigger background data re-indexing |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the training prices?"}'
```

```json
{
  "answer": "Training prices at Sinbi Muay Thai start from ...",
  "sources": ["https://www.sinbimuaythai.com/training-prices/"]
}
```

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- OpenAI API key

### Run with Docker Compose

```bash
# Set environment variables
cp .env.example .env
# Add your OPENAI_API_KEY and optionally LANGCHAIN_API_KEY

# Start all services (app + ChromaDB + Redis)
docker compose up --build

# Index the business data (one-time)
curl -X POST http://localhost:8000/admin/index
```

### Local Development

```bash
# Install dependencies
uv sync

# Start ChromaDB and Redis
docker compose up redis chromadb -d

# Run the API server
uvicorn app.main:app --reload

# Run tests
pytest app/test/
```

## Observability

- **LangSmith** — Full trace visibility for every LLM call, retrieval, and chain execution
- **Structured logging** — Rotating file logs (`logs/app.log`, 5MB max, 3 backups) + stdout
- **Response time headers** — `X-Response-Time` header on `/query` responses
- **Metrics dashboard** — Live token/cost tracking via `/metrics` endpoint