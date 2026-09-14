<h1 align="center"> ResearchPilot </h1>
<p align="center">
  <strong style="background-color: green;">English</strong>
  |
  <a href="./README_zh.md" target="_Self">中文</a>
</p>

ResearchPilot is a research assistant for scientific papers (currently: vehicle / person
re-identification literature). It started as the AI-ChatKit engineering template and was
rebuilt layer by layer into a domain agent: paper ingestion, hybrid retrieval, a corrective-RAG
research workflow, an evaluation harness, and one-command deployment.

> Design rationale for every layer lives in `reference/` —
> start with `reference/design-decisions.md` and `reference/roadmap-status.md`.

## What it does

1. **Document knowledge base** - upload PDF / DOCX / Markdown / TXT via `POST /documents`
   (202 + background indexing); page-aware PDF parsing (pdfminer, figure text stripped),
   citation-grade locators (`p.5` / `sec.2` / `blk.4`).
2. **Hybrid retrieval** - BM25 (hand-rolled, filter-aware) + vector recall fused with RRF,
   cross-encoder reranking (ONNX int8, local). Vector backend selectable: Chroma (default,
   embedded) or Qdrant (`VECTOR_STORE=qdrant`).
3. **Two agents** - `research-workflow` (Corrective-RAG variant: analyze, retrieve, assess,
   refine, synthesize; hard round budget; three-state evidence verdict) and `react-assistant`
   (ReAct baseline for comparison). Registry served at `GET /agents`.
4. **Evaluation harness** - 30-item A/B/C eval set with LLM judge (defined verdicts, corpus
   fingerprint staleness check, absence probes) plus an LLM-free deterministic rank
   benchmark (`rank_bench.py`) and a black-box API acceptance script.
5. **Persistent sessions** - LangGraph checkpointer on PostgreSQL; conversation list & history
   served via `/conversations`, frontend keeps no state of its own.
6. **Ops** - `/health` (code rev, reranker state, index stats), rotating file logs,
   unified exception layer, Docker Compose (api / postgres / web, plus optional qdrant).

## Quick start

> **The corpus is not bundled.** `backend/resource/papers/` ships empty on purpose - the
> papers used during development are published work, so they are not redistributed here.
> Drop your own PDFs into that folder and import them with
> `python app/ai/rag/ingest.py` (titles come from PDF metadata / filenames; a
> `titles.json` is generated on first run for manual correction).

```
# 1. infrastructure (PostgreSQL is required for sessions & documents)
docker compose up -d postgres

# 2. backend (needs Ollama with bge-m3 on the host).
#    Use run_server.py rather than `python -m uvicorn`: on Windows the async psycopg
#    checkpointer only runs on a Selector event loop, which uvicorn installs only in
#    its --reload subprocess path. run_server.py sets the policy before uvicorn.run()
#    and needs no --reload (see NOTES.md).
cd backend
.venv-py311/Scripts/python.exe run_server.py

# 3. frontend
cd ../frontend
pnpm install && pnpm dev

# or everything at once (Windows): double-click 启动 Docker.cmd
```

API docs: `http://127.0.0.1:8002/docs` (FastAPI auto docs).

## Configuration

`backend/.env` is **not** in the repo (it is git-ignored, so a clone never carries
anyone else's keys). Create your own from the template:

```bash
cd backend
cp .env.example .env      # Windows: copy .env.example .env
```

Then set **two** things:

| What | Value | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | your own key | Any OpenAI-compatible provider works - see the alternatives in `.env.example`. The lines are commented out, so **uncomment the provider you use**; filling in a key without uncommenting does nothing. |
| `EMBEDDING_MODEL` | `bge-m3` | Embeddings run locally through Ollama, no API key: `ollama pull bge-m3`. |

`DATABASE_URL` defaults to SQLite (zero setup). Switch it to the PostgreSQL line
in the template if you want the session/document features with the docker-compose
database.

## Learn more

| Doc | Content |
|---|---|
| `reference/design-decisions.md` | why hybrid retrieval / rerank / pdfminer / the eval harness - with measurements |
| `reference/transformation-*.md` | per-transformation design docs (workflow, document API, sessions, logging, qdrant) |
| `reference/roadmap-status.md` | live status against the 10-item mission |
| `reference/deployment.md` | ops manual & troubleshooting |
