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

It now also includes **Scientific Research Harness V1**. The existing RAG remains available and
acts as the Literature Evidence Engine, while research questions, hypotheses, evidence,
experiments, runs, observations and conclusions are persisted as separate, reviewable objects.
See `docs/current_system_audit.md`, `docs/modification_plan.md` and
`docs/project_explanation.md`.

Run the fully offline structured-research demo from `backend/`:

```powershell
.venv-py311\Scripts\python.exe scripts/scientific_harness_demo.py
```

The Workbench exports a complete research archive with a SHA-256 seal. Verify an archive
or run a read-only deployment smoke test from `backend/`:

```powershell
.venv-py311\Scripts\python.exe scripts/verify_research_archive.py RQ-xxx-research-archive.json
.venv-py311\Scripts\python.exe scripts/research_harness_smoke.py --base-url http://127.0.0.1:8002
# Restore only into a missing RQ; the explicit ID confirmation prevents selecting the wrong archive.
.venv-py311\Scripts\python.exe scripts/restore_research_archive.py archive.json --confirm-question-id RQ-xxx
# Isolated deployments only: creates one question and one approved hypothesis.
.venv-py311\Scripts\python.exe scripts/research_workbench_browser_smoke.py `
  --url http://127.0.0.1:3000/research --browser-executable "C:\Path\To\msedge.exe"
```

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
7. **Research Workbench** - `/research` is a lightweight UI for research questions,
   hypotheses, three-way literature evidence, human approvals, integrity metrics and
   structured reports. PostgreSQL remains the sole source of scientific state.

## Quick start

> **The corpus is not bundled.** `backend/resource/papers/` ships empty on purpose - the
> papers used during development are published work, so they are not redistributed here.
> Drop your own PDFs into that folder and import them with
> `python app/ai/rag/ingest.py` (titles come from PDF metadata / filenames; a
> `titles.json` is generated on first run for manual correction).

```
# 1. infrastructure (PostgreSQL is required for sessions & documents)
docker compose up -d postgres

# 2. backend (needs Ollama with bge-m3 on the host), Python 3.11.
#    Create the venv once, from the pinned versions in requirements.lock.
#    Those pins are 3.11-specific, so do NOT use `uv sync` here - a fresh
#    resolve would pull langchain 1.x and the app would not run (see pyproject.toml).
cd backend
python3.11 -m venv .venv-py311
.venv-py311/Scripts/python.exe -m pip install -r requirements.lock

#    Start it with run_server.py rather than `python -m uvicorn`: on Windows the
#    async psycopg checkpointer only runs on a Selector event loop, which uvicorn
#    installs only in its --reload subprocess path (see NOTES.md).
.venv-py311/Scripts/python.exe run_server.py

# 3. frontend
cd ../frontend
pnpm install && pnpm dev

# or everything at once (Windows): double-click 启动 Docker.cmd
```

API docs: `http://127.0.0.1:8002/docs` (FastAPI auto docs).
Research Workbench: `http://127.0.0.1:3000/research`.

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
