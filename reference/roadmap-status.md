# ResearchPilot 路线图完成情况

对照 `MISSION.md` 第 13–24 行的 10 项改造。
本表由代码实况核对得出（不是凭记忆），核对时间：HEAD = `4082c63`，工作树干净。

**总览：4 项完成、5 项部分完成、1 项未开始。**

| # | 改造项 | 状态 | 已完成的部分 | 还差什么 |
|---|---|---|---|---|
| 1 | 业务重设计（OA → 科研） | 🟡 部分 | 工具层已整体换成论文检索（`list_papers` / `resolve_paper_sources` / `all_titles` / `search_documents`，全在 `ai/tools/research_tools.py`）；`DEFAULT_AGENT` 已是 `research-workflow`；前端有知识库抽屉 | **OA 残留没清干净**，见下方清单 |
| 2 | 重写 RAG | ✅ 完成 | `parsers.py`（PDF/MD/DOCX/TXT + magic 嗅探）、`ingest.py`（800/150 切块）、`hybrid.py`（BM25 + RRF k=60）、`rerank.py`（cross-encoder，ONNX int8）、`pipeline.py`（二段式召回 → RRF → 重排 → 两信号证据评估 → 引用）、`textnorm.py`、位置标签抽象（`p.` / `sec.` / `blk.`） | — |
| 3 | LangGraph research workflow | ✅ 完成 | `research_workflow.py`（26.5 KB）：`analyze → retrieve → assess → refine → synthesize`，`MAX_RETRIEVE_ROUNDS = 3`，三态判定 `sufficient` / `absent` / `insufficient` | — |
| 4 | 接 MCP | ⬜ 未开始 | 仓库里 **0 个** mcp 文件 | 全部。押后到真要接 arXiv / PubMed 时再做 |
| 5 | 完善 FastAPI | 🟡 部分 | `/documents` 五个端点（`POST` 202 / `GET` 列表 / `GET` 详情 / `DELETE` 204 / `POST reindex` 202）、`/health`、`/chat/invoke`、`/chat/stream` | **会话 API 完全没有**——数据库里没有 conversation / message 表，前端会话全靠 `localStorage`；Agent Task API 没有；`chat_routes.py` / `chatSchema.py` 还是 OA 命名 |
| 6 | 完善数据层 | 🟡 部分 | SQLite → PostgreSQL（asyncpg）已切完；compose 里的 `postgres:16-alpine`；`paper_repo.py` 全套（含 `delete_not_in` + `path_prefix`） | **Qdrant 对比没做**（只有 Chroma）；迁移还是手写 `ALTER TABLE` shim（`_add_missing_columns`），没上 Alembic |
| 7 | 完善工程（异常/日志/任务状态/流式） | 🟡 部分 | 文档任务状态机 `pending / indexing / indexed / failed` + `error` 字段 + 前端轮询；SSE 流式输出；`requirements.lock` 锁依赖 | **日志没落文件**——全项目 0 处 `logging` / `FileHandler` 配置，容器一重启日志就没了；异常处理没有统一层 |
| 8 | RAG / Agent 评测 | ✅ 完成 | `run_eval.py`（20 题 A/B/C 类 + LLM judge + 语料指纹 `verified_on` 过期告警）、`rank_bench.py`（确定性、无 LLM）、`compare_runs.py`（重复运行取均值/极差 + 可比性告警）、`reid_paper_eval_set.json` | — |
| 9 | Docker Compose 一键部署 | ✅ 完成 | `docker-compose.yml` 三服务、两个 `Dockerfile`、`requirements.lock`、`启动 Docker.cmd` + `scripts/docker-up.ps1`（含镜像源回退、`GIT_REV` 构建参数、真探活 `/health`） | — |
| 10 | 最后调整前端 | 🟡 部分 | 知识库抽屉（上传 / 列表 / 删除 / 重建索引 / 状态轮询）、SSE 流式渲染、Agent 选择器 | `AgentSelector.tsx` 里 `AGENTS` 是**后端注册表的手工镜像**（注释里自己标了要改成 `get_all_agent_info()` 动态取）；`layout.tsx` 默认 agent 硬编码 `"research-workflow"`；品牌还是 AI ChatKit（`APP_NAME` 也没改）；会话列表在 `localStorage` |

## 第 1 项还欠的 OA 残留清单

| 文件 | 情况 |
|---|---|
| `backend/app/main.py` | 第 2/4 行 import、第 29/31 行 `include_router` 仍挂着 department / employee |
| `backend/app/api/department_routers.py` | 整个文件是 OA |
| `backend/app/api/employee_routers.py` | 整个文件是 OA |
| `backend/app/db/models/department.py` | 整个文件是 OA |
| `backend/app/db/models/employee.py` | 整个文件是 OA |
| `backend/app/db/repository/department_repo.py` | 整个文件是 OA |
| `backend/app/db/repository/employee_repo.py` | 整个文件是 OA |
| `backend/app/db/models/__init__.py` | 第 10/11 行仍导出 `Department` / `Employee` |
| `backend/app/ai/agent/oa_assistant.py` | 文件名还是 OA（内容是 ReAct 图，本身有用，只是该改名） |
| `backend/app/ai/agent/multi_agent.py` | OA 时期的多 agent 试验，未接入注册表 |
| `backend/app/api/chat_routes.py`、`api/schema/chatSchema.py` | 命名还是 OA 语境 |
| `backend/app/core/config.py` | `APP_NAME = "AI ChatKit"` |
| `backend/app/api/document_routes.py` | 第 430/461 行的注释在拿 `DELETE /employee/delete/{id}` 当风格先例——删了 OA 之后这两处注释要改 |

## 建议顺序（按「一小时能看见变化」排）

1. **#1 残留清理 + #10 去硬编码** —— 清完这个仓库的故事才自洽，是唯一还能一眼看出「抄自 OA 模板」的地方
2. **#7 日志落文件** —— 现在日志只在控制台，出故障无法倒查
3. **#5 会话 / Agent Task API** —— 体感最强的缺口（`MemorySaver` 只在进程内存，刷新即丢）
4. **#6 Qdrant 对比**、**#4 MCP** —— 押后，等有真实需求再做
