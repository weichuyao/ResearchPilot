# ResearchPilot 路线图完成情况

对照 `MISSION.md` 第 13–24 行的 10 项改造。
本表由代码实况核对得出（不是凭记忆），核对时间：HEAD = `4082c63`，工作树干净。

**总览：4 项完成、5 项部分完成、1 项未开始。**

| # | 改造项 | 状态 | 已完成的部分 | 还差什么 |
|---|---|---|---|---|
| 1 | 业务重设计（OA → 科研） | ✅ 完成（2026-09-12） | 工具层已整体换成论文检索（`list_papers` / `resolve_paper_sources` / `all_titles` / `search_documents`，全在 `ai/tools/research_tools.py`）；`DEFAULT_AGENT` 已是 `research-workflow`；前端有知识库抽屉；OA 残留已全部清除（见下文「OA 残留清理记录」） | — |
| 2 | 重写 RAG | ✅ 完成 | `parsers.py`（PDF/MD/DOCX/TXT + magic 嗅探）、`ingest.py`（800/150 切块）、`hybrid.py`（BM25 + RRF k=60）、`rerank.py`（cross-encoder，ONNX int8）、`pipeline.py`（二段式召回 → RRF → 重排 → 两信号证据评估 → 引用）、`textnorm.py`、位置标签抽象（`p.` / `sec.` / `blk.`） | — |
| 3 | LangGraph research workflow | ✅ 完成 | `research_workflow.py`（26.5 KB）：`analyze → retrieve → assess → refine → synthesize`，`MAX_RETRIEVE_ROUNDS = 3`，三态判定 `sufficient` / `absent` / `insufficient` | — |
| 4 | 接 MCP | ⬜ 未开始 | 仓库里 **0 个** mcp 文件 | 全部。押后到真要接 arXiv / PubMed 时再做 |
| 5 | 完善 FastAPI | 🟡 部分 | `/documents` 五个端点（`POST` 202 / `GET` 列表 / `GET` 详情 / `DELETE` 204 / `POST reindex` 202）、`/health`、`/agents`、`/chat/invoke`、`/chat/stream` | **会话 API 完全没有**——数据库里没有 conversation / message 表，前端会话全靠 `localStorage`；Agent Task API 没有 |
| 6 | 完善数据层 | 🟡 部分 | SQLite → PostgreSQL（asyncpg）已切完；compose 里的 `postgres:16-alpine`；`paper_repo.py` 全套（含 `delete_not_in` + `path_prefix`） | **Qdrant 对比没做**（只有 Chroma）；迁移还是手写 `ALTER TABLE` shim（`_add_missing_columns`），没上 Alembic |
| 7 | 完善工程（异常/日志/任务状态/流式） | 🟡 部分 | 文档任务状态机 `pending / indexing / indexed / failed` + `error` 字段 + 前端轮询；SSE 流式输出；`requirements.lock` 锁依赖 | **日志没落文件**——`oa_assistant.py`（现 `react_assistant.py`）里有一处 `logging.basicConfig`，但没有任何 FileHandler，容器一重启日志就没了；异常处理没有统一层 |
| 8 | RAG / Agent 评测 | ✅ 完成 | `run_eval.py`（20 题 A/B/C 类 + LLM judge + 语料指纹 `verified_on` 过期告警）、`rank_bench.py`（确定性、无 LLM）、`compare_runs.py`（重复运行取均值/极差 + 可比性告警）、`reid_paper_eval_set.json` | — |
| 9 | Docker Compose 一键部署 | ✅ 完成 | `docker-compose.yml` 三服务、两个 `Dockerfile`、`requirements.lock`、`启动 Docker.cmd` + `scripts/docker-up.ps1`（含镜像源回退、`GIT_REV` 构建参数、真探活 `/health`） | — |
| 10 | 最后调整前端 | 🟡 部分（2026-09-12 大头已清） | 知识库抽屉（上传 / 列表 / 删除 / 重建索引 / 状态轮询）、SSE 流式渲染；Agent 选择器已改为从 `GET /agents` **动态取**（手工镜像已删）；默认 agent 不再硬编码（空值省略 `agent_id`，后端 `DEFAULT_AGENT` 兜底）；品牌文案已改 ResearchPilot（侧边栏 / 欢迎语 / `APP_NAME`） | 会话列表仍在 `localStorage`（属 #5 的会话持久化）；知识库抽屉 `accept` 扩展名仍是后端解析器注册表的手工镜像（改造 #5 设计文档第十节早就标了，建议加 `GET /documents/formats`） |

## OA 残留清理记录（2026-09-12，对应上表 #1 / #10）

**删掉的文件**（9 个）：

- `api/department_routers.py`、`api/employee_routers.py` —— OA 路由，`main.py` 已摘掉挂载
- `db/models/department.py`、`db/models/employee.py`、`db/repository/department_repo.py`、`db/repository/employee_repo.py`
- `ai/agent/multi_agent.py` —— OA 时期的 langgraph-supervisor 试验（数学/编程双 agent），**注意：它其实已接在注册表里**（key `multi-agent-supervisor`），交接文档说"未接入注册表"不准确，删除时连注册表一起清的
- `tests/rag/importRag.py` —— 引用已不存在的 `EmployeeHandbook.pdf`，跑不了
- `tests/rag/queryChroma.py` —— 唯一还在用 OA handbook collection 的死脚本

**改名**：`ai/agent/oa_assistant.py` → **`react_assistant.py`**，注册 key `oa-assistant` → **`react-assistant`**。它是 ReAct 自由循环对照组，本身有用，只是名字错了。`run_eval.py --agent` 的默认值已跟着改。⚠️ **旧的评估报告 JSON 里 `summary.agent` 仍是 `oa-assistant`** —— 那是历史数据，不要改；`compare_runs.py` 按名字分组，新旧各归各的组正合适。

**顺手修的活残留**：`chromaClient.py` 里的 `hand_book_vector_store` 句柄带着
`create_collection_if_not_exists=True`，意味着**每次启动都会把废弃的 "handbook" collection 重建出来**。句柄已删；磁盘上 Chroma 里遗留的旧 collection 数据不受影响（想清掉就重建 CHROMA_PATH）。

**判定不改的**：`chat_routes.py` / `chatSchema.py` 文件名保留 —— 内容是通用聊天管道，没有任何 OA 语义，命名风格与 `documentSchema.py` 一致；`research_tools.py` / `paper.py` / `collection.py` / `paper_repo.py` 里提到 OA 的**注释**保留 —— 那是踩坑记录（缺陷 A2/A5/A6 的出处），不是残留。

**新增**：`GET /agents`（`api/system_routes.py`）—— 注册表的只读视图，前端 agent 下拉从这取，消灭「下拉里能选、后端不认识」这类镜像漂移。

**前端去硬编码**（对应上表 #10）：`AgentSelector.tsx` 改为挂载时拉 `GET /agents`，拿不到就显示「后端 agent 列表不可用」而不是兜底造假数据；`layout.tsx` 的 `agentId` 初始为空、由列表回填第一个选项；`useStreamChat.ts` 在 `agentId` 为空时省略 `agent_id` 字段（发空字符串会让 `get_agent("")` 500）。

## 建议顺序（按「一小时能看见变化」排）

1. ~~**#1 残留清理 + #10 去硬编码**~~ —— ✅ 2026-09-12 完成
2. **#7 日志落文件** —— 现在日志只在控制台，出故障无法倒查
3. **#5 会话 / Agent Task API** —— 体感最强的缺口（`MemorySaver` 只在进程内存，刷新即丢）；做的时候顺手修掉 `baseline-defects.md` B4（前后端会话状态不一致的根因就是它）
4. **#6 Qdrant 对比**、**#4 MCP** —— 押后，等有真实需求再做
