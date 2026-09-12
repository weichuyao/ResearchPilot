# ResearchPilot 路线图完成情况

对照 `MISSION.md` 第 13–24 行的 10 项改造。
本表由代码实况核对得出（不是凭记忆），核对时间：HEAD = `4082c63`，工作树干净。

**总览：4 项完成、5 项部分完成、1 项未开始。**

| # | 改造项 | 状态 | 已完成的部分 | 还差什么 |
|---|---|---|---|---|
| 1 | 业务重设计（OA → 科研） | ✅ 完成（2026-09-12） | 工具层已整体换成论文检索（`list_papers` / `resolve_paper_sources` / `all_titles` / `search_documents`，全在 `ai/tools/research_tools.py`）；`DEFAULT_AGENT` 已是 `research-workflow`；前端有知识库抽屉；OA 残留已全部清除（见下文「OA 残留清理记录」） | — |
| 2 | 重写 RAG | ✅ 完成 | `parsers.py`（PDF/MD/DOCX/TXT + magic 嗅探）、`ingest.py`（800/150 切块）、`hybrid.py`（BM25 + RRF k=60）、`rerank.py`（cross-encoder，ONNX int8）、`pipeline.py`（二段式召回 → RRF → 重排 → 两信号证据评估 → 引用）、`textnorm.py`、位置标签抽象（`p.` / `sec.` / `blk.`） | — |
| 3 | LangGraph research workflow | ✅ 完成 | `research_workflow.py`（26.5 KB）：`analyze → retrieve → assess → refine → synthesize`，`MAX_RETRIEVE_ROUNDS = 3`，三态判定 `sufficient` / `absent` / `insufficient` | — |
| 4 | 接 MCP | ✅ 完成（2026-09-12 第一刀：arXiv） | `ai/tools/mcp_tools.py`：arxiv-mcp-server（stdio，uvx 拉起）经 langchain-mcp-adapters 转成 LangChain 工具，白名单只挂 search_papers / get_abstract / citation_graph（server 共 19 个工具，其中 list_papers 与本地工具重名、download/read 系列与自有入库链路冲突——按设计文档全不挂）；只接 react-assistant，research-workflow 保持纯本地语料（B/C 类评测语义不变）；`MCP_ARXIV_ENABLED` 开关 + 评估/验收入口强制关闭 | 批量重索引/文档预览等 server 端工具按需再加；容器内需在镜像里装 uvx |
| 5 | 完善 FastAPI | 🟡 部分（2026-09-12 会话已补） | `/documents` 五个端点（`POST` 202 / `GET` 列表 / `GET` 详情 / `DELETE` 204 / `POST reindex` 202）、`/health`、`/agents`、`/chat/invoke`、`/chat/stream`；**会话 API**（`GET /conversations`、`GET /conversations/{id}/messages`、`DELETE /conversations/{id}` 204）+ checkpointer 落 PostgreSQL（AsyncPostgresSaver，图惰性编译），B4 已修（见 `reference/transformation-05-conversations-design.md`）；统一异常层（4xx detail 原样 / 5xx 日志带 traceback） | **Agent Task API 刻意不做**（没有后台 agent 运行的场景，理由在设计文档第三节）；`GET /documents/formats` 已加（前端 accept 改为动态取，2026-09-12 收尾） |
| 6 | 完善数据层 | 🟡 部分（2026-09-12 Qdrant 对比已做，决策不切换） | SQLite → PostgreSQL（asyncpg）已切完；compose 里的 `postgres:16-alpine`；`paper_repo.py` 全套（含 `delete_not_in` + `path_prefix`）；checkpointer 落 PostgreSQL；**Qdrant 对比已完成**（`VECTOR_STORE` 后端选择 + 同接口适配器，名次基准三配置九轮全一致，决策默认保持 Chroma，见 `transformation-06-qdrant-design.md`；顺带查明 Chroma 实际跑在 l2 空间、阈值 0.35 的真实口径是 `1-d²/√2` 而非余弦） | Alembic 仍押后（paper / conversation / chroma 全部可重建，见设计文档开头） |
| 7 | 完善工程（异常/日志/任务状态/流式） | 🟡 部分（2026-09-12 日志已落盘） | 文档任务状态机 `pending / indexing / indexed / failed` + `error` 字段 + 前端轮询；SSE 流式输出；`requirements.lock` 锁依赖；**日志落文件**（`core/logging_config.py`：console + `RotatingFileHandler`，`backend/logs/app.log` 10MB×5，compose bind mount `./backend/logs:/app/logs`，uvicorn 日志一并进文件；配置从 `react_assistant.py` 的导入副作用里收敛出来，见 `reference/transformation-07-logging-design.md`） | 异常处理没有统一层；无请求 ID 关联（等 #5 用 thread_id 天然解决）；无结构化日志（接聚合平台时再说） |
| 8 | RAG / Agent 评测 | ✅ 完成（2026-09-12 扩容后扩充至 30 题，11 道新题全过） | `run_eval.py`（20 题 A/B/C 类 + LLM judge + 语料指纹 `verified_on` 过期告警）、`rank_bench.py`（确定性、无 LLM）、`compare_runs.py`（重复运行取均值/极差 + 可比性告警）、`reid_paper_eval_set.json` | — |
| 9 | Docker Compose 一键部署 | ✅ 完成 | `docker-compose.yml` 三服务、两个 `Dockerfile`、`requirements.lock`、`启动 Docker.cmd` + `scripts/docker-up.ps1`（含镜像源回退、`GIT_REV` 构建参数、真探活 `/health`） | — |
| 10 | 最后调整前端 | 🟡 部分（2026-09-12 大头已清） | 知识库抽屉（上传 / 列表 / 删除 / 重建索引 / 状态轮询）、SSE 流式渲染；Agent 选择器已改为从 `GET /agents` **动态取**（手工镜像已删）；默认 agent 不再硬编码（空值省略 `agent_id`，后端 `DEFAULT_AGENT` 兜底）；品牌文案已改 ResearchPilot（侧边栏 / 欢迎语 / `APP_NAME`） | 会话列表仍在 `localStorage`（属 #5 的会话持久化）；知识库抽屉 `accept` 已改为从 `GET /documents/formats` 动态取（手工镜像删除，2026-09-12 收尾）；README/README_zh 已重写为 ResearchPilot 实际形态 |

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
2. ~~**#7 日志落文件**~~ —— ✅ 2026-09-12 完成（`transformation-07-logging-design.md`）
3. ~~**#5 会话 API（含 B4 / 统一异常层）**~~ —— ✅ 2026-09-12 完成（`transformation-05-conversations-design.md`；Agent Task API 刻意不做，理由在设计文档）
4. ~~**#6 Qdrant 对比**、**#4 MCP**~~ —— ✅ 2026-09-12 完成（Qdrant 决策不切换；MCP 接 arXiv，见 transformation-04-mcp-arxiv-design.md）
5. ~~收尾候选：`/documents/formats` + 前端 accept 动态取、README 品牌更新~~ —— ✅ 2026-09-12 完成；Alembic 仍押后（等数据不再可重建，见 #6 行）
