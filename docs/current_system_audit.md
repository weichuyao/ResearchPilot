# Scientific Research Harness V1：当前系统审计

> Phase 0 产物。审计日期：2026-09-23。本文只描述当前仓库中已经存在的实现，不把规划中的能力写成现状。

## 1. 审计范围与结论

本次实际阅读了以下入口与关键链路：

- 项目说明与依赖：`README.md`、`README_zh.md`、`backend/pyproject.toml`、`backend/requirements.lock`
- 后端启动与配置：`backend/run_server.py`、`backend/app/main.py`、`backend/app/core/config.py`
- 文档解析与入库：`backend/app/ai/rag/parsers.py`、`backend/app/ai/rag/ingest.py`
- 检索链：`backend/app/ai/rag/hybrid.py`、`pipeline.py`、`rerank.py`、`chromaClient.py`、`qdrantClient.py`
- LLM 与 Agent：`backend/app/ai/llm.py`、`models.py`、`agent/agents.py`、`react_assistant.py`、`research_workflow.py`
- API 与 schema：`chat_routes.py`、`document_routes.py`、`conversation_routes.py` 及对应 schema
- 数据层：`db/database.py`、现有 SQLModel 模型、repository、LangGraph checkpointer
- 测试与评测：`backend/tests/`、`qa_blackbox_test.py`、`backend/app/ai/eval/`
- 前端引用定位：`frontend/app/chat/components/CitationChips.tsx`

结论：当前系统已经是一个工程化的科研文献问答系统，而不是最小 RAG 示例。它具备多格式解析、位置标签、混合检索、交叉编码器重排、Corrective-RAG 工作流、会话持久化、文档管理和评测能力。它尚不具备独立、持久、可审计的科研对象模型，也没有把检索命中保存为 Evidence、把实验运行保存为 Run、或把 Conclusion 与来源建立数据库级关联。因此最合适的改造方式是：**保持现有问答链不变，在其旁边增加 Research Domain，并把当前结构化检索入口包装成 Literature Evidence Engine。**

## 2. 当前真实架构

### 2.1 运行时组成

| 层 | 当前实现 | 主要职责 |
|---|---|---|
| Web 前端 | Next.js / React / Ant Design | 会话、流式聊天、文档管理、引用页原文弹窗 |
| HTTP API | FastAPI | `/chat`、`/conversations`、`/documents`、`/agents`、`/health` |
| Agent | LangGraph | `research-workflow` 与 `react-assistant` 两个图 |
| LLM | LangChain 模型适配层 | DeepSeek、OpenAI、Ollama、通义及测试模型 |
| 检索 | 自研 BM25 + 向量检索 + RRF + ONNX reranker | 召回、过滤、阈值拒答、重排、相邻块补全 |
| 向量存储 | Chroma 默认，Qdrant 可切换 | 保存 chunk 文本、metadata 与向量 |
| 业务关系库 | SQLModel + SQLAlchemy Async | Paper、Collection、Conversation 元数据 |
| 会话存储 | LangGraph checkpointer | PostgreSQL 时持久化完整消息；否则 MemorySaver |
| 文档文件 | 本地上传目录 | 保存上传的原文件 |

FastAPI 在 `backend/app/main.py` 的 startup 中建业务表、启动 checkpointer，并预编译两个 Agent 图。当前 router 只有 chat、conversation、document、system 四组，没有 `/research/*` API。

### 2.2 当前数据源的职责边界

- `paper` 表保存论文级结构化属性与索引状态；它不是正文仓库。
- Chroma/Qdrant 保存切块后的正文和定位 metadata；它是检索来源。
- LangGraph checkpointer 保存某个 `thread_id` 的消息/图状态；它是会话历史来源。
- `conversation` 表只是会话列表索引，不保存消息正文。
- 上传目录保存原文件。

当前没有 ResearchQuestion、Hypothesis、Evidence、Experiment、Run、Observation、Conclusion、EvidenceRelation、审批记录或科研状态事件表。

## 3. 当前文档入库流程

```text
POST /documents
  -> 校验扩展名、文件头、大小、重复状态
  -> 文件按内容哈希落盘
  -> Paper 记录进入 pending/indexing
  -> FastAPI BackgroundTasks
  -> parse_document
  -> format-specific clean
  -> RecursiveCharacterTextSplitter(800, overlap 150)
  -> 写 Chroma 或 Qdrant
  -> BM25 缓存失效
  -> 更新 Paper 状态与 chunk_count
```

解析器注册表支持 PDF、Markdown、DOCX 和纯文本：

- PDF：pdfminer 按物理页抽取，跳过 `LTFigure` 子树；失败时回退 pypdf。会做重复页眉页脚去除和断词修复。
- Markdown/DOCX：按章节输出可定位 Section。
- TXT：按段落块输出。

切块时写入的 metadata 是：

- `source`：与 `Paper.source_file` 对接的文件名
- `paper_title`
- `page`：兼容旧代码的 0 基位置
- `page_label`：PDF 页码、章节号或块号
- `locator_prefix`：`p`、`sec` 或 `blk`
- `locator_kind`：`page`、`section` 或 `block`
- `kind`：`text` 或按数字密度粗略识别的 `table`

重要限制：当前 metadata **没有稳定的显式 `chunk_id`**，也不保留 PDF 章节标题或页内坐标。向量库自己的内部 ID 没有通过检索结果暴露为领域标识。

## 4. 当前问答流程

默认 Agent 是 `research-workflow`，真实流程为：

```text
用户问题
  -> analyze：LLM 输出结构化 ResearchPlan / 子问题 / 检索 query / facets
  -> retrieve：调用公共 retrieve()，不调用 LLM
  -> assess：交叉编码器相关性 + facet 字面覆盖，三态判定
  -> [不足且未超过 3 轮] refine：LLM 改写 query
  -> retrieve / assess 循环
  -> synthesize：LLM 基于格式化检索材料生成最终回答与引用
```

其中 `ResearchState` 是 LangGraph 的一次问答图状态，包含 messages、plan、rounds、listing、evidence verdict。这里的 `evidence` 只是 `sufficient / insufficient / absent` 一类流程判定，不是 Harness 要求的持久化 Evidence 实体。

`react-assistant` 是对照入口，通过 `list_papers`、`search_documents` 工具自行决定调用顺序。两个 Agent 共用相同底层检索能力。

## 5. 当前检索流程

公共入口是 `backend/app/ai/rag/pipeline.py::retrieve()`：

```text
query
  -> hybrid_search
       -> Chroma/Qdrant vector top-k
       -> 内存 BM25 top-k
       -> RRF 融合
       -> 可选 allowed_sources 过滤
  -> 以 vector_top1 做域外阀门（当前阈值 0.35）
  -> 跨副本内容去重
  -> ONNX cross-encoder rerank（不可用时有降级路径）
  -> 补同论文相邻上下文块
  -> Outcome(query, hits, vector_top1, rejected, ambiguous)
```

`Outcome.hits` 已是结构化结果，每项为 `(Document, origin, score)`。这是 Literature Evidence Engine 最重要的复用点，比解析 `search_documents` 的字符串输出可靠。

当前检索存在两种“分数”：

- `vector_top1` 用于知识库相关性阀门；
- cross-encoder 分数用于重排和工作流证据充分性辅助判断。

二者均不能直接证明一条材料 SUPPORT 或 CONTRADICT 某假设。

## 6. 页码、定位与引用如何产生

1. 解析阶段把 PDF 物理页转成 `Section(label="1..N", kind="page")`。
2. 切块阶段继承 `page_label`、`locator_prefix`、`locator_kind` 到每个 chunk metadata。
3. `format_hits()` 把命中格式化为：

   ```text
   [source N | 论文标题 | p.5 | relevance ...]
   原始 chunk 文本
   ```

4. synthesize prompt 要求模型只使用检索材料并输出定位引用。
5. 前端 `CitationChips.tsx` 从模型回答中扫描 `p.N` 或“第 N 页”，再用标题近似匹配调用 `/documents/resolve` 与 `/documents/{id}/passages?page=N`，展示该页已索引原文。
6. 评测脚本通过与 `format_hits()` 耦合的正则解析来源，并检查 citation 是否命中期望论文/页。

因此当前“引用”由两部分组成：**可信的检索 metadata 定位** + **LLM 在答案文本中复述该定位**。页面 chip 不是由后端返回的结构化 citation 数组驱动，而是前端再次解析自然语言答案。这对聊天可用，但不适合作为 Harness 的权威 provenance；新 Evidence 必须直接保存结构化 locator。

## 7. 当前 LLM 与 prompt 边界

- `research_workflow.py` 内包含 analyze、refine、synthesize prompt。
- analyze/refine 使用结构化 Pydantic 输出，温度为 0，并允许单独配置本地改写模型。
- synthesize 负责最终措辞，读取格式化来源。
- 现有 prompt 已强调：没找到只能说“当前知识库未找到”，不能推断事实不存在。
- 没有针对 Hypothesis、Evidence Relation、Experiment、Observation、Conclusion 的 schema 或 prompt。
- 没有任何代码可将 LLM 的科学判断经过规则验证和人工批准后写入状态库。

## 8. 当前 API 与输出格式

现有主要 API：

- `POST /chat/invoke`：返回 `ChatMessage`
- `POST /chat/stream`：SSE 输出 message/token/error/end
- `/conversations`：会话列表、历史、重命名、删除
- `/documents`：上传、列表、详情、重索引、删除、格式、标题解析、按页 passages
- `/agents`、`/health`

聊天响应 schema 是通用消息结构，最终答案仍是 Markdown 字符串；工具调用与 tool result 可附带在消息中。当前没有返回结构化 Evidence 或 Research Report 的 API 契约。

## 9. 当前持久化与迁移能力

- 业务模型使用 SQLModel，异步 engine 由 `settings.DATABASE_URL` 创建。
- `create_all` 负责新表；`_add_missing_columns` 只支持给已有表追加简单列，不是完整迁移系统。
- 现有部署使用 PostgreSQL；代码路径也能使用 SQLAlchemy 支持的异步 SQLite，但 README 中“默认 SQLite”与 `Settings.DATABASE_URL = None` 本身并不一致，实际启动依赖环境配置。
- LangGraph checkpoint 在 PostgreSQL 下持久化，否则明确降级为内存。

对 Harness 的含义：科研状态应落在业务关系库的显式表中，而不是 checkpoint。新增全新表可由 `create_all` 建立；一旦后续要改变枚举、约束或关联，当前手写补列机制不足，需在进入生产数据前决定迁移方案。

## 10. 当前测试与评测

仓库中的测试主要是可直接执行的验收脚本，不是标准 pytest suite：

- `backend/tests/api/uploadApiTest.py`：多格式上传、状态、索引、检索、重索引、删除、HTTP 错误路径。
- `backend/tests/api/uploadChatE2ETest.py`：上传到聊天的端到端链路。
- `backend/tests/db/SqlModelTest.py`：早期 SQLModel 脚本，覆盖有限。
- `qa_blackbox_test.py`：运行中 API 的黑盒验收。
- `ai/eval/run_eval.py`：LLM judge、引用正确性、语料指纹与缺席探针。
- `rank_bench.py`、`rewrite_bench.py`：确定性检索名次和 query rewrite 基准。

当前 Python 3.11 环境未安装 pytest。Phase 0 执行的无外部服务基线为：

- `python -m compileall -q app`：通过。
- 导入 `ai.rag.parsers` 与 `ai.rag.pipeline`：通过。

未在 Phase 0 运行上传 E2E、在线 LLM 评测或黑盒测试，因为它们会连接实际数据库/向量库、写入临时文档或要求服务与模型在线，不属于只读审计。

## 11. 可以直接复用的模块

1. `parse_document()` 与格式注册表。
2. `chunk_document()` 的位置 metadata 设计。
3. `retrieve()` 返回的结构化 `Outcome`。
4. BM25 + vector + RRF、source filter、reranker、相邻块补全。
5. `PaperRepository` 与 `source_file` 到论文记录的映射。
6. FastAPI 的路由、异常和依赖注入方式。
7. SQLModel/AsyncSession 基础设施。
8. LangGraph（只用于新 Controller 的流程编排，不作为科研事实存储）。
9. 现有 citation/rank/eval fixture，可用于 RAG 回归。

## 12. 需要扩展的模块

1. 新增独立 Research Domain schema 与 repository。
2. 用 `Outcome.hits` 构造 Literature Evidence，并保存原始 excerpt 与结构化 locator。
3. 为新入库 chunk 增加稳定 chunk ID；为旧索引提供内容指纹兼容值。
4. 新增 EvidenceRelation，并区分建议关系和经确认关系。
5. 新增 primary / contradiction / limitation 三路搜索策略。
6. 新增 Experiment、Run 标准导入协议、Observation 生成。
7. 新增显式状态机、审批请求和状态变更审计事件。
8. 新增受规则约束的 Hypothesis Update proposal。
9. 新增 provenance traversal 与结构化 Research Report。
10. 新增 `/research/*` API 与离线 workflow fixture。

## 13. 不应修改的模块

除非回归测试证明必须修复，以下模块在 V1 应保持行为兼容：

- 原 `/chat/invoke`、`/chat/stream` 契约和两个 Agent ID。
- 当前 `research-workflow` 的 analyze/retrieve/assess/refine/synthesize 语义。
- PDF/Markdown/DOCX/TXT 解析策略。
- hybrid retrieval、阈值、RRF、reranker 参数。
- Chroma/Qdrant 可切换能力。
- `format_hits()` 现有文本格式；评测正则依赖它。
- 文档上传、删除、重索引生命周期。
- 会话 checkpointer 与 conversation index。
- 当前前端聊天页面。

允许的最小兼容扩展是：入库 metadata 增加字段、公共检索之上新增适配器、`main.py` 注册新 router、`db.models` 导入新表。

## 14. 当前能力与 Harness 目标的差距

| Harness 目标 | 当前状态 | 差距 |
|---|---|---|
| ResearchQuestion | 无 | 需要独立实体和状态 |
| Hypothesis + lineage | 无 | 需要实体、父子关系、受控状态迁移 |
| Literature Evidence | 仅临时检索命中 | 需要持久化 excerpt、source、locator、query、rank、score |
| SUPPORT/CONTRADICT/LIMITATION | 无 | 搜索意图不能自动等于已确认科学关系 |
| Experiment / Run | 无 | 需要设计、审批和标准导入协议 |
| Observation | 无 | 需要与原始 Run 分离并可复算 |
| Conclusion | 聊天自然语言答案 | 需要强制绑定 Evidence/Observation |
| Provenance | 页面定位可追踪但未形成图 | 需要数据库关联与 traversal API |
| Human approval | chat 支持 interrupt 基础能力 | 没有领域审批记录和规则 |
| Research report | 普通回答 | 需要从权威状态确定性组装 |

## 15. 审计风险与需验证事项

- 当前语料不随仓库发布，Evidence Engine 的集成测试必须用固定小 fixture，不能依赖本地真实论文库。
- chunk ID 的兼容方案需验证 Chroma 与 Qdrant 返回一致性。
- Qdrant 适配器是否完整保留新增 metadata 需在 Phase 2 两种后端各测一次。
- citation 前端目前只正确处理 PDF 页；Harness 报告应使用结构化链接，不能继续依赖自然语言正则。
- 现有数据库迁移只适合新增简单列；状态枚举建议以字符串 + 应用校验保存，避免数据库 enum 演进困难。
- 当前测试没有统一 pytest runner。新 Harness 测试应独立、离线、使用临时 SQLite，并在后续决定是否补 pytest 依赖。

