# Scientific Research Harness V1：增量改造计划

> Phase 0 产物。计划以 `docs/current_system_audit.md` 所记录的真实源码为依据。Phase 0 不修改现有运行逻辑。

## 1. 总体设计决定

### 1.1 保留两条产品路径

```text
原聊天路径（保持兼容）
Question -> existing research-workflow/react-assistant -> Answer + textual citation

新增 Harness 路径
ResearchQuestion -> Research Controller -> structured state
                                   -> LiteratureEvidenceEngine -> existing retrieve()
                                   -> Evidence + provenance
                                   -> Experiment/Run/Observation
                                   -> reviewed hypothesis update
                                   -> Research Report
```

Harness 不接管、重命名或删除原 RAG。它只把现有结构化检索结果提升为可持久化 Literature Evidence。

### 1.2 权威状态与临时上下文分离

- Research repository 中的表是科研状态 authority。
- LangGraph state 只保存当前流程需要的 ID 和临时 proposal。
- LLM 不直接写状态表，不直接确认 Evidence relation，不直接把 Hypothesis 改为 SUPPORTED。
- 所有重大状态变更通过 service 规则、审批记录和 audit event 完成。

### 1.3 复用现有关系库基础设施

需求建议 SQLite；真实项目已经有 SQLModel + async SQLAlchemy，并在部署中使用 PostgreSQL。V1 将科研表加入同一业务关系库，使其在本地 SQLite 和现有 PostgreSQL 配置下都能运行，而不增加第二套数据库连接和事务边界。离线测试统一使用临时 SQLite。

这满足“关系数据库是 authority、LLM context 只是投影”的核心约束，同时避免为了指定存储品牌破坏当前部署。若最终必须物理隔离为独立 SQLite，可在 repository 接口不变的情况下换 engine；Phase 1 不先制造双数据库复杂度。

### 1.4 搜索意图与科学关系分离

Contradiction Search 命中的文段只是“反证候选”，不能仅因 query 中出现 `failure` 就自动成为已确认 CONTRADICT Evidence。设计为：

- `Evidence.search_intent`: PRIMARY / CONTRADICTION / LIMITATION
- `EvidenceRelation.relation`: SUPPORT / CONTRADICT / LIMITATION / RELATED
- `EvidenceRelation.review_status`: PROPOSED / CONFIRMED / REJECTED

报告的正式 Evidence For/Against 只使用已确认关系；候选材料单独显示待审。

### 1.5 原文与总结分离

Literature Evidence 至少同时保存：

- `statement`：对证据的结构化表述，可由人或模型提出；
- `excerpt`：检索命中的原始文本，不允许被 LLM 覆盖；
- source 与 locator：paper ID/source file/title、page/section/locator、chunk ID；
- retrieval provenance：query、search intent、rank、origin、score、检索时间。

## 2. 核心对象与约束

### 2.1 ResearchQuestion

- 状态：OPEN -> ACTIVE -> ANSWERED -> ARCHIVED。
- ANSWERED 需要至少一个已批准 Conclusion。
- ARCHIVED 不接受新的直接子对象，除非先恢复。

### 2.2 Hypothesis

- 保存 `parent_hypothesis_id` 形成 lineage。
- 状态采用字符串 enum，由 service 校验跃迁。
- 新建只允许 PROPOSED；正式注册、重大状态更新需要 approval。
- SUPPORTED 不允许由 LLM proposal 直接落库。

### 2.3 Evidence 与 EvidenceRelation

- Evidence type：LITERATURE / EXPERIMENTAL / DERIVED_ANALYSIS / MODEL_HYPOTHESIS / HUMAN_NOTE。
- LITERATURE 必须有 `source_id/source_type`、原始 excerpt 和 locator；PDF 必须有 page。
- MODEL_HYPOTHESIS 永远不能伪装成 LITERATURE 或 EXPERIMENTAL。
- EvidenceRelation 使用独立表表达多对多关系，不把 ID 数组当唯一关系来源。

### 2.4 Experiment / Run / Observation

- Experiment 通过关联表绑定一个或多个 Hypothesis。
- Experiment 从 DRAFT 到 APPROVED 需要人工审批，未批准不能进入 RUNNING。
- Run 标准导入只接收 JSON/YAML/CSV 的明确 schema；原始导入 payload 与 artifact path 保留。
- Observation 引用一个或多个 Run，统计量由确定性代码计算；解释文本与测量结果分开。

### 2.5 Conclusion

- 数据库写入前强制至少绑定一个已确认 Evidence 或 Observation。
- 保存 supporting/contradicting Evidence 关联和 Observation 关联。
- 正式进入报告必须通过人工审批。
- confidence 是受限枚举或范围值，不使用没有口径的自由文本。

### 2.6 ApprovalRequest 与 ResearchEvent

虽然不是七个科学对象之一，但它们是实现 Human-in-the-loop 与可追踪状态变化所必需的最小控制对象：

- ApprovalRequest：entity、action、before/after proposal、状态、reviewer、reason、timestamps。
- ResearchEvent：actor、entity、event type、before/after、关联 approval、timestamp。

## 3. 稳定 ID 与 provenance 方案

### 3.1 实体 ID

使用带类型前缀的 UUID/ULID 字符串，例如 `RQ-...`、`H-...`、`E-...`、`EXP-...`、`RUN-...`、`O-...`、`C-...`。数据库主键不依赖展示顺序。

### 3.2 chunk ID 兼容

- 新入库：在 `chunk_document()` 分块完成后，为每个 chunk 写稳定 `chunk_id`，输入包含 source、locator、chunk 内容指纹和同页序号。
- 旧索引：Evidence Engine 在 metadata 无 `chunk_id` 时使用同一算法按现有字段与内容生成 legacy-compatible ID，不要求立即全库重建。
- ID 只标识原始 chunk，不包含检索分数或 query。

### 3.3 provenance traversal

`get_provenance(entity_id)` 由 repository 关联确定性遍历，不由 LLM 生成：

- Conclusion -> Evidence -> Paper/source -> locator -> chunk/excerpt
- Conclusion -> Observation -> Experiment -> Runs -> imported metrics/artifacts
- Hypothesis -> parent lineage、EvidenceRelation、Experiment links、status events

返回结构化节点与边；Markdown 树只是它的展示投影。

## 4. 分阶段实现顺序

### Phase 0：源码审计（本阶段）

- 输出当前系统审计与本计划。
- 只执行无副作用的 compile/import smoke test。
- 不改现有运行逻辑。

退出条件：文档明确现状、复用点、禁改点、文件清单、冲突和测试基线。

### Phase 1：Research Object Model + State Store

1. 定义 enum、七个核心对象、关系表、ApprovalRequest、ResearchEvent。
2. 将新表纳入现有 SQLModel metadata。
3. 实现 repository 的事务化 CRUD、关联校验和状态跃迁。
4. 用临时 SQLite 写完全离线的 object/state tests。
5. 不接 LLM，不接 RAG，不加复杂 controller。

退出条件：非法状态、缺来源 Evidence、无来源 Conclusion、Observation 引用不存在 Run 等均被拒绝；持久化可重载。

### Phase 2：Literature Evidence Engine

1. 封装现有 `retrieve()`，输出 Evidence draft，而不是答案。
2. 解析 `Document.metadata`，保存 excerpt 与检索 provenance。
3. 增加稳定 chunk ID，新旧索引兼容。
4. 保持 `search_documents()` 与聊天入口原样。
5. 增加使用 fake retriever / 固定 Document 的离线集成测试，以及原 RAG 格式回归。

退出条件：一个 PDF hit 能稳定生成带 paper/source/page/chunk/excerpt 的 Evidence，原 `format_hits()` 输出不变。

### Phase 3：Hypothesis-Evidence Relation 与反证搜索

1. Primary、Contradiction、Limitation 三组透明 query strategy。
2. 保存每次 search attempt，覆盖度可审计。
3. 创建 PROPOSED relation，人工/规则确认后才用于正式判断。
4. Evidence API 按 FOR / AGAINST / LIMITATIONS / RELATED / PENDING 分组。

退出条件：配置要求反证搜索时，三个 channel 均有执行记录；没有把搜索意图误当科学关系。

### Phase 4：Experiment / Run / Observation

1. Experiment 设计校验和审批。
2. JSON 为首个标准导入协议；随后增加 YAML/CSV adapter。
3. 保存原始 payload hash、metrics、environment、artifacts。
4. 从多个 Run 确定性计算 mean/delta/count 等 Observation 字段。

退出条件：未批准实验不能运行；未知 experiment/run 导入失败；Observation 可追到全部 Run。

### Phase 5：Research Controller + Human Approval

1. 实现显式 workflow state 与允许跃迁。
2. LangGraph 只编排 ID 和节点，不保存权威对象正文。
3. 在注册 Hypothesis、批准 Experiment、确认 import、重大 hypothesis update、批准 Conclusion 处创建 interrupt/approval。
4. 每次变更写 ResearchEvent。

退出条件：流程可暂停、拒绝、编辑、恢复；绕过 approval 的写入被 service 阻止。

### Phase 6：Hypothesis Update + Research Report

1. Scientific Reasoner 只生成 `HypothesisUpdateProposal`。
2. 规则层校验引用 ID、关系确认状态、Observation 完整性和允许跃迁。
3. 人工批准后 service 更新状态。
4. Report 从 repository 确定性组装章节，LLM 只可润色非事实连接语。

退出条件：LLM 不能自行落库；报告中的正式 Conclusion 100% 有来源。

### Phase 7：API + Demo

1. 增量注册 `/research`、`/hypotheses`、`/experiments` 路由。
2. 保持现有聊天前端；只提供最小 Harness API 与 CLI/demo script。
3. 使用固定 ReID fixture 演示完整链路和 provenance。

退出条件：需求中的建议 API 有等价实现；离线 demo 不需要真实 LLM 也可跑状态机。

### Phase 8：一致性审计

对照设计、数据库 schema、API、测试和 demo，计算 traceability、citation integrity、state consistency、unsupported conclusion rate、contradiction coverage、reload reproducibility。

## 5. 拟新增文件清单

文件名可在实现时因 SQLModel 循环依赖作小幅合并，但职责不变。

```text
docs/
  current_system_audit.md                 # 已新增（Phase 0）
  modification_plan.md                    # 已新增（Phase 0）
  project_explanation.md                  # Phase 8

backend/app/research/
  __init__.py
  enums.py
  schemas.py                              # API/input/proposal schemas，不作为表模型
  validators.py                           # 跨对象不变量与状态迁移
  evidence_engine.py                      # 现有 retrieve() -> Evidence draft
  search_strategy.py                      # primary/contradiction/limitation query
  hypothesis_service.py
  experiment_service.py
  observation_service.py
  conclusion_service.py
  approval_service.py
  provenance.py
  controller.py
  report.py

backend/app/db/models/
  research.py                             # 七类对象及最小关系/审计表

backend/app/db/repository/
  research_repository.py

backend/app/api/
  research_routes.py

backend/app/api/schema/
  researchSchema.py

backend/tests/research/
  __init__.py
  conftest.py                             # 临时 SQLite engine/session fixture
  test_models.py
  test_repository.py
  test_state_transitions.py
  test_evidence_engine.py
  test_contradiction_search.py
  test_experiment_import.py
  test_observation.py
  test_provenance.py
  test_workflow.py
  test_report.py
  fixtures/
    research_case.json
    metrics_run_001.json
    metrics_run_002.json
    metrics_run_003.json

backend/scripts/
  scientific_harness_demo.py
```

## 6. 拟修改文件清单

| 文件 | 最小修改 | 阶段 |
|---|---|---|
| `backend/app/db/models/__init__.py` | 导入 research tables，确保 create_all 可见 | 1 |
| `backend/app/ai/rag/ingest.py` | 新 chunk 写稳定 `chunk_id` metadata | 2 |
| `backend/app/main.py` | 注册 research router | 7 |
| `backend/pyproject.toml` | 仅在确定测试 runner 后增加必要测试依赖 | 1 |
| `backend/requirements.lock` | 若增加测试依赖，同步固定版本 | 1 |
| `README.md` / `README_zh.md` | 增加 Harness 定位、API、demo 与迁移说明 | 7/8 |
| `reference/roadmap-status.md` | 记录各 Phase 状态 | 每阶段 |

现阶段不计划修改 `pipeline.py`、`hybrid.py`、`rerank.py`、`research_workflow.py`、现有 chat/document routes 或前端聊天组件。

## 7. API 规划

在需求建议基础上补足审批和 provenance：

```text
POST /research/questions
GET  /research/questions/{id}
POST /research/questions/{id}/hypotheses
POST /hypotheses/{id}/registration-proposals
POST /hypotheses/{id}/evidence/search
GET  /hypotheses/{id}/evidence
POST /evidence-relations/{id}/review
POST /experiments
POST /experiments/{id}/approve
POST /experiments/{id}/runs/import
POST /experiments/{id}/runs/{run_id}/confirm
GET  /experiments/{id}
POST /hypotheses/{id}/evaluate
POST /approvals/{id}/approve
POST /approvals/{id}/reject
POST /approvals/{id}/edit
GET  /research/questions/{id}/report
GET  /research/entities/{id}/provenance
```

所有 mutation 经过 service；route 不直接操作 SQLModel 表。

## 8. 测试策略与命令规划

### 离线核心测试

建议在 Phase 1 引入 pytest 后使用：

```powershell
cd backend
.venv-py311\Scripts\python.exe -m pytest tests/research -q
```

使用临时 SQLite、fake LLM、固定 Documents；不得访问真实语料、外部 API 或生产数据库。

### RAG 回归

```powershell
cd backend
.venv-py311\Scripts\python.exe -m compileall -q app
.venv-py311\Scripts\python.exe app/ai/eval/rank_bench.py
.venv-py311\Scripts\python.exe tests/api/uploadApiTest.py
```

在线 LLM eval 与黑盒验收单独标记，不混入离线单元测试。

### Harness 验收指标

- Evidence Traceability：正式 claim 可追到 source locator 的比例。
- Citation Integrity：LITERATURE Evidence 的 source 与 locator 在索引中存在。
- State Consistency：非法状态跃迁计数为 0。
- Unsupported Conclusion Rate：必须为 0。
- Contradiction Coverage：要求反证搜索的 hypothesis 有对应 search attempt。
- Reproducibility：同一数据库快照重载后 provenance graph 等价。

## 9. 与真实源码的冲突及调整

### 冲突 A：需求写 `src/`，项目实际是 `backend/app/`

调整：新增 `backend/app/research/`，沿用现有 top-level import 与启动方式，不引入第二个源码根。

### 冲突 B：需求建议 SQLite，现有部署主路径是 PostgreSQL

调整：使用现有 async SQLModel repository，支持 SQLite 测试/本地与 PostgreSQL 部署。科研状态仍是关系库 authority，不放入 LangGraph messages。

### 冲突 C：当前没有 chunk ID

调整：新 chunk 增加稳定 ID，旧 chunk 由 Evidence Engine 确定性补算，不强迫用户立即重建语料。

### 冲突 D：当前 `ResearchState` 名称已被占用

调整：现有类型保留；新长期状态使用 `ResearchWorkflowSnapshot` 或 `ResearchProjectState`，避免导入和概念混淆。

### 冲突 E：当前 citation 是自然语言文本

调整：保留聊天格式；Harness Evidence/Report 使用结构化 locator 与 provenance API，不从 LLM 输出反向解析权威来源。

### 冲突 F：当前 `search_documents` 返回字符串

调整：Evidence Engine 直接调用已存在的结构化 `retrieve()`，不解析工具字符串，也不改变原工具。

### 冲突 G：当前测试不是 pytest suite

调整：新 Harness 测试独立采用标准 runner；旧验收脚本继续保留并作为集成回归运行。

### 冲突 H：当前 schema migration 能力有限

调整：Phase 1 只新增表，避免修改既有表；状态用可演进字符串并由应用层校验。生产数据形成后再评估 Alembic，不在 V1 先引入重型迁移体系。

### 冲突 I：Contradiction query 不能证明 CONTRADICT

调整：搜索 channel 只产生候选关系，必须经过 assessment/human review 才成为正式关系，防止把关键词命中冒充科学反证。

## 10. Phase 1 开始前的确认点

本计划默认以下选择：

1. 科研状态复用现有业务关系库连接，而不是另建一份固定路径 SQLite。
2. 新测试引入 pytest，并锁定版本；旧脚本不改写。
3. V1 API-first + CLI demo，不做大型 Harness 前端。
4. Evidence relation 需要确认态；检索结果默认只是候选。
5. 新 ID 使用带类型前缀的字符串，不复用数据库自增整数。

如这些默认选择可接受，下一步按 Phase 1 只实现对象模型、repository 与离线测试，不接 LLM/RAG。

