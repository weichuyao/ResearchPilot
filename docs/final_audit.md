# Scientific Research Harness V1：最终一致性审计

审计日期：2026-09-23 至 2026-09-24。

## 1. 设计—代码—测试—Demo 对照

| 设计目标 | 代码实现 | 验证证据 |
|---|---|---|
| 七类科研对象 | `db/models/research.py` | object/repository tests |
| 关系库为 authority | SQLModel tables + transactional repository | 临时 SQLite commit 后重新建 engine 加载 |
| RAG 降级为 Literature Evidence Engine | `research/evidence_engine.py` 调用原 `retrieve()` | fixed Document test；原 `format_hits()` 回归 |
| 页码/chunk provenance | 新 chunk metadata + legacy fallback | ingest/legacy chunk ID tests |
| 三路证据搜索 | `search_strategy.py` + SearchAttempt/SearchHit | contradiction coverage tests |
| 搜索不自动确认科学关系 | EvidenceRelation `PROPOSED` | 三路关系均保持待审的测试 |
| 实验人工批准 | ApprovalRequest + Experiment transition | approval tests/API test |
| 标准结果导入 | JSON/YAML/CSV parser + payload hash | import protocol/duplicate tests |
| Run 与 Observation 分层 | deterministic `ObservationService` | group mean/delta tests |
| 受控 Hypothesis 更新 | proposal→rule validation→approval→transition | source relation/approval tests |
| Conclusion 必须有来源 | repository invariant | unsupported/model-only tests |
| 双链 provenance | `ProvenanceService` | Conclusion→Evidence→page/chunk；Conclusion→Observation→Experiment→Run |
| Research Report | deterministic `ResearchReportService` | section/source/provenance assertions |
| Human-in-the-loop | registration/import/experiment/update/conclusion gates | repository/controller/API tests |
| 离线完整 Demo | `scripts/scientific_harness_demo.py` | 实际运行成功并输出 report + JSON graph |
| 轻量科研工作台 | `frontend/app/research` + state projection API | Next production build + PostgreSQL smoke test |

## 2. Harness 自评口径

`research/evaluation.py` 实现：

- Evidence Traceability
- Citation Integrity（离线结构完整性；真实索引存在性由 RAG 集成测试验证）
- State Consistency
- Unsupported Conclusion Rate
- Contradiction Coverage
- State Fingerprint/Reproducibility

固定测试案例结果：traceability 1.0、structural citation integrity 1.0、invalid state 0、unsupported conclusion rate 0、contradiction coverage 1.0，数据库重载前后 fingerprint 一致。
State Consistency 覆盖 Question、Hypothesis、Experiment、Run 与 Conclusion；fingerprint
覆盖证据定位、实验判据、Run 指标/导入哈希及 Observation 统计等核心科研状态。

## 3. 原系统回归边界

- 未改 `pipeline.py`、`hybrid.py`、`rerank.py`、两个原 Agent、chat routes 或聊天前端。
- `ingest.py` 只增加 metadata 字段；chunk 文本、切分参数、向量、排序和格式化输出不变。
- 测试明确断言新增 metadata 不出现在既有 `[source ...]` 文本协议中。
- FastAPI 原 router 保留，仅增量注册 research router。

## 4. 已验证

```powershell
cd backend
.venv-py311\Scripts\python.exe -m pytest tests/research -q
.venv-py311\Scripts\python.exe -m compileall -q app scripts tests/research
.venv-py311\Scripts\python.exe scripts/scientific_harness_demo.py
```

此外已验证 FastAPI app 可导入，全部 Harness 路由已注册，`git diff --check` 无内容错误。

## 5. 真实环境集成验收

- 用户现有 PostgreSQL 已增量创建 18 张 Harness 表；原表和原服务未被替换。
- 针对 asyncpg 实测修复了两项 SQLite 不会暴露的问题：时间戳统一以 UTC 语义写入
  `TIMESTAMP WITHOUT TIME ZONE`；运行导入审批用 32 位十六进制短指纹作为动作键，
  并继续在审批载荷中保存、确认时校验完整 SHA-256。
- 现有 Chroma 中 72 篇论文、6818 个旧 chunk 均可读取；旧 metadata 缺少 `chunk_id`
  时，Literature Evidence Engine 会生成确定性 fallback ID。
- 在隔离的临时 Qdrant collection 中验证了新 `chunk_id` metadata 写入、相似度检索和
  round-trip，随后删除临时 collection；未覆盖现有 `papers` collection。
- 使用 PostgreSQL、真实 Chroma 语料和本地 reranker 跑通完整 HTTP 生命周期：问题、
  假设注册审批、PRIMARY/CONTRADICTION/LIMITATION 三路检索、关系人工确认、实验审批、
  结果导入审批、Observation、假设更新审批、Conclusion 审批、报告、溯源和工作流完成。
- 实测终态：Hypothesis=`PARTIALLY_SUPPORTED`、Run=`COMPLETED`、Workflow=`COMPLETE`；
  报告长度 41,611 字符，Conclusion 溯源图为 7 个节点、6 条边。
- 生命周期联动：确认首个 Run 导入后 Experiment 从 `APPROVED` 进入 `RUNNING`；形成
  Observation 后仍允许继续导入其他 Run，只有显式调用完成操作且所有 Run 已完成、至少有
  一个 Observation 时才进入 `COMPLETED`，避免首个观察过早封死多次运行实验。
- 在线自评：traceability=1.0、structural citation integrity=1.0、contradiction coverage=1.0、
  invalid state=0、unsupported conclusion rate=0。
- 验收产生的数据仅位于新增 `research_*` 表，验收结束后已清理；schema 保留。

## 6. Research Workbench 增量验收

- 新增 `GET /research/questions` 与 `GET /research/questions/{id}/state`；后者是只读 UI
  projection，不引入第二套状态存储。
- `/research` 支持项目创建/选择、Hypothesis 提交、审批处理、三路 Evidence 搜索、
  EvidenceRelation 人工确认、Experiment 设计与审批、标准 Run JSON 导入确认、
  Observation 确定性生成、Hypothesis Update、Conclusion 审批、workflow 推进、
  provenance 查看、完整性评分和 Research Report 查看。
- 现有聊天页、Agent 选择器和知识库入口保留；侧边栏只增加科研工作台入口。
- Next.js production build 完成，`/research` 被静态生成；真实运行态返回 HTTP 200，
  PostgreSQL 中创建项目后，聚合 state 的 question ID 与 workflow 均一致。
- Windows 无管理员权限时 pnpm standalone 的符号链接复制会失败；配置现会在 Windows
  本地自动关闭 standalone，而 Docker/Linux 生产构建仍自动输出 standalone。
- 烟测产生的 4 条 `research_*` 记录已清理，真实 Harness 表恢复为空。

## 7. 保留项

- 未运行依赖外部 LLM judge 的质量评测；当前评估聚焦可确定复现的结构、状态和溯源完整性。
- 按 V1 边界只做轻量工作台；Run 支持标准 JSON 表单导入，但任意日志解析、拖拽式
  lineage 编辑器、团队权限和大型项目管理仍未引入。
- 当前 schema 以新增表为主，生产数据形成后的复杂迁移仍建议引入 Alembic。

## 8. 可移交档案与部署验收

- `GET /research/questions/{id}/export` 导出 schema-versioned JSON，包含作用域内的核心
  records、关系表、审批、ResearchEvent、workflow、evaluation、Conclusion provenance
  与 Markdown report。
- 档案同时保存 state fingerprint 与覆盖完整档案内容的 SHA-256；修改任何导出字段都会
  导致 `ResearchArchiveService.verify()` 和独立校验 CLI 失败。
- `scripts/verify_research_archive.py` 可在不连接数据库的情况下核验交付档案。
- `scripts/research_harness_smoke.py` 是只读部署检查：验证 `/health`、OpenAPI 的 9 个
  必需 Harness 路由、项目列表，并在已有项目时校验 state/evaluation/export 一致性。
- 真实环境 smoke 结果：72 papers、6818 chunks、9/9 必需路由存在、0 次数据库写入。

## 9. 受控档案恢复与 Schema Audit

- `ResearchArchiveService.restore()` 只接受 schema version 正确且完整 SHA-256 校验通过的
  档案，目标 ResearchQuestion 已存在时立即拒绝，不执行覆盖或 merge。
- 记录按外键顺序恢复；Hypothesis lineage 使用拓扑顺序，循环或缺失 parent 会拒绝。
- 全部记录 flush 后重新计算 state fingerprint；与档案不一致时抛错，由 CLI 回滚整个事务。
- 恢复前会验证所有直接记录与关系外键都属于同一 ResearchQuestion；即使攻击者篡改内容后
  重新计算 SHA-256，夹带跨项目记录仍会在写库前被拒绝。审批事件也已纳入导出范围。
- `scripts/restore_research_archive.py` 要求使用者显式输入档案内的 RQ ID，防止选错文件；
  本次只在两个隔离 SQLite 数据库间验证，未向真实 PostgreSQL 写入恢复数据。
- `GET /research/system/schema` 只读检查 18 张 Harness 表、缺失列和字符串容量缩窄。
- 部署 smoke 已将 schema compatibility 纳入硬门槛。真实 PostgreSQL 结果：18/18 表兼容，
  9/9 必需路由存在，72 papers、6818 chunks。

## 10. 浏览器级 Workbench 验收

- `scripts/research_workbench_browser_smoke.py` 使用真实 Chromium/Edge 页面而非 HTTP 模拟。
- 已验证：工作台加载、研究问题 Modal、PostgreSQL 创建、workflow 初始化、Hypothesis
  Modal、注册审批出现、人工批准以及最终 `TESTABLE` 状态渲染。
- 页面无 JavaScript `pageerror`；1440×1000 截图经人工可视检查，项目栏、指标卡、
  Hypothesis 卡片、审批反馈和 workflow controller 均正常。
- 自动化早期选择器调试及最终通过运行共生成 30 条隔离 `research_*` 记录，服务停止后
  已统一清理；18 张表当前记录数恢复为 0。

## 11. 最终交付门禁

- Harness tests：33 passed。（这是 V1 交付当时的数；截至 §13 的补强，
  `pytest tests/research tests/eval -q` 为 **63 passed** —— 科研域 45 + 越界形态分类器 18。）
- Python `compileall`：通过。
- Next.js production build、lint 与 TypeScript：通过；`/research` 静态路由 42.4 kB。
- `git diff --check`：通过，仅有 Git 的 LF→CRLF 工作区提示。
- 最终真实数据库状态：18 张 Harness 表、0 条验收夹具。
- 8002 后端已恢复运行并通过只读 smoke；3000 前端 `/research` 返回 HTTP 200。

## 12. 2026-09-24 反向路径审计修复

- 通用审批端点与实验专用端点统一应用状态变化；实验/结论被拒绝时分别进入
  `CANCELLED`/`REJECTED`，不会留下已审批但实体仍停在旧状态的悬挂状态。
- 同一实体、动作和载荷的待审批请求幂等复用；同动作不同载荷会被拒绝，避免并行审批互相覆盖。
- Experiment 只能绑定已注册 Hypothesis；SearchHit、ObservationHypothesis 均增加同项目或
  实验实际检验关系校验，封堵跨项目和未测试假设的错误引用。
- Literature Evidence 必须具有合法的 page/section/block 定位器；检索结果缺少原文定位时直接
  丢弃，不再用 `?` 伪造一个结构上看似有效的页码。
- workflow 初始化会把 Question 从 `OPEN` 推进到 `ACTIVE`，完成时推进到 `ANSWERED`；证据搜索
  阶段只接受已注册假设。
- Workbench 支持 RUNNING 实验继续导入 Run 和显式完成；取消 Run 导入确认会自动拒绝审批，
  报告缓存会在状态刷新后失效，档案下载的 Blob URL 在点击调度后再释放。
- PostgreSQL 实测：同一实验成功导入 2 个 Run，Observation 后状态为 `RUNNING`，显式完成后为
  `COMPLETED`；夹具已按精确 RQ ID 清理，research 表恢复为空。

## 13. 观察解读的确认态（2026-09-24 补）

V1 原始实现有一处不对称：`EvidenceRelation` 有 `review_status`，
`ObservationHypothesisRelation` 却没有 —— 观察的**数值**是确定性算出来的，但"这条观察
支持/反驳哪个假设"同样是判断，却可以直接解锁假设状态跃迁。现已补齐：

- 该表新增 `review_status`（默认 `PROPOSED`）、`reviewed_by`、`reviewed_at`。
- `repository._require_hypothesis_basis` 与 `HypothesisUpdateService` 只计 `CONFIRMED`
  的观察关系：人工批准是必要条件，观察关系确认也是。
- 新增 `POST /observations/{id}/relations/{hypothesis_id}/review`（复合主键没有独立 id，
  用这一对定位）；工作台增加确认/驳回入口，状态投影原样带出 `review_status`。
- Research Report 的 Observations 章节只列已确认解读。

真实环境验收（PostgreSQL + 真实语料 + 本地重排）：问题 → 假设注册 → 实验批准 →
两个 Run 导入并各自批准 → 确定性观察（组间差 0.6）→ **未确认时 `/evaluate` 返回 422
`observation ... is not linked as confirmed SUPPORT`** → 确认后返回 200 生成审批请求。
只读烟测 9/9 必需路由、18/18 表兼容；验收数据已按 research 表清空，`paper` 72 行未受影响。
离线侧 63 passed（含新约束的正反两个方向）。

**一条必须记下的迁移隐患**：新列由 `_add_missing_columns` 建（`ALTER TABLE ADD COLUMN`
不带 server default），因此对**已有行** `review_status` 会是 `NULL` 而非 `PROPOSED`。
当前这些表为空所以不可见；一旦有生产数据，`NULL` 会被 `!= PROPOSED` 判成"已复核过"，
既不能确认也不能重判。方向是 fail-closed（不会误放行），但应在真实数据产生之前
补一次回填或让应用层把 `NULL` 视同 `PROPOSED`。
