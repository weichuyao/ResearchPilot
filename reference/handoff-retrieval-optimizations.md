# 交接总纲：ResearchPilot 项目全貌 + 当前任务（写给新会话）

> 交接时间：2026-09-12 深夜。**先读这份建立全局，再按第五节干活。**
> 项目的一切设计理由都在 `reference/` 下，本文件是入口地图。

## 一、这个项目是什么、要做到什么程度

**Mission**（详见 `MISSION.md`）：把 AI-ChatKit（OA 企业助手模板）改造成
**ResearchPilot——面向 ReID 方向科研文献的研究助手**，并把每处改动讲清楚
"为什么"，最终成为能扛住面试追问的简历项目。

**学习模式**：用户本人是学习者，AI 按导师/承包商角色干活；硬性规矩是
**任何改造先写设计文档（为什么/代价/验证标准），再动手，最后用数据验证**。
设计文档全在 `reference/transformation-*.md`，决策总账在
`reference/design-decisions.md`（10 个决策，全部带实测数据）。

**铁律**（违反会毁掉这个项目的价值）：
1. 先测量后改：任何优化必须有护栏验证（rank_bench + run_eval 30 题不退）
2. 把测量缺陷和系统缺陷分开：历史上 3 次"指标说坏了"其实都是尺子坏了
3. 降级必须看得见：任何静默降级都要打日志/暴露状态
4. 评估集期望值依赖语料状态：语料一变，B 类题必须复核（有探针机制辅助，
   见下文）

## 二、十项改造全部完成（当前形态）

| # | 改造 | 状态 |
|---|---|---|
| 1 | 业务重设计 | ✅ OA 残留清零，品牌「科研领航」（前端）/ResearchPilot（代码标识） |
| 2 | RAG 重写 | ✅ pdfminer 版面感知解析、BM25+RRF 混合、交叉编码器重排（ONNX int8）、连字还原、页码引用 |
| 3 | Research Workflow | ✅ Corrective-RAG 变体（analyze→retrieve→assess→refine→synthesize，轮次预算 3，三态判定） |
| 4 | MCP | ✅ arXiv 检索（只进 react-assistant），白名单 7 工具，uvx --offline 自动降级 |
| 5 | FastAPI | ✅ 文档五端点 + 会话三端点 + /agents + /formats + 统一异常层 |
| 6 | 数据层 | ✅ PostgreSQL（业务表 + LangGraph checkpointer 落库）；Qdrant 已对比（决策保持 Chroma） |
| 7 | 工程化 | ✅ 日志落盘（轮转）、状态机、requirements.lock、统一异常层 |
| 8 | 评估 | ✅ 30 题评估集 + 确定性名次基准 + 探针机制 + 阈值校准脚本 |
| 9 | Docker | ✅ api/pg/web 三服务 + 可选 qdrant |
| 10 | 前端 | ✅ 苹果风主题、agent/格式动态取、会话走后端、重命名、多选上传、引用 chips |

**近期新增能力**（2026-09-12）：**检索优化三项全部落地**（结构化元数据 +
年份过滤、相邻块两轮分配 + 检索层内容去重、表格块标注——全量重索引后
kind 6818/6818 块（table 108）、year 21/73、external_id 10/73，见决策十
及后续一/二/三）；语料 80→73 篇（清理 7 对重复入库 + ingest 层内容指纹
防重）；arXiv 编号→年份解码与年份筛选（两个模式都有）、引用可点击看原文、
主题订阅、综述生成入口、后端黑盒测试脚本 `qa_blackbox_test.py`。

## 三、检索优化三项（✅ 已全部完成，2026-09-12）

- **项 1 结构化元数据** ✅（`cb2d46e` 代码 + `dc32ec8` 重索引落地：元数据回填
  `refresh_paper_metadata` 挂进索引后台任务收口，上传/重索引两路径一致）
- **项 2 相邻块** ✅（两轮分配：每命中先各拿一个续块 i+1、有剩再补前块，
  防头部命中独占 cap；配检索层内容去重——语料 7 对重复副本曾把候选池
  减半并瓜分名额，A08 由此根治，`a9a4418`）
- **项 3 表格标注** ✅（重索引后 108 个 table 块，`[experimental table]`
  前缀已修活——WIP 提交里它是死代码）
- 收尾时的三个疑问全部闭合：A08 端到端 GROUNDED、A05 锚点经修订
  （`maximal gradient magnitude`）全绿、B05 复跑判定为评委噪声。
  全量护栏：30 题 A/B/C verdict 全对、evidence_recall 1.0、黑盒 11/11。
  详见决策十及"后续之一/二/三"。

## 四、改完之后的全项目脉络（下一步方向）

1. ~~**收尾当前三项**~~ —— ✅ 2026-09-12 完成（30 题全量不退 + rank_bench +
   决策十；`verified_on` 已复核更新到 73 篇/6818 块）
2. **语料稳定后补评估**：用户还在增删文献——每轮变动后跑探针（B 类缺席
   告警）+ 阈值校准脚本（`calibrate_threshold.py`，当前 0.35 是 4 篇时代
   校准的，语料稳定后建议重校）
3. **第二梯队功能候选**（用户已知悉，未排期）：原文定位阅读器（PDF 第 X 页
   跳转）、相关论文推荐（引用图/向量共现）、语料仪表盘（年份/主题分布，
   年份解码数据现成）
4. **第三梯队（需设计文档 + 评估扩充）**：实验数字精确检索（表格标注是
   其前置，已铺）、对话内指代解析
5. **MCP 扩展候选**：download/latex 系列 server 工具与自有入库链路的
   整合、暴露自身为 MCP server、容器内 uvx
6. **长期工程债**（书面押后，别主动做）：Alembic、Agent Task API、
   文档预览、鉴权、结构化日志、语义缓存

## 五、环境须知（关键，血泪换来的）

| 事项 | 现状与命令 |
|---|---|
| 后端 | **8002 端口**（不是 8001）：`cd backend && NO_PROXY="127.0.0.1,localhost,::1" .venv-py311\Scripts\python.exe run_server.py` |
| **为什么必须用 run_server.py** | Windows 裸 `python -m uvicorn` = psycopg PoolTimeout（psycopg 异步只支持 Selector 循环，uvicorn 0.34 只在 `--reload` 子进程自带 Selector）；run_server.py 在 `uvicorn.run` 前设策略，无 --reload 依赖 |
| **热加载不可信** | --reload 实测多次看漏变更（并发编辑还会崩 worker）。改完后端**重启进程**，用 `/health` 的 `started_at` 核对（git_rev 是请求时读的，不能判进程新旧） |
| 8001 僵尸 | 杀不掉的旧服务（PID 归属错乱，疑似 WSL/docker 转发层），跑旧代码。**别往 8001 发请求**；用户重启电脑清理 |
| 8000 端口 | 有个用户其他项目的服务（曾被误杀过一次，用户已重启）——**清理进程时别碰 8000** |
| 前端 | 3000，`pnpm dev`；`.env.local` 指向 8002。改 .env.local 要重启前端 |
| MCP | 依赖 uvx 拉 pypi：网络不通时自动降级 `--offline`（缓存在，实测可用）；挂载失败 = WARNING + 仅本地工具 |
| 评估 | `run_eval.py`（30 题，--only 调试）；`rank_bench.py`（确定性名次）；探针告警在 run_eval 启动时打印 |
| B 类复核 | 语料变更 → `verified_on` 过期告警 → 人工核对 B 类期望值 → `--mark-verified` |
| 批量上传 | `scripts/batch_upload.py <文件夹> --wait`；用户真实数据在库里，**测试只允许自建自删** |

## 六、本项目特有的坑（全部踩过，别再踩）

1. 修竞态的代码会成为新的竞态源（abort effect 掐死新会话的流）
2. 改 `format_hits` 格式必须同步 `run_eval.py` 的 `_SOURCE_RE`（掉过 0.25）
3. 列表推导重新绑定：`items = [过滤]` 后 `d["items"]` 不变（30 题静默变 20）
4. bash heredoc 写正则会吃 `\b`（正则按行号写入或用 `(?<!\d)` 形式）
5. 跨事件循环复用 async engine = 静默炸（应用外访问数据库自建 engine）
6. Chroma 的 relevance 实际是 l2 换算（1-d²/√2），不是余弦——阈值 0.35
   的口径，Qdrant 适配器已对齐
7. git_rev 在 `/health` 里是请求时读的，不能用来判断进程新旧；用 started_at
8. Windows 裸 `python -m uvicorn` 起服务 = psycopg PoolTimeout（Proactor 循环 +
   psycopg 异步不兼容，uvicorn 0.34 只在 `--reload` 子进程自带 Selector）；
   用 `run_server.py`，别再加 `--reload` 绕。启动失败先看是不是这个

## 七、用户偏好

- 界面全中文、苹果风简约（已完成，用户在持续体验中反馈细节）
- 沟通直接说结论和证据，别绕
- 用户会自己在页面上测试并截图报问题——他的实测是最有效的 bug 发现渠道
