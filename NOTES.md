# Learning preferences

- 黑盒学习法：先观察运行行为、API 和输入输出，不从源码开始。
- 学习目标是以后亲自构建 ResearchPilot，重点是 Agent、RAG、LLM 后端。
- **阶段 A（看懂）已结束，现在进入阶段 B（动手）：由本人修改代码，导师做评审。**

---

# Progress

## 课程进度

| 阶段 | 课程 | 状态 |
|---|---|---|
| A | 第 1 课 系统地图 | ✅ |
| A | 第 2 课 谁负责什么 | ✅ 三情景已批改 |
| A | 第 3 课 Agent 为什么决定调用工具 | ✅ 4 个实验 + 自测 |
| A | 第 4 课 HTTP / JSON / SSE 真相 | ✅ 5 个实验 + 叙述题已批改 |
| A | 第 5 课 多轮对话靠什么记住 | ✅ 3 个实验 + 叙述题已批改 |
| **B** | **第 6 课 第一次动手：修三个 500** | ⬅ 当前 |

## 阶段 A 的三条"主线"（比知识点更重要）

1. **分层**：看到现象先问"这是哪一层的问题"。前端 / HTTP / Graph / 模型决策 / Tool / Embedding / 向量库，各层的失败长得几乎一样。
2. **静默失败**：用户看到的状态 ≠ 系统的真实状态。三种现象，同一个模式：
   - Tool 报错被吞成文本（HTTP 200、error 事件 0）
   - 检索无阈值，给了无关材料却像有依据
   - 后端重启，前端还有历史但后端已失忆
   → **以后每设计一个功能，先问："它坏掉的时候，用户能看出来吗？"**
3. **用显式状态替代隐式历史**：别让模型从 80 轮对话里"回忆"事实。把关键信息放进图的 state（如"已重试次数""本轮用过的查询"），它不占 token、而且是确定的。

## 已修正的概念（复习用）

1. **层边界**：LLM 只看到工具的**接口**（名字、描述、参数），看不到**实现**。它不知道 bge-m3、也不知道 SQLite 存在。
2. **失败观**：查不到 **≠** 报错。空库时模型收到空字符串；有库但不相关时模型收到 10 段材料。**最危险的失败不是"空"，是"非空但不相关"。**
3. **失败模式**：不是"凭空编造"，而是**"结论超出证据"**（从"手册未提及"跳到"公司不提供"）——**无根据的否定**。
4. **Graph 的两个视角**：
   - 时间顺序：模型判断 → Graph 路由 → 工具执行 → 模型再判断 → 模型写答案
   - 调用归属：FastAPI → **Graph** → 〔model 节点 → LLM〕+〔tools 节点 → Tool〕，**Graph 在最外层**
   - 分清的意义：Graph 若不是最外层的框，就管不了循环、管不了"能走哪条边"。
5. **记忆机制**：记忆 = 钥匙（`thread_id`）+ 仓库（checkpointer）+ **每轮回放**。**模型本身没有记忆。**
6. **状态码**：2xx 我做成了 · **4xx 我拒绝（但我没坏）** · 5xx 我坏了。详见 `reference/http-status-codes.md`。

## 全链路分层（含确定性标注）

| # | 谁在做 | 确定性 / 概率性 |
|---|---|---|
| 1 | 前端拼 JSON，`POST /chat/stream` | 确定性 |
| 2 | FastAPI 解析 + Pydantic 校验（`StreamInput`） | 确定性 |
| 3 | `get_agent("oa-assistant")` 取出编译好的 Graph | 确定性 |
| 4 | Graph 把该 `thread_id` 的历史消息交给 model 节点 | 确定性（编排） |
| 5 | **LLM 决定：要不要调工具、参数是什么** | 🔴 **概率性** |
| 6 | Graph 条件边读 `tool_calls`，路由到 tools 节点 | 确定性 |
| 7 | ToolNode 执行：embedding → Chroma，或查 SQLite | 确定性 |
| 8 | 结果包成 ToolMessage，回到 model 节点 | 确定性 |
| 9 | **LLM 读材料，写最终回答** | 🔴 **概率性** |
| 10 | 回答经 SSE（`token` / `message`）流回前端 | 确定性 |

## 关键实测结论（可直接用于面试/简历）

- **检索层是确定性的**：同一 query 连跑两次，结果完全相同（667 字符 / 10 段）。
- **LLM 层是概率性的**：检索结果、提示词、问题全相同，措辞仍可能不同（「手册未提及」vs「公司不提供」）。
- **推论**：确定性层可测试、可回归、可量化；概率性层**必须靠测试集 + 断言**。这就是改造 #8 的立项依据。
- **Ollama 只是 `search_handbook` 的实现细节**：Ollama 关着，`get_user_department` 照样成功。
- **Graph 的作用边界**：图上没有的节点，系统就没有那个能力。
- **流式接口的状态码是锁死的**：SSE 响应头一旦发出，就无法再改状态码。业务失败只能作为流里的一个事件传出去——所以 `/chat/stream` 在工具彻底坏掉时**仍然是 200、error 事件 0 个**。
- **三个部门写接口全部返回 500**（已复现），且 **500 的响应体只有一行 `Internal Server Error`，没有任何线索**——真正的原因只在后端日志里。

## 环境事实

- 后端必须**手动启动**才能看到日志（一键脚本用隐藏窗口启动，日志不可见）。
  改造 #5 之二起会话记忆走 psycopg，Windows 下必须带 `--reload`
  （uvicorn 0.34 只在该模式用 SelectorEventLoop，否则 psycopg 异步连接全部失败）；
  且要先有 PostgreSQL（`docker compose up -d postgres`）：
  ```powershell
  cd D:\Code\agent\ai-chatkit-master\backend
  $env:NO_PROXY = '127.0.0.1,localhost,::1'
  .\.venv-py311\Scripts\python.exe -m uvicorn main:app --app-dir app --host 127.0.0.1 --port 8001 --reload
    ⚠️ **--reload 不可信**：实测多次改了后端代码（agents.py、conversation_routes.py）页面行为不变 —— reload 看漏变更。改完后端后**手动重启**验证，别信热加载。（判别：/health 的 started_at 没变 = 没重启成功。）
  ```
- 数据库基线（动手改代码前请记住，改完要核对没被破坏）：
  - PostgreSQL（compose 的 researchpilot-pg，端口 5433）；`department` / `employee` 等 OA 表已随 OA 清理退役
  - `paper`：7 篇 / 636 块（4 篇种子语料 + 3 篇上传）
  - `conversation`：会话索引（消息本体在 LangGraph 的 checkpoint 表里）
- LangGraph checkpointer：**PostgreSQL 版已接入**（AsyncPostgresSaver，改造 #5 之二）；
  `DATABASE_URL` 不是 postgres 时降级 MemorySaver 并打 WARNING。

## 参考件（reference/）

| 文件 | 内容 |
|---|---|
| `agent-rag-system-map.html` | 第 1 课系统速查表 |
| `data-and-retrieval-roles.html` | 数据与检索职责 |
| `baseline-defects.md` | baseline 缺陷清单（**动手阶段的起点**） |
| `transformation-03-research-workflow-design.md` | 改造 #3 设计草稿 |
| `http-status-codes.md` | 状态码速查 |
