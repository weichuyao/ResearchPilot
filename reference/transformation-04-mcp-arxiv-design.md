# 改造 #4：MCP（arXiv 检索，第一刀）

> roadmap 里 MCP 的触发条件是「真要接 arXiv / PubMed 这类**进程外**工具」——
> 本文就是那个时刻。只接 arXiv 检索，最小闭环。

## 一、架构决策

### 1.1 方向：消费，不是暴露

把 arXiv 当成进程外的工具提供方（MCP server），本项目的 agent 当 MCP 客户端。
这回答了设计文档里「不用 MCP 会怎样」：arXiv 工具不在本进程里，MCP 的标准化
发现/调用第一次有真实收益。

- server：社区维护的 **arxiv-mcp-server**（PyPI，MIT），stdio 传输，
  用 `uvx` 拉起（隔离环境，不污染项目 venv，uv 本来就在）
- 客户端：**langchain-mcp-adapters**（LangChain 官方）——把 MCP 工具转成
  LangChain `BaseTool`

### 1.2 只给 react-assistant，不给 research-workflow（关键取舍）

两个 agent 的结构差异决定了这一刀的边界：

- `react-assistant` 是 ToolNode 自由循环 —— 新工具 append 进 tools 列表即可用
- `research-workflow` 的检索是**图上的确定性节点**（retrieve 调 hybrid_search），
  没有 ToolNode 循环；接外部工具要改 analyze/assess 的三态语义
  （"语料里没有"和"arXiv 上有"是两个不同的结论，B 类题的评测口径会被破坏）

所以 research-workflow 保持**纯本地语料**（B/C 类评估语义不变），arXiv 工具
只进 react-assistant。等 workflow 要接时，那是「analyze 增加本地/外部路由」
的新设计，不是这次的顺手活。

### 1.3 生命周期与降级

- MCP server 进程由适配器在**获取工具时**拉起（stdio 子进程），配置里一条
  命令（`MCP_ARXIV_COMMAND`，默认 `uvx arxiv-mcp-server`）
- 降级要看得见：server 拉不起来 / 包没装 → **WARNING 日志 + 空工具列表**，
  agent 照常工作（与重排模型缺失静默降级同一规矩，但这次必须有日志）
- 开关：`MCP_ARXIV_ENABLED`。**评估必须关掉**（run_eval 模块级设 env）——
  评估走网络会引入不可复现的延迟与失败；TestClient 验收脚本同理
- uvx 首次运行会下载包（几秒~几十秒），之后走缓存

### 1.4 工具绑定

react_assistant 的 `tools = [list_papers, search_documents]` 变为
`base_tools + mcp_tools`。ToolNode 与 bind_tools 用同一个列表 —— 图的其余
部分不动。工具命名冲突不存在（arxiv 的工具名以 search_arxiv / download_paper
等命名），但给 react_assistant 的 instructions 不动 —— 工具自身 description
足够让模型决策（实测验证点之一）。

## 二、验证标准

1. 后端启动日志出现「MCP 工具已挂载：N 个」
2. 用 react-assistant 问"arXiv 上关于 occluded person re-ID 的最新论文" →
   模型真的调用了 arXiv 工具并返回论文标题（真实网络调用）
3. `MCP_ARXIV_ENABLED=false` 或 server 不可用时：WARNING + agent 正常答本地问题
4. run_eval 一题照常绿（证明评估路径隔离生效）
5. 本地两工具照常（没把原有检索打坏）
