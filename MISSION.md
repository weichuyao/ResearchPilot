# Mission: 从 baseline 走到 ResearchPilot，并把它变成简历上的项目

## Why
把当前能运行的 AI ChatKit 逐步改造成 ResearchPilot——一个科研文献与技术资料研究 Agent。
原项目只是**工程骨架**；主要业务逻辑与核心 Agent/RAG 必须变成你自己的。

学习目的（2026-09-10 明确）：

1. 掌握 **Agent 开发必备的知识**，重点是 Agent / RAG / LLM 后端调用链。
2. 把 ResearchPilot **写进简历**，并且能扛住面试追问——不是"让 AI 帮我改完"。
3. 判断标准：能讲清楚每一处改动的**为什么**，而不只是它能不能跑。

## 最终形态：ResearchPilot 的 10 项改造

1. **重新设计业务**：OA 企业助手 → 科研文献研究助手。
2. **重写 RAG**：PDF/DOCX/Markdown → metadata → chunk → embedding → retrieval → rerank → citation。
3. **增加 LangGraph research workflow**：问题分析 → 检索 → 判断证据充分性 → 工具调用 → 补充检索 → 综合回答。
4. **增加 MCP**：文件、搜索、数据库等工具通过 MCP Server 接入。
5. **完善 FastAPI**：知识库、文档、会话、Agent Task、Health 等 API。
6. **完善数据层**：SQLite → PostgreSQL；Chroma 先保留，之后比较 Qdrant。
7. **完善工程**：异常、日志、任务状态、流式输出。
8. **增加 RAG/Agent Evaluation**：Recall@K、citation accuracy、tool success rate、bad cases。
9. **Docker Compose 一键部署**。
10. **最后调整前端**，让它成为你自己的产品。

起步阶段发现的 baseline 缺陷（见 `reference/baseline-defects.md`）也属于优化范围，在这条路径上一并修掉。

## Success looks like

- 能独立启动、观察并排查 Agent/RAG 应用的每一层。
- 能解释聊天请求、Tool 调用、RAG 检索、SSE 各自的职责与边界。
- 能亲手把 OA Assistant 逐步替换为 ResearchPilot，并解释每处改动的取舍。
- 能不看代码画出完整调用链；遇到故障先定位到"哪一层"。
- 能用面试语言讲清楚：为什么这样设计、替代方案是什么、怎么验证。

## 教学方式（延续黑盒路径）

每一块都走同一条循环：

```
观察（跑起来看现象）→ 原理（讲透为什么）→ 动手（你改，我 review）→ 验收（按面试标准追问）
```

**硬性规则：任何一块改造，先由你讲清楚为什么这么改，再动手。**

## Constraints

- 学习阶段不改业务源码；动手改造阶段由你本人修改，我做评审。
- 黑盒优先：运行、接口、输入输出与实验先于读源码；源码只用来验证你的判断。
- 学习重点是 Agent/RAG/LLM 后端，不以成为前端工程师为目标。

## Out of scope（暂时）

在讲清 OA 完整调用链之前，不展开 Docker、PostgreSQL、Qdrant、MCP、多 Agent、Kubernetes 与复杂鉴权。
