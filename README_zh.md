<h1 align="center"> ResearchPilot </h1>
<p align="center">
  <a href="./README.md"  target="_Self">English</a> |
  <strong style="background-color: green;">中文</strong>

</p>

ResearchPilot 是一个面向科研文献（当前：行人/车辆重识别方向论文）的研究助手。
它由 AI-ChatKit 工程模板逐层改造而来：论文入库、混合检索、Corrective-RAG 研究
工作流、评估体系、一键部署。

> 每一层的设计理由都在 `reference/` 下 ——
> 建议从 `reference/design-decisions.md` 和 `reference/roadmap-status.md` 读起。

## 它能做什么

1. **文档知识库** —— `POST /documents` 上传 PDF / DOCX / Markdown / TXT（202 + 后台
   索引）；版面感知的 PDF 解析（pdfminer，剔除图表文字），引用级位置标签
   （`p.5` / `sec.2` / `blk.4`）。
2. **混合检索** —— 自研 BM25（支持按论文过滤）+ 向量召回，RRF 融合，交叉编码器
   重排（ONNX int8 本地推理）。向量后端可选：Chroma（默认，嵌入式）或 Qdrant
   （`VECTOR_STORE=qdrant`）。
3. **两个 agent** —— `research-workflow`（Corrective-RAG 变体：分析 → 检索 → 判
   证据充分性 → 补充检索 → 综合，硬性轮次预算，三态证据判定）和 `react-assistant`
   （ReAct 对照组）。注册表经 `GET /agents` 暴露。
4. **评估体系** —— 30 题 A/B/C 评估集 + LLM 评委（verdict 有定义、语料指纹过期
   告警、缺席探针）+ 不走 LLM 的确定性名次基准（`rank_bench.py`）与后端黑盒
   验收脚本。
5. **会话持久化** —— LangGraph checkpointer 落 PostgreSQL；会话列表与历史经
   `/conversations` 提供，前端不保存任何会话状态。
6. **工程与部署** —— `/health`（代码版本 / 重排模型状态 / 索引统计）、轮转文件
   日志、统一异常层、Docker Compose（api / postgres / web，另有可选 qdrant）。

## 快速开始

> **仓库不含语料。** `backend/resource/papers/` 是空的，这是有意的 —— 开发时用的
> 论文都是已发表作品，不在仓库里再分发。把你自己的 PDF 放进该目录，然后
> `python app/ai/rag/ingest.py` 导入即可（标题取自 PDF 元数据 / 文件名，首次运行
> 会生成 `titles.json` 供人工校正）。

```
# 1. 基础设施（会话与文档都依赖 PostgreSQL）
docker compose up -d postgres

# 2. 后端（需要宿主机 Ollama + bge-m3）。
#    用 run_server.py，不要用 `python -m uvicorn`：Windows 上异步 psycopg
#    checkpointer 只能跑在 Selector 事件循环，而 uvicorn 只在 --reload 子进程
#    路径里装这个策略。run_server.py 在 uvicorn.run() 之前设好，且不需要 --reload
#    （原因见 NOTES.md）。
cd backend
.venv-py311/Scripts/python.exe run_server.py

# 3. 前端
cd ../frontend
pnpm install && pnpm dev

# 或 Windows 一键：双击 启动 Docker.cmd
```

API 文档：`http://127.0.0.1:8002/docs`（FastAPI 自动生成）。

## 更多文档

| 文档 | 内容 |
|---|---|
| `reference/design-decisions.md` | 混合检索 / 重排 / pdfminer / 评估体系为什么这么设计 —— 附实测数据 |
| `reference/transformation-*.md` | 各改造的设计文档（工作流、文档 API、会话、日志、Qdrant） |
| `reference/roadmap-status.md` | 对照 10 项改造的实时状态 |
| `reference/deployment.md` | 部署手册与排错 |
