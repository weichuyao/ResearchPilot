# 交接：检索优化项 2/3 收尾（写给下一个会话）

> 交接时间：2026-09-12 深夜。前一会话完成了三项检索层优化的项 1 和项 2/3 的
> 主体代码，**剩 3 个未闭合的评估问题**。本文件是唯一的交接依据，读完即可开工。

## 一、当前状态（务必先核对）

- HEAD = `670c682`（WIP 提交，含项 1 `cb2d46e`），工作树干净
- **后端跑在 8002 端口**（不是 8001！8001 是个杀不掉的僵尸服务，跑旧代码，
  用户重启电脑后清理；前端 `.env.local` 已指向 8002）
- 前端 dev 在 3000；MCP 挂载带 `uvx --offline` 自动降级（当前挂 7/19 工具）
- 后端改动**必须手动重启**（`--reload` 不可信，见 NOTES.md），重启后核对
  `/health` 的 `started_at`
- ⚠️ 曾有一次清理误杀了用户另一个项目在 8000 端口的服务（`uvicorn.exe
  app.main:app --port 8000`），已告知用户

## 二、已验证生效的（不要回退）

- **项 1**（`cb2d46e`）：ingest 三个建行点 + 上传路由，arXiv 编号/年份自动沉淀
  （`_decode_arxiv_date` → year + `arXiv:原始编号`）；`search_documents` 新增
  `year_from`
- **项 2 核心机制**：`pipeline.py` 的 `_attach_context_chunks()` —— 重排 top_n
  后补同论文相邻块（origin="context"，分数 None，cap=6）。**直连探针实测：
  A08 查询的三个锚点（ship type / imaging perspective / loading and equipment
  configuration）首次全部命中** —— 决策七的跨块枚举问题在检索层已根治
- **项 3**：ingest 数字密度 > 0.25 → metadata `kind=table`；`format_hits` 加
  `[experimental table]` 前缀；context 块的 how 文案独立（"adjacent context
  chunk..."）。注意：已入库的 80 篇没有 kind 字段，重索引后才生效

## 三、待排查的 3 个问题（按此顺序）

### 问题 1：A08 端到端仍 NOT_FOUND（最重要）

- **现象**：检索层探针三锚点全中，但 run_eval 里 agent 答"第三项被截断"
- **疑点 A：`tool_queries` 在评估记录里是空的**（A05/A08 都是）——
  `run_eval.py` 的 harvest 解析工具输出提取查询；本轮 `format_hits` 改了
  how 文案和表格前缀，**先查 harvest 的解析正则是否被新格式弄坏**
  （历史教训：改输出格式没同步解析正则，`retrieval_recall` 静默掉 0.25）
- **疑点 B：context cap=6 的分配**——10 个重排命中跨多篇论文，每个吃 2 个
  邻居，cap 先到先得，A2RNet (iii) 块的邻居可能被别的论文挤占。备选修法：
  按论文分组配额，或每个命中只取 1 个紧邻
- **排查入口**：`python app/ai/eval/run_eval.py --only A08`，然后读最新
  报告 JSON 里 A08 的 tool_queries/answer；直连探针对照：
  ```python
  from ai.rag.pipeline import retrieve
  out = retrieve('A2RNet SAP module three semantic attributes inferred global feature vector')
  # 三个锚点应全 True（当前已验证）
  ```

### 问题 2：A05 锚点 `first-order difference` 字面缺失

- 直连探针：`kmin` 命中、`first-order difference` 未命中 → 语料里该串
  字面不存在（agent 的回答内容方向是对的，缺字面术语）
- **查语料变体**：DPEFormer 第 4 页附近找 "first order difference" /
  "first-order differences" / 被连字或断词的变体（textnorm 已还原 ﬁ，但
  可能是别的形态）。若确认语料写法不同 → 改评估集 A05 的 required_evidence
  为语料中的真实字面串（人工核对后 --mark-verified）

### 问题 3：B05 翻 GROUNDED（判噪声还是真失效）

- 语料含 "GAN" ×154（探针已报警）。之前 B05 通过；本轮翻车
- **动作**：复跑 `--only B05` 3 次看稳定性。稳定 GROUNDED → 读答案里引用的
  论文，判断语料是否真有 GAN 相关方法（那 B05 期望值要人工修订 + 记录）；
  抖动 → 属评委噪声，记录即可

## 四、收尾标准（全部完成才算完）

1. 三问题闭合（修复或人工修订评估集并记录）
2. 全量护栏：`run_eval.py`（30 题，A/B/C verdict 全对）+ `rank_bench.py`
   （注意：其数字与扩容前不可比，因为评估集变了；rank_bench 走
   hybrid_search，不受相邻块影响）
3. 更新 `design-decisions.md`（相邻块与表格标注的决策记录，含本次踩坑）
4. 提交；`roadmap-status.md` 的检索优化补一条
5. 测试脚本：`qa_blackbox_test.py`（后端黑盒，含 cleanup 子命令）

## 五、本次会话的重要教训（写给未来的自己）

1. **修竞态的代码会成为新的竞态源**：上一轮加的 abort effect 把新建会话
   的流掐死了（用户实测抓到）——加生命周期钩子必须想清楚它触发时机
2. **`--reload` 不可信**：三次看漏后端变更；8001 僵尸服务就是这么来的
   （TaskStop 杀 shell 不杀 uvicorn 子树，supervisor 还会复活 worker）
3. **bash heredoc 写正则会吃 `\b`**：正则一律按行号写入文件或用
   `(?<!\d)` 形式
4. **列表推导重新绑定**：`items = [过滤]` 后 `d["items"]` 不变——
   评估集 30 题静默变 20 题就是这么来的
