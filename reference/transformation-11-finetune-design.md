# 改造 #11：规划节点的微调闭环（设计文档）

> 状态：**阶段 1+2 已完成并实测**（`46321dc`）。阶段 3-5（数据构造 / 训练 /
> 闭环评测）待启动 —— 见第七节的启动条件。
>
> 本文件按项目规矩写：**为什么做、代价是什么、怎么验证**。所有数字都是实测的。

---

## 一、目标与范围

把 `research_workflow` 的**规划节点**（`analyze` / `refine`）从「每次调用
DeepSeek API」换成「本地微调过的小模型」，并用项目自己的评估体系回答一个具体
问题：**微调到底有没有用、值不值。**

交付形态（最小可信闭环）：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 服务接入层：规划节点可分模型 + 失败回落 + 可见性 | ✅ `46321dc` |
| 2 | 确定性改写器基准 `rewrite_bench.py` | ✅ `46321dc` |
| 3 | SFT 数据构造（语料蒸馏 + 硬案例 + 清洗） | ⏳ |
| 4 | LLaMA-Factory 训练（LoRA / QLoRA 两配置） | ⏳ |
| 5 | 闭环评测 + 实验记录 | ⏳ |

---

## 二、为什么只做改写器（以及为什么不微调别的）

### 为什么不微调「生成」（synthesize）

RAG 的事实来自检索，不是来自权重。把一个 6818 块的语料「训进」一个 1.5B 模型
既不可能也不必要；而 `synthesize` 要读十几个块、产出带页码引用的长回答，小模型
会直接把引用能力丢掉。**用一个小模型替掉生成环节是事故。**

### 为什么不微调重排模型（虽然它的指标最硬）

重排（`bge-reranker` 交叉编码器）能用 `rank_bench` 的确定性指标验证，证据最干净。
但它不是生成式任务 —— LLaMA-Factory 不适用，要另用 FlagEmbedding 训练并自行导出
ONNX int8 才能接进现有链路（`rerank.py` 有固定的磁盘契约：`tokenizer.json` +
`onnx/` 下四个候选名之一 + 单输出张量）。**本轮明确不做**，留作后续可选项。

### 为什么改写器是合适的对象

`analyze` 每个用户问题调用一次；证据不足时 `refine` 最多再触发
`MAX_RETRIEVE_ROUNDS = 3` 轮 —— **每个问题 1~4 次 API 调用，全是窄任务**：
中文问题进、固定 JSON schema 的英文检索查询出。格式约束强、不需要世界知识，
是「小模型能顶替大模型」的少数场景之一。

### 诚实的反对意见（必须写下来）

DeepSeek 在 `temperature=0` + 结构化输出下**已经工作得很好**。1.5B 微调后
很可能打不过它。预期收益主要在**延迟与成本**，不是质量。

如果实测质量明显下降 → 保留 API 做改写，把负面结论写在本文件里。这个项目里
负面结果是有价值的记录（见决策十的 B05 评委噪声、`compare_runs` 的可比性告警），
不是失败。

---

## 三、核心设计决策：按节点分模型，而不是全局换模型

**现状（改造前）**：`research_workflow.py` 的 `_model()` 让 `analyze`、`refine`、
`synthesize` 共用同一个全局模型（`settings.DEFAULT_MODEL`）。

**改动**：新增 `ANALYZE_MODEL` 配置与 `LocalModelName` 模型族 ——
`analyze`/`refine` 走它，`synthesize` 不受影响。取值留空则完全保持旧行为
（改造前的所有历史评估报告都是这个口径，可直接对比）。

这不是「多一个配置项」，而是**承认一件事**：图上不同节点对模型能力的要求不同，
用一个模型通吃是偷懒。窄任务与生成任务该分开。

---

## 四、阶段 1：服务接入层（含两个既有缺陷）

### 顺带修掉的两个真缺陷

1. `ai/llm.py` 的 OpenAI 分支**不传 `base_url` / `api_key`** —— 任何自部署的
   OpenAI 兼容端点都接不进来。
2. `backend/.env.example` 声明了 `OPENAI_BASE_URL` / `OPENAI_API_KEY`，但
   `core/config.py` **没有对应字段**。pydantic-settings 只把这些值读进 settings
   对象、**不导出到 `os.environ`**，所以 SDK 拿不到 —— 配置项写了等于没写。

### 失败回落必须看得见

`_invoke_structured` 的分层：

1. 模型不可用 / 调用抛异常 → 回落
2. 结构化输出解析失败 → 回落
3. **语义为空**（无子问题、查询为空串）→ 回落
4. **违反任务约定**（query 是中文，而任务要求英文）→ 回落

第 3、4 层是这次改造的度量核心。回落时打 WARNING（进日志文件）+ 进程内计数，
`/health` 暴露 `rewrite_model: {configured, target, calls, fallbacks, last_error}`。

**为什么必须暴露**：本地模型一直失败、每次悄悄回落 API 时，系统表现**完全正常**，
只是没省下任何延迟。这与 reranker 静默降级是同一类问题（`system_routes.py` 里
`_reranker_state()` 的注释写过这件事）。不暴露的话这次改造等于白做且无人察觉。

实测（`tmp/probe18.py`、冒烟请求）：

```
/health → rewrite_model: {configured: "local-rewriter", target: "qwen:4b",
                          calls: 3, fallbacks: 1,
                          last_error: "ValueError: query 不是英文（任务约定要求英文查询）：
                                       'A²RNet 的 SAP 模块推断的三类语义属性是什么？'"}
```

### 技术选型：语法约束解码，不是「schema 进提示词」

实测对比（`tmp/probe19.py`，基座 `qwen:4b`）：

| 方式 | 结果 |
|---|---|
| `json_mode` + schema 写进提示词 | **失败** —— 弱模型把 schema 片段当答案回吐：`{"type": "string", "title": "Query", ...}` |
| `json_schema`（Ollama 原生语法约束解码） | **成功**，1.1 秒，结构 100% 合法 |

所以本地路径用 `with_structured_output(schema, method="json_schema")`，
`llm.py` 的本地模型分支**刻意不设 `format`** —— 由调用方按需指定。

但结构合法 ≠ 内容可用：同一个测试里基座模型返回的 `query` 是**中文原文**、
`facets` 为空。**这正是微调要解决的缺口**，也是第 4 层校验存在的理由。

---

## 五、阶段 2：确定性改写器基准（`rewrite_bench.py`）

### 为什么必须新建（而不是复用 rank_bench）

`rank_bench.py` 把**评估题原文**直接喂给检索：

```python
hits, _ = hybrid_search(item["question"], top_n=top_k)   # 不经过改写器
```

它测的是「混合检索 + 重排」，**完全不经过改写器**。拿它比较微调前后的改写器，
两组数字会一模一样 —— 报告看起来像「微调无效」，实际是**尺子测错了对象**。

这个项目在这一点上有前科：`retrieval_rank` 曾被当排序指标用，后来发现它同时被
查询改写与多轮拼接污染（决策六）。**原则：被微调的组件，要有专门量它的尺子。**

### 测什么（全流程不含 LLM 评委）

```
中文问题 → plan_question()（改写器）→ query
         → hybrid_search(query) → 命中列表
         → 期望论文+页码的名次 / required_evidence 锚点是否齐全
```

| 指标 | 含义 |
|---|---|
| `回落到主力模型的比例` | **头号数字** —— 本地模型到底有没有在干活 |
| `recall@1 / recall@5 / mrr / mean_rank` | 改写后「该找的东西」有多好找（分母含未命中） |
| `锚点全中` | `required_evidence` 齐全的题数，直接对应 A08 那类跨块枚举问题 |
| `结构化输出失败` | 本地与回落后都失败 —— 硬门槛 |
| `改写平均耗时` | 收益侧的主要指标 |

`--baseline raw` 不调改写器、直接用问题原文检索。**已核对它与 `rank_bench`
口径精确一致**：

```
rank_bench  混合检索: {"hit": 17, "recall@1": 0.22727, "recall@5": 0.63636,
                       "mrr": 0.37449, "mean_rank": 3.64706}
rewrite_bench --baseline raw: hit 17, recall@1 0.2273, recall@5 0.6364,
                               mrr 0.3745, mean_rank 3.647
```

### 第一版实测暴露的报告缺陷（已修）

初版只打印「结构化输出失败 0/N」。而 `_invoke_structured` 在本地失败时会**回落
并成功返回**，于是 `plan_question` 看起来一切正常 —— A08 显示「失败 0/1」，
真实情况却是本地模型输出中文查询、被校验拒绝后回落。

**只看失败数会把「本地模型完全没干活」读成「本地模型工作正常」。** 已改为报告
回落率，并在回落率 ≥ 50% 时明确打印「下面的指标主要是主力模型的成绩」。

---

## 六、基线数字（微调前的起点）

语料 72 篇 / 6818 块；评估集 30 题；改写器配置 `ANALYZE_MODEL=local-rewriter`，
本地 tag 用现成的 `qwen:4b`（**不是**微调产物，只用来跑通链路拿基线）。

| 配置 | 回落率 | recall@1 | recall@5 | mrr | mean_rank | 锚点全中 | 改写耗时 |
|---|---|---|---|---|---|---|---|
| `raw`（不改写） | — | 0.2273 | 0.6364 | 0.3745 | 3.647 | 15/22 | — |
| 改写（本地 83% 回落） | 83% | **0.5** | **0.7727** | **0.6247** | **2.421** | **19/22** | 2.93s/题 |

两个结论：

1. **改写器本身的价值第一次被量化**：recall@1 从 0.2273 翻倍到 0.5，锚点全中
   15/22 → 19/22。改造前没有尺子量得到这一点 —— 这是阶段 2 的直接产出。
2. **基座模型（未微调）顶不住这个任务**：30 次调用只成功 5 次，回落率 83%。
   结构能过语法约束，内容过不了任务约定。**这正是微调要填的缺口**，
   也是阶段 3-5 的验证目标：回落率应当显著下降。

⚠️ 上表第二行的指标**主要是 DeepSeek 的成绩**（83% 的调用回落到它），
不是本地模型的成绩。基准报告里已经明确标注了这一点。

---

## 七、阶段 3-5 计划

### 阶段 3：SFT 数据构造（工作量主体）

`backend/app/ai/finetune/build_sft_data.py`

1. **问题来源**：从 72 篇论文的标题 / 摘要 / 块生成中文研究型问题，覆盖改写器
   要处理的各种形态
2. **教师蒸馏**：用现有 DeepSeek `plan_question()` 产出 `ResearchPlan` 作为标签
   （**注意成本**：每条一次 API 调用，先小批量试跑）
3. **硬案例注入**：把项目已记录的失败模式编进训练集 —— A08 类跨块枚举、
   表格数字类（`kind=table` 的 108 个块）、B 类「语料里有没有 X」
   （必须**不**编造断言存在的查询）
4. **清洗**：去重（用项目自己的 `textnorm.norm_for_match` 口径）、剔除退化 plan
   （空 / 非英文 query、子问题 >4、facets 非字面短语）、`needs_paper_list` 错标
5. **训练/评估严格隔离**：脚本里**断言** 30 道评估题从未进入训练集
   （做进代码，不靠自觉），另留 held-out 集

输出：LLaMA-Factory `sharegpt` 格式 JSONL（含 system role）。
**产物不进 git**：训练数据由受版权保护的论文派生，与移除 PDF 的理由一致。

### 阶段 4：训练（LLaMA-Factory）

- **独立环境**，绝不装进 `backend/.venv-py311` —— 项目明确记录「后端 venv 没有
  torch 是有意为之（torch 要下 2.5GB）」，污染它会破坏既有设计
- 硬件：RTX 5060 Laptop **8GB 显存**（实测可用约 6GB）→ 只能 QLoRA 4-bit，
  基座 ≤3B 舒适、7B 紧张
- 基座：`Qwen2.5-1.5B-Instruct`（中文原生、结构化输出友好）
- **两个配置 = 方案比较**：`LoRA(bf16, 1.5B)` vs `QLoRA(4bit, 3B)`
- **必须含第三个对照：未微调的同一基座** —— 否则分不清「微调有用」还是
  「基座本来就够」
- 模型下载走 `HF_ENDPOINT=https://hf-mirror.com`（`scripts/fetch-reranker.ps1`
  已有先例，本机直连 huggingface.co 实测超时）

### 阶段 5：闭环评测

- 导出 GGUF → `ollama create` → 改 `ANALYZE_MODEL` / `OLLAMA_REWRITE_MODEL`
- 四条线对比：**基座未微调 / LoRA / QLoRA / DeepSeek（上限参考）**
- 工具：`rewrite_bench`（确定性的主力证据）、`run_eval` 全 30 题、`compare_runs`
- **`compare_runs` 需扩展**：它只按 `summary.agent` 分组，而 `summary.model`
  虽已写入却未参与分组 —— 微调前后会混成一组，而它们恰恰是要对比的两组。
  `run_eval` 已记录 `summary.analyze_model`（本改造加的）供分组使用

---

## 八、护栏（项目铁律：任何改造必须有验证标准）

| 检查项 | 门槛 | 现状 |
|---|---|---|
| 结构化输出解析成功率 | ≥ 99% | 100%（语法约束解码保证），阶段 4 后复验 |
| 回落率 | 显著低于基线 83% | 待阶段 5 |
| `rewrite_bench` 指标 | 不低于未微调同基座；目标接近 DeepSeek | 待阶段 5 |
| `run_eval` 30 题 | A/B/C verdict 全对不退（基线 22+6+2 全对）、`evidence_recall` ≥ 1.0 | ✅ **阶段 1+2 回归已确认**（见下） |
| `rank_bench` | 锚点全中不下降 | ✅ 15/22 不变 |
| `process.avg_seconds` | 相对 API 基线下降（本次改造的主要收益） | 待阶段 5 |
| 回落机制 | 本地模型挂掉时自动切回 API，WARNING 可见 | ✅ 已实测 |

### 阶段 1+2 的回归确认（2026-09-16）

改动集中在服务接入层与新增基准，理论上不影响默认路径（`ANALYZE_MODEL` 留空
即改造前口径 —— 这一点也由报告里的 `analyze_model: ""` 记录了）。实测确认：

| 指标 | 改造前 | 改造后 |
|---|---|---|
| A.evidence_recall | 1.0 | 1.0 |
| A.verdict_accuracy | 1.0 | 1.0 |
| A.citation_ok_rate | 1.0 | 1.0 |
| A.retrieval_recall | 0.955 | 0.955 |
| B / C verdict | 全对 | 全对 |
| `rank_bench` 锚点全中 | 15/22 | 15/22 |
| 不达标题目 | 无 | 无 |

报告：`resource/eval/eval-20260916-220902.json`、
`resource/eval/rank-bench-20260916-221135.json`。

（`A.avg_concept_coverage` 在 0.977~1.0 之间随运行波动 —— 历史多次运行都在这个
区间，属评委噪声，不是回归。）

---

## 九、风险

1. **微调可能不提升质量** —— 已备好负面结论写法，不算失败
2. **教师蒸馏要花 API 费用** —— 按条数计，阶段 3 先小批量试跑
3. **8GB 显存** —— QLoRA 4-bit；训练期间占用本机显存
4. **`run_eval` 有 LLM 评委噪声** —— 主力证据用阶段 2 的确定性基准，
   `compare_runs` 多轮取均值
5. **数据版权** —— 训练数据与 adapter 权重都不进公开仓库

---

## 十、明确不做

- 不微调生成节点（`synthesize`）—— 会丢掉引用能力
- 不微调重排模型 —— 本轮范围外（见第二节）
- 不做 DPO、不做 LLaMA-Factory WebUI 集成、不做多基座矩阵
- 不把 `torch` / `peft` 装进后端 venv
- 不把训练数据与 adapter 权重提交进公开仓库

---

## 附：本改造新增的工具与命令

```bash
# 确定性改写器基准（不调 LLM 评委）
python app/ai/eval/rewrite_bench.py                  # 当前配置的改写器
python app/ai/eval/rewrite_bench.py --baseline raw    # 对照：不改写
python app/ai/eval/rewrite_bench.py --json -          # 写报告

# 开启本地改写模型（未微调时回落率会很高，属预期）
ANALYZE_MODEL=local-rewriter OLLAMA_REWRITE_MODEL=qwen:4b python run_server.py
curl -s localhost:8002/health | python -m json.tool   # 看 rewrite_model 字段
```
