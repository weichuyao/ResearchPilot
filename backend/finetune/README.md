# 规划节点改写器微调（改造 #11 阶段 3-5）

本目录是**微调实验**的配置与说明。它不是后端运行时依赖 —— 后端只需在
`.env` 里把 `ANALYZE_MODEL` 指过去（见下），训练栈完全不进 `backend/`。

## 它解决什么问题

`research_workflow` 的 `analyze` / `refine` 节点是**窄任务**：中文问题进、
固定 JSON schema 的英文检索查询出。每个用户问题要调 1~4 次，全是这一类。

这个实验的回答是：**能不能用本地小模型替掉这些 API 调用。**

结论（实测，见 `reference/transformation-11-finetune-design.md` 第七节）：

| | 未微调同基座 | 微调后 |
|---|---|---|
| 回落到 API 的比例 | 93% | **0%** |
| 改写耗时 | 3.15s/题 | **1.28s/题** |
| recall@5（检索质量） | 0.8636 | 0.6818 |
| 锚点全中 | 20/22 | 16/22 |

**微调解决了「能不能用」（格式合规、零回落）和延迟（2.5×），没有解决
「用得好不好」**：模型学会了输出英文查询，但丢掉了字面领域术语
（`instance bank`、`DIST` 这类可逐字匹配的罕见词），于是召回下降。

所以正确用法是**按场景切换**，不是无条件替换。

## 目录结构

```
finetune/
├── configs/
│   ├── qlora_qwen2.5-1.5b.yaml    # QLoRA 训练配置（8GB 显存下的唯一可行解）
│   └── export_merged.yaml          # 合并 LoRA → 完整模型（供 Ollama 导入）
├── data/          # ← 生成物，不进 git（train.jsonl / holdout.jsonl / dataset_info.json）
├── models/        # ← 不进 git（基座、合并后的模型）
└── output/        # ← 不进 git（LoRA 适配器、checkpoint）
```

`data/` 与 `models/` 不进 git 有两个理由：训练数据由**受版权保护的论文**派生
（与仓库移除 PDF 的理由一致），权重是大文件。

## 复现步骤

### 1. 训练环境（独立，**不要**装进 backend venv）

后端 venv 刻意没有 torch（装它要下 2.5GB），这是既有设计决策 —— 别破坏它。
本实验用 conda 克隆一个已有 torch+CUDA 的环境，避免重下 torch：

```bash
conda create -n llamaft --clone <已有 torch 的环境> -y
conda run -n llamaft pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    peft trl datasets accelerate bitsandbytes llamafactory
```

> ⚠️ 实测教训：**PyPI 直连只有 82 kB/s**（39MB 的 bitsandbytes 下了 12 分钟），
> LLaMA-Factory 那 250MB 依赖根本下不动。换清华镜像后 1.1 MB/s。
> 另外 LLaMA-Factory 会把 transformers 降到 4.52、numpy 降到 1.26 —— 在
> **克隆体**里发生，原环境不受影响，这正是克隆而不是直接装的理由。

### 2. 基座模型（走 hf-mirror）

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='finetune/models/Qwen2.5-1.5B-Instruct',
                  allow_patterns=['*.json','*.safetensors','*.txt'])"
```

### 3. 构造数据（两步，第二步会调 DeepSeek）

```bash
cd backend
export PYTHONPATH=app
# 从语料反向生成中文研究型问题（约 5 分钟）
python -m ai.finetune.build_sft_data gen-questions --per-paper 7 --out finetune/data/questions.jsonl
# 教师蒸馏 + 清洗 + 训练/评估隔离断言 → sharegpt JSONL
python -m ai.finetune.build_sft_data distill --in finetune/data/questions.jsonl --out-dir finetune/data
```

第二步会**断言** 30 道评估题没有混进训练集 —— 这道断言在首次运行时真的抓到了
3 条逐字重合 + 1 条语义等同的泄漏（不改的话，阶段 5 的对比会全部作废，
且表现为"微调效果惊人"）。

### 4. 训练（8GB 显存约 5.5 分钟）

```bash
cd backend
llamafactory-cli train finetune/configs/qlora_qwen2.5-1.5b.yaml
```

### 5. 合并 + 导入 Ollama

```bash
cd backend
llamafactory-cli export finetune/configs/export_merged.yaml
cd finetune/models/qwen2.5-1.5b-rewriter-merged && ollama create research-rewriter -f Modelfile

# 未微调的同基座对照（必需 —— 否则分不清是微调的功劳还是基座本来就行）
cp Modelfile ../Qwen2.5-1.5B-Instruct/Modelfile
cd ../Qwen2.5-1.5B-Instruct && ollama create qwen2.5-1.5b-base -f Modelfile
```

> 为什么不用 GGUF：LLaMA-Factory 的 GGUF 导出要 llama.cpp 的编译工具链，
> 而 **Ollama 0.6+ 能直接吃 Safetensors** —— 少一步转换就少一处失败点。

### 6. 启用与评测

```bash
cd backend
# 启用（写进 .env 更持久）
ANALYZE_MODEL=local-rewriter OLLAMA_REWRITE_MODEL=research-rewriter python run_server.py
curl -s localhost:8002/health | python -m json.tool     # 看 rewrite_model 字段

# 确定性评测（不含 LLM 评委）
python app/ai/eval/rewrite_bench.py                    # 微调后
python app/ai/eval/rewrite_bench.py --baseline raw      # 对照：不改写
ANALYZE_MODEL=local-rewriter OLLAMA_REWRITE_MODEL=qwen2.5-1.5b-base \
    python app/ai/eval/rewrite_bench.py                 # 对照：未微调同基座
```

## 如果要把效果做上去

按性价比排（见设计文档第七节的分析）：

1. **数据侧**：训练数据里显式强化「查询必须包含 facets 的字面术语」，
   或把任务改成 facet-conditioned —— 这是最对症的
2. **工程侧**：本地模型出查询骨架、术语用更可靠的抽取补上（绕过模型短板）
3. 接受现状，按场景切换（延迟敏感用本地、质量敏感用 API）
