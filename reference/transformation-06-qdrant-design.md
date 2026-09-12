# 改造 #6 之二设计：Qdrant 对比（先比较，再决定换不换）

> MISSION 第 6 项的另一半：「Chroma 先保留，之后比较 Qdrant」。
> 本文先定义**怎么比才公平**，再记录结果与决策。
> （#6 的 Alembic 部分仍然押后：paper / conversation 索引 / chroma 全部可重建，
> 手写 `_add_missing_columns` 的代价还没超过引入 Alembic 的双轨制，见改造 #6 提交记录。）

---

## 零、为什么值得比（一个真实的痛点驱动）

`design-decisions.md` 决策六记录了一个没有修的缺陷：**Chroma 的 HNSW 近似检索
在 top-k 边界处不可复现** —— 同一查询两次运行，前 20 条只重叠 17 条，而重叠部分
的分数逐位相同。后果是「±1 名的差异不可信，必须重复运行」。

Qdrant 支持 `exact=true` 的**精确检索**（放弃近似换全量比对）。在 636 块的
语料上，精确检索的代价可以忽略 —— 如果 Qdrant 能在同样召回质量下给出
**确定性的名次**，它解决的就是这套评估体系的一个方法论级痛点。

## 一、怎么比才公平

| 维度 | 控制 |
|---|---|
| **向量必须逐位相同** | 不重新嵌入 —— 从 Chroma `get(include=["embeddings"])` 原样导出，按 id 上传 Qdrant。embedding 的任何重算都会引入差异，毁掉对比 |
| **语料相同** | 同一份 7 篇 / 636 块（含全部 metadata） |
| **查询集相同** | `rank_bench.py` 的评估题与流程原样复用 —— 它只依赖 `hybrid_search`，后者只依赖 `document_vector_store` 的五个方法 |
| **BM25 路相同** | BM25 索引从 `get(include=["documents","metadatas"])` 构建，两边同源 |
| **重排相同** | rank_bench 的 ± 重排两轮都跑 |
| **重复运行** | Chroma 3 轮（验证它的不可复现），Qdrant HNSW 3 轮 + exact 3 轮 |

**Qdrant 三种形态都要测**：HNSW 默认（对照 Chroma 的近似）、`exact=true`
（确定性候选）。对比的判据：

1. 名次基准四指标（recall@1/3/5、MRR）—— 召回质量不能掉；
2. **跨运行名次稳定性** —— Qdrant exact 是否做到「跑一百次一个样」；
3. 单次检索延迟 —— 同机对比；
4. 运维成本 —— 嵌入式文件（Chroma，零服务）vs 独立服务（Qdrant，compose 多一个容器）。

## 二、接入方式：同接口适配器，不平行实现

代码对 Chroma 的耦合面已经收敛到五个方法（`hybrid.py` / `ingest.py`）：

```
get(include=["documents","metadatas"], where=…)
similarity_search_with_relevance_scores(query, k, filter=…)
add_documents(chunks)  /  delete(ids)  /  count()
```

所以做成**后端选择**而不是重写检索层：

- `settings.VECTOR_STORE = "chroma" | "qdrant"`（默认 chroma，行为不变）
- `chromaClient.py` 按配置导出同名的 `document_vector_store` ——
  `hybrid` / `ingest` / `rank_bench` 一行不改
- Qdrant 的分数口径对齐 Chroma：`similarity_search_with_relevance_scores`
  返回**余弦相似度**（Chroma 的 relevance score 在 cosine space 下同义）
- Chroma 侧补一个 `count()`（原来 hybrid 直接摸 `client.get_collection`），
  让计数也走统一接口

`$in` / `$eq` 的 Chroma filter 语法在适配器里翻译成 Qdrant 的 `Filter/MatchAny`。
**翻译层只此一处** —— 让 Chroma 方言语法留在 Chroma 文件里，Qdrant 文件不出现它。

## 三、不做的事

- **不默认切换**。除非结果明确更好，否则 Chroma 仍是默认后端 ——
  嵌入式零运维在当前规模是真实优势，为一个 636 块的语料引入常驻服务需要理由。
- **不重新嵌入**、不引入双写（两套索引同时写是分布式的开端，comparison 不需要）。
- 不比较标量量化 / 磁盘压缩（636 块用不上，那是百万级向量的话题）。

## 四、预期与判据（写在这一版跑之前，防止事后挑数字）

- recall 类指标两边应该**几乎相同**（同一批向量 + 同一个重排）—— 差异只会来自
  HNSW 边界舍取；
- Qdrant exact 的价值点在**可复现性**（决策六的痛点），不在召回率；
- 如果 Qdrant exact 名次跨运行完全一致且 recall 不低于 Chroma →
  「评估用 Qdrant exact、生产看规模」或「整体切换」就有依据；
  如果没差异 → 记录「Chroma 的不可复现在本规模下影响多大」也算结论。

## 五、结果与决策

### 对比前先撞出一个更重要的发现：Chroma 一直跑在 L2 空间

迁移后抽样核对发现**同一个块两边分数不同**（Qdrant 报 0.6002，Chroma 报
0.4346）。追查结果：`papers` collection 创建时没有传 `collection_metadata`，
Chroma 默认 **l2** 度量，langchain 把它换算成 relevance = **1 - d²/√2**
（d 为欧氏距离；实测手算逐位吻合，含 d² 的平方项 —— Chroma 的 l2 返回的是
平方距离）。手算余弦与 Qdrant 的余弦分数逐位一致，证明两边向量本身一致。

两个后果：

1. **阈值 0.35 的真实口径**：它校准在 L2-换算分数上，不是余弦相似度 ——
   之前所有文档把它当"余弦阈值"讨论是不精确的（排序结论不受影响：
   bge-m3 向量实测全部归一化，L2 与余弦排序单调等价）。
2. **对比必须同度量**：Qdrant collection 因此用 EUCLID（不是 COSINE），
   适配器复现 `1 - d²/√2` 变换。对齐后抽样核对：**分数差 ~1e-7（浮点精度级），
   名次与块完全一致**。这本身就是适配器正确性的证明。

### 名次基准（20 题 rank_bench，每配置 3 轮独立进程）

| 配置 | recall@1 | recall@3 | recall@5 | MRR | mean_rank | 3 轮输出 |
|---|---|---|---|---|---|---|
| Chroma（l2） | 0.500 | 0.583 | 0.667 | 0.597 | 3.18 | **逐字节一致** |
| Qdrant HNSW（euclid） | 0.500 | 0.583 | 0.667 | 0.597 | 3.18 | **逐字节一致** |
| Qdrant exact（全量比对） | 0.500 | 0.583 | 0.667 | 0.597 | 3.18 | **逐字节一致** |

逐条名次跨配置 diff 为空。延迟未单独测 —— 本规模下操作层面无感知差异。

### 决策：默认后端保持 Chroma，Qdrant 适配器保留待用

- **质量**：完全打平（同一批向量、同度量、同变换 —— 这是设计保证的，不是巧合）。
- **确定性**：决策六记录的「Chroma top-k 跨进程不可复现」在当前规模（636 块、
  3 独立进程 × 3 配置）**没有复现**，Qdrant exact 没有带来额外收益。旧结论
  是在 411 块语料上观察到的；不排除与规模/索引参数相关，保留为已知观察。
- **运维**：Chroma 嵌入式零服务 vs Qdrant 多一个常驻容器 —— 636 块付这个成本不值。

**什么时候重新评估**：语料到十万块级（HNSW 参数调优有实际收益）、
需要标量量化 / 磁盘压缩、或需要跨进程共享索引（Chroma 嵌入式不允许两个后端
进程挂同一目录 —— compose 文档里已经因此警告过）。

### 过程中被推翻的假设（留档）

- "Chroma cosine-space relevance = 余弦相似度" —— 错，实为 L2 + 换算（见上）。
  设计文档第一节预先写的判据没有预见到这一点；是**迁移后的逐位核对**把它抓出来
  的 —— 这正是"先抄向量再重嵌"的迁移设计的价值：向量相同，分数不同，
  只能是口径问题，一眼定位。
