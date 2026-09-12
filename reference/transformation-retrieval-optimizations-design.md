# 检索层优化：结构化元数据 / 相邻块 / 表格标注（与 agent 无关的三项）

> 用户需求：做与 agent 无关的文献查询优化。这三项全部是**确定性层**（ingest /
> 检索管线）的改动，效果用 rank_bench 与证据锚点量化验证，不依赖 LLM 发挥。

## 项 1：结构化元数据入库

**问题**：年份/作者筛选目前是查询时解码（`submission_year`）+ 源列表过滤。上传的
论文 `year` 字段基本为空，结构化信息没有在入库时沉淀。

**做法**：
- ingest 建记录时用 `arxiv_submission_date()` 解码 source_file/title，
  写入 `Paper.arxiv_id` / `Paper.pub_year`（可空列，`_add_missing_columns`
  自动迁移；已有 80 篇不动，只对今后上传生效）
- `search_documents` 工具增加 `year_from` 参数：用 `papers_for_year` 解析出
  source 列表做限定（复用现有 `allowed_sources` 机制，不引入新的过滤通道）

**验证**：上传一篇带 arXiv 编号的新测试 PDF → 记录里 arxiv_id/pub_year 正确；
search_documents(year_from=2025) 只返回 2025+ 论文的段落。

## 项 2：命中块带相邻块（根治 A08 遗留）

**问题**：决策七诚实清单——枚举跨 931 字符 > chunk_size 800，答案被切块边界
切断。A08「能过靠运气」。

**做法**：`pipeline.retrieve` 重排取 top_n 后，对每个最终命中块补**同论文、
相邻 page_label** 的块（去重、总量封顶 +6），分数记 None、来源标记加
`[context]` 前缀。它们不参与重排与阀门判定，只作为上下文进入 LLM 上下文
—— 锚点能匹配上即可。

**验证**：rank_bench 证据报告里 A08 的缺失锚点
（`loading and equipment configuration`）从缺失变为命中；run_eval A08 保持
GROUNDED；全量锚点/verdict 不退。

## 项 3：表格块标注

**问题**：表格数字行既答不了数字问题又会污染排序（决策五）。

**做法**（保守版）：入库时按数字密度给块打 `kind: table` 标注；
`format_hits` 对表格块加 `[表格/实验数据]` 前缀提示 LLM；工具描述写明
「性能数字查 table 块」。不动重排与 RRF（防过拟合调参，那是决策五的教训）。

**验证**：重新索引一篇含实验表的论文 → 抽查表格块带 kind 标注；
run_eval 全量不退（纯标注不应改变行为）。

## 护栏

所有改动跑完后：`rank_bench`（3 轮）与 `run_eval`（30 题全量）的指标
**不得低于当前基线**（A 类 verdict 1.0 / B+C 全对 / 名次基准 recall@5 0.667）。
任何回退要么修掉，要么把改动回滚 —— 不接受「新功能换旧质量」。
