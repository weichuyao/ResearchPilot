# 四篇 ReID 论文评估集（v1）

## 使用约定

- **题量与分布：20 题**（A 类 12 题、B 类 6 题、C 类 2 题）。
- **页码口径：**下文“PDF 第 X 页”均指输入 PDF 文件的物理页序号（从 1 开始），不是论文印刷页码。
- **A 类通过条件：**回答须覆盖“可判定要点”全部内容，且引用指定论文和 PDF 页码；同义改写可接受。
- **B 类通过条件：**回答须明确限定在“提供的四篇论文/当前文档库”，表达“未找到相关证据（`NOT_FOUND_IN_CORPUS`）”。不得把未检索到证据改写为“没有使用/不存在/作者未使用”。
- **C 类通过条件：**回答须明确拒绝，并标注 `OUT_OF_SCOPE`；不得给出实质性答案或编造论文联系。

## 文献短名

| 短名 | 论文 |
| --- | --- |
| MCL | *Unsupervised Maritime Vessel Re-Identification With Multi-Level Contrastive Learning* |
| DPEFormer | *Dynamic Patch-aware Enrichment Transformer for Occluded Person Re-Identification* |
| A²RNet | *A²RNet: Association-Aware Feature Reinforcement Network for Long-Term Ship Re-Identification* |
| ViV-ReID | *ViV-ReID: Bidirectional Structural-Aware Spatial-Temporal Graph Networks on Large-Scale Video-Based Vessel Re-Identification Dataset* |

## A. 文档里明确有答案（12 题）

### A01

问题：hard positive problem（HPP）在无监督船舶 ReID 中指什么？

类型：A

期望行为：解释为：聚类得到的不准确伪标签使同一真实船舶的图像难以被视为正样本；常规实例级对比学习还会把同伪标签样本当作不同类别，从而难以发现真正正样本。必须引用 MCL，PDF 第 2 页。

期望出处：MCL，PDF 第 2 页。

### A02

问题：MCL 的训练过程交替进行的两个阶段分别是什么？

类型：A

期望行为：回答“伪标签生成”和“多层次对比学习”，并说明前者按前一轮编码器特征聚类生成伪标签，后者结合实例级与簇级对比。必须引用 MCL，PDF 第 3 页。

期望出处：MCL，PDF 第 3 页。

### A03

问题：VesselReID 数据集最终包含多少张图像、多少个船舶身份？

类型：A

期望行为：回答 30,587 张图像和 1,248 个船舶身份（约每船 25 张），不得把原始收集的 361,760 张/1,540 个身份当作最终数据集。必须引用 MCL，PDF 第 6 页。

期望出处：MCL，PDF 第 6 页。

### A04

问题：DPEFormer 的 DPSM 如何从遮挡图像中选择有用的 patch token？

类型：A

期望行为：说明它以 label-guided proxy token 与各 patch token 的相似度评估重要性，并动态选出信息量大的、较少遮挡的 token；还应指出这是二值权重的 hard attention。必须引用 DPEFormer，PDF 第 2 页。

期望出处：DPEFormer，PDF 第 2 页。

### A05

问题：DPSM 如何决定保留多少个 patch token？

类型：A

期望行为：说明它将 proxy 相似度降序排列，取一阶差分最大的分界点 k，将前 k 个视为身体 token；再以 kmin 作为最小保留数量。必须引用 DPEFormer，PDF 第 4 页。

期望出处：DPEFormer，PDF 第 4 页。

### A06

问题：ROA 使用什么来源生成遮挡掩码？它是否在推理阶段使用？

类型：A

期望行为：回答掩码来自在自然图像上运行的 Segment Anything Model（SAM）；ROA 是训练期辅助增强，不参与推理。必须引用 DPEFormer，PDF 第 2 页或 PDF 第 6 页。

期望出处：DPEFormer，PDF 第 2 页、PDF 第 6 页。

### A07

问题：A²RNet 认为长期船舶 ReID 中外观大幅变化的典型原因有哪些？

类型：A

期望行为：至少列出 cargo changes（货物变化）、viewpoint shifts（视角变化）、occlusions（遮挡）、environmental factors（环境因素）中的三项，并说明这些变化破坏短期外观稳定性假设。必须引用 A²RNet，PDF 第 1 页。

期望出处：A²RNet，PDF 第 1 页。

### A08

问题：A²RNet 的 SAP 模块推断的三类语义属性是什么？

类型：A

期望行为：完整回答 ship type、imaging perspective/viewpoint、loading and equipment configuration/cargo status。必须引用 A²RNet，PDF 第 3 页。

期望出处：A²RNet，PDF 第 3 页。

### A09

问题：A²RNet 的 TAR 如何组织历史特征，以进行跨时间的匹配？

类型：A

期望行为：说明它建立按身份-视角对 (identity, viewpoint) 索引的动态 instance bank，存储 feature map；视角来自 SAP，以保证视角对齐的跨时间关联。必须引用 A²RNet，PDF 第 4 页。

期望出处：A²RNet，PDF 第 4 页。

### A10

问题：ViV-ReID 数据集的规模是多少？

类型：A

期望行为：完整回答 480 个船舶身份、20 个跨港/海事区域视角、7,165 条 tracklet、1,145,469（约 114 万）帧；不应误答成 MCL 的图像数据集规模。必须引用 ViV-ReID，PDF 第 2 页。

期望出处：ViV-ReID，PDF 第 2 页。

### A11

问题：ViV-ReID 数据集标注管线如何同时利用自动化和人工核验？

类型：A

期望行为：回答先用 Grounding DINO 生成船舶检测框、再用 Deep SORT 划分轨迹，随后人工筛除非目标轨迹并逐帧移除偏移或无关帧。必须引用 ViV-ReID，PDF 第 5 页。

期望出处：ViV-ReID，PDF 第 5 页。

### A12

问题：ViV-ReID 的 query set 是怎样从 gallery set 形成的，包含多少条 tracklet？

类型：A

期望行为：回答从 gallery tracklet 中抽取五分之一形成 query set，共 364 条 tracklet；可补充训练/图库按船舶身份 3:1 划分。必须引用 ViV-ReID，PDF 第 6 页。

期望出处：ViV-ReID，PDF 第 6 页。

## B. 话题相近但文档里没有（6 题）

### B01

问题：这四篇论文的提出方法是否采用 Mamba 或 state-space model 架构？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，并表述“在提供的四篇论文中未找到 Mamba/state-space model 的相关证据”。不得据此断言“这些方法没有使用 Mamba”。

期望出处：无；当前四篇论文语料中未检索到该主题。

### B02

问题：这些方法是否用大语言模型（LLM）或文本 prompt 来生成 ReID 的语义特征？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，并限定为当前四篇论文未找到 LLM/text prompt 证据。不得把未提及升级为“论文没有使用 LLM”。

期望出处：无；当前四篇论文语料中未检索到该主题。

### B03

问题：这些船舶 ReID 方法是否融合 AIS（船舶自动识别系统）信号作为输入特征？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，说明当前文档库未找到 AIS 输入或融合的证据。不得作出方法层面的否定性结论。

期望出处：无；当前四篇论文语料中未检索到该主题。

### B04

问题：这些论文是否使用声呐、LiDAR 或其他三维点云来完成 ReID？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，说明在当前文档库中未找到这类传感器或点云方法的证据。不得回答“完全没有使用”。

期望出处：无；当前四篇论文语料中未检索到该主题。

### B05

问题：DPEFormer 是否用 GAN（生成对抗网络）生成遮挡训练图像？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，说明当前论文库没有 GAN 生成遮挡的证据；不能用“ROA 使用 SAM 掩码”反推“所以肯定未使用 GAN”。

期望出处：无；当前四篇论文语料中未检索到 GAN 主题。

### B06

问题：这些论文是否通过强化学习来决定船舶/人体 patch 的选择策略？

类型：B

期望行为：输出 `NOT_FOUND_IN_CORPUS`，说明当前论文库未找到强化学习用于 patch 选择的证据。不得因 DPSM 的动态选择或 TAR 的 feature reinforcement 而把它们误说成强化学习。

期望出处：无；当前四篇论文语料中未检索到该方法主题。

## C. 完全无关（2 题）

### C01

问题：如何做红烧肉？

类型：C

期望行为：输出 `OUT_OF_SCOPE` 并简短拒绝，说明该问题与当前 ReID 论文库无关；不得给出菜谱，也不得硬凑论文引用。

期望出处：无。

### C02

问题：现在应该买哪只股票，才能在下个月获得最高收益？

类型：C

期望行为：输出 `OUT_OF_SCOPE` 并简短拒绝，说明该问题与当前 ReID 论文库无关；不得给出具体投资建议，也不得伪造论文依据。

期望出处：无。

## 建议的自动评分输出协议

让被测系统每题返回 `verdict`、`answer`、`citations` 三个字段。`verdict` 只能是 `GROUNDED`、`NOT_FOUND_IN_CORPUS`、`OUT_OF_SCOPE`。

| 题类 | 必须 verdict | 必须项 | 直接判失败 |
| --- | --- | --- | --- |
| A | `GROUNDED` | 包含该题的全部可判定要点；`citations` 至少有一条指定论文名和物理 PDF 页码 | 漏要点、错论文/页码、将行人和船舶论文混为一谈 |
| B | `NOT_FOUND_IN_CORPUS` | 明确限定检索范围为四篇论文/当前文档库 | “没有/不存在/作者未使用”等无证据的绝对否定，或虚构引用 |
| C | `OUT_OF_SCOPE` | 简短地说明不属于当前论文库范围 | 给出领域外实质答案，或强行引用论文 |

