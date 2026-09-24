# Scientific Research Harness V1：项目解释

## Q1：原来的科研文献 RAG 和现在的 Scientific Research Harness 有什么本质区别？

原 RAG 的工作单元是一次问题：检索相关段落，让模型基于段落生成带引用的回答。Harness 的工作单元是一个长期研究项目：ResearchQuestion、Hypothesis、Evidence、Experiment、Run、Observation 和 Conclusion 都是独立、可持久化、可关联的科研对象。RAG 仍然存在，但职责收缩为 Literature Evidence Engine，不再直接决定科研结论。

## Q2：为什么不能只靠聊天历史保存科研状态？

聊天历史是自然语言事件流，存在省略、歧义、上下文裁剪和模型重新解释。它难以可靠回答“当前有效假设是什么”“哪个结论经过了批准”“这次实验用了哪些 runs”。Harness 使用关系数据库作为 authority；聊天和 LangGraph state 只是按需加载的临时投影。

## Q3：Evidence 和 Conclusion 有什么区别？

Evidence 是可定位的材料或测量，例如论文第 7 页的一段原文。Conclusion 是研究者基于一组 Evidence 和 Observation 作出的受限解释。Evidence 保存来源，不自动等于科学判断；Conclusion 必须绑定来源并经过批准。

## Q4：为什么 Literature Evidence 和 Experimental Evidence 必须分开？

Literature Evidence 表示其他工作在特定论文中的报告，来源链是 Paper→Page→Chunk。Experimental Evidence 表示本项目实验产生的数据，来源链是 Experiment→Run→artifact/metrics。混在一起会把“论文声称”误写成“我们观察到”，破坏责任和可复现边界。

## Q5：Hypothesis 为什么需要版本和 lineage？

科研假设会随着证据收窄、拆分或修正。`parent_hypothesis_id` 保留 H1→H1.1 的来源，使报告能区分原始假设和后续可检验版本，避免模型静默改写研究命题。

## Q6：为什么要主动搜索 contradictory evidence？

只用假设原句检索会天然偏向相似、支持性材料。Harness 显式执行 Primary、Contradiction 和 Limitation 三个搜索通道，并保存每次 Search Attempt。反向通道的命中只是待审候选，必须经人工确认才能成为正式 CONTRADICT 或 LIMITATION 关系。

## Q7：实验结果为什么不能直接变成 scientific conclusion？

Run metrics 是原始结果；Observation 是对一个或多个 Run 的确定性汇总；Conclusion 才是解释。一次 mAP 上升可能受 seed、对照设置或数据泄漏影响。分层之后，数值不会因为一句模型解释就被升级为”证明有效”。

追问常落在”那 Observation 里就没有主观成分吗”——有。均值和组间差是算出来的，但”这条观察支持哪个假设”是判断，所以 `ObservationHypothesisRelation` 与文献证据关系同级，带 `PROPOSED/CONFIRMED/REJECTED` 审核态；未被人工确认的观察关系不能解锁假设状态跃迁，即使审批已经通过。

## Q8：Harness 如何保证一条 conclusion 可以追踪到论文页码或者实验 run？

Conclusion 通过关系表连接 Evidence 和 Observation。Literature Evidence 保存 source、page/section、chunk ID 和原文 excerpt；Observation 连接 Experiment，Experiment 再连接具体 Runs、metrics、artifact path 和 import hash。`ProvenanceService.get_provenance()` 从数据库关系确定性遍历这些边，不让 LLM 编造 lineage。

## Q9：项目中哪些部分是原项目已有能力？

已有能力包括多格式文档解析、PDF 页码定位、切块、Embedding、Chroma/Qdrant、BM25+向量+RRF、交叉编码器重排、Corrective-RAG、聊天 API、会话 checkpointer、论文元数据表、引用原文弹窗和 RAG 评测。它们被保留，只有入库 metadata 增加了不影响检索行为的稳定 chunk ID。

## Q10：此次真正新增的技术贡献是什么？

新增贡献是结构化科研状态和约束层：七类核心对象、关系数据库 authority、EvidenceRelation
与 ObservationHypothesisRelation 两套审核态、三路证据搜索记录、标准 Run 导入与哈希确认、
确定性 Observation、审批驱动的 Hypothesis 更新、显式 Research Controller、双链 provenance、
结构化 Research Report 以及 Harness 完整性指标。核心价值不是”让模型自动做科研”，而是让模型
参与的科研过程可追踪、可验证、可更新。

