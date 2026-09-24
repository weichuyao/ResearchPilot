"use client";

import {
  Alert, Button, Card, Empty, Form, Input, InputNumber, List, Modal, Progress, Select, Space,
  Spin, Tabs, Tag, Typography, message,
} from "antd";
import {
  CheckOutlined, ExperimentOutlined, FileSearchOutlined, PlusOutlined,
  ReloadOutlined, SafetyCertificateOutlined,
} from "@ant-design/icons";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Approval, Evidence, ExperimentDraft, HarnessEvaluation, Hypothesis, ResearchQuestion, ResearchState,
  advanceWorkflow, approveExperiment, completeExperiment, confirmRunImport, createConclusion, createExperiment,
  createHypothesis, createObservation, createResearchQuestion, getHarnessEvaluation,
  getProvenance, getResearchArchive, getResearchReport, getResearchState, listResearchQuestions,
  prepareRunImport, proposeHypothesisUpdate, reviewApproval, reviewEvidenceRelation, searchEvidence,
} from "../lib/researchApi";
import styles from "./research.module.css";

const { Title, Text, Paragraph } = Typography;

const statusColor: Record<string, string> = {
  PROPOSED: "default", TESTABLE: "blue", UNDER_TEST: "processing",
  PARTIALLY_SUPPORTED: "gold", SUPPORTED: "success", REFUTED: "error",
  CONFIRMED: "success", REJECTED: "error", PENDING: "warning", APPROVED: "success",
};

const relationColor: Record<string, string> = {
  SUPPORT: "green", CONTRADICT: "red", LIMITATION: "orange", RELATED: "blue",
};

export default function ResearchWorkbench() {
  const [questions, setQuestions] = useState<ResearchQuestion[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [state, setState] = useState<ResearchState | null>(null);
  const [evaluation, setEvaluation] = useState<HarnessEvaluation | null>(null);
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [questionOpen, setQuestionOpen] = useState(false);
  const [hypothesisOpen, setHypothesisOpen] = useState(false);
  const [experimentOpen, setExperimentOpen] = useState(false);
  const [runOpen, setRunOpen] = useState<string | null>(null);
  const [observationOpen, setObservationOpen] = useState<string | null>(null);
  const [updateOpen, setUpdateOpen] = useState<string | null>(null);
  const [conclusionOpen, setConclusionOpen] = useState(false);
  const [questionForm] = Form.useForm();
  const [hypothesisForm] = Form.useForm();
  const [experimentForm] = Form.useForm();
  const [runForm] = Form.useForm();
  const [observationForm] = Form.useForm();
  const [updateForm] = Form.useForm();
  const [conclusionForm] = Form.useForm();

  const loadQuestions = useCallback(async (preferId?: string) => {
    const data = await listResearchQuestions();
    setQuestions(data.items);
    setSelectedId((current) => preferId ?? current ?? data.items[0]?.id ?? "");
    if (!selectedId && !preferId && data.items[0]) setSelectedId(data.items[0].id);
  }, [selectedId]);

  const refresh = useCallback(async (id: string) => {
    if (!id) { setState(null); setLoading(false); return; }
    setLoading(true);
    setReport("");
    try {
      const [snapshot, metrics] = await Promise.all([
        getResearchState(id), getHarnessEvaluation(id),
      ]);
      setState(snapshot); setEvaluation(metrics);
    } catch (error: any) {
      message.error(error.message);
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { loadQuestions().catch((e) => { message.error(e.message); setLoading(false); }); }, []);
  useEffect(() => { if (selectedId) refresh(selectedId); }, [selectedId, refresh]);

  const evidenceById = useMemo(
    () => new Map((state?.evidence ?? []).map((item) => [item.id, item])), [state?.evidence],
  );
  const proposedRelations = (state?.evidence_relations ?? []).filter((x) => x.review_status === "PROPOSED");
  const pendingApprovals = (state?.approvals ?? []).filter((x) => x.status === "PENDING");

  async function run(key: string, action: () => Promise<unknown>) {
    setBusy(key);
    try {
      await action();
      await refresh(selectedId);
      message.success("科研状态已更新");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitQuestion() {
    const values = await questionForm.validateFields();
    setBusy("question");
    try {
      const created = await createResearchQuestion(values);
      questionForm.resetFields(); setQuestionOpen(false);
      await loadQuestions(created.id); setSelectedId(created.id);
      message.success("研究问题已创建");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitHypothesis() {
    if (!selectedId) return;
    const values = await hypothesisForm.validateFields();
    setBusy("hypothesis");
    try {
      await createHypothesis(selectedId, values);
      hypothesisForm.resetFields(); setHypothesisOpen(false);
      await refresh(selectedId);
      message.success("假设已提交，等待人工注册审批");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function decideApproval(approval: Approval, decision: "APPROVED" | "REJECTED") {
    return approval.action === "APPROVE_EXPERIMENT"
      ? approveExperiment(approval.entity_id, decision)
      : reviewApproval(approval.id, decision);
  }

  async function submitExperiment() {
    if (!state) return;
    const values = await experimentForm.validateFields();
    const csv = (value = "") => value.split(",").map((x: string) => x.trim()).filter(Boolean);
    const payload: ExperimentDraft = {
      research_question_id: state.question.id,
      tested_hypotheses: values.tested_hypotheses,
      purpose: values.purpose,
      independent_variable: values.independent_variable,
      dependent_variables: csv(values.dependent_variables),
      control: values.control,
      treatment: values.treatment,
      controlled_variables: csv(values.controlled_variables),
      metrics: csv(values.metrics),
      success_criteria: { criterion: values.success_criteria },
    };
    setBusy("experiment");
    try {
      await createExperiment(payload); experimentForm.resetFields(); setExperimentOpen(false);
      await refresh(selectedId); message.success("实验方案已创建，等待人工审批");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitRun() {
    if (!runOpen) return;
    const values = await runForm.validateFields();
    let metrics: Record<string, unknown>;
    try { metrics = JSON.parse(values.metrics); }
    catch { message.error("Metrics 必须是合法 JSON 对象"); return; }
    const content = JSON.stringify({
      experiment_id: runOpen, run_id: values.run_id || undefined,
      seed: values.seed, config: { group: values.group }, metrics,
      artifact_paths: values.artifact_paths ? values.artifact_paths.split(",").map((x: string) => x.trim()).filter(Boolean) : [],
      environment: { imported_from: "research-workbench" },
    });
    setBusy("run");
    try {
      const prepared = await prepareRunImport(runOpen, content);
      Modal.confirm({
        title: "确认导入实验结果？",
        content: `完整载荷哈希：${prepared.payload_hash}. 此操作会形成正式 Run 记录。`,
        okText: "批准并导入", cancelText: "暂不导入",
        onOk: async () => {
          try {
            await reviewApproval(prepared.approval.id, "APPROVED");
            await confirmRunImport(runOpen, content, prepared.approval.id);
            runForm.resetFields(); setRunOpen(null); await refresh(selectedId);
            message.success("实验结果已确认导入");
          } catch (error: any) {
            message.error(error.message);
            throw error;
          }
        },
        onCancel: async () => {
          try {
            await reviewApproval(prepared.approval.id, "REJECTED");
            await refresh(selectedId);
          } catch (error: any) { message.error(error.message); }
        },
      });
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitObservation() {
    if (!state || !observationOpen) return;
    const values = await observationForm.validateFields();
    const linked = state.experiment_hypotheses.filter((x) => x.experiment_id === observationOpen);
    const relations = Object.fromEntries(linked.map((x) => [x.hypothesis_id, values.relation]));
    setBusy("observation");
    try {
      await createObservation(observationOpen, values.run_ids, relations);
      observationForm.resetFields(); setObservationOpen(null); await refresh(selectedId);
      message.success("Observation 已由 Run 指标确定性生成");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitUpdate() {
    if (!state || !updateOpen) return;
    const values = await updateForm.validateFields();
    setBusy("update");
    try {
      await proposeHypothesisUpdate(updateOpen, {
        proposed_status: values.proposed_status,
        supporting_evidence: values.supporting_evidence ?? [],
        contradicting_evidence: values.contradicting_evidence ?? [],
        supporting_observations: values.supporting_observations ?? [],
        reasoning: values.reasoning,
        remaining_uncertainty: values.remaining_uncertainty,
        suggested_next_experiment: values.suggested_next_experiment || null,
      });
      updateForm.resetFields(); setUpdateOpen(null); await refresh(selectedId);
      message.success("假设更新已提出，等待人工审批");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function submitConclusion() {
    const values = await conclusionForm.validateFields();
    setBusy("conclusion");
    try {
      await createConclusion(selectedId, {
        statement: values.statement, confidence: values.confidence,
        supporting_evidence_ids: values.supporting_evidence_ids ?? [],
        contradicting_evidence_ids: values.contradicting_evidence_ids ?? [],
        observation_ids: values.observation_ids ?? [],
        limitations: values.limitations ? [values.limitations] : [],
        unresolved_questions: values.unresolved_questions ? [values.unresolved_questions] : [],
      });
      conclusionForm.resetFields(); setConclusionOpen(false); await refresh(selectedId);
      message.success("Conclusion 已创建，等待最终人工审批");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function showProvenance(id: string) {
    try {
      const graph = await getProvenance(id);
      Modal.info({ title: "可追溯证据链", width: 760,
        content: <pre className={styles.provenance}>{JSON.stringify(graph, null, 2)}</pre> });
    } catch (error: any) { message.error(error.message); }
  }

  async function downloadArchive() {
    if (!selectedId) return;
    setBusy("archive");
    try {
      const archive = await getResearchArchive(selectedId);
      const url = URL.createObjectURL(new Blob([JSON.stringify(archive, null, 2)], { type: "application/json" }));
      const link = document.createElement("a");
      link.href = url; link.download = `${selectedId}-research-archive.json`;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      message.success("科研档案已导出并附带 SHA-256 校验值");
    } catch (error: any) { message.error(error.message); }
    finally { setBusy(""); }
  }

  async function loadReport() {
    if (!selectedId) return;
    try { setReport((await getResearchReport(selectedId)).report); }
    catch (error: any) { message.error(error.message); }
  }

  const overview = state ? (
    <>
      {pendingApprovals.length > 0 && (
        <Alert className="mb-4" type="warning" showIcon
          message={`${pendingApprovals.length} 项科研状态等待人工确认`}
          description={<Space direction="vertical" className="mt-2">
            {pendingApprovals.map((approval: Approval) => (
              <Space key={approval.id} wrap>
                <Text code>{approval.action}</Text>
                <Button size="small" type="primary" loading={busy === approval.id}
                  disabled={approval.action.startsWith("CONFIRM_RUN_IMPORT")}
                  onClick={() => run(approval.id, () => decideApproval(approval, "APPROVED"))}>批准</Button>
                <Button size="small" danger disabled={!!busy}
                  onClick={() => run(approval.id, () => decideApproval(approval, "REJECTED"))}>拒绝</Button>
              </Space>
            ))}
          </Space>}
        />
      )}
      <div className={styles.metricGrid}>
        <Metric label="假设" value={state.hypotheses.length} />
        <Metric label="证据" value={state.evidence.length} />
        <Metric label="实验 / Runs" value={`${state.experiments.length} / ${state.runs.length}`} />
        <Metric label="工作流阶段" value={state.workflow?.stage?.replaceAll("_", " ") ?? "未初始化"} compact />
      </div>
      <Space className="mb-4" wrap>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setHypothesisOpen(true)}>提出假设</Button>
        <Button icon={<ExperimentOutlined />} disabled={!state.hypotheses.some((x) => x.status !== "PROPOSED")} onClick={() => setExperimentOpen(true)}>设计实验</Button>
        <Button disabled={!state.evidence.length && !state.observations.length} onClick={() => setConclusionOpen(true)}>形成结论</Button>
        <Button loading={busy === "archive"} onClick={downloadArchive}>导出科研档案</Button>
        <Button icon={<ReloadOutlined />} onClick={() => refresh(selectedId)}>刷新状态</Button>
      </Space>
      <Title level={4}>Hypotheses</Title>
      {state.hypotheses.length ? state.hypotheses.map((hypothesis: Hypothesis) => (
        <Card key={hypothesis.id} className={styles.entityCard} size="small"
          title={<Space><Text code>{hypothesis.id}</Text><Tag color={statusColor[hypothesis.status]}>{hypothesis.status}</Tag></Space>}
          extra={<Space wrap><Button icon={<FileSearchOutlined />} disabled={hypothesis.status === "PROPOSED" || !!busy}
            loading={busy === `search-${hypothesis.id}`}
            onClick={() => run(`search-${hypothesis.id}`, () => searchEvidence(hypothesis.id))}>三路证据检索</Button>
            <Button disabled={!state.observations.length || hypothesis.status === "PROPOSED"}
              onClick={() => setUpdateOpen(hypothesis.id)}>评估假设</Button></Space>}>
          <Paragraph strong>{hypothesis.statement}</Paragraph>
          <Text type="secondary">可检验预测：{hypothesis.prediction}</Text>
        </Card>
      )) : <Empty description="尚未提出结构化假设" />}
      {(state.experiments.length > 0 || state.observations.length > 0) && <>
        <Title level={4} className="mt-6">Experiment state</Title>
        {state.experiments.map((experiment) => (
          <Card key={experiment.id} className={styles.entityCard} size="small"
            title={<Space><ExperimentOutlined /><Text code>{experiment.id}</Text><Tag>{experiment.status}</Tag></Space>}
            extra={<Space wrap>
              {["APPROVED", "RUNNING"].includes(experiment.status) && <Button size="small" onClick={() => setRunOpen(experiment.id)}>导入 Run</Button>}
              {experiment.status === "RUNNING" && state.runs.some((x) => x.experiment_id === experiment.id && x.status === "COMPLETED") &&
                <Button size="small" onClick={() => setObservationOpen(experiment.id)}>生成 Observation</Button>}
              {experiment.status === "RUNNING" && state.observations.some((x) => x.experiment_id === experiment.id) &&
                <Button size="small" type="primary" onClick={() => run(`complete-${experiment.id}`, () => completeExperiment(experiment.id))}>完成实验</Button>}
            </Space>}>
            <Paragraph>{experiment.purpose}</Paragraph>
            <Text type="secondary">Runs：{state.runs.filter((x) => x.experiment_id === experiment.id).length} · Observations：{state.observations.filter((x) => x.experiment_id === experiment.id).length}</Text>
          </Card>
        ))}
      </>}
      {state.conclusions.length > 0 && <>
        <Title level={4} className="mt-6">Conclusions</Title>
        {state.conclusions.map((conclusion) => <Card key={conclusion.id} className={styles.entityCard} size="small"
          title={<Space><Text code>{conclusion.id}</Text><Tag color={statusColor[conclusion.status]}>{conclusion.status}</Tag><Tag>{conclusion.confidence}</Tag></Space>}
          extra={<Button size="small" onClick={() => showProvenance(conclusion.id)}>查看 provenance</Button>}>
          {conclusion.statement}
        </Card>)}
      </>}
      {state.workflow && <Card className={styles.entityCard} title="Workflow controller" size="small">
        <WorkflowActions stage={state.workflow.stage} busy={busy === "workflow"}
          onAdvance={(target) => run("workflow", () => advanceWorkflow(selectedId, target))} />
      </Card>}
    </>
  ) : null;

  const evidenceTab = state ? (
    <>
      {proposedRelations.length > 0 && <Alert className="mb-4" type="info" showIcon
        message={`${proposedRelations.length} 条 EvidenceRelation 等待人工判断`} />}
      {state.evidence_relations.length ? state.evidence_relations.map((relation) => {
        const evidence = evidenceById.get(relation.evidence_id);
        if (!evidence) return null;
        return <EvidenceCard key={relation.id} evidence={evidence} relation={relation}
          busy={busy === relation.id}
          onReview={(decision) => run(relation.id, () => reviewEvidenceRelation(relation.id, decision))} />;
      }) : <Empty description="执行证据检索后，这里会分别展示支持、反对和局限证据" />}
    </>
  ) : null;

  const evaluationTab = evaluation ? (
    <div className={styles.metricGrid}>
      <Score label="证据可追溯率" value={evaluation.evidence_traceability} />
      <Score label="引用结构完整率" value={evaluation.citation_integrity_structural} />
      <Score label="反证检索覆盖率" value={evaluation.contradiction_coverage} />
      <Score label="有依据结论率" value={1 - evaluation.unsupported_conclusion_rate} />
      <Card className={styles.entityCard} style={{ gridColumn: "1 / -1" }}>
        <Space><SafetyCertificateOutlined style={{ color: evaluation.state_consistency.invalid_count ? "#cf1322" : "#2f806a" }} />
          <Text strong>状态一致性</Text><Tag color={evaluation.state_consistency.invalid_count ? "error" : "success"}>
            {evaluation.state_consistency.invalid_count ? `${evaluation.state_consistency.invalid_count} 个非法状态` : "通过"}
          </Tag></Space>
        <Paragraph className="mt-3" copyable={{ text: evaluation.state_fingerprint }}>
          <Text type="secondary">State fingerprint：</Text><Text code>{evaluation.state_fingerprint.slice(0, 20)}…</Text>
        </Paragraph>
      </Card>
    </div>
  ) : <Empty />;

  return <div className={styles.shell}>
    <header className={styles.hero}>
      <div className={styles.eyebrow}>Scientific Research Harness</div>
      <Title level={2} className={styles.title}>科研状态工作台</Title>
      <Text type="secondary">证据、实验和结论各自独立；每次关键状态变化都保留人工确认。</Text>
    </header>
    <div className={styles.workspace}>
      <aside className={styles.projectRail}>
        <Button type="primary" block icon={<PlusOutlined />} onClick={() => setQuestionOpen(true)}>新建研究问题</Button>
        <List dataSource={questions} locale={{ emptyText: "还没有研究项目" }}
          renderItem={(item) => <button onClick={() => { setSelectedId(item.id); setReport(""); }}
            className={`${styles.projectItem} ${selectedId === item.id ? styles.projectActive : ""}`}>
            <Text strong ellipsis style={{ display: "block" }}>{item.title}</Text>
            <Space className="mt-2"><Tag>{item.status}</Tag><Text type="secondary" style={{ fontSize: 11 }}>{item.id.slice(0, 11)}</Text></Space>
          </button>} />
      </aside>
      <main className={styles.main}>
        {loading ? <div className="flex justify-center p-20"><Spin size="large" /></div> : state ? <>
          <Space align="start" className="w-full justify-between" wrap>
            <div><Title level={3} style={{ marginBottom: 4 }}>{state.question.title}</Title><Paragraph type="secondary">{state.question.description}</Paragraph></div>
            <Tag color="cyan">{state.question.status}</Tag>
          </Space>
          <Tabs onChange={(key) => { if (key === "report") loadReport(); }} items={[
            { key: "overview", label: "研究状态", children: overview },
            { key: "evidence", label: `证据关系${proposedRelations.length ? ` (${proposedRelations.length})` : ""}`, children: evidenceTab },
            { key: "evaluation", label: "完整性评估", children: evaluationTab },
            { key: "report", label: "研究报告", children: report ? <pre className={styles.report}>{report}</pre> : <Empty description="尚无可展示报告" /> },
          ]} />
        </> : <Empty description="创建或选择一个研究问题，开始结构化科研流程" />}
      </main>
    </div>

    <Modal title="新建 Research Question" open={questionOpen} onCancel={() => setQuestionOpen(false)}
      onOk={submitQuestion} confirmLoading={busy === "question"} okText="创建">
      <Form form={questionForm} layout="vertical">
        <Form.Item name="title" label="标题" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item name="description" label="明确问题" rules={[{ required: true }]}><Input.TextArea rows={3} /></Form.Item>
        <Form.Item name="background" label="研究背景"><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="scope" label="范围"><Input /></Form.Item>
        <Form.Item name="out_of_scope" label="排除范围"><Input /></Form.Item>
      </Form>
    </Modal>
    <Modal title="提出可检验假设" open={hypothesisOpen} onCancel={() => setHypothesisOpen(false)}
      onOk={submitHypothesis} confirmLoading={busy === "hypothesis"} okText="提交注册审批">
      <Form form={hypothesisForm} layout="vertical">
        <Form.Item name="statement" label="假设陈述" rules={[{ required: true }]}><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="prediction" label="可检验预测" rules={[{ required: true }]}><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="rationale" label="提出依据"><Input.TextArea rows={3} /></Form.Item>
      </Form>
    </Modal>
    <Modal title="设计受控实验" open={experimentOpen} onCancel={() => setExperimentOpen(false)}
      onOk={submitExperiment} confirmLoading={busy === "experiment"} okText="提交审批" width={720}>
      <Form form={experimentForm} layout="vertical">
        <Form.Item name="tested_hypotheses" label="检验假设" rules={[{ required: true }]}>
          <Select mode="multiple" options={(state?.hypotheses ?? []).filter((x) => x.status !== "PROPOSED").map((x) => ({ value: x.id, label: `${x.id} · ${x.statement}` }))} />
        </Form.Item>
        <Form.Item name="purpose" label="实验目的" rules={[{ required: true }]}><Input.TextArea rows={2} /></Form.Item>
        <Space className="w-full" align="start">
          <Form.Item name="independent_variable" label="自变量" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="dependent_variables" label="因变量（逗号分隔）" rules={[{ required: true }]}><Input /></Form.Item>
        </Space>
        <Form.Item name="control" label="对照条件" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item name="treatment" label="处理条件" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item name="controlled_variables" label="控制变量（逗号分隔）"><Input /></Form.Item>
        <Form.Item name="metrics" label="指标（逗号分隔）" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item name="success_criteria" label="成功判据" rules={[{ required: true }]}><Input /></Form.Item>
      </Form>
    </Modal>
    <Modal title="导入标准 Run JSON" open={!!runOpen} onCancel={() => setRunOpen(null)}
      onOk={submitRun} confirmLoading={busy === "run"} okText="校验并申请导入">
      <Form form={runForm} layout="vertical" initialValues={{ group: "treatment", metrics: "{\n  \"mAP\": 88.2\n}" }}>
        <Form.Item name="run_id" label="Run ID（可选）"><Input placeholder="留空则自动生成" /></Form.Item>
        <Space align="start"><Form.Item name="seed" label="Seed"><InputNumber /></Form.Item>
          <Form.Item name="group" label="实验组"><Select options={[{ value: "control" }, { value: "treatment" }]} /></Form.Item></Space>
        <Form.Item name="metrics" label="Metrics JSON" rules={[{ required: true }]}><Input.TextArea rows={5} /></Form.Item>
        <Form.Item name="artifact_paths" label="产物路径（逗号分隔）"><Input /></Form.Item>
      </Form>
    </Modal>
    <Modal title="从已完成 Run 生成 Observation" open={!!observationOpen} onCancel={() => setObservationOpen(null)}
      onOk={submitObservation} confirmLoading={busy === "observation"} okText="确定性计算">
      <Form form={observationForm} layout="vertical" initialValues={{ relation: "INCONCLUSIVE" }}>
        <Form.Item name="run_ids" label="Runs" rules={[{ required: true }]}>
          <Select mode="multiple" options={(state?.runs ?? []).filter((x) => x.experiment_id === observationOpen && x.status === "COMPLETED").map((x) => ({ value: x.id, label: `${x.id} · ${JSON.stringify(x.metrics)}` }))} />
        </Form.Item>
        <Form.Item name="relation" label="与实验所检验假设的关系" rules={[{ required: true }]}>
          <Select options={[{ value: "SUPPORT" }, { value: "CONTRADICT" }, { value: "INCONCLUSIVE" }]} />
        </Form.Item>
      </Form>
    </Modal>
    <Modal title="提出 Hypothesis 状态更新" open={!!updateOpen} onCancel={() => setUpdateOpen(null)}
      onOk={submitUpdate} confirmLoading={busy === "update"} okText="提交人工审批" width={720}>
      <Form form={updateForm} layout="vertical">
        <Form.Item name="proposed_status" label="建议状态" rules={[{ required: true }]}>
          <Select options={["PARTIALLY_SUPPORTED", "SUPPORTED", "WEAKENED", "REFUTED", "INCONCLUSIVE"].map((value) => ({ value }))} />
        </Form.Item>
        <Form.Item name="supporting_evidence" label="支持证据">
          <Select mode="multiple" options={(state?.evidence_relations ?? []).filter((x) => x.hypothesis_id === updateOpen && x.relation === "SUPPORT" && x.review_status === "CONFIRMED").map((x) => ({ value: x.evidence_id, label: x.evidence_id }))} />
        </Form.Item>
        <Form.Item name="contradicting_evidence" label="反对证据">
          <Select mode="multiple" options={(state?.evidence_relations ?? []).filter((x) => x.hypothesis_id === updateOpen && x.relation === "CONTRADICT" && x.review_status === "CONFIRMED").map((x) => ({ value: x.evidence_id, label: x.evidence_id }))} />
        </Form.Item>
        <Form.Item name="supporting_observations" label="实验观察">
          <Select mode="multiple" options={(state?.observations ?? []).map((x) => ({ value: x.id, label: `${x.id} · ${x.description}` }))} />
        </Form.Item>
        <Form.Item name="reasoning" label="理由" rules={[{ required: true }]}><Input.TextArea rows={3} /></Form.Item>
        <Form.Item name="remaining_uncertainty" label="剩余不确定性" rules={[{ required: true }]}><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="suggested_next_experiment" label="建议下一实验"><Input /></Form.Item>
      </Form>
    </Modal>
    <Modal title="形成受证据约束的 Conclusion" open={conclusionOpen} onCancel={() => setConclusionOpen(false)}
      onOk={submitConclusion} confirmLoading={busy === "conclusion"} okText="提交最终审批" width={720}>
      <Form form={conclusionForm} layout="vertical" initialValues={{ confidence: "LOW" }}>
        <Form.Item name="statement" label="结论陈述" rules={[{ required: true }]}><Input.TextArea rows={3} /></Form.Item>
        <Form.Item name="confidence" label="置信度"><Select options={["LOW", "MEDIUM", "HIGH"].map((value) => ({ value }))} /></Form.Item>
        <Form.Item name="supporting_evidence_ids" label="支持证据">
          <Select mode="multiple" options={(state?.evidence_relations ?? []).filter((x) => x.relation === "SUPPORT" && x.review_status === "CONFIRMED").map((x) => ({ value: x.evidence_id, label: x.evidence_id }))} />
        </Form.Item>
        <Form.Item name="contradicting_evidence_ids" label="反对证据">
          <Select mode="multiple" options={(state?.evidence_relations ?? []).filter((x) => x.relation === "CONTRADICT" && x.review_status === "CONFIRMED").map((x) => ({ value: x.evidence_id, label: x.evidence_id }))} />
        </Form.Item>
        <Form.Item name="observation_ids" label="实验观察"><Select mode="multiple" options={(state?.observations ?? []).map((x) => ({ value: x.id, label: x.id }))} /></Form.Item>
        <Form.Item name="limitations" label="局限"><Input.TextArea rows={2} /></Form.Item>
        <Form.Item name="unresolved_questions" label="未解决问题"><Input.TextArea rows={2} /></Form.Item>
      </Form>
    </Modal>
  </div>;
}

function Metric({ label, value, compact = false }: { label: string; value: React.ReactNode; compact?: boolean }) {
  return <div className={styles.metric}><div className={styles.metricLabel}>{label}</div><div className={styles.metricValue} style={compact ? { fontSize: 14, marginTop: 9 } : undefined}>{value}</div></div>;
}

function Score({ label, value }: { label: string; value: number }) {
  return <div className={styles.metric}><Progress type="dashboard" size={86} percent={Math.round(value * 100)} strokeColor="#397c6a" /><div className={styles.metricLabel}>{label}</div></div>;
}

function EvidenceCard({ evidence, relation, busy, onReview }: {
  evidence: Evidence; relation: ResearchState["evidence_relations"][number]; busy: boolean;
  onReview: (decision: "CONFIRMED" | "REJECTED") => void;
}) {
  return <Card className={styles.entityCard} size="small"
    title={<Space wrap><Tag color={relationColor[relation.relation]}>{relation.relation}</Tag><Tag color={statusColor[relation.review_status]}>{relation.review_status}</Tag><Text code>{evidence.id}</Text></Space>}
    extra={relation.review_status === "PROPOSED" && <Space><Button size="small" type="primary" icon={<CheckOutlined />} loading={busy} onClick={() => onReview("CONFIRMED")}>确认关系</Button><Button size="small" danger disabled={busy} onClick={() => onReview("REJECTED")}>拒绝</Button></Space>}>
    <Text strong>{evidence.statement}</Text>
    <div className={styles.evidenceExcerpt}>{evidence.excerpt}</div>
    <Space wrap><Text type="secondary">{evidence.source_title ?? evidence.source_id ?? "未知来源"}</Text>{evidence.page && <Tag>p.{evidence.page}</Tag>}{evidence.section && <Tag>{evidence.section}</Tag>}{evidence.chunk_id && <Text code>{evidence.chunk_id}</Text>}</Space>
  </Card>;
}

const workflowNext: Record<string, string[]> = {
  DEFINE_QUESTION: ["LITERATURE_REVIEW"], LITERATURE_REVIEW: ["GENERATE_HYPOTHESIS"],
  GENERATE_HYPOTHESIS: ["SEARCH_SUPPORTING_EVIDENCE"],
  SEARCH_SUPPORTING_EVIDENCE: ["SEARCH_CONTRADICTING_EVIDENCE"],
  SEARCH_CONTRADICTING_EVIDENCE: ["ASSESS_EVIDENCE"],
  ASSESS_EVIDENCE: ["SEARCH_SUPPORTING_EVIDENCE", "DESIGN_EXPERIMENT"],
  DESIGN_EXPERIMENT: ["HUMAN_APPROVAL"], HUMAN_APPROVAL: ["WAIT_FOR_RESULT", "DESIGN_EXPERIMENT"],
  WAIT_FOR_RESULT: ["IMPORT_RESULT"], IMPORT_RESULT: ["BUILD_OBSERVATION", "WAIT_FOR_RESULT"],
  BUILD_OBSERVATION: ["UPDATE_HYPOTHESIS"], UPDATE_HYPOTHESIS: ["DESIGN_EXPERIMENT", "GENERATE_RESEARCH_REPORT"],
  GENERATE_RESEARCH_REPORT: ["COMPLETE", "UPDATE_HYPOTHESIS"], COMPLETE: [],
};

function WorkflowActions({ stage, busy, onAdvance }: { stage: string; busy: boolean; onAdvance: (target: string) => void }) {
  const next = workflowNext[stage] ?? [];
  return <Space wrap><Tag color="geekblue">{stage}</Tag>{next.length ? next.map((target) =>
    <Button key={target} size="small" loading={busy} onClick={() => onAdvance(target)}>推进至 {target}</Button>
  ) : <Text type="secondary">工作流已完成</Text>}</Space>;
}
