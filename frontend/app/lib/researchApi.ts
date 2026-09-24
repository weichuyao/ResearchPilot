const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8002";

export interface ResearchQuestion {
  id: string;
  title: string;
  description: string;
  background: string;
  scope: string;
  out_of_scope: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Hypothesis {
  id: string;
  statement: string;
  rationale: string;
  prediction: string;
  status: string;
  parent_hypothesis_id?: string | null;
}

export interface Evidence {
  id: string;
  evidence_type: string;
  statement: string;
  excerpt: string;
  source_id?: string | null;
  source_title?: string | null;
  page?: string | null;
  section?: string | null;
  chunk_id?: string | null;
  search_intent: string;
}

export interface EvidenceRelation {
  id: string;
  hypothesis_id: string;
  evidence_id: string;
  relation: "SUPPORT" | "CONTRADICT" | "LIMITATION" | "RELATED";
  review_status: "PROPOSED" | "CONFIRMED" | "REJECTED";
  rationale: string;
}

export interface Approval {
  id: string;
  entity_type: string;
  entity_id: string;
  action: string;
  status: string;
  proposed_changes: Record<string, unknown>;
}

export interface ResearchState {
  question: ResearchQuestion;
  workflow: { stage: string; revision: number } | null;
  hypotheses: Hypothesis[];
  evidence: Evidence[];
  evidence_relations: EvidenceRelation[];
  search_attempts: Array<{ id: string; search_intent: string; result_count: number }>;
  experiments: Array<{ id: string; purpose: string; status: string; metrics: string[] }>;
  experiment_hypotheses: Array<{ experiment_id: string; hypothesis_id: string }>;
  runs: Array<{ id: string; experiment_id: string; status: string; metrics: Record<string, unknown> }>;
  observations: Array<{ id: string; experiment_id: string; description: string; derived_statistics: Record<string, unknown> }>;
  observation_hypotheses: Array<{
    observation_id: string;
    hypothesis_id: string;
    relation: string;
    review_status: string;
    reviewed_by: string | null;
  }>;
  conclusions: Array<{ id: string; statement: string; confidence: string; status: string }>;
  approvals: Approval[];
}

export interface ExperimentDraft {
  research_question_id: string;
  tested_hypotheses: string[];
  purpose: string;
  independent_variable: string;
  dependent_variables: string[];
  control: string;
  treatment: string;
  controlled_variables: string[];
  metrics: string[];
  success_criteria: Record<string, unknown>;
}

export interface HarnessEvaluation {
  evidence_traceability: number;
  citation_integrity_structural: number;
  state_consistency: { invalid_count: number; invalid_entity_ids: string[] };
  unsupported_conclusion_rate: number;
  contradiction_coverage: number;
  state_fingerprint: string;
}

async function unwrap<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail ?? body?.message;
    throw new Error(typeof detail === "string" ? detail : `请求失败（HTTP ${response.status}）`);
  }
  return body as T;
}

function request<T>(path: string, init?: RequestInit): Promise<T> {
  return fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  }).then(unwrap<T>);
}

export async function listResearchQuestions() {
  return request<{ total: number; items: ResearchQuestion[] }>("/research/questions");
}

export async function createResearchQuestion(values: Partial<ResearchQuestion>) {
  const question = await request<ResearchQuestion>("/research/questions", {
    method: "POST",
    body: JSON.stringify(values),
  });
  await request(`/research/questions/${question.id}/workflow/initialize`, {
    method: "POST",
    body: "{}",
  });
  return question;
}

export function getResearchState(id: string) {
  return request<ResearchState>(`/research/questions/${id}/state`);
}

export function createHypothesis(questionId: string, values: Partial<Hypothesis>) {
  return request<{ hypothesis: Hypothesis; registration_approval: Approval }>(
    `/research/questions/${questionId}/hypotheses`,
    { method: "POST", body: JSON.stringify(values) },
  );
}

export function reviewApproval(id: string, decision: "APPROVED" | "REJECTED") {
  return request(`/approvals/${id}/review`, {
    method: "POST",
    body: JSON.stringify({ reviewer: "workbench-user", decision }),
  });
}

export function searchEvidence(hypothesisId: string) {
  return request(`/hypotheses/${hypothesisId}/evidence/search`, {
    method: "POST",
    body: JSON.stringify({ search_mode: "balanced", include_contradiction: true, include_limitation: true }),
  });
}

export function reviewEvidenceRelation(id: string, decision: "CONFIRMED" | "REJECTED") {
  return request(`/evidence-relations/${id}/review`, {
    method: "POST",
    body: JSON.stringify({ reviewer: "workbench-user", decision, rationale: "Reviewed in Research Workbench" }),
  });
}

/** 观察对假设的解读同样要人工确认；链接没有独立 id，用 (observation, hypothesis) 定位。 */
export function reviewObservationRelation(
  observationId: string,
  hypothesisId: string,
  decision: "CONFIRMED" | "REJECTED",
) {
  return request(`/observations/${observationId}/relations/${hypothesisId}/review`, {
    method: "POST",
    body: JSON.stringify({ reviewer: "workbench-user", decision }),
  });
}

export function getHarnessEvaluation(questionId: string) {  return request<HarnessEvaluation>(`/research/questions/${questionId}/evaluation`);
}

export function getResearchReport(questionId: string) {
  return request<{ report: string }>(`/research/questions/${questionId}/report`);
}

export function createExperiment(values: ExperimentDraft) {
  return request<{ experiment: ResearchState["experiments"][number]; approval: Approval }>(
    "/experiments", { method: "POST", body: JSON.stringify(values) },
  );
}

export function approveExperiment(experimentId: string, decision: "APPROVED" | "REJECTED") {
  return request(`/experiments/${experimentId}/approve`, {
    method: "POST",
    body: JSON.stringify({ reviewer: "workbench-user", decision }),
  });
}

export function prepareRunImport(experimentId: string, content: string) {
  return request<{ payload_hash: string; approval: Approval }>(`/experiments/${experimentId}/runs/import`, {
    method: "POST",
    body: JSON.stringify({ source_format: "json", content, requested_by: "workbench-user" }),
  });
}

export function confirmRunImport(experimentId: string, content: string, approvalRequestId: string) {
  return request(`/experiments/${experimentId}/runs/import/confirm`, {
    method: "POST",
    body: JSON.stringify({ source_format: "json", content, requested_by: "workbench-user", approval_request_id: approvalRequestId, actor: "workbench-user" }),
  });
}

export function createObservation(experimentId: string, runIds: string[], relations: Record<string, string>) {
  return request(`/experiments/${experimentId}/observations`, {
    method: "POST",
    body: JSON.stringify({ run_ids: runIds, hypothesis_relations: relations, actor: "workbench-user" }),
  });
}

export function completeExperiment(experimentId: string) {
  return request(`/experiments/${experimentId}/complete`, {
    method: "POST",
    body: JSON.stringify({ actor: "workbench-user" }),
  });
}

export function proposeHypothesisUpdate(hypothesisId: string, values: Record<string, unknown>) {
  return request<Approval>(`/hypotheses/${hypothesisId}/evaluate`, {
    method: "POST",
    body: JSON.stringify({ hypothesis_id: hypothesisId, ...values }),
  });
}

export function createConclusion(questionId: string, values: Record<string, unknown>) {
  return request<{ conclusion: ResearchState["conclusions"][number]; approval: Approval }>(
    `/research/questions/${questionId}/conclusions`,
    { method: "POST", body: JSON.stringify(values) },
  );
}

export function getProvenance(entityId: string) {
  return request<{ root: string; nodes: Array<Record<string, any>>; edges: Array<{ from: string; to: string; type: string }> }>(
    `/research/entities/${entityId}/provenance`,
  );
}

export function advanceWorkflow(questionId: string, target: string) {
  return request(`/research/questions/${questionId}/workflow/advance`, {
    method: "POST",
    body: JSON.stringify({ target, actor: "workbench-user" }),
  });
}

export function getResearchArchive(questionId: string) {
  return request<Record<string, any>>(`/research/questions/${questionId}/export`);
}
