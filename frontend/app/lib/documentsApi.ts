/**
 * 知识库（文档）接口的客户端。
 *
 * 对应后端 `backend/app/api/document_routes.py`。
 *
 * ## 为什么上传之后要轮询、而不是等一个长请求
 *
 * 一篇 PDF 要解析 → 清洗 → 切块 → 逐块算 embedding，411 个块的论文要一分钟以上。
 * 后端因此返回 **202 + 记录 id**，而不是让请求干等（网关和浏览器都会先超时）。
 * 所以上传接口的"完成"语义在这里必须由**轮询 GET /documents/{id}** 补上 ——
 * 拿到 202 不代表文件已经能检索了。
 *
 * ## 为什么错误信息要原样透传给用户
 *
 * 后端的错误体是 FastAPI 的 `{detail: "…"}`，而那句 detail 是**特意写成给人看的**
 * （比如「文件超过 50 MB 上限」「这篇论文已经在知识库里了（id=5）」）。
 * 前端如果统一换成「上传失败，请重试」，这些信息就全丢了 ——
 * 用户就不知道到底是文件太大、格式不对、还是重复了。
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL;

export type DocStatus = "pending" | "indexing" | "indexed" | "failed";

export interface DocumentOut {
  id: number;
  title: string;
  source_file: string;
  status: number;
  status_text: DocStatus;
  chunk_count: number;
  error: string;
  collection_id: number;
  indexed_at: string | null;
}

export interface UploadAccepted {
  id: number;
  title: string;
  status: number;
  status_text: DocStatus;
  poll: string;
}

export interface DocumentList {
  total: number;
  items: DocumentOut[];
}

/** 把 FastAPI 的 `{detail}` 抽出来当错误信息；它不是内部细节，是写给用户看的。 */
async function unwrap(res: Response): Promise<any> {
  if (res.status === 204) return null;          // DELETE 成功没有响应体
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && (body.detail || body.message);
    throw new Error(
      typeof detail === "string" ? detail : `请求失败（HTTP ${res.status}）`
    );
  }
  return body;
}

export async function listDocuments(limit = 100): Promise<DocumentList> {
  return unwrap(await fetch(`${BASE}/documents?limit=${limit}`));
}

export async function getDocument(id: number): Promise<DocumentOut> {
  return unwrap(await fetch(`${BASE}/documents/${id}`));
}

export async function uploadDocument(file: File, title = ""): Promise<UploadAccepted> {
  const form = new FormData();
  form.append("file", file);
  if (title) form.append("title", title);
  // 不要手动设 Content-Type —— 浏览器要自己加 multipart 的 boundary，
  // 手动设成 "multipart/form-data" 会丢掉 boundary，后端直接解析失败。
  return unwrap(await fetch(`${BASE}/documents`, { method: "POST", body: form }));
}

export async function reindexDocument(id: number): Promise<UploadAccepted> {
  return unwrap(await fetch(`${BASE}/documents/${id}/reindex`, { method: "POST" }));
}

export async function deleteDocument(id: number): Promise<void> {
  await unwrap(await fetch(`${BASE}/documents/${id}`, { method: "DELETE" }));
}

/** 还在处理中的状态 —— 只有这两种需要继续轮询。 */
export function isInProgress(doc: DocumentOut): boolean {
  return doc.status_text === "pending" || doc.status_text === "indexing";
}

export const STATUS_LABEL: Record<DocStatus, string> = {
  pending: "排队中",
  indexing: "处理中",
  indexed: "已索引",
  failed: "失败",
};

export const STATUS_COLOR: Record<DocStatus, string> = {
  pending: "default",
  indexing: "processing",
  indexed: "success",
  failed: "error",
};
