/**
 * 会话（conversations）接口的客户端。
 *
 * 对应后端 `backend/app/api/conversation_routes.py`。
 *
 * ## 为什么会话从后端取、不再存 localStorage
 *
 * 之前 sessions / 消息历史都在 localStorage（baseline-defects B4）：
 * 后端重启后界面还显示历史，但后端已经完全失忆 —— **看起来有记忆，实际没有**。
 * 现在消息的 source of truth 是后端的 checkpointer（PostgreSQL），会话列表是
 * 后端的索引表。localStorage 退役，后端拿不到就显示错误，不造本地假数据
 * （与 AgentSelector 动态取 agent 列表是同一条原则）。
 *
 * ## 为什么后端返回的 type 要映射
 *
 * 后端 ChatMessage 的 type 是 LangGraph 口径（"human" / "ai" / "tool"），
 * 前端 Message 用的是界面口径（"user" / "ai" / "tool"）。在客户端边界做一次
 * 显式映射，而不是让两层共用一个词 —— 两边语义并不真的相同。
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL;

export interface ConversationInfo {
  thread_id: string;
  title: string;
  agent_id: string;
  last_message_at: string;
}

interface BackendChatMessage {
  type: "human" | "ai" | "tool" | "custom";
  content: string;
  tool_calls?: { name: string; args: Record<string, unknown>; id: string | null }[];
  tool_call_id?: string | null;
}

function toFrontendMessage(msg: BackendChatMessage, index: number) {
  return {
    id: `hist_${index}_${Date.now()}`,
    type: msg.type === "human" ? "user" : msg.type === "custom" ? "ai" : msg.type,
    content: msg.content,
    // 后端 ChatMessage.tool_calls 是裸列表；前端 Message.toolCall 是 {calls:[...]}
    // 包装 —— 与 useStreamChat 处理 SSE 消息事件时的形状保持一致。
    ...(msg.tool_calls?.length ? { toolCall: { calls: msg.tool_calls } } : {}),
  };
}

export async function fetchConversations(): Promise<ConversationInfo[]> {
  const response = await fetch(`${BASE}/conversations`);
  if (!response.ok) throw new Error(`GET /conversations HTTP ${response.status}`);
  return response.json();
}

export async function fetchConversationMessages(threadId: string) {
  const response = await fetch(`${BASE}/conversations/${threadId}/messages`);
  if (!response.ok) throw new Error(`GET messages HTTP ${response.status}`);
  const list: BackendChatMessage[] = await response.json();
  return list.map(toFrontendMessage);
}

export async function deleteConversation(threadId: string): Promise<void> {
  const response = await fetch(`${BASE}/conversations/${threadId}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    // 404 也算成功：列表数据可能落后于后端（另一个窗口刚删过）。
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail || `DELETE HTTP ${response.status}`);
  }
}
