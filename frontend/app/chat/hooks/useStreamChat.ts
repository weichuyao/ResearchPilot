import { message } from "antd";
import { Dispatch, SetStateAction, useRef } from "react";
import { Message } from "../types/chat.types";

interface UseStreamChatProps {
  currentThreadId: string;
  agentId: string;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  isStreaming: boolean;
  setIsStreaming: (value: boolean) => void;
}

export const useStreamChat = ({
  currentThreadId,
  agentId,
  setMessages,
  isStreaming,
  setIsStreaming,
}: UseStreamChatProps) => {
  // 流式请求的中止控制：切换会话 / 新建对话时把还在跑的旧流掐掉，
  // 否则旧流的 setState 会写进新会话的消息列表（竞态实测存在）。
  const abortRef = useRef<AbortController | null>(null);

  const abort = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const handleStream = async (input: string, threadIdOverride?: string) => {
    if (!input.trim() || isStreaming) return;
    setIsStreaming(true);
    abort();

    const newUserMessage: Message = {
      id: `user_${Date.now()}`,
      type: "user",
      content: input,
    };
    const newAiMessage: Message = {
      id: `ai_${Date.now()}`,
      type: "ai",
      content: "",
    };
    setMessages((prev: Message[]) => [...prev, newUserMessage, newAiMessage]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // agentId 为空（/agents 列表还没回来）时省略 agent_id，
      // 让后端的 DEFAULT_AGENT 决定 —— 不能发空字符串过去，后端 get_agent("") 会 500。
      const requestMsg = {
        thread_id: threadIdOverride ?? currentThreadId,
        role: "user",
        message: input,
        ...(agentId ? { agent_id: agentId } : {}),
      };

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestMsg),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      // SSE 按 TCP 分包到达，一条 "data: {...}" 可能被切成两个 chunk ——
      // 必须维护行缓冲，只处理完整行，残余留给下一轮拼接。
      let buffer = "";
      let streamEnded = false;

      const processLine = (line: string) => {
        if (!line.startsWith("data: ")) return;
        const payload = line.slice(6).trim();
        if (!payload) return;
        let data: any;
        try {
          data = JSON.parse(payload);
        } catch {
          console.warn("跳过无法解析的 SSE 行:", payload.slice(0, 80));
          return;
        }
        switch (data.type) {
          case "message":
            handleMessageData(data.content);
            break;
          case "token":
            handleTokenData(data.content);
            break;
          case "error":
            // 后端在工具/生成出错时会推 error 事件 —— 之前被静默丢弃，
            // 用户只看到流"正常结束"却没有答案。
            message.error(`回答出错：${data.content ?? "未知错误"}`);
            break;
          case "end":
            streamEnded = true;
            reader.cancel().catch(() => {});
            break;
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? ""; // 最后一段可能是半行，留到下一轮
        lines.forEach(processLine);
        if (streamEnded) break;
      }
      // 流结束但服务端没发 end（异常断开）：不能让 isStreaming 永远卡住
      if (!streamEnded) {
        message.warning("连接中断，回答可能不完整");
      }
    } catch (error: any) {
      if (error?.name === "AbortError") {
        // 主动中止（切换会话/新建对话）：不是错误，静默
      } else {
        console.error(" Request Failed:", error);
        message.error("请求失败" + (error?.message ? `：${error.message}` : "，请稍后重试"));
      }
    } finally {
      // 兜底：无论成功、出错还是中断，输入框都必须解锁
      if (abortRef.current === controller) abortRef.current = null;
      setIsStreaming(false);
    }
  };

  const handleMessageData = (content: any) => {
    if (content.type === "ai" && content.tool_calls?.length > 0) {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last) return prev;
        const calls = last.toolCall?.calls || [];
        // 去重：已有 calls 里不存在同 id 的才加入（旧逻辑比较方向写反，会重复添加）
        const addCalls = content.tool_calls.filter(
          (toolCall: any) => !calls.some((c: any) => c.id === toolCall.id)
        );
        if (addCalls.length === 0) return prev;
        return prev.map((msg, i) =>
          i === prev.length - 1
            ? { ...msg, toolCall: { calls: [...(msg.toolCall?.calls || []), ...addCalls] } }
            : msg
        );
      });
    } else if (content.type === "ai" && content.content) {
      setMessages((prev) =>
        prev.map((msg, i) =>
          i === prev.length - 1 ? { ...msg, content: content.content } : msg
        )
      );
    }
    if (content.type === "tool") {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last?.toolCall?.calls) return prev; // tool 结果先于带 tool_calls 的消息到达时别炸
        const updatedCalls = last.toolCall.calls.map((call: any) =>
          call.id === content.tool_call_id ? { ...call, result: content.content } : call
        );
        return prev.map((msg, i) =>
          i === prev.length - 1 ? { ...msg, toolCall: { calls: [...updatedCalls] } } : msg
        );
      });
    }
  };

  const handleTokenData = (token: string) => {
    setMessages((prev) =>
      prev.map((msg, i) =>
        i === prev.length - 1 ? { ...msg, content: msg.content + token } : msg
      )
    );
  };

  return { handleStream, abort };
};
