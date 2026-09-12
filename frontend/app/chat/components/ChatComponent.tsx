import React, { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import MessageInput from "../components/MessageInput";
import { useLayoutContext } from '../../layout-context';
import { Message, ChatComponentProps } from '../types/chat.types';
import { useStreamChat } from '../hooks/useStreamChat';
import MessageBubble from '../components/MessageBubble';
import useChatActions from '../hooks/useChatActions';
import { fetchConversationMessages } from '../../lib/conversationsApi';

const ChatComponent: React.FC<ChatComponentProps> = ({
  threadId,
}) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef(null);
  const { agentId, setAgentId, currentThreadId, setCurrentThreadId } = useLayoutContext()

  // URL 参数 -> 会话状态。必须依赖 [threadId]：无依赖数组时每次渲染都执行，
  // 会把「新建对话」刚置空的 currentThreadId 又改回旧值（新建对话因此失效）。
  useEffect(() => {
    if (threadId) {
      setCurrentThreadId(threadId);
    } else {
      setCurrentThreadId(null); // URL 是 /chat 时显式清空，保持 URL 与状态一致
    }
  }, [threadId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const { handleNewChat } = useChatActions({ setMessages, setInput, isStreaming, setIsStreaming });

  // 进入会话时从后端拉历史（改造 #5 之二）。
  // 之前读 localStorage：后端重启后界面还显示历史、后端却已失忆（缺陷 B4）。
  // 现在消息的 source of truth 是后端 checkpointer；拉取失败就空着并报错。
  useEffect(() => {
    if (!currentThreadId || currentThreadId === "") {
      handleNewChat();
      return;
    }
    let cancelled = false;
    fetchConversationMessages(currentThreadId)
      .then((history) => { if (!cancelled) setMessages(history as Message[]); })
      .catch((err) => {
        console.error("加载会话历史失败", err);
        if (!cancelled) setMessages([]);
      });
    return () => { cancelled = true; };
  }, [currentThreadId]);

  const { handleStream, abort } = useStreamChat({ currentThreadId, agentId, setMessages, isStreaming, setIsStreaming });

  // 切换会话时掐掉还在跑的旧流：不 abort 的话，旧流的 token 会继续写进
  // 新会话的消息列表（两个会话共用同一个 messages state，竞态实测存在）。
  useEffect(() => {
    abort();
  }, [currentThreadId]);

  const handleSend = async () => {
    const text = input;
    setInput("");

    // 关键：threadId 必须先落局部变量。setCurrentThreadId 是异步的，
    // 闭包里的 currentThreadId 此时还是旧值（新建会话时是 null）——
    // 之前 add-session 事件和 handleStream 拿到的 id 跟实际写入后端的
    // thread_id 三处各不相同，会话与回答永远对不上。
    const newThreadId = currentThreadId || uuidv4();
    if (!currentThreadId) {
      setCurrentThreadId(newThreadId);
      window.dispatchEvent(
        new CustomEvent("add-session", {
          detail: { threadId: newThreadId, msg: text },
        })
      );
    }
    await handleStream(text, newThreadId);
  };

  return (
    <div className="h-full">
      <div className="chat-messages overflow-y-auto p-4 h-[calc(100vh-280px)]">
        {messages.length === 0 && (
          <div className="flex flex-col justify-center items-center min-h-full text-gray-600 space-y-2">
            <div className="text-2xl font-medium">欢迎使用科研领航</div>
            <div className="text-base">
            在下方输入问题，我会基于文献库为你检索和解答          </div>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} isStreaming={isStreaming} />
        ))}
        <div ref={messagesEndRef} />
      </div>
      <MessageInput
        input={input}
        setInput={setInput}
        handleSend={handleSend}
        isStreaming={isStreaming}
      />
    </div>
  );
};

export default ChatComponent;
