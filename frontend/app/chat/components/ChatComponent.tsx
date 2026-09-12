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

  useEffect(() => {
    if(threadId){
      setCurrentThreadId(threadId)
    }
  });
  
  console.log("chat agentId", agentId)
  console.log("chat threadId", currentThreadId)
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages]);

  const { handleNewChat } = useChatActions({ setMessages, setInput, isStreaming, setIsStreaming });

  // 进入会话时从后端拉历史（改造 #5 之二）。
  // 之前读 localStorage：后端重启后界面还显示历史、后端却已失忆（缺陷 B4）。
  // 现在消息的 source of truth 是后端 checkpointer；拉取失败就空着并报错。
  useEffect(() => {
    if(!currentThreadId || currentThreadId === "") {
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

  const { handleStream } = useStreamChat({ currentThreadId, agentId, setMessages, isStreaming, setIsStreaming });

  const handleSend = async () => {
    setInput("");

    setIsStreaming(true);
    if (!currentThreadId) {
      setCurrentThreadId(uuidv4());
      window.dispatchEvent(
        new CustomEvent("add-session", {
          detail: { threadId: currentThreadId, msg: input },
        })
      );
    }
    await handleStream(input);
  };

  return (
    <div className="h-full">
      <div className="chat-messages overflow-y-auto p-4 h-[calc(100vh-280px)]">
        {messages.length === 0 && (
          <div className="flex flex-col justify-center items-center min-h-full text-gray-600 space-y-2">
            <div className="text-2xl font-medium">Welcome to ResearchPilot</div>
            <div className="text-base">
            You can start typing your questions now and I'll be here to help you!            </div>
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
