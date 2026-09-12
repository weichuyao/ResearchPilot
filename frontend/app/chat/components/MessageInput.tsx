import React from "react";
import { Button, Input } from "antd";

interface MessageInputProps {
  input: string;
  setInput: (value: string) => void;
  handleSend: () => void;
  isStreaming: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ input, setInput, handleSend, isStreaming }) => {
  return (
    <div className="p-4 border-t">
      <div className="flex gap-2 items-center">
        <Input.TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入你的问题…"
          onKeyDown={(e) => {
            // keyPress 不区分修饰键也不感知输入法：Shift+Enter 会被当发送、
            // 中文 IME 确认候选词也会被当发送 —— 对中文用户是高频事故。
            if (
              e.key === "Enter" &&
              !(e.nativeEvent as any).isComposing
            ) {
              e.preventDefault();
              if (e.shiftKey || e.ctrlKey) {
                setInput(input + "\n"); // shift/ctrl + enter 换行
              } else {
                handleSend();
              }
            }
          }}
          disabled={isStreaming}
          className="flex-1 min-h-[80px] p-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition-colors"
          autoSize={{ minRows: 3, maxRows: 5 }}
        />
        <Button
          type="primary"
          className="h-10 px-6 rounded-lg font-medium shadow-sm"
          onClick={handleSend}
          disabled={!input.trim() || isStreaming}
        >
          {isStreaming ? "回答中…" : "发送"}
        </Button>
      </div>
    </div>
  );
};

export default MessageInput;