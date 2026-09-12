import React from 'react';
import { Collapse, Spin } from 'antd';
import ReactMarkdown from 'react-markdown';
import { Message } from '../types/chat.types';


interface MessageBubbleProps {
  message: Message;
  isStreaming: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isStreaming }) => {
  const { type, content, toolCall } = message;

  return (
    <div className={`mb-5 flex ${type === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`px-4 py-3 text-[15px] leading-relaxed ${
          type === 'user'
            ? 'bg-[#0071e3] text-white rounded-[18px] rounded-br-md max-w-[75%]'
            : 'bg-white text-[#1d1d1f] rounded-[18px] rounded-tl-md shadow-[0_1px_3px_rgba(0,0,0,0.06)] border border-black/5 max-w-[85%]'
        } message-body`}
      >
          {type === 'ai' && isStreaming && content === '' ? (
            toolCall ? <div><Spin size="small" /> invoking tool...</div> : <Spin size="small" />
          ) : (
            <> 
              {toolCall?.calls && (
                <Collapse defaultActiveKey={['0']} className="mt-2">
                  {toolCall.calls.map((call: any, index: number) => (
                    <Collapse.Panel header={`Tool ${index + 1}: ${call.name}`} key={index}>
                      <p className="mb-2">input：{JSON.stringify(call.args)}</p>
                      {call.result && <p>result：{call.result}</p>}
                    </Collapse.Panel>
                  ))}
                </Collapse>
              )}
              <ReactMarkdown>{content}</ReactMarkdown>
            </>
          )}
        </div>
    </div>
  );
};

export default MessageBubble;