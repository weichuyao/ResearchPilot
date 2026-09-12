"use client";

import React from "react";

import { Layout, Menu, Button, Select } from "antd";
import { useState, useEffect, useRef } from "react";
import { BarsOutlined, DatabaseOutlined, PlusOutlined } from "@ant-design/icons";
import "./globals.css";
import { v4 as uuidv4 } from "uuid";
import { LayoutContext } from "./layout-context";
import SessionListItem from './components/SessionListItem';
import AgentSelector from './components/AgentSelector';
import SiderComponent from './components/SiderComponent';
import KnowledgeBaseDrawer from './components/KnowledgeBaseDrawer';

const { Header, Content } = Layout;

  // Since ReactNode may not be imported correctly, use the more generic type 'any' instead
export default function RootLayout({ children }: { children: any }) {
  const [collapsed, setCollapsed] = useState(false);
  const [sessions, setSessions] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("chatSessions") || "[]");
    } catch (e) {
      return [];
    }
  });


  const [currentThreadId, setCurrentThreadId] = useState(null);

  // 打开页面时默认用哪个 agent。
  // 初始为空：AgentSelector 挂载后从 GET /agents 拉取列表，并把第一个选项
  // 回填进来 —— 「默认是哪个」由后端注册表的顺序决定，前端不再硬编码 key。
  // 列表到达前用户就发消息的话，请求里省略 agent_id，后端用 DEFAULT_AGENT 兜底
  // （见 useStreamChat.ts 与 agents.py）。
  const [agentId, setAgentId] = useState("");

  // 知识库抽屉的开合。state 放在 layout 层而不是抽屉内部，
  // 这样以后想从别处（比如"没有相关文档"的提示里）打开它也有地方挂。
  const [kbOpen, setKbOpen] = useState(false);


  //listen new-chat event
  useEffect(() => {
    const addSession = (event: CustomEvent) => {
      const { threadId, msg } = event.detail;
      handleAddSession(threadId, msg);
    };
    window.addEventListener("add-session", addSession);
    return () => {
      window.removeEventListener("add-session", addSession);
    };
  }, []);

  const handleAddSession = (newThreadId: string, startMsg: string) => {
    if (!newThreadId) {
      newThreadId = uuidv4();
    }
    if (!startMsg) {
      startMsg = `greet ${new Date().toLocaleString()}`;
    }
    const newSession = {
      threadId: newThreadId,
      name: startMsg.substring(0, 10),
      lastUpdated: Date.now(),
    };
    // left sider auto select new session
    setSessions((prev) => [...prev, newSession]);
    setCurrentThreadId(newThreadId);
    localStorage.setItem(
      "chatSessions",
      JSON.stringify([...sessions, newSession])
    );
    window.history.pushState({}, "", `/chat/${newThreadId}`);
  };

  // delete session
  const handleDeleteSession = (delThreadId: string) => {
    const newSessions = sessions.filter(
      (session) => session.threadId !== delThreadId
    );
    setSessions(newSessions);
    localStorage.setItem("chatSessions", JSON.stringify(newSessions));
    localStorage.removeItem( "chatMessages-" + delThreadId);
    if(newSessions.length > 0){
      setCurrentThreadId([...newSessions].reverse()[0]?.threadId || "");
      window.history.pushState({}, "", `/chat/${currentThreadId}`);
    }else{
      setCurrentThreadId(null);
      window.history.pushState({}, "", "/chat");
    }
  };

  const handlerNewChat = () => {
    setCurrentThreadId(null);
    window.history.pushState({}, "", "/chat");
  };

  const selectAgent = (value: string) => {
    console.log("selectAgent", value);
    setAgentId(value);
    handlerNewChat();
  };

  const [items, setItems] = useState([]);
  useEffect(() => {
    const reversedSessions = [...sessions].reverse();
    setItems(() => {
      return reversedSessions.map((session) => ({
        key: session.threadId,
        label: <SessionListItem session={session} onDelete={handleDeleteSession} />,
      }));
    });
  }, [sessions]);

  return (
    <LayoutContext.Provider value={{ agentId, setAgentId, currentThreadId, setCurrentThreadId }}>
      <html>
        <body className="min-h-screen">
          <Layout style={{ minHeight: "auto" }}>
            <SiderComponent
              collapsed={collapsed}
              onCollapse={setCollapsed}
              sessions={sessions}
              handleDeleteSession={handleDeleteSession}
              handlerNewChat={handlerNewChat}
              items={items}
              onSelectSession={(key) => {
                setCurrentThreadId(key);
                const newPath = `/chat/${key}`;
                window.history.pushState({}, "", newPath);
              }}
            />
            <Layout>
              <Header className="bg-white p-0 flex flex-nowrap">
                <BarsOutlined
                  onClick={() => setCollapsed(!collapsed)}
                  className="ml-4 text-xl"
                />
                <div className="flex items-center ml-8 flex-none shrink-0">
                  <span className="text-base">AI-Agent:</span>
                  <AgentSelector value={agentId} onChange={selectAgent} />
                </div>
                <div className="flex items-center ml-4 flex-none shrink-0">
                  {/* 知识库入口。放在 Header 而不是侧边栏：它和"选哪个 agent"一样，
                      属于"这次对话在哪个上下文里进行"的设置，而不是会话列表的一部分。 */}
                  <Button icon={<DatabaseOutlined />} onClick={() => setKbOpen(true)}>
                    知识库
                  </Button>
                </div>
              </Header>
              <Content className="m-4 p-6 bg-white min-h-[calc(100vh-120px)]">
                  {children}
              </Content>
              <KnowledgeBaseDrawer open={kbOpen} onClose={() => setKbOpen(false)} />
            </Layout>
          </Layout>
        </body>
      </html>
    </LayoutContext.Provider>

  );
}
