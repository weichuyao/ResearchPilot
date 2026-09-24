import React from 'react';
import Link from 'next/link';
import { Button, Layout, Menu } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';
import NewChatButton from './NewChatButton';
import { useLayoutContext } from '../layout-context'


interface SiderComponentProps {
  collapsed: boolean;
  onCollapse: (collapsed: boolean) => void;
  sessions: Array<{ threadId: string; name: string; lastUpdated: number }>;
  handleDeleteSession: (threadId: string) => void;
  handlerNewChat: () => void;
  items: Array<{ key: string; label: React.ReactNode }>;
  onSelectSession: (key: string) => void;
}

const { Sider } = Layout;

const SiderComponent: React.FC<SiderComponentProps> = ({ 
  collapsed, 
  onCollapse, 
  sessions, 
  handleDeleteSession, 
  handlerNewChat, 
  items,
  onSelectSession
}) => {
  const { currentThreadId, setCurrentThreadId } = useLayoutContext()

  return (
    <Sider
      theme="light"
      collapsible
      collapsed={collapsed}
      onCollapse={onCollapse}
      width={200}
    >
      {!collapsed && (
        <div className="logo flex items-center justify-center h-16 text-[#1d1d1f] text-base font-semibold tracking-wide">
          RESEARCHPILOT
        </div>
      )}
      <NewChatButton collapsed={collapsed} onClick={handlerNewChat} />
      <Link href="/research" style={{ display: "block", margin: "0 16px 12px" }}>
        <Button
          icon={<ExperimentOutlined />}
          block={!collapsed}
          shape={collapsed ? "circle" : "round"}
          style={collapsed ? { width: 40 } : undefined}
        >
          {!collapsed && "科研工作台"}
        </Button>
      </Link>
      {!collapsed && (
        <Menu
          theme="light"
          className="max-h-[calc(100vh-180px)] overflow-y-auto"
          defaultSelectedKeys={[currentThreadId]}
          selectedKeys={[currentThreadId]}
          mode="inline"
          items={items}
          onSelect={({ key }) => {
            onSelectSession(key);
          }}
        />
      )}
    </Sider>
  );
};

export default SiderComponent;
