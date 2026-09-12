import React from 'react';
import { Layout, Menu } from 'antd';
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