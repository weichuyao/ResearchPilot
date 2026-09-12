import React, { useState } from 'react';
import { Dropdown, Menu as AntMenu, Button, Input, Modal, message } from 'antd';
import { EllipsisOutlined } from '@ant-design/icons';
import { renameConversation } from '../lib/conversationsApi';

interface SessionListItemProps {
  session: { threadId: string; name: string; lastUpdated: number };
  onDelete: (threadId: string) => void;
  onRename: (threadId: string, title: string) => void;
}

const SessionListItem: React.FC<SessionListItemProps> = ({ session, onDelete, onRename }) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(session.name);
  const [saving, setSaving] = useState(false);

  const save = async () => {
    const title = draft.trim();
    if (!title) return;
    setSaving(true);
    try {
      await renameConversation(session.threadId, title);
      onRename(session.threadId, title);
      setEditing(false);
    } catch (err: any) {
      message.error("重命名失败" + (err?.message ? `：${err.message}` : ""));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex items-center gap-2 w-full min-w-0 flex-1 overflow-visible">
      <span className="flex-1 overflow-hidden text-clip whitespace-nowrap min-w-0">
        {session.name}
      </span>
      <Dropdown
        className="shrink-0 w-6 ml-2 flex-none"
        overlay={
          <AntMenu>
            <AntMenu.Item key="rename" onClick={() => { setDraft(session.name); setEditing(true); }}>
              重命名
            </AntMenu.Item>
            <AntMenu.Item key="delete" danger onClick={() => onDelete(session.threadId)}>
              删除会话
            </AntMenu.Item>
          </AntMenu>
        }
        trigger={["click"]}
      >
        <Button
          icon={<EllipsisOutlined />}
          shape="circle"
          size="small"
          style={{ flexShrink: 0, backgroundColor: "transparent", color: "#6e6e73" }}
        />
      </Dropdown>
      <Modal
        title="重命名会话"
        open={editing}
        onOk={save}
        onCancel={() => setEditing(false)}
        okText="保存"
        cancelText="取消"
        confirmLoading={saving}
        destroyOnClose
      >
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onPressEnter={save}
          maxLength={100}
          placeholder="输入新的会话标题"
        />
      </Modal>
    </div>
  );
};

export default SessionListItem;
