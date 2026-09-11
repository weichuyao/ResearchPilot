"use client";

/**
 * 知识库抽屉：上传文档、看索引进度、重索引、删除。
 *
 * ## 和后端的分工

 * 后端 `POST /documents` 返回 **202 + 记录 id**（受理，不等于做完），
 * 所以这里的"上传完成"必须由**轮询**补上：只要列表里还有 pending/indexing
 * 的文档，就每 1.5 秒拉一次。上传成功 ≠ 能检索到，这两件事在界面上要分开表达。
 *
 * ## 为什么不在前端做文件大小校验

 * 上限（50 MB）是后端的规则，前端再抄一份就会漂移 —— 改了一边忘了另一边，
 * 用户会遇到"界面说可以传、后端说太大"。所以这里只做 `accept` 过滤，
 * 真正的拒绝交给后端，前端负责把它的 `detail` 原样显示出来。
 *
 * ## 为什么 accept 和解析器分不开

 * `accept` 里列的扩展名是后端解析器注册表支持的格式（见 ai/rag/parsers.py）。
 * 两者不同步的话，用户会选到一个后端不认识的格式然后收到 415 ——
 * 所以列表变化时这里也要跟着改。改造 #10 应该把这个列表也改成从后端取。
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Button, Drawer, Empty, List, Popconfirm, Space, Tag, Tooltip, Typography, Upload, message,
} from "antd";
import {
  DeleteOutlined, FileTextOutlined, ReloadOutlined, UploadOutlined,
} from "@ant-design/icons";

import {
  DocumentOut, STATUS_COLOR, STATUS_LABEL,
  deleteDocument, isInProgress, listDocuments, reindexDocument, uploadDocument,
} from "../lib/documentsApi";

const { Text } = Typography;

// 和后端解析器注册表支持的扩展名保持一致（ai/rag/parsers.py）
const ACCEPT = ".pdf,.md,.markdown,.txt,.text,.docx";

// 只要有文档在处理中就轮询。1.5 秒是"看起来实时"和"不把后端问爆"之间的折中；
// 索引一篇论文要几十秒，再快也没意义。
const POLL_MS = 1500;

interface Props {
  open: boolean;
  onClose: () => void;
}

const KnowledgeBaseDrawer: React.FC<Props> = ({ open, onClose }) => {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<Record<number, boolean>>({});
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async (silent = true) => {
    if (!silent) setLoading(true);
    try {
      const data = await listDocuments();
      setDocs(data.items);
    } catch (err: any) {
      // 轮询失败不弹提示：网络抖一下弹一次错误框会把界面变成弹窗工厂。
      // 只有用户主动操作触发的加载失败才报。
      if (!silent) message.error(err.message);
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  // 打开时加载，关闭时停掉轮询（别让关掉的抽屉继续打后端）
  useEffect(() => {
    if (!open) {
      if (timer.current) { clearInterval(timer.current); timer.current = null; }
      return;
    }
    refresh(false);
  }, [open, refresh]);

  // 有文档在处理中就开轮询 —— 没有就停，避免空转
  useEffect(() => {
    const pending = docs.some(isInProgress);
    if (!open || !pending) {
      if (timer.current) { clearInterval(timer.current); timer.current = null; }
      return;
    }
    if (!timer.current) {
      timer.current = setInterval(() => refresh(true), POLL_MS);
    }
    return () => {
      if (timer.current) { clearInterval(timer.current); timer.current = null; }
    };
  }, [open, docs, refresh]);

  const handleUpload = async (file: File) => {
    try {
      const accepted = await uploadDocument(file);
      message.success(`已受理《${accepted.title}》，正在后台索引`);
      await refresh(true);
    } catch (err: any) {
      // 后端把 413/415/409 的 detail 写成了给人看的话，原样显示
      message.error(err.message);
    }
    return false;           // 阻止 antd 自己上传，我们走上面的 fetch
  };

  const withBusy = async (id: number, fn: () => Promise<any>, ok: string) => {
    setBusy((prev) => ({ ...prev, [id]: true }));
    try {
      await fn();
      message.success(ok);
      await refresh(true);
    } catch (err: any) {
      message.error(err.message);
    } finally {
      setBusy((prev) => ({ ...prev, [id]: false }));
    }
  };

  return (
    <Drawer
      title="知识库"
      width={620}
      open={open}
      onClose={onClose}
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => refresh(false)} loading={loading}>
            刷新
          </Button>
          <Upload
            accept={ACCEPT}
            showUploadList={false}
            beforeUpload={handleUpload}
          >
            <Button type="primary" icon={<UploadOutlined />}>上传文档</Button>
          </Upload>
        </Space>
      }
    >
      <Text type="secondary" style={{ fontSize: 12 }}>
        支持 PDF / Markdown / 纯文本 / DOCX。上传后会在后台解析入库，可以关掉这个面板。
      </Text>

      <List
        style={{ marginTop: 12 }}
        loading={loading}
        dataSource={docs}
        locale={{ emptyText: <Empty description="知识库还是空的" /> }}
        renderItem={(doc) => {
          const inProgress = isInProgress(doc);
          return (
            <List.Item
              key={doc.id}
              actions={[
                <Tooltip title={inProgress ? "处理中不能重索引" : "重新解析并覆盖索引"} key="reindex">
                  <Button
                    type="text" size="small" icon={<ReloadOutlined />}
                    disabled={inProgress || busy[doc.id]}
                    onClick={() => withBusy(doc.id, () => reindexDocument(doc.id), "已开始重新索引")}
                  />
                </Tooltip>,
                <Popconfirm
                  key="delete"
                  title="删除这篇文档？"
                  description="向量块、上传的文件、记录都会被删掉。"
                  okText="删除" cancelText="取消"
                  onConfirm={() => withBusy(doc.id, () => deleteDocument(doc.id), "已删除")}
                >
                  <Button type="text" size="small" danger icon={<DeleteOutlined />}
                          disabled={busy[doc.id]} />
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                avatar={<FileTextOutlined style={{ fontSize: 18 }} />}
                title={
                  <Space>
                    <span>{doc.title}</span>
                    <Tag color={STATUS_COLOR[doc.status_text]}>
                      {STATUS_LABEL[doc.status_text]}
                    </Tag>
                  </Space>
                }
                description={
                  <>
                    <div>
                      {doc.status_text === "indexed" && `${doc.chunk_count} 个块`}
                      {inProgress && "正在解析入库…"}
                      {doc.status_text === "failed" && (
                        // 失败原因必须显示出来。后端特意把它存在记录里
                        // （而不是只打日志）就是为了让用户能在这里看到。
                        <Text type="danger">失败：{doc.error || "未知原因"}</Text>
                      )}
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {doc.source_file}
                    </Text>
                  </>
                }
              />
            </List.Item>
          );
        }}
      />
    </Drawer>
  );
};

export default KnowledgeBaseDrawer;
