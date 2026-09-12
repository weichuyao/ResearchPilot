import React, { useMemo, useState } from 'react';
import { Button, Modal, Spin, Tag } from 'antd';
import { FileTextOutlined } from '@ant-design/icons';
import { ensureDocIndex, lookupPassages, PassageLookup } from '../../lib/documentsApi';

interface Props {
  content: string; // AI 回答的 markdown 原文，从中提取引用
}

const CITE_RE = /(?:p\.(\d{1,3})|第\s*(\d{1,3})\s*页)/g;

/**
 * 引用定位 chips：扫描回答里的「p.5 / 第 5 页」引用，点击直接展示该页原文。
 *
 * 论文归属的判定是启发式：取引用前 120 字符窗口，与文档索引里的标题算词重合度，
 * 取最匹配的。回答里通常在引用前不远就写了论文标题，实战够用；解析失败会在
 * 弹窗里如实说，不会硬凑。
 */
const CitationChips: React.FC<Props> = ({ content }) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PassageLookup | null>(null);
  const [error, setError] = useState<string | null>(null);

  const citations = useMemo(() => {
    const seen = new Set<number>();
    const list: { page: number; hint: string }[] = [];
    for (const m of Array.from(content.matchAll(CITE_RE))) {
      const page = Number(m[1] ?? m[2]);
      if (!page || seen.has(page)) continue;
      seen.add(page);
      const windowStart = Math.max(0, (m.index ?? 0) - 120);
      list.push({ page, hint: content.slice(windowStart, m.index ?? 0) });
    }
    return list.slice(0, 6); // 一条回答最多给 6 个，防刷屏
  }, [content]);

  if (citations.length === 0) return null;

  const open_ = async (page: number, hint: string) => {
    setOpen(true);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const docs = await ensureDocIndex();
      const words = hint.toLowerCase().match(/[a-z\u4e00-\u9fff]{3,}/g) ?? [];
      let best: { id: number; title: string } | null = null;
      let bestScore = 0;
      for (const doc of docs) {
        const titleL = doc.title.toLowerCase();
        const hit = words.filter((w) => titleL.includes(w)).length;
        if (hit > bestScore) {
          bestScore = hit;
          best = doc;
        }
      }
      if (!best || bestScore === 0) {
        setError("没能从回答中定位到论文 —— 试试在提问时写明论文标题");
        return;
      }
      setResult(await lookupPassages(best.title, page));
    } catch (err: any) {
      setError(err?.message || "原文获取失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {citations.map(({ page, hint }) => (
          <Button
            key={`${page}-${hint.slice(0, 8)}`}
            size="small"
            icon={<FileTextOutlined />}
            onClick={() => open_(page, hint)}
          >
            查看第 {page} 页原文
          </Button>
        ))}
      </div>
      <Modal
        title={result ? `${result.title} · 第 ${result.page} 页原文` : "原文定位"}
        open={open}
        onCancel={() => setOpen(false)}
        footer={null}
        width={640}
      >
        {loading && <Spin />}
        {error && <div style={{ color: "#b91c1c" }}>{error}</div>}
        {result && (
          <>
            {result.passages.length === 0 ? (
              <div style={{ color: "#6e6e73" }}>
                该页没有已索引的文本块。这篇论文已索引的页码：
                {result.known_pages.map((p) => (
                  <Tag key={p} style={{ margin: 2 }}>{p}</Tag>
                ))}
              </div>
            ) : (
              result.passages.map((t, i) => (
                <p key={i} style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, marginBottom: 12 }}>
                  {t}
                </p>
              ))
            )}
          </>
        )}
      </Modal>
    </>
  );
};

export default CitationChips;
