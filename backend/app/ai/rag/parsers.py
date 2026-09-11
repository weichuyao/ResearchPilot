"""文档解析器注册表：把各种格式拆成「可定位的片段」。

## 为什么需要「位置标签」这个抽象，而不是直接叫「页码」

PDF 有页，Markdown 有章节，纯文本有段落 —— **它们没有共同的分页概念**。
如果为了统一而把这个字段叫 `page`，那就等于对 Markdown 撒谎：
引用里会写「第 2 页」，而那个文件根本没有页。

所以每个解析器输出 `Section(label, text, kind)`：

| 格式 | kind | 标签前缀 | 标签含义 | 引用长这样 |
|---|---|---|---|---|
| PDF | page | `p` | 物理页码（1 起） | `p.5` |
| Markdown | section | `sec` | 第 N 个标题段落 | `sec.2` |
| DOCX | section | `sec` | 第 N 个标题段落 | `sec.3` |
| 纯文本 | block | `blk` | 第 N 个段落块 | `blk.4` |

标签前缀会写进块的 metadata（`locator_prefix`），检索工具据此拼出引用。
**引用格式是用户唯一会核对的东西，这里说错比不说更糟。**

## 为什么解析和清洗分开

`parse()` 只负责「把文件变成文本片段」，不做任何清洗。
清洗（页眉页脚、连字还原、换行断词）是**格式相关的**，所以放在 `clean()`：
PDF 需要跨页重复行检测和 `represen-\ntation` 这种断词合并，
而把同一套规则套到 Markdown 上会把正常的硬换行也吃掉。

默认 `clean()` 只做字符还原（见 ai/rag/textnorm.py），PdfParser 覆写它加更多步骤。
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass

from ai.rag.textnorm import normalize_typography

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Section:
    """文档里一个可定位的片段。"""

    label: str    # 位置标签，引用里显示在 `p.` / `sec.` 之后
    text: str     # 原始文本（清洗由 clean() 做）
    kind: str     # page | section | block


@dataclass(frozen=True)
class ParserInfo:
    """给接口层做校验用的信息。"""

    kind: str
    prefix: str
    extensions: tuple[str, ...]
    magic: bytes | None      # 文件头；None 表示这种格式没有可靠的魔数
    magic_note: str


class Parser:
    """解析器基类。子类覆写 parse()，需要时覆写 clean()。"""

    info: ParserInfo

    def parse(self, path: str) -> list[Section]:
        raise NotImplementedError

    def clean(self, sections: list[Section]) -> list[Section]:
        """默认清洗：只做字符还原（连字、破折号）。

        多页格式的页眉页脚剥离、断词合并由子类覆写加入。
        """
        return [Section(s.label, normalize_typography(s.text), s.kind) for s in sections]


# ---------------------------------------------------------------------------
# PDF 专属的清洗
#
# 这三个函数放在这里而不是 ai/rag/ingest.py：它们是「PDF 这个格式怎么读才干净」的
# 知识，属于解析器。放在 ingest 里会让 parsers 反过来 import ingest，形成循环。
# ---------------------------------------------------------------------------
def _norm(line: str) -> str:
    """归一化一行，用于判断它是否在每页重复。数字会被抹掉，因为页码会变。"""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip().lower()


def find_repeated_edge_lines(pages: list[str]) -> set[str]:
    """找出在超过一半页面上重复出现的首行/末行 —— 也就是页眉和页脚。

    用「跨页重复」来识别，而不是硬编码某本刊物的页眉格式，这样换一批论文也能用。

    少于 3 页时直接返回空集：样本太小，"重复"可能只是巧合。
    """
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for text in pages:
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            continue
        for edge in {_norm(lines[0]), _norm(lines[-1])}:
            if edge:
                counts[edge] = counts.get(edge, 0) + 1
    threshold = max(2, len(pages) // 2)
    return {key for key, count in counts.items() if count >= threshold}


def clean_page_text(text: str, noise: set[str]) -> str:
    """去掉页眉页脚、还原连字、并修复被换行拆开的单词。"""
    kept = [line for line in text.split("\n") if not (line.strip() and _norm(line) in noise)]
    joined = "\n".join(kept)
    # represen-\ntation → representation
    # 注意：这也会把行尾的真实连字符合并（well-\nknown → wellknown），属于已知取舍。
    # 换行拆词在两端对齐的论文里非常常见，而真实连字符恰好落在行尾的情况少得多。
    return normalize_typography(re.sub(r"(\w)-\n(\w)", r"\1\2", joined))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class PdfParser(Parser):
    info = ParserInfo(
        kind="page", prefix="p", extensions=(".pdf",),
        magic=b"%PDF-",
        magic_note="PDF 文件头",
    )

    def parse(self, path: str) -> list[Section]:
        # 延迟导入：pypdf / pdfminer 都比较重，不解析 PDF 的进程不必付这个代价
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = _extract_pdf_pages(path, fallback=reader)
        return [
            Section(label=str(index + 1), text=text, kind="page")
            for index, text in enumerate(pages)
        ]

    def clean(self, sections: list[Section]) -> list[Section]:
        """PDF 的清洗比别的格式多两步，都是 PDF 特有的：

          1. **跨页重复行**（页眉页脚）。判断依据是同一行在超过一半的页面上出现 —— 
             数字会被抹掉再比，因为页码每页都不同。
          2. **断词合并**：两端对齐的排版会把单词从行尾切开（`represen-\ntation`）。

        这两步都**不能**套到 Markdown/纯文本上：那里没有页眉，硬换行也是作者写的。
        """
        noise = find_repeated_edge_lines([s.text for s in sections])
        return [
            Section(s.label, clean_page_text(s.text, noise), s.kind)
            for s in sections
        ]


def _collect_text_lines(obj, out: list[str]) -> None:
    """递归收集 LTTextLine 的文本，遇到 LTFigure 整棵子树跳过。"""
    from pdfminer.layout import LTContainer, LTFigure, LTTextLine

    if isinstance(obj, LTFigure):
        return
    if isinstance(obj, LTTextLine):
        text = obj.get_text().rstrip()
        if text:
            out.append(text)
        return
    if isinstance(obj, LTContainer):
        for child in obj:
            _collect_text_lines(child, out)


def _extract_pdf_pages(path: str, fallback=None) -> list[str]:
    """用 pdfminer 抽取每一页的文本，跳过图表子树。

    为什么不用 pypdf 的 `page.extract_text()`：它是**流式抽取，没有版面感知**。
    A²RNet 第 3 页上，正文那句

        "...(i) ship type, (ii) imaging perspective, and (iii) loading and
         equipment configuration"

    被 Fig. 3 的图形标签、图注和一张概率表隔开了约 **1600 字符**，而 pypdf
    忠实地还原了这个错乱的顺序。后果有两层：句子被切断（模型只能答出前两项），
    而且含答案的那一块被数字稀释、embedding 质量下降、排不进候选。

    pdfminer 会把用 Form XObject 画的图包成 LTFigure 节点，**整棵子树跳过**即可。
    图内文字（`Instance Bank` / `Conv` / `MP` / `0.9977` 这类）根本不会出现。
    实测同一页：图内标签和概率表数字全部消失，`(iii) loading and equipment
    configuration` 回到正文里，与前半句的距离从约 1600 字符降到约 350 字符 ——
    足够让它们落进同一个 800 字符的块。

    pypdf 仍然保留：读 `/Title` 元数据比 pdfminer 方便，而且这里是它的兜底路径。
    抽取失败时退回 pypdf（结果差，但总比整篇丢掉好）—— 和 rerank 的降级是同一个思路。
    """
    from pdfminer.high_level import extract_pages

    try:
        pages: list[str] = []
        for layout in extract_pages(path):
            lines: list[str] = []
            _collect_text_lines(layout, lines)
            pages.append("\n".join(lines))
        if pages:
            return pages
        logger.warning("pdfminer 没有抽出任何页面，改用 pypdf：%s", path)
    except Exception as exc:
        logger.warning("pdfminer 抽取失败（%s: %s），改用 pypdf：%s",
                       type(exc).__name__, exc, path)

    if fallback is not None:
        return [page.extract_text() or "" for page in fallback.pages]
    return []


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownParser(Parser):
    """按标题切段。

    为什么按标题而不是按行数：Markdown 作者用标题**表达了结构**，
    那是最自然的定位单位。按固定行数切出来的标签对读者毫无意义。
    """

    info = ParserInfo(
        kind="section", prefix="sec", extensions=(".md", ".markdown"),
        magic=None,
        magic_note="Markdown 是纯文本，没有可靠的魔数；只看扩展名",
    )

    def parse(self, path: str) -> list[Section]:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()

        sections: list[Section] = []
        current: list[str] = []
        for line in lines:
            if HEADING_RE.match(line) and current:
                sections.append(_section(sections, current))
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append(_section(sections, current))
        return [s for s in sections if s.text.strip()]


def _section(done: list[Section], lines: list[str]) -> Section:
    return Section(label=str(len(done) + 1), text="\n".join(lines), kind="section")


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
class DocxParser(Parser):
    """按 Word 的标题样式切段，表格内容也抽出来。

    表格为什么不跳过：技术文档里的参数表、对比表常常是**唯一**写清某个事实的地方
    （"支持哪些格式"这类问题答案往往就在一张表里）。丢掉表格等于丢掉内容。
    """

    info = ParserInfo(
        kind="section", prefix="sec", extensions=(".docx",),
        magic=b"PK\x03\x04",
        magic_note="docx 本质是 zip，文件头是 PK；但 .zip 也是这个头，"
                   "所以还要靠扩展名做第二重判断",
    )

    def parse(self, path: str) -> list[Section]:
        import docx

        document = docx.Document(path)
        sections: list[Section] = []
        current: list[str] = []

        for paragraph in document.paragraphs:
            text = (paragraph.text or "").strip()
            is_heading = (paragraph.style is not None
                          and (paragraph.style.name or "").lower().startswith("heading"))
            if is_heading and current:
                sections.append(_section(sections, current))
                current = []
            if text:
                current.append(text)

        for index, table in enumerate(document.tables, start=1):
            rows = []
            for row in table.rows:
                cells = [(cell.text or "").strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                current.append("[table %d]\n%s" % (index, "\n".join(rows)))

        if current:
            sections.append(_section(sections, current))
        return [s for s in sections if s.text.strip()]


# ---------------------------------------------------------------------------
# 纯文本
# ---------------------------------------------------------------------------
class TextParser(Parser):
    """按段落块切。

    标签用**块序号**而不是行号：行号看起来更精确，但它依赖编辑器怎么折行 ——
    同一个文件在不同编辑器里行号不同，对不上。块序号至少是稳定的。
    块按空行分组、累积到约 BLOCK_CHARS 字符就切一刀，避免一个几千行的日志
    变成一整块（那样引用没有意义，检索也没法定位）。
    """

    BLOCK_CHARS = 1500

    info = ParserInfo(
        kind="block", prefix="blk", extensions=(".txt", ".text"),
        magic=None,
        magic_note="纯文本没有魔数；只看扩展名。不改扩展名冒充的二进制文件会在解码时露馅",
    )

    def parse(self, path: str) -> list[Section]:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        blocks: list[Section] = []
        buffer: list[str] = []
        size = 0
        for paragraph in re.split(r"\n\s*\n", raw):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if buffer and size + len(paragraph) > self.BLOCK_CHARS:
                blocks.append(_section(blocks, buffer))
                buffer, size = [], 0
            buffer.append(paragraph)
            size += len(paragraph)
        if buffer:
            blocks.append(_section(blocks, buffer))
        return [s for s in blocks if s.text.strip()]


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
PARSERS: tuple[Parser, ...] = (PdfParser(), MarkdownParser(), DocxParser(), TextParser())

_BY_EXTENSION = {ext: parser for parser in PARSERS for ext in parser.info.extensions}


def supported_extensions() -> tuple[str, ...]:
    """接口层校验用。从注册表推导而不是另写一份 —— 两份会漂移。"""
    return tuple(sorted(_BY_EXTENSION))


def parser_for(filename: str) -> Parser | None:
    ext = os.path.splitext(filename or "")[1].lower()
    return _BY_EXTENSION.get(ext)


def describe_formats() -> str:
    """给错误信息用的一句话格式说明。"""
    return "、".join(supported_extensions())


def parse_document(path: str) -> tuple[list[Section], ParserInfo]:
    """解析一个文件，返回 (清洗过的片段, 解析器信息)。"""
    parser = parser_for(path)
    if parser is None:
        # 只报扩展名，不报整个路径 —— 路径太长时会被日志截断，
        # 反而看不出到底缺的是什么（实测踩过：一个 133 字符的没有扩展名的路径，
        # 报错信息里只看到一串论文标题，看不到"没有扩展名"这个事实）
        ext = os.path.splitext(path)[1].lower()
        raise ValueError("不支持的格式：%s（支持 %s）"
                         % (ext or "文件没有扩展名", describe_formats()))
    sections = parser.clean(parser.parse(path))
    return sections, parser.info
