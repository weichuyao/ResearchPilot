"""论文导入管道：PDF → 清洗 → 切块 → 向量化 → Chroma。

放在 app/ai/rag/ 而不是 tests/ 下，是为了让以后 FastAPI 的文档上传接口
（改造 #5）能复用同一套解析与切块逻辑，避免出现两份实现。

标题的三种来源，优先级从高到低：

    1. titles.json 里人工填写的 title      ← 最高，人工确认过
    2. PDF 内置元数据 /Title               ← 实测 4 篇里只有 2 篇有
    3. 文件名清洗                          ← 兜底，可能不准确

titles.json 不用手写：脚本第一次运行时会**自动生成**，把每篇论文的自动推导结果
和来源都列出来，你只需要把 title 填上。填完重跑即可。

命令行用法：

    python app/ai/rag/ingest.py
    python app/ai/rag/ingest.py --folder resource/papers --no-reset
"""

from __future__ import annotations

import glob
import html
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ---- 切块参数 ----------------------------------------------------------------
# 实测：这批论文共 50 页 / 约 26.8 万字符，每页中位约 5400 字符。
# 学术论文一个标准段落大约 400~800 字符，800 大致能装下一个完整论述；
# 重叠 150 是为了避免把一个论点从中间切断。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 标题清单的文件名（相对于论文目录）
MANIFEST_NAME = "titles.json"

# 一个位置片段短于这个字符数就丢掉。
#
# 页眉页脚剥离、图表文字过滤之后，PDF 里会剩下一堆只有一个词甚至一个公式符号的
# 残片。它们进向量库只会污染检索 —— 实测这类残片的余弦分数虚高
# （和查询字面重合度高但没有信息量），会把真正有内容的段落挤下去。
MIN_SECTION_CHARS = 50

_MANIFEST_HELP = (
    "只需要填写每篇论文的 title 字段（留空则使用脚本自动推导的结果）。"
    "带下划线前缀的字段由脚本生成，每次运行都会被覆写，请不要修改它们。"
)


# ============================================================================
# 文本清洗
# ============================================================================
#
# PDF 专属的清洗（页眉页脚剥离、换行断词合并、连字还原）已经搬到 ai/rag/parsers.py，
# 由 PdfParser.clean() 负责 —— 那里才是"格式相关的处理"该待的地方。
# 放在这里会导致 parsers 反过来 import ingest，形成循环。
#
# 这里的 normalize_typography 只是一层转发，给还在用它的旧调用点留个名字。


# ============================================================================
# 标题：三层优先级
# ============================================================================

def _iter_supported_files(folder: str) -> list[str]:
    """列出目录下所有**注册表认识的**文件。

    以前这里是 `glob("*.pdf")`。改成查注册表之后，目录里放 Markdown / DOCX
    也能一起导入 —— 而「支持哪些格式」只有 `parsers.py` 一处定义，
    不会出现「上传接口收了 .docx 但批量导入只认 .pdf」这种不一致。
    """
    from ai.rag.parsers import supported_extensions

    found: list[str] = []
    for ext in supported_extensions():
        found.extend(glob.glob(os.path.join(folder, "*" + ext)))
    return found


def _pdf_title_metadata(path: str) -> dict:
    """读 PDF 的 `/Title`。非 PDF 返回空 dict（它们没有这个概念）。"""
    if os.path.splitext(path)[1].lower() != ".pdf":
        return {}
    try:
        from pypdf import PdfReader

        return PdfReader(path).metadata or {}
    except Exception as exc:
        logger.warning("读 PDF 元数据失败（%s），改用文件名推标题：%s",
                       type(exc).__name__, path)
        return {}


def _auto_title(path: str, metadata: dict | None = None) -> tuple[str, str]:
    """自动推导标题，返回 (标题, 来源)。来源只会是 metadata 或 filename。

    `metadata` 可以显式传入（批量导入时读一次就够），不传就按格式自己读。
    """
    if metadata is None:
        metadata = _pdf_title_metadata(path)

    raw = metadata.get("/Title")
    if raw is not None:
        title = html.unescape(str(raw)).strip()
        if title and title.lower() != "none":
            return title, "metadata"

    stem = os.path.splitext(os.path.basename(path))[0]
    # 文件名清洗：去掉下载产生的 (1) / (3) 后缀，下划线转空格
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return re.sub(r"_+", " ", stem).strip(), "filename"


def load_manifest(folder: str) -> dict:
    """读取 titles.json。文件不存在或读不动时返回空清单，不阻塞导入。"""
    path = os.path.join(folder, MANIFEST_NAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("papers", {}) if isinstance(data, dict) else {}
    except Exception as exc:
        print("  [警告] %s 读取失败（%s），本次忽略它" % (MANIFEST_NAME, exc))
        return {}


# 兼容旧版 manifest：早先的字段名是 auto / auto_source（没有下划线前缀）
def _legacy_auto(entry: dict) -> str:
    return str(entry.get("_auto", entry.get("auto", "")) or "").strip()


def _legacy_auto_source(entry: dict) -> str:
    return str(entry.get("_auto_source", entry.get("auto_source", "")) or "").strip()


# 清单里「人工可填」的元数据字段。title 之外的这些会写进 paper 表，
# 供 list_papers / get_paper 这类结构化查询使用。
MANUAL_FIELDS = ("title", "authors", "venue", "year", "external_id")


def save_manifest(folder: str, rows: list[dict]) -> str:
    """写回 titles.json：保留人工填写的字段，并把新出现的论文补进去。

    字段顺序刻意把人工字段放在最前面、生成字段加下划线前缀。早先的版本把 auto
    排在 title 前面且没有标注，结果被误改了 auto —— 而 auto 每次运行都会覆写，
    改错等于白改。这里如果检测到这种误改，会明确提示。
    """
    existing = load_manifest(folder)
    papers = {}
    for row in rows:
        name = row["file"]
        entry = existing.get(name, {}) or {}
        manual = {field: entry.get(field, "") for field in MANUAL_FIELDS}
        manual["title"] = str(manual.get("title") or "").strip()

        stored_auto = _legacy_auto(entry)
        if not manual["title"] and stored_auto and stored_auto != row["auto_title"]:
            print(
                "  [提示] %s 的 _auto 与脚本推导结果不一致 —— 可能是把标题填到了 _auto 字段。"
                "这个字段每次运行都会被覆写，请改填 title。" % name[:44]
            )

        papers[name] = {
            **manual,
            "year": manual.get("year") or "",
            "_auto": row["auto_title"],
            "_auto_source": row["auto_source"],
        }
    path = os.path.join(folder, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_说明": _MANIFEST_HELP, "papers": papers}, fh, ensure_ascii=False, indent=2)
    return path


# ============================================================================
# 导入
# ============================================================================

def build_splitter() -> RecursiveCharacterTextSplitter:
    """切块器。抽成函数是为了让「整库导入」和「上传单个文件」用**同一套**参数 ——
    两处各写一份的话，上传进来的块和批量导入的块粒度会悄悄不一致。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_document(path: str, source_file: str, title: str,
                   splitter: RecursiveCharacterTextSplitter | None = None
                   ) -> tuple[list[Document], dict]:
    """解析**单个**文件 → (切好的块, 报告行)。

    格式由 `ai/rag/parsers.py` 的注册表决定（PDF / Markdown / DOCX / 纯文本），
    这里只负责「切块 + 打 metadata」，不关心文件是什么格式。

    `source_file` 和 `title` 由调用方决定，不在这里从文件名推 ——
    上传的文件名是带哈希前缀的存储名，不能拿来当标题用。

    metadata 里的三个位置字段：
      · `page_label`     —— 位置标签（PDF 是页码，Markdown 是章节号，纯文本是块号）
      · `locator_prefix` —— 引用里的前缀（`p` / `sec` / `blk`），检索工具据此拼引用
      · `locator_kind`   —— page / section / block，让模型知道自己在说什么

    **为什么不统一叫 page**：Markdown 没有页。硬套的话引用会写「第 2 页」，
    而那个文件根本没有页 —— 引用是用户唯一会核对的东西，说错比不说更糟。
    """
    from ai.rag.parsers import parse_document

    splitter = splitter or build_splitter()
    sections, info = parse_document(path)

    chunks: list[Document] = []
    for section in sections:
        text = section.text
        if len(text.strip()) < MIN_SECTION_CHARS:
            continue
        piece = Document(
            page_content=text,
            metadata={
                "source": source_file,
                "paper_title": title,
                # `page` 保留成 0 基序号，只为兼容既有代码（Chroma 里的旧数据也有它）；
                # 真正表达位置的是下面两个。
                "page": int(section.label) - 1 if section.label.isdigit() else 0,
                "page_label": section.label,
                "locator_prefix": info.prefix,
                "locator_kind": info.kind,
            },
        )
        chunks.extend(splitter.split_documents([piece]))

    return chunks, {
        "file": source_file,
        "title": title,
        "pages": len(sections),
        "sections": len(sections),
        "chunks": len(chunks),
        "locator_kind": info.kind,
    }


def load_folder_chunks(folder: str, manifest: dict) -> tuple[list[Document], list[dict]]:
    """读取目录下所有**受支持格式**的文件，返回 (切好的块, 每篇的处理报告)。"""
    splitter = build_splitter()

    chunks: list[Document] = []
    report: list[dict] = []

    for path in sorted(_iter_supported_files(folder)):
        name = os.path.basename(path)
        auto_title, auto_source = _auto_title(path)

        entry = manifest.get(name, {}) or {}
        manual_title = str(entry.get("title", "") or "").strip()
        title = manual_title or auto_title
        title_source = "manual" if manual_title else auto_source

        file_chunks, row = chunk_document(path, name, title, splitter)
        chunks.extend(file_chunks)
        row["title_source"] = title_source
        row["auto_title"] = auto_title
        row["auto_source"] = auto_source
        row["meta"] = {
            "authors": str(entry.get("authors", "") or "").strip(),
            "venue": str(entry.get("venue", "") or "").strip(),
            "year": _parse_year(entry.get("year")),
            "external_id": str(entry.get("external_id", "") or "").strip(),
        }
        report.append(row)

    return chunks, report


def index_pdf(pdf_path: str, source_file: str, title: str) -> dict:
    """**阻塞**部分：解析 → 切块 → 写向量库 → 让 BM25 索引失效。

    ## 为什么不能复用 ingest()

    `ingest()` 的语义是**整库重建**：`reset=True` 时先把整个 Chroma collection 清空。
    上传接口绝不能那么干 —— 用户传第 5 篇，前 4 篇会被抹掉。
    所以这里走一条独立的增量路径：只解析这一个文件、只追加它的块。

    ## 为什么必须让 BM25 索引失效（这条最容易漏）

    BM25 索引按 Chroma 集合大小做签名缓存（见 ai/rag/hybrid.py 的 get_index()），
    签名没变就复用旧索引。新导入的文档如果不显式 `invalidate_index()`，
    **关键词那一路在进程重启前根本看不见它**；而向量那一路是直接查 Chroma 的，
    所以会命中 —— 于是现象变成「语义搜得到、关键词搜不到」，非常难查。

    ## 为什么刻意不碰关系库

    这个函数会在**线程池**里跑（见 record_paper 的说明），那个线程里没有事件循环，
    不能 `asyncio.run()` 去写库。关系库的写回由调用方在主循环上做。
    """
    from ai.rag.chromaClient import document_vector_store
    from ai.rag.hybrid import invalidate_index

    chunks, row = chunk_document(pdf_path, source_file=source_file, title=title)
    if not chunks:
        raise ValueError("没有解析出任何内容，检查一下是不是扫描件（没有文本层）")

    # ---- 先清掉同 source 的旧块，让这个操作**幂等** ----
    #
    # 不清的话，下面 add_documents 是**追加**：同一段内容在向量库里存两份，
    # 检索时互相竞争，而引用看起来完全正常（同一页、两个不同分数）——
    # 一次不会报错的静默退化。
    #
    # 什么时候会出现「记录没了但块还在」？实测踩过一次：整库重建的剪枝逻辑
    # 没有限定目录，把上传的文档记录删了、块留下了。那种情况下重新索引
    # 就会产生重复块。直接改数据库而没走 DELETE /documents 也是同一类。
    deleted = 0
    try:
        got = document_vector_store.get(where={"source": source_file})
        ids = got.get("ids") or []
        if ids:
            document_vector_store.delete(ids=ids)
            deleted = len(ids)
    except Exception as exc:
        logger.warning("清理旧块失败（%s），继续索引：%s", type(exc).__name__, source_file)

    batch = 64
    for start in range(0, len(chunks), batch):
        document_vector_store.add_documents(chunks[start:start + batch])
        time.sleep(0.2)

    invalidate_index()
    if deleted:
        row["replaced_chunks"] = deleted
    return row


def _decode_arxiv_date(file: str, title: str):
    """从文件名/标题解出 arXiv 编号信息：(year, month, "arXiv:原始编号") 或 None。"""
    from ai.tools.research_tools import _ARXIV_ID_RE, arxiv_submission_date

    for text in (file or "", title or ""):
        m = _ARXIV_ID_RE.search(text)
        if m:
            d = arxiv_submission_date(text)
            return d[0], d[1], "arXiv:" + m.group(0)
    return None


async def record_paper(
    row: dict,
    pdf_path: str,
    meta: dict | None = None,
    collection_name: str = "reid-papers",
    status: int | None = None,
    error: str = "",
) -> None:
    """把单个文件的元数据写进 paper 表。**必须在主事件循环上调用。**

    为什么不在线程里顺手写完：项目的 `async_engine` 是模块级全局，连接池里的连接
    是在主事件循环上建立的。在线程里另起一个 loop 去用它会抛
    「Future attached to a different loop」—— 所以线程只做解析和 embedding 这些
    重活，数据库写回留在主循环。

    实测教训（今天踩过两次同类问题）：**跨事件循环复用异步资源是静默的**
    —— 它不一定当场报错，可能只是随机地在某次请求上炸。
    """
    import db.database as database
    from db.models.paper import STATUS_INDEXED, Paper
    from db.repository.collection_repo import CollectionRepository
    from db.repository.paper_repo import PaperRepository

    meta = meta or {}
    async with database.async_session_maker() as session:
        collection = await CollectionRepository.get_or_create(
            session=session, name=collection_name, description="ResearchPilot 论文知识库"
        )
        paper = Paper(
            source_file=row["file"],
            title=row["title"],
            authors=str(meta.get("authors") or ""),
            venue=str(meta.get("venue") or ""),
            # meta 清单没有的，从 arXiv 编号解码兜底（上传件基本没有清单）
            year=_parse_year(meta.get("year"))
            or (lambda d: d[0] if d else None)(_decode_arxiv_date(row["file"], row["title"])),
            external_id=str(meta.get("external_id") or "")
            or (lambda d: d[2] if d else "")(_decode_arxiv_date(row["file"], row["title"])),
            pdf_path=os.path.relpath(pdf_path, os.getcwd()),
            collection_id=collection.id or 0,
            status=STATUS_INDEXED if status is None else status,
            chunk_count=row.get("chunks") or 0,
            error=error,
            indexed_at=datetime.now(),
        )
        await PaperRepository.upsert(session=session, paper=paper)


def _parse_year(value) -> int | None:
    """清单里的 year 可能被填成字符串或留空，这里统一成 int|None。"""
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip()[:4])
    except Exception:
        return None


def ingest(folder: str, reset: bool = True) -> tuple[int, list[dict], str]:
    """把目录下的 PDF 导入 papers collection。reset=True 时先清空。"""
    from ai.rag.chromaClient import document_vector_store

    if not os.path.isdir(folder):
        raise SystemExit("找不到论文目录: %s" % folder)

def ingest(folder: str, reset: bool = True, collection_name: str = "reid-papers") -> tuple[int, list[dict], str]:
    """把目录下的 PDF 导入 papers collection，并把论文元数据写进 paper 表。

    向量库和关系库各管一段：
      · Chroma 存切块文本 + 向量（语义检索用）
      · paper 表存论文属性（精确查询、metadata filter 用）
    两边靠 paper.source_file ↔ Chroma 块的 metadata["source"] 对应起来。
    """
    from ai.rag.chromaClient import document_vector_store

    if not os.path.isdir(folder):
        raise SystemExit("找不到论文目录: %s" % folder)

    manifest = load_manifest(folder)
    chunks, report = load_folder_chunks(folder, manifest)
    if not chunks:
        raise SystemExit("没有解析出任何内容，检查一下 PDF 是不是扫描件")

    if reset:
        existing = document_vector_store.get()
        if existing.get("ids"):
            document_vector_store.delete(ids=existing["ids"])

    # 分批写入，避免一次性提交上万个 embedding 请求
    batch = 64
    for start in range(0, len(chunks), batch):
        document_vector_store.add_documents(chunks[start:start + batch])
        print("  已写入 %d / %d" % (min(start + batch, len(chunks)), len(chunks)))
        time.sleep(0.2)

    manifest_path = save_manifest(folder, report)
    _sync_paper_table(report, folder, collection_name, prune=reset)
    return len(chunks), report, manifest_path


def _sync_paper_table(report: list[dict], folder: str, collection_name: str,
                      prune: bool = False) -> None:
    """把这一批论文的结构化元数据写进 paper 表（按 source_file upsert）。

    `prune=True`（整库重建时）会顺带删掉**这个知识库里已经不存在的**记录，
    让关系库和向量库保持一致。理由见 PaperRepository.delete_not_in 的说明 ——
    不删的话，从语料目录移走一篇论文之后，`list_papers` 还会把它列出来。
    """
    import asyncio

    from db.database import async_session_maker, create_db_and_tables
    from db.models.paper import STATUS_INDEXED, Paper
    from db.repository.collection_repo import CollectionRepository
    from db.repository.paper_repo import PaperRepository

    async def run() -> None:
        await create_db_and_tables()
        async with async_session_maker() as session:
            collection = await CollectionRepository.get_or_create(
                session=session, name=collection_name, description="ResearchPilot 论文知识库"
            )
            if prune:
                # path_prefix 是**必须的护栏**：不加的话会把上传目录里的文档
                # 也当成"已删除的论文"清掉（实测踩过，用户上传的 3 篇消失了）。
                # 详见 PaperRepository.delete_not_in 的说明。
                removed = await PaperRepository.delete_not_in(
                    session=session,
                    collection_id=collection.id or 0,
                    keep_source_files=[row["file"] for row in report],
                    path_prefix=folder,
                )
                for name in removed:
                    print("  已清理 paper 表中不再存在的记录: %s" % name)
            for row in report:
                meta = row.get("meta") or {}
                paper = Paper(
                    source_file=row["file"],
                    title=row["title"],
                    authors=meta.get("authors") or "",
                    venue=meta.get("venue") or "",
                    year=meta.get("year")
                    or (lambda d: d[0] if d else None)(_decode_arxiv_date(row["file"], row["title"])),
                    external_id=meta.get("external_id") or ""
                    or (lambda d: d[2] if d else "")(_decode_arxiv_date(row["file"], row["title"])),
                    pdf_path=os.path.relpath(os.path.join(folder, row["file"]), os.getcwd()),
                    collection_id=collection.id or 0,
                    status=STATUS_INDEXED,
                    chunk_count=row["chunks"],
                    indexed_at=datetime.now(),
                )
                await PaperRepository.upsert(session=session, paper=paper)
            print("  已写入 paper 表 %d 条（知识库：%s）" % (len(report), collection_name))

    asyncio.run(run())


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))

    # .env 查找和数据库相对路径都跟工作目录有关，统一切到 backend/
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))

    folder = "resource/papers"
    reset = "--no-reset" not in sys.argv
    for index, arg in enumerate(sys.argv):
        if arg == "--folder" and index + 1 < len(sys.argv):
            folder = sys.argv[index + 1]

    print("论文目录 : %s" % os.path.abspath(folder))
    print("切块参数 : chunk_size=%d, overlap=%d" % (CHUNK_SIZE, CHUNK_OVERLAP))
    print("清空重建 : %s" % reset)
    print("")

    total, report, manifest_path = ingest(folder, reset=reset)

    print("")
    print("%-50s %6s %6s %9s %8s" % ("标题", "片段数", "块数", "位置类型", "标题来源"))
    print("-" * 88)
    for row in report:
        print("%-50s %6d %6d %9s %8s" % (
            row["title"][:48], row["sections"], row["chunks"],
            row.get("locator_kind", "?"), row["title_source"]
        ))
    print("-" * 88)
    print("共 %d 篇，%d 块" % (len(report), total))
    print("")

    need_review = [row for row in report if row["title_source"] != "manual"]
    print("标题清单 : %s" % os.path.relpath(manifest_path, backend_root))
    if need_review:
        print("")
        print("[需要确认] 有 %d 篇的标题不是人工确认的，请打开上面的文件检查：" % len(need_review))
        for row in need_review:
            print("      [%s] %s" % (row["auto_source"], row["auto_title"][:66]))
        print("      填写 title 字段后重新运行本脚本即可生效。")


if __name__ == "__main__":
    # Windows 控制台默认是 GBK，遇到编码不下的字符（emoji 之类）会直接抛
    # UnicodeEncodeError 让脚本中断。这里降级成替换字符，保证导入本身不会因为
    # 打印一句话而失败。
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    main()
