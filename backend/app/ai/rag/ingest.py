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
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTContainer, LTFigure, LTTextLine
from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ---- 切块参数 ----------------------------------------------------------------
# 实测：这批论文共 50 页 / 约 26.8 万字符，每页中位约 5400 字符。
# 学术论文一个标准段落大约 400~800 字符，800 大致能装下一个完整论述；
# 重叠 150 是为了避免把一个论点从中间切断。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 标题清单的文件名（相对于论文目录）
MANIFEST_NAME = "titles.json"

_MANIFEST_HELP = (
    "只需要填写每篇论文的 title 字段（留空则使用脚本自动推导的结果）。"
    "带下划线前缀的字段由脚本生成，每次运行都会被覆写，请不要修改它们。"
)


# ============================================================================
# 文本清洗
# ============================================================================

def _norm(line: str) -> str:
    """归一化一行，用于判断它是否在每页重复。数字会被抹掉，因为页码会变。"""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line)).strip().lower()


def find_repeated_edge_lines(pages: list[str]) -> set[str]:
    """找出在超过一半页面上重复出现的首行/末行 —— 也就是页眉和页脚。

    用「跨页重复」来识别，而不是硬编码某本刊物的页眉格式，这样换一批论文也能用。
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


def normalize_typography(text: str) -> str:
    """还原 PDF 里的连字和特殊破折号。实现与理由见 ai/rag/textnorm.py。

    这里只做一层转发，是为了让 ingest 能在**导入时**就用上同一份归一化 ——
    归一化的三个使用方（导入、评估、名次基准）必须完全一致，所以实现在
    textnorm.py 里只写一份。延迟导入是因为 app/ 要等 main() 才进 sys.path。
    """
    from ai.rag.textnorm import normalize_typography as impl

    return impl(text)


def clean_page_text(text: str, noise: set[str]) -> str:
    """去掉页眉页脚、还原连字、并修复被换行拆开的单词。"""
    kept = [line for line in text.split("\n") if not (line.strip() and _norm(line) in noise)]
    joined = "\n".join(kept)
    # represen-\ntation → representation
    # 注意：这也会把行尾的真实连字符合并（well-\nknown → wellknown），属于已知取舍。
    # 换行拆词在两端对齐的论文里非常常见，而真实连字符恰好落在行尾的情况少得多。
    return normalize_typography(re.sub(r"(\w)-\n(\w)", r"\1\2", joined))


# ============================================================================
# 标题：三层优先级
# ============================================================================

def _auto_title(path: str, metadata: dict) -> tuple[str, str]:
    """自动推导标题，返回 (标题, 来源)。来源只会是 metadata 或 filename。"""
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

def _collect_text_lines(obj, out: list[str]) -> None:
    """递归收集 LTTextLine 的文本，遇到 LTFigure 整棵子树跳过。"""
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


def extract_pages_text(path: str, fallback: PdfReader | None = None) -> list[str]:
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

    注意 pypdf 仍然保留：`/Title` 元数据的读取比 pdfminer 方便，且抽取失败时要靠它兜底。

    抽取失败时退回 pypdf（结果差，但总比整篇丢掉好）—— 和 rerank 的降级是同一个思路。
    """
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


def load_pdf_chunks(folder: str, manifest: dict) -> tuple[list[Document], list[dict]]:
    """读取目录下所有 PDF，返回 (切好的块, 每篇的处理报告)。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[Document] = []
    report: list[dict] = []

    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        name = os.path.basename(path)
        reader = PdfReader(path)
        auto_title, auto_source = _auto_title(path, reader.metadata or {})

        entry = manifest.get(name, {}) or {}
        manual_title = str(entry.get("title", "") or "").strip()
        title = manual_title or auto_title
        title_source = "manual" if manual_title else auto_source

        raw_pages = extract_pages_text(path, fallback=reader)
        noise = find_repeated_edge_lines(raw_pages)

        per_file = 0
        for page_index, raw in enumerate(raw_pages):
            cleaned = clean_page_text(raw, noise)
            if len(cleaned.strip()) < 50:
                continue
            page_doc = Document(
                page_content=cleaned,
                metadata={
                    "source": name,
                    "paper_title": title,
                    "page": page_index,
                    "page_label": str(page_index + 1),
                },
            )
            split = splitter.split_documents([page_doc])
            per_file += len(split)
            chunks.extend(split)

        report.append(
            {
                "file": name,
                "title": title,
                "title_source": title_source,
                "auto_title": auto_title,
                "auto_source": auto_source,
                "pages": len(raw_pages),
                "chunks": per_file,
                "noise_lines": len(noise),
                # 人工填写的元数据（可能为空），写入 paper 表时使用
                "meta": {
                    "authors": str(entry.get("authors", "") or "").strip(),
                    "venue": str(entry.get("venue", "") or "").strip(),
                    "year": _parse_year(entry.get("year")),
                    "external_id": str(entry.get("external_id", "") or "").strip(),
                },
            }
        )

    return chunks, report


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
    chunks, report = load_pdf_chunks(folder, manifest)
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
    _sync_paper_table(report, folder, collection_name)
    return len(chunks), report, manifest_path


def _sync_paper_table(report: list[dict], folder: str, collection_name: str) -> None:
    """把这一批论文的结构化元数据写进 paper 表（按 source_file upsert）。"""
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
            for row in report:
                meta = row.get("meta") or {}
                paper = Paper(
                    source_file=row["file"],
                    title=row["title"],
                    authors=meta.get("authors") or "",
                    venue=meta.get("venue") or "",
                    year=meta.get("year"),
                    external_id=meta.get("external_id") or "",
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
    print("%-50s %5s %6s %7s %8s" % ("标题", "页数", "块数", "噪声行", "标题来源"))
    print("-" * 82)
    for row in report:
        print("%-50s %5d %6d %7d %8s" % (
            row["title"][:48], row["pages"], row["chunks"], row["noise_lines"], row["title_source"]
        ))
    print("-" * 82)
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
