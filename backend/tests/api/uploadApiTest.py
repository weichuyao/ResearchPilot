"""改造 #5 的端到端测试：上传（多格式）/ 状态 / 检索 / 删除 / 重索引。

    python tests/api/uploadApiTest.py

## 为什么现场造文件而不是用现成的语料

用现成的会被 409 挡住（去重按内容哈希），而且就算改名绕过，也会往真实知识库里塞
重复内容、改变 411 个块这个基线。所以这里**现场生成**四种格式的测试文件，
跑完把它们产生的 paper 行、向量块、磁盘文件都删掉 —— 测试不该改变被测试系统的状态。

## 为什么用 TestClient 而不是起一个服务

避免和用户正在跑的后端抢 8001 端口、抢 SQLite 写锁。TestClient 在进程内把应用
跑起来，验证的是同一份代码。

## 验收标准（对应设计文档第七节）

    **上传的文档真的能被检索到** ← 这一条才是目的，其余全是手段
    **位置标签跟着格式走**（PDF→p. / Markdown→sec. / 纯文本→blk.）
    **重索引不会留下重复块**（旧块必须先清掉）
    415 扩展名与文件头 / 413 超限 / 202 受理 / 409 重复与处理中 / 204 删除 / 404
"""

from __future__ import annotations

import io
import os
import sys
import time

# 每种格式一个独特的关键词，用来验证"上传的东西真的能被检索到"。
# 这些词在整个语料里不存在，所以命中必然是本次测试带进去的。
MARKER = "xylophone-calibration"


def _bootstrap() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(here))
    os.chdir(backend_root)
    sys.path.insert(0, os.path.join(backend_root, "app"))
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    return backend_root


# ---------------------------------------------------------------------------
# 造四种格式的测试文件
# ---------------------------------------------------------------------------
def make_pdf() -> bytes:
    """手写 + pypdf 转存。

    手写 PDF 的交叉引用表很难算对，但 pypdf 读的时候会重建它，
    所以「手写 + 转存」就能得到一个完全合法的 PDF，不需要额外依赖。
    """
    from pypdf import PdfReader, PdfWriter

    body = """ResearchPilot upload test document.

This paragraph exists only to give the extraction pipeline something to chunk.
It mentions a distinctive term, %s, which appears nowhere in the real corpus.

A second paragraph mentions the same term again so the keyword index has
something to match: %s.""" % (MARKER, MARKER)

    ops = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
    for line in body.strip().splitlines():
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append("(%s) Tj T*" % escaped)
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    for index, payload in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % index + payload + b"\nendobj\n"
    out += b"xref\n0 6\ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

    reader = PdfReader(io.BytesIO(bytes(out)))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_markdown() -> bytes:
    return ("""# ResearchPilot 上传测试（Markdown）

这是第一节的内容，用来验证按标题切段是对的。里面出现了一个独特术语
%s，它在真实语料里不存在。

## 第二节：关于检索

这一节再次提到 %s，这样关键词那一路也有东西可以匹配。
Markdown 的定位标签应该是 sec.N（章节序号），而不是页码 ——
这个文件根本没有页。
""" % (MARKER, MARKER)).encode("utf-8")


def make_text() -> bytes:
    return ("""ResearchPilot upload test (plain text).

    First block. The distinctive term is %s and it appears again below so that
    the keyword index has something to match.

    Second block. %s.
""" % (MARKER, MARKER)).encode("utf-8")


def make_docx() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("ResearchPilot upload test (DOCX)", level=1)
    document.add_paragraph(
        "第一节内容。独特术语 %s 在这里出现，用来验证 DOCX 也能被解析和检索。" % MARKER
    )
    document.add_heading("第二节：表格也要抽出来", level=2)
    document.add_paragraph("技术文档里的事实常常只写在表格里，所以表格不能跳过。")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "term"
    table.cell(0, 1).text = "value"
    table.cell(1, 0).text = MARKER
    table.cell(1, 1).text = "present"

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_empty_pdf() -> bytes:
    """一个合法但**没有文本层**的 PDF —— 用来构造"索引失败"的文档。

    有文本层的话它会正常索引，测不出失败路径。空页面 → 解析出 0 个片段 →
    index_pdf 抛「没有解析出任何内容」→ 记录变成 failed。
    """
    from pypdf import PdfReader, PdfWriter

    stream = b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    for index, payload in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % index + payload + b"\nendobj\n"
    out += b"xref\n0 5\ntrailer << /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

    reader = PdfReader(io.BytesIO(bytes(out)))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


CASES = [
    ("pdf", "upload-test.pdf", make_pdf, "p", "application/pdf"),
    ("markdown", "upload-test.md", make_markdown, "sec", "text/markdown"),
    ("text", "upload-test.txt", make_text, "blk", "text/plain"),
    ("docx", "upload-test.docx", make_docx, "sec",
     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
]


# ---------------------------------------------------------------------------
def _run_isolated(coro_factory) -> None:
    """在**自己的事件循环 + 自己的 engine** 上跑一小段数据库操作。

    ## 为什么不能直接用 `db.database.async_session_maker`

    全局 engine 的连接池是绑在**第一个使用它的那个事件循环**上的。
    这个测试用 TestClient 在进程内跑应用（它有自己的事件循环），
    而 cleanup / set_status 是在 `with TestClient(...)` 之外调用的 ——
    那时那个循环已经关了。

    在 SQLite（aiosqlite）上这么做**一直没出问题**，换成 PostgreSQL（asyncpg）之后立刻炸：

        RuntimeError: Event loop is closed
        AttributeError: 'NoneType' object has no attribute 'send'     ← asyncpg 去写一个已死的 transport

    所以带外操作必须**自建 engine**，用完 dispose。
    这不只是测试的问题 —— 任何"应用之外再去访问数据库"的脚本
    （恢复脚本、维护脚本）都该守这条规矩。
    """
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from core.config import settings

    async def run() -> None:
        engine = create_async_engine(settings.DATABASE_URL)
        try:
            maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
            async with maker() as session:
                await coro_factory(session)
        finally:
            await engine.dispose()

    asyncio.run(run())


def set_status(paper_id: int, status: int) -> None:
    """直接改库里的状态，用来确定性地构造「处理中」这种中间态。

    靠真实上传去抢那个时间窗是不稳定的（索引可能几十毫秒就跑完了），
    测试会随机地通过或失败 —— 那比没有测试更糟。
    """
    from db.repository.paper_repo import PaperRepository

    async def work(session) -> None:
        await PaperRepository.update_progress(
            session=session, paper_id=paper_id, status=status
        )

    _run_isolated(work)


def wait_for(client, paper_id: int, timeout: float = 180.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/documents/%s" % paper_id).json()
        if state.get("status_text") in ("indexed", "failed"):
            return state
        time.sleep(0.4)
    return None


TRACKED: list[tuple[int, str]] = []


def cleanup(client=None) -> None:
    """把测试产生的一切都清掉：paper 行、向量块、磁盘文件。

    ## 两条规矩

    **① 记录走 API 删，不直接改库。**
    那正是用户会用的接口（`DELETE /documents/{id}`），而且它自己负责连带删掉向量块
    和磁盘文件 —— 测试和真实路径用同一套逻辑，清理就不可能"看起来清干净了其实没有"。

    **② 不带 client 时才退回直接改库，而且必须用独立 engine。**
    见 `_run_isolated` 的说明：TestClient 的事件循环和应用全局 engine 的连接池不是同一个，
    跨循环在 asyncpg 上会炸。
    """
    from core.config import settings

    if client is not None:
        deleted = 0
        for paper_id, source_file in TRACKED:
            try:
                r = client.delete("/documents/%s" % paper_id)
                if r.status_code in (204, 404):
                    deleted += 1
            except Exception as exc:
                print("   清理失败 id=%s: %s" % (paper_id, exc))
        files = 0
        for _, source_file in TRACKED:
            path = os.path.join(settings.UPLOAD_DIR, source_file)
            if os.path.isfile(path):
                # DELETE 只删上传目录里的文件；测试用的文件都在那儿，所以正常情况
                # 这一步不该有残留。有残留说明删除接口没清磁盘，值得看见。
                os.remove(path)
                files += 1
        if TRACKED:
            print("   清理：%d 条记录（走 DELETE 接口）/ %d 个残留文件" % (deleted, files))
        TRACKED.clear()
        return

    # ---- 兜底路径：没有 client，直接改库 + 直接删向量 ----
    from ai.rag.chromaClient import document_vector_store
    from ai.rag.hybrid import invalidate_index
    from db.repository.paper_repo import PaperRepository

    chunks = 0
    for paper_id, source_file in TRACKED:
        try:
            got = document_vector_store.get(where={"source": source_file})
            ids = got.get("ids") or []
            if ids:
                document_vector_store.delete(ids=ids)
                chunks += len(ids)
        except Exception as exc:
            print("   清理向量失败 %s: %s" % (source_file, exc))

    if chunks:
        invalidate_index()

    async def work(session) -> None:
        for paper_id, _ in TRACKED:
            paper = await PaperRepository.get_by_id(session=session, paper_id=paper_id)
            if paper is not None:
                await session.delete(paper)
                await session.commit()

    _run_isolated(work)

    files = 0
    for _, source_file in TRACKED:
        path = os.path.join(settings.UPLOAD_DIR, source_file)
        if os.path.isfile(path):
            os.remove(path)
            files += 1

    if TRACKED:
        print("   清理（兜底）：%d 条记录 / %d 个向量块 / %d 个文件"
              % (len(TRACKED), chunks, files))
    TRACKED.clear()


def report(failures: list[str]) -> int:
    print()
    if failures:
        print("FAILED (%d):" % len(failures))
        for item in failures:
            print("   - %s" % item)
        return 1
    print("OK 全部通过")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    _bootstrap()

    from fastapi.testclient import TestClient

    import main
    from core.config import settings as cfg
    from db.models.paper import STATUS_INDEXED, STATUS_INDEXING

    failures: list[str] = []

    with TestClient(main.app) as client:
        health = client.get("/health").json()
        print("[health] %s" % health)
        baseline_chunks = health["index"]["chunks"]
        baseline_papers = health["index"]["papers"]

        # ---------- 校验分支 ----------
        r = client.post("/documents", files={"file": ("notes.exe", b"xx", "application/octet-stream")})
        print("[415 扩展名] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 415:
            failures.append("不支持的扩展名应为 415，实际 %s" % r.status_code)

        r = client.post("/documents", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
        print("[415 文件头] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 415:
            failures.append("假 PDF 应为 415，实际 %s" % r.status_code)

        r = client.post("/documents", files={"file": ("fake.md", b"\x00\x01\x02binary", "text/markdown")})
        print("[415 伪装文本] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 415:
            failures.append("含 NUL 的伪文本应为 415，实际 %s" % r.status_code)

        saved_cap = cfg.MAX_UPLOAD_MB
        cfg.MAX_UPLOAD_MB = 1
        r = client.post("/documents",
                        files={"file": ("big.pdf", b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024), "application/pdf")})
        print("[413 超限] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 413:
            failures.append("超限文件应为 413，实际 %s" % r.status_code)
        cfg.MAX_UPLOAD_MB = saved_cap

        # ---------- 四种格式：上传 → 索引 → 检索 ----------
        from ai.rag.pipeline import retrieve

        pdf_bytes = None
        pdf_paper_id = None
        for label, filename, maker, want_prefix, mime in CASES:
            payload = maker()
            if label == "pdf":
                pdf_bytes = payload
            r = client.post("/documents",
                            files={"file": (filename, payload, mime)},
                            data={"title": "Upload Test (%s)" % label, "collection": "reid-papers"})
            if r.status_code != 202:
                failures.append("%s 上传应为 202，实际 %s %s" % (label, r.status_code, r.text[:120]))
                continue
            paper_id = r.json()["id"]
            state = wait_for(client, paper_id)
            if not state or state["status_text"] != "indexed":
                failures.append("%s 索引失败：%s" % (label, state))
                TRACKED.append((paper_id, state["source_file"] if state else ""))
                continue
            TRACKED.append((paper_id, state["source_file"]))
            if label == "pdf":
                pdf_paper_id = paper_id

            # 检索：不只验证"存进去了"，而是验证"能查出来"
            outcome = retrieve(MARKER, allowed_sources=[state["source_file"]])
            hits_text = "\n".join(d.page_content for d, _o, _s in outcome.hits)
            found = MARKER in hits_text
            prefixes = {d.metadata.get("locator_prefix") for d, _o, _s in outcome.hits}
            print("[%s] id=%s 块数=%s 检索命中=%s 位置前缀=%s"
                  % (label, paper_id, state["chunk_count"], found, prefixes or "—"))
            if not found:
                failures.append("%s 上传后检索不到" % label)
            if prefixes and prefixes != {want_prefix}:
                failures.append("%s 的位置前缀应为 %s，实际 %s" % (label, want_prefix, prefixes))

            # BM25 那一路也要看得见（验证 invalidate_index 生效）
            from ai.rag.hybrid import hybrid_search

            bm, _vt = hybrid_search(MARKER, allowed_sources=[state["source_file"]], top_n=5)
            if not any(MARKER in d.page_content for d, _o, _s in bm):
                failures.append("%s 的 BM25 索引没看到新文档" % label)

        # ---------- 长文件名（回归：截断不能把扩展名切掉）----------
        #
        # 实测踩过：_safe_name 第一版写的是 base[:120]，把**整个文件名**截断，
        # 于是长文件名的 .pdf 被切掉，下游认不出格式，1.4 MB 的论文直接上传失败。
        # 上面那些用例的文件名都很短，所以没覆盖到 —— 这条就是补这个洞。
        long_name = ("MVReID-Former_Multi-View_Vision_Transformer_for_Cross-Camera_"
                     "Person_and_Vehicle_Re-Identification_in_Surveillance_Networks_"
                     "Journal_Version_Final_Accepted_Manuscript_2024.pdf")
        print("[长文件名] %d 字符" % len(long_name))
        r = client.post("/documents",
                        files={"file": (long_name, make_pdf(), "application/pdf")},
                        data={"title": "长文件名回归测试"})
        if r.status_code != 202:
            failures.append("长文件名上传应为 202，实际 %s：%s" % (r.status_code, r.text[:160]))
        else:
            long_id = r.json()["id"]
            state = wait_for(client, long_id)
            if not state or state["status_text"] != "indexed":
                failures.append("长文件名文档索引失败：%s" % state)
            else:
                sf = state["source_file"]
                print("[长文件名] source_file 长度 %d，结尾 %r" % (len(sf), sf[-8:]))
                if not sf.endswith(".pdf"):
                    failures.append("长文件名的扩展名被截掉了：%r" % sf)
                TRACKED.append((long_id, sf))

        # ---------- 失败的文档可以重新上传（重试语义，不该被 409 挡住）----------
        #
        # 一个索引失败的文档并没有真的进知识库（chunk_count=0、检索不到）。
        # 把它当"已存在"会让用户陷入死循环：想重试却被 409 挡住，只能先删再传。
        empty = make_empty_pdf()
        r = client.post("/documents",
                        files={"file": ("no-text-layer.pdf", empty, "application/pdf")},
                        data={"title": "无文本层测试"})
        if r.status_code != 202:
            failures.append("无文本层 PDF 上传应为 202，实际 %s" % r.status_code)
        else:
            failed_id = r.json()["id"]
            state = wait_for(client, failed_id)
            print("[失败路径] 状态=%s error=%s"
                  % (state and state["status_text"], (state or {}).get("error", "")[:60]))
            if not state or state["status_text"] != "failed":
                failures.append("无文本层的 PDF 应该索引失败，实际 %s" % state)
            else:
                if not state.get("error"):
                    failures.append("失败记录里没有 error —— 失败必须是看得见的")
                TRACKED.append((failed_id, state["source_file"]))

                r = client.post("/documents",
                                files={"file": ("no-text-layer.pdf", empty, "application/pdf")})
                print("[失败后重传] HTTP %s %s"
                      % (r.status_code, r.json().get("detail", "（受理了）")[:50]))
                if r.status_code != 202:
                    failures.append("上传一个索引失败过的文档应该被受理（重试），实际 %s：%s"
                                    % (r.status_code, r.text[:160]))

        # ---------- 409 重复上传 ----------
        if pdf_bytes:
            r = client.post("/documents", files={"file": ("upload-test.pdf", pdf_bytes, "application/pdf")})
            print("[409 重复] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
            if r.status_code != 409:
                failures.append("重复上传应为 409，实际 %s" % r.status_code)

        # ---------- 重索引 ----------
        if pdf_paper_id:
            before = client.get("/health").json()["index"]["chunks"]

            # 处理中不允许重索引
            set_status(pdf_paper_id, STATUS_INDEXING)
            r = client.post("/documents/%s/reindex" % pdf_paper_id)
            print("[409 重索引中] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:60]))
            if r.status_code != 409:
                failures.append("处理中的文档应拒绝重索引，实际 %s" % r.status_code)
            set_status(pdf_paper_id, STATUS_INDEXED)

            # 正常重索引
            r = client.post("/documents/%s/reindex" % pdf_paper_id)
            print("[202 重索引] HTTP %s  %s" % (r.status_code, r.json().get("status_text")))
            if r.status_code != 202:
                failures.append("重索引应为 202，实际 %s" % r.status_code)
            else:
                state = wait_for(client, pdf_paper_id)
                after = client.get("/health").json()["index"]["chunks"]
                print("[重索引后] 状态=%s 块数=%s  总块数 %s -> %s"
                      % (state and state["status_text"], state and state["chunk_count"],
                         before, after))
                if not state or state["status_text"] != "indexed":
                    failures.append("重索引后状态不是 indexed：%s" % state)
                # 这条是重索引最容易错的地方：旧块没清掉的话，同一段内容会有两份，
                # 检索时互相竞争，而引用看起来完全正常 —— 一次看不见的退化。
                if after != before:
                    failures.append("重索引后总块数变了（%s -> %s），可能留下了重复块"
                                    % (before, after))

            # 源文件不在磁盘上时不允许重索引
            state = client.get("/documents/%s" % pdf_paper_id).json()
            disk = os.path.join(cfg.UPLOAD_DIR, state["source_file"])
            os.remove(disk)
            r = client.post("/documents/%s/reindex" % pdf_paper_id)
            print("[409 源文件缺失] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:60]))
            if r.status_code != 409:
                failures.append("源文件缺失时应拒绝重索引，实际 %s" % r.status_code)

        # ---------- 删除 ----------
        if pdf_paper_id:
            r = client.delete("/documents/%s" % pdf_paper_id)
            print("[204 删除] HTTP %s  响应体 %d 字节" % (r.status_code, len(r.content)))
            if r.status_code != 204:
                failures.append("删除应为 204，实际 %s" % r.status_code)
            if r.content:
                failures.append("204 不该有响应体")
            r = client.get("/documents/%s" % pdf_paper_id)
            print("[404 已删] HTTP %s" % r.status_code)
            if r.status_code != 404:
                failures.append("已删除的文档应返回 404，实际 %s" % r.status_code)
            TRACKED[:] = [t for t in TRACKED if t[0] != pdf_paper_id]

        # ---------- 收尾：列表 + 清理 + 基线 ----------
        #
        # ⚠️ 清理和基线核对都放在**同一个 TestClient 块里**。
        #
        # 原因：每次 `with TestClient(...)` 都会起一个新的事件循环，而应用那个全局
        # engine 的连接池绑在**第一个**循环上。跨块再去用它，asyncpg 会炸：
        #
        #     RuntimeError: Event loop is closed
        #     AttributeError: 'NoneType' object has no attribute 'send'
        #
        # SQLite（aiosqlite）对这种用法很宽容，换 Postgres 才暴露 —— 见 _run_isolated。
        # 而且清理走 API 本身就更合理：**用用户会用的那个接口去清理**。
        listing = client.get("/documents").json()
        print("[列表] total=%s" % listing["total"])
        if listing["total"] < baseline_papers:
            failures.append("文档列表丢东西了")

        cleanup(client)

        # 清理之后再确认一次基线 —— 只删记录不删向量的"删除成功"是最糟的一种
        index = client.get("/health").json()["index"]
        print("[清理后] papers=%s chunks=%s（基线 %s / %s）"
              % (index["papers"], index["chunks"], baseline_papers, baseline_chunks))
        if index["papers"] != baseline_papers or index["chunks"] != baseline_chunks:
            failures.append("清理后没回到基线：%s/%s != %s/%s"
                            % (index["papers"], index["chunks"], baseline_papers, baseline_chunks))

    return report(failures)


if __name__ == "__main__":
    raise SystemExit(main())
