"""改造 #5 上传接口的端到端测试。

    python tests/api/uploadApiTest.py

## 为什么是"造一个 PDF"而不是用现成的那 4 篇

用现成的会被 409 挡住（去重按内容哈希），而且就算改名绕过，也会往真实知识库里塞
重复内容、改变 411 个块这个基线。所以这里**现场生成**一个有文本层的 PDF，
跑完把它产生的 paper 行和向量块都删掉 —— 测试不该改变被测试系统的状态。

手写 PDF 的交叉引用表很难算对，但 pypdf 读的时候会重建它，
因此「手写 + pypdf 转存」就能得到一个完全合法的 PDF，不需要额外依赖。

## 为什么用 TestClient 而不是起一个服务

避免和用户正在跑的后端抢 8001 端口、抢 SQLite 写锁。TestClient 在进程内把应用
跑起来，验证的是同一份代码。

## 验收标准（对应设计文档第七节）

    415 扩展名 / 415 文件头 / 413 超限 / 202 受理 / 状态变 indexed
    → **检索得到**（向量那一路）
    → **BM25 也看得到**（验证 invalidate_index 生效）
    → 409 重复 / 列表可见

中间那条「检索得到」才是真正的验收点，前面全是手段。
"""

from __future__ import annotations

import io
import os
import sys
import time


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


BODY = """ResearchPilot upload test document.

This paragraph exists only to give the extraction pipeline something to chunk.
It mentions a distinctive term, xylophone-calibration, which appears nowhere in
the real corpus. That makes it possible to verify end to end that a newly
uploaded document is actually retrievable, rather than merely stored.

The second paragraph repeats a few technical words so that the BM25 keyword
index has something to match: calibration, xylophone, upload, pipeline.
"""


def make_pdf() -> bytes:
    from pypdf import PdfReader, PdfWriter

    text_ops = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
    for line in BODY.strip().splitlines():
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        text_ops.append("(%s) Tj T*" % escaped)
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")

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


def cleanup(paper_id: int | None, source_file: str | None) -> None:
    """把测试产生的记录和向量块删掉。"""
    import asyncio

    from ai.rag.chromaClient import document_vector_store
    from ai.rag.hybrid import invalidate_index
    from db.database import async_session_maker
    from db.repository.paper_repo import PaperRepository

    if source_file:
        got = document_vector_store.get(where={"source": source_file})
        ids = got.get("ids") or []
        if ids:
            document_vector_store.delete(ids=ids)
            print("   清理：删掉 %d 个向量块" % len(ids))
        invalidate_index()

    if paper_id:
        async def run() -> None:
            async with async_session_maker() as session:
                paper = await PaperRepository.get_by_id(session=session, paper_id=paper_id)
                if paper:
                    await session.delete(paper)
                    await session.commit()

        asyncio.run(run())
        print("   清理：删掉 paper 行 id=%s" % paper_id)


def set_status(paper_id: int, status: int) -> None:
    """直接改库里的状态，用来确定性地构造「处理中」这种中间态。

    靠真实上传去抢那个时间窗是不稳定的（索引可能几十毫秒就跑完了），
    测试会随机地通过或失败 —— 那比没有测试更糟。
    """
    import asyncio

    from db.database import async_session_maker
    from db.repository.paper_repo import PaperRepository

    async def run() -> None:
        async with async_session_maker() as session:
            await PaperRepository.update_progress(
                session=session, paper_id=paper_id, status=status
            )

    asyncio.run(run())


def report(failures: list[str]) -> int:
    print()
    if failures:
        print("FAILED (%d):" % len(failures))
        for item in failures:
            print("   - %s" % item)
        return 1
    print("OK 全部通过")
    return 0


def main() -> int:
    _bootstrap()

    from fastapi.testclient import TestClient

    import main
    from core.config import settings as cfg
    from db.models.paper import STATUS_INDEXED, STATUS_INDEXING

    failures: list[str] = []
    paper_id = None
    source_file = None
    baseline_chunks = 0

    with TestClient(main.app) as client:
        r = client.get("/health")
        print("[health] HTTP %s  %s" % (r.status_code, r.json()))
        if r.status_code != 200:
            failures.append("health 不是 200")
        # 上传前记下基线，删除之后要比对它 —— 这才是"删干净了"的判据
        baseline_chunks = r.json()["index"]["chunks"]

        r = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
        ok = r.status_code == 415
        print("[415 扩展名] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:60]))
        if not ok:
            failures.append("非 PDF 扩展名应为 415，实际 %s" % r.status_code)

        r = client.post("/documents", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
        print("[415 文件头] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:60]))
        if r.status_code != 415:
            failures.append("假 PDF 应为 415，实际 %s" % r.status_code)

        saved_cap = cfg.MAX_UPLOAD_MB
        cfg.MAX_UPLOAD_MB = 1          # 临时压到 1MB，免得真造一个 50MB 文件
        big = b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024)
        r = client.post("/documents", files={"file": ("big.pdf", big, "application/pdf")})
        print("[413 超限] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 413:
            failures.append("超限文件应为 413，实际 %s" % r.status_code)
        cfg.MAX_UPLOAD_MB = saved_cap

        pdf = make_pdf()
        print("[上传] PDF %d 字节" % len(pdf))
        r = client.post(
            "/documents",
            files={"file": ("upload-test.pdf", pdf, "application/pdf")},
            data={"title": "ResearchPilot Upload Test", "collection": "reid-papers"},
        )
        print("[202 受理] HTTP %s  %s" % (r.status_code, r.json()))
        if r.status_code != 202:
            failures.append("正常上传应为 202，实际 %s" % r.status_code)
            return report(failures)
        paper_id = r.json()["id"]

        deadline = time.time() + 120
        final = None
        while time.time() < deadline:
            state = client.get("/documents/%s" % paper_id).json()
            if state["status_text"] in ("indexed", "failed"):
                final = state
                break
            time.sleep(0.5)
        print("[状态] %s" % final)
        if not final or final["status_text"] != "indexed":
            failures.append("索引没有变成 indexed：%s" % final)
            cleanup(paper_id, None)
            return report(failures)

        source_file = final["source_file"]
        if final["chunk_count"] <= 0:
            failures.append("chunk_count 为 0")

        from ai.rag.hybrid import hybrid_search
        from ai.rag.pipeline import retrieve

        outcome = retrieve("xylophone calibration upload pipeline", allowed_sources=[source_file])
        hit = any("xylophone" in d.page_content.lower() for d, _o, _s in outcome.hits)
        print("[检索] 命中 %d 条，含关键词: %s（top1 向量分 %.4f）"
              % (len(outcome.hits), hit, outcome.vector_top1))
        if not hit:
            failures.append("上传的文档检索不到（向量那一路）")

        hits, _vt = hybrid_search("xylophone-calibration", allowed_sources=[source_file], top_n=5)
        bm25_ok = any("xylophone" in d.page_content.lower() for d, _o, _s in hits)
        print("[BM25] 命中 %d 条，含关键词: %s" % (len(hits), bm25_ok))
        if not bm25_ok:
            failures.append("BM25 索引没看到新文档（invalidate_index 可能没生效）")

        r = client.post("/documents", files={"file": ("upload-test.pdf", pdf, "application/pdf")})
        print("[409 重复] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 409:
            failures.append("重复上传应为 409，实际 %s" % r.status_code)

        r = client.get("/documents")
        ids = [item["id"] for item in r.json()["items"]]
        print("[列表] total=%s，含测试文档: %s" % (r.json()["total"], paper_id in ids))

        # ---- 10) 409：处理中的文档不允许删 ----
        # 直接改库把状态改成「处理中」，模拟"后台任务还在跑"。这样测是确定性的，
        # 靠真实上传去抢那个时间窗是不稳定的。
        set_status(paper_id, STATUS_INDEXING)
        r = client.delete("/documents/%s" % paper_id)
        print("[409 处理中] HTTP %s  %s" % (r.status_code, r.json().get("detail", "")[:70]))
        if r.status_code != 409:
            failures.append("处理中的文档应拒绝删除（409），实际 %s" % r.status_code)
        set_status(paper_id, STATUS_INDEXED)

        # ---- 11) 正常删除 → 204 ----
        r = client.delete("/documents/%s" % paper_id)
        print("[204 删除] HTTP %s  响应体长度 %d" % (r.status_code, len(r.content)))
        if r.status_code != 204:
            failures.append("删除应为 204，实际 %s" % r.status_code)
        if r.content:
            failures.append("204 不该有响应体")

        # ---- 12) 删完再查 → 404 ----
        r = client.get("/documents/%s" % paper_id)
        print("[404 已删] HTTP %s" % r.status_code)
        if r.status_code != 404:
            failures.append("已删除的文档应返回 404，实际 %s" % r.status_code)

        # ---- 13) 向量块真的没了（这条才是删除的实质）----
        # 记录没了不代表向量没了；如果只删了行，向量会变成永远清不掉的孤儿，
        # 而且它还会被检索到 —— 那是最糟的一种"删除成功"。
        health = client.get("/health").json()["index"]
        print("[删除后] papers=%s chunks=%s" % (health["papers"], health["chunks"]))
        if health["chunks"] != baseline_chunks:
            failures.append(
                "删除后向量块数没回到基线：%s != %s" % (health["chunks"], baseline_chunks)
            )

        # ---- 14) 再删一次 → 404 ----
        r = client.delete("/documents/%s" % paper_id)
        print("[404 重复删] HTTP %s" % r.status_code)
        if r.status_code != 404:
            failures.append("重复删除应为 404，实际 %s" % r.status_code)

        paper_id = None          # 已删干净，不需要 cleanup 再处理

    cleanup(paper_id, source_file)
    return report(failures)


if __name__ == "__main__":
    raise SystemExit(main())
