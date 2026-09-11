"""文档上传与知识库 API（改造 #5）。

设计见 `reference/transformation-05-document-api-design.md`，这里只写实现要点。

## 接口

    POST   /documents        上传 PDF → **202** + 文档记录（前端拿着 id 轮询）
    GET    /documents        列出知识库里的文档
    GET    /documents/{id}   查单个文档状态

## 为什么是 202 而不是 200

一篇 PDF 要解析 → 清洗 → 切块 → 逐块算 embedding 写向量库，实测 411 个块耗时以分钟计。
**不可能让一个 HTTP 请求干等** —— 网关和浏览器都会先超时。
所以受理（202）只表示"收下了、开始处理了"，不表示"做完了"。

## 为什么用 paper 记录本身当"任务"

对一篇论文而言，同一时刻最多只应该有一个索引入库操作在进行。
既然操作天然按文档唯一，`paper.status` 就是任务状态 —— 多建一张 task 表
只会多一处不一致的可能（task 说完成、paper 说失败）。这也让 409 的语义变得自然。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi import status as http

from ai.rag.ingest import index_pdf
from api.schema.documentSchema import (
    DocumentList,
    DocumentOut,
    UploadAccepted,
    status_text,
)
from core.config import settings
from db.database import async_session_maker
from db.models.paper import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_PENDING,
    Paper,
)
from db.repository.collection_repo import CollectionRepository
from db.repository.paper_repo import PaperRepository

logger = logging.getLogger(__name__)

document_router = APIRouter(prefix="/documents", tags=["documents"])

# 只收 PDF。为什么写死扩展名而不是 Content-Type：浏览器给出的 MIME 常常是
# application/octet-stream，甚至是错的，靠它判断会既拒绝合法文件又放过非法文件。
# 真正的判据是内容（PDF 头是 %PDF-），所以这里只看扩展名，下面再验文件头。
ALLOWED_EXTENSIONS = (".pdf",)
PDF_MAGIC = b"%PDF-"

# 读上传流的分片大小。用分片读而不是 file.read() 一把梭：
# UploadFile 背后是 SpooledTemporaryFile，**不读它就不占内存**；
# 一次性读会让我们在上限检查生效之前就把整个文件放进内存。
READ_CHUNK = 1024 * 1024


def _safe_name(name: str) -> str:
    """把用户给的文件名清洗成安全的磁盘名。

    用户提供的文件名不可信 —— 可能带路径分隔符（`../../etc/passwd`）、
    控制字符、超长串。文件名只用来给人看，**永远不直接参与路径拼接**。
    """
    base = os.path.basename(name or "").strip() or "upload.pdf"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base[:120] or "upload.pdf"


def _derive_title(filename: str, provided: str) -> str:
    """标题优先级：表单里填的 > 文件名清洗。

    和批量导入（titles.json > PDF /Title > 文件名）的顺序一致：
    人工确认过的优先，自动推导的兜底。
    """
    if provided.strip():
        return provided.strip()[:300]
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    stem = re.sub(r"[_]+", " ", stem).strip()
    return (stem or "untitled")[:300]


async def _read_with_cap(file: UploadFile, max_bytes: int) -> bytes:
    """边读边算大小，超限立刻中断。

    顺序很重要：**先检查价格便宜的（扩展名），再做贵的（读内容）**。
    而且不能读完再判大小 —— 那样一个 10 GB 的请求就能把我们打爆。
    """
    total = 0
    parts: list[bytes] = []
    while True:
        chunk = await file.read(READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=http.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="文件超过 %d MB 上限（已读 %d MB）" % (
                    max_bytes // (1024 * 1024), total // (1024 * 1024)),
            )
        parts.append(chunk)
    return b"".join(parts)


def _to_out(paper: Paper) -> DocumentOut:
    return DocumentOut(
        id=paper.id or 0,
        title=paper.title,
        source_file=paper.source_file,
        status=paper.status,
        status_text=status_text(paper.status),
        chunk_count=paper.chunk_count,
        error=paper.error or "",
        collection_id=paper.collection_id,
        indexed_at=paper.indexed_at,
    )


@document_router.post("", status_code=http.HTTP_202_ACCEPTED, response_model=UploadAccepted)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    collection: str = Form("reid-papers"),
) -> UploadAccepted:
    """上传一个 PDF 并开始后台索引。成功受理返回 **202**。"""
    # ---- 1) 便宜的检查：扩展名（415）----
    original = file.filename or ""
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=http.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="只支持 PDF（收到 %r）。其它格式的解析器还没接。" % (ext or original),
        )

    # ---- 2) 读内容，边读边算大小（413）----
    max_bytes = max(1, settings.MAX_UPLOAD_MB) * 1024 * 1024
    payload = await _read_with_cap(file, max_bytes)

    # ---- 3) 验文件头。扩展名叫 .pdf 不代表内容是 PDF ----
    if not payload.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=http.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="文件内容不是 PDF（缺少 %%PDF- 文件头）。改个扩展名是没用的。",
        )

    # ---- 4) 落盘。用内容哈希命名 ----
    # 用它而不是用户给的名字，有两个原因：
    #   · 用户的名字可能带路径分隔符，直接拼进路径就是目录穿越
    #   · 内容哈希天然去重，同一篇论文重复上传会落到同一个文件
    digest = hashlib.sha256(payload).hexdigest()
    source_file = "%s_%s" % (digest[:12], _safe_name(original))

    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, source_file)

    async with async_session_maker() as session:
        # ---- 5) 去重（409）----
        existing = await PaperRepository.get_by_source_file(session=session, source_file=source_file)
        if existing is not None:
            # 两种情况分开报：正在处理 vs 已经处理完。
            # 都用 409，但消息要能让人知道下一步该干什么。
            if existing.status in (STATUS_PENDING, STATUS_INDEXING):
                raise HTTPException(
                    status_code=http.HTTP_409_CONFLICT,
                    detail="这篇论文正在处理中（id=%s）。请查 GET /documents/%s 看进度。"
                           % (existing.id, existing.id),
                )
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail="这篇论文已经在知识库里了（id=%s, 状态 %s）。"
                       "需要重新索引请先删除该文档。"
                       % (existing.id, status_text(existing.status)),
            )

        with open(pdf_path, "wb") as fh:
            fh.write(payload)

        # ---- 6) 先建记录再排后台任务 ----
        # 顺序不能反：如果先排任务再建记录，后台可能跑完了却找不到要更新的行。
        # 先落记录（status=待处理），后台任务只负责推进状态。
        paper = Paper(
            source_file=source_file,
            title=_derive_title(original, title),
            pdf_path=os.path.relpath(pdf_path, os.getcwd()),
            status=STATUS_PENDING,
        )
        found = await CollectionRepository.get_or_create(
            session=session, name=collection, description="ResearchPilot 论文知识库"
        )
        paper.collection_id = found.id or 0
        paper = await PaperRepository.upsert(session=session, paper=paper)
        paper_id = paper.id or 0
        paper_title = paper.title

    # ---- 7) 排后台任务并立刻返回 202 ----
    #
    # 传进去的是**同步**函数：Starlette 会把同步的 background task 丢进线程池执行。
    # 这正是我们要的 —— 解析（CPU 密集）和 embedding（同步 HTTP）都是阻塞的，
    # 直接在事件循环上跑会让整个后端冻结几十秒，连别人正在用的 SSE 聊天都会停。
    #
    # 如果写成 async 函数、里面又不 to_thread，就会踩这个坑，而且现象是
    # 「上传本身很快，但期间所有接口都不响应」，很难联想到上传。
    background.add_task(
        _ingest_in_background, paper_id, pdf_path, source_file, paper_title
    )

    logger.info("upload: 受理 %s（id=%s, %d KB），开始后台索引",
                paper_title, paper_id, len(payload) // 1024)
    return UploadAccepted(
        id=paper_id,
        title=paper_title,
        status=STATUS_PENDING,
        status_text=status_text(STATUS_PENDING),
        poll="/documents/%s" % paper_id,
    )


async def _ingest_in_background(paper_id: int, pdf_path: str, source_file: str, title: str) -> None:
    """后台导入。

        标成「处理中」→ 解析 + 嵌入（重活，扔线程）→ 成功：标「已索引」+ 块数
                                                    → 失败：标「失败」+ 原因

    ## 为什么这个函数是 async 的，但重活仍然扔线程

    两件事必须**同时**成立，缺一个都会坏：

    1. **重活不能跑在事件循环上。** 解析（CPU 密集）和 embedding（同步 HTTP 调 Ollama）
       都是阻塞的。直接在事件循环上跑，会让整个后端冻结几十秒 ——
       现象是「上传很快返回 202，但期间所有接口都不响应」，包括别人正在用的 SSE 聊天。
       → 所以 `index_pdf` 走 `asyncio.to_thread`，落到线程池。

    2. **数据库写回也不能跑在别的线程里。** 项目的 `async_engine` / `async_session_maker`
       是**模块级全局**，连接池里的连接是在主事件循环上建立的。在线程里另起一个
       loop（`asyncio.run`）去复用它，会抛「Future attached to a different loop」——
       而且是**随机地**在某次请求上炸，不是每次都炸，极难查。
       → 所以 mark() 留在主循环上跑。

    结论：**线程只做纯计算和外部 IO，不碰数据库。** 这也是为什么
    `ai/rag/ingest.py` 里把逻辑拆成了 `index_pdf`（阻塞、不碰关系库）和
    `record_paper`（异步、碰关系库）两部分。

    （真正的生产做法是独立 worker 进程 + 队列，服务重启不丢任务、还能多副本。
    这里先用线程，因为它是单机下最小且正确的改动；分队列属于改造 #6/#9 的范畴。）
    """
    async def mark(status: int, error: str = "", chunk_count: int | None = None) -> None:
        async with async_session_maker() as session:
            await PaperRepository.update_progress(
                session=session, paper_id=paper_id, status=status,
                error=error, chunk_count=chunk_count,
            )

    await mark(STATUS_INDEXING)
    try:
        row = await asyncio.to_thread(index_pdf, pdf_path, source_file, title)
    except Exception as exc:
        reason = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        logger.exception("upload: 索引失败 id=%s", paper_id)
        await mark(STATUS_FAILED, error=reason)
        return
    await mark(STATUS_INDEXED, chunk_count=row.get("chunks") or 0)
    logger.info("upload: 索引完成 id=%s，%d 块", paper_id, row.get("chunks") or 0)


@document_router.get("", response_model=DocumentList)
async def list_documents(limit: int = 50) -> DocumentList:
    async with async_session_maker() as session:
        papers = await PaperRepository.list_papers(session=session, limit=limit)
        total = await PaperRepository.count(session=session)
    return DocumentList(total=total, items=[_to_out(p) for p in papers])


@document_router.get("/{paper_id}", response_model=DocumentOut)
async def get_document(paper_id: int) -> DocumentOut:
    async with async_session_maker() as session:
        paper = await PaperRepository.get_by_id(session=session, paper_id=paper_id)
    if paper is None:
        raise HTTPException(
            status_code=http.HTTP_404_NOT_FOUND, detail="没有 id=%s 的文档" % paper_id
        )
    return _to_out(paper)
