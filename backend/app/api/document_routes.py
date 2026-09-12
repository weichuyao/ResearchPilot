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
from ai.rag.parsers import describe_formats, parser_for, supported_extensions
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

# 支持哪些格式**不在接口层定义**，而是问解析器注册表（ai/rag/parsers.py）。
#
# 为什么：格式清单只能有一处。写两份的话，必然出现「上传接口收了 .docx 但
# 导入管道不认」或者反过来 —— 而那种不一致的表现是"文件传上去了、状态一直失败"，
# 排查起来要跨两个文件。
#
# 校验分两步，都从注册表取：
#   扩展名 → 决定用哪个解析器（不认识就 415）
#   文件头 → 确认内容真的是那个格式（参数化的魔数）
# 为什么不看 Content-Type：浏览器给出的 MIME 常常是 application/octet-stream，
# 甚至干脆是错的，靠它判断会既拒绝合法文件又放过非法文件。

# 读上传流的分片大小。用分片读而不是 file.read() 一把梭：
# UploadFile 背后是 SpooledTemporaryFile，**不读它就不占内存**；
# 一次性读会让我们在上限检查生效之前就把整个文件放进内存。
READ_CHUNK = 1024 * 1024


def _safe_name(name: str, max_length: int = 120) -> str:
    """把用户给的文件名清洗成安全的磁盘名。

    用户提供的文件名不可信 —— 可能带路径分隔符（`../../etc/passwd`）、
    控制字符、超长串。文件名只用来给人看，**永远不直接参与路径拼接**。

    ## ⚠️ 截断必须保留扩展名

    第一版写的是 `base[:120]`，把**整个文件名**截断 —— 于是长文件名的扩展名被切掉：

        MVReID-Former_..._Surveillance_Networ     ← 没有 .pdf

    下游 `parser_for()` 靠扩展名选解析器，认不出来就报「不支持的格式」，
    用户看到的是一句莫名其妙的错误。实测踩过：一个 133 字符的 source_file
    结尾没有 .pdf，1.4 MB 的论文直接上传失败。

    正确做法是**截词干、留扩展名**：`stem[:120-len(ext)] + ext`。
    """
    base = os.path.basename(name or "").strip() or "upload.pdf"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    stem, ext = os.path.splitext(base)
    if not stem:
        return "upload.pdf"
    return stem[: max(1, max_length - len(ext))] + ext


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
    parser = parser_for(original)
    if parser is None:
        raise HTTPException(
            status_code=http.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="只支持 %s（收到 %r）。" % (describe_formats(), ext or original),
        )

    # ---- 2) 读内容，边读边算大小（413）----
    max_bytes = max(1, settings.MAX_UPLOAD_MB) * 1024 * 1024
    payload = await _read_with_cap(file, max_bytes)

    # ---- 3) 验文件头。扩展名叫 .pdf 不代表内容是 PDF ----
    magic = parser.info.magic
    if magic and not payload.startswith(magic):
        raise HTTPException(
            status_code=http.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="文件内容不是 %s（缺少%s）。改个扩展名是没用的。"
                   % (ext, parser.info.magic_note),
        )
    if magic is None and b"\x00" in payload[:4096]:
        # 没有魔数的格式（Markdown / 纯文本）做一次粗筛：
        # 前 4KB 里出现 NUL 字节，基本可以断定不是文本文件。
        # 这拦不住所有伪装，但能把「把 .exe 改名成 .md」这种挡在外面 ——
        # 否则它会被当作乱码文本索引进去，污染检索。
        raise HTTPException(
            status_code=http.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="%s 是文本格式，但内容里含二进制数据 —— 大概率只是改了扩展名。" % ext,
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
        if existing is not None and existing.status in (STATUS_PENDING, STATUS_INDEXING):
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail="这篇论文正在处理中（id=%s）。请查 GET /documents/%s 看进度。"
                       % (existing.id, existing.id),
            )
        if existing is not None and existing.status != STATUS_FAILED:
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail="这篇论文已经在知识库里了（id=%s, 状态 %s）。"
                       "需要重新索引请先删除该文档。"
                       % (existing.id, status_text(existing.status)),
            )
        # 走到这里有两种情况：全新文件，或者**上次索引失败**。
        #
        # 失败的那条为什么放行：它并没有真的进知识库（chunk_count=0、检索不到），
        # 把它当成「已存在」会让用户陷入死循环 —— 想重试却被 409 挡住，
        # 只能先删除再传。**重新上传一个失败的文件，用户的意图就是重试。**
        # 下面 upsert 会按 source_file 命中同一行，把状态和 error 一起重置掉，
        # 所以不会产生第二条记录。

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


@document_router.post("/{paper_id}/reindex", status_code=http.HTTP_202_ACCEPTED,
                      response_model=UploadAccepted)
async def reindex_document(background: BackgroundTasks, paper_id: int) -> UploadAccepted:
    """重新索引一个已存在的文档 → 202。

    ## 什么时候需要它

    改了切块参数、换了 embedding 模型、或者上次索引失败之后。上传接口对同一份文件
    返回 409（"已经在知识库里了，需要重新索引请先删除该文档"）—— 在没有这个接口之前，
    那句话让用户去做一件他做不到的事。

    ## 为什么要先删旧块

    **不删的话新块会和旧块叠加**：同一段内容会在向量库里存在两份，检索时互相竞争，
    引用看起来还很正常（同一页、两个不同分数）。这种重复不会报错，只会让结果变差 ——
    又是一次静默失败。

    删在前面的代价是：如果重新解析失败，旧索引已经没了，文档变成 status=failed。
    这个取舍是**故意**的：
      · 重新索引的前提就是旧内容已经过时，留着它并没有价值
      · 失败会明确落在记录里（status=2 + error），用户看得到、可以重试
      · 反过来（先加后删）会在加成功、删失败时留下重复，而那是**看不见**的坏状态
    宁可要一个看得见的失败，不要一个看不见的重复。

    ## 为什么不需要额外的 task 表

    设计文档当时预计需要，但真做下来发现不需要：**「一篇文档同时只有一个索引操作」
    这个不变量对重索引同样成立**，所以 `paper.status` 依然够用。
    task 表真正会被需要是在这些场景：需要**操作历史**（第几次、什么时候、失败原因）、
    需要**批量重索引整个知识库**（任务就不挂在单个文档上了）、或者要**多 worker 并发**
    （那时需要跨进程的任务认领机制）。现在一个都不成立，提前建表只会多一处不一致。
    """
    async with async_session_maker() as session:
        paper = await PaperRepository.get_by_id(session=session, paper_id=paper_id)
        if paper is None:
            raise HTTPException(
                status_code=http.HTTP_404_NOT_FOUND, detail="没有 id=%s 的文档" % paper_id
            )
        if paper.status in (STATUS_PENDING, STATUS_INDEXING):
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail="这篇文档已经在处理中（%s），等它结束再重试。" % status_text(paper.status),
            )

        source_file = paper.source_file
        pdf_path = paper.pdf_path
        title = paper.title

        # 源文件可能已经被删掉了（比如手动清理过 uploads 目录）。
        # 没有源文件就无从重新解析 —— 这不是 404（文档记录还在），是状态不允许这个操作。
        if not pdf_path or not os.path.isfile(pdf_path):
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail="源文件不在磁盘上了（%s），无法重新索引。"
                       "请重新上传，或者先删除这条记录。" % (pdf_path or "路径为空"),
            )

        # 先删旧块（理由见 docstring）
        from ai.rag.chromaClient import document_vector_store

        removed = 0
        got = document_vector_store.get(where={"source": source_file})
        ids = got.get("ids") or []
        if ids:
            document_vector_store.delete(ids=ids)
            removed = len(ids)

        # 状态回到「待处理」，后台任务会把它推进到「处理中」→「已索引」
        await PaperRepository.update_progress(
            session=session, paper_id=paper_id, status=STATUS_PENDING, chunk_count=0
        )

    background.add_task(_ingest_in_background, paper_id, pdf_path, source_file, title)
    logger.info("reindex: 受理 id=%s（先清掉 %d 个旧块）", paper_id, removed)

    return UploadAccepted(
        id=paper_id,
        title=title,
        status=STATUS_PENDING,
        status_text=status_text(STATUS_PENDING),
        poll="/documents/%s" % paper_id,
    )


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


# ⚠️ 这个函数**故意不写返回注解**。
#
# 写成 `async def delete_document(...) -> None:` 会让 FastAPI 从注解推断出
# `response_model=NoneType`，而 `NoneType` 这个类对象是 truthy 的 ——
# 于是它认为"有响应体"，直接断言失败：
#
#     AssertionError: Status code 204 must not have a response body
#
# 所以这里也不写返回注解（注解写 `-> None` 都会触发那个断言）。
@document_router.delete("/{paper_id}", status_code=http.HTTP_204_NO_CONTENT)
async def delete_document(paper_id: int, force: bool = False):
    """删除一个文档：向量块、上传的文件、paper 记录，三样都要删。

    ## 为什么删的顺序不能反

    **先删向量块，再删 paper 行。** 反过来的话，万一删向量失败，
    `source_file` 这个唯一能把两边对起来的键就没了 —— 那些向量块变成
    永远找不到、也永远清不掉的孤儿。先删向量则失败时记录还在，可以重试。

    ## 为什么只删「自己管的」文件

    `resource/papers/` 里那 4 篇是操作者放进来的种子语料，由 `ingest.py` 管理；
    `resource/uploads/` 才是本接口的产物。所以磁盘删除**只允许发生在上传目录之内**，
    并且要先把路径规范化再判断（防 `..` 之类）。删库接口顺手把语料删了是不可逆的事故。

    ## 为什么默认拒绝删除「处理中」的文档

    后台索引任务手里攥着 `paper_id`，跑完会去写状态。此时把记录删掉：
      · 状态更新会落空（`update_progress` 找不到行，静默返回 None）
      · 但**向量块已经写进去了** —— 又一批孤儿
    所以状态是「待处理 / 处理中」时返回 **409**，这正是状态码速查里
    「当前状态不允许这个操作」那一类。

    `force=true` 是给**卡住的**文档留的出口：进程重启会让后台任务消失，
    记录永远停在「处理中」（`/health` 的 `index.by_status.indexing` 就是在报这个）。
    那种情况下没有任务会再写，强制删是安全的 —— 但也**只在那时**才安全。

    ## 为什么返回 204 而不是 200 + body

    删除是幂等且无副作用的查询语义之外的操作，204 足够，不需要回显 body。
    「删掉了几个向量块」这类细节进日志；要看聚合效果有 `/health` 的 `index`。
    """
    async with async_session_maker() as session:
        paper = await PaperRepository.get_by_id(session=session, paper_id=paper_id)
        if paper is None:
            raise HTTPException(
                status_code=http.HTTP_404_NOT_FOUND, detail="没有 id=%s 的文档" % paper_id
            )

        if paper.status in (STATUS_PENDING, STATUS_INDEXING) and not force:
            raise HTTPException(
                status_code=http.HTTP_409_CONFLICT,
                detail=(
                    "文档正在处理中（%s），此时删除会留下无法清理的向量块。"
                    "等它结束，或者如果确认是进程重启后卡住的，用 ?force=true。"
                    % status_text(paper.status)
                ),
            )

        source_file = paper.source_file
        pdf_path = paper.pdf_path
        title = paper.title

        # 1) 先删向量块（顺序理由见 docstring）
        from ai.rag.chromaClient import document_vector_store
        from ai.rag.hybrid import invalidate_index

        deleted_chunks = 0
        try:
            got = document_vector_store.get(where={"source": source_file})
            ids = got.get("ids") or []
            if ids:
                document_vector_store.delete(ids=ids)
                deleted_chunks = len(ids)
        except Exception as exc:
            logger.exception("delete: 删除向量块失败 id=%s", paper_id)
            raise HTTPException(
                status_code=http.HTTP_503_SERVICE_UNAVAILABLE,
                detail="向量库删除失败，文档记录已保留，可以重试：%s" % str(exc)[:120],
            )

        # 2) 让 BM25 索引重建 —— 否则被删掉的文档还能被关键词搜出来
        invalidate_index()

        # 3) 删磁盘文件。**只删上传目录里的**，种子语料不碰。
        removed_file = _remove_uploaded_file(pdf_path)

        # 4) 最后删记录
        await session.delete(paper)
        await session.commit()

    logger.info("delete: 删除 id=%s《%s》—— 向量块 %d 个，磁盘文件 %s",
                paper_id, title, deleted_chunks, "已删" if removed_file else "未动")


def _remove_uploaded_file(pdf_path: str) -> bool:
    """删掉上传目录里的 PDF。返回是否真的删了。

    两重保险：
      · `commonpath` 而不是字符串 startswith —— `startswith("/a/uploads")`
        会被 `/a/uploads-evil/x.pdf` 骗过去
      · 种子语料（resource/papers/）因此天然落在允许范围之外
    """
    if not pdf_path:
        return False
    try:
        upload_root = os.path.abspath(settings.UPLOAD_DIR)
        target = os.path.abspath(pdf_path)
        if os.path.commonpath([upload_root, target]) != upload_root:
            logger.info("delete: %s 不在上传目录内，不动磁盘文件", target)
            return False
        if os.path.isfile(target):
            os.remove(target)
            return True
    except Exception as exc:
        # 文件删不掉不该让整个删除失败：记录和向量才是"能不能检索到"的决定因素，
        # 残留一个文件只是占点磁盘。所以这里记警告，不抛。
        logger.warning("delete: 删除文件失败 %s：%s", pdf_path, exc)
    return False
