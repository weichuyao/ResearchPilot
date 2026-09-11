"""论文的结构化元数据。

和向量库的分工：
  · 向量库（Chroma 的 papers collection）存的是论文**切块后的文本 + 向量**，用于语义检索。
  · 这张表存的是论文**本身的属性**（标题、作者、年份、状态……），用于精确查询与过滤。

为什么要分开：语义检索擅长「按意思找」，但答不了「库里有哪几篇」「2023 年之后的」
「作者是谁」「哪篇导入失败了」这类问题 —— 那些需要结构化查询。
而检索时的 metadata filter 也依赖这里的 source_file 去圈定某一篇论文。
"""

from datetime import datetime

from sqlmodel import Field, SQLModel

from db.models.base import DBBaseModel


# status 取值
#
# 为什么「待处理」和「处理中」必须分开（改造 #5 加的 3）：
# 后台导入任务跑在 API 进程内，**进程重启会让任务丢失**。那时记录会永远停在 3。
# 如果 0 和 3 合并成一个值，就分不清「排队等着」和「重启前卡死了」——
# 前者会自己好，后者需要人工重试。分开之后 /health 和启动逻辑就能识别后者。
STATUS_PENDING = 0
STATUS_INDEXED = 1
STATUS_FAILED = 2
STATUS_INDEXING = 3


class Paper(DBBaseModel, table=True):
    """知识库里的一篇论文。"""

    __tablename__ = "paper"

    # 主键可空 + 默认 None，交给 SQLite 自增。
    # 反例：Department.id 写成了必填的 int，导致客户端被迫提供主键（缺陷 A6）。
    id: int | None = Field(default=None, primary_key=True, description="论文 id")

    # source_file 是和向量库对接的关键：
    # Chroma 里每个块的 metadata["source"] 就是这个文件名。
    source_file: str = Field(
        max_length=300, index=True, unique=True, description="导入用的 PDF 文件名"
    )
    title: str = Field(max_length=300, index=True, description="论文标题")

    authors: str = Field(default="", max_length=500, description="作者，英文逗号分隔")
    venue: str = Field(default="", max_length=200, description="会议或期刊")
    year: int | None = Field(default=None, index=True, description="发表年份")
    external_id: str = Field(default="", max_length=120, description="arXiv ID 或 DOI")

    pdf_path: str = Field(default="", max_length=500, description="相对 backend/ 的 PDF 路径")
    collection_id: int = Field(default=0, index=True, description="所属知识库 id")

    status: int = Field(
        default=STATUS_PENDING, description="0 待处理 / 1 已索引 / 2 失败 / 3 处理中"
    )
    chunk_count: int = Field(default=0, description="已写入向量库的块数")
    indexed_at: datetime | None = Field(default=None, description="最近一次索引完成时间")

    # 导入失败的原因（成功时为空串）。
    #
    # 为什么失败必须落在记录里、而不是只打一条日志：用户看到的是「上传了但一直没反应」，
    # 而日志在服务器的另一个地方。没有这一列，失败就是一个**静默失败** ——
    # 这个项目一路都在跟这类问题较劲（工具报错变成文本、HTTP 200 + 0 error 事件……）。
    error: str = Field(default="", max_length=500, description="最近一次导入失败的原因")
