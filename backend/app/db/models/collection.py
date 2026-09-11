"""知识库（论文集合）。

对应原来 OA 结构里的 Department：一个「归属方」，下挂多个被管理的实体。
"""

from sqlmodel import Field, SQLModel

from db.models.base import DBBaseModel


class Collection(DBBaseModel, table=True):
    """一个知识库 = 一组论文。

    刻意不用 @dataclass：Employee / Department 上加 @dataclass 是为了让 asdict() 能用，
    副作用是 asdict() 只按子类注解取字段，DBBaseModel 的 create_time / edit_time 会被
    静默丢掉。新模型直接用 Pydantic 的 model_dump()，不需要那个装饰器。
    """

    __tablename__ = "collection"

    id: int | None = Field(default=None, primary_key=True, description="知识库 id")
    name: str = Field(max_length=100, index=True, unique=True, description="知识库名称")
    description: str = Field(default="", max_length=500, description="知识库说明")
