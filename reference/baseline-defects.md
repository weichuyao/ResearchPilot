# Baseline 缺陷清单（已实测验证）

这份清单是**动手改造阶段的起点**。每一项都标注了证据来源：[实测] 表示我在本机跑过并观察到了结果，[读码] 表示从源码逻辑推导（未单独跑验证）。

改造原则：**先讲清为什么这么改，再动手。** 尤其是 Bug 2 和 Bug 3，它们不是"bug"，而是 ResearchPilot 改造 #2 / #3 的**动机本身**。

---

## A. 会直接导致功能失败的缺陷

### A1 · `search_handbook` 的"没找到"分支是死代码 [实测]

- 位置：`backend/app/ai/tools/oa_tools.py:46`
- 现象：`if result.__len__ == 0:` —— `result.__len__` 是**方法对象**，永远不等于 `0`。
- 实测：`[].__len__ == 0` → `False`；`len([]) == 0` → `True`。
- 后果：查不到内容时返回 `"\n\n".join([])` = **空字符串**，而不是 `"no result found"`。模型收到一条空的 ToolMessage。
- 归属：改造 #2（重写 RAG）顺手修掉。

### A2 · 部门写接口三个全坏 [实测]

| 位置 | 代码 | 后果 |
|---|---|---|
| `backend/app/db/repository/department_repo.py:11` | `session.add(departmnt)` | 拼写错误 → `NameError` → `POST /department/add` 必 500 |
| `department_repo.py:31` | `cls.get_department(department_id)` | 少传 `session` → `TypeError` → `PUT /department/update/{id}` 必 500 |
| `department_repo.py:40` | `cls.get_department(department_id)` | 同上 → `DELETE /department/del/{id}` 必 500 |

- 对照组：`employee_repo.py` 的同类方法**都正确传了 `session`**。所以这不是设计缺陷，是复制粘贴漏改。
- 归属：改造 #5（完善 FastAPI）的入门练习题——边界清晰、可立刻验证、不需理解全系统。

> **重要补充（动手时才发现）**：A2 这三个 500 里，`POST /department/add` 修好拼写后**会立刻暴露出 A5**。
> 因为 `NameError` 发生在第 11 行，第 12 行的 `commit()`（INSERT 真正发生的地方）永远执行不到。
> **修好一个 bug 会暴露被它掩盖的下一个 bug** —— 这是真实的调试常态。

### A5 · `DBBaseModel` 的时间默认值写成了「函数本身」，导致所有创建接口都插不进去 [实测]

- 位置：`backend/app/db/models/base.py:12-13`
  ```python
  create_time: datetime | None = Field(default=datetime.now, ...)     # ← 少了 ()
  edit_time:   datetime | None = Field(default=datetime.now, ...)     # ← 少了 ()
  ```
- **现象**：`datetime.now` 是一个**函数对象**，不是调用结果。Pydantic v2 不会替你调用它，所以字段的默认值**就是这个函数本身**。
- 实测：
  ```
  Employee.create_time 的默认值 = <built-in method now of type object at ...>
  类型 = builtin_function_or_method
  ```
- 后果：SQLAlchemy 要求 `DateTime` 列必须是 `datetime` 对象，收到函数对象 → 抛
  `sqlalchemy.exc.StatementError: (builtins.TypeError) SQLite DateTime type only accepts Python datetime and date objects as input.`
- **注意不能简单加括号**：`default=datetime.now()` 会在**模块导入时**算一次，之后所有记录共用同一个时间戳。正确写法是：
  ```python
  Field(default_factory=datetime.now)
  ```
  `default_factory` 的含义是「**需要的时候再调用它**」。
- 实测修法有效：改 `default_factory` 后，FastAPI 校验出的 `create_time` 是真正的 `datetime`，INSERT 成功。
- **影响面（比看起来大）**：`Employee` 和 `Department` 都继承 `DBBaseModel`，所以下面这些接口全都有同一个问题：
  - `POST /department/add`（被 A2 的 NameError 掩盖着，修完 A2 才暴露）
  - **`POST /employee/add`**（实测等价操作失败，同一个异常）← 原清单漏掉了这一条
- 归属：改造 #5；也是改造 #6（数据层）要一起处理的模型设计问题。

### A6 · 用数据库模型直接当请求体模型（反模式）[实测]

- 位置：`backend/app/api/department_routers.py:19-23`、`employee_routers.py:12`
  ```python
  async def create_department(department: Department, session: SessionDep):
  ```
  请求体类型直接用了 `table=True` 的数据库模型。
- 后果一：客户端被要求提供 `id`、`create_time`、`edit_time` —— **服务端生成的字段变成了客户端的义务**。
- 后果二：**客户端传来的时间字符串不会被转换成 `datetime`**。实测（修好 A5 之后）：
  | 请求体 | 结果 |
  |---|---|
  | `{"id":100,"name":"x"}` | `create_time` = `datetime` ✅ |
  | `{"name":"x"}` | `id` = `None`（SQLite 会自增），`create_time` = `datetime` ✅ |
  | `{"id":101,"name":"x","create_time":"2026-…Z",…}` | `create_time` = **`str`** ❌ 仍然失败 |
- 实测三条 FastAPI 校验路径的差异（这是根因所在）：
  ```
  Department.model_validate(payload)        -> create_time 类型 = datetime
  TypeAdapter(Department).validate_python() -> create_time 类型 = str     ← FastAPI 用的是这条
  ```
- 正确做法：单独定义**创建请求模型**，只含客户端该填的字段（如 `DepartmentCreate: name / parent_id / manager_id`），`id` 与时间戳由服务端负责。
- 归属：**改造 #5** 的核心内容之一。不在第一次动手的范围内。

### A7 · `edit_time` 是个摆设：更新记录时从不刷新 [实测]

- 位置：`backend/app/db/repository/department_repo.py:30-37`（`update_department`）
  ```python
  for key, value in department_data.items():
      setattr(department, key, value)      # 只改了客户端传进来的字段
  await session.commit()
  ```
  `edit_time` 从不在更新时被赋值。
- 实测证据：
  - 对 `id=900` 执行 `PUT {"name":"YanShouGaiMing"}` 后，返回体里
    `create_time` 与 `edit_time` **完全相同**（`2026-09-11T15:12:12.208296`）。
  - `id=1` 被改名为 `试一下` 之后，`edit_time` 仍然是 `2025-06-17 07:30:35`——**没变**。
- 后果：字段名承诺"更新时间"，实际永远等于创建时间。任何依赖 `edit_time` 做增量同步、审计或缓存失效的逻辑都会失效。
- 归属：改造 #5（API 与业务逻辑）、改造 #6（数据层，`updated_at` 应该由数据库或 ORM 自动维护）。
- 同类风险：`Employee` 的更新路径（`employee_repo.update_employee`）有同样的问题。

### A3 · `GET /employee/get_by_name/{name}` 必 500 [实测]

- 位置：`backend/app/api/employee_routers.py:56`
- 现象：`await get_user_info(name)` 把 `@tool` 生成的 `StructuredTool` 当普通函数调用。
- 实测：抛 `NotImplementedError: StructuredTool does not support sync invocation.`
- 正确写法：`await get_user_info.ainvoke({"user_name": name})`
- 归属：改造 #5。

### A4 · `GET /department/get_by_name/{department_name}` 无视路径里的名字 [实测]

- 位置：`backend/app/api/department_routers.py:39-44`
- 实测（来自 `app.openapi()`）：该路由的参数是 **`department_id: query`**，路径里的 `department_name` 根本没有被声明。
- 后果：请求 `/department/get_by_name/人事部` → 422 缺参数；加 `?department_id=1` 能通，但返回的是 id=1 的部门，名字被彻底丢掉。
- 归属：改造 #5。

---

## B. 不是 bug，但是 ResearchPilot 的核心动机

### B1 · RAG 没有相关性阈值：查不到也会硬塞 10 条无关内容 [实测]

- 位置：`backend/app/ai/tools/oa_tools.py:44`（`similarity_search(query, k=10)`）
- 实测：用完全不存在的主题查询

  ```
  search_handbook.ainvoke({"query": "zzzzz nonexistent topic qqqq"})
  → 'absence regardless of the length of the absence.\n\nmessages or other non-job-related purposes...'
  ```

- 后果：模型**永远**收到 10 段材料，它无从知道这些材料其实无关，于是照着编。检索层没有"证据不足"这个概念。
- **这就是"答案看起来像编的、但系统不报任何错"的机制之一。**
- 归属：**改造 #3 的核心**（判断证据充分性 → 补充检索）。也是面试高频题："你的 RAG 检索不到内容时怎么办？"

### B2 · chunk 质量差：切得太碎，还带不可见字符 [实测]

- 位置：`backend/tests/rag/importRag.py:27`（`chunk_size=100, chunk_overlap=20`）
- 实测：`handbook` collection 共 **875 条**向量；最相关的片段长这样：
  - `'Vacation\xa0benefits.'`（18 字符）
  - `'to\xa0use\xa0accrued\xa0vacation\xa0or\xa0personal\xa0leave.'`（42 字符）
- 大量 `\xa0`（不换行空格）干扰了递归切分，句子被从中间切断。
- 归属：改造 #2 的第一件事。

### B3 · `search_handbook` 丢弃文档 metadata [读码]

- 位置：`backend/app/ai/tools/oa_tools.py:49` —— 只 `join(doc.page_content)`，丢掉 `source` / `page` 等字段。
- 实测：Chroma 里每条文档的 metadata 包含 `source`、`page`、`page_label`、`total_pages` 等。
- 后果：**无法做 citation**。这是改造 #2（… → citation）必须解决的。

### B4 · 会话状态前后端不一致 [读码]

- 后端：`oa_assistant.py:82` 用 `MemorySaver()`，进程重启即清空。
- 前端：`ChatComponent.tsx:36` 用 `localStorage` 持久化。
- 后果：重启后端后，前端还显示历史消息，但后端完全不记得上下文。
- 归属：改造 #6（数据层）/ #7（工程化）。

---

## C. 结构性坑（不是 bug，但会绊住你）

| # | 问题 | 位置 | 说明 |
|---|---|---|---|
| C1 | `db/models.py` 与 `db/models/` 同名 | `backend/app/db/` | [实测] `import db.models` 解析到**包**（`db/models/__init__.py`，空的）。所以 `db/models.py` 是**永不可达的死代码**；且它 `from .database import Base`，而 `database.py` 里没有 `Base`，一旦真被导入就 ImportError |
| C2 | FastAPI 教程残留死代码 | `api/services.py`、`api/schemas.py`、`db/models.py`、`db/__init__.py` | 没有任何文件 import 它们；`db/__init__.py` 还在 import 时建了个没人用的 `sqlite:///./sql_app.db` engine |
| C3 | `@dataclass` 是"歪打正着" | `db/models/employee.py`、`department.py` | 加 `@dataclass` 是为了让 `asdict()` 能用；副作用是 `asdict()` 只按子类自己的注解取字段，`DBBaseModel` 的 `create_time`/`edit_time` 被**静默丢弃**（实测 `asdict(employee)` 确实没有这两个键）。正因如此 `json.dumps` 才能成功 |
| C4 | Prompt 占位符没被替换 | `ai/agent/oa_assistant.py:39` vs `:34` | `instructions` 里写了 `{current_time}`，但用的是 `SystemMessage(content=instructions)`，没 `.format()`。**模型看到的是字面量 `{current_time}`，不是真实时间** |
| C5 | 全局开启 LangChain debug | `oa_assistant.py:19`、`multi_agent.py:14` | `set_debug(True)`，后台日志非常吵 |
| C6 | 热重载导致残留 worker | `run_server.py:18` | `reload=settings.is_dev()` 而 `DEV=True`。这是之前"8000 端口残留多个 worker、请求随机落到旧进程"的直接原因 |
| C7 | 前端 toolCall 解引用不安全 | `frontend/app/chat/hooks/useStreamChat.ts:121` | `prev[prev.length-1].toolCall.calls.map` 没有 `?.`（第 87 行同样的写法有兜底）。若 `tool` 消息先到而 `toolCall` 未建立，会抛 TypeError |
| C8 | Chroma 路径依赖工作目录 | `backend/.env` 的 `CHROMA_PATH=resource/chroma_db` | 相对路径靠进程 CWD 解析。这是 `backend/resource/`、`backend/app/resource/`、项目根 `resource/` 三处都有 chroma 产物的原因——**只有 `backend/resource/`（875 条向量）是有效的** |
| C9 | 数据库列与模型不一致 | `backend/resource/create_tables.sql` vs `db/models/employee.py` | DB 的 `employee` 表有 `birth_date` 列，SQLModel 模型里没有。无害，但不一致 |
| C10 | 无用的 `tools_condition` 导入 | `ai/agent/oa_assistant.py:10` | 实际用的是自定义的 `pending_tool_calls`；`tools_condition` 导入了但没用 |

---

## D. 环境类问题（已在脚本中修复，不属于项目代码）

### D1 · 系统代理导致 httpx 访问 Ollama 返回 502 [实测，已修复]

- 机器状态：Windows 系统代理开启（`ProxyEnable=1`，`ProxyServer=127.0.0.1:7897`，`ProxyOverride` 含 `127.*;localhost`）
- 机制：`PowerShell` 与 `urllib` 会读 `ProxyOverride` 白名单；**`httpx` 不读**。而 `langchain_ollama.OllamaEmbeddings` → `ollama` python 客户端 → `httpx`。
- 实测 A/B：不加 `NO_PROXY` → 502；加 `NO_PROXY=127.0.0.1,localhost,::1` → 200。
- 已修复于：`scripts/start-ai-chatkit.ps1`（设置 `NO_PROXY` + 启动前自检），以及用户级环境变量 `NO_PROXY`。
- **教学价值**：修复前那次请求是 `HTTP 200`、SSE 正常 `end`、`error` 事件为 0 —— 错误只在 Tool 层被模型吞成一句道歉。这是"分层排查"的真实案例。
