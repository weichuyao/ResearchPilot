# 改造 #5 之二设计：会话持久化（conversations API + checkpointer 落库）

> 对应 MISSION 第 5 项的"会话、Agent Task"部分。文档 API 那一半见
> `transformation-05-document-api-design.md`。同时清掉 `baseline-defects.md` B4
>（前后端会话状态不一致）和 #7 的欠账（统一异常层）。

---

## 零、要解决的问题（都实测过）

1. **`MemorySaver` 只在进程内存**：后端一重启，所有会话上下文消失。前端却把历史
   存在 `localStorage` 里 —— 重启后界面还显示历史，后端完全失忆。这就是
   baseline-defects B4，**看起来有记忆，实际没有**（又是静默失败的老朋友）。
2. **会话列表只在前端**：`chatSessions` 存 localStorage，换浏览器/清缓存全丢，
   后端对"有哪些会话"一无所知。
3. **异常处理没有统一层**：未知 agent 返回 500 + `Internal Server Error` 纯文本
   （SSE 场景下连 JSON 都不是），全局异常没有统一日志出口。

## 一、消息的 source of truth = checkpointer，不建 message 表

LangGraph 的 checkpointer 本来就**按 thread_id 全量持久化每轮消息**。再建一张
message 表等于同一份数据存两处 —— 项目里这类镜像已经吃过三次亏（agent 列表
手工镜像、上传格式手工镜像、chroma 重复块互相竞争）。

所以数据模型只建一张**索引表**：

```
Conversation(thread_id PK, title, agent_id, created_at, last_message_at)
```

| 读什么 | 从哪读 |
|---|---|
| 会话列表（侧边栏） | conversation 表 |
| 某个会话的全部消息 | `agent.aget_state(config)` → checkpoint 里的 messages |
| 删除会话 | conversation 行 + `checkpointer.adelete_thread()` |

conversation 表不存任何消息内容，坏了可以随时从 checkpoint 重建（它只是索引，
和 paper 表可重建是同一个思路）。

## 二、Checkpointer：MemorySaver → AsyncPostgresSaver

### 为什么跟着 DATABASE_URL 走，且只支持 postgres

- 项目的主库已经是 PostgreSQL（改造 #6），会话持久化没有理由用第二套存储。
- SQLite 路径要另写一套连接生命周期（aiosqlite 的连接必须在事件循环里打开，
  而 agent 图在**导入期**编译，拿不到异步连接）——为一个只在冒烟测试里出现的
  模式维护双代码路径，不值。
- **fallback：`DATABASE_URL` 不是 postgres 时用 `MemorySaver` 并打 WARNING 日志** ——
  行为退回现状（内存记忆），但日志里说清楚。测试脚本不受影响。

### 连接生命周期（本次实现最容易错的地方，三次实测修正）

两个 agent 图原来在**模块导入时** compile，而 `AsyncPostgresSaver.__init__`
会抓**当前运行中的事件循环** —— 没有循环的上下文（TestClient 导入应用、
评估脚本的 import 阶段）里构造它直接 RuntimeError。这套约束逼出了三层惰性：

```python
# 模块导入（无循环）——什么都不连
pool = AsyncConnectionPool(conninfo, open=False)

# startup / 评估脚本开头（循环在跑）
get_checkpointer()          # Saver 实例在这里才创建（要抓循环）
await pool.open()           # 连接在这里才建立
await saver.setup()         # 幂等 DDL
```

1. **Saver 惰性单例**：`get_checkpointer()` 首次调用才创建。调用点只有两个，
   都保证在循环内 —— main 的 startup 事件、agent 图的惰性编译。
2. **图惰性编译**：`graph.compile(checkpointer=...)` 从导入期挪进
   `build_xxx()` 工厂，由 `agents.py` 注册表在首次 `get_agent()` 时编译；
   main 的 startup 会预热全部图（编译错误在启动时暴露，而不是留给第一个请求）。
   这是对"图必须在导入期编译"这一旧假设的推翻 —— 旧假设在 TestClient / 评估
   脚本两条路径上都被验收抓了包。
3. **Windows 事件循环**：psycopg 异步模式只支持 SelectorEventLoop，而 uvicorn
   先建循环后导入应用，应用内设策略救不了 `python -m uvicorn` 直启。
   最终口径：策略在引入 psycopg 的 `checkpointer.py` 模块级设置（覆盖之后
   创建的一切循环），本地直启 uvicorn 需带 `--reload`（uvicorn 0.34 只在该
   模式自带 Selector）；run_eval 在模块顶层另设一次（它对 checkpointer 的
   导入发生在 asyncio.run 内部，晚了）。容器是 Linux，不受影响。

另有两个实测坑已写进代码注释：池要 `kwargs={"autocommit": True}`
（`setup()` 的迁移含 `CREATE INDEX CONCURRENTLY`，事务块内直接报错）；
`startup_checkpointer()` 必须幂等且被 run_eval 调用（独立进程没有 startup
事件，不调则第一笔 checkpoint 抛 PoolClosed）。

### 评估隔离（不做会静默污染评估）

`run_eval.py` 现在用 `thread_id = "eval-<题号>"`。checkpointer 持久化之后，
**第二轮评估会吃到第一轮的对话历史** —— A/B 类题的答案会在"上下文里已经有
上一轮的回答"的条件下产生，评估结果静默失真。B 类题尤其危险（上一轮说过
"未找到"，这一轮模型可能顺着说）。

修法：thread_id 加每次运行的随机前缀 `eval-<run8>-<题号>`，每轮评估都是
全新线程。这比"评估前清库"好 —— 清库会连带删掉真实用户的会话。

## 三、API

```
GET    /conversations                        列表（按 last_message_at 降序）
GET    /conversations/{thread_id}/messages   从 checkpoint 读全量消息
DELETE /conversations/{thread_id}            删索引行 + adelete_thread → 204
```

写入时机：`/chat/invoke` 和 `/chat/stream` 的**成功路径**结束后 upsert
conversation 行（标题 = 首条用户消息前 50 字）。流式中断（GeneratorExit /
CancelledError）不写 —— 会话列表里不该有一条只有半句话的会话；
checkpoint 里可能有部分状态，但没有索引行它就不可见，等价于"没聊过"。

**Agent Task API：不做。** 理由与 MCP 的推迟同类：系统目前没有"后台 agent
运行"这个概念，唯一的长时任务（文档索引）已经有 `/documents` 状态机 +
`/health` 承载。等真的出现异步 agent 任务再建，到时字段从真实需求里长出来。

## 四、统一异常层（#7 的欠账，顺手清）

`main.py` 注册两个 handler：

| 异常 | 行为 |
|---|---|
| `StarletteHTTPException` | 日志记 warning，detail **原样**返回（项目原则：detail 是特意写给人看的） |
| 其它一切 `Exception` | `logger.exception`（带完整 traceback 进日志文件 —— 刚做的 #7 派上用场）+ 500 `{"detail": "Internal server error"}` |

未知 agent 从"500 Internal Server Error 纯文本"改为**路由入口显式判 404**：
`get_agent` 抛 KeyError 让 ai 层不依赖 fastapi，chat 路由先查注册表再取图。

顺手清残留：`message_generator` 里 `supervisor` / `math_agent` / `code_agent`
的分支是已删除的 multi_agent 的遗产，永远走不到，摘掉。

## 五、前端（B4 的另一半）

原则与 AgentSelector 相同：**后端是唯一事实源，不造本地假数据**。

- 会话列表：`layout.tsx` 挂载时拉 `GET /conversations`；localStorage 的
  `chatSessions` 删除。发送首条消息时先本地乐观插入（此刻后端还没写行），
  之后以拉取为准。
- 消息历史：进入会话时从 `GET /conversations/{id}/messages` 拉取；
  localStorage 的 `chatMessages-*` 删除。
- 删除会话：调 DELETE 接口，失败则报错且**不**在界面上假装删掉。
- 旧的 localStorage 键不做迁移：会话本来就是进程内存里的易失数据，
  没有值得迁移的东西，直接退役。

## 六、验证标准

1. 后端重启前问一个问题 → 重启后 `GET /conversations/{id}/messages` 还能
   读出完整对话（checkpointer 落库的证据）。
2. 同一 thread 连续两轮对话 → 第二轮模型知道第一轮的内容（真恢复，不只是消息存档）。
3. `run_eval.py` 两轮连跑 → 第二轮 thread 与第一轮不同（隔离生效）。
4. 未知 agent → 404 + JSON detail；未知路径 → 404；全局异常 → 日志文件里有 traceback。
5. DELETE 会话后：列表消失、消息 404、checkpoint 里 thread 消失。
6. sqlite 环境（现有 TestClient 脚本）→ 退回 MemorySaver + WARNING，测试照常绿。
