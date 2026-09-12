# 改造 #7 设计：日志落文件（logging 到 RotatingFileHandler）

> 对应 MISSION 第 7 项：异常、日志、任务状态、流式。本文只做**日志落盘**这一件事；
> 统一异常处理层仍然欠着（见文末）。

---

## 一、要解决的问题（两条实测证据）

### 证据一：容器一重启，日志就没了

Docker 容器重启/删除后，stdout 里的日志全部消失，`docker compose logs` 只能看到
**还在跑的进程**的输出。而故障排查最需要的恰恰是重启前发生了什么 ——
改造 #5 期间那个"后端跑的是几小时前的旧代码"的事故里，最后是靠手工比对进程
启动时间才确认的。当时要是有文件日志，`/health` 的 `started_at` 加日志第一行
就能直接回答。

### 证据二：根日志配置藏在 agent 模块的导入副作用里

改造 #7 之前，唯一的 `logging.basicConfig` 在 `react_assistant.py`（更早是
`oa_assistant.py`）的**模块顶层**。三个问题：

| 问题 | 后果 |
|---|---|
| 配置位置不可发现 | `/api/system_routes.py` 打的日志，格式居然由 `ai/agent/` 下某个文件决定 |
| 不导入就不配置 | 任何不经过 agent 模块的路径（脚本、测试、部分工具），日志退回 Python 默认裸格式，且不落盘 |
| 导入顺序影响全局行为 | 调整 agent 的导入顺序可能静默改变整个应用的日志行为 |

---

## 二、决策

### 决策一：配置收敛到 `core/logging_config.py`，由 `main.py` 导入时执行

任何一条启动路径（uvicorn / 脚本 / 测试）只要导入 `main` 就有统一格式 + 文件。
agent 模块只保留 LangChain 自己的 `set_debug`（那是 LangChain 的调试开关，
不是 stdlib logging 的事）。`setup_logging()` 幂等：给文件 handler 打标再查重，
**不清空已有 handlers** —— 后者会顺手扔掉 pytest / uvicorn 装的 handler。

### 决策二：console 行为不变，文件是**增量**

根日志同时接 console（沿用原格式，行为不变）+ `RotatingFileHandler`
（`backend/logs/app.log`，10 MB × 5 份，UTF-8）。轮转上界防止磁盘被慢写满；
10 MB 是"一天跑不出一个文件"的量级，5 份够倒查一周。

### 决策三：目录锚定到 backend/，容器里 bind mount

和 `chromaClient.py` 的 CHROMA_PATH 同一个教训：相对路径按进程 CWD 解析。
compose 里挂 `./backend/logs:/app/logs`（bind mount，不是 named volume）——
落文件要的就是"容器没了，日志还在宿主机上"。

### 决策四：uvicorn 的日志只在文件里"追加"，console 保持 uvicorn 原样

uvicorn 的 access / error 日志自带 handlers 且不向根传播，默认进不了文件。
把文件 handler 追加到它们的 logger 上，console 保持 uvicorn 自己的格式
（带颜色对齐）。**实测踩过一个坑**：第一版把 handler 挂到了
`("uvicorn", "uvicorn.error", "uvicorn.access")` 三个上 —— 但 `uvicorn.error`
没有自己的 handler、会向上传播给 `uvicorn`，于是每条启动日志在文件里
**重复两遍**。修正为只挂 `uvicorn` 和 `uvicorn.access`（后者不传播）。

> 教训：往 logger 树上加 handler 之前，先弄清每个节点**自带什么、向谁传播**。
> 这和 chroma 重复块、RRF 并列分数是同一类问题：同一份数据走了两条路，没人去重。

### 决策五：噪音库压制跟着配置走

`httpx / httpcore / openai / urllib3 / chromadb` 压到 WARNING（它们每次 HTTP
调用都会打完整请求体，含用户提问 —— 既是噪音也是隐私问题）。这条规则原来
也在 react_assistant 里，属于 app 级策略，跟着日志配置走。

### 决策六：评估脚本静音而不是配置

`run_eval.py` 保持 `logging.disable(logging.INFO)`：评估的输出就在它自己的
报告里，应用日志只会碍事（而且 logging 直写 stderr，`redirect_stdout` 拦不住）。

---

## 三、验证（实测）

| 检查 | 结果 |
|---|---|
| `setup_logging()` 连调 3 次 | root 恰好 1 file + 1 console handler，无重复 |
| uvicorn / uvicorn.access 的文件 handler | 各恰好 1 个 |
| 启动 + 2 次请求后 `backend/logs/app.log` | 7 行：启动 4 行 + 重排模型加载 1 行 + access 2 行，**每行恰好出现一次**（修正前启动行 ×2） |
| 日志目录 | 本机 `backend/logs/`；容器里 `/app/logs`（bind mount 到宿主机同一位置） |
| 既有行为 | console 输出与改造前一致（uvicorn 原格式），`/health`、`/agents` 正常 |

---

## 四、还没做的（#7 的剩余部分）

| 项 | 说明 |
|---|---|
| **统一异常处理层** | FastAPI 的 exception handler + 业务异常类型还没统一；现在各路由自己 raise HTTPException，工具层错误仍会以 SSE 事件或 500 的形式漏出。等 #5 会话 API 一起做（都会动 api 层） |
| **结构化日志（JSON）** | 现在是人读格式。真要接日志聚合（ELK / Loki）时再改 formatter —— 单机部署先不引入这个复杂度 |
| **请求 ID 关联** | 一次请求的多行日志还没法串起来。等做会话 API 时用 thread_id 天然关联，先不加中间件 |
