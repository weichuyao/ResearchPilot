# 部署：用 Docker 跑起来

> 改造 #9 的操作手册。设计取舍在 `transformation-06-09-data-and-docker-design.md`，
> 这里只讲**怎么做**和**出问题怎么查**。

---

## 最快的一条路：双击

仓库根目录有个 **`启动 Docker.cmd`**，**双击它**就行。

它内部做四件事：

1. 检查 Docker Desktop 是否在跑（没跑就明确告诉你，而不是给你一串报错）
2. 按根目录 `.env` 里的镜像源准备基础镜像
3. 构建两个镜像（自动把当前 git 提交号传进去）
4. 启动，并**等到后端真的健康**（连上数据库、加载完 266MB 重排模型）才报完成

跳过构建直接启动（镜像没变时快很多）：

```
启动 Docker.cmd -NoBuild
```

---

## Docker Desktop 的图形界面能做什么、不能做什么

**这个问题值得先说清楚**，因为它决定了你要不要跟命令行打交道。

| 操作 | GUI | 说明 |
|---|---|---|
| 看容器状态 | ✅ | Containers 标签页 |
| **看日志** | ✅ | 点容器 → Logs。**出问题时这里最重要** |
| 进容器摸一摸 | ✅ | 点容器 → Exec，比命令行方便 |
| 看构建进度 | ✅ | **Builds** 标签页，有正在进行的构建和历史记录 |
| 看数据卷 | ✅ | Volumes 标签页 |
| 调 CPU / 内存 | ✅ | Settings → Resources |
| 拉单个镜像 | ✅ | Images → Pull |
| **compose 编排** | ❌ | GUI 不读 `docker-compose.yml`。三个服务一起起、按依赖顺序起、变量替换 —— 都做不了 |
| **构建镜像** | ⚠️ | 较新版本 Images 页有 Build 按钮，但那是**单镜像**的，不处理 compose 里的构建参数 |

**核心矛盾**：这个项目是**多服务 + 有启动顺序依赖**的（数据库先健康，后端才能连），
这正是 compose 存在的理由，而 compose 是命令行工具。

所以实际的分工是：**双击 `.cmd` 启动 → 用 GUI 看状态和日志**。
GUI 负责"看"，脚本负责"做"。

---

## 三个必须先知道的前提

### ① Ollama 不在 Docker 里，必须由宿主机提供

embedding 用宿主机的 Ollama（`127.0.0.1:11434`）。容器里的 `127.0.0.1` 是容器自己，
所以后端通过 `host.docker.internal` 反着找宿主机。

**这套栈不是自包含的。** `docker compose up` 之后系统仍然依赖宿主机跑着 Ollama。

为什么不把 Ollama 也塞进容器：要 GPU 直通，Windows 上很折腾；而且它本来就是个
该被多个项目共享的服务。**与其假装一键就绪，不如把这个依赖写在明处。**

故障表现：问问题返回 502 或者 "connection refused" → 先确认宿主机的 Ollama 还在。

### ② 前端 API 地址是构建期烘进 bundle 的

Next.js 把 `NEXT_PUBLIC_*` 在 `next build` 时替换成字面量，**运行时改环境变量没用**。

而且这个地址必须是**浏览器能访问的**：

| 写法 | 结果 |
|---|---|
| `http://localhost:8001` | ✅ 浏览器跑在宿主机上，能访问 |
| `http://backend:8001` | ❌ 容器名只在容器网络里能解析 → 页面能打开，所有请求都失败 |

要改的话改根目录 `.env` 里的 `NEXT_PUBLIC_API_BASE_URL`，然后**重新构建前端**。

### ③ 不要同时跑本地后端和容器后端

Chroma 是**嵌入式**的（一个目录，不是服务）。两边挂同一个目录会互相破坏索引，
而且 8001 端口会冲突。

切换：

```powershell
scripts\stop-ai-chatkit.ps1     # 停掉本地那两个
启动 Docker.cmd
```

---

## 常用操作

```powershell
docker compose ps                      # 看状态
docker compose logs -f backend         # 跟后端日志
docker compose logs -f frontend
docker compose restart backend         # 只重启后端
docker compose down                    # 全停（数据保留）
docker compose down -v                 # 全停 + 删数据库数据（慎用）
docker compose up -d --build backend   # 改完后端代码后重建
```

> ⚠️ **改了前端代码要重建镜像**，不能只 `restart` —— 见前提 ②。

---

## 日志在哪里

后端日志同时写两处：stdout（`docker compose logs -f backend`）和
**宿主机的 `backend/logs/app.log`**（compose 里 bind mount 进容器的 /app/logs，
见 core/logging_config.py）。轮转上界 10 MB × 5 份 —— **容器重启、删除之后
日志仍然在宿主机上**，这正是落文件要解决的问题：倒查故障要的是重启前的输出。

评估脚本（run_eval.py）会把根日志整体静音，它的日志只在自己的报告里。

## 出问题怎么查

### 第一步：分清是"构建期"还是"运行期"

- 构建失败 → 看 `docker compose build <服务>` 的输出
- 运行失败 → 看 `docker compose logs <服务>`

### 第二步：进容器手动试

```powershell
docker compose exec backend bash
# 进去之后
python -c "import sys; print(sys.version)"
curl http://host.docker.internal:11434/api/tags     # 能不能摸到宿主机的 Ollama
curl http://127.0.0.1:8001/health                   # 后端自己觉得健康吗
```

### 第三步：`/health` 说什么

```powershell
curl.exe http://127.0.0.1:8001/health
```

| 字段 | 说明 |
|---|---|
| `git_rev` | **线上跑的是哪份代码**。容器里读不到 `.git`，是靠构建参数传进去的 |
| `reranker` | `unavailable` 说明重排模型没加载 —— 系统会**静默降级**成纯混合检索 |
| `index.by_status.indexing` | 大于 0 说明有导入任务卡住了（进程重启丢了后台任务） |

---

## 已知的坑

### Docker Hub 认证连不上（本机实测）

```
failed to fetch oauth token: Post "https://auth.docker.io/token": ... timeout
```

拉**公共**镜像也要先取 token，而 Docker daemon 没走系统代理。

**两个解法，推荐第一个**：

**解法一（推荐）：让基础镜像从国内镜像源拉。** 根目录 `.env`：

```
PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
NODE_BASE_IMAGE=docker.m.daocloud.io/library/node:20-alpine
POSTGRES_IMAGE=docker.m.daocloud.io/library/postgres:16-alpine
```

Dockerfile 和 compose 里仍然写官方名，**文件保持可移植**。
换到网络正常的机器：删掉这三行即可。

**解法二：手动拉 + 打标签。** 不想用 `.env` 的话：

```powershell
docker pull docker.m.daocloud.io/library/python:3.11-slim
docker tag  docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim
# 前端、数据库同理
```

### 依赖装不上（`ResolutionImpossible`）

```
ERROR: Cannot install -r requirements.txt (line 4) and fastapi>=0.115.12
       because these package versions have conflicting dependencies.
```

**根因是项目没有锁文件**：`pyproject.toml` 全是 `>=` 宽松约束，全新解析会挑到
chromadb 1.x 和 fastapi 撞车。本机 venv 能用是因为它是更早某个时间点解析出来的。

修法已经落地：`backend/requirements.lock` 锁住了实测可用的精确版本。
**加了新依赖之后要重新生成**：

```powershell
uv pip freeze --python backend\.venv-py311\Scripts\python.exe
# 然后手工剔掉 Windows 专有的包（文件末尾有说明）
```

### 改了 compose 的项目名，数据"消失"了

Compose 的项目名（默认是**目录名**）决定了数据卷的名字 `<项目名>_pgdata`。

**改项目名等于换了一套数据。** 所以 `docker-compose.yml` 里刻意**没有**写 `name:`。

---

## 换个机器要重做哪几步

1. 装 Docker Desktop
2. 按上面的「Docker Hub 认证连不上」建好根目录 `.env`（如果那台机器也访问不了）
3. 宿主机的 Ollama 要跑着，并且 `ollama pull bge-m3`
4. 重排模型 266MB —— 跑一次 `scripts\fetch-reranker.ps1`（它挂载进容器）
5. 双击 `启动 Docker.cmd`

第 4 步容易漏：`resource/models/` 被 gitignore 了，新 clone 没有这个模型。
**漏了不会报错** —— 系统会静默降级成纯混合检索，从回答质量上完全看不出来。
所以启动后一定要看一眼 `/health` 的 `reranker` 字段。
