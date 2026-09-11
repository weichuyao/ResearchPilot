# 改造 #6 + #9：PostgreSQL 与 Docker Compose

> 一起做的原因：Docker 逼着把「Python 版本」和「数据库」这两件一直含糊的事定下来。
> 单独做 #6 会留下「本地 SQLite、容器 Postgres」的双轨，而双轨必然漂移。

---

## 一、#6：换 PostgreSQL

### 1.1 数据层几乎没改代码 —— 但这不等于"没活干"

`SQLModel/SQLAlchemy` 把方言差异抽象掉了，所以：

```bash
DATABASE_URL=postgresql+asyncpg://... python -c "from db.database import create_db_and_tables; ..."
```

建表、查询、`get_or_insert` 全部直接工作，**一行代码没改**。

这能成立是因为之前刻意避开了 SQLite 特有的写法。真正要干的活是别的：

| 事 | 为什么必须做 |
|---|---|
| 装 `asyncpg` | SQLAlchemy 的 `postgresql+asyncpg://` 要它（选 asyncpg 而不是 psycopg：官方 async 文档首选） |
| 修跨事件循环的 bug | 见下面 —— **这是换库真正的收益** |
| `pyproject.toml` 的 Python 版本 | 写 `>=3.13` 而实际跑 3.11，Docker 逼着定下来 → 改成 `>=3.11` |
| 数据迁移 | 重跑一次导入即可（种子语料 4 篇 + uploads 里 3 篇） |

### 1.2 换库真正的收益：**它把一直是坏的东西暴露出来了**

SQLite（aiosqlite）对「跨事件循环复用连接池」这件事**很宽容**。
asyncpg 一点都不宽容，于是今天连续炸出三处：

```
RuntimeError: Event loop is closed
AttributeError: 'NoneType' object has no attribute 'send'      ← asyncpg 去写一个已死的 transport
```

**这三处在 SQLite 上一直都是坏的，只是不报错。**

1. **测试的 `cleanup()`**：在 `with TestClient(...)` 之外用 `asyncio.run` 访问数据库，
   而全局 engine 的连接池绑在 TestClient 那个**已经关闭**的事件循环上。
   → 改成走 `DELETE /documents` 接口清理（本来就更合理：用用户会用的接口清理）。
2. **测试的 `set_status()`**：同样的问题。
   → 抽了 `_run_isolated()`：**带外操作用自己的 engine，用完 dispose**。
3. **测试末尾又开了一个 TestClient**：新循环 + 旧连接池。
   → 把清理和基线核对都收进同一个 TestClient 块。

> **这条规律的适用范围比测试大得多。** 恢复脚本、维护脚本、任何"应用之外再访问数据库"
> 的代码都该守：**自建 engine，别复用应用那个全局的。**
>
> 我写恢复脚本时也踩了同一个坑，而且更隐蔽 —— 报错发生在**最后一步的校验**里，
> 让我一度以为数据没写进去（其实写进去了）。**错误发生的位置不等于错误造成的位置。**

### 1.3 数据迁移踩到的真 bug：剪枝没有限定目录

`ingest(reset=True)` 会重建 Chroma，并顺带清理 `paper` 表里"已经不存在"的记录
（`PaperRepository.delete_not_in`）。

**第一版的判据是「不在本次报告里的记录一律删」—— 它把用户上传的 3 篇论文删掉了。**

原因：`ingest()` 只扫描 `resource/papers/`（种子语料），而上传的文档在
`resource/uploads/`。这些文档**本来就不该出现在本次报告里**，于是被当成
"已删除的论文"清理掉了。

修法：**剪枝必须限定在被重建的那个目录内**（`path_prefix` 参数）。

> 这个 bug 说明了一件事：**"整库重建"这个动作的语义边界必须写清楚** ——
> 它是"重建**这个目录**"，不是"重建**这个知识库**"。
> 两者在只有种子语料时看不出区别，有了第二种来源之后立刻致命。

数据是可恢复的（源文件都还在 `resource/uploads/`），已完整重建：

```
postgres 里：7 篇 / 636 块，全部 indexed，零失败
```

### 1.4 为什么没上 Alembic

设计文档当时写「改造 #6 换 PostgreSQL 时应该同时引入 Alembic」。真做下来**先不上**，
理由是：

> Alembic 的价值是**保护那些丢了就没了的数据**。
> 而这里的 `paper` 表完全可重建 —— 重跑一次导入就回来了（PDF 都在）。

代价要说清楚：现在靠的是 `_add_missing_columns()` 这个手写的小工具，
**它只覆盖「加一个可空列」这一种情况**（SQLite/PG 都支持 `ALTER TABLE ADD COLUMN`）。
改类型、删列、建索引它都做不了。

什么时候该上：**当数据不再是可重建的时候**（有了真实用户的笔记、标注、会话历史）。
那时 Alembic 才有它该保护的东西。

### 1.5 默认库仍然是 SQLite

`.env` 的默认值改成了 Postgres（本机就是这个配置），但 SQLite 保留为**一条可选路径**：

- 不需要 Docker 就能跑测试
- 定式：`.env.example` 里三种 URL 都列了（SQLite / Postgres 宿主机 / Postgres 容器内）

⚠️ 代价是「现在测的是哪个库」变成一个新的漂移来源。所以 `.env.example` 里
把三种写法的**使用场景**都注明了，而不是只列 URL。

---

## 二、#9：Docker Compose

### 2.1 三个必须先说清楚的前提

这不是"一键就绪"，有三件事必须写在明处：

**① Ollama 不进容器，由宿主机提供。**

embedding 用宿主机的 Ollama（`127.0.0.1:11434`）。容器里的 `127.0.0.1` 是容器自己，
所以后端通过 `host.docker.internal` 反着找宿主机。

**为什么不做成自包含**：把 Ollama 塞进容器要 GPU 直通，Windows 上很折腾；
而且它本来就是个该被多个项目共享的服务。**与其假装一键就绪，不如把这个依赖写在明处。**

顺带修了一个真缺口：`OllamaEmbeddings(model=...)` **没有传 `base_url`** ——
项目里定义了 `settings.OLLAMA_BASE_URL`（`ai/llm.py` 的聊天模型用了它），
但 embedding 那一路被忽略了。宿主机上跑没问题，一进容器就指向容器自己。

> 这类"设置项存在但一半代码没读它"的缺口很难发现：不报错、不告警，
> 只在换环境时以「连接被拒绝」的形式出现，而且很容易被误认为是网络问题。

**② 前端 API 地址是构建期烘进 bundle 的。**

Next.js 把 `NEXT_PUBLIC_*` 在 `next build` 时替换成字面量，运行时改环境变量没用。
而且这个地址必须是**浏览器能访问的**（`http://localhost:8001`），不是容器内部的 ——
写 `http://backend:8001` 的话页面能打开、所有请求都会失败（容器名只在容器网络里能解析）。

所以走 build args 传，默认 `http://localhost:8001`。

**③ 不要同时跑宿主机的后端和容器里的后端。**

Chroma 是**嵌入式**的（一个目录，不是服务）。两边挂同一个目录会互相破坏索引，
而且 8001 端口会冲突。切换时先停掉另一边。

### 2.2 为什么用 bind mount 而不是 named volume

`resource/chroma_db`、`resource/uploads`、`resource/models` 都挂宿主机目录。
这样容器一启动就能看到本地已经索引好的 7 篇论文、已上传的 PDF、以及 266MB 的重排模型，
不用在容器里重跑一遍导入。

代价就是上面第 ③ 条。

### 2.3 后端镜像：依赖清单不另抄一份

Dockerfile 里用标准库 `tomllib` 从 `pyproject.toml` 里读依赖：

```dockerfile
RUN python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); \
    print('\n'.join(d['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt
```

抄一份 `requirements.txt` 的话两份必然漂移，而漂移的表现是
**「本地能跑、容器里 import 失败」** —— 排查时要跨两个文件对版本。

先 COPY `pyproject.toml` 再 COPY 代码，是为了让依赖层能被缓存：改代码不会重装依赖。

### 2.4 前端镜像：三阶段 + standalone

| 阶段 | 需要什么 |
|---|---|
| deps | 只要 `package.json` + `pnpm-lock.yaml` → 这一层能缓存 |
| builder | 全部源码 + node_modules → 产出 `.next` |
| runner | 只留运行时那点东西 → 不要源码、不要 devDependencies、不要 pnpm |

分阶段不是为了好看，是三个阶段的**需求互相冲突**。合成一个阶段的话，
最终镜像里会带着源码、pnpm 和几百 MB 的 `node_modules`。

配套加了 `next.config.mjs`（项目之前**根本没有** next.config）开启
`output: 'standalone'` —— 它把这些依赖里真正被用到的那部分单独挑出来，
镜像体积大概从 ~1GB 降到 ~150MB。对本地 `next start` 没有影响。

`pnpm install --frozen-lockfile`：lockfile 和 package.json 不一致时**直接失败**，
而不是悄悄更新 lockfile。镜像构建必须可复现。

### 2.5 在 Docker Hub 不可达的网络里怎么建

本机实测：**Docker daemon 连不上 `auth.docker.io`**（拉公共镜像也要先取 token）：

```
failed to fetch oauth token: Post "https://auth.docker.io/token": dial tcp ... timeout
```

但宿主机的系统代理（`127.0.0.1:7897`）能到 Docker Hub，国内镜像源也能用。
**不用改 Docker Desktop 设置**（那会重启 Docker、把别的项目的容器一起停掉），
用镜像源拉下来再打回官方标签即可：

```powershell
docker pull docker.m.daocloud.io/library/postgres:16-alpine
docker tag  docker.m.daocloud.io/library/postgres:16-alpine postgres:16-alpine
# 同理：python:3.11-slim、node:20-alpine
```

这样 compose 文件里仍然写官方名（**保持可移植**），本机也能构建。
这件事应该写进部署文档 —— 不然换台机器会卡在同一个地方。

---

## 三、验收

### #6（PostgreSQL）

```
建表 / 查询 / get_or_create           零代码改动直接工作
13 篇 636 块迁移完成（7 篇真实 + 测试残留，残留已清）
uploadApiTest.py 在 Postgres 上全绿（退出码 0）
run_eval.py 在 Postgres 上：A/B/C verdict 全 1.0，fixture.stale = false
```

### #9（Docker）—— 待构建完成后补

```
docker compose config         ✅ 语法通过，变量解析正确
docker compose build          进行中
docker compose up + 端到端验收  待做
```

---

## 四、还没做

| 项 | 说明 |
|---|---|
| **Alembic** | 等数据不再可重建时再上（见 1.4） |
| **独立 worker + 队列** | 现在导入任务跑在 API 进程内，进程重启会丢。要跨进程就需要它 |
| **不在 compose 里跑 Ollama** | 有意的，见 2.1 ① |
| **CI** | 没有自动化测试流水线；现在靠手跑那几个测试脚本 |
