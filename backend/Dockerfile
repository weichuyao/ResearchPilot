# ResearchPilot 后端镜像（改造 #9）
#
# ## 为什么是 3.11 而不是 pyproject 里原来写的 3.13
#
# `pyproject.toml` 当时声明的是 `requires-python = ">=3.13"`，而本机实际跑的是 3.11
# （`.venv-py311`）。这个不一致拖了很久，Docker 逼着必须定下来。
# 选 **3.11**，因为它是**已知全部依赖可用的版本** —— 之前所有实测都跑在它上面。
# pyproject 已同步改成 >=3.11（2026-09 又把那个 3.13 的 venv 删掉了，
# 详见 reference/design-decisions.md 的「环境版本统一（3.11）」一节）。
#
# ## 为什么用 requirements.lock 而不是从 pyproject 现解析
#
# 第一次构建就是这么干的（用 tomllib 从 pyproject 读依赖列表然后 pip install），
# **结果失败了**：
#
#     ERROR: Cannot install -r requirements.txt (line 4) and fastapi>=0.115.12
#            because these package versions have conflicting dependencies.
#
# 第 4 行是 chromadb>=0.6.3 —— 全新解析挑到了 chromadb 1.x，和 fastapi 撞了。
# 本机 venv 一直能用，是因为它是更早某个时间点解析出来的。
# **宽松约束 + 无锁文件 = 不可复现安装**，这就是活生生的例子。
#
# 详细的取舍写在 requirements.lock 的文件头里。

# 基础镜像做成构建参数：本机 docker daemon 连不上 Docker Hub 的认证
# （`auth.docker.io` 超时），但能连国内镜像源。做成参数之后，只要在根目录
# .env 里写一行 PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
# 就能直接构建 —— 不用先 `docker pull` 再 `docker tag` 手动打标签。
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# 当前代码的 git 提交号，给 /health 用。
#
# 容器里读不到 `.git` —— 它在仓库根目录，而构建上下文是 backend/，不在里面。
# 不传的话 /health 会报 null，恰好丢掉它最想回答的那个问题（"线上跑的是哪份代码"）。
# compose 会用宿主机的 GIT_REV 环境变量传进来，没设就留空（退回读 .git，容器里读不到）。
ARG GIT_REV=
ENV GIT_REV=${GIT_REV}

# - PYTHONUNBUFFERED：让日志立刻出来。不加的话 print/logging 会被缓冲，
#   容器日志看起来"卡住不动"，排查时非常误导。
# - PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl 给 healthcheck 用
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先只 COPY 依赖清单再装依赖：这样改代码不会让依赖层失效，
# 重建时能命中缓存（Docker 的层缓存按文件内容算）
COPY requirements.lock ./
RUN pip install -r requirements.lock

# 再 COPY 代码。.dockerignore 排除了 venv / 运行时数据 / .env
COPY app ./app
COPY resource/papers ./resource/papers
COPY tests ./tests

# 运行时数据目录先建好，compose 会把 volume 挂到这些位置。
# 不建的话首次启动时应用可能因为目录不存在而失败。
RUN mkdir -p resource/chroma_db resource/uploads resource/models

EXPOSE 8001

# 健康检查用 /health 而不是"端口能连上"：那个接口会真的去查数据库、
# 确认重排模型加载状态、统计索引内容。端口开着但数据库连不上时，
# 前者会挂、后者不会 —— 而我们要的是前者。
HEALTHCHECK --interval=15s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8001/health || exit 1

# --host 0.0.0.0：容器里必须监听所有网卡，绑 127.0.0.1 的话
# 端口映射进来也连不上（这是容器网络最常见的坑之一）
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "app", \
     "--host", "0.0.0.0", "--port", "8001"]
