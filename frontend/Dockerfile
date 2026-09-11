# ResearchPilot 前端镜像（改造 #9）
#
# 三阶段构建。分阶段的目的不是"显得专业"，是三个阶段的**需求互相冲突**：
#
#   deps    只需要 package.json + lockfile  →  这一层能被缓存，改代码不用重装依赖
#   builder 需要全部源码 + node_modules     →  产物是 .next
#   runner  只需要运行时的那点东西          →  不要源码、不要 devDependencies、不要 pnpm
#
# 合成一个阶段的话，最终镜像里会带着源码、pnpm、dev 依赖 —— 以及几百 MB 的 node_modules。

# ---------------------------------------------------------------- deps
# 基础镜像做成构建参数：本机 docker daemon 连不上 Docker Hub 的认证
# （auth.docker.io 超时），但能连国内镜像源。做成参数之后只要在根目录 .env
# 里写 NODE_BASE_IMAGE=docker.m.daocloud.io/library/node:20-alpine 就行，
# 不用先 pull 再手动打标签。三个阶段共用同一个参数。
ARG BASE_IMAGE=node:20-alpine

FROM ${BASE_IMAGE} AS deps
WORKDIR /app

# corepack 是 Node 自带的包管理器转发器，用它拿到和 package.json 里
# packageManager 字段一致的 pnpm 版本，不用在镜像里 `npm i -g pnpm`
RUN corepack enable

COPY package.json pnpm-lock.yaml ./
# --frozen-lockfile：lockfile 和 package.json 不一致时**直接失败**，
# 而不是悄悄更新 lockfile。镜像构建必须可复现，不能容忍"顺手升级了依赖"
RUN pnpm install --frozen-lockfile

# ---------------------------------------------------------------- builder
FROM ${BASE_IMAGE} AS builder
WORKDIR /app

RUN corepack enable

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# ⚠️ 前端 API 地址是**构建期**烘进 bundle 的。
#
# Next.js 会把代码里出现的 `process.env.NEXT_PUBLIC_XXX` 在 `next build` 时
# 替换成字面量 —— 运行时再设环境变量**完全没用**。
#
# 而且这个地址必须是**浏览器能访问的**，不是容器内部的：
# 浏览器跑在宿主机上，它看到的 127.0.0.1 是宿主机自己。
# 写成 http://backend:8001 的话，页面能打开、所有请求都会失败
# （容器名只在容器网络里能解析）。
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

# ---------------------------------------------------------------- runner
FROM ${BASE_IMAGE} AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# 不用 root 跑应用。
#
# ⚠️ `nextjs` 这个用户**基础镜像里没有** —— 它是 Next.js 官方 Dockerfile 示例里
# 自己 addgroup/adduser 建的。第一版我直接写了 `USER nextjs`，构建时炸：
#
#     unable to find user nextjs: no matching entries in passwd file
#
# 所以必须先建出来。用固定的 uid/gid（1001）而不是让系统自动分配：
# 挂载卷的时候宿主机和容器里的 uid 需要能对上，自动分配的 uid 每次重建都可能变。
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

USER nextjs

# standalone 模式的三块产物。缺任何一个都会以很难懂的方式失败：
#   standalone/     server.js + 被挑出来的运行时依赖（含 package.json）
#   static/         静态资源；standalone 不会自动带上它，不拷的话页面没样式、没 JS
#   public/         本项目没有 public 目录，所以不拷
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD node -e "fetch('http://127.0.0.1:3000').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# standalone 的入口是 server.js，不是 `next start`
CMD ["node", "server.js"]
