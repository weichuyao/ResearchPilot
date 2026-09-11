/** @type {import('next').NextConfig} */

/**
 * 这一份配置是**做 Docker 时加的**（改造 #9），之前项目根本没有 next.config。
 *
 * ## output: 'standalone' 解决什么
 *
 * 默认的 `next build` 产物需要**完整的 node_modules** 才能跑 `next start` ——
 * 那是几百 MB。standalone 模式会把这些依赖里真正被用到的那一部分单独挑出来，
 * 放进 `.next/standalone`，配一个 `server.js` 直接 `node server.js` 就能起。
 *
 * 镜像体积大概从 ~1GB 降到 ~150MB。代价是 Dockerfile 要按 standalone 的目录结构
 * 来 COPY（见 frontend/Dockerfile）。
 *
 * ## 对本地开发没有影响
 *
 * `next start` 照常工作 —— standalone 只是**额外**多产出一份
 * `.next/standalone`，不改变原来的产物。所以 scripts/start-ai-chatkit.ps1
 * 那条路径不用动。
 */
const nextConfig = {
  output: 'standalone',
};

export default nextConfig;
