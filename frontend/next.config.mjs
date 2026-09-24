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
 * Windows 本地构建默认关闭 standalone：pnpm 的依赖树需要目录符号链接，
 * 普通 Windows 终端通常没有创建权限。Docker 构建运行在 Linux 中，仍会
 * 自动启用 standalone；也可用 NEXT_ENABLE_STANDALONE=1 显式强制开启。
 */
const useStandalone =
  process.env.NEXT_ENABLE_STANDALONE === '1' ||
  (process.platform !== 'win32' && process.env.NEXT_DISABLE_STANDALONE !== '1');

const nextConfig = {
  ...(useStandalone ? { output: 'standalone' } : {}),
};

export default nextConfig;
