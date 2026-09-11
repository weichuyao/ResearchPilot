<#
.SYNOPSIS
    一键把 ResearchPilot 用 Docker 起起来。

.DESCRIPTION
    通常不需要直接运行这个脚本 —— 双击仓库根目录的「启动 Docker.cmd」就行，
    那个 .cmd 会带着 -ExecutionPolicy Bypass 调这里。

    ## 它做四件事

      1. 检查 Docker 是否在跑
      2. 准备基础镜像（本机 docker daemon 连不上 Docker Hub 的认证，
         需要走国内镜像源 —— 见 scripts/README 或 reference/transformation-06-09）
      3. 构建两个镜像
      4. 启动整套并等它就绪

    ## 为什么基础镜像那一步是必须的

    本机实测，docker daemon 直连 Docker Hub 会在取 token 时超时：

        failed to fetch oauth token: Post "https://auth.docker.io/token": ... timeout

    而宿主机的系统代理能到 Docker Hub、国内镜像源也能用。
    比起去改 Docker Desktop 的代理设置（那会重启 Docker，把别的项目的容器一起停掉），
    让基础镜像从镜像源拉更省事。

    镜像名通过根目录 .env 里的 PYTHON_BASE_IMAGE / NODE_BASE_IMAGE / POSTGRES_IMAGE
    传给 compose，所以 Dockerfile 和 compose 里仍然写的是官方名 —— 文件保持可移植。

.PARAMETER NoBuild
    跳过构建，只启动（镜像没变的时候快很多）。

.PARAMETER Rebuild
    强制重新构建（不用缓存）。

.EXAMPLE
    pwsh -File scripts\docker-up.ps1
#>

[CmdletBinding()]
param(
    [switch]$NoBuild,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    $text" -ForegroundColor Yellow }
function Die($text)  { Write-Host ""; Write-Host "!! $text" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. Docker 在跑吗
Step "检查 Docker"
try {
    $server = & docker version --format "{{.Server.Version}}" 2>$null
} catch {
    $server = $null
}
if (-not $server) {
    Die @"
Docker 没有在跑（或者 Docker Desktop 还没启动完）。

请先打开 Docker Desktop，等左下角变成绿色「Engine running」，再试一次。
"@
}
Ok "Docker Desktop 已就绪（server $server）"

# ---------------------------------------------------------------- 2. 基础镜像
Step "准备基础镜像"
# 从根目录 .env 读镜像名（没有就用官方的 —— 那种情况下不需要这里的准备工作）
$imageMap = @{}
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding UTF8) {
        if ($line -match '^\s*(PYTHON_BASE_IMAGE|NODE_BASE_IMAGE|POSTGRES_IMAGE)\s*=\s*(.+)$') {
            $imageMap[$matches[1]] = $matches[2].Trim()
        }
    }
}

if ($imageMap.Count -eq 0) {
    Warn "根目录没有 .env，使用官方镜像名（需要能正常访问 Docker Hub）"
} else {
    Ok "使用 .env 里指定的镜像源："
    foreach ($k in $imageMap.Keys) { Write-Host "       $k = $($imageMap[$k])" }
}

# 逐个确认基础镜像在本地。不在就从镜像源拉 —— 这里用 pull 而不是让 build 自己去拉，
# 是为了把"拉不动"和"构建失败"这两个问题分开报，混在一起很难查。
$needed = @()
if ($imageMap.ContainsKey("PYTHON_BASE_IMAGE"))   { $needed += $imageMap["PYTHON_BASE_IMAGE"] }
if ($imageMap.ContainsKey("NODE_BASE_IMAGE"))     { $needed += $imageMap["NODE_BASE_IMAGE"] }
if ($imageMap.ContainsKey("POSTGRES_IMAGE"))      { $needed += $imageMap["POSTGRES_IMAGE"] }

$local = & docker images --format "{{.Repository}}:{{.Tag}}"
foreach ($img in $needed) {
    if ($local -contains $img) {
        Ok "已在本地: $img"
        continue
    }
    Warn "本地没有 $img，正在拉取（第一次会比较久）..."
    & docker pull $img
    if ($LASTEXITCODE -ne 0) {
        Die "拉取 $img 失败。检查网络，或者换一个镜像源（改根目录 .env 里的那三行）。"
    }
    Ok "拉取完成: $img"
}

# ---------------------------------------------------------------- 3. 构建
if (-not $NoBuild) {
    Step "构建镜像"
    # GIT_REV 传进镜像，让 /health 能回答"线上跑的是哪份代码"。
    # 容器里读不到 .git（它在仓库根目录，不在 backend/ 这个构建上下文里）。
    Push-Location $root
    try {
        $rev = & git rev-parse --short HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $rev) {
            $env:GIT_REV = $rev.Trim()
            Ok "代码版本 GIT_REV=$env:GIT_REV"
        } else {
            Warn "读不到 git 提交号，/health 里的 git_rev 会是 null"
        }
    } catch { }
    Pop-Location

    $buildArgs = @("compose", "build")
    if ($Rebuild) { $buildArgs += "--no-cache" }
    & docker @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Die @"
构建失败。先看上面的错误信息 —— 常见的两类：

  · 依赖装不上     → 检查 backend/requirements.lock（它锁的是实测能用的版本）
  · 基础镜像拉不到 → 检查根目录 .env 里那三行，或者 Docker Desktop 的代理设置

完整的构建日志：docker compose build backend
"@
    }
    Ok "两个镜像都构建完成"
}

# ---------------------------------------------------------------- 4. 启动
Step "启动"
& docker compose up -d
if ($LASTEXITCODE -ne 0) { Die "启动失败，先跑 docker compose logs 看日志。" }

# 等后端健康。/health 会真的去查数据库、确认重排模型加载状态，比"端口开着"可靠得多。
Write-Host ""
Write-Host "    等待服务就绪（后端要连数据库、加载 266MB 的重排模型，约 20~40 秒）"
$deadline = (Get-Date).AddSeconds(120)
$healthy = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:8001/health" -TimeoutSec 5
        if ($r.index) { $healthy = $true; break }
    } catch { }
}

Write-Host ""
& docker compose ps --format "    {{.Name}}`t{{.Status}}`t{{.Ports}}"

if ($healthy) {
    Write-Host ""
    Write-Host "    后端报告：" -ForegroundColor Green
    Write-Host ("      代码版本 : {0}" -f $r.git_rev)
    Write-Host ("      重排模型 : {0}" -f $r.reranker)
    Write-Host ("      知识库   : {0} 篇 / {1} 块" -f $r.index.papers, $r.index.chunks)
}

Write-Host ""
if ($healthy) {
    Write-Host "==> 启动完成" -ForegroundColor Green
} else {
    Write-Host "==> 容器起来了，但后端还没就绪（120 秒超时）" -ForegroundColor Yellow
    Write-Host "    看日志： docker compose logs -f backend" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "    前端     http://localhost:3000"
Write-Host "    后端     http://localhost:8001/docs"
Write-Host "    健康检查 http://localhost:8001/health"
Write-Host ""
Write-Host "    停掉     docker compose down"
Write-Host "    看日志   docker compose logs -f backend"
Write-Host ""
