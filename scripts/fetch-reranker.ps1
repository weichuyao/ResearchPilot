<#
.SYNOPSIS
    下载交叉编码器重排模型（bge-reranker-base 的 int8 ONNX 版本）。

.DESCRIPTION
    模型权重是二进制，不进 git（见 .gitignore 的「模型权重」一节）。
    全新 clone 出来的仓库需要跑一次这个脚本，否则 ai/rag/rerank.py 会静默降级成
    纯混合检索 —— 系统能跑，但没有重排。

    为什么不用 huggingface.co：本机连 huggingface.co 会超时（实测 15s ConnectTimeout），
    所以走 hf-mirror.com 镜像（实测 3.2 MB/s，266 MB 约 90 秒）。

    为什么是 ONNX 而不是 sentence-transformers：环境里已经有 onnxruntime /
    tokenizers / numpy（随 Chroma 装进来的），但没有 torch。装 torch 要下 2.5 GB。

.PARAMETER ModelDir
    模型存放目录。默认 backend/resource/models/bge-reranker-base-int8。

.PARAMETER Python
    python.exe 的路径。默认用 backend/.venv-py311/Scripts/python.exe。

.EXAMPLE
    pwsh -File scripts/fetch-reranker.ps1
#>

[CmdletBinding()]
param(
    [string]$ModelDir,
    [string]$Python
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $ModelDir) { $ModelDir = Join-Path $root "backend\resource\models\bge-reranker-base-int8" }
if (-not $Python)   { $Python   = Join-Path $root "backend\.venv-py311\Scripts\python.exe" }

if (-not (Test-Path $Python)) {
    Write-Error "找不到 python：$Python`n用 -Python 参数指定实际在用的解释器。"
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

# 用镜像。huggingface_hub 认这个环境变量，不用改调用代码。
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 只下需要的文件：int8 权重 + 分词器 + 配置。
# 同一个仓库里还有 fp32 (1.1 GB) / fp16 (557 MB) / q4 等版本，都不下。
$files = @("onnx/model_int8.onnx", "tokenizer.json", "config.json", "tokenizer_config.json")

Write-Host "下载 bge-reranker-base (int8) 到 $ModelDir"
Write-Host "镜像: $env:HF_ENDPOINT"
Write-Host ""

$script = @"
from huggingface_hub import hf_hub_download
for f in $($files | ForEach-Object { "'$_'" } | Join-String -Separator ', '):
    path = hf_hub_download('Xenova/bge-reranker-base', f, local_dir=r'$ModelDir')
    print('  ok', f)
"@

& $Python -c $script
if ($LASTEXITCODE -ne 0) { Write-Error "下载失败（exit $LASTEXITCODE）" }

Write-Host ""
Write-Host "已下载："
Get-ChildItem -Recurse $ModelDir -File |
    Where-Object { $_.FullName -notmatch '\\\.cache\\' } |
    ForEach-Object { "  {0,10:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Replace($ModelDir, "") }

Write-Host ""
Write-Host "自检："
$verify = @"
import sys
sys.path.insert(0, 'app')
from ai.rag.rerank import available, score_pairs
print('  available() ->', available())
if available():
    s = score_pairs('什么是 hard positive problem?', [
        'The hard positive problem refers to samples of the same identity that look very different.',
        'Bring the water to a boil, then add the pork belly and simmer for two hours.',
    ])
    print('  相关段落 %.3f / 无关段落 %.3f  ->  %s' % (
        s[0], s[1], 'OK' if s[0] > s[1] else '排序异常！'))
"@

Push-Location (Join-Path $root "backend")
try { & $Python -c $verify } finally { Pop-Location }
