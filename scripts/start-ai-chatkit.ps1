[CmdletBinding()]
param(
    [switch]$RebuildFrontend,
    [switch]$NoBrowser,
    [string]$Browser = 'edge'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot 'backend'
$frontendDir = Join-Path $projectRoot 'frontend'
$backendPython = Join-Path $backendDir '.venv-py311\Scripts\python.exe'

# ---------------------------------------------------------------------------
# Let local services bypass the Windows system proxy.
#
# The backend reaches Ollama on 127.0.0.1:11434 through httpx
# (langchain-ollama -> ollama -> httpx). httpx reads the Windows system proxy
# from the registry, but it does NOT honour that proxy's "ProxyOverride"
# bypass list. Loopback requests are therefore handed to the proxy, which
# answers HTTP 502, and every RAG embedding call fails.
#
# Declaring NO_PROXY makes httpx connect directly to loopback, while traffic
# to external hosts (for example the LLM API) still goes through the proxy.
#
# These variables are inherited by every process started further down, so no
# application source file has to be changed. Any NO_PROXY value that already
# exists is preserved.
# ---------------------------------------------------------------------------
$loopbackHosts = @('127.0.0.1', 'localhost', '::1')
$noProxyEntries = @()
if ($env:NO_PROXY) { $noProxyEntries += ($env:NO_PROXY -split ',') }
if ($env:no_proxy) { $noProxyEntries += ($env:no_proxy -split ',') }
$noProxyEntries = $noProxyEntries |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ }
$env:NO_PROXY = (($noProxyEntries + $loopbackHosts) | Select-Object -Unique) -join ','
$env:no_proxy = $env:NO_PROXY
Write-Host "Local proxy bypass: NO_PROXY=$($env:NO_PROXY)"

function Test-LocalPort {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne(1000)) { return $false }
        $client.EndConnect($connection)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-ForLocalPort {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 30,
        [string]$ServiceName
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-LocalPort $Port) { return }
        Start-Sleep -Milliseconds 500
    }
    throw "$ServiceName did not start on port $Port within $TimeoutSeconds seconds."
}

function Test-LoopbackHttp {
    param([string]$Python)

    # Reproduce exactly what the backend's embedding client does: an httpx
    # request to the loopback Ollama endpoint. Returns the HTTP status code as
    # a string, or $null when the request could not be completed at all.
    #
    # The URL is passed as a command-line argument on purpose: PowerShell
    # strips the double quotes out of an inline -c payload, which would turn
    # the probe itself into a SyntaxError and always report a failure.
    $probe = 'import sys,httpx;sys.stdout.write(str(httpx.get(sys.argv[1],timeout=10).status_code))'
    $result = & $Python -c $probe 'http://127.0.0.1:11434/api/tags' 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    if (-not $result) { return $null }
    return "$result".Trim()
}

if (-not (Test-Path $backendPython)) {
    throw "Backend environment is missing: $backendPython"
}

# ---------------------------------------------------------------------------
# Open the frontend in an explicitly chosen browser.
#
# Start-Process on a URL goes through ShellExecute, which resolves the Windows
# default handler for the http/https scheme. Third-party clients register
# themselves as browser clients and take that association over -- Quark
# (quark.exe --brand-clouddrive), Doubao, Lenovo SLBrowser and friends all do
# it -- so a plain Start-Process 'http://...' ends up in whichever of them last
# won the association.
#
# Launching a concrete browser executable makes this script independent of the
# machine's default-browser setting, so a hijack cannot change where the
# frontend opens. Choose with -Browser edge|chrome|quark, or pass a full path
# to any .exe. -NoBrowser (or -Browser none) skips opening a window entirely.
# ---------------------------------------------------------------------------
function Get-BrowserCandidates {
    param([string]$Name)

    $programFiles = $env:ProgramFiles
    $programFilesX86 = ${env:ProgramFiles(x86)}
    $localAppData = $env:LOCALAPPDATA

    $candidates = @()
    switch ($Name.ToLowerInvariant()) {
        'edge' {
            if ($programFilesX86) { $candidates += Join-Path $programFilesX86 'Microsoft\Edge\Application\msedge.exe' }
            if ($programFiles) { $candidates += Join-Path $programFiles 'Microsoft\Edge\Application\msedge.exe' }
        }
        'chrome' {
            if ($localAppData) { $candidates += Join-Path $localAppData 'Google\Chrome\Application\chrome.exe' }
            if ($programFiles) { $candidates += Join-Path $programFiles 'Google\Chrome\Application\chrome.exe' }
            if ($programFilesX86) { $candidates += Join-Path $programFilesX86 'Google\Chrome\Application\chrome.exe' }
        }
        'quark' {
            $candidates += 'D:\App\Quark\quark.exe'
        }
        default {
            $candidates += $Name
        }
    }
    return $candidates
}

function Resolve-BrowserExecutable {
    param([string]$Name)

    foreach ($candidate in Get-BrowserCandidates -Name $Name) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

$ollama = (Get-Command ollama -ErrorAction Stop).Source
if (-not (Test-LocalPort 11434)) {
    Write-Host 'Starting Ollama...'
    Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
    Wait-ForLocalPort -Port 11434 -ServiceName 'Ollama'
}

$installedModels = (& $ollama list | Out-String)
if ($installedModels -notmatch '(?m)^bge-m3(?::|\s)') {
    throw 'bge-m3 is not installed. Run: ollama pull bge-m3'
}

# Confirm with the real client library that the bypass actually works here.
$loopbackStatus = Test-LoopbackHttp -Python $backendPython
if ($loopbackStatus -eq '200') {
    Write-Host 'Loopback check: httpx reaches Ollama directly. OK' -ForegroundColor Green
}
else {
    if (-not $loopbackStatus) { $loopbackStatus = 'no response' }
    Write-Warning ("httpx could not reach Ollama on 127.0.0.1:11434 (result: {0}). RAG embeddings will fail with HTTP 502. Turn the system proxy off, add 127.0.0.1 to its bypass list, or set NO_PROXY manually." -f $loopbackStatus)
}

if (-not (Test-LocalPort 8002)) {
    Write-Host 'Starting backend at http://localhost:8002 ...'
    # 必须经 run_server.py 启动：Windows 上裸 `python -m uvicorn` 是 Proactor
    # 循环，psycopg 异步（checkpointer）连不上，30 秒 PoolTimeout 后退出。
    # run_server.py 在 uvicorn.run 之前设 Selector 策略（见 backend/run_server.py）。
    Start-Process -FilePath $backendPython `
        -ArgumentList 'run_server.py' `
        -WorkingDirectory $backendDir `
        -WindowStyle Hidden
    Wait-ForLocalPort -Port 8002 -ServiceName 'Backend'
}
else {
    Write-Host 'Backend already listening on 8002; its environment is left untouched.' -ForegroundColor Yellow
    Write-Host 'If it was started before this proxy fix, restart it (stop-ai-chatkit.ps1, then this script) so NO_PROXY applies.' -ForegroundColor Yellow
}

$nextCli = Join-Path $frontendDir 'node_modules\.bin\next.cmd'
if (-not (Test-Path $nextCli)) {
    throw "Frontend dependencies are missing: $nextCli. Run pnpm install from the frontend directory once."
}

$nextBuildId = Join-Path $frontendDir '.next\BUILD_ID'
if ($RebuildFrontend -or -not (Test-Path $nextBuildId)) {
    Write-Host 'Building frontend...'
    Push-Location $frontendDir
    try {
        & $nextCli build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-LocalPort 3000)) {
    Write-Host 'Starting frontend at http://localhost:3000 ...'
    Start-Process -FilePath $nextCli -ArgumentList 'start' -WorkingDirectory $frontendDir -WindowStyle Hidden
    Wait-ForLocalPort -Port 3000 -ServiceName 'Frontend'
}

Write-Host ''
Write-Host 'AI ChatKit is ready.' -ForegroundColor Green
Write-Host 'Frontend: http://localhost:3000'
Write-Host 'Backend:  http://localhost:8002/docs'

if (-not $NoBrowser -and $Browser -ne 'none') {
    $browserExecutable = Resolve-BrowserExecutable -Name $Browser
    if ($browserExecutable) {
        Write-Host "Opening http://localhost:3000 in $browserExecutable"
        Start-Process -FilePath $browserExecutable -ArgumentList 'http://localhost:3000'
    }
    else {
        # Deliberately NOT falling back to Start-Process <url>: that is the
        # system default handler, which is how a hijacked association (Quark
        # and similar) gets to answer this request in the first place.
        Write-Warning "Browser '$Browser' was not found. Tried:"
        foreach ($candidate in Get-BrowserCandidates -Name $Browser) {
            Write-Warning "  $candidate"
        }
        Write-Warning 'Open http://localhost:3000 manually, or pass a full .exe path with -Browser.'
    }
}
