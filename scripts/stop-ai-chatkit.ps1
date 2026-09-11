[CmdletBinding()]
param(
    [switch]$StopOllama
)

$ErrorActionPreference = 'Stop'
$ports = @(3000, 8001)
if ($StopOllama) { $ports += 11434 }

function Get-ListeningProcessId {
    param([int]$Port)

    # Preferred: the cmdlet. Some shells (locked-down sessions, restricted
    # sandboxes) cannot read the perf counters behind it and throw
    # "access denied", so fall back to parsing netstat instead of silently
    # reporting that nothing is listening.
    try {
        $fromCmdlet = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique)
        if ($fromCmdlet.Count -gt 0) { return $fromCmdlet }
    }
    catch {
        Write-Verbose "Get-NetTCPConnection unavailable for port ${Port}: $($_.Exception.Message)"
    }

    # A listening socket is identified by an empty peer address, so this
    # matches on the "0.0.0.0:0" / "[::]:0" foreign address rather than on the
    # state word, which can be localised.
    $pattern = '^\s*TCP\s+\S+:' + $Port + '\s+(?:0\.0\.0\.0|\[::\]):0\s+\S+\s+(\d+)\s*$'
    $fromNetstat = @(netstat -ano -p TCP |
        Select-String -Pattern $pattern |
        ForEach-Object { [int]$_.Matches[0].Groups[1].Value } |
        Select-Object -Unique)
    return $fromNetstat
}

$failures = @()

foreach ($port in $ports) {
    $processIds = @(Get-ListeningProcessId -Port $port)

    if ($processIds.Count -eq 0) {
        Write-Host "Nothing is listening on port $port."
        continue
    }

    foreach ($processId in $processIds) {
        $processName = 'unknown process'
        try { $processName = (Get-Process -Id $processId -ErrorAction Stop).ProcessName } catch { }

        try {
            Stop-Process -Id $processId -Confirm:$false -ErrorAction Stop
            Write-Host "Stopped $processName (pid $processId) on port $port."
        }
        catch {
            Write-Warning "Could not stop $processName (pid $processId) on port ${port}: $($_.Exception.Message)"
            $failures += "port ${port}: pid $processId could not be stopped"
        }
    }
}

Start-Sleep -Milliseconds 500

# Verify the result instead of assuming it. A stop script that reports success
# while a stale service keeps running is worse than one that fails loudly: the
# start script then skips that service and leaves its old environment in place.
foreach ($port in $ports) {
    if (@(Get-ListeningProcessId -Port $port).Count -gt 0) {
        $failures += "port $port is still listening"
    }
}

if ($failures.Count -gt 0) {
    Write-Host ''
    Write-Host ("Stop did not complete: {0}." -f ($failures -join '; ')) -ForegroundColor Red
    Write-Host 'Services started from an elevated or protected session must be stopped from an elevated PowerShell window.' -ForegroundColor Yellow
    exit 1
}

Write-Host ''
Write-Host 'All requested ports are free.' -ForegroundColor Green
