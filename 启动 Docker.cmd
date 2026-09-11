@echo off
REM ============================================================================
REM  Double-click this file to start ResearchPilot with Docker.
REM
REM  Why this .cmd exists instead of just a .ps1:
REM    Double-clicking a .ps1 opens it in Notepad instead of running it.
REM    This wrapper runs it with the execution policy bypassed.
REM
REM  Why the messages here are English/ASCII only:
REM    cmd.exe reads .bat/.cmd files using the OEM code page. Non-ASCII text
REM    written as UTF-8 comes out as garbage. The actual script
REM    (scripts\docker-up.ps1) IS UTF-8 with a BOM and can use Chinese safely.
REM
REM  Skip the build next time:  docker-up.cmd -NoBuild
REM ============================================================================

chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "PS1=%ROOT%scripts\docker-up.ps1"

if not exist "%PS1%" (
    echo.
    echo   ERROR: cannot find scripts\docker-up.ps1
    echo   Make sure this file sits in the repository root.
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "CODE=%ERRORLEVEL%"

echo.
if not "%CODE%"=="0" (
    echo   Exited with code %CODE%. Read the messages above.
) else (
    echo   Done. You can close this window.
)
echo.
pause
exit /b %CODE%
