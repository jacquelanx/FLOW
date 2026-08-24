@echo off
rem Double-click this file to start FLOW (Windows).
rem
rem It does everything the old two-terminal dance did, in one window:
rem   1. installs FLOW into .venv\ the first time
rem   2. checks that Docker Desktop is running
rem   3. starts the web app + API on http://localhost:8000 and opens your browser
rem
rem Ctrl+C in this window (or closing it) stops FLOW.

cd /d "%~dp0"
echo.
echo   FLOW
echo   ====
echo.

if not exist ".venv\Scripts\flow.exe" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo   Python 3 isn't installed. Get it from https://www.python.org/downloads/
        echo   During install, tick "Add python.exe to PATH". Then double-click this file again.
        goto :fail
    )
    echo   First-time setup - this takes a few minutes. Leave the window open.
    echo.
    python -m venv .venv || goto :venvfail
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e ".[dev]" || goto :installfail
    echo.
    echo   Setup done.
    echo.
)

rem Docker runs every analysis, so a stopped Docker means runs fail later, not now.
docker info >nul 2>&1
if errorlevel 1 (
    echo   Docker Desktop is not running.
    echo   Open the Docker Desktop app, wait for the whale icon to stop animating,
    echo   then double-click this file again.
    echo.
    echo   (Continuing anyway - you can browse the app, but runs will not start.^)
    echo.
)

".venv\Scripts\flow.exe" serve

rem A double-clicked window closes on exit, so hold it open if FLOW failed to start.
if errorlevel 1 (
    echo.
    pause
)
goto :eof

:venvfail
echo   Couldn't create the Python environment (.venv^).
goto :fail

:installfail
echo   Install failed. Check your internet connection and try again.
goto :fail

:fail
echo.
pause
exit /b 1
