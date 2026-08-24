#!/bin/bash
# Double-click this file to start FLOW (macOS). On Linux, run: ./"Start FLOW.command"
#
# It does everything the old two-terminal dance did, in one window:
#   1. installs FLOW into .venv/ the first time
#   2. checks that Docker Desktop is running
#   3. starts the web app + API on http://localhost:8000 and opens your browser
#
# Ctrl+C in this window (or closing it) stops FLOW.

set -u
cd "$(dirname "$0")" || exit 1

# A double-clicked window vanishes on exit, taking the error message with it.
die() {
    printf '\n  %s\n\n' "$1"
    printf '  Press Return to close this window. '
    read -r _
    exit 1
}

printf '\n  FLOW\n  ====\n\n'

if [ ! -x .venv/bin/flow ]; then
    PY=""
    for cand in python3 python3.12 python3.11 python; do
        if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
    [ -n "$PY" ] || die "Python 3 isn't installed. Get it from https://www.python.org/downloads/ and double-click this file again."

    printf '  First-time setup — this takes a few minutes. Leave the window open.\n\n'
    "$PY" -m venv .venv || die "Couldn't create the Python environment (.venv)."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install -e ".[dev]" || die "Install failed. Check your internet connection and try again."
    printf '\n  Setup done.\n\n'
fi

# Docker runs every analysis, so a stopped Docker means runs fail later, not now.
if ! docker info >/dev/null 2>&1; then
    printf '  Docker Desktop is not running.\n'
    printf '  Open the Docker Desktop app, wait for the whale icon to stop animating,\n'
    printf '  then double-click this file again.\n\n'
    printf '  (Continuing anyway — you can browse the app, but runs will not start.)\n\n'
fi

exec .venv/bin/flow serve
