"""Serving the built web UI from the backend, so FLOW is a single process.

The UI fetches the API with same-origin relative paths (``fetch("/api/...")``), so mounting
the Vite build at ``/`` makes one server enough: no second terminal, no dev proxy, no CORS.
``npm run dev`` still works for UI development (it proxies /api back to this server).

The build is looked up in three places, first hit wins:
  1. ``$FLOW_WEB_DIR``            -- explicit override (packaging, tests)
  2. ``flow/web/``                -- bundled inside an installed wheel
  3. ``<repo>/frontend/dist/``    -- a git clone / editable install (the common case)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_HERE = Path(__file__).resolve()
# .../src/flow/api/webui.py -> parents: [api, flow, src, <repo root>]
_PACKAGED_DIR = _HERE.parents[1] / "web"
_REPO_DIST_DIR = _HERE.parents[3] / "frontend" / "dist"

# Shown instead of the UI when no build is present, rather than a bare 404.
_NO_BUILD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>FLOW — UI not built</title>
<style>
 body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      max-width:44rem;margin:5rem auto;padding:0 1.5rem;color:#22303c}
 code{background:#eef2f5;padding:.15rem .4rem;border-radius:4px;font-size:.9em}
 pre{background:#eef2f5;padding:.9rem 1rem;border-radius:6px;overflow-x:auto}
 a{color:#2f5673}
</style></head><body>
<h1>The API is running — the web UI isn't built</h1>
<p>The backend is up (try <a href="/docs">/docs</a>), but no compiled front-end was found, so
there is nothing to show at this address.</p>
<p>Build it once, then reload this page:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>Developing the UI? Run <code>npm run dev</code> in <code>frontend/</code> instead and use
<a href="http://localhost:5173">localhost:5173</a> for hot reload.</p>
</body></html>
"""


def web_dir() -> Path | None:
    """Return the directory holding the built UI, or None if it hasn't been built."""
    override = os.environ.get("FLOW_WEB_DIR", "").strip()
    candidates = [Path(override).expanduser()] if override else [_PACKAGED_DIR, _REPO_DIST_DIR]
    for cand in candidates:
        if (cand / "index.html").is_file():
            return cand
    return None


def mount_web_ui(app: FastAPI) -> Path | None:
    """Mount the built UI at ``/``. Call this AFTER every /api route is registered.

    Starlette matches routes in registration order, so the catch-all mount here never
    shadows /api/*, /docs, or /openapi.json. Returns the directory served, or None.
    """
    directory = web_dir()
    if directory is None:
        @app.get("/", include_in_schema=False)
        def _no_build() -> HTMLResponse:
            return HTMLResponse(_NO_BUILD_HTML, status_code=200)

        return None

    # html=True serves index.html for "/" and for unknown paths, which is what a
    # single-page app needs.
    app.mount("/", StaticFiles(directory=str(directory), html=True), name="web")
    return directory
