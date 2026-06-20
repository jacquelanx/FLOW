"""SQLite persistence for projects and runs (stdlib only).

Stores project/run *metadata* so the UI's RunHistory survives refresh/restart. Uploaded
files and run artifacts live on the filesystem (paths from env); this DB only indexes
them. No analysis logic is stored here.

Storage roots (env-overridable):
  FLOW_DATA_ROOT       — project uploads        (default: ./var/projects)
  FLOW_ARTIFACTS_ROOT  — run artifact dirs       (default: ./var/runs)
  FLOW_DB_PATH         — sqlite file             (default: ./var/flow.sqlite3)
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def data_root() -> Path:
    return Path(os.environ.get("FLOW_DATA_ROOT", "var/projects")).resolve()


def artifacts_root() -> Path:
    return Path(os.environ.get("FLOW_ARTIFACTS_ROOT", "var/runs")).resolve()


def db_path() -> Path:
    return Path(os.environ.get("FLOW_DB_PATH", "var/flow.sqlite3")).resolve()


def _connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and storage roots if they do not exist."""
    data_root().mkdir(parents=True, exist_ok=True)
    artifacts_root().mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  REAL NOT NULL,
                path        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id             TEXT PRIMARY KEY,
                project_id     TEXT NOT NULL,
                question       TEXT NOT NULL,
                provider       TEXT NOT NULL,
                model          TEXT NOT NULL,
                status         TEXT NOT NULL,
                created_at     REAL NOT NULL,
                finished_at    REAL,
                submitted      INTEGER DEFAULT 0,
                failure_reason TEXT,
                artifact_dir   TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            """
        )


# ----------------------------------------------------------------- projects
def create_project(project_id: str, name: str) -> dict[str, Any]:
    path = data_root() / project_id
    path.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at, path) VALUES (?, ?, ?, ?)",
            (project_id, name, time.time(), str(path)),
        )
    return get_project(project_id)


def get_project(project_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    return dict(row) if row else None


def list_projects() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------- runs
def create_run(
    run_id: str,
    project_id: str,
    question: str,
    provider: str,
    model: str,
    artifact_dir: str,
) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO runs
               (id, project_id, question, provider, model, status, created_at, artifact_dir)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (run_id, project_id, question, provider, model, time.time(), artifact_dir),
        )
    return get_run(run_id)


def update_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [run_id]
    with _connect() as conn:
        conn.execute(f"UPDATE runs SET {cols} WHERE id = ?", vals)


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(project_id: Optional[str] = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if project_id:
            rows = conn.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]
