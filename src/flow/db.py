"""SQLite persistence for projects, consensus batches, and trajectories (stdlib only).

A **batch** is one consensus run: N independent trajectories analyzing the same data, plus
a meta-analysis that synthesizes their conclusions. Each **trajectory** is a single agent
run with fully isolated state (its own artifact dir + Docker container).

Only metadata is stored here so the UI's history survives restart; notebooks, logs, plots,
and the consensus text live on the filesystem (paths from env). No analysis logic is stored.

Storage roots (env-overridable):
  FLOW_DATA_ROOT       — project uploads        (default: ./var/projects)
  FLOW_ARTIFACTS_ROOT  — batch artifact dirs     (default: ./var/runs)
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
            CREATE TABLE IF NOT EXISTS batches (
                id              TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL,
                question        TEXT NOT NULL,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                n_trajectories  INTEGER NOT NULL,
                status          TEXT NOT NULL,
                created_at      REAL NOT NULL,
                finished_at     REAL,
                consensus_ok    INTEGER DEFAULT 0,
                n_submitted     INTEGER DEFAULT 0,
                failure_reason  TEXT,
                artifact_dir    TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS trajectories (
                id             TEXT PRIMARY KEY,
                batch_id       TEXT NOT NULL,
                idx            INTEGER NOT NULL,
                status         TEXT NOT NULL,
                submitted      INTEGER DEFAULT 0,
                failure_reason TEXT,
                steps          INTEGER DEFAULT 0,
                artifact_dir   TEXT NOT NULL,
                created_at     REAL NOT NULL,
                finished_at    REAL,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
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
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return dict(row) if row else None


def list_projects() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_project(project_id: str) -> None:
    """Remove a project and all its batches/trajectories from the DB.

    On-disk files (the project dir and run artifacts) are removed by the API layer.
    """
    with _connect() as conn:
        batch_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM batches WHERE project_id = ?", (project_id,)
            ).fetchall()
        ]
        for bid in batch_ids:
            conn.execute("DELETE FROM trajectories WHERE batch_id = ?", (bid,))
        conn.execute("DELETE FROM batches WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


# ----------------------------------------------------------------- batches
def create_batch(
    batch_id: str,
    project_id: str,
    question: str,
    provider: str,
    model: str,
    n_trajectories: int,
    artifact_dir: str,
) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO batches
               (id, project_id, question, provider, model, n_trajectories, status,
                created_at, artifact_dir)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (batch_id, project_id, question, provider, model, n_trajectories,
             time.time(), artifact_dir),
        )
    return get_batch(batch_id)


def update_batch(batch_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE batches SET {cols} WHERE id = ?", [*fields.values(), batch_id])


def get_batch(batch_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return dict(row) if row else None


def list_batches(project_id: Optional[str] = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if project_id:
            rows = conn.execute(
                "SELECT * FROM batches WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------- trajectories
def create_trajectory(
    traj_id: str, batch_id: str, idx: int, artifact_dir: str
) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO trajectories
               (id, batch_id, idx, status, artifact_dir, created_at)
               VALUES (?, ?, ?, 'pending', ?, ?)""",
            (traj_id, batch_id, idx, artifact_dir, time.time()),
        )
    return get_trajectory(traj_id)


def update_trajectory(traj_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE trajectories SET {cols} WHERE id = ?", [*fields.values(), traj_id]
        )


def get_trajectory(traj_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM trajectories WHERE id = ?", (traj_id,)
        ).fetchone()
    return dict(row) if row else None


def list_trajectories(batch_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trajectories WHERE batch_id = ? ORDER BY idx ASC", (batch_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_batch(batch_id: str) -> None:
    """Remove a batch and its trajectory rows from the DB (artifacts deleted separately)."""
    with _connect() as conn:
        conn.execute("DELETE FROM trajectories WHERE batch_id = ?", (batch_id,))
        conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
