"""Upload validation and dataset profiling.

Validates the files a lab member uploads and produces a lightweight *profile* (row
counts, columns, recognized file kinds) for display and for the agent's dataset
description. This is structural validation only — it NEVER interprets file contents as
analysis instructions, and it imposes no scientific meaning on any column or channel.

It also provides path-traversal-safe filename handling for the upload API.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Recognized project files. Cytometry data is either a combined event table or a
# directory of FCS files — both are accepted; neither is interpreted.
KNOWN_FILES = {
    "metadata.json",
    "flow.csv",
    "alc.csv",
    "config.yaml",
    "true_lab_results.csv",
}

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB per file (API also enforces)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class UnsafePathError(ValueError):
    """Raised when an uploaded filename would escape the project directory."""


def safe_filename(name: str) -> str:
    """Return a traversal-safe basename, or raise UnsafePathError.

    Rejects absolute paths, ``..`` components, path separators, and any character
    outside ``[A-Za-z0-9._-]``. Uploaded filenames must be plain basenames.
    """
    if not name or name in {".", ".."}:
        raise UnsafePathError(f"Unsafe filename: {name!r}")
    if "/" in name or "\\" in name or ".." in name:
        raise UnsafePathError(f"Unsafe filename (path separators not allowed): {name!r}")
    if not _SAFE_NAME.match(name):
        raise UnsafePathError(
            f"Filename {name!r} contains characters that are not allowed."
        )
    return name


@dataclass
class FileReport:
    """Validation result for one file (or the FCS directory)."""

    name: str
    kind: str  # metadata|timepoints|alc|event_table|fcs_dir|config|ground_truth|unknown
    ok: bool
    rows: int | None = None
    columns: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ProjectProfile:
    """Aggregate profile of a project directory."""

    files: list[FileReport] = field(default_factory=list)
    ok: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "notes": self.notes,
            "files": [fr.__dict__ for fr in self.files],
        }

    def dataset_description(self) -> str:
        """Build a neutral, structural description of the dataset for the agent.

        Lists files, shapes, and columns only — no analysis hints whatsoever.
        """
        lines = ["The dataset directory contains the following files:"]
        for fr in self.files:
            if not fr.ok:
                lines.append(f"  - {fr.name}: (could not parse: {fr.detail})")
                continue
            if fr.columns:
                cols = ", ".join(fr.columns)
                rows = f"{fr.rows} rows" if fr.rows is not None else ""
                lines.append(f"  - {fr.name} [{fr.kind}] {rows}; columns: {cols}")
            else:
                lines.append(f"  - {fr.name} [{fr.kind}] {fr.detail}".rstrip())
        lines.append(
            "\nExperiment facts (which detector carries which stain, key dates, etc.) "
            "are in metadata.json. Inspect every file before interpreting it."
        )
        return "\n".join(lines)


def _read_csv_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        count = sum(1 for _ in reader)
    return [h.strip() for h in header], count


def _validate_csv_columns(
    path: Path, kind: str, required: set[str]
) -> FileReport:
    try:
        cols, rows = _read_csv_header_and_count(path)
    except Exception as e:
        return FileReport(path.name, kind, ok=False, detail=str(e))
    lower = {c.lower() for c in cols}
    missing = required - lower
    if missing:
        return FileReport(
            path.name,
            kind,
            ok=False,
            columns=cols,
            rows=rows,
            detail=f"missing required column(s): {sorted(missing)}",
        )
    return FileReport(path.name, kind, ok=True, columns=cols, rows=rows)


def validate_file(path: Path) -> FileReport:
    """Validate a single known file structurally."""
    name = path.name
    if name == "metadata.json":
        try:
            data = json.loads(path.read_text())
            keys = list(data.keys()) if isinstance(data, dict) else []
            return FileReport(name, "metadata", ok=True, columns=keys,
                              detail=f"{len(keys)} fields")
        except Exception as e:
            return FileReport(name, "metadata", ok=False, detail=str(e))
    if name == "flow.csv":
        return _validate_csv_columns(path, "timepoints", {"label", "date"})
    if name == "alc.csv":
        return _validate_csv_columns(path, "alc", {"date", "alc"})
    if name == "true_lab_results.csv":
        try:
            cols, rows = _read_csv_header_and_count(path)
            return FileReport(name, "ground_truth", ok=True, columns=cols, rows=rows,
                              detail="evaluation only; never read during analysis")
        except Exception as e:
            return FileReport(name, "ground_truth", ok=False, detail=str(e))
    if name == "config.yaml":
        from flow.config import ConfigError, load_config

        try:
            load_config(path)
            return FileReport(name, "config", ok=True, detail="valid, analysis-free")
        except ConfigError as e:
            return FileReport(name, "config", ok=False, detail=str(e))
    # Cytometry event table (combined per-event CSV/parquet).
    if path.suffix.lower() in {".csv", ".tsv", ".parquet"}:
        if path.suffix.lower() == ".parquet":
            return FileReport(name, "event_table", ok=True,
                              detail="parquet event table (columns read at runtime)")
        try:
            cols, rows = _read_csv_header_and_count(path)
            return FileReport(name, "event_table", ok=True, columns=cols, rows=rows)
        except Exception as e:
            return FileReport(name, "event_table", ok=False, detail=str(e))
    if path.suffix.lower() == ".fcs":
        return FileReport(name, "fcs", ok=True, detail="FCS file (parsed at runtime)")
    return FileReport(name, "unknown", ok=True, detail="accepted; not interpreted")


def profile_project(project_dir: str | Path) -> ProjectProfile:
    """Validate and profile every file in a project directory."""
    root = Path(project_dir)
    profile = ProjectProfile()
    if not root.exists():
        profile.ok = False
        profile.notes.append(f"Project directory not found: {root}")
        return profile

    entries = sorted(p for p in root.iterdir())
    fcs_files = [p for p in entries if p.is_file() and p.suffix.lower() == ".fcs"]
    fcs_subdirs = [p for p in entries if p.is_dir()]

    for p in entries:
        if p.is_dir():
            fcs_in_dir = list(p.glob("*.fcs"))
            if fcs_in_dir:
                profile.files.append(
                    FileReport(
                        p.name + "/",
                        "fcs_dir",
                        ok=True,
                        rows=len(fcs_in_dir),
                        detail=f"directory of {len(fcs_in_dir)} FCS file(s)",
                    )
                )
            continue
        profile.files.append(validate_file(p))

    if fcs_files:
        profile.notes.append(
            f"Found {len(fcs_files)} FCS file(s) alongside the sample sheet."
        )

    # Soft guidance (not enforcement): note missing recommended files.
    present = {fr.name for fr in profile.files}
    if "metadata.json" not in present:
        profile.notes.append("metadata.json not found — the agent will lack experiment facts.")
    has_events = any(
        fr.kind in {"event_table", "fcs_dir"} or fr.kind == "fcs" for fr in profile.files
    ) or bool(fcs_files)
    if not has_events:
        profile.notes.append("No cytometry data (event table or FCS files) detected.")

    profile.ok = all(fr.ok for fr in profile.files)
    return profile
