#!/usr/bin/env python3
"""Run FLOW-Jo into a discoverable, write-once project-local directory.

Example
-------
python scripts/run_project_first_run.py \
  --project /path/to/UPN27 \
  --data /path/to/UPN27/analysis/flow/run_inputs/input_snapshot \
  --run-id upn27_flowjo_v10_organized
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def project_run_dir(project_root: str | Path, run_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in str(run_id)).strip("_")
    if not safe:
        raise ValueError("run_id must contain at least one letter or number")
    return Path(project_root).resolve() / "analysis" / "flow" / "runs" / safe / "first_run"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Patient/project root, for example UPN27.")
    parser.add_argument("--data", required=True, help="Frozen first-run input snapshot.")
    parser.add_argument("--run-id", required=True, help="Stable, human-readable run identifier.")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    data = Path(args.data).resolve()
    if not project.is_dir():
        raise SystemExit(f"project directory not found: {project}")
    if not data.is_dir():
        raise SystemExit(f"input snapshot not found: {data}")
    output = project_run_dir(project, args.run_id)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to replace non-empty project run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "project_run_location.json").write_text(json.dumps({
        "schema_version": "flow.project_run_location.v1",
        "project_root": str(project),
        "run_id": args.run_id,
        "first_run_dir": str(output),
        "layout": "<project>/analysis/flow/runs/<run_id>/first_run",
        "write_policy": "new run id required for a changed canonical result",
    }, indent=2, sort_keys=True))

    repo = Path(__file__).resolve().parents[1]
    runner = repo / "src" / "flow" / "firstrun" / "anchored_nk_panel.py"
    command = [
        sys.executable,
        str(runner),
        "--data", str(data),
        "--out", str(output),
        "--plots", str(output / "outputs" / "plots"),
    ]
    completed = subprocess.run(command, cwd=repo, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"PROJECT_FIRST_RUN_OK: {output}")


if __name__ == "__main__":
    main()
