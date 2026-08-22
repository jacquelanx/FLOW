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
    parser.add_argument(
        "--compensation-controls", default="",
        help="Optional directory containing acquisition/reference compensation-control FCS.",
    )
    parser.add_argument(
        "--analysis-contract", default="",
        help="Optional validated analysis-contract JSON to freeze into provenance.",
    )
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
    compensation_controls = (
        Path(args.compensation_controls).resolve() if args.compensation_controls else None
    )
    analysis_contract = Path(args.analysis_contract).resolve() if args.analysis_contract else None
    if compensation_controls is not None and not compensation_controls.is_dir():
        raise SystemExit(f"compensation-control directory not found: {compensation_controls}")
    if analysis_contract is not None and not analysis_contract.is_file():
        raise SystemExit(f"analysis contract not found: {analysis_contract}")
    (output / "project_run_location.json").write_text(json.dumps({
        "schema_version": "flow.project_run_location.v1",
        "project_root": str(project),
        "run_id": args.run_id,
        "first_run_dir": str(output),
        "layout": "<project>/analysis/flow/runs/<run_id>/first_run",
        "write_policy": "new run id required for a changed canonical result",
        "compensation_control_root": (
            str(compensation_controls) if compensation_controls is not None else None
        ),
        "analysis_contract": str(analysis_contract) if analysis_contract is not None else None,
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
    if compensation_controls is not None:
        command.extend(["--compensation-controls", str(compensation_controls)])
    if analysis_contract is not None:
        command.extend(["--analysis-contract", str(analysis_contract)])
    completed = subprocess.run(command, cwd=repo, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"PROJECT_FIRST_RUN_OK: {output}")


if __name__ == "__main__":
    main()
