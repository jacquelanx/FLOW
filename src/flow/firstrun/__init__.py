"""First-run harness.

FLOW runs a deterministic *first-run script* at the start of a trajectory to do the heavy
lifting (all the flow-cytometry analysis); the agent then only interprets the results.

This module invokes a script in the sandbox, then loads the tables that the script
produced so the agent can interpret them.

  The script is invoked inside the container as::

      python /data/first_run.py --data /data --out /work/first_run --plots /work/plots

  It should read the dataset from ``--data`` and write result CSVs (and any plots) to
  ``--out`` / ``--plots``. Every CSV it writes is auto-loaded into the agent's notebook (into
  the ``first_run_tables`` dict, and as a variable named after each CSV), with its schema
  printed so the agent uses the exact column names.

An example script (``example_nk_panel.py``) ships as a *starting template* the biologist can
adapt — the harness never imports or depends on it; it is only offered as editor seed content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FIRSTRUN_DIR = Path(__file__).resolve().parent
# The script, saved in the project directory (which mounts read-only at /data).
SCRIPT_FILENAME = "first_run.py"
CONTAINER_SCRIPT = "/data/first_run.py"
# Starting template offered in the editor (never executed by the harness)
EXAMPLE_TEMPLATE = FIRSTRUN_DIR / "example_nk_panel.py"
# Anchored template: the reference-anchoring pipeline (one deterministic cutoff per
# marker, locked at a reference timepoint and transferred across all timepoints). It depends
# on the vendored ``anchored/`` package, which ``provision_anchored`` copies next to the
# saved first_run.py so ``import anchored`` resolves inside the sandbox.
ANCHORED_TEMPLATE = FIRSTRUN_DIR / "anchored_nk_panel.py"
ANCHORED_PACKAGE = FIRSTRUN_DIR / "anchored"


@dataclass
class FirstRun:
    """A first-run scaffold: opening notebook cells + a prompt note for the agent."""

    cells: list[str] = field(default_factory=list)
    prompt_note: str = ""
    kind: str = ""


def has_script(data_dir: str | Path) -> bool:
    """True if the project provides a first-run script."""
    return (Path(data_dir) / SCRIPT_FILENAME).is_file()


def example_script() -> str:
    """The example first-run script, used to seed the editor (empty string if missing)."""
    try:
        return EXAMPLE_TEMPLATE.read_text()
    except OSError:
        return ""


def anchored_script() -> str:
    """The anchored (reference-anchoring) first-run template, for seeding the editor."""
    try:
        return ANCHORED_TEMPLATE.read_text()
    except OSError:
        return ""


def provision_anchored(project_dir: str | Path) -> Path:
    """Install the anchored template as this project's ``first_run.py`` + its dependency.

    Writes ``<project>/first_run.py`` (the anchored entry point) and copies the vendored
    ``anchored/`` package to ``<project>/anchored`` so ``import anchored`` works when the
    sandbox runs ``python /data/first_run.py``. Returns the first_run.py path.
    """
    import shutil

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / SCRIPT_FILENAME).write_text(ANCHORED_TEMPLATE.read_text())
    dest_pkg = project_dir / "anchored"
    if dest_pkg.exists():
        shutil.rmtree(dest_pkg)
    shutil.copytree(
        ANCHORED_PACKAGE, dest_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return project_dir / SCRIPT_FILENAME


def _seed_cell() -> str:
    """The opening notebook cell: run the project's script, then auto-load its CSV outputs."""
    return (
        "# === FLOW first-run (this project's script; you did NOT write this) ===\n"
        "import os, glob, subprocess\n"
        "import pandas as pd\n"
        "pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60)\n"
        "DATA = os.environ.get('FLOW_DATA_DIR', '/data')\n"
        "WORK = os.environ.get('FLOW_WORK_DIR', '/work')\n"
        "out_dir = os.path.join(WORK, 'first_run'); os.makedirs(out_dir, exist_ok=True)\n"
        "plots_dir = os.path.join(WORK, 'plots'); os.makedirs(plots_dir, exist_ok=True)\n"
        "cmd = ['python', '" + CONTAINER_SCRIPT + "', '--data', DATA, '--out', out_dir,\n"
        "       '--plots', plots_dir]\n"
        "print('Running the first-run script provided for this project...')\n"
        "_p = subprocess.run(cmd, capture_output=True, text=True)\n"
        "print(_p.stdout[-4000:])\n"
        "if _p.returncode != 0:\n"
        "    print('FIRST-RUN STDERR (tail):', _p.stderr[-3000:])\n"
        "# Auto-load every table the script produced. USE THESE EXACT COLUMN NAMES — do not\n"
        "# guess, and do not recompute the analysis; interpret these tables.\n"
        "first_run_tables = {}\n"
        "for _f in sorted(glob.glob(os.path.join(out_dir, '*.csv'))):\n"
        "    _name = os.path.splitext(os.path.basename(_f))[0]\n"
        "    try:\n"
        "        first_run_tables[_name] = pd.read_csv(_f)\n"
        "    except Exception as _e:\n"
        "        print('could not load', _f, _e)\n"
        "for _name, _df in first_run_tables.items():\n"
        "    if _name.isidentifier():\n"
        "        globals()[_name] = _df   # also expose as a variable named after the CSV\n"
        "    print(f'\\n=== {_name}  shape={_df.shape}  columns={list(_df.columns)} ===')\n"
        "    print(_df.head(12).to_string())\n"
        "if not first_run_tables:\n"
        "    print('NOTE: the first-run script produced no CSV tables in', out_dir)\n"
    )


# Generic, biology-free interpretive guidance. The SPECIFIC biology (panel, populations,
# comparisons) reaches the agent separately, from the Analysis spec (prompt.md).
PROMPT_NOTE = (
    "A deterministic FIRST-RUN script (provided for this project) has already run in the "
    "opening cell and produced the result tables shown above. They are loaded into the "
    "`first_run_tables` dict, and each is also a variable named after its CSV; their exact "
    "columns are printed above — USE THOSE EXACT NAMES, do not guess.\n\n"
    "YOUR ROLE IS INTERPRETIVE — the heavy analysis is ALREADY DONE, ONCE, deterministically. "
    "The headline numbers are the first-run's; report them from these tables — do NOT replace "
    "them with your own re-derivation:\n"
    "  • If the tables include QC columns/flags, start there and note any unreliable samples, "
    "then down-weight or exclude them (and say which).\n"
    "  • Base every reported number on these tables; do not use raw event counts.\n"
    "  • You MAY inspect the first-run's parameters (e.g. any cutoffs it records) and the raw "
    "data to AUDIT its choices. If a choice looks wrong, refine it in a cell, show the "
    "before/after, and clearly flag the discrepancy — but keep the first-run tables as the "
    "authoritative headline unless you have strong evidence to override a specific value.\n"
    "  • Follow the analysis specification in the additional guidance below (the panel, the "
    "populations of interest, and the comparisons the lab wants).\n"
    "  • Any NEW number you report must be computed in a cell. Submit a clear, evidence-backed "
    "conclusion."
)


def build_first_run(data_dir: str | Path, mode: str = "auto") -> FirstRun | None:
    """Build the first-run scaffold for a project, or None if there is nothing to run.

    mode: "none" disables the first-run entirely (agent starts from a blank notebook);
    "auto" (the default) runs the project's ``first_run.py`` if one has been provided.
    Returns None when disabled or when no script is present.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "none":
        return None
    if not has_script(data_dir):
        return None
    return FirstRun(cells=[_seed_cell()], prompt_note=PROMPT_NOTE, kind="script")
