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
# Optional companion script, run AFTER the first-run under the same --data/--out/--plots
# contract. The harness stays agnostic about what it computes: if a project provides one, it
# runs and its CSV outputs are loaded alongside the first-run's. Projects that do not ship one
# are unaffected.
DIAGNOSTICS_FILENAME = "first_run_diagnostics.py"
CONTAINER_DIAGNOSTICS = "/data/first_run_diagnostics.py"
# Starting template offered in the editor (never executed by the harness)
EXAMPLE_TEMPLATE = FIRSTRUN_DIR / "example_nk_panel.py"
# Anchored template: the reference-anchoring pipeline (one deterministic cutoff per
# marker, locked at a reference timepoint and transferred across all timepoints). It depends
# on the vendored ``anchored/`` package, which ``provision_anchored`` copies next to the
# saved first_run.py so ``import anchored`` resolves inside the sandbox.
ANCHORED_TEMPLATE = FIRSTRUN_DIR / "anchored_nk_panel.py"
ANCHORED_PACKAGE = FIRSTRUN_DIR / "anchored"
# Companion diagnostics entry point for anchored projects, installed as
# ``first_run_diagnostics.py`` by ``provision_anchored``. It audits the first-run's locked
# cutoffs numerically and never modifies them.
ANCHORED_DIAGNOSTICS_TEMPLATE = FIRSTRUN_DIR / "anchored_nk_diagnostics.py"


# Where the scripts write their outputs, relative to the writable work dir (``/work`` in the
# sandbox, the run's artifact directory on the host — the same directory, bind-mounted). Named
# here because both the seed cell and the host-side obligations lookup depend on it.
OUT_SUBDIR = "first_run"
# Optional machine-readable companion to the worklist: the same obligations as a manifest.
# The harness reads this from the HOST side after the seed cells run, to enforce generically
# whatever a project's diagnostics chose to require. A project that writes no manifest is not
# gated at all.
OBLIGATIONS_FILENAME = "diagnostics_obligations.json"


@dataclass
class FirstRun:
    """A first-run scaffold: opening notebook cells + a prompt note for the agent."""

    cells: list[str] = field(default_factory=list)
    prompt_note: str = ""
    kind: str = ""
    # Path, relative to the work dir, where the diagnostics may have left an obligations
    # manifest. Empty when the project ships no diagnostics script.
    obligations_relpath: str = ""


def has_script(data_dir: str | Path) -> bool:
    """True if the project provides a first-run script."""
    return (Path(data_dir) / SCRIPT_FILENAME).is_file()


def has_diagnostics(data_dir: str | Path) -> bool:
    """True if the project provides a post-first-run diagnostics script."""
    return (Path(data_dir) / DIAGNOSTICS_FILENAME).is_file()


def anchored_diagnostics_script() -> str:
    """The anchored diagnostics entry point, for seeding the editor (empty if missing)."""
    try:
        return ANCHORED_DIAGNOSTICS_TEMPLATE.read_text()
    except OSError:
        return ""


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
    # Companion numeric audit of the first-run's locked cutoffs. Installed alongside rather
    # than merged into first_run.py, so the first-run script stays byte-identical.
    diagnostics_src = anchored_diagnostics_script()
    if diagnostics_src:
        (project_dir / DIAGNOSTICS_FILENAME).write_text(diagnostics_src)
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
        "out_dir = os.path.join(WORK, '" + OUT_SUBDIR + "'); os.makedirs(out_dir, exist_ok=True)\n"
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


# The diagnostics report can run to hundreds of KB, while the agent sees a truncated
# observation window. So the cell prints the script's DIGEST file — a bounded index that
# accounts for every finding by group and names the files holding the per-row detail — and
# falls back to the tail of stdout only if no digest was written. Printing the whole report
# would keep only its head and tail, deleting the middle, which is where the findings are.
DIGEST_FILENAME = "cutoff_diagnostics_digest.txt"
# Optional companion to the digest: the diagnostics script's WORKLIST — the short list of
# items it leaves the reader to open, each naming the rows behind it. Printed LAST in the
# cell, after the table inventory, so it is the final thing in the opening observation: a
# digest that counts 375 findings honestly is still not a finishable amount of work, and two
# production runs read the counts and opened three of eleven tables. Optional in exactly the
# same way the digest is — a project that writes no worklist file is unaffected.
WORKLIST_FILENAME = "diagnostics_worklist.txt"
# Characters of the digest to print. Generous relative to the digest's own cap so a project
# whose digest grows is still shown in full, and bounded so a runaway file cannot flood the
# observation. The seed-cell observation limit in NotebookEnvironment is larger than this.
#
# Reduced from 15000 to move that headroom to the worklist, which shares the same window. The
# digest caps itself at ``emit.DIGEST_MAX_CHARS`` (11500), so the slack above that figure was
# never used, while the worklist was rendering 14 of 24 obligations and pointing the other ten
# at a JSON file. The digest COUNTS findings; the worklist is what gets them read.
DIGEST_PRINT_CHARS = 12000
# Characters of the worklist to print. The worklist bounds itself when it is rendered, so
# this is a backstop against a runaway file rather than the working limit.
WORKLIST_PRINT_CHARS = 15500
# Tables wider/longer than this get their schema printed but no rows: ten wide diagnostics
# tables x 12 rows each is what used to consume the whole observation window.
INVENTORY_ROWS = 4
# Column names shown per table in the inventory. The inventory exists so the agent uses EXACT
# column names instead of guessing, and on a real study the full lists cost 5768 chars of a
# 32000-char window — a 24-column geometry table and a 44-column transfer table between them
# spent 2545 of it. That is the cheapest space in the cell to buy back, because it is the only
# part with a substitute: the worklist's `read:` lines already carry the exact expressions for
# the columns an obligation turns on, and any other column is one `list(df.columns)` away. The
# worklist has no substitute — an obligation that does not fit is one the reader never sees —
# so the trade is columns for items, and the truncation says so rather than implying the table
# is narrower than it is.
INVENTORY_COLUMNS = 12


def _diagnostics_cell() -> str:
    """Second opening cell: run the project's diagnostics script, then load its new tables.

    Kept separate from the first-run cell so a diagnostics failure cannot affect the
    first-run's own outputs, and so the harness stays agnostic about what is being measured.

    Output is written to fit the agent's observation window: the script's digest (if it wrote
    one), then a compact inventory of the new tables rather than a head() of each. The full
    tables are in the notebook as DataFrames either way, so nothing is lost — only unprinted.
    """
    src = '''\
# === FLOW first-run diagnostics (this project's script; you did NOT write this) ===
import glob as _glob
import os as _os
import subprocess as _subprocess
import pandas as _pd

_out_dir = _os.path.join(_os.environ.get('FLOW_WORK_DIR', '/work'), '__OUT_SUBDIR__')
_plots_dir = _os.path.join(_os.environ.get('FLOW_WORK_DIR', '/work'), 'plots')
try:
    first_run_tables
except NameError:
    first_run_tables = {}
_before = set(first_run_tables)

print('Running the diagnostics script provided for this project...')
_d = _subprocess.run(
    ['python', '__DIAG_SCRIPT__', '--data', _os.environ.get('FLOW_DATA_DIR', '/data'),
     '--out', _out_dir, '--plots', _plots_dir],
    capture_output=True, text=True,
)
# Prefer the script's own digest file: it is written to fit a bounded window, whereas the
# full stdout is not. Fall back to the tail of stdout when no digest exists.
_digest_path = _os.path.join(_out_dir, '__DIGEST_FILENAME__')
_digest = ''
if _os.path.isfile(_digest_path):
    try:
        _digest = open(_digest_path).read()
    except Exception as _e:
        print('could not read the diagnostics digest:', _e)
if _digest:
    print(_digest[:__DIGEST_PRINT_CHARS__])
    if len(_digest) > __DIGEST_PRINT_CHARS__:
        print(f'[digest truncated at __DIGEST_PRINT_CHARS__ chars; read the rest with '
              f'open({_digest_path!r}).read()]')
    _tail = _d.stdout[-1500:]
    if _tail.strip():
        print('\\nDIAGNOSTICS RUN LOG (tail):\\n' + _tail)
else:
    print('NOTE: no digest file at', _digest_path, '- showing the tail of stdout instead.')
    print(_d.stdout[-8000:])
if _d.returncode != 0:
    print('DIAGNOSTICS STDERR (tail):', _d.stderr[-3000:])

# Load any NEW tables the diagnostics wrote (the first-run cell already loaded its own).
for _f in sorted(_glob.glob(_os.path.join(_out_dir, '*.csv'))):
    _name = _os.path.splitext(_os.path.basename(_f))[0]
    if _name in _before:
        continue
    try:
        first_run_tables[_name] = _pd.read_csv(_f)
    except Exception as _e:
        print('could not load', _f, _e)
_new = sorted(set(first_run_tables) - _before)
if _new:
    # An inventory, not a dump: schemas plus a few rows of the small tables. Every one of
    # these is a full DataFrame in this notebook - print what you need from it in a cell.
    print(f'\\n=== {len(_new)} diagnostics table(s) loaded (each is a DataFrame variable) ===')
    for _name in _new:
        _df = first_run_tables[_name]
        if _name.isidentifier():
            globals()[_name] = _df
        _cols = list(_df.columns)
        _shown = _cols[:__INVENTORY_COLUMNS__]
        _rest = len(_cols) - len(_shown)
        _more = (f"  (+{_rest} more column(s); run list({_name}.columns) for all of them)"
                 if _rest > 0 else '')
        print(f'\\n{_name}  shape={_df.shape}\\n  columns: {_shown}{_more}')
        if len(_df) <= __INVENTORY_ROWS__:
            print(_df.to_string())
else:
    print('NOTE: the diagnostics script produced no additional tables in', _out_dir)

# The worklist LAST, so the final thing in this observation is the short list of items to
# open rather than a schema dump. Optional: absent file, no output, no complaint.
_worklist_path = _os.path.join(_out_dir, '__WORKLIST_FILENAME__')
if _os.path.isfile(_worklist_path):
    try:
        _worklist = open(_worklist_path).read()
    except Exception as _e:
        print('could not read the diagnostics worklist:', _e)
        _worklist = ''
    if _worklist:
        print('\\n' + '=' * 78)
        print(_worklist[:__WORKLIST_PRINT_CHARS__])
        if len(_worklist) > __WORKLIST_PRINT_CHARS__:
            print(f'[worklist truncated at __WORKLIST_PRINT_CHARS__ chars; read the rest '
                  f'with open({_worklist_path!r}).read()]')
'''
    return (
        src.replace("__DIAG_SCRIPT__", CONTAINER_DIAGNOSTICS)
        .replace("__OUT_SUBDIR__", OUT_SUBDIR)
        .replace("__DIGEST_FILENAME__", DIGEST_FILENAME)
        .replace("__DIGEST_PRINT_CHARS__", str(DIGEST_PRINT_CHARS))
        .replace("__WORKLIST_FILENAME__", WORKLIST_FILENAME)
        .replace("__WORKLIST_PRINT_CHARS__", str(WORKLIST_PRINT_CHARS))
        .replace("__INVENTORY_ROWS__", str(INVENTORY_ROWS))
        .replace("__INVENTORY_COLUMNS__", str(INVENTORY_COLUMNS))
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

# Appended to PROMPT_NOTE only when the project ships a diagnostics script. Stays generic: it
# describes the CONTRACT of a diagnostics table (measurements + stated rules, with explicit
# missingness), not any particular measurement.
DIAGNOSTICS_NOTE = (
    "\n\nA second script has also already run: a DIAGNOSTICS pass over the first-run's own "
    "parameter choices. Its tables are loaded alongside the first-run's. What is printed above "
    "is a DIGEST: a deliberately short INDEX of its findings, not the whole report. How to use "
    "it:\n"
    "  • The digest accounts for EVERY finding by group, with a full count and the worst "
    "case in each group, and it names the file holding the per-row detail. A group listed as "
    "'x43' means 43 findings exist and you have seen ONE of them. To judge the other 42, read "
    "the rows in a cell — the tables are already DataFrames, so filter the flags frame by "
    "its `code` column. Do not treat the printed example as the whole group, and do not "
    "characterize a group you have not read.\n"
    "  • Read the PER-TIMEPOINT block FIRST, before any headline number. If a timepoint is "
    "listed as off-trend, name it in your answer and state whether you are including or "
    "excluding it and why. Silently leaving a timepoint out of a longitudinal claim is the one "
    "thing you must not do — a trend that only holds because a contradicting sample went "
    "unmentioned is worse than no trend.\n"
    "  • Then read the flagged items, before interpreting any headline number.\n"
    "  • Every entry is a MEASUREMENT, and every flag carries the exact RULE that surfaced it. "
    "They are not conclusions. You may disagree with a rule — say so and say why.\n"
    "  • A null is 'could not be measured', NOT zero. Do not read a missing value, or the "
    "absence of a flag, as evidence that something is fine.\n"
    "  • Where a diagnostics table reports an uncertainty for a reported number, quote the "
    "number WITH that uncertainty rather than as a point estimate.\n"
    "  • If the diagnostics report a reproduction or fidelity check, state its outcome — if it "
    "failed, the rest of the diagnostics may not describe the run that produced the tables.\n"
    "  • The diagnostics do NOT replace the first-run's numbers and do not change them. Use "
    "them to decide how much weight each number carries, and which items would need another "
    "iteration.\n"
    "  • PRINTING: a diagnostics table can run to hundreds of rows, and this notebook keeps "
    "only the head and tail of a cell's output — so dumping a filtered frame deletes its "
    "MIDDLE, which is where the rows you filtered for usually are. Aggregate first "
    "(`.groupby('code').size()`, `.value_counts()`, a few named columns via `[[...]]`), then "
    "print the specific rows you decided you needed. A truncated dump is worse than a "
    "summary, because you cannot tell what it removed.\n"
    "  • If the diagnostics printed a WORKLIST at the end of that output, treat it as the "
    "minimum, not the maximum. Each item names measurements that already exist and that the "
    "digest only counted; each carries the exact expression that retrieves them. Work through "
    "it and account for every item in your answer — concluding 'this does not change the "
    "result' is a valid outcome and stating it is required; going silent on an item is not. "
    "Anything the worklist reports as NOT PROMOTED is not addressed by working through it, so "
    "do not describe your review as complete on that basis."
)


def build_first_run(data_dir: str | Path, mode: str = "auto") -> FirstRun | None:
    """Build the first-run scaffold for a project, or None if there is nothing to run.

    mode: "none" disables the first-run entirely (agent starts from a blank notebook);
    "auto" (the default) runs the project's ``first_run.py`` if one has been provided.
    Returns None when disabled or when no script is present.

    When the project also ships ``first_run_diagnostics.py``, a second opening cell runs it and
    loads its tables. Both cells execute before the agent acts and neither counts against its
    step budget.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "none":
        return None
    if not has_script(data_dir):
        return None
    cells = [_seed_cell()]
    note = PROMPT_NOTE
    kind = "script"
    obligations_relpath = ""
    if has_diagnostics(data_dir):
        cells.append(_diagnostics_cell())
        note = PROMPT_NOTE + DIAGNOSTICS_NOTE
        kind = "script+diagnostics"
        obligations_relpath = f"{OUT_SUBDIR}/{OBLIGATIONS_FILENAME}"
    return FirstRun(cells=cells, prompt_note=note, kind=kind,
                    obligations_relpath=obligations_relpath)
