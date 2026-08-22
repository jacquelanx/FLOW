"""Tests for the anchored (reference-anchoring) first-run template.

The anchored analysis code is container-only (it needs pandas/numpy/matplotlib/flowkit, which
live in the sandbox image, not FLOW's own env) — exactly like ``example_nk_panel.py``. So the
dev-env tests here cover the FLOW-side WIRING by contract and by file operations, without
importing the heavy analysis modules. The full data-dependent reproduction is a separate,
skip-gated integration test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from flow.firstrun import (
    ANCHORED_PACKAGE,
    ANCHORED_TEMPLATE,
    SCRIPT_FILENAME,
    anchored_script,
    provision_anchored,
)


def test_anchored_template_exists_and_follows_contract():
    s = anchored_script()
    assert s, "anchored template is empty/missing"
    # Honors the FLOW first-run contract and names the new outputs the plan requires.
    for token in ("--data", "--out", "--plots", "multilineage.csv",
                  "unified_cutoffs.csv", "composition_shift.csv"):
        assert token in s, f"anchored template missing {token!r}"
    assert ANCHORED_TEMPLATE.is_file()
    assert ANCHORED_PACKAGE.is_dir()


def test_anchored_template_documents_the_role_split():
    """The template must tell the agent the numbers are fixed and its job is to evaluate."""
    s = anchored_script().lower()
    assert "deterministic" in s
    assert "audit" in s or "evaluate" in s


def test_provision_anchored_installs_script_and_package(tmp_path: Path):
    fr = provision_anchored(tmp_path)
    assert fr == tmp_path / SCRIPT_FILENAME
    assert (tmp_path / SCRIPT_FILENAME).read_text() == ANCHORED_TEMPLATE.read_text()
    # The dependency package must be copied so `import anchored` resolves in the sandbox.
    for mod in ("__init__.py", "gates.py", "reference_anchor.py", "fcs_io.py",
                "flow_outputs.py", "upn18_operator_anchor.json"):
        assert (tmp_path / "anchored" / mod).is_file(), f"missing anchored/{mod}"
    # No stray bytecode copied.
    assert not (tmp_path / "anchored" / "__pycache__").exists()


def test_provision_is_idempotent(tmp_path: Path):
    provision_anchored(tmp_path)
    # A second call must overwrite cleanly (no error, package still intact).
    provision_anchored(tmp_path)
    assert (tmp_path / "anchored" / "gates.py").is_file()


def test_package_init_is_lazy_no_flowkit_import_needed():
    """anchored/__init__ must not eagerly import flowkit (so pure helpers stay importable)."""
    init_src = (ANCHORED_PACKAGE / "__init__.py").read_text()
    # No column-0 (module-level) import of .run — that would pull flowkit at import time.
    # A lazy import inside __getattr__ (indented) is fine.
    assert not any(ln.startswith("from .run import") or ln.startswith("import .run")
                   for ln in init_src.splitlines())
    assert "__getattr__" in init_src


# --- heavy, data + deps dependent: skip unless everything is available --------------------
_HAVE_DEPS = all(
    importlib.util.find_spec(m) is not None
    for m in ("pandas", "numpy", "matplotlib", "flowkit")
)
_UPN27 = Path("/private/tmp") / "does-not-exist"  # set to a real dataset dir to enable locally


@pytest.mark.skipif(not (_HAVE_DEPS and _UPN27.is_dir()),
                    reason="needs pandas/numpy/matplotlib/flowkit + a UPN27-style dataset")
def test_end_to_end_reproduces_anchored_numbers(tmp_path: Path):  # pragma: no cover
    import subprocess
    import sys

    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(ANCHORED_TEMPLATE), "--data", str(_UPN27),
         "--out", str(out), "--plots", str(out / "plots")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    for name in ("multilineage.csv", "unified_cutoffs.csv", "composition_shift.csv"):
        assert (out / "outputs" / "tables" / name).is_file()
