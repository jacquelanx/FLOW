"""Source-level contract for anchored QC bivariate rendering.

The anchored plotting module is container-only because it imports Matplotlib and SciPy.
These tests intentionally inspect its AST instead of importing the heavy stack, matching
the existing first-run wiring tests while keeping the host test environment lightweight.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


QC_PLOTS = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "flow"
    / "firstrun"
    / "anchored"
    / "qc_plots.py"
)
EXAMPLE_NK_PANEL = QC_PLOTS.parent.parent / "example_nk_panel.py"


def _tree() -> ast.Module:
    return ast.parse(QC_PLOTS.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_calls(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_density_contour_levels_match_validated_upn27_style():
    tree = _tree()
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "DENSITY_CONTOUR_FRACTIONS"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == (0.20, 0.40, 0.65)


def test_rectangular_gate_fill_alpha_is_twenty_percent():
    tree = _tree()
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "GATE_ALPHA"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == 0.20

    qc_page = _function(tree, "qc_page")
    for call in (
        node for node in ast.walk(qc_page)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_shade_rect"
    ):
        alpha = next(keyword.value for keyword in call.keywords if keyword.arg == "alpha")
        assert isinstance(alpha, ast.Name) and alpha.id == "GATE_ALPHA"


def test_rectangular_gate_regions_are_directly_named():
    source = QC_PLOTS.read_text(encoding="utf-8")
    for label in ("T gate", "NK gate", "CD4 gate", "CD8 gate", "Donor gate", "CAR+ gate"):
        assert f'"{label}"' in source


def test_all_anchored_bivariate_panels_use_shared_heat_density_renderer():
    tree = _tree()
    for name in ("_hb", "_lymph_density_panel", "_live_cd45_lymph_geometry_panel"):
        assert "_heat_scatter_density" in _direct_calls(_function(tree, name))


def test_paired_scatter_views_name_parent_modes_low_ssc_and_applied_gate():
    source = QC_PLOTS.read_text(encoding="utf-8")
    for label in (
        "All analyzed events → viable cellular scatter gate",
        "Live-CD45 parent → lymph density geometry",
        "Parent-density contours",
        "Fitted mode centers",
        "Low-SSC contour (95%)",
        "Applied gate",
        "diagnostic only",
    ):
        assert label in source


def test_legacy_hist2d_renderer_is_not_used_in_anchored_qc():
    tree = _tree()
    legacy_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "hist2d"
    ]
    assert not legacy_calls


def test_population_highlight_sampling_is_deterministic():
    source = QC_PLOTS.read_text(encoding="utf-8")
    assert "_deterministic_mask_indices" in source
    assert "np.random.choice" not in source


def test_timepoint_qc_pages_use_letter_landscape_size():
    function = _function(_tree(), "qc_page")
    figure_call = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "figure"
    )
    figsize = next(keyword.value for keyword in figure_call.keywords
                   if keyword.arg == "figsize")
    assert ast.literal_eval(figsize) == (11.0, 8.5)


def test_cd4_cd8_title_wraps_within_its_panel():
    source = QC_PLOTS.read_text(encoding="utf-8")
    assert '"9. CD4/CD8 of T"' in source
    assert 'f"CD4={pct.get' in source
    assert "clip_on=False" in source


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None
            for name in ("matplotlib", "scipy", "sklearn")),
    reason="density-geometry dependencies live in the FLOW analysis container",
)
def test_live_cd45_mode_fit_is_deterministic_and_selects_lowest_ssc_mode():
    import numpy as np

    from flow.firstrun.anchored.qc_plots import _fit_live_cd45_density_modes

    rng = np.random.RandomState(2701)
    fsc = np.r_[rng.normal(0.35, 0.025, 1200), rng.normal(0.58, 0.035, 1800)]
    ssc = np.r_[rng.normal(0.10, 0.015, 1200), rng.normal(0.47, 0.040, 1800)]
    parent = np.ones(len(fsc), dtype=bool)
    first = _fit_live_cd45_density_modes(fsc, ssc, parent)
    second = _fit_live_cd45_density_modes(fsc, ssc, parent)
    assert 1 <= len(first["modes"]) <= 4
    assert first["fit_method"] == second["fit_method"]
    assert np.allclose(
        [mode["mean"] for mode in first["modes"]],
        [mode["mean"] for mode in second["modes"]],
    )
    assert first["modes"][0]["is_low_ssc_mode"] is True
    assert first["modes"][0]["mean"][1] == min(
        mode["mean"][1] for mode in first["modes"]
    )


def test_generic_first_run_flow_scatter_uses_heat_density_contract():
    tree = ast.parse(EXAMPLE_NK_PANEL.read_text(encoding="utf-8"))
    gating_plot = _function(tree, "gating_plot")
    assert "_heat_scatter_density" in _direct_calls(gating_plot)
    assert "_overlay_accepted_density_envelope" in _direct_calls(gating_plot)
    raw_scatter_calls = [
        node for node in ast.walk(gating_plot)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scatter"
    ]
    assert not raw_scatter_calls


def test_umap_keeps_semantic_cluster_coloring():
    tree = ast.parse(EXAMPLE_NK_PANEL.read_text(encoding="utf-8"))
    unsupervised = _function(tree, "unsupervised")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scatter"
        for node in ast.walk(unsupervised)
    )


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None for name in ("matplotlib", "scipy")),
    reason="anchored plotting dependencies live in the FLOW analysis container",
)
def test_bivariate_cd4_cd8_fixture_uses_heat_density_and_contours():
    """Exercise the bivariate CD4×CD8 renderer even when UPN27 uses CD4-only."""
    import matplotlib.pyplot as plt
    import numpy as np

    from flow.firstrun.anchored.qc_plots import _hb

    rng = np.random.RandomState(2701)
    cd4 = np.r_[rng.normal(900, 160, 2500), rng.normal(250, 90, 1200)]
    cd8 = np.r_[rng.normal(300, 100, 2500), rng.normal(1100, 180, 1200)]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    _hb(ax, cd4, cd8, "CD4", "CD8", full_axes=True)
    try:
        assert len(ax.collections) >= 2  # heat mesh + contour set
    finally:
        plt.close(fig)
