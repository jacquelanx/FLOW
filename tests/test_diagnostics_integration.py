"""End-to-end test: run the anchored first-run, then audit it with the diagnostics.

Skip-gated on the container-only scientific stack (pandas / matplotlib / flowkit / sklearn /
scipy), exactly like ``test_anchored_firstrun.py``'s heavy cases. When those are installed this
builds a small synthetic CAR-NK dataset with a DELIBERATELY INJECTED defect, runs the real
pipeline over it, and checks that the diagnostics detect the defect.

The assertions are about the diagnostics' contract, not about the pipeline's gating quality:

  * the replay reproduces the pipeline's own reported percentages EXACTLY (the pipeline is
    deterministic, so anything else means the audit is describing a different gating);
  * every tier computes;
  * the injected negative-population drift is the largest drift measured anywhere, and it is
    flagged;
  * the known-positive and known-negative control tubes come out on the right sides;
  * every flag carries the rule that produced it and a resolution hint.

Deliberately NOT asserted: absolute population percentages. Those depend on how well the
pipeline gates synthetic data, which is not what this test is for.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HEAVY = ("pandas", "numpy", "matplotlib", "flowkit", "flowio", "sklearn", "scipy")
_HAVE_DEPS = all(importlib.util.find_spec(m) is not None for m in _HEAVY)

pytestmark = pytest.mark.skipif(
    not _HAVE_DEPS,
    reason=f"needs the container-only stack: {', '.join(_HEAVY)}",
)

FIRSTRUN_DIR = Path(__file__).resolve().parent.parent / "src" / "flow" / "firstrun"

# Injected defect: the CD56-negative population is shifted at this timepoint only.
DRIFT_TIMEPOINT = "D28"
DRIFT_UNITS = 300.0
N_EVENTS = 9000


def _build_dataset(out: Path) -> None:
    """A minimal synthetic CAR-NK panel with one injected negative-population drift."""
    import flowio
    import numpy as np
    from flowkit import transforms as T

    biex = T.WSPBiexTransform(negative=0.0, width=-10.0, positive=4.418539922,
                              max_value=262144.0)
    lin_t = 262144.0
    channels = [
        "FSC-A", "SSC-A", "SSC-W", "SSC-H",
        "BV510-A", "PE-Cy5-A", "Qdot 800-A", "APC-Cy7-A", "PE-Texas Red-A",
        "UV 450 L/D-A", "Alexa Fluor 488-A", "Alexa Fluor 647-A", "BV650-A",
        "Alexa Fluor 700-A", "PE-Cy5-5-A", "Time",
    ]
    NEG, POS = (620.0, 55.0), (2450.0, 230.0)
    rng = np.random.default_rng(7)

    def inv(vals):
        return biex.inverse(np.asarray(vals, float).reshape(-1, 1)).ravel()

    def tube(parts):
        """parts: list of (n, fsc, ssc, {channel: (mu, sd)})."""
        blocks = []
        for n, fsc, ssc, spec in parts:
            if n <= 0:
                continue
            d = {c: np.zeros(n) for c in channels}
            d["FSC-A"] = np.clip(rng.normal(fsc, 0.035, n), 0.01, 0.99) * lin_t
            ssca = np.clip(rng.normal(ssc, 0.03, n), 0.005, 0.95)
            d["SSC-A"] = ssca * lin_t
            w = np.where(rng.random(n) < 0.03, rng.normal(0.55, 0.04, n),
                         rng.normal(0.30, 0.02, n))
            d["SSC-W"] = np.clip(w, 0.05, 0.95) * lin_t
            d["SSC-H"] = np.clip(ssca * 0.9 + rng.normal(0, 0.012, n), 0.004, 0.95) * lin_t
            for chan in channels[4:-1]:
                mu_sd = spec.get(chan, NEG)
                d[chan] = inv(rng.normal(mu_sd[0], mu_sd[1], n))
            blocks.append(d)
        out_d = {c: np.concatenate([b[c] for b in blocks]) for c in channels}
        total = len(out_d["FSC-A"])
        order = rng.permutation(total)
        out_d = {c: v[order] for c, v in out_d.items()}
        out_d["Time"] = np.sort(rng.uniform(0, 60000, total))
        return out_d

    def write(path: Path, data):
        mat = np.column_stack([data[c] for c in channels]).astype(np.float32)
        with open(path, "wb") as fh:
            flowio.create_fcs(fh, mat.flatten().tolist(), channel_names=channels)

    fcs = out / "fcs"
    fcs.mkdir(parents=True, exist_ok=True)

    plan = [("Baseline", 0.10, 0.00), (DRIFT_TIMEPOINT, 0.35, 0.80), ("D90", 0.28, 0.70)]
    for label, nk_f, donor_f in plan:
        cd56_neg = (NEG[0] + DRIFT_UNITS, NEG[1]) if label == DRIFT_TIMEPOINT else NEG
        n_lymph = int(0.62 * N_EVENTS)
        n_nk = int(nk_f * n_lymph)
        n_t = int(0.55 * n_lymph)
        n_b = int(0.05 * n_lymph)
        n_other = max(0, n_lymph - n_nk - n_t - n_b)
        n_donor = int(donor_f * n_nk)
        lymph = {"BV510-A": POS, "PE-Texas Red-A": cd56_neg}
        write(fcs / f"Specimen_001_{label}.fcs", tube([
            (int(0.10 * N_EVENTS), 0.05, 0.05, {"UV 450 L/D-A": POS}),
            (int(0.16 * N_EVENTS), 0.45, 0.45, {"BV510-A": POS}),
            (int(0.12 * N_EVENTS), 0.50, 0.25, {"BV510-A": POS, "PE-Cy5-A": POS}),
            (n_b, 0.35, 0.10, {**lymph, "Qdot 800-A": POS}),
            (n_t, 0.35, 0.10, {**lymph, "APC-Cy7-A": POS, "Alexa Fluor 700-A": POS}),
            (n_other, 0.35, 0.10, lymph),
            (n_nk - n_donor, 0.35, 0.10, {"BV510-A": POS, "PE-Texas Red-A": POS,
                                          "BV650-A": POS}),
            (n_donor, 0.35, 0.10, {"BV510-A": POS, "PE-Texas Red-A": POS, "BV650-A": POS,
                                   "Alexa Fluor 488-A": POS, "Alexa Fluor 647-A": POS}),
        ]))

    # Control tubes: NT-NK is CAR-negative by design, the product is CAR-bright.
    for fname, car_spec in (("Specimen_001_NT-NK.fcs", NEG), ("Specimen_001_Car.fcs", POS)):
        n_nk = int(0.9 * N_EVENTS)
        write(fcs / fname, tube([
            (N_EVENTS - n_nk, 0.05, 0.05, {"UV 450 L/D-A": POS}),
            (n_nk, 0.35, 0.10, {"BV510-A": POS, "PE-Texas Red-A": POS, "BV650-A": POS,
                                "Alexa Fluor 488-A": POS, "Alexa Fluor 647-A": car_spec}),
        ]))

    (out / "metadata.json").write_text(json.dumps({
        "patient_study": "UPNTEST",
        "hla_specificity": "HLA-A2",
        "hla_polarity": "donor",
        "reference_timepoint": "Baseline",
        "anchor_mode": "negative",
        "subsample": N_EVENTS,
        "diagnostics_sensitivity_events": 6000,
    }, indent=2))
    (out / "flow.csv").write_text("label\nBaseline\n" + DRIFT_TIMEPOINT + "\nD90\n")


def _run(script: str, data: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FIRSTRUN_DIR / script),
         "--data", str(data), "--out", str(out), "--plots", str(out / "plots")],
        capture_output=True, text=True, env={"PYTHONPATH": str(FIRSTRUN_DIR), "PATH": "/usr/bin:/bin"},
    )


@pytest.fixture(scope="module")
def diagnostics_report(tmp_path_factory) -> dict:
    """Build the dataset, run the first-run, then run the diagnostics. Returns the report."""
    root = tmp_path_factory.mktemp("diag")
    data, out = root / "data", root / "out"
    out.mkdir(parents=True, exist_ok=True)
    _build_dataset(data)

    first = _run("anchored_nk_panel.py", data, out)
    if "FIRST_RUN_OK" not in first.stdout:
        pytest.skip(f"first-run did not complete on synthetic data:\n{first.stdout[-2500:]}\n"
                    f"{first.stderr[-2500:]}")

    diag = _run("anchored_nk_diagnostics.py", data, out)
    report_path = out / "cutoff_diagnostics.json"
    assert report_path.is_file(), f"no report written:\n{diag.stdout[-3000:]}\n{diag.stderr[-2000:]}"
    return json.loads(report_path.read_text())


# ── the replay must audit the run that actually happened ──────────────────────
def test_replay_reproduces_the_pipeline_numbers_exactly(diagnostics_report):
    """The pipeline is deterministic; anything but an exact match means a different gating."""
    rep = diagnostics_report["reproduction"]
    assert rep["available"], "no multilineage table found to verify against"
    assert rep["checked"] > 0
    assert rep["faithful"], f"replay diverged: {rep['mismatches'][:5]}"
    assert rep["max_abs_delta_pp"] == 0.0


def test_every_tier_computes(diagnostics_report):
    assert diagnostics_report["tier_failures"] == []
    for key in ("tier0_data_adequacy", "tier1_cutoff_foundation", "tier2_transfer_validity",
                "tier3_sensitivity", "tier4_internal_controls",
                "tier5_operator_concordance"):
        assert key in diagnostics_report


# ── the injected defect must be detected ──────────────────────────────────────
def test_the_injected_drift_is_the_largest_drift_measured(diagnostics_report):
    drifts = []
    for marker, m in diagnostics_report["tier2_transfer_validity"]["markers"].items():
        if not m.get("available"):
            continue
        for tp, row in m["timepoints"].items():
            if row.get("is_reference"):
                continue
            d = row.get("negative_mode_drift_in_negative_sd")
            if d is not None:
                drifts.append((d, marker, tp))
    assert drifts, "no drift was measured anywhere"
    drifts.sort(reverse=True)
    worst_value, worst_marker, worst_tp = drifts[0]
    assert (worst_marker, worst_tp) == ("cd56", DRIFT_TIMEPOINT), (
        f"expected the injected cd56@{DRIFT_TIMEPOINT} drift to dominate, got {drifts[:3]}")
    assert worst_value > 0.5, f"injected drift measured only {worst_value} negative-SD"


def test_the_injected_drift_produces_a_flag(diagnostics_report):
    subjects = [f["subject"] for f in diagnostics_report["flags"]
                if f["code"] == "NEGATIVE_MODE_DRIFT"]
    assert any("cd56" in s and DRIFT_TIMEPOINT in s for s in subjects), subjects


def test_drift_at_the_undisturbed_timepoint_is_not_flagged(diagnostics_report):
    """Guards against a rule so loose that everything trips it."""
    subjects = [f["subject"] for f in diagnostics_report["flags"]
                if f["code"] == "NEGATIVE_MODE_DRIFT"]
    assert not any("cd56 @ D90" in s for s in subjects), subjects


# ── internal controls must land on the right side ─────────────────────────────
def test_control_tubes_come_out_on_the_expected_sides(diagnostics_report):
    tubes = diagnostics_report["tier4_internal_controls"]["control_tubes"]
    ntnk, car = tubes.get("ntnk", {}), tubes.get("car", {})
    if ntnk.get("available"):
        assert ntnk["car_positive_rate_of_nk"] < 0.10, "designed CAR-negative tube reads positive"
    if car.get("available"):
        assert car["car_positive_rate_of_nk"] > 0.50, "designed CAR-positive tube reads low"


def test_pre_infusion_donor_rate_is_near_zero(diagnostics_report):
    """No donor cells exist before infusion; this is a specificity floor for the HLA cutoff."""
    rows = [r for r in diagnostics_report["tier4_internal_controls"]["pre_infusion_donor"]
            if r.get("available")]
    for r in rows:
        assert r["donor_rate_of_nk"] < 0.05, r


# ── discipline holds in real output ──────────────────────────────────────────
def test_every_flag_carries_its_rule_and_a_resolution_hint(diagnostics_report):
    for f in diagnostics_report["flags"]:
        assert f["rule"], f
        assert f["resolution_hint"], f
        assert isinstance(f["measured"], dict)
        assert f["tier"] in (0, 1, 2, 3, 4, 5)


def test_thresholds_are_all_documented_in_the_emitted_audit_trail(diagnostics_report):
    rows = diagnostics_report["thresholds"]
    assert rows
    for r in rows:
        assert r["provenance"] in ("self-calibrating", "pipeline-constant", "settable-default",
                                   "cost-knob")
        assert r["rationale"]


def test_limitations_are_stated_rather_than_implied(diagnostics_report):
    assert len(diagnostics_report["limitations"]) >= 4


# ── artifacts the agent will actually read ────────────────────────────────────
# ── a cutoff the reference cannot audit is still audited somewhere ────────────
# This dataset has ZERO donor cells at the Baseline reference (donor_f = 0.00 in the plan
# above), which is not a quirk of the synthetic data — it is what a pre-infusion reference
# always looks like. The CAR cutoff is applied inside Donor NK, so the reference pool cannot
# carry its audit, and before the fallback existed that produced no tier-1 row, no tier-2 rows
# and no tier-3 sweep for the one marker the study reports as its headline.
def test_every_available_cutoff_is_audited_somewhere(diagnostics_report):
    t1 = diagnostics_report["tier1_cutoff_foundation"]["markers"]
    assert "car" in t1, "the CAR cutoff produced no tier-1 entry at all"
    for key, m in t1.items():
        assert "audited_at_reference" in m, f"{key} does not say where it was audited"
        if m.get("available"):
            assert m.get("audited_at_timepoint"), f"{key} is available but names no timepoint"


def test_an_empty_reference_pool_moves_the_audit_and_says_so(diagnostics_report):
    car = diagnostics_report["tier1_cutoff_foundation"]["markers"]["car"]
    assert car["available"], "the CAR cutoff was not audited anywhere"
    assert car["audited_at_reference"] is False
    assert car["audited_at_timepoint"] != "Baseline"
    assert "events" in (car["reference_audit_unavailable_reason"] or "")
    # The substitution is labelled, not silent: the note has to say the numbers are placement
    # in another timepoint's distribution rather than a check on the derivation.
    assert "do NOT verify how it was derived" in car["audit_scope_note"]


def test_a_moved_audit_is_flagged_because_an_absent_row_looks_like_a_clean_one(
        diagnostics_report):
    subjects = [f["subject"] for f in diagnostics_report["flags"]
                if f["code"] == "REFERENCE_AUDIT_UNAVAILABLE"]
    assert any(s.startswith("car @") for s in subjects), subjects


def test_a_moved_audit_does_not_claim_a_derivation_fingerprint(diagnostics_report):
    """A locked cutoff's percentile rank in a pool it was transferred to is a coincidence."""
    car = diagnostics_report["tier1_cutoff_foundation"]["markers"]["car"]
    assert car.get("suspected_percentile_fallback") is None
    assert "away from the reference" in (car.get("fingerprint_note") or "")
    for f in diagnostics_report["flags"]:
        if f["code"] in ("PERCENTILE_FALLBACK_SUSPECTED", "NO_VALLEY_EVIDENCE"):
            assert not f["subject"].startswith("car @"), (
                "a derivation finding was raised from a fallback audit")


def test_the_transfer_tier_measures_drift_against_the_sample_it_could_audit(
        diagnostics_report):
    """Comparing against a pool tier 1 rejected as too small would have no baseline behind it."""
    t1 = diagnostics_report["tier1_cutoff_foundation"]["markers"]["car"]
    t2 = diagnostics_report["tier2_transfer_validity"]["markers"]["car"]
    assert t2["available"], "CAR transfer is still unmeasured"
    assert t2["baseline_timepoint"] == t1["audited_at_timepoint"]
    assert t2["baseline_is_reference"] is False
    assert "NOT 'unchanged since the cutoff was derived'" in t2["baseline_note"]
    measured = [tp for tp, r in t2["timepoints"].items() if r.get("available")]
    assert len(measured) >= 2, f"only {measured} carried a CAR transfer measurement"


def test_the_headline_car_number_now_carries_a_range_from_its_own_cutoff(diagnostics_report):
    """The defect this closes: 'cutoff' named as the dominant source of %CAR+ uncertainty,
    computed by moving CD14, CD45 and the viability gate while CAR itself was never touched."""
    t3 = diagnostics_report["tier3_sensitivity"]
    swept = {(c["timepoint"], c["marker"]) for c in (t3.get("coverage") or {}).get(
        "completed", [])}
    assert any(m == "car" for _tp, m in swept), "the CAR cutoff was never perturbed"
    drivers = {row["marker"] for row in (t3.get("perturbations") or [])
               if row["metric"] == "%CAR+ (of Donor NK)"}
    assert "car" in drivers, sorted(drivers)


# ── the HLA x CAR gate has a numeric surrogate ────────────────────────────────
def test_the_donor_car_gate_is_measured_at_every_timepoint(diagnostics_report):
    """It defines %Donor NK and %CAR+, and had no 2-D numeric surrogate of any kind."""
    geom = diagnostics_report["tier2_transfer_validity"]["gate_geometry"]
    assert geom, "no gate geometry was computed"
    for tp, g in geom.items():
        assert "donor_car_quadrant" in g, f"{tp} has no HLA x CAR geometry"
        assert "available" in g["donor_car_quadrant"]
    measured = [tp for tp, g in geom.items() if g["donor_car_quadrant"].get("available")]
    assert measured, "the HLA x CAR gate was measurable at no timepoint"


def test_the_donor_car_gate_reports_margins_for_the_clouds_that_exist(diagnostics_report):
    geom = diagnostics_report["tier2_transfer_validity"]["gate_geometry"]
    engrafted = [g["donor_car_quadrant"] for tp, g in geom.items()
                 if g["donor_car_quadrant"].get("available")
                 and (g["donor_car_quadrant"].get("n_donor") or 0) >= 500]
    assert engrafted, "no timepoint had a donor cloud to measure"
    for q in engrafted:
        assert q.get("hla_boundary_margin_donor_in_sd") is not None
        # The CAR margin is measured from the CAR-NEGATIVE mode inside Donor NK, which a
        # fully-transduced product does not have. Either the margin exists, or the row says
        # why it could not be measured — never a bare null, which reads as a wide margin.
        assert (q.get("car_boundary_margin_in_sd") is not None
                or "UNDEFINED here, not wide" in (q.get("car_reason") or "")), q
        occ = q.get("quadrant_occupancy") or {}
        assert set(occ) >= {"donor_car_pos", "donor_car_neg", "host_car_pos", "host_car_neg"}
        assert abs(sum(occ[k] for k in ("donor_car_pos", "donor_car_neg",
                                        "host_car_pos", "host_car_neg")) - 1.0) < 0.01


def test_a_missing_donor_cloud_is_reported_as_unmeasured_not_as_well_placed(
        diagnostics_report):
    """Pre-infusion there is no donor cloud; that is not evidence the CAR gate is fine."""
    geom = diagnostics_report["tier2_transfer_validity"]["gate_geometry"]
    pre = geom.get("Baseline", {}).get("donor_car_quadrant") or {}
    if pre.get("available") and (pre.get("n_donor") or 0) < 500:
        assert pre.get("car_boundary_margin_in_sd") is None
        assert "NOT evidence" in (pre.get("car_reason") or "")


# ── a cutoff the pipeline never claimed to derive from a valley ───────────────
def test_an_asserted_cutoff_is_surfaced_rather_than_left_silent(diagnostics_report):
    """HLA and CAR come from controls, so every valley check above passes over them."""
    t1 = diagnostics_report["tier1_cutoff_foundation"]["markers"]
    asserted = [f["subject"] for f in diagnostics_report["flags"]
                if f["code"] == "ASSERTED_CUTOFF_NO_VALLEY_EVIDENCE"]
    for key in ("hla", "car"):
        m = t1.get(key) or {}
        if m.get("available") and m.get("valley_supported") is False:
            assert any(s.startswith(f"{key} @") for s in asserted), (
                f"{key} has no valley evidence and no flag says so; asserted={asserted}")


def test_an_asserted_cutoff_points_at_the_only_checks_that_can_judge_it(diagnostics_report):
    for f in diagnostics_report["flags"]:
        if f["code"] == "ASSERTED_CUTOFF_NO_VALLEY_EVIDENCE":
            assert "tier 4" in f["resolution_hint"]
            assert f["measured"]["valley_derived_by_pipeline"] is False


def test_the_expected_files_are_written(diagnostics_report):
    files = diagnostics_report["files"]
    names = {Path(p).name for p in files.values()}
    for expected in ("cutoff_diagnostics_summary.txt", "cutoff_diagnostics.json",
                     "diagnostics_cutoff_audit.csv", "diagnostics_transfer.csv",
                     "diagnostics_controls.csv", "diagnostics_thresholds.csv",
                     "diagnostics_metrics.csv"):
        assert expected in names, f"{expected} missing from {sorted(names)}"
    for p in files.values():
        assert Path(p).is_file()


def test_a_real_run_leaves_a_worklist_pointing_at_tables_that_exist(diagnostics_report):
    """The unit tests build the manifest from a hand-made report; this one from a real run.

    The property that only an end-to-end run can check is that every table an obligation
    names was actually written under the names the notebook will expose — a worklist whose
    ``read:`` line raises NameError sends the reader back to the digest it came from.
    """
    files = diagnostics_report["files"]
    assert "diagnostics_obligations.json" in {Path(p).name for p in files.values()}
    assert "diagnostics_worklist.txt" in {Path(p).name for p in files.values()}

    manifest = json.loads(Path(files["obligations"]).read_text())
    assert manifest["schema"] == "flow_diagnostics_obligations/v1"
    assert manifest["n_promoted"] + manifest["n_not_promoted"] == manifest["n_candidates"]

    # This dataset carries an INJECTED drift, so something must have been left to read.
    assert manifest["items"], "an injected defect produced no obligation"

    on_disk = {Path(p).stem for p in files.values()}
    for item in manifest["items"]:
        for name in item["must_read"]:
            assert name in on_disk, f"{item['id']} points at {name}, which was not written"
        assert item["statement"], item["id"]
        assert item["how"], f"{item['id']} states an obligation with no way to discharge it"

    worklist = Path(files["worklist"]).read_text()
    assert "WORKLIST" in worklist
    for item in manifest["items"][:1]:
        assert item["id"] in worklist


def test_the_environment_can_enforce_the_manifest_a_real_run_wrote(diagnostics_report):
    """The contract between the diagnostics and the harness, checked against a real file.

    Everything else about the submission check is unit-tested against hand-written manifests.
    The one thing those cannot catch is a schema drift between the module that WRITES the
    manifest and the module that READS it — at which point the check silently stops
    enforcing, which looks exactly like a run with nothing to enforce.
    """
    from flow.env.kernel import InProcessKernel
    from flow.env.notebook_env import NotebookEnvironment

    manifest_path = Path(diagnostics_report["files"]["obligations"])
    env = NotebookEnvironment(
        kernel=InProcessKernel(plots_dir=manifest_path.parent / "plots"),
        question="q", dataset_description="d",
        obligations_file=manifest_path,
    )
    env._load_obligations()

    manifest = json.loads(manifest_path.read_text())
    # Advisory items carry no discharge condition and are deliberately not enforced, so the
    # harness's total is the checkable subset — not the whole worklist.
    checkable = [i for i in manifest["items"] if not i["advisory"]]
    assert checkable, "a real run produced no enforceable obligation at all"
    assert len(env._obligations) == len(checkable), "the harness dropped checkable items"
    assert env._obligation_tables == manifest["tables"]
    assert env.evidence_coverage()["enforced"] is True

    # An answer that reads nothing must leave every checkable item undischarged, or the
    # discharge check is matching on something the manifest did not ask for.
    pending = env._undischarged_obligations("NK% rises over time.")
    assert len(pending) == len(checkable)


def test_the_digest_a_real_run_writes_announces_its_worklist(diagnostics_report):
    digest = Path(diagnostics_report["files"]["digest"]).read_text()
    assert "WORKLIST:" in digest
    assert "diagnostics_obligations.json" in digest


def test_the_summary_is_readable_and_leads_with_the_discipline(diagnostics_report):
    summary = Path(diagnostics_report["files"]["summary"]).read_text()
    assert "HOW TO READ THIS FILE" in summary
    assert "MEASUREMENT, not a conclusion" in summary
    assert "REPRODUCTION CHECK" in summary
    assert "WHAT THIS CANNOT CATCH" in summary


def test_uncertainty_envelope_labels_two_separate_error_sources(diagnostics_report):
    env = diagnostics_report["tier3_sensitivity"].get("envelope") or []
    if not env:
        pytest.skip("sensitivity produced no envelope on this dataset")
    for row in env:
        assert "cutoff_range_pp" in row
        assert "counting_ci_halfwidth_pp" in row
        assert row.get("dominant_uncertainty") in ("cutoff", "counting", "comparable", None)
