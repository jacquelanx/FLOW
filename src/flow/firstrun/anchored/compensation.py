"""Auditable FCS spillover parsing and compensation provenance.

The canonical first run uses the spillover matrix embedded in each specimen FCS.  This
module makes that choice explicit and machine-checkable; acquisition control files are
inventoried as independent verification evidence, never silently substituted for an
embedded matrix.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class CompensationError(RuntimeError):
    """Raised when a present compensation matrix is invalid or cannot be applied."""


def _stain_identity(value: str) -> str:
    normalized = Path(str(value)).stem.lower().replace("2f", "")
    if "stained control" in normalized:
        normalized = normalized.split("stained control", 1)[0]
    normalized = normalized.replace("compensation controls", "")
    normalized = normalized.replace("stained control", "")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized[:-1] if len(normalized) > 1 and normalized.endswith("a") else normalized


@dataclass(frozen=True)
class SpilloverMatrix:
    channels: tuple[str, ...]
    values: np.ndarray
    sha256: str
    condition_number: float
    state: str
    reasons: tuple[str, ...]

    def as_audit(self) -> dict[str, Any]:
        return {
            "channels": list(self.channels),
            "dimension": len(self.channels),
            "matrix_sha256": self.sha256,
            "condition_number": self.condition_number,
            "validation_state": self.state,
            "validation_reasons": list(self.reasons),
        }


def _fields(value: str) -> list[str]:
    try:
        row = next(csv.reader(StringIO(str(value)), skipinitialspace=True))
    except Exception as exc:  # pragma: no cover - csv is deliberately permissive
        raise CompensationError(f"spillover CSV parse failed: {exc}") from exc
    return [part.strip() for part in row]


def parse_spillover(value: str) -> SpilloverMatrix:
    """Parse and validate the conventional ``$SPILL``/``$SPILLOVER`` value."""
    fields = _fields(value)
    if not fields or not fields[0]:
        raise CompensationError("spillover value is empty")
    try:
        n = int(fields[0])
    except ValueError as exc:
        raise CompensationError("spillover dimension is not an integer") from exc
    if n < 1:
        raise CompensationError("spillover dimension must be positive")
    expected = 1 + n + n * n
    if len(fields) != expected:
        raise CompensationError(
            f"spillover field count mismatch: expected {expected}, observed {len(fields)}"
        )
    channels = tuple(fields[1 : 1 + n])
    if any(not channel for channel in channels):
        raise CompensationError("spillover contains an empty channel name")
    if len(set(channels)) != n:
        raise CompensationError("spillover channel names are not unique")
    try:
        matrix = np.asarray(fields[1 + n :], dtype=float).reshape(n, n)
    except ValueError as exc:
        raise CompensationError("spillover matrix contains a non-numeric value") from exc
    if not np.isfinite(matrix).all():
        raise CompensationError("spillover matrix contains a non-finite value")
    diag = np.diag(matrix)
    if np.any(diag <= 0):
        raise CompensationError("spillover matrix diagonal must be positive")
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError as exc:
        raise CompensationError("spillover matrix condition number is unavailable") from exc
    if not math.isfinite(condition):
        raise CompensationError("spillover matrix is singular")

    reasons: list[str] = []
    if not np.allclose(diag, 1.0, atol=0.15, rtol=0.15):
        reasons.append("diagonal_not_near_one")
    if condition > 100.0:
        reasons.append("ill_conditioned_gt_100")
    elif condition > 50.0:
        reasons.append("condition_number_gt_50")
    state = "BLOCKED" if condition > 100.0 else ("REVIEW" if reasons else "PASS")
    normalized = {
        "channels": channels,
        "values": [[float(v) for v in row] for row in matrix.tolist()],
    }
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SpilloverMatrix(channels, matrix, digest, condition, state, tuple(reasons))


def acquisition_id(path: str | Path) -> str:
    """Recover source acquisition folder from FLOW's ``folder__file.fcs`` convention."""
    name = Path(path).name
    if "__" in name:
        prefix = name.split("__", 1)[0].strip()
        return prefix or "UNASSIGNED"
    return "UNASSIGNED"


def control_inventory(paths: Iterable[str | Path], classifier) -> list[dict[str, Any]]:
    """Inventory compensation controls without using them as biological specimens."""
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(p) for p in paths), key=lambda p: p.name.lower()):
        if classifier(path.name) != "comp":
            continue
        lower = path.stem.lower()
        target = None
        if "unstained" not in lower and "stained control" in lower:
            target = _stain_identity(lower)
        rows.append({
            "file": path.name,
            "acquisition_id": acquisition_id(path),
            "role": "compensation_control",
            "control_target": target,
            "used_for_canonical_compensation": False,
            "verification_state": "INVENTORIED_NOT_EVALUATED",
        })
    return rows


def summarize_compensation(audits: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize specimen-to-matrix assignment and within-acquisition agreement."""
    records = list(audits)
    by_acquisition: dict[str, set[str]] = {}
    blockers: list[str] = []
    reviews: list[str] = []
    for record in records:
        aid = str(record.get("acquisition_id") or "UNASSIGNED")
        digest = record.get("matrix_sha256")
        if digest:
            by_acquisition.setdefault(aid, set()).add(str(digest))
        compensation_state = str(record.get("compensation_state", ""))
        matrix_state = str(record.get("matrix_validation_state", ""))
        if compensation_state.startswith("BLOCKED") or compensation_state == "APPLICATION_FAILED":
            blockers.append(str(record.get("file") or "unknown"))
        elif compensation_state.startswith("REVIEW") or matrix_state == "REVIEW":
            reviews.append(str(record.get("file") or "unknown"))
    disagreements = {
        aid: sorted(hashes) for aid, hashes in by_acquisition.items()
        if aid != "UNASSIGNED" and len(hashes) > 1
    }
    unassigned = sum(1 for record in records if record.get("acquisition_id") == "UNASSIGNED")
    if blockers or disagreements:
        state = "BLOCKED"
    elif reviews or unassigned:
        state = "REVIEW"
    else:
        state = "PASS"
    return {
        "state": state,
        "n_specimens": len(records),
        "n_matrix_hashes": len({r.get("matrix_sha256") for r in records if r.get("matrix_sha256")}),
        "unassigned_acquisition_count": unassigned,
        "within_acquisition_matrix_disagreement": disagreements,
        "blocked_files": sorted(set(blockers)),
        "review_files": sorted(set(reviews)),
        "canonical_source": "embedded_fcs_spillover",
        "control_policy": "controls inventoried for verification; never silently substituted",
    }


def verify_control_settings(
    specimen_audits: Iterable[dict[str, Any]],
    control_audits: Iterable[dict[str, Any]],
    inventory_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Require acquisition-linked unstained/single-stain controls and voltage agreement."""
    specimens = list(specimen_audits)
    controls = list(control_audits)
    inventory = list(inventory_rows)
    assessments = []
    acquisition_ids = sorted({str(row.get("acquisition_id") or "UNASSIGNED")
                              for row in specimens})
    for aid in acquisition_ids:
        spec = [row for row in specimens if row.get("acquisition_id") == aid]
        ctrl = [row for row in controls if row.get("acquisition_id") == aid]
        inventory_acq = [row for row in inventory if row.get("acquisition_id") == aid]
        ctrl_names = [str(row.get("file", "")) for row in inventory_acq]
        unstained = [name for name in ctrl_names if "unstained control" in name.lower()]
        single_stain_rows = [row for row in inventory_acq
                             if "stained control" in str(row.get("file", "")).lower()
                             and "unstained" not in str(row.get("file", "")).lower()]
        single_stain = [str(row.get("file", "")) for row in single_stain_rows]
        single_targets = [str(row.get("control_target") or "").strip().lower()
                          for row in single_stain_rows]
        if any(not target for target in single_targets):
            # Backward-compatible derivation for older inventory rows.
            single_targets = [_stain_identity(name) for name in single_stain]
        else:
            single_targets = [_stain_identity(target) for target in single_targets]
        target_counts = {target: single_targets.count(target) for target in set(single_targets)}
        duplicate_targets = sorted(target for target, count in target_counts.items() if count > 1)
        expected_dimension = max((int(row.get("matrix_dimension") or 0) for row in spec), default=0)
        expected_channels = sorted({
            str(channel) for row in spec for channel in row.get("matrix_channels", [])
        })
        voltage_values: dict[str, set[float]] = {}
        missing_voltage_files = []
        for row in [*spec, *ctrl]:
            voltage_rows = row.get("detector_voltages", [])
            by_channel = {str(voltage.get("channel")): voltage.get("voltage")
                          for voltage in voltage_rows}
            required = expected_channels or [str(v.get("channel")) for v in voltage_rows]
            missing = [channel for channel in required if by_channel.get(channel) is None]
            if not voltage_rows or missing:
                missing_voltage_files.append({
                    "file": str(row.get("file") or "unknown"),
                    "channels": sorted(set(missing or required)),
                })
            for voltage in voltage_rows:
                value = voltage.get("voltage")
                if value is not None:
                    voltage_values.setdefault(str(voltage.get("channel")), set()).add(float(value))
        mismatches = {channel: sorted(values) for channel, values in voltage_values.items()
                      if len(values) > 1}
        reasons = []
        if aid == "UNASSIGNED":
            reasons.append("acquisition_mapping_ambiguous")
        if not ctrl:
            reasons.append("run_matched_controls_missing")
        if not unstained:
            reasons.append("unstained_control_missing")
        if duplicate_targets:
            reasons.append("duplicate_single_stain_controls")
        if expected_dimension and len(set(single_targets)) < expected_dimension:
            reasons.append(
                f"single_stain_controls_incomplete_{len(set(single_targets))}_of_{expected_dimension}"
            )
        if expected_channels:
            normalized_targets = {_stain_identity(target) for target in single_targets}
            missing_stains = [channel for channel in expected_channels
                              if _stain_identity(channel) not in normalized_targets]
            if missing_stains:
                reasons.append("spill_matrix_channels_without_unique_single_stain")
        else:
            missing_stains = []
        if missing_voltage_files:
            reasons.append("detector_voltage_metadata_missing")
        if mismatches:
            reasons.append("detector_settings_mismatch")
        specimen_review = any(
            str(row.get("compensation_state", "")).startswith("REVIEW")
            or row.get("matrix_validation_state") == "REVIEW" for row in spec
        )
        if specimen_review:
            reasons.append("embedded_matrix_requires_review")
        blocking_reasons = [reason for reason in reasons if reason != "embedded_matrix_requires_review"]
        assessments.append({
            "acquisition_id": aid,
            "state": "BLOCKED" if blocking_reasons else ("REVIEW" if reasons else "PASS"),
            "reasons": reasons,
            "specimen_count": len(spec),
            "control_count": len(ctrl),
            "unstained_control_count": len(unstained),
            "single_stain_control_count": len(single_stain),
            "unique_single_stain_target_count": len(set(single_targets)),
            "duplicate_single_stain_targets": duplicate_targets,
            "missing_spill_matrix_stain_channels": missing_stains,
            "expected_matrix_dimension": expected_dimension,
            "detector_voltage_mismatches": mismatches,
            "detector_voltage_metadata_missing": missing_voltage_files,
        })
    return {
        "state": (
            "BLOCKED" if not assessments or any(row["state"] == "BLOCKED" for row in assessments)
            else "REVIEW" if any(row["state"] == "REVIEW" for row in assessments)
            else "PASS"
        ),
        "acquisitions": assessments,
        "interpretation": (
            "Embedded matrices may be applied provisionally, but fluorescence identity is "
            "blocked until run-matched controls and settings pass verification."
        ),
    }
