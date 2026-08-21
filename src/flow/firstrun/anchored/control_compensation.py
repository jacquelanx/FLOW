"""Derive and audit compensation candidates from single-stain control acquisitions.

The embedded specimen matrix remains canonical when it is valid. A control-derived matrix
is eligible only for a specimen acquired on the same date and cytometer, with a complete,
unambiguous control set and evaluable detector settings. Historical controls are retained as
reference evidence and can never silently replace an embedded matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import flowkit as fk
import numpy as np
import pandas as pd

from .acquisition_qc import detector_voltages
from .compensation import (
    CompensationError,
    SpilloverMatrix,
    _stain_identity,
    acquisition_id,
    acquisition_metadata,
    parse_spillover,
)


def _spillover_text(channels: Sequence[str], values: np.ndarray) -> str:
    fields = [str(len(channels)), *map(str, channels)]
    fields.extend(format(float(value), ".17g") for value in values.ravel())
    return ",".join(fields)


def _raw_frame(path: Path) -> tuple[fk.Sample, pd.DataFrame]:
    sample = fk.Sample(str(path))
    frame = sample.as_dataframe(source="raw")
    frame.columns = [column[0] if isinstance(column, tuple) else column
                     for column in frame.columns]
    return sample, frame


def infer_control_fluorescence_channels(paths: Iterable[str | Path]) -> list[str]:
    """Infer area fluorescence detectors when no specimen embeds a channel list."""
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item).lower()):
        try:
            sample = fk.Sample(str(path))
        except Exception:
            continue
        channels = []
        for channel in map(str, sample.pnn_labels):
            normalized = channel.strip().lower()
            if normalized in {"time", "time-a"}:
                continue
            if normalized.startswith("fsc-") or normalized.startswith("ssc-"):
                continue
            if normalized.endswith("-a"):
                channels.append(channel)
        if channels:
            return channels
    return []


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    if math.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    standard = float(np.nanstd(values))
    return standard if math.isfinite(standard) and standard > 0 else 1.0


def _positive_negative_masks(primary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(primary)
    if int(finite.sum()) < 250:
        raise CompensationError("fewer than 250 finite events in single-stain control")
    lower, upper = np.nanquantile(primary[finite], [0.20, 0.80])
    negative = finite & (primary <= lower)
    positive = finite & (primary >= upper)
    if int(negative.sum()) < 50 or int(positive.sum()) < 50:
        raise CompensationError("single-stain positive/negative quantiles are too small")
    return positive, negative


def _control_target(path: Path) -> str | None:
    lower = path.stem.lower()
    if "unstained" in lower:
        return None
    if "stained control" not in lower:
        return None
    return _stain_identity(lower)


def _event_identity(metadata: Mapping[str, Any], path: Path) -> str:
    guid = metadata.get("guid") or metadata.get("$GUID")
    if guid not in (None, ""):
        return f"guid:{guid}"
    # The fallback distinguishes files conservatively. Whole-file hash differences may be
    # export metadata only, so they are not treated as proof of different event content.
    return "path:" + hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def inspect_control_files(
    paths: Iterable[str | Path], classifier: Callable[[str], str]
) -> list[dict[str, Any]]:
    """Read control provenance/settings and identify repeated exports by event GUID."""
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item).lower()):
        if classifier(path.name) != "comp":
            continue
        try:
            sample = fk.Sample(str(path))
            record = acquisition_metadata(sample.metadata)
            target = _control_target(path)
            rows.append({
                "file": path.name,
                "source_path": str(path.resolve()),
                "acquisition_id": acquisition_id(path, sample.metadata),
                **record,
                "role": "unstained_control" if target is None else "single_stain_control",
                "control_target": target,
                "event_identity": _event_identity(sample.metadata, path),
                "event_count": int(sample.event_count),
                "channels": list(map(str, sample.pnn_labels)),
                "detector_voltages": detector_voltages(sample.metadata, sample.pnn_labels),
                "metadata_state": "READABLE",
                "used_for_canonical_compensation": False,
            })
        except Exception as exc:
            rows.append({
                "file": path.name,
                "source_path": str(path.resolve()),
                "acquisition_id": "UNASSIGNED",
                "role": "compensation_control",
                "control_target": _control_target(path),
                "event_identity": None,
                "event_count": None,
                "channels": [],
                "detector_voltages": [],
                "metadata_state": "UNREADABLE",
                "reason": str(exc),
                "used_for_canonical_compensation": False,
            })
    return rows


def _channel_target_map(channels: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for channel in channels:
        identity = _stain_identity(channel)
        if identity in mapping and mapping[identity] != channel:
            raise CompensationError(f"ambiguous normalized channel identity: {identity}")
        mapping[identity] = str(channel)
    return mapping


def _deduplicate_acquisition_rows(
    rows: Sequence[dict[str, Any]], channels: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    reasons: list[str] = []
    channel_map = _channel_target_map(channels)
    selected: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, Any] = {}
    for identity, channel in channel_map.items():
        matches = [row for row in rows if row.get("control_target") == identity]
        by_event: dict[str, list[dict[str, Any]]] = {}
        for row in matches:
            by_event.setdefault(str(row.get("event_identity")), []).append(row)
        if not matches:
            reasons.append(f"missing_single_stain:{channel}")
            continue
        if len(by_event) != 1:
            reasons.append(f"ambiguous_single_stain:{channel}:{len(by_event)}_event_sets")
            continue
        chosen_group = next(iter(by_event.values()))
        chosen = sorted(chosen_group, key=lambda row: row["source_path"])[0]
        selected[channel] = chosen
        duplicates[channel] = {
            "event_identity": chosen.get("event_identity"),
            "export_count": len(chosen_group),
            "selected_source_path": chosen["source_path"],
        }
    unstained = [row for row in rows if row.get("role") == "unstained_control"]
    unstained_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in unstained:
        unstained_by_event.setdefault(str(row.get("event_identity")), []).append(row)
    if not unstained:
        reasons.append("missing_unstained_control")
    elif len(unstained_by_event) != 1:
        reasons.append(f"ambiguous_unstained:{len(unstained_by_event)}_event_sets")
    else:
        group = next(iter(unstained_by_event.values()))
        duplicates["unstained"] = {
            "event_identity": group[0].get("event_identity"),
            "export_count": len(group),
            "selected_source_path": sorted(row["source_path"] for row in group)[0],
        }
    return selected, duplicates, reasons


def _control_voltage_signature(
    rows: Sequence[dict[str, Any]], channels: Sequence[str]
) -> tuple[dict[str, float], list[str]]:
    values: dict[str, set[float]] = {str(channel): set() for channel in channels}
    missing: dict[str, int] = {str(channel): 0 for channel in channels}
    for row in rows:
        by_channel = {
            str(item.get("channel")): item.get("voltage")
            for item in row.get("detector_voltages", [])
        }
        for channel in channels:
            value = by_channel.get(str(channel))
            if value is None:
                missing[str(channel)] += 1
            else:
                values[str(channel)].add(float(value))
    reasons = []
    for channel in channels:
        if missing[str(channel)]:
            reasons.append(f"detector_voltage_missing:{channel}")
        if len(values[str(channel)]) > 1:
            reasons.append(f"detector_voltage_mismatch_within_controls:{channel}")
    signature = {
        channel: next(iter(channel_values))
        for channel, channel_values in values.items() if len(channel_values) == 1
    }
    return signature, reasons


def derive_control_matrix(
    rows: Sequence[dict[str, Any]], channels: Sequence[str]
) -> dict[str, Any]:
    """Derive a robust median-difference spillover candidate for one acquisition."""
    acquisition_ids = sorted({str(row.get("acquisition_id")) for row in rows})
    if len(acquisition_ids) != 1:
        raise CompensationError("control derivation requires exactly one acquisition")
    selected, duplicate_exports, reasons = _deduplicate_acquisition_rows(rows, channels)
    voltage_signature, voltage_reasons = _control_voltage_signature(rows, channels)
    reasons.extend(voltage_reasons)
    if reasons:
        return {
            "acquisition_id": acquisition_ids[0],
            "state": "BLOCKED",
            "reasons": reasons,
            "channels": list(channels),
            "matrix_sha256": None,
            "spillover": None,
            "eligible_for_same_acquisition_fallback": False,
            "eligibility_scope": "exact acquisition ID plus detector-setting match only",
            "duplicate_exports": duplicate_exports,
            "control_files": {},
            "detector_voltage_signature": voltage_signature,
        }

    values = np.eye(len(channels), dtype=float)
    separations: list[dict[str, Any]] = []
    control_frames: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for row_index, primary_channel in enumerate(channels):
        source = Path(selected[primary_channel]["source_path"])
        _, frame = _raw_frame(source)
        missing = sorted(set(channels) - set(map(str, frame.columns)))
        if missing:
            reasons.append(
                f"{primary_channel}:control_missing_channels:" + "|".join(missing)
            )
            continue
        numeric = frame.loc[:, list(channels)].apply(pd.to_numeric, errors="coerce")
        primary = numeric[primary_channel].to_numpy(dtype=float)
        try:
            positive, negative = _positive_negative_masks(primary)
        except CompensationError as exc:
            reasons.append(f"{primary_channel}:{exc}")
            continue
        pos_median = numeric.loc[positive].median(axis=0).to_numpy(dtype=float)
        neg_median = numeric.loc[negative].median(axis=0).to_numpy(dtype=float)
        delta = pos_median - neg_median
        primary_delta = float(delta[row_index])
        separation = primary_delta / _robust_scale(primary[negative])
        separations.append({
            "primary_channel": primary_channel,
            "negative_event_count": int(negative.sum()),
            "positive_event_count": int(positive.sum()),
            "primary_delta": primary_delta,
            "robust_separation": float(separation),
        })
        if not math.isfinite(primary_delta) or primary_delta <= 0:
            reasons.append(f"{primary_channel}:nonpositive_primary_signal_delta")
            continue
        if not math.isfinite(separation) or separation < 5.0:
            reasons.append(f"{primary_channel}:robust_separation_below_5")
            continue
        values[row_index, :] = delta / primary_delta
        values[row_index, row_index] = 1.0
        control_frames[primary_channel] = (numeric, positive, negative)

    if reasons:
        return {
            "acquisition_id": acquisition_ids[0],
            "state": "BLOCKED",
            "reasons": reasons,
            "channels": list(channels),
            "matrix_sha256": None,
            "spillover": None,
            "eligible_for_same_acquisition_fallback": False,
            "eligibility_scope": "exact acquisition ID plus detector-setting match only",
            "duplicate_exports": duplicate_exports,
            "control_files": {
                channel: row["source_path"] for channel, row in selected.items()
            },
            "positive_negative_separation": separations,
            "detector_voltage_signature": voltage_signature,
        }

    parsed = parse_spillover(_spillover_text(channels, values))
    state = "PASS" if parsed.state == "PASS" else parsed.state
    residuals = evaluate_control_residuals(parsed, control_frames)
    return {
        "acquisition_id": acquisition_ids[0],
        "state": state,
        "reasons": list(parsed.reasons),
        "channels": list(parsed.channels),
        "values": parsed.values.tolist(),
        "matrix_sha256": parsed.sha256,
        "condition_number": parsed.condition_number,
        "spillover": _spillover_text(parsed.channels, parsed.values),
        "eligible_for_same_acquisition_fallback": state == "PASS",
        "eligibility_scope": "exact acquisition ID plus detector-setting match only",
        "derivation_method": "top20_bottom20_robust_median_difference.v1",
        "duplicate_exports": duplicate_exports,
        "control_files": {
            channel: row["source_path"] for channel, row in selected.items()
        },
        "positive_negative_separation": separations,
        "detector_voltage_signature": voltage_signature,
        "residual_summary": residuals["summary"],
        "residual_details": residuals["details"],
    }


def evaluate_control_residuals(
    matrix: SpilloverMatrix,
    control_frames: Mapping[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Measure off-diagonal positive-minus-negative residuals after compensation."""
    details: list[dict[str, Any]] = []
    inverse = np.linalg.inv(matrix.values)
    channels = list(matrix.channels)
    for primary_index, primary_channel in enumerate(channels):
        if primary_channel not in control_frames:
            continue
        numeric, positive, negative = control_frames[primary_channel]
        raw = numeric.loc[:, channels].to_numpy(dtype=float)
        compensated = raw @ inverse
        delta = np.nanmedian(compensated[positive], axis=0) - np.nanmedian(
            compensated[negative], axis=0
        )
        primary_delta = float(delta[primary_index])
        for secondary_index, secondary_channel in enumerate(channels):
            if secondary_index == primary_index:
                continue
            residual = (
                float(delta[secondary_index] / primary_delta)
                if math.isfinite(primary_delta) and abs(primary_delta) > 1e-12 else math.nan
            )
            details.append({
                "primary_channel": primary_channel,
                "secondary_channel": secondary_channel,
                "residual_ratio": residual,
                "absolute_residual_ratio": abs(residual) if math.isfinite(residual) else None,
            })
    finite = np.asarray([
        row["absolute_residual_ratio"] for row in details
        if row["absolute_residual_ratio"] is not None
    ], dtype=float)
    summary = {
        "off_diagonal_pair_count": int(len(finite)),
        "median_absolute_residual_ratio": (
            float(np.median(finite)) if len(finite) else None
        ),
        "p95_absolute_residual_ratio": (
            float(np.quantile(finite, 0.95)) if len(finite) else None
        ),
        "maximum_absolute_residual_ratio": (
            float(np.max(finite)) if len(finite) else None
        ),
    }
    return {"summary": summary, "details": details}


def build_control_repository(
    paths: Iterable[str | Path],
    classifier: Callable[[str], str],
    expected_channels: Sequence[str],
) -> dict[str, Any]:
    """Inventory repeated exports and derive one candidate per acquisition."""
    rows = inspect_control_files(paths, classifier)
    candidates = []
    readable = [row for row in rows if row.get("metadata_state") == "READABLE"]
    for aid in sorted({str(row["acquisition_id"]) for row in readable}):
        group = [row for row in readable if row["acquisition_id"] == aid]
        candidate = derive_control_matrix(group, expected_channels)
        first = group[0]
        candidate.update({
            "acquisition_date": first.get("acquisition_date"),
            "cytometer": first.get("cytometer"),
            "cytometer_serial": first.get("cytometer_serial"),
            "control_export_count": len(group),
            "unique_event_identity_count": len({row.get("event_identity") for row in group}),
        })
        candidates.append(candidate)
    state = (
        "BLOCKED" if not candidates or all(row["state"] == "BLOCKED" for row in candidates)
        else "REVIEW" if any(row["state"] != "PASS" for row in candidates)
        else "PASS"
    )
    return {
        "schema_version": "flow.compensation_control_repository.v1",
        "state": state,
        "expected_channels": list(expected_channels),
        "controls": rows,
        "control_sets": candidates,
    }


def candidate_for_acquisition(
    repository: Mapping[str, Any],
    aid: str,
    *,
    specimen_voltages: Sequence[dict[str, Any]] | None = None,
    require_voltage_match: bool = False,
) -> dict[str, Any] | None:
    matches = [row for row in repository.get("control_sets", [])
               if row.get("acquisition_id") == aid and row.get("state") == "PASS"]
    if len(matches) == 1:
        candidate = dict(matches[0])
        if require_voltage_match:
            if specimen_voltages is None:
                return None
            specimen_by_channel = {
                str(row.get("channel")): row.get("voltage") for row in specimen_voltages
            }
            expected = candidate.get("detector_voltage_signature", {})
            if not expected or any(
                specimen_by_channel.get(channel) is None
                or float(specimen_by_channel[channel]) != float(value)
                for channel, value in expected.items()
            ):
                return None
        return candidate
    return None


def _matrix_from_audit(audit: Mapping[str, Any]) -> SpilloverMatrix | None:
    channels = audit.get("matrix_channels") or []
    values = audit.get("matrix_values")
    if not channels or values is None:
        return None
    array = np.asarray(values, dtype=float)
    return parse_spillover(_spillover_text(channels, array))


def compare_specimens_to_control_repository(
    specimen_audits: Sequence[dict[str, Any]], repository: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind each specimen to run-matched or historical control evidence."""
    control_sets = list(repository.get("control_sets", []))
    rows: list[dict[str, Any]] = []
    residual_details: list[dict[str, Any]] = []
    residual_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for audit in specimen_audits:
        aid = str(audit.get("acquisition_id"))
        matched = [row for row in control_sets if row.get("acquisition_id") == aid]
        eligible_candidate = candidate_for_acquisition(
            repository,
            aid,
            specimen_voltages=audit.get("detector_voltages"),
            require_voltage_match=True,
        )
        valid_matched = [eligible_candidate] if eligible_candidate is not None else []
        matrix = _matrix_from_audit(audit)
        comparison_state = "REFERENCE_NOT_RUN_MATCHED"
        embedded_residual = None
        candidate_residual = None
        matrix_max_delta = None
        matrix_frobenius_delta = None
        recommendation = "KEEP_EMBEDDED_PROVISIONAL"
        matched_candidate_hash = None
        if len(valid_matched) == 1:
            candidate = valid_matched[0]
            matched_candidate_hash = candidate.get("matrix_sha256")
            candidate_residual = candidate.get("residual_summary")
            if matrix is not None and tuple(matrix.channels) == tuple(candidate.get("channels", [])):
                candidate_values = np.asarray(candidate["values"], dtype=float)
                delta = matrix.values - candidate_values
                matrix_max_delta = float(np.max(np.abs(delta)))
                matrix_frobenius_delta = float(np.linalg.norm(delta))
                cache_key = (matrix.sha256, str(candidate.get("matrix_sha256")))
                if cache_key not in residual_cache:
                    frames = {}
                    for primary, source in candidate.get("control_files", {}).items():
                        _, frame = _raw_frame(Path(source))
                        numeric = frame.loc[:, list(matrix.channels)].apply(
                            pd.to_numeric, errors="coerce"
                        )
                        positive, negative = _positive_negative_masks(
                            numeric[primary].to_numpy(dtype=float)
                        )
                        frames[primary] = (numeric, positive, negative)
                    residual_cache[cache_key] = evaluate_control_residuals(matrix, frames)
                evaluated = residual_cache[cache_key]
                embedded_residual = evaluated["summary"]
                for detail in evaluated["details"]:
                    residual_details.append({
                        "file": audit.get("file"),
                        "matrix_source": "embedded_fcs_spillover",
                        "matrix_sha256": matrix.sha256,
                        **detail,
                    })
                embedded_p95 = embedded_residual.get("p95_absolute_residual_ratio")
                candidate_p95 = (candidate_residual or {}).get(
                    "p95_absolute_residual_ratio"
                )
                materially_better = (
                    embedded_p95 is not None and candidate_p95 is not None
                    and embedded_p95 > 0.02
                    and candidate_p95 <= 0.5 * embedded_p95
                )
                if materially_better:
                    comparison_state = "REVIEW_CONTROL_DERIVED_MATERIALLY_LOWER_RESIDUAL"
                    recommendation = "ANALYST_REVIEW_CONTROL_DERIVED_OVERRIDE"
                else:
                    comparison_state = "RUN_MATCHED_CONTROL_COMPARISON_PASS"
                    recommendation = "KEEP_EMBEDDED"
            elif audit.get("compensation_source") == "control_derived_run_matched":
                comparison_state = "CONTROL_DERIVED_USED_FOR_MISSING_OR_INVALID_EMBEDDED"
                recommendation = "KEEP_CONTROL_DERIVED"
            else:
                comparison_state = "BLOCKED_MATRIX_CHANNEL_MISMATCH"
                recommendation = "BLOCK_FLUORESCENCE"
        elif matched and not valid_matched:
            comparison_state = "BLOCKED_RUN_MATCHED_CONTROL_SET_OR_SETTINGS_INVALID"
            recommendation = "BLOCK_FLUORESCENCE"
        elif len(matched) > 1 or len(valid_matched) > 1:
            comparison_state = "BLOCKED_AMBIGUOUS_RUN_MATCHED_CONTROL_SET"
            recommendation = "BLOCK_FLUORESCENCE"
        rows.append({
            "file": audit.get("file"),
            "acquisition_id": aid,
            "acquisition_date": audit.get("acquisition_date"),
            "cytometer_serial": audit.get("cytometer_serial"),
            "selected_compensation_source": audit.get("compensation_source"),
            "selected_matrix_sha256": audit.get("matrix_sha256"),
            "run_matched_control_set_count": len(matched),
            "matched_control_candidate_sha256": matched_candidate_hash,
            "control_comparison_state": comparison_state,
            "recommendation": recommendation,
            "matrix_max_absolute_delta": matrix_max_delta,
            "matrix_frobenius_delta": matrix_frobenius_delta,
            "embedded_p95_absolute_residual_ratio": (
                (embedded_residual or {}).get("p95_absolute_residual_ratio")
            ),
            "control_derived_p95_absolute_residual_ratio": (
                (candidate_residual or {}).get("p95_absolute_residual_ratio")
            ),
            "reference_control_acquisition_ids": "|".join(sorted({
                str(row.get("acquisition_id")) for row in control_sets
                if row.get("acquisition_id") != aid
            })),
        })
    overall = (
        "BLOCKED" if any(str(row["control_comparison_state"]).startswith("BLOCKED")
                         for row in rows)
        else "REVIEW" if any(row["control_comparison_state"] in {
            "REFERENCE_NOT_RUN_MATCHED",
            "REVIEW_CONTROL_DERIVED_MATERIALLY_LOWER_RESIDUAL",
        } for row in rows)
        else "PASS"
    )
    return {
        "schema_version": "flow.compensation_selection.v1",
        "state": overall,
        "policy": {
            "canonical_default": "embedded_fcs_spillover",
            "control_derived_fallback": "only complete PASS run-matched controls",
            "valid_embedded_override": "analyst approval required after residual comparison",
            "reference_controls_can_override": False,
        },
        "specimens": rows,
        "embedded_residual_details": residual_details,
        "content_sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
