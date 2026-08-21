"""Human-review report for deterministic Time-channel cleaning sensitivity results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATUS_TEXT = "PROVISIONAL / TIME-CLEANING SENSITIVITY / NOT BIOLOGICALLY VALIDATED"


def _status_banner(fig, compensation_state: str) -> None:
    fig.text(
        0.5, 0.985,
        f"{STATUS_TEXT}  ·  COMPENSATION {compensation_state}",
        ha="center", va="top", fontsize=10, weight="bold", color="#8b1a1a",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff0f0",
              "edgecolor": "#b22222", "linewidth": 1.0},
    )


def write_acquisition_cleaning_report(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    intervals: pd.DataFrame,
    output: str | Path,
    *,
    patient_id: str,
    compensation_state: str,
) -> Path:
    """Write a compact before/after report without replacing canonical outputs."""
    output = Path(output)
    with PdfPages(output, metadata={
        "Creator": "FLOW governed Time-cleaning sensitivity",
        "Producer": "FLOW",
        "CreationDate": None,
        "ModDate": None,
    }) as pdf:
        fig = plt.figure(figsize=(14, 8.5))
        _status_banner(fig, compensation_state)
        fig.suptitle(
            f"{patient_id} – acquisition-instability cleaning sensitivity",
            fontsize=20, weight="bold", color="#18395f", y=0.92,
        )
        fig.text(
            0.06, 0.84,
            "The canonical first run remains unchanged. This secondary analysis removes "
            "only events in predeclared Time bins flagged by an event-rate burst or a "
            "concurrent multi-channel signal excursion, then reapplies the same gate "
            "parameters without refitting. These events are not claimed to be voltage spikes.",
            ha="left", va="top", fontsize=11, wrap=True,
        )
        show_columns = [
            "timepoint", "acquisition_qc_state", "analyzed_event_count",
            "excluded_analyzed_event_count", "excluded_analyzed_event_percent",
            "sensitivity_result",
        ]
        display = summary.reindex(columns=show_columns).copy()
        if not display.empty:
            display["excluded_analyzed_event_percent"] = display[
                "excluded_analyzed_event_percent"
            ].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}%")
            display["sensitivity_result"] = display["sensitivity_result"].map(
                lambda value: str(value).replace("_", " ").title()
            )
            display.columns = [
                "Timepoint", "QC", "Analyzed", "Excluded", "Excluded %", "Result"
            ]
            axis = fig.add_axes([0.05, 0.12, 0.90, 0.59])
            axis.axis("off")
            table = axis.table(
                cellText=display.astype(str).values,
                colLabels=display.columns,
                loc="upper center", cellLoc="center", colLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1, 1.28)
            for (row, _), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#dce8f5")
                    cell.set_text_props(weight="bold", color="#18395f")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        flagged = summary.loc[summary["excluded_analyzed_event_count"] > 0]
        for _, sample in flagged.iterrows():
            file_name = str(sample["file"])
            timepoint = str(sample["timepoint"])
            sample_cmp = comparisons.loc[comparisons["file"] == file_name].copy()
            sample_cmp = sample_cmp.loc[sample_cmp["canonical_percent"].notna()]
            sample_cmp["absolute_delta"] = sample_cmp["delta_percentage_points"].abs()
            sample_cmp = sample_cmp.sort_values("absolute_delta", ascending=False).head(10)
            sample_intervals = intervals.loc[intervals["file"] == file_name].copy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [1.2, 1]})
            _status_banner(fig, compensation_state)
            fig.suptitle(
                f"{patient_id} {timepoint}: canonical versus Time-cleaned sensitivity",
                fontsize=18, weight="bold", color="#18395f", y=0.92,
            )
            if sample_cmp.empty:
                axes[0].text(0.5, 0.5, "No comparable population percentages", ha="center")
                axes[0].axis("off")
            else:
                positions = np.arange(len(sample_cmp))
                width = 0.36
                axes[0].barh(
                    positions + width / 2, sample_cmp["canonical_percent"], height=width,
                    label="Canonical", color="#4c78a8",
                )
                axes[0].barh(
                    positions - width / 2, sample_cmp["cleaned_percent"], height=width,
                    label="Time-cleaned sensitivity", color="#f28e2b",
                )
                axes[0].set_yticks(positions)
                axes[0].set_yticklabels(sample_cmp["metric"], fontsize=8)
                axes[0].invert_yaxis()
                axes[0].set_xlabel("Percent (metric-specific denominator)")
                axes[0].set_title("Largest absolute percentage-point changes")
                axes[0].grid(axis="x", alpha=0.25)
                handles, labels = axes[0].get_legend_handles_labels()

            axes[1].axis("off")
            lines = [
                f"Analyzed events: {int(sample['analyzed_event_count']):,}",
                f"Excluded events: {int(sample['excluded_analyzed_event_count']):,} "
                f"({float(sample['excluded_analyzed_event_percent']):.3f}%)",
                f"Full-file candidate events: {int(sample['full_file_candidate_event_count']):,} "
                f"({float(sample['full_file_candidate_event_percent']):.3f}%)",
                f"Candidate Time intervals: {int(sample['candidate_interval_count'])}",
                "",
                "Exact excluded event IDs are stored in",
                "outputs/membership/time_cleaning_excluded_events.csv.gz.",
                "Canonical membership and canonical percentages are unchanged.",
            ]
            if not sample_intervals.empty:
                lines.extend(["", "Flagged intervals (Time units):"])
                for _, interval in sample_intervals.head(14).iterrows():
                    flags = []
                    if bool(interval.get("rate_burst")):
                        flags.append("rate burst")
                    if bool(interval.get("signal_spike_proxy")):
                        flags.append("signal proxy")
                    lines.append(
                        f"{float(interval['start']):.3f}–{float(interval['end']):.3f}: "
                        + ", ".join(flags)
                    )
                if len(sample_intervals) > 14:
                    lines.append(f"… plus {len(sample_intervals) - 14} more intervals")
            axes[1].text(
                0.02, 0.96, "\n".join(lines), ha="left", va="top", fontsize=10.5,
                linespacing=1.35,
            )
            if sample_cmp.empty:
                handles, labels = [], []
            if handles:
                fig.legend(
                    handles, labels, loc="lower center", bbox_to_anchor=(0.38, 0.025),
                    ncol=2, frameon=True,
                )
            fig.subplots_adjust(top=0.82, left=0.10, right=0.96, bottom=0.15, wspace=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output
