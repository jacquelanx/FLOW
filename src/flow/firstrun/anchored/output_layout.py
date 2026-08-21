"""Discoverable, typed output layout for governed FLOW-Jo first runs."""
from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FirstRunOutputLayout:
    """Paths for human-facing outputs while governance manifests remain at run root."""

    root: Path
    outputs: Path
    reports: Path
    report_pages_png: Path
    report_pages_pdf: Path
    tables: Path
    qc: Path
    membership: Path
    provenance: Path
    plots: Path

    @classmethod
    def create(cls, root: str | Path) -> "FirstRunOutputLayout":
        root_path = Path(root)
        outputs = root_path / "outputs"
        layout = cls(
            root=root_path,
            outputs=outputs,
            reports=outputs / "reports",
            report_pages_png=outputs / "reports" / "pages" / "png",
            report_pages_pdf=outputs / "reports" / "pages" / "pdf",
            tables=outputs / "tables",
            qc=outputs / "qc",
            membership=outputs / "membership",
            provenance=outputs / "provenance",
            plots=outputs / "plots",
        )
        for directory in (
            layout.reports, layout.report_pages_png, layout.report_pages_pdf,
            layout.tables, layout.qc, layout.membership, layout.provenance,
            layout.plots,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout

    def write_catalog(self, patient_id: str) -> tuple[Path, Path, Path]:
        """Write a deterministic start-here guide, legacy map, and file index."""
        readme = self.outputs / "README.md"
        readme.write_text(
            "# FLOW-Jo first-run outputs\n\n"
            "Start with `reports/QC_Report_" + patient_id + ".pdf`.\n\n"
            "- `reports/QC_Report_" + patient_id + ".pdf`: multi-page composite QC report.\n"
            "- `reports/Interactive_Longitudinal_QC_" + patient_id + ".html`: interactive "
            "longitudinal dashboard (not an HTML rendering of the composite PDF).\n"
            "- `reports/`: longitudinal PDF plus named composite-page PDFs/PNGs.\n"
            "- `tables/`: population, temporal-change, gate-parameter, and geometry tables.\n"
            "- `qc/`: acquisition, compensation, detector, and release-state evidence.\n"
            "- `membership/`: event-level canonical membership and its coverage manifest.\n"
            "- `plots/`: cross-timepoint marker overlays.\n"
            "- `provenance/`: the operator anchor used by the deterministic run.\n\n"
            "The run-root first-run bundle governs these files. Files ending in "
            "`_SSA` are deprecated aliases and are not emitted by default.\n",
            encoding="utf-8",
        )

        legacy_map = self.outputs / "legacy_name_map.json"
        legacy_map.write_text(json.dumps({
            "schema_version": "flow.legacy_output_name_map.v1",
            "policy": "deprecated or misleading names are mapped, not duplicated",
            "mappings": [
                {
                    "legacy_name": f"QC_Report_{patient_id}_SSA.pdf",
                    "canonical_relative_path": f"reports/QC_Report_{patient_id}.pdf",
                },
                {
                    "legacy_name": f"QC_Report_{patient_id}_SSA.html",
                    "canonical_relative_path": (
                        f"reports/Interactive_Longitudinal_QC_{patient_id}.html"
                    ),
                },
                {
                    "legacy_name": f"QC_Report_{patient_id}.html",
                    "canonical_relative_path": (
                        f"reports/Interactive_Longitudinal_QC_{patient_id}.html"
                    ),
                },
                {
                    "legacy_name": f"Multilineage_SSA_{patient_id}.csv",
                    "canonical_relative_path": "tables/multilineage.csv",
                },
            ],
        }, indent=2, sort_keys=True), encoding="utf-8")

        index = self.outputs / "output_index.csv"
        rows = []
        for path in sorted(self.outputs.rglob("*")):
            if not path.is_file() or path == index:
                continue
            relative = path.relative_to(self.outputs).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "category": relative.split("/", 1)[0],
                "relative_path": relative,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            })
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "category", "relative_path", "media_type", "sha256", "size_bytes",
            ])
            writer.writeheader()
            writer.writerows(rows)
        return readme, legacy_map, index
