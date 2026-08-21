# Governed flow-automation first run

This repository keeps cytometry-specific work under `src/flow/firstrun/`; the generic
ReAct/runtime path remains analysis-free. The governed first-run sequence is:

1. Run the anchored first run once per batch.
2. Freeze `first_run_bundle.json`, including SHA-256 hashes of inputs, source, configuration,
   runtime/dependencies, tables, figures, PDF, and page renders. The manifest is write-once;
   a conflicting rebuild fails instead of overwriting it.
3. Keep the canonical run immutable and require a new run identifier for any changed input,
   source, configuration, or artifact.
4. Require human flow-analyst review before biological interpretation or release.

## Plot contract

- Bivariate event displays use deterministic heat-density fields with 20%, 40%, and 65%
  contours.
- Rectangular retained regions use named colored boxes with fill alpha `0.20`.
- Population highlights are deterministic and do not alter event membership.
- Each sample page begins with two explicitly different scatter views:
  `All analyzed events -> viable cellular scatter gate`, followed by
  `Live-CD45 parent -> lymph density geometry`. The second view converts the linear scatter
  coordinates back to instrument units and separately labels Live-CD45 parent contours,
  deterministic fitted mode centers, the 95% low-SSC fitted-mode contour, and the applied
  lymph-gate membership envelope. It is diagnostic only and cannot alter the canonical gate.
- `lymph_density_geometry.csv` records every plotted fitted-mode center, covariance, weight,
  low-SSC designation, parent count, and applied-gate retention for deterministic replay.
- The composite QC PDF includes methods, acquisition stability, a cross-sample subset table,
  and a full gate sequence for every timepoint. Each page is also rendered to PNG and indexed
  in `report_page_index.csv`.
- Every composite page is also saved as a named PDF under `report_pages/pdf/`. Filenames carry
  the patient, three-digit composite page number, timepoint or page role, and content role.
  `report_page_index.csv` is the authoritative composite-page-to-file mapping.
- `temporal_cell_type_summary.csv` contains one row per timepoint and configured cell-type
  region, with percentage denominator, gated-event count, reference/previous changes, and
  QC/identity state. It also contains K/uL estimates only when `flow.csv` maps the specimen to
  an exact date with one unambiguous `alc.csv` value; missing dates are never imputed.
- `Longitudinal_Cell_Composition_<study>.pdf` places percentages on the top row and exact-date
  ALC-calibrated K/uL estimates on the bottom row when available. Gated event counts are retained
  in the CSV and are the PDF fallback when no validated clinical conversion is present. ALC
  calibrates lymphoid regions; CD14+ abundance requires an appropriate CBC/differential source.
- Every page carries `PROVISIONAL / NOT BIOLOGICALLY VALIDATED` plus deterministic run,
  compensation, and sample-acquisition states. Blocked HLA/CAR outputs are labeled configured
  technical regions rather than validated identities.

## Project-local output layout

Temporary directories are appropriate for disposable development and adversarial replay, but
reviewable runs belong inside the patient/project tree:

```text
<project>/analysis/flow/runs/<run_id>/first_run/
  outputs/
    README.md
    output_index.csv
    legacy_name_map.json
    reports/
      QC_Report_<study>.pdf
      Interactive_Longitudinal_QC_<study>.html
      Longitudinal_Cell_Composition_<study>.pdf
      Acquisition_Cleaning_Sensitivity_<study>.pdf
      report_page_index.csv
      pages/pdf/<study>_QC_page_<NNN>_<timepoint-or-role>.pdf
      pages/png/page_<NNN>.png
    tables/
      multilineage.csv
      multilineage_time_cleaned_sensitivity.csv
      acquisition_cleaning_population_comparison.csv
      temporal_cell_type_summary.csv
      lymph_density_geometry.csv
      composition_shift.csv
      gate_parameters.csv
      unified_cutoffs.csv
    qc/
    membership/
    plots/
    provenance/
  first_run_bundle.json
```

The multi-page composite has one unambiguous name: `QC_Report_<study>.pdf`. The interactive
longitudinal dashboard is `Interactive_Longitudinal_QC_<study>.html`; it is not an HTML
rendering of the composite report. Historical `QC_Report_<study>.html` and `_SSA` aliases are
no longer emitted. `legacy_name_map.json` records their canonical replacements without
duplicating files.

Longitudinal plots group comparable percentages by parent population. B, T, CD4, CD8, NK,
and Donor NK use the CD14-negative lymph region; Lymphocytes and CD14+ use viable live events;
the configured CAR region uses the configured Donor NK region. Titles, legends, hover labels, table
headers, and `temporal_cell_type_summary.csv` all state these denominators explicitly.

The project-local launcher refuses to replace a non-empty run directory:

```bash
python scripts/run_project_first_run.py \
  --project /path/to/UPN27 \
  --data /path/to/frozen/input_snapshot \
  --run-id upn27_flowjo_v22_compensation_time_cleaning \
  --compensation-controls /path/to/machine/readout/control_exports \
  --analysis-contract /path/to/frozen/data_contract.json
```

Use a new run ID whenever canonical inputs, code, configuration, or artifacts change.

## Compensation policy

The embedded `$SPILL`/`$SPILLOVER` matrix in each specimen is the canonical default because it
travels with that specimen and preserves channel order. FLOW validates its dimensions, unique
channels, finite values, diagonal, condition number, application result, and SHA-256.

When a separate control root is supplied, FLOW groups controls by FCS acquisition date plus
cytometer serial, collapses repeated exports by event GUID, requires one unstained and one
unique single-stain control per spill channel, checks detector voltages, derives a robust
median-difference candidate matrix, and measures post-compensation residual spillover. A
control-derived candidate can serve as fallback only for a specimen with the exact acquisition
ID and matching detector settings when its embedded matrix is missing or invalid. A valid
embedded matrix remains selected unless an analyst approves an override after an objective
residual comparison. The selected source, hashes, comparison, and recommendation are written to
`compensation_selection.csv`.

Separate unstained and single-stain controls are never treated as biological specimens. An
applied embedded matrix is
`PROVISIONAL_EMBEDDED_UNVERIFIED` until all of these pass:

- exact specimen-to-acquisition mapping;
- at least one run-matched unstained control;
- a complete single-stain set for the matrix dimensions;
- unique (not duplicate) stain identities matching the spill-matrix channels;
- compatible channels and `$P#V` detector settings;
- no within-acquisition matrix disagreement.

Controls from a different date/acquisition are labeled `REFERENCE_NOT_RUN_MATCHED`; they can
support method comparison but cannot replace a specimen matrix. Missing run-matched controls or
ambiguous acquisition mapping blocks fluorescence identity/release while allowing scatter-only
technical QC to continue.

## Time and detector-setting QC

`acquisition_qc.py` bins the Time channel and reports event-rate gaps/bursts plus concurrent
multi-channel median excursions. These are **signal-instability proxies**, not proof of dynamic
voltage changes. Static `$P#V` values are recorded separately and compared across acquisition-
linked specimens and controls.

Canonical event removal remains `false`. The first run also emits a separately labeled
Time-cleaned sensitivity analysis: only events in predeclared candidate Time bins are removed,
the exact event IDs/reasons are written to a compressed ledger, and canonical gate parameters
are reapplied without refitting. Canonical versus cleaned counts and percentages are reported in
CSV plus `Acquisition_Cleaning_Sensitivity_<study>.pdf`. These candidate exclusions are not
called voltage spikes and cannot be promoted into the canonical run without human review.
Missing Time or voltage metadata is `NOT_EVALUABLE`, never `PASS`.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
pytest -q
```

The analysis container already installs these scientific dependencies. Sensitive outputs,
manual workspaces, membership exports, and rendered patient reports must stay outside Git and
outside unapproved hosted-model prompts.
