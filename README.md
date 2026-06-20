# FLOW

**FLOW** is a clean, minimal, faithful re-implementation of **Finch** — FutureHouse's
autonomous, Jupyter-native data-analysis agent ("A multi-agent system for automating
scientific discovery") — plus a simple web UI so non-computational lab members can drive
it.

FLOW is a **generative reasoning engine**, not a pipeline. Given a dataset and a research
question, the agent writes original, executable Python/R code line-by-line inside a Docker
container, observes the output, and adapts — exactly like Finch. There is **zero hardcoded
analysis logic** anywhere in `src/`: no gates, thresholds, cell-population definitions,
transforms, or dataset-specific branches. The agent derives all analysis at runtime from
the data + the question + `metadata.json`.

---

## What Finch is, and where each piece lives (fidelity checklist)

| Finch spec | FLOW implementation |
|---|---|
| Autonomous, Jupyter-native data-analysis agent | [`agent/react.py`](src/flow/agent/react.py) + [`env/notebook_env.py`](src/flow/env/notebook_env.py) |
| Built on the **Aviary** framework (Environment / Tool / Message) | [`flow.aviary`](src/flow/aviary/) (minimal Aviary-shaped core; see ADR-001) |
| Processes bioinformatics workflows (RNA-seq DE, flow cytometry, …) | Generic; the bio stack lives in [`docker/Dockerfile`](docker/Dockerfile), not in code |
| **Generative**: writes ORIGINAL code in real time, not templates | [`providers/`](src/flow/providers/) + [`agent/prompts.py`](src/flow/agent/prompts.py); **no analysis in src** (enforced by [`tests/test_no_hardcoded_analysis.py`](tests/test_no_hardcoded_analysis.py)) |
| Adapts to the data distribution | The model decides everything from observations; FLOW supplies only the environment |
| **ReAct** prompting strategy | [`agent/react.py`](src/flow/agent/react.py), [`agent/prompts.py`](src/flow/agent/prompts.py) |
| Each trajectory runs inside a pre-built Docker container (BixBench-env) | [`docker/Dockerfile`](docker/Dockerfile) → image `flow-bixbench-env:1.0`; [`env/kernel.py`](src/flow/env/kernel.py) |
| Can install new dependencies competently | network opt-in (`safety.allow_network`) in [`config.py`](src/flow/config.py) + [`docker_runtime.py`](src/flow/env/docker_runtime.py) |
| Interacts via **exactly two tools** | [`env/notebook_env.py`](src/flow/env/notebook_env.py) registers only `edit_cell` + `submit_answer` |
| `edit_cell`: select, modify, execute cells | `NotebookEnvironment._edit_cell` |
| `submit_answer`: finalize the conclusion | `NotebookEnvironment._submit_answer` |
| Persistent kernel; state persists across cell edits | [`env/kernel_server.py`](src/flow/env/kernel_server.py) (persistent namespace) + `DockerKernel` (one long-lived container) |
| Observations = truncated outputs; images summarized | `notebook_env._format_observation` (truncation) + `kernel_server` (`[image: …]`) |
| Environment never completes the analysis; no-submit ⇒ fail | `react.py` (budget exhaustion ⇒ failed trajectory) |
| Auditable trajectory artifacts | [`trajectory.py`](src/flow/trajectory.py): `notebook.ipynb`, `actions.jsonl`, `answer.txt`, `run.json` (image digest, limits) |
| Safety: Docker-only, RO mount, network none, limits, non-root | [`env/docker_runtime.py`](src/flow/env/docker_runtime.py) + [`tests/test_safety.py`](tests/test_safety.py) |
| API + UI for lab members | [`api/app.py`](src/flow/api/app.py), [`frontend/`](frontend/) |

---

## Non-negotiable principles

1. **No hardcoded analysis.** Zero domain/analysis logic in `src/`. Grepping for analysis
   terms in `src/` returns nothing outside prompts/tests/docs (enforced by a test).
2. **Exactly two agent tools:** `edit_cell` and `submit_answer`. No "run the pipeline"
   shortcut, no templated-analysis engine.
3. **All agent code runs in Docker, never on the host.** If Docker is unavailable, FLOW
   refuses with an actionable message.
4. Everything is config/argument/upload-driven; no scientific domain is baked in.
5. The user-editable config contains **no** analysis logic — only dataset locations, the
   research question, runtime, and safety. The loader actively rejects analysis keys.

---

## Data formats (what lab members upload)

A project is a folder of:

- **`metadata.json`** — experiment FACTS (not analysis): e.g. which detector carries which
  stain, key dates. Example fields: `patient_study`, `infusion_date`, `hla_specificity`,
  `hla_channel`, `hla_polarity`, `car_channel`, `panel_notes`. These let the agent
  *interpret* channels; they are **not** gating instructions.
- **`flow.csv`** — timepoint sheet: columns `label,date` (e.g. `Baseline,6/25/2025`).
- **`alc.csv`** — absolute lymphocyte counts over time: columns `date,alc`. The agent
  decides whether/how to use it.
- **cytometry data** — either (a) a combined per-event table (CSV/parquet) with `sample_id`,
  `label`, and one column per detector/channel; or (b) a directory of FCS files + `flow.csv`.
- **`config.yaml`** — analysis-free: dataset locations, the research question, runtime, safety.
- **`true_lab_results.csv`** (optional) — ground truth for EVALUATION only; never read during
  analysis.

A small **synthetic** demo (no PHI, generic detector names) is generated by
`scripts/generate_synthetic_demo.py`.

---

## Install

```bash
python -m pip install -e ".[dev]"
```

## Build the container (BixBench-env equivalent)

```bash
docker build -t flow-bixbench-env:1.0 -f docker/Dockerfile .
flow doctor          # verifies Docker + the image
```

> The image pins a broad Python + R/Bioconductor bio stack (numpy/pandas/scipy/scikit-learn,
> matplotlib/seaborn/statsmodels, anndata/scanpy, fcsparser/flowio/flowkit, umap-learn/leidenalg,
> jupyter/nbclient; R with flowCore/openCyto/CATALYST/FlowSOM/DESeq2/edgeR/limma/IRkernel). It
> contains **no** analysis logic — only libraries and a non-root user.

## Generate the demo project

```bash
python scripts/generate_synthetic_demo.py     # writes examples/demo/
```

## Run via the CLI

```bash
# Offline mock loop end-to-end (proves the loop; performs no real analysis):
flow run --data examples/demo --question "Characterize the CAR-detector signal trend across timepoints." --model mock

# Real analysis with a hosted free-tier model (key on the host; see below):
flow run --data examples/demo --question "<your question>" --provider gemini --model gemini-2.0-flash
```

Each run produces a trajectory directory containing `notebook.ipynb`, `actions.jsonl`,
`answer.txt`, `run.json`, and any `plots/`.

## Run via the web UI

```bash
flow serve --port 8000           # backend API + interactive docs at http://localhost:8000/docs
cd frontend && npm install && npm run dev    # UI at http://localhost:5173 (proxies /api)
```

Then in the UI: create a project → drag-drop the files → edit the question/config → launch a
run (choose provider/model) → watch the live trajectory → view results and artifacts. Run
history persists in the backend DB and survives refresh/restart.

---

## Provider / model setup (including free options)

LLM calls run **host-side**; API keys live on the host and never enter the container. Keys
are read from environment variables only.

| Provider | env var | example model | notes |
|---|---|---|---|
| `mock` | — | `mock` | offline; loop only, no real analysis (tests/demos) |
| `ollama` | — | `qwen2.5-coder` | local; `ollama serve` + `ollama pull qwen2.5-coder` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` | Google AI Studio free tier |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Groq free tier |
| `openrouter` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct` | many free models |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` | |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | |

```bash
export GEMINI_API_KEY=...     # then: flow run --provider gemini --model gemini-2.0-flash ...
```

---

## Safety model

- **Docker-only execution.** `flow doctor` and runtime checks hard-fail with guidance if
  Docker/the image is missing. The backend enforces this too.
- **Mounts.** The uploaded dataset is mounted **read-only**; there is one writable workdir.
  No other host paths are exposed.
- **Resource limits.** Per container: `--memory`, `--cpus`, `--pids-limit`, per-cell and
  per-trajectory wall-clock timeouts, and a hard kill switch on timeout.
- **Network.** Default `--network none`; dependency installs require explicit opt-in
  (`safety.allow_network: true`) — documented as a risk.
- **Hardening.** Non-root user, `--cap-drop=ALL`, `no-new-privileges`, read-only rootfs +
  tmpfs workdir.
- **Secrets / privacy.** LLM keys stay on the host; never passed into the container. The
  privacy boundary defaults to sending only schemas/summaries to a hosted model; sending
  full raw event data is gated behind `safety.allow_raw_data_to_model`.
- **Uploads.** Validated, size-limited, and traversal-checked (`safe_filename`).
- **Audit trail.** Every tool call, cell source, output, the image digest, and the limits
  are recorded in the trajectory directory.

---

## Limitations

- The agent's quality depends entirely on the chosen model; the `mock` provider performs no
  real analysis.
- FLOW reproduces Finch's single-agent ReAct data-analysis loop and the two-tool interface;
  it does not implement FutureHouse's broader multi-agent orchestration.
- The Bioconductor image is large and slow to build; some R packages may need a runtime
  install (network opt-in) if a prebuilt binary is unavailable.
- On macOS, Docker Desktop only bind-mounts shared paths (e.g. under your home directory);
  keep project folders under a shared location.

See [`DESIGN.md`](DESIGN.md) for architecture decisions (ADRs).
