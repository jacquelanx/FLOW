# FLOW

**FLOW** is a tool that lets you ask a question about a flow-cytometry dataset, run your
dataset through a first-run analysis script, and have an AI agent interpret and analyze your
results. It works in two stages:

1. A **first-run script** (default provided but can modify) does standard flow-cytometry analysis
   — gating, QC, populations — the same way every time.
2. An **AI agent** then reads those results and does the interpretation: checks the gating,
   makes plots, compares timepoints, and writes a conclusion. Several agents run independently
   and their answers are merged into one consensus.

> **This guide takes you through everything, step by step.** For a quick introduction, follow
> the **[Quickstart](#quickstart-step-by-step)** exactly as written and use the **web app** —
> that is by far the easiest way. If you are interested, the command-line instructions are here
> too.

---

## Table of contents

- [Before you start: what to install](#before-you-start-what-to-install)
- [Quickstart (step by step)](#quickstart-step-by-step)
- [Connecting an AI model (the `.env` file)](#connecting-an-ai-model-the-env-file)
- [Using the web app (detailed walkthrough)](#using-the-web-app-detailed-walkthrough)
- [Using the command line (optional)](#using-the-command-line-optional)
- [How a run works](#how-a-run-works)
- [Data you upload](#data-you-upload)
- [Safety & privacy (please read for patient data)](#safety--privacy-please-read-for-patient-data)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Before you start: what to install

You need four things installed on your computer. Install each one (they're all free), then
continue. **You only do this once.**

| Tool | What it's for | Get it |
|---|---|---|
| **Docker Desktop** | Runs the analysis safely in a sandbox | <https://www.docker.com/products/docker-desktop/> |
| **Python 3.10 or newer** | Runs FLOW | <https://www.python.org/downloads/> |
| **Node.js (v18+)** | Runs the web app | <https://nodejs.org/> (choose "LTS") |
| **git** | Downloads the code | <https://git-scm.com/downloads> |

**Check what you already have.** Open a terminal (macOS: *Terminal* app; Windows: *PowerShell*)
and paste these one at a time — each should print a version number:

```bash
docker --version
python3 --version
node --version
git --version
```

If any command says "command not found," install that tool from the link above, then re-check.

> **Important:** After installing Docker Desktop, **open the Docker Desktop app and leave it
> running.** FLOW runs all analysis inside Docker and will refuse to start work if Docker isn't
> running. You'll know it's ready when the Docker whale icon in your menu bar/taskbar is steady
> (not animating).

---

## Quickstart (step by step)

Do these steps in order. Commands are meant to be **copy-pasted** into your terminal.

### Step 1 — Download the code

Replace `<REPO_URL>` with the address of this this repository!! (it looks like
`https://github.com/your-lab/flow.git`).

```bash
git clone <REPO_URL>
cd flow
```

Everything from here on is run from inside this `flow` folder.

### Step 2 — Install FLOW

We create an isolated Python environment so FLOW doesn't interfere with anything else on your
computer, then install it.

```bash
python3 -m venv .venv                 # create the environment (once)
source .venv/bin/activate             # turn it on  (Windows: .venv\Scripts\activate)
python -m pip install -e ".[dev]"     # install FLOW
```

> You'll need to run `source .venv/bin/activate` again each time you open a **new** terminal to
> use FLOW (or it might auto activate). Your prompt will show `(.venv)` when it's active.

Quick check that it installed:

```bash
flow doctor
```

This checks Docker and reports whether the analysis image is built yet (it isn't — that's the
next step).

### Step 3 — Build the analysis container (one time, ~30 minutes)

This builds the sandboxed environment where all analysis runs. **It's slow the first time**
(it installs a large scientific + bioinformatics toolkit), but you only do it once.

```bash
docker build -t flow-bixbench-env:1.0 -f docker/Dockerfile .
flow doctor        # should now say "All checks passed"
```

> Keep Docker Desktop running while this builds. If it fails partway, just run the same
> `docker build` command again.

### Step 4 — Connect an AI model

FLOW needs an AI model to do the analysis. You tell it which one by creating a small settings
file called `.env`. Copy the provided template:

```bash
cp .env.example .env
```

Now open `.env` in any text editor and fill in your provider(s)'s key. **Which one to pick and
exactly what to paste is fully explained in the next section:
[Connecting an AI model](#connecting-an-ai-model-the-env-file).** Go read that, fill in your
key, save the file, then come back here.

> **Just want to check everything works first?** You can skip the key for now and use the
> built-in `mock` model, which runs offline and needs no key (it won't do real analysis, but it
> proves the whole app works end-to-end). You can add a real key when you're ready.

### Step 5 — Start FLOW (the web app — easiest)

You'll use **two terminal windows**: one for the backend, one for the web app.

**Terminal 1 — start the backend:**

```bash
source .venv/bin/activate     # if not already active
flow serve
```

Leave this running. It automatically reads your `.env` keys.

**Terminal 2 — start the web app:**

```bash
cd flow                       # the project folder, if you're not already there
cd frontend
npm install                   # first time only
npm run dev
```

Now open **<http://localhost:5173>** in your web browser. You should see the FLOW app with a
**"Docker ready"** badge in the bottom-left corner.

### Step 6 — Run your first analysis

In the browser, work top-to-bottom through the four numbered steps in the left sidebar. There's
a full walkthrough in [Using the web app](#using-the-web-app-detailed-walkthrough) below —
in short:

1. **Project & Upload** — create a project, drag in your data files.
2. **Configure** — type your research question, pick the AI model.
3. **Analysis** — provide/adjust the first-run analysis script and describe your panel.
4. **Launch Run** — press **Start run**, then watch it work and read the results.

**When you're done:** press `Ctrl+C` in both terminals to stop FLOW. Next time, you only need
Step 5 (start the two terminals); Steps 1–4 are one-time setup.

---

## Connecting an AI model (the `.env` file)

FLOW talks to an AI model ("provider"). Your key goes in the `.env` file you created in Step 4.
Keys are read **only on your computer** and are **never** sent into the analysis sandbox.

Open `.env` in a text editor, find your provider below, paste your key, and **save** manually.

| Provider | Cost | Good for | What to put in `.env` |
|---|---|---|---|
| `mock` | free | testing that the app works (no real analysis) | nothing — just choose `mock` in the app |
| `azure` | your org's account | **real patient/clinical data** (approved) | `AZURE_OPENAI_*` (see below) |
| `ollama` | free, runs on your machine | private data, no internet | nothing — just run Ollama locally |
| `gemini` | free tier | quick tests with **non-sensitive** data | `GEMINI_API_KEY=...` |
| `groq` | free tier | quick tests with **non-sensitive** data | `GROQ_API_KEY=...` |
| `openai` | paid | high quality with **non-sensitive** data | `OPENAI_API_KEY=...` |

> **Patient data:** Free hosted models (Gemini, Groq, OpenAI, etc.) are **not** approved for
> PHI. For real clinical data with sensitive PHI that might be exposed, use **`azure`** or a
> **local** model (`ollama`), or test the app with `mock`.

### Where do keys come from?

- **Google Gemini** (free): sign in at <https://aistudio.google.com/apikey>, click *Create API
  key*, copy it, and set `GEMINI_API_KEY=` that value in `.env`.
- **Groq** (free): sign in at <https://console.groq.com/keys>, create a key, set
  `GROQ_API_KEY=`.
- **OpenAI** (paid): <https://platform.openai.com/api-keys>, set `OPENAI_API_KEY=`.
- **Ollama** (local, no key): install from <https://ollama.com>, then in a terminal run
  `ollama serve` and `ollama pull qwen2.5-coder`. No key needed.

### Azure OpenAI

Azure needs a few lines instead of one key. In `.env`, set:

```bash
AZURE_OPENAI_API_KEY=<your key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-10-21          # optional; a sensible default is used
# AZURE_OPENAI_AUTH_HEADER=api-key           # only if a gateway needs a different header
```

- Get the **key** and **endpoint** from the Azure portal.
- The endpoint might also include the route prefix, e.g. `https://apimd.mdanderson.edu/dig/foundry`
  Copy the base URL shown for the "Create chat completion" operation in the developer portal.
- The **model** you choose in the app is your Azure **deployment name** (from the portal's
  *Deployments* list), *not* the base model id.
- The deployment must be a **chat model that supports tool/function calling** — `gpt-35-turbo`
  (version 0613+), `gpt-4o`, or any `gpt-4*`/`gpt-5*` chat model. Old completion models
  (`text-davinci-003`) will not work.
- Steps might need to be tweaked a little depending on Azure configs

**After editing `.env`, restart the backend** (`Ctrl+C` in Terminal 1, then `flow serve` again)
so it picks up the new keys.

---

## Using the web app 

The left sidebar is a numbered workflow. Go top to bottom.

### 1 · Project & Upload
- Under **Create a new project**, type a name and click **Create project**.
- Under **Upload data files**, drag your files onto the drop zone (or click to browse). FLOW
  validates each one and shows a table. Typical files: a **folder of FCS files** (you can drop a
  `.zip` of them), plus `metadata.json`, `flow.csv`, and `alc.csv`
  (see [Data you upload](#data-you-upload)).

### 2 · Configure
- **Research question** — the single most important field. Describe, in plain language, what you
  want to find out. Be specific (which populations, which comparisons, exclude which controls).
- **Model** — choose your provider and model. For Azure, type your **deployment name** in the
  model box.
- **First-run** — leave on **Auto** (runs your project's first-run script before the agent).
- **Advanced settings** (optional) — step budget, timeouts, memory/CPU limits, and whether the
  sandbox may access the network.

### 3 · Analysis
Two parts:
- **First-run script** — the Python script that does the actual flow-cytometry analysis. FLOW
  seeds a worked **example** you can edit, or you can **upload your own** `.py`. It's run as
  `python first_run.py --data /data --out OUT --plots PLOTS`, reads your data, and writes result
  tables the agent then interprets. Edit it to match your assay and click **Save script**.
- **Analysis spec** — describe your panel (which detector = which marker), the populations of
  interest, the comparisons you care about, and what the write-up should include. This is turned
  into instructions that steer how the agent interprets the results. Click **Save & continue**.

### 4 · Launch Run
- Confirm the question and model.
- **Trajectories** — how many independent agents analyze the data before their answers are
  merged. Start with **3** for a quick first test; use **8** for a robust consensus.
- Click **Start run.**

### Watching and reading results (left sidebar, "Review run")
- **Live Trajectory** — watch each agent write and run code in real time.
- **Notebook** — the exact cells an agent ran.
- **Results** — the research question, the **consensus** answer up top, and each agent's own
  conclusion below.
- **Artifacts** — the first-run tables (QC, gating, populations, etc.) and plots; **Download run
  (.zip)** saves everything.

**Run History** (sidebar) lists every past run for the project and survives closing the app.

---

## Using the command line (optional)

Prefer the terminal? Everything the app does is available via the `flow` command. Keys are read
from `.env` automatically.

```bash
# Health check (Docker + image ready?)
flow doctor

# Generate a tiny synthetic demo dataset to try things out (no patient data)
python scripts/generate_synthetic_demo.py            # writes examples/demo/

# Prove the whole loop works offline, no key needed:
flow run --data examples/demo \
  --question "Characterize how the CAR-detector signal changes across timepoints." \
  --model mock

# A real run with your data + a model (8 independent trajectories + consensus):
flow run --data path/to/your/project \
  --question "<your research question>" \
  --provider gemini --model gemini-2.0-flash \
  --trajectories 8

# Start just the backend API (e.g. to use the web app):
flow serve --port 8000        # docs at http://localhost:8000/docs
```

Useful `flow run` options: `--provider`, `--model`, `--trajectories`, `--max-steps`,
`--meta-provider`/`--meta-model` (use a different model for the consensus), `--out`. Each run
writes a folder with `notebook.ipynb`, `actions.jsonl`, `answer.txt`, `run.json`, and `plots/`.

---

## How a run works

1. **First-run script** (your project's `first_run.py`) runs first, inside the sandbox, and
   writes result tables (QC, gating, populations, marker expression, …).
2. Those tables are loaded into the agent's notebook, and the agent **interprets** them: reviews
   the gating, checks QC, compares timepoints, makes plots, and writes a conclusion. It has
   exactly two actions — write/run a notebook cell, and submit its answer.
3. **Several agents** do this independently in isolated sandboxes; a final step synthesizes their
   conclusions into **one consensus** answer (pure text — it runs no code and reads no data).

FLOW is a **reproducible scaffold + interpretive agent**: the script does the heavy lifting
deterministically; the agent does the interpretation.

---

## Data you upload

A project is a folder of files. The core ones:

- **A folder of FCS files** (or a combined per-event table `events.csv` with `sample_id`,
  `label`, and one column per detector) — your cytometry data.
- **`metadata.json`** — facts about the experiment (e.g. which detector carries which stain, key
  dates). These help the agent *interpret* channels; they are not gating instructions.
- **`flow.csv`** — timepoint sheet: columns `label,date` (e.g. `Baseline,6/25/2025`).
- **`alc.csv`** — WBC (K/µL) over time: columns `date,alc`. Used for absolute counts.

The **first-run analysis script** and the **analysis spec** are provided on the **Analysis**
page in the app (not uploaded as data files).

A small synthetic demo (no patient data) is generated by
`python scripts/generate_synthetic_demo.py`.

---

## Safety & privacy (please read for patient data)

- **All analysis runs in a locked-down Docker sandbox** — non-root, no network by default,
  resource-limited, read-only access to your data, with a hard timeout. FLOW refuses to run
  analysis on your machine directly.
- **Your API keys stay on your computer** and are never passed into the sandbox.
- **Uploads** are validated, size-limited, and path-checked.
- **Audit trail:** every agent action, code cell, output, and the image digest are saved with
  each run.
- **PHI / clinical data:** the AI provider you choose *does* see summaries/tables derived from
  your data. It's a good idea to make sure that no PHI are in the data you're sending to AI
  providers!

---

## Troubleshooting

- **"Docker not ready" / analysis won't start** — open the Docker Desktop app and wait until the
  whale icon is steady, then reload. Verify with `flow doctor`.
- **`flow: command not found`** — activate the environment first: `source .venv/bin/activate`
  (Windows: `.venv\Scripts\activate`).
- **Backend won't start: "address already in use"** — a previous `flow serve` is still running.
  Stop it (`Ctrl+C` in its terminal) or start on another port: `flow serve --port 8010`. Note:
  `flow serve` does **not** auto-reload on code changes — restart it after editing `.env` or the
  code.
- **The AI keeps failing with "rate limit" (HTTP 429)** — free tiers (e.g. Groq) have small
  daily limits and many trajectories use a lot of tokens. Use fewer trajectories, wait for the
  limit to reset, or switch to a model with more quota (e.g. Azure).
- **macOS: "no such file or directory" when running** — Docker Desktop only shares certain
  folders. Keep your project folder under your home directory (e.g. `~/…`), not `/tmp`.
- **The Docker image build failed** — re-run the `docker build …` command (Step 3); partial
  builds resume. Keep Docker Desktop running throughout.

---

## Limitations

- The quality of the analysis depends on the AI model you choose; the `mock` model does no real
  analysis (it only proves the app works).
- The analysis container is large and slow to build the first time.
- Trajectories run one at a time (for strict isolation), so more trajectories take longer.
- The web app is a single-user local app (no login) — intended to run on your own machine.
