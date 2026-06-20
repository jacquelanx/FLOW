# FLOW — Design & Architecture Decision Records

FLOW reproduces Finch's behavior with the smallest, cleanest codebase that stays faithful.
This document records the key decisions.

## System overview

```
                 ┌──────────────┐      one ToolCall/step      ┌────────────────────────┐
   provider ◄────┤  ReActAgent  ├────────────────────────────►│  NotebookEnvironment   │
 (host-side)     └──────────────┘   obs / reward / done       │  (Aviary-shaped)       │
   gemini/groq/         ▲                                      │  tools: edit_cell,     │
   ollama/mock          │ system prompt + task                │         submit_answer  │
                        │                                      └───────────┬────────────┘
                        │                                                  │ cell source
                        │                                                  ▼
                        │                                        ┌──────────────────┐
                        │                                        │   DockerKernel    │  persistent
                        │                                        │  (docker run -i)  │  kernel,
                        │                                        └─────────┬─────────┘  RO data mount,
                        │                                                  │            net=none,
                        │                                       ┌──────────▼──────────┐ limits, non-root
                        │                                       │  kernel_server.py    │
                        │                                       │  (in container)      │
                        │                                       └──────────────────────┘
   FastAPI (api/app.py) ── delegates runs to runner.py ── artifacts (trajectory.py) ── SQLite (db.py)
   React UI (frontend/) ── talks only to /api
```

---

## ADR-001 — Aviary-shaped minimal core, not `fhaviary`/`ldp`

**Decision.** Implement a small, dependency-light `flow.aviary` package (`Message`, `Tool`,
`ToolCall`, `Environment`) that mirrors FutureHouse's `fhaviary` interface, and a minimal
ReAct loop, instead of depending on `fhaviary` + `ldp`.

**Why.** Priorities are faithful behavior, safety, docs, zero hardcoding, and an accessible
UI — with a *clean, minimal* codebase. The Aviary contract that matters for fidelity is the
environment/tool boundary (`reset()/step()/tools()`, one tool call per step, observations as
returns). That is small and stable. Pulling the full frameworks adds a large transitive
dependency tree and indirection without changing the agent's observable behavior.

**Consequences.** The boundary is intentionally identical to `fhaviary`, so
`NotebookEnvironment` can be re-parented onto the real `fhaviary.Environment` and the loop
swapped for `ldp` with no change to tools, kernel, runtime, API, or UI. Tests assert the
two-tool constraint and the loop semantics directly.

---

## ADR-002 — Persistent kernel via one long-lived container process

**Decision.** Each trajectory starts **one** container (`docker run -i`) that runs
`kernel_server.py`, which keeps a persistent Python namespace and serves cells over
stdin/stdout (line-delimited JSON, sentinel-terminated). We do **not** re-execute the whole
notebook each step, and we do **not** use `docker exec` per cell (which would not share
state).

**Why.** Real notebook semantics require state to persist across `edit_cell` calls
(variables, imports, loaded data). A single persistent interpreter is the simplest faithful
model and avoids re-running expensive setup on every step.

**Consequences.** Per-cell timeouts are enforced host-side; on timeout the container is
force-killed. The same `execute_code` function is reused by the host-side `InProcessKernel`
(tests only), so there is exactly one "run a cell" code path. Matplotlib figures are saved
to the workdir and summarized to the agent as `[image]` (no raw bytes in the context).

---

## ADR-003 — Docker-only execution, enforced everywhere

**Decision.** Agent code executes only inside the hardened container. `flow doctor`, the
runner, and the API all call `require_docker()` and refuse to run on the host if Docker or
the image is missing.

**Why.** Safety is priority #2. Agent-written code is untrusted; running it on the host is
unacceptable. There is no host-execution escape hatch in the product path — the in-process
kernel exists solely for tests and is never selected by `runner.py`.

**Consequences.** The container is launched with a read-only dataset mount, one writable
workdir, `--network none` (opt-in to enable), resource limits, non-root user,
`--cap-drop=ALL`, `no-new-privileges`, and a read-only rootfs with tmpfs. Tests assert the
exact `docker run` flags.

---

## ADR-004 — Provider abstraction; keys host-side only

**Decision.** A tiny `Provider.generate(messages, tools) -> ToolCall` interface, with a
deterministic offline `MockProvider`, a local `OllamaProvider`, and one
`OpenAICompatibleProvider` covering Gemini/Groq/OpenRouter/DeepSeek/OpenAI via
(`base_url`, `api_key_env`). Keys are read from the host environment at call time.

**Why.** Free-tier flexibility and local-first operation without vendor lock-in or
provider-specific hacks. The OpenAI-compatible shape covers most hosted free tiers with one
implementation. The mock provider drives the full loop in CI without Docker or network and
**never** performs real analysis.

**Consequences.** Provider calls happen host-side, so keys never enter the container
(privacy boundary). Unknown providers and missing keys fail fast with clear errors.

---

## ADR-005 — Config is analysis-free, by construction

**Decision.** `config.yaml` declares only dataset locations, the research question, runtime,
and safety. The loader rejects any analysis keys (gates, populations, thresholds,
transforms, markers, pipelines, …).

**Why.** Principle #5: the config must never become a place to hardcode the science.
Experiment *facts* go in `metadata.json`; the agent derives all analysis at runtime.

**Consequences.** `config.py` is the one library file allowed to *name* forbidden terms —
only to refuse them. The no-hardcoded-analysis test allowlists it for that reason.

---

## ADR-006 — Privacy boundary for hosted models

**Decision.** The dataset description sent to a hosted model is a neutral, **structural**
profile (file names, shapes, columns) plus `metadata.json` facts — not raw event data.
Sending full raw rows is gated behind `safety.allow_raw_data_to_model` (default off).

**Why.** Lab data may be sensitive. The agent can still reason well from schemas/summaries
and read the raw data *inside the sandbox*; only what crosses to the hosted model is limited.

**Consequences.** `validation.ProjectProfile.dataset_description()` is purely structural (a
test asserts it contains no analysis hints). Raw event values stay in the container.

---

## ADR-007 — API/UI architecture

**Decision.** FastAPI backend + React/Vite/TypeScript frontend. SQLite (stdlib) stores
project/run *metadata* (so RunHistory survives refresh/restart); uploads and artifacts live
on the filesystem (paths from env). Runs execute in a background thread that delegates to the
Docker runner; the API never runs agent code on the host. The UI separates the **active**
run (just launched this session) from a **selected** historical run, so a persisted
selection can never hijack the run launcher.

**Why.** Lab members need to upload, launch, watch, and review without a terminal. Durable
history requires a DB; live trajectory viewing requires incremental artifact writes (the
runner appends to `actions.jsonl` and rewrites `notebook.ipynb` after every step, which the
UI tails).

**Consequences.** Status-aware UI: "running…" until complete, a clear failure banner if the
agent didn't submit or Docker is missing, and never a vague dead-end. Uploads are validated,
size-limited, and traversal-checked.
