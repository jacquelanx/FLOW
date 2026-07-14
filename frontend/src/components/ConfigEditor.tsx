import { useEffect, useState } from "react";
import { api } from "../api";
import type { ProvidersInfo } from "../types";
import ModelPicker from "./ModelPicker";

// Lab members set research question and settings; FLOW builds the analysis-free
// config.yaml on the server. A read-only YAML preview is available but unneeded.
export default function ConfigEditor({
  projectId,
  onNext,
}: {
  projectId: string;
  onNext: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const [info, setInfo] = useState<ProvidersInfo | null>(null);

  // Form fields
  const [question, setQuestion] = useState("");
  const [description, setDescription] = useState("");
  const [provider, setProvider] = useState("mock");
  const [model, setModel] = useState("mock");
  const [diffMeta, setDiffMeta] = useState(false);
  const [metaProvider, setMetaProvider] = useState("mock");
  const [metaModel, setMetaModel] = useState("mock");
  const [maxSteps, setMaxSteps] = useState(30);
  const [firstRun, setFirstRun] = useState("auto");
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [perCell, setPerCell] = useState(120);
  const [perTraj, setPerTraj] = useState(1800);
  const [memory, setMemory] = useState("4g");
  const [cpus, setCpus] = useState("2");

  const [customPrompt, setCustomPrompt] = useState("");
  const [promptSaved, setPromptSaved] = useState<boolean | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showYaml, setShowYaml] = useState(false);
  const [yamlPreview, setYamlPreview] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; error?: string } | null>(null);

  useEffect(() => {
    api.getPrompt(projectId).then((p) => setCustomPrompt(p.content)).catch(() => {});
    Promise.all([api.providersInfo(), api.getConfig(projectId)]).then(([pi, cfg]) => {
      setInfo(pi);
      const c = cfg.config;
      setQuestion(c.question || "");
      setDescription(c.dataset?.description || "");
      const p = c.runtime?.provider || "mock";
      setProvider(p);
      setModel(c.runtime?.model || pi.default_models[p] || "mock");
      const mp = c.runtime?.meta_provider || "";
      const mm = c.runtime?.meta_model || "";
      if (mp || mm) {
        setDiffMeta(true);
        setMetaProvider(mp || p);
        setMetaModel(mm || pi.default_models[mp] || "");
      } else {
        setMetaProvider(p);
        setMetaModel(c.runtime?.model || pi.default_models[p] || "mock");
      }
      setMaxSteps(c.runtime?.max_steps ?? 30);
      setFirstRun(c.runtime?.first_run || "auto");
      setPerCell(c.runtime?.per_cell_timeout ?? 120);
      setPerTraj(c.runtime?.per_trajectory_timeout ?? 1800);
      setAllowNetwork(Boolean(c.safety?.allow_network));
      setMemory(c.safety?.memory || "4g");
      setCpus(c.safety?.cpus || "2");
      setYamlPreview(cfg.content);
      setLoaded(true);
    });
  }, [projectId]);

  const models = info?.catalog[provider] ?? [model];

  function changeProvider(p: string) {
    setProvider(p);
    setModel(info?.default_models[p] ?? "");
  }

  async function save(thenNext = false) {
    const r = await api.saveConfigForm(projectId, {
      question,
      description,
      provider,
      model,
      meta_provider: diffMeta ? metaProvider : "",
      meta_model: diffMeta ? metaModel : "",
      first_run: firstRun,
      max_steps: maxSteps,
      per_cell_timeout: perCell,
      per_trajectory_timeout: perTraj,
      memory,
      cpus,
      allow_network: allowNetwork,
    });
    setStatus(r);
    if (r.ok && r.content) setYamlPreview(r.content);
    if (r.ok && thenNext) onNext();
  }

  if (!loaded) return <p className="muted">Loading…</p>;

  return (
    <div className="page">
      <header className="page-head">
        <h2>Configure the run</h2>
        <p className="subtitle">
          Set what you want to ask and how FLOW should run. You don’t specify any analysis
          steps here; the agent works those out from your data and the question.
        </p>
      </header>

      <section className="card">
        <label htmlFor="q">Research question</label>
        <p className="hint">The single most important field — describe what you want to find out.</p>
        <textarea
          id="q"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How does the CAR-detector signal change across the timepoints?"
        />

        <label htmlFor="desc" style={{ marginTop: 18 }}>
          Dataset note <span className="optional">optional</span>
        </label>
        <input
          id="desc"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="A short note about this dataset (not analysis instructions)."
        />
      </section>

      <section className="card">
        <h3 className="card-title">Model</h3>
        <div className="grid-2">
          <div>
            <label htmlFor="prov">Provider</label>
            <select id="prov" value={provider} onChange={(e) => changeProvider(e.target.value)}>
              {(info?.ui_providers ?? ["mock"]).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="model">Model</label>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </div>
        </div>
        <p className="hint">
          Pick a model from the list, or choose “Custom…” to enter any id the provider serves.
          <code>mock</code> runs the loop offline with no real analysis.
        </p>

        <label className="toggle" style={{ fontWeight: 600, marginTop: 18 }}>
          <input
            type="checkbox"
            checked={diffMeta}
            onChange={(e) => setDiffMeta(e.target.checked)}
          />
          <span>
            Use a different model for the consensus
            <span className="hint"> — synthesize the final answer with a separate model.</span>
          </span>
        </label>
        {diffMeta && (
          <div className="grid-2" style={{ marginTop: 12 }}>
            <div>
              <label htmlFor="mprov">Consensus provider</label>
              <select
                id="mprov"
                value={metaProvider}
                onChange={(e) => {
                  setMetaProvider(e.target.value);
                  setMetaModel(info?.default_models[e.target.value] ?? "");
                }}
              >
                {(info?.ui_providers ?? ["mock"]).map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="mmodel">Consensus model</label>
              <ModelPicker
                id="mmodel"
                models={info?.catalog[metaProvider] ?? [metaModel]}
                value={metaModel}
                onChange={setMetaModel}
              />
            </div>
          </div>
        )}
      </section>

      <section className="card">
        <h3 className="card-title">First-run</h3>
        <select
          id="cfg-firstrun"
          value={firstRun}
          onChange={(e) => setFirstRun(e.target.value)}
        >
          <option value="auto">Auto — run this project’s first-run script before the agent</option>
          <option value="none">None — agent starts from a blank notebook</option>
        </select>
        <p className="hint">
          The first-run script (set on the <strong>Analysis</strong> page) does the flow analysis
          <em> before</em> the agent, and the agent interprets its output. “Auto” runs it if the
          project has one; “None” skips it.
        </p>
      </section>

      <section className="card">
        <h3 className="card-title">Custom LLM prompt (optional)</h3>
        <p className="hint" style={{ marginTop: 0 }}>
          Free-form extra instructions for the agent, <strong>added on top of</strong> FLOW’s
          default prompt. For the structured biology (panel, gating, populations), use the
          <strong> Analysis</strong> page — this box is only for anything that doesn’t fit there.
        </p>
        <textarea
          rows={8}
          value={customPrompt}
          onChange={(e) => {
            setCustomPrompt(e.target.value);
            setPromptSaved(null);
          }}
          placeholder="e.g. Gating hierarchy: 1) Debris removal FSC-A > 10000 …"
        />
        <div className="actions" style={{ marginTop: 12 }}>
          <label className="secondary" style={{ display: "inline-block", cursor: "pointer" }}>
            Upload .md/.txt
            <input
              type="file"
              accept=".md,.txt,text/plain,text/markdown"
              style={{ display: "none" }}
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setCustomPrompt(await f.text());
                  setPromptSaved(null);
                }
              }}
            />
          </label>
          <button
            className="secondary"
            onClick={async () => {
              await api.savePrompt(projectId, customPrompt);
              setPromptSaved(true);
            }}
          >
            Save prompt
          </button>
          {promptSaved && <span className="muted">Saved.</span>}
        </div>
      </section>

      <section className="card">
        <button
          className="disclosure"
          aria-expanded={showAdvanced}
          onClick={() => setShowAdvanced((s) => !s)}
        >
          {showAdvanced ? "▾" : "▸"} Advanced settings
        </button>
        {showAdvanced && (
          <div className="advanced">
            <div className="grid-2">
              <div>
                <label htmlFor="steps">Max steps</label>
                <input
                  id="steps"
                  type="number"
                  min={1}
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value || "30", 10))}
                />
              </div>
              <div>
                <label htmlFor="pcell">Per-cell timeout (s)</label>
                <input
                  id="pcell"
                  type="number"
                  min={1}
                  value={perCell}
                  onChange={(e) => setPerCell(parseInt(e.target.value || "120", 10))}
                />
              </div>
              <div>
                <label htmlFor="ptraj">Per-run timeout (s)</label>
                <input
                  id="ptraj"
                  type="number"
                  min={1}
                  value={perTraj}
                  onChange={(e) => setPerTraj(parseInt(e.target.value || "1800", 10))}
                />
              </div>
              <div>
                <label htmlFor="mem">Memory limit</label>
                <input id="mem" value={memory} onChange={(e) => setMemory(e.target.value)} />
              </div>
              <div>
                <label htmlFor="cpus">CPU limit</label>
                <input id="cpus" value={cpus} onChange={(e) => setCpus(e.target.value)} />
              </div>
            </div>

            <div className="toggles">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={allowNetwork}
                  onChange={(e) => setAllowNetwork(e.target.checked)}
                />
                <span>
                  Allow network in the sandbox
                  <span className="hint"> — lets the agent install packages (off by default).</span>
                </span>
              </label>
            </div>
          </div>
        )}
      </section>

      <div className="actions">
        <button className="secondary" onClick={() => save(false)}>
          Save
        </button>
        <button className="primary" onClick={() => save(true)} disabled={!question.trim()}>
          Save &amp; continue →
        </button>
        <button
          className="ghost"
          onClick={() => setShowYaml((s) => !s)}
          style={{ marginLeft: "auto" }}
        >
          {showYaml ? "Hide" : "View"} generated config
        </button>
      </div>

      {status && (
        <div className={`banner ${status.ok ? "ok" : "err"}`}>
          {status.ok ? "Saved." : `Could not save: ${status.error}`}
        </div>
      )}

      {showYaml && (
        <section className="card">
          <h3 className="card-title">config.yaml (generated, read-only)</h3>
          <pre className="code">{yamlPreview}</pre>
        </section>
      )}
    </div>
  );
}
