import { useEffect, useState } from "react";
import { api } from "../api";
import type { ProvidersInfo } from "../types";
import ModelPicker from "./ModelPicker";

// Page 2: a friendly, form-based config editor — no raw YAML. Lab members set the research
// question and a few run settings with labeled controls; FLOW builds the analysis-free
// config.yaml on the server. A collapsible read-only YAML preview is available for the
// curious, but no one needs to touch it.
export default function ConfigEditor({
  projectId,
  onNext,
}: {
  projectId: string;
  onNext: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  const [info, setInfo] = useState<ProvidersInfo | null>(null);

  // Form fields.
  const [question, setQuestion] = useState("");
  const [description, setDescription] = useState("");
  const [provider, setProvider] = useState("mock");
  const [model, setModel] = useState("mock");
  const [diffMeta, setDiffMeta] = useState(false);
  const [metaProvider, setMetaProvider] = useState("mock");
  const [metaModel, setMetaModel] = useState("mock");
  const [maxSteps, setMaxSteps] = useState(30);
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [allowRaw, setAllowRaw] = useState(false);
  const [perCell, setPerCell] = useState(120);
  const [perTraj, setPerTraj] = useState(1800);
  const [memory, setMemory] = useState("4g");
  const [cpus, setCpus] = useState("2");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showYaml, setShowYaml] = useState(false);
  const [yamlPreview, setYamlPreview] = useState("");
  const [status, setStatus] = useState<{ ok: boolean; error?: string } | null>(null);

  useEffect(() => {
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
      setPerCell(c.runtime?.per_cell_timeout ?? 120);
      setPerTraj(c.runtime?.per_trajectory_timeout ?? 1800);
      setAllowNetwork(Boolean(c.safety?.allow_network));
      setAllowRaw(Boolean(c.safety?.allow_raw_data_to_model));
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
      max_steps: maxSteps,
      per_cell_timeout: perCell,
      per_trajectory_timeout: perTraj,
      memory,
      cpus,
      allow_network: allowNetwork,
      allow_raw_data_to_model: allowRaw,
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
          steps here — the agent works those out from your data and the question.
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
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={allowRaw}
                  onChange={(e) => setAllowRaw(e.target.checked)}
                />
                <span>
                  Allow raw data to the model
                  <span className="hint"> — by default only schemas/summaries are shared.</span>
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
