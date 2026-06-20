import { useEffect, useState } from "react";
import { api } from "../api";
import type { DockerHealth, ProvidersInfo } from "../types";
import ModelPicker from "./ModelPicker";

// Page 3: confirm the question, pick provider/model (from the catalog) + max steps, and
// start a run. Starting always begins a FRESH run for this session — a persisted selection
// can never hijack the launcher (that separation lives in App's state).
export default function RunLauncher({
  projectId,
  health,
  onRunStarted,
}: {
  projectId: string;
  health: DockerHealth | null;
  onRunStarted: (rid: string) => void;
}) {
  const [question, setQuestion] = useState("");
  const [info, setInfo] = useState<ProvidersInfo | null>(null);
  const [provider, setProvider] = useState("mock");
  const [model, setModel] = useState("mock");
  const [maxSteps, setMaxSteps] = useState(30);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.providersInfo(), api.getConfig(projectId)]).then(([pi, cfg]) => {
      setInfo(pi);
      const c = cfg.config;
      setQuestion(c.question || "");
      const p = c.runtime?.provider || "mock";
      setProvider(p);
      setModel(c.runtime?.model || pi.default_models[p] || "mock");
      setMaxSteps(c.runtime?.max_steps ?? 30);
    });
  }, [projectId]);

  const models = info?.catalog[provider] ?? [model];

  function changeProvider(p: string) {
    setProvider(p);
    setModel(info?.default_models[p] ?? "");
  }

  async function start() {
    setErr(null);
    setStarting(true);
    try {
      const r = await api.startRun({
        project_id: projectId,
        question,
        provider,
        model,
        max_steps: maxSteps,
      });
      onRunStarted(r.run_id);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  const dockerReady = health?.ready ?? false;

  return (
    <div className="page">
      <header className="page-head">
        <h2>Launch a run</h2>
        <p className="subtitle">
          Confirm the question and model, then start. The agent writes and runs code inside a
          sandboxed Docker container.
        </p>
      </header>

      {!dockerReady && (
        <div className="banner err">
          Docker isn’t ready, so the agent can’t execute code safely. {health?.notes.join(" ")}
          <br />
          Build the image: <code>docker build -t flow-bixbench-env:1.0 -f docker/Dockerfile .</code>
        </div>
      )}

      <div className="split">
        <section className="card">
          <div className="field">
            <label htmlFor="rq">Research question</label>
            <textarea
              id="rq"
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>

          <div className="grid-3">
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
          </div>

          <p className="hint" style={{ marginTop: 16 }}>
            <code>mock</code> runs the loop offline with no real analysis. Hosted models
            (gemini / groq / …) need their API key set on the server; <code>ollama</code> runs
            locally.
          </p>

          <div className="actions" style={{ marginTop: 20 }}>
            <button className="primary" onClick={start} disabled={starting || !question.trim()}>
              {starting ? <span className="spinner" /> : "Start run"}
            </button>
          </div>
          {err && <div className="banner err">{err}</div>}
        </section>

        <aside className="aside">
          <section className="card">
            <div className="card-title">Before you start</div>
            <ul className="reflist">
              <li>
                <div className="name">Sandbox</div>
                <div className="desc">
                  {dockerReady ? "Docker is ready." : "Docker is not ready — see the notice."}
                </div>
              </li>
              <li>
                <div className="name">Question</div>
                <div className="desc">{question.trim() ? "Set." : "Add a question to enable Start."}</div>
              </li>
              <li>
                <div className="name">Model</div>
                <div className="desc">
                  {provider} / {model || "—"}
                </div>
              </li>
            </ul>
          </section>

          <section className="card">
            <div className="card-title">What happens</div>
            <ol className="steps-list">
              <li>A sandboxed container starts with your data mounted read-only.</li>
              <li>The agent writes and runs code, step by step.</li>
              <li>You watch each step live on the next page.</li>
              <li>It submits a conclusion; artifacts are saved.</li>
            </ol>
          </section>
        </aside>
      </div>
    </div>
  );
}
