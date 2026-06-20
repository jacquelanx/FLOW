import { useEffect, useState } from "react";
import { api } from "../api";
import type { DockerHealth } from "../types";

// Page 3: confirm the question, pick provider/model + max steps, and start a run.
// Starting always begins a FRESH run for this session — no persisted selection can
// hijack the launcher (that concern lives entirely in App's state separation).
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
  const [providers, setProviders] = useState<string[]>(["mock"]);
  const [provider, setProvider] = useState("mock");
  const [model, setModel] = useState("mock");
  const [maxSteps, setMaxSteps] = useState(30);
  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.providers().then(setProviders).catch(() => {});
    api.getConfig(projectId).then((c) => {
      const m = c.content.match(/^question:\s*"?(.*?)"?\s*$/m);
      if (m) setQuestion(m[1]);
    });
  }, [projectId]);

  // Sensible default models per provider (editable).
  useEffect(() => {
    const defaults: Record<string, string> = {
      mock: "mock",
      ollama: "qwen2.5-coder",
      gemini: "gemini-2.0-flash",
      google: "gemini-2.0-flash",
      groq: "llama-3.3-70b-versatile",
      openrouter: "meta-llama/llama-3.3-70b-instruct",
      deepseek: "deepseek-chat",
      openai: "gpt-4o-mini",
    };
    setModel(defaults[provider] ?? "");
  }, [provider]);

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
  const needsDocker = provider !== undefined; // all runs execute in Docker

  return (
    <div>
      <h2>Launch Run</h2>

      {needsDocker && !dockerReady && (
        <div className="banner err">
          Docker is not ready, so the agent cannot execute code safely. {health?.notes.join(" ")}
          <br />
          Build the image: <code>docker build -t flow-bixbench-env:1.0 -f docker/Dockerfile .</code>
        </div>
      )}

      <div className="card">
        <label htmlFor="rq">Research question</label>
        <textarea id="rq" rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} />

        <div className="row" style={{ marginTop: 12 }}>
          <div>
            <label htmlFor="prov">Provider</label>
            <select id="prov" value={provider} onChange={(e) => setProvider(e.target.value)}>
              {providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="model">Model</label>
            <input id="model" value={model} onChange={(e) => setModel(e.target.value)} />
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

        <p className="hint" style={{ marginTop: 10 }}>
          Code executes inside the Docker BixBench-env container. The <code>mock</code> provider
          runs the loop offline with no real analysis. For real analysis use a hosted model
          (gemini / groq / openrouter / deepseek) with the key set on the server, or local
          <code> ollama</code>.
        </p>

        <div style={{ marginTop: 12 }}>
          <button onClick={start} disabled={starting || !question.trim()}>
            {starting ? <span className="spinner" /> : "Start run"}
          </button>
        </div>
        {err && (
          <div className="banner err" style={{ marginTop: 12 }}>
            {err}
          </div>
        )}
      </div>
    </div>
  );
}
