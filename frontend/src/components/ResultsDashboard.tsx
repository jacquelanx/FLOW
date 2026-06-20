import { useEffect, useState } from "react";
import { api } from "../api";
import type { Artifact, RunStatus } from "../types";

// Status-aware results view: shows "running…" until complete, the submitted answer when
// done, rendered plots and saved tables, and a clear error if the run failed.
export default function ResultsDashboard({
  runId,
  onBrowse,
}: {
  runId: string;
  onBrowse: () => void;
}) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  useEffect(() => {
    let active = true;
    async function load() {
      const r = await api.getRun(runId);
      if (!active) return;
      setRun(r);
      setArtifacts(await api.listArtifacts(runId));
      if (!["completed", "failed", "error"].includes(r.status)) {
        setTimeout(load, 2000);
      }
    }
    load();
    return () => {
      active = false;
    };
  }, [runId]);

  if (!run) return <p>Loading…</p>;

  const status = run.status;
  const images = artifacts.filter((a) => a.kind === "image");
  const tables = artifacts.filter((a) => a.kind === "table");

  return (
    <div className="page">
      <header className="page-head">
        <h2>Results</h2>
        <p className="subtitle">
          The agent’s submitted conclusion, plus any plots and tables it produced.
        </p>
      </header>

      {!["completed", "failed", "error"].includes(status) && (
        <div className="banner warn">
          <span className="spinner" /> Run is {status}… results will appear when it finishes.
        </div>
      )}

      {(status === "failed" || status === "error") && (
        <div className="banner err">
          <strong>Run did not produce an answer.</strong>
          <div style={{ marginTop: 6 }}>
            {String((run.run_meta as any)?.failure_reason || (run.live as any)?.failure_reason ||
              "The agent did not submit an answer, or execution could not start (e.g. Docker missing).")}
          </div>
        </div>
      )}

      {status === "completed" && run.answer && (
        <div className="card">
          <h3>Submitted conclusion</h3>
          <div className="code">{run.answer}</div>
        </div>
      )}

      {images.length > 0 && (
        <div className="card">
          <h3>Plots</h3>
          {images.map((a) => (
            <div key={a.path}>
              <div className="muted">{a.path}</div>
              <img className="plot" src={api.artifactUrl(runId, a.path)} alt={a.name} />
            </div>
          ))}
        </div>
      )}

      {tables.length > 0 && (
        <div className="card">
          <h3>Saved tables</h3>
          <ul>
            {tables.map((a) => (
              <li key={a.path}>
                <a href={api.artifactUrl(runId, a.path)} target="_blank" rel="noreferrer">
                  {a.path}
                </a>{" "}
                <span className="muted">({a.size} bytes)</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card">
        <div style={{ display: "flex", gap: 10 }}>
          <button className="secondary" onClick={onBrowse}>
            Browse all artifacts →
          </button>
          <a href={api.downloadUrl(runId)}>
            <button>Download run (.zip)</button>
          </a>
        </div>
      </div>
    </div>
  );
}
