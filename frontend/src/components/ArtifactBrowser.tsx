import { useEffect, useState } from "react";
import { api } from "../api";
import type { Artifact, TrajectorySummary } from "../types";
import TrajectoryPicker from "./TrajectoryPicker";

// Lists every artifact a chosen trajectory produced, with a type-aware preview
// (notebook/json/csv/image/text), plus one-click download of the whole consensus run.
export default function ArtifactBrowser({ runId }: { runId: string }) {
  const [trajectories, setTrajectories] = useState<TrajectorySummary[]>([]);
  const [idx, setIdx] = useState(0);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [preview, setPreview] = useState<string>("");

  useEffect(() => {
    api.getTrajectories(runId).then(setTrajectories).catch(() => setTrajectories([]));
  }, [runId]);

  useEffect(() => {
    setSelected(null);
    setPreview("");
    api.listArtifacts(runId, idx).then(setArtifacts).catch(() => setArtifacts([]));
  }, [runId, idx]);

  async function open(a: Artifact) {
    setSelected(a);
    if (a.kind === "image") {
      setPreview("");
      return;
    }
    const res = await fetch(api.artifactUrl(runId, idx, a.path));
    let text = await res.text();
    if (a.kind === "json" || a.kind === "notebook") {
      try {
        text = JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        /* leave as-is */
      }
    }
    setPreview(text.slice(0, 20000));
  }

  const images = artifacts.filter((a) => a.kind === "image");

  return (
    <div className="page">
      <header className="page-head">
        <h2>Artifacts</h2>
        <p className="subtitle">
          Every file a chosen trajectory produced. Select one to preview, or download the whole
          consensus run.
        </p>
      </header>

      <TrajectoryPicker trajectories={trajectories} idx={idx} onChange={setIdx} />

      <div className="card">
        <div className="actions" style={{ margin: 0 }}>
          <a href={api.downloadUrl(runId)}>
            <button>Download run (.zip)</button>
          </a>
          <span className="hint" style={{ margin: 0 }}>
            One archive with every trajectory’s notebook, tables, and plots.
          </span>
        </div>
      </div>

      {/* Contact sheet of the plots: the fastest way to find the figure you want. */}
      {images.length > 0 && (
        <section className="card">
          <div className="card-title">
            Plots — {images.length} figure{images.length === 1 ? "" : "s"}
          </div>
          <div className="thumbgrid">
            {images.map((a) => (
              <button
                key={a.path}
                className={`thumb ${selected?.path === a.path ? "sel" : ""}`}
                title={a.path}
                onClick={() => open(a)}
              >
                <img
                  src={api.artifactUrl(runId, idx, a.path)}
                  alt={a.name}
                  loading="lazy"
                />
                <span className="thumb-name">{a.name}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      <div className="artifact-split">
        <div className="card">
          <table className="artifact-list">
            <thead>
              <tr>
                <th>Artifact</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {artifacts.map((a) => {
                const sel = selected?.path === a.path;
                return (
                  <tr
                    key={a.path}
                    className={sel ? "sel" : ""}
                    aria-selected={sel}
                    onClick={() => open(a)}
                  >
                    <td>{a.path}</td>
                    <td>{a.kind}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {artifacts.length === 0 && (
            <div className="empty">No artifacts for this trajectory.</div>
          )}
        </div>

        <div className="card">
          {!selected && <div className="empty">Select an artifact to preview.</div>}
          {selected && selected.kind === "image" && (
            <img
              className="plot"
              style={{ width: "100%" }}
              src={api.artifactUrl(runId, idx, selected.path)}
              alt={selected.name}
            />
          )}
          {selected && selected.kind === "table" && <CsvTable text={preview} />}
          {selected && selected.kind !== "image" && selected.kind !== "table" && (
            <pre className="code" style={{ maxHeight: 560, overflow: "auto" }}>{preview}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

function CsvTable({ text }: { text: string }) {
  const rows = text.trim().split("\n").slice(0, 200).map((r) => r.split(","));
  if (rows.length === 0) return null;
  const [head, ...body] = rows;
  return (
    <div style={{ overflow: "auto", maxHeight: 500 }}>
      <table>
        <thead>
          <tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
