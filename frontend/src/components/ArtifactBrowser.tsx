import { useEffect, useState } from "react";
import { api } from "../api";
import type { Artifact } from "../types";

// Lists every artifact with a type-aware preview (notebook/json/csv/image/text) and
// one-click download of the whole run as a zip.
export default function ArtifactBrowser({ runId }: { runId: string }) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [preview, setPreview] = useState<string>("");

  useEffect(() => {
    api.listArtifacts(runId).then(setArtifacts).catch(() => setArtifacts([]));
  }, [runId]);

  async function open(a: Artifact) {
    setSelected(a);
    if (a.kind === "image") {
      setPreview("");
      return;
    }
    const res = await fetch(api.artifactUrl(runId, a.path));
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

  return (
    <div className="page">
      <header className="page-head">
        <h2>Artifacts</h2>
        <p className="subtitle">
          Every file this run produced. Select one to preview, or download the whole run.
        </p>
      </header>
      <div className="card">
        <a href={api.downloadUrl(runId)}>
          <button>Download run (.zip)</button>
        </a>
      </div>

      <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
        <div className="card" style={{ flex: "0 0 260px" }}>
          <table>
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
                    onClick={() => open(a)}
                    style={{
                      cursor: "pointer",
                      background: sel ? "var(--ink)" : undefined,
                      color: sel ? "#fff" : undefined,
                    }}
                  >
                    <td>{a.path}</td>
                    <td>{a.kind}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="card" style={{ flex: 1, minWidth: 0 }}>
          {!selected && <div className="empty">Select an artifact to preview.</div>}
          {selected && selected.kind === "image" && (
            <img
              className="plot"
              style={{ width: "100%" }}
              src={api.artifactUrl(runId, selected.path)}
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
