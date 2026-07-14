import { useEffect, useState } from "react";
import { api } from "../api";
import type { TrajectorySummary } from "../types";
import TrajectoryPicker from "./TrajectoryPicker";

// Renders the notebook.ipynb of a chosen trajectory within the consensus run
export default function NotebookViewer({ runId }: { runId: string }) {
  const [trajectories, setTrajectories] = useState<TrajectorySummary[]>([]);
  const [idx, setIdx] = useState(0);
  const [nb, setNb] = useState<any | null>(null);

  useEffect(() => {
    api.getTrajectories(runId).then(setTrajectories).catch(() => setTrajectories([]));
  }, [runId]);

  useEffect(() => {
    setNb(null);
    api.getNotebook(runId, idx).then(setNb).catch(() => setNb(null));
  }, [runId, idx]);

  const cells: any[] = nb?.cells || [];

  return (
    <div className="page">
      <header className="page-head">
        <h2>Notebook</h2>
        <p className="subtitle">
          The notebook a chosen trajectory built — code cells with their text and plot outputs.
        </p>
      </header>

      <TrajectoryPicker trajectories={trajectories} idx={idx} onChange={setIdx} />

      {cells.length > 0 && (
        <div className="banner ok">
          The <strong>first cell is the automated first-run pipeline</strong> (preprocessing +
          gating). Everything after it is the <strong>agent’s interpretive work</strong>.
        </div>
      )}

      {!nb && <p className="muted">Loading notebook…</p>}
      {nb && cells.length === 0 && <div className="empty">No cells in this trajectory yet.</div>}
      {cells.map((c, i) => (
        <div className="notebook-cell" key={i}>
          <div className="src">{Array.isArray(c.source) ? c.source.join("") : c.source}</div>
          {(c.outputs || []).map((o: any, j: number) => (
            <Output key={j} output={o} runId={runId} idx={idx} />
          ))}
        </div>
      ))}
    </div>
  );
}

function Output({ output, runId, idx }: { output: any; runId: string; idx: number }) {
  if (output.output_type === "stream") {
    return <div className="out">{textOf(output.text)}</div>;
  }
  if (output.output_type === "execute_result" || output.output_type === "display_data") {
    const imgPath = output.metadata?.flow_image_path as string | undefined;
    if (imgPath) {
      const name = imgPath.split("/").slice(-2).join("/");
      return <img className="plot" src={api.artifactUrl(runId, idx, name)} alt="plot" />;
    }
    return <div className="out">{textOf(output.data?.["text/plain"])}</div>;
  }
  if (output.output_type === "error") {
    return (
      <div className="out" style={{ color: "var(--err)" }}>
        {(output.traceback || []).join("\n")}
      </div>
    );
  }
  return null;
}

function textOf(t: unknown): string {
  if (Array.isArray(t)) return t.join("");
  return String(t ?? "");
}
