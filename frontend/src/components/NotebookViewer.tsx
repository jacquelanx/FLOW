import { useEffect, useState } from "react";
import { api } from "../api";

// Renders the produced notebook.ipynb: code cells + their text/plot outputs. Plot images
// are loaded from the artifact endpoint by their saved path.
export default function NotebookViewer({ runId }: { runId: string }) {
  const [nb, setNb] = useState<any | null>(null);

  useEffect(() => {
    api.getNotebook(runId).then(setNb).catch(() => setNb(null));
  }, [runId]);

  if (!nb) return <p>Loading notebook…</p>;
  const cells: any[] = nb.cells || [];

  return (
    <div className="page">
      <header className="page-head">
        <h2>Notebook</h2>
        <p className="subtitle">
          The notebook the agent built — code cells with their text and plot outputs.
        </p>
      </header>
      {cells.length === 0 && <div className="empty">No cells yet.</div>}
      {cells.map((c, i) => (
        <div className="notebook-cell" key={i}>
          <div className="src">
            {Array.isArray(c.source) ? c.source.join("") : c.source}
          </div>
          {(c.outputs || []).map((o: any, j: number) => (
            <Output key={j} output={o} runId={runId} />
          ))}
        </div>
      ))}
    </div>
  );
}

function Output({ output, runId }: { output: any; runId: string }) {
  if (output.output_type === "stream") {
    return <div className="out">{textOf(output.text)}</div>;
  }
  if (output.output_type === "execute_result" || output.output_type === "display_data") {
    const imgPath = output.metadata?.flow_image_path as string | undefined;
    if (imgPath) {
      // The container writes plots into /work; the artifact server exposes them by name.
      const name = imgPath.split("/").slice(-2).join("/");
      return <img className="plot" src={api.artifactUrl(runId, name)} alt="plot" />;
    }
    return <div className="out">{textOf(output.data?.["text/plain"])}</div>;
  }
  if (output.output_type === "error") {
    return <div className="out" style={{ color: "var(--err)" }}>{(output.traceback || []).join("\n")}</div>;
  }
  return null;
}

function textOf(t: unknown): string {
  if (Array.isArray(t)) return t.join("");
  return String(t ?? "");
}
