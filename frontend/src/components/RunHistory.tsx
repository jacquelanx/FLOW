import { useEffect, useState } from "react";
import { api } from "../api";
import type { Batch } from "../types";

// Persistent, clickable list of past consensus runs (read from the backend DB, so it
// survives refresh/restart). Selecting one opens its results — without hijacking the launcher.
export default function RunHistory({
  projectId,
  onSelectRun,
  onRunDeleted,
}: {
  projectId: string;
  onSelectRun: (rid: string) => void;
  onRunDeleted?: (rid: string) => void;
}) {
  const [runs, setRuns] = useState<Batch[]>([]);

  useEffect(() => {
    api.listRuns(projectId).then(setRuns).catch(() => setRuns([]));
  }, [projectId]);

  async function remove(r: Batch) {
    if (!confirm("Delete this run and all its artifacts? This cannot be undone.")) return;
    try {
      await api.deleteRun(r.id);
      setRuns((rs) => rs.filter((x) => x.id !== r.id));
      onRunDeleted?.(r.id);
    } catch (e) {
      alert(`Could not delete run: ${(e as Error).message}`);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h2>Run history</h2>
        <p className="subtitle">
          Every consensus run for this project is saved here and survives a refresh. Open one to
          revisit its consensus, trajectories, and artifacts.
        </p>
      </header>
      <div className="card">
        {runs.length === 0 && (
          <div className="empty">
            No runs yet. Launch one from “Launch run” and it will appear here.
          </div>
        )}
        {runs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Question</th>
                <th>Model</th>
                <th>Trajectories</th>
                <th>Status</th>
                <th></th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="muted">{new Date(r.created_at * 1000).toLocaleString()}</td>
                  <td title={r.question}>{truncate(r.question, 48)}</td>
                  <td className="muted">{r.model}</td>
                  <td>
                    {r.n_trajectories}
                    {r.status === "completed" && (
                      <span className="muted"> ({r.n_submitted} answered)</span>
                    )}
                  </td>
                  <td>
                    <StatusBadge status={r.status} consensusOk={r.consensus_ok} />
                  </td>
                  <td>
                    <button className="secondary" onClick={() => onSelectRun(r.id)}>
                      Open
                    </button>
                  </td>
                  <td>
                    <button
                      className="ghost"
                      disabled={["running", "pending", "synthesizing"].includes(r.status)}
                      title={
                        ["running", "pending", "synthesizing"].includes(r.status)
                          ? "Can't delete a run in progress"
                          : "Delete this run"
                      }
                      onClick={() => remove(r)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function StatusBadge({ status, consensusOk }: { status: string; consensusOk: number }) {
  if (status === "completed" && consensusOk) return <span className="badge ok">consensus</span>;
  if (["running", "pending", "synthesizing"].includes(status))
    return <span className="badge run">{status}</span>;
  return <span className="badge err">{status}</span>;
}
