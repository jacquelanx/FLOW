import { useEffect, useState } from "react";
import { api } from "../api";
import type { Run } from "../types";

// Persistent, clickable list of past runs (read from the backend DB, so it survives
// refresh/restart). Selecting one opens its results — without hijacking the launcher.
export default function RunHistory({
  projectId,
  onSelectRun,
}: {
  projectId: string;
  onSelectRun: (rid: string) => void;
}) {
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    api.listRuns(projectId).then(setRuns).catch(() => setRuns([]));
  }, [projectId]);

  return (
    <div>
      <h2>Run History</h2>
      <div className="card">
        {runs.length === 0 && <p className="muted">No runs yet for this project.</p>}
        {runs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Question</th>
                <th>Provider / Model</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="muted">{new Date(r.created_at * 1000).toLocaleString()}</td>
                  <td title={r.question}>{truncate(r.question, 60)}</td>
                  <td className="muted">
                    {r.provider} / {r.model}
                  </td>
                  <td>
                    <StatusBadge status={r.status} submitted={r.submitted} />
                  </td>
                  <td>
                    <button className="secondary" onClick={() => onSelectRun(r.id)}>
                      Open
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

function StatusBadge({ status, submitted }: { status: string; submitted: number }) {
  if (status === "completed" && submitted) return <span className="badge ok">completed</span>;
  if (status === "running" || status === "pending") return <span className="badge run">{status}</span>;
  return <span className="badge err">{status}</span>;
}
