import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { BatchStatus, StepRecord, TrajectorySummary } from "../types";

// Page: watch all N trajectories of a consensus run progress live. A grid shows each
// trajectory's status + step count; selecting one streams its code/output below. When the
// trajectories finish, a meta-analysis synthesizes the consensus and "View results" opens.
export default function LiveTrajectory({
  runId,
  onViewResults,
}: {
  runId: string;
  onViewResults: () => void;
}) {
  const [batch, setBatch] = useState<BatchStatus | null>(null);
  const [selected, setSelected] = useState(0);
  const [steps, setSteps] = useState<StepRecord[]>([]);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setBatch(null);
    setSelected(0);
    setSteps([]);

    async function poll() {
      try {
        const b = await api.getRun(runId);
        setBatch(b);
        if (["completed", "failed", "error"].includes(b.status) && timer.current) {
          window.clearInterval(timer.current);
        }
      } catch {
        /* keep polling */
      }
    }
    poll();
    timer.current = window.setInterval(poll, 1500);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [runId]);

  // Stream the selected trajectory's steps.
  useEffect(() => {
    let active = true;
    async function pollSteps() {
      try {
        const s = await api.getSteps(runId, selected);
        if (active) setSteps(s.steps);
      } catch {
        /* ignore */
      }
    }
    pollSteps();
    const t = window.setInterval(pollSteps, 1500);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, [runId, selected]);

  const terminal = batch && ["completed", "failed", "error"].includes(batch.status);
  const synthesizing = batch?.phase === "synthesizing";

  return (
    <div className="page">
      <header className="page-head">
        <h2>Live trajectories</h2>
        <p className="subtitle">
          {batch ? batch.run.n_trajectories : "—"} independent agents analyzing the data.
          Watch each one reason, then a consensus is synthesized from all of them.
        </p>
      </header>

      <div className="card">
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <BatchStatusBadge status={batch?.status} />
          {synthesizing && (
            <span className="muted">
              <span className="spinner" /> synthesizing consensus…
            </span>
          )}
          {terminal && (
            <button className="primary" style={{ marginLeft: "auto" }} onClick={onViewResults}>
              View results →
            </button>
          )}
        </div>
      </div>

      {batch && (
        <div className="card">
          <div className="card-title">Trajectories</div>
          <div className="traj-grid">
            {batch.trajectories.map((t) => (
              <button
                key={t.idx}
                className={`traj-chip ${selected === t.idx ? "active" : ""}`}
                onClick={() => setSelected(t.idx)}
              >
                <span className="traj-idx">#{t.idx}</span>
                <TrajBadge t={t} />
                <span className="traj-steps">{t.steps} steps</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">Trajectory #{selected}</div>
        {steps.length === 0 ? (
          <div className="empty">
            {terminal ? "No steps recorded." : "Waiting for this trajectory to start…"}
          </div>
        ) : (
          steps.map((s) => (
            <div className="step" key={s.step}>
              <div className="step-head">
                <span className="badge run">step {s.step}</span>
                <span>{s.tool}</span>
              </div>
              {s.tool === "edit_cell" && <pre>{String((s.arguments as any).source ?? "")}</pre>}
              {s.tool === "submit_answer" && (
                <pre>{String((s.arguments as any).answer ?? "")}</pre>
              )}
              <div className="step-head" style={{ background: "transparent" }}>
                observation
              </div>
              <pre>{s.observation}</pre>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function BatchStatusBadge({ status }: { status?: string }) {
  if (!status) return <span className="badge run">loading</span>;
  const cls =
    status === "completed" ? "ok" : ["pending", "running", "synthesizing"].includes(status) ? "run" : "err";
  return <span className={`badge ${cls}`}>{status}</span>;
}

function TrajBadge({ t }: { t: TrajectorySummary }) {
  if (t.status === "completed" || t.submitted) return <span className="badge ok">done</span>;
  if (t.status === "running") return <span className="badge run">running</span>;
  if (t.status === "pending") return <span className="badge run">pending</span>;
  return <span className="badge err">{t.status}</span>;
}
