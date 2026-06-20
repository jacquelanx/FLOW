import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { StepRecord } from "../types";

// Page 4: stream the agent's steps live — each edit_cell's code and the cell's output,
// so a lab member can watch the agent reason. Polls only the run passed in (the active
// run from this session), never a stale selection.
export default function LiveTrajectory({
  runId,
  onViewResults,
}: {
  runId: string;
  onViewResults: () => void;
}) {
  const [steps, setSteps] = useState<StepRecord[]>([]);
  const [status, setStatus] = useState<string>("pending");
  const [executing, setExecuting] = useState<boolean>(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setSteps([]);
    setStatus("pending");

    async function poll() {
      try {
        const s = await api.getSteps(runId);
        setSteps(s.steps);
        const run = await api.getRun(runId);
        setStatus(run.status);
        setExecuting(Boolean((run.live as any)?.executing_in_docker));
        if (["completed", "failed", "error"].includes(run.status)) {
          if (timer.current) window.clearInterval(timer.current);
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

  const terminal = ["completed", "failed", "error"].includes(status);

  return (
    <div className="page">
      <header className="page-head">
        <h2>Live trajectory</h2>
        <p className="subtitle">
          Watch the agent reason in real time — each cell of code it writes and the output it
          gets back.
        </p>
      </header>
      <div className="card">
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <StatusBadge status={status} />
          {executing && (
            <span className="muted">
              <span className="spinner" /> executing in Docker…
            </span>
          )}
          <span className="muted">run {runId}</span>
          {terminal && (
            <button style={{ marginLeft: "auto" }} onClick={onViewResults}>
              View results →
            </button>
          )}
        </div>
      </div>

      {steps.length === 0 && !terminal && (
        <p className="muted">
          <span className="spinner" /> Waiting for the agent's first step…
        </p>
      )}

      {steps.map((s) => (
        <div className="step" key={s.step}>
          <div className="step-head">
            <span className="badge run">step {s.step}</span>
            <span>{s.tool}</span>
          </div>
          {s.tool === "edit_cell" && (
            <pre>{String((s.arguments as any).source ?? "")}</pre>
          )}
          {s.tool === "submit_answer" && (
            <pre>{String((s.arguments as any).answer ?? "")}</pre>
          )}
          <div className="step-head" style={{ background: "transparent" }}>
            observation
          </div>
          <pre>{s.observation}</pre>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "completed" ? "ok" : status === "running" || status === "pending" ? "run" : "err";
  return <span className={`badge ${cls}`}>{status}</span>;
}
