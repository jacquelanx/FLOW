import { useEffect, useState } from "react";
import { api } from "../api";
import type { BatchStatus } from "../types";

// Status-aware results view for a consensus run: the synthesized consensus up top, then the
// individual trajectory conclusions it was distilled from. Clear messaging while running or
// on failure — never a dead end.
export default function ResultsDashboard({
  runId,
  onBrowse,
}: {
  runId: string;
  onBrowse: () => void;
}) {
  const [batch, setBatch] = useState<BatchStatus | null>(null);
  const [open, setOpen] = useState<Record<number, boolean>>({});

  useEffect(() => {
    let active = true;
    async function load() {
      const b = await api.getRun(runId);
      if (!active) return;
      setBatch(b);
      if (!["completed", "failed", "error"].includes(b.status)) setTimeout(load, 2000);
    }
    load();
    return () => {
      active = false;
    };
  }, [runId]);

  if (!batch) return <p className="muted">Loading…</p>;

  const { status, consensus } = batch;
  const running = !["completed", "failed", "error"].includes(status);
  const nSubmitted = batch.trajectories.filter((t) => t.submitted).length;

  return (
    <div className="page">
      <header className="page-head">
        <h2>Results</h2>
        <p className="subtitle">
          A consensus synthesized from {batch.run.n_trajectories} independent trajectories,
          with each trajectory’s own conclusion below it.
        </p>
      </header>

      {batch.run.question && (
        <section className="card">
          <div className="card-title">Research question</div>
          <div className="code" style={{ fontSize: "0.95rem" }}>{batch.run.question}</div>
          <p className="hint">
            {batch.run.model && <>Model: {batch.run.model}. </>}
            {batch.run.n_trajectories} trajector{batch.run.n_trajectories === 1 ? "y" : "ies"}.
          </p>
        </section>
      )}

      {running && (
        <div className="banner warn">
          <span className="spinner" /> Run is {status}… {nSubmitted}/{batch.run.n_trajectories}{" "}
          trajectories done. The consensus appears once all finish.
        </div>
      )}

      {(status === "failed" || status === "error") && !consensus.consensus && (
        <div className="banner err">
          <strong>No consensus could be formed.</strong>
          <div style={{ marginTop: 6 }}>
            {consensus.failure_reason ||
              batch.run.failure_reason ||
              "No trajectory produced an answer (or execution could not start, e.g. Docker missing)."}
          </div>
        </div>
      )}

      {consensus.consensus && (
        <section className="card">
          <div className="card-title">
            Consensus
            {consensus.synthesized === false && " (single trajectory)"}
          </div>
          <div className="code" style={{ fontSize: "0.95rem" }}>{consensus.consensus}</div>
          <p className="hint">
            {consensus.synthesized
              ? `Synthesized from ${consensus.n_submitted} of ${consensus.n_total} trajectories that produced an answer.`
              : `Based on ${nSubmitted} trajectory${nSubmitted === 1 ? "" : "ies"} that produced an answer.`}
          </p>
        </section>
      )}

      <section className="card">
        <div className="card-title">Trajectory conclusions</div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Status</th>
              <th>Steps</th>
              <th>Conclusion</th>
            </tr>
          </thead>
          <tbody>
            {batch.trajectories.map((t) => {
              const isOpen = open[t.idx];
              const text = t.answer || t.failure_reason || "—";
              return (
                <tr key={t.idx} onClick={() => setOpen({ ...open, [t.idx]: !isOpen })} style={{ cursor: "pointer" }}>
                  <td>#{t.idx}</td>
                  <td>
                    {t.submitted ? (
                      <span className="badge ok">answered</span>
                    ) : (
                      <span className="badge err">{t.status}</span>
                    )}
                  </td>
                  <td>{t.steps}</td>
                  <td>{isOpen ? text : truncate(text, 90)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="hint">Click a row to expand its full conclusion.</p>
      </section>

      <div className="card">
        <div className="actions">
          <button className="secondary" onClick={onBrowse}>
            Browse artifacts →
          </button>
          <a href={api.downloadUrl(runId)}>
            <button className="primary">Download run (.zip)</button>
          </a>
        </div>
      </div>
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
