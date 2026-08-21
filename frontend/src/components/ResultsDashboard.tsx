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

  // Only trajectories whose first-run actually left obligations can report coverage. A null
  // total means the question was never asked, so those rows show "—" rather than a clean
  // score they did not earn.
  const gated = batch.trajectories.filter((t) => t.obligations_total != null);
  const nUnread = gated.reduce((n, t) => n + (t.obligations_undischarged ?? 0), 0);

  return (
    <div className="page">
      <header className="page-head">
        <h2>Results</h2>
        <p className="subtitle">
          A consensus synthesized from {batch.run.n_trajectories} independent trajectories,
          with each trajectory’s own conclusion below it.
        </p>
      </header>

      {/* Headline numbers first — the shape of the run before any of its prose. */}
      <section className="card">
        <div className="statgrid">
          <div className="stat">
            <span className="stat-label">Status</span>
            <span className="stat-value" style={{ fontSize: "0.95rem" }}>
              {status}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Trajectories</span>
            <span className="stat-value">{batch.run.n_trajectories}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Answered</span>
            <span className="stat-value">
              {nSubmitted}
              <span className="muted" style={{ fontWeight: 400 }}>
                {" "}
                / {batch.run.n_trajectories}
              </span>
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Agent steps</span>
            <span className="stat-value">
              {batch.trajectories.reduce((n, t) => n + t.steps, 0)}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Consensus</span>
            <span className="stat-value" style={{ fontSize: "0.95rem" }}>
              {consensus.consensus ? (consensus.synthesized ? "synthesized" : "single") : "—"}
            </span>
          </div>
          {gated.length > 0 && (
            <div className="stat">
              <span className="stat-label">Evidence unread</span>
              <span className="stat-value">
                {nUnread}
                <span className="muted" style={{ fontWeight: 400 }}>
                  {" "}
                  / {gated.reduce((n, t) => n + (t.obligations_total ?? 0), 0)}
                </span>
              </span>
            </div>
          )}
          <div className="stat">
            <span className="stat-label">Model</span>
            <span className="stat-value small" title={batch.run.model}>
              {batch.run.model || "—"}
            </span>
          </div>
        </div>
      </section>

      {batch.run.question && (
        <section className="card">
          <div className="card-title">Research question</div>
          <div className="prose">{batch.run.question}</div>
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
          <div
            className="card-title"
            style={{ display: "flex", alignItems: "center", gap: 12 }}
          >
            <span>
              Consensus
              {consensus.synthesized === false && " (single trajectory)"}
            </span>
            <CopyButton text={consensus.consensus} />
          </div>
          <div className="prose">{consensus.consensus}</div>
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
              {gated.length > 0 && <th>Evidence</th>}
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
                  {gated.length > 0 && (
                    <td>
                      {t.obligations_total == null ? (
                        <span className="muted">—</span>
                      ) : t.obligations_undischarged ? (
                        <span className="badge err">
                          {t.obligations_undischarged}/{t.obligations_total} unread
                        </span>
                      ) : (
                        <span className="badge ok">all read</span>
                      )}
                    </td>
                  )}
                  <td>
                    <span className="caret">
                      <span className={`tri ${isOpen ? "open" : ""}`} aria-hidden="true" />
                    </span>
                    {isOpen ? text : truncate(text, 90)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="hint">
          Click a row to expand its full conclusion.
          {gated.length > 0 &&
            " “Evidence” counts measurements this project’s first-run required the analyst to" +
              " consult; unread items are listed in each trajectory’s evidence_coverage.json."}
        </p>
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

// Copy the consensus text so it can be pasted into a lab notebook or email.
function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="secondary"
      style={{ padding: "2px 10px", fontSize: "0.76rem", textTransform: "none" }}
      title="Copy the consensus text"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setDone(true);
          setTimeout(() => setDone(false), 1600);
        } catch {
          /* clipboard unavailable — nothing to do */
        }
      }}
    >
      {done ? "Copied" : "Copy"}
    </button>
  );
}
