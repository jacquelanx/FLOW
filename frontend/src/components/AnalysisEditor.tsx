import { useEffect, useState } from "react";
import { api } from "../api";
import type { AnalysisProfile, GateStep, MarkerDef } from "../types";

// Customized instructions fed to the agent loop; seeded with the NK
// panel the first-run assumes, so an expert can review and take it over.
export default function AnalysisEditor({
  projectId,
  onNext,
}: {
  projectId: string;
  onNext: () => void;
}) {
  const [p, setP] = useState<AnalysisProfile | null>(null);
  const [seeded, setSeeded] = useState(false);
  const [preview, setPreview] = useState("");
  const [showPreview, setShowPreview] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; error?: string } | null>(null);
  const [saving, setSaving] = useState(false);

  // The deterministic first-run script
  const [script, setScript] = useState<string>("");
  const [scriptSeeded, setScriptSeeded] = useState(false);
  const [scriptSaved, setScriptSaved] = useState<boolean | null>(null);

  useEffect(() => {
    api.getAnalysis(projectId).then((r) => {
      setP(r.profile);
      setSeeded(r.seeded);
      setPreview(r.guidance_preview);
    });
    api.getFirstRunScript(projectId).then((r) => {
      setScript(r.content);
      setScriptSeeded(r.seeded);
    });
  }, [projectId]);

  async function saveScript() {
    const r = await api.saveFirstRunScript(projectId, script);
    setScriptSaved(r.ok);
    if (r.ok) setScriptSeeded(false);
  }

  if (!p) return <p className="muted">Loading…</p>;

  // Generic field setter
  const set = <K extends keyof AnalysisProfile>(k: K, v: AnalysisProfile[K]) =>
    setP({ ...p, [k]: v });

  async function save(thenNext = false) {
    if (!p) return;
    setSaving(true);
    const r = await api.saveAnalysis(projectId, p);
    setStatus(r);
    if (r.guidance_preview) setPreview(r.guidance_preview);
    if (r.ok) setSeeded(false);
    setSaving(false);
    if (r.ok && thenNext) onNext();
  }

  return (
    <div className="page">
      <header className="page-head">
        <h2>Analysis specification</h2>
        <p className="subtitle">
          Here, you can specify the <strong>first-run script</strong> that does the actual
          flow analysis as well as the <strong>spec below</strong> (panel, populations, comparisons)
          that guides how the agent interprets the results.
        </p>
      </header>

      {seeded && (
        <div className="banner ok">
          <strong>Starting point loaded.</strong> This is pre-filled with the NK cell-therapy
          panel the first-run pipeline assumes. Review and edit every field to match{" "}
          <em>your</em> assay, then Save — your version is what the agent will follow.
        </div>
      )}

      {/* First-run script, the deterministic analysis */}
      <section className="card">
        <h3 className="card-title">First-run script (the deterministic analysis)</h3>
        <p className="hint" style={{ marginTop: 0 }}>
          This Python script does <strong>all the flow-cytometry analysis</strong> — FLOW runs
          it before the agent, and the agent only interprets its output. You can edit the existing
          template here or upload your own.
        </p>
        {scriptSeeded && (
          <div className="banner ok" style={{ marginBottom: 12 }}>
            <strong>Example loaded.</strong> This is a worked NK-panel script as a starting point —
            edit it to match your assay (or replace it entirely), then Save.
          </div>
        )}
        <textarea
          className="code-editor"
          spellCheck={false}
          rows={18}
          value={script}
          onChange={(e) => {
            setScript(e.target.value);
            setScriptSaved(null);
          }}
          placeholder="# Your first-run analysis script (Python). Reads --data, writes CSVs to --out."
        />
        <div className="actions" style={{ marginTop: 12 }}>
          <label className="secondary" style={{ display: "inline-block", cursor: "pointer" }}>
            Upload .py
            <input
              type="file"
              accept=".py,text/x-python,text/plain"
              style={{ display: "none" }}
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setScript(await f.text());
                  setScriptSaved(null);
                }
              }}
            />
          </label>
          <button className="secondary" onClick={saveScript}>
            Save script
          </button>
          {scriptSaved === true && <span className="muted">Saved.</span>}
          {scriptSaved === false && <span className="badge err">Save failed</span>}
        </div>
      </section>

      {/* Panel name */}
      <section className="card">
        <label htmlFor="panel">Panel / assay name</label>
        <input
          id="panel"
          value={p.panel_name}
          onChange={(e) => set("panel_name", e.target.value)}
          placeholder="e.g. NK cell-therapy 18-color panel"
        />
      </section>

      {/* Marker map */}
      <section className="card">
        <h3 className="card-title">Panel — detector → marker</h3>
        <p className="hint" style={{ marginTop: 0 }}>
          Which fluorophore/detector carries which marker. These are facts about your panel.
        </p>
        <ObjList<MarkerDef>
          rows={p.markers}
          onChange={(rows) => set("markers", rows)}
          blank={{ detector: "", marker: "", notes: "" }}
          columns={[
            { key: "detector", label: "Detector", placeholder: "e.g. PE-Cy7-A" },
            { key: "marker", label: "Marker", placeholder: "e.g. CD70 (CAR)" },
            { key: "notes", label: "Notes (optional)", placeholder: "e.g. CAR readout" },
          ]}
        />
      </section>

      {/* Gating strategy */}
      <section className="card">
        <h3 className="card-title">Gating strategy (in order)</h3>
        <p className="hint" style={{ marginTop: 0 }}>
          The ordered gating hierarchy in your own words. The agent applies this on top of the
          first-run’s standard pass, and refines gates that look wrong.
        </p>
        <ObjList<GateStep>
          rows={p.gating}
          onChange={(rows) => set("gating", rows)}
          blank={{ name: "", definition: "" }}
          columns={[
            { key: "name", label: "Gate", placeholder: "e.g. NK cells" },
            {
              key: "definition",
              label: "Definition",
              placeholder: "e.g. CD3- CD19- CD14- and CD56+",
              wide: true,
            },
          ]}
        />
      </section>

      {/* Populations of interest */}
      <section className="card">
        <h3 className="card-title">Populations of interest</h3>
        <StrList
          items={p.populations_of_interest}
          onChange={(v) => set("populations_of_interest", v)}
          placeholder="e.g. CAR+ Donor NK"
        />
      </section>

      {/* Comparisons */}
      <section className="card">
        <h3 className="card-title">Comparisons / questions to answer</h3>
        <StrList
          items={p.comparisons}
          onChange={(v) => set("comparisons", v)}
          placeholder="e.g. Donor vs Patient NK across timepoints"
        />
      </section>

      {/* QC expectations */}
      <section className="card">
        <h3 className="card-title">Quality-control expectations</h3>
        <textarea
          rows={3}
          value={p.qc_expectations}
          onChange={(e) => set("qc_expectations", e.target.value)}
          placeholder="e.g. Flag samples with few CD45+ events or low viability; down-weight them."
        />
      </section>

      {/* Desired outputs */}
      <section className="card">
        <h3 className="card-title">The write-up should include</h3>
        <StrList
          items={p.desired_outputs}
          onChange={(v) => set("desired_outputs", v)}
          placeholder="e.g. A per-timepoint table of % and absolute counts"
        />
      </section>

      {/* Context + presentation */}
      <section className="card">
        <h3 className="card-title">Biological context & instructions</h3>
        <label htmlFor="ctx">Biological context / hypotheses <span className="optional">optional</span></label>
        <textarea
          id="ctx"
          rows={4}
          value={p.biological_context}
          onChange={(e) => set("biological_context", e.target.value)}
          placeholder="Anything the agent should know: prior expectations, what a positive result looks like, caveats…"
        />
        <label htmlFor="extra" style={{ marginTop: 16 }}>
          Additional instructions <span className="optional">optional</span>
        </label>
        <textarea
          id="extra"
          rows={3}
          value={p.extra_instructions}
          onChange={(e) => set("extra_instructions", e.target.value)}
          placeholder="Any other analysis instructions for the agent."
        />
        <label className="toggle" style={{ fontWeight: 600, marginTop: 16 }}>
          <input
            type="checkbox"
            checked={p.plain_language}
            onChange={(e) => set("plain_language", e.target.checked)}
          />
          <span>
            Explain results in plain language
            <span className="hint"> — the agent defines terms and writes for a non-specialist.</span>
          </span>
        </label>
      </section>

      <div className="actions">
        <button className="secondary" onClick={() => save(false)} disabled={saving}>
          {saving ? <span className="spinner" /> : "Save"}
        </button>
        <button className="primary" onClick={() => save(true)} disabled={saving}>
          Save &amp; continue →
        </button>
        <button
          className="ghost"
          style={{ marginLeft: "auto" }}
          onClick={() => setShowPreview((s) => !s)}
        >
          {showPreview ? "Hide" : "Preview"} agent guidance
        </button>
      </div>

      {status && (
        <div className={`banner ${status.ok ? "ok" : "err"}`}>
          {status.ok ? "Saved — the agent will follow this specification." : `Could not save: ${status.error}`}
        </div>
      )}

      {showPreview && (
        <section className="card">
          <h3 className="card-title">Agent guidance (generated from your spec)</h3>
          <p className="hint" style={{ marginTop: 0 }}>
            This is the exact natural-language instruction compiled from the fields above and
            handed to the agent. Save to refresh it.
          </p>
          <pre className="code">{preview}</pre>
        </section>
      )}
    </div>
  );
}

// ── Editable list of short strings (add / edit / remove) ──────────────────────
function StrList({
  items,
  onChange,
  placeholder,
}: {
  items: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
}) {
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="row-edit">
          <input
            value={it}
            placeholder={placeholder}
            onChange={(e) => {
              const next = [...items];
              next[i] = e.target.value;
              onChange(next);
            }}
          />
          <button
            className="ghost"
            title="Remove"
            onClick={() => onChange(items.filter((_, j) => j !== i))}
          >
            ✕
          </button>
        </div>
      ))}
      <button className="secondary" onClick={() => onChange([...items, ""])}>
        + Add
      </button>
    </div>
  );
}

// ── Editable list of objects (a small table with typed columns) ───────────────
type Col<T> = { key: keyof T; label: string; placeholder?: string; wide?: boolean };

function ObjList<T extends object>({
  rows,
  onChange,
  columns,
  blank,
}: {
  rows: T[];
  onChange: (rows: T[]) => void;
  columns: Col<T>[];
  blank: T;
}) {
  return (
    <div>
      <div className="objlist-head">
        {columns.map((c) => (
          <div key={String(c.key)} className={c.wide ? "wide" : ""}>
            {c.label}
          </div>
        ))}
        <div className="objlist-x" />
      </div>
      {rows.map((row, i) => (
        <div key={i} className="objlist-row">
          {columns.map((c) => (
            <input
              key={String(c.key)}
              className={c.wide ? "wide" : ""}
              value={row[c.key] as string}
              placeholder={c.placeholder}
              onChange={(e) => {
                const next = [...rows];
                next[i] = { ...row, [c.key]: e.target.value };
                onChange(next);
              }}
            />
          ))}
          <button
            className="ghost objlist-x"
            title="Remove"
            onClick={() => onChange(rows.filter((_, j) => j !== i))}
          >
            ✕
          </button>
        </div>
      ))}
      <button className="secondary" onClick={() => onChange([...rows, { ...blank }])}>
        + Add row
      </button>
    </div>
  );
}
