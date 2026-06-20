import { useEffect, useState } from "react";
import { api } from "../api";

// Page 2: edit config.yaml. The research question is the prominent field. We make it
// explicit that this is NOT where gates/populations/thresholds are specified.
export default function ConfigEditor({
  projectId,
  onNext,
}: {
  projectId: string;
  onNext: () => void;
}) {
  const [content, setContent] = useState("");
  const [exists, setExists] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; error?: string } | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    api.getConfig(projectId).then((c) => {
      setContent(c.content);
      setExists(c.exists);
      setLoaded(true);
    });
  }, [projectId]);

  async function save() {
    const r = await api.saveConfig(projectId, content);
    setStatus(r);
    if (r.ok) setExists(true);
  }

  // Pull the question line out for a prominent dedicated editor.
  const questionMatch = content.match(/^question:\s*"?(.*?)"?\s*$/m);
  const question = questionMatch ? questionMatch[1] : "";

  function setQuestion(q: string) {
    const safe = q.replace(/"/g, "'");
    if (questionMatch) {
      setContent(content.replace(/^question:.*$/m, `question: "${safe}"`));
    } else {
      setContent(`question: "${safe}"\n` + content);
    }
  }

  if (!loaded) return <p>Loading config…</p>;

  return (
    <div>
      <h2>Config</h2>
      <div className="banner ok">
        This config is <strong>analysis-free</strong> by design. It says <em>where</em> the
        data is, <em>what</em> to ask, and <em>how</em> to run — never <em>how to analyze</em>.
        Gates, populations, and thresholds are <strong>not</strong> set here; the agent
        derives all analysis from your data, question, and metadata.json.
      </div>

      <div className="card">
        <label htmlFor="q">Research question (the most important field)</label>
        <textarea
          id="q"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How does the CAR-detector signal change across timepoints?"
        />
      </div>

      <div className="card">
        <label htmlFor="cfg">
          Full config.yaml{" "}
          <span className="hint">{exists ? "(saved in project)" : "(default — not yet saved)"}</span>
        </label>
        <textarea
          id="cfg"
          rows={20}
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
          <button onClick={save}>Save config</button>
          <button className="secondary" onClick={onNext}>
            Next: Launch Run →
          </button>
        </div>
        {status && (
          <div className={`banner ${status.ok ? "ok" : "err"}`} style={{ marginTop: 12 }}>
            {status.ok ? "Saved and validated." : `Rejected: ${status.error}`}
          </div>
        )}
      </div>
    </div>
  );
}
