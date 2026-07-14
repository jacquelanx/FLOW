import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Profile, Project } from "../types";

// Accepted project files, shown as an always-visible reference so the page is informative
// even before anything is uploaded.
const ACCEPTED_FILES: { name: string; desc: string }[] = [
  { name: "metadata.json", desc: "Experiment facts (which detector carries which stain, key dates)." },
  { name: "flow.csv", desc: "Timepoint sheet — columns label, date." },
  { name: "alc.csv", desc: "WBC count (K/µL) over time — columns date, alc. Used for absolute counts." },
  { name: "FCS files / events.csv", desc: "A folder of FCS files, or a combined per-event table." },
  { name: "cbc.csv", desc: "Optional per-timepoint WBC override — columns label, wbc_kul." },
  { name: "prompt.md", desc: "Optional custom LLM prompt (added on top of the default)." },
  { name: "true_lab_results.csv", desc: "Optional ground truth — used for evaluation only." },
];

// Page 1: create/select a project and drag-drop the lab's files, with per-file validation.
export default function ProjectUpload({
  projects,
  projectId,
  setProjectId,
  refreshProjects,
  onNext,
}: {
  projects: Project[];
  projectId: string | null;
  setProjectId: (id: string) => void;
  refreshProjects: () => Promise<void>;
  onNext: () => void;
}) {
  const [name, setName] = useState("");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (projectId) api.profile(projectId).then(setProfile).catch(() => setProfile(null));
    else setProfile(null);
  }, [projectId]);

  async function create() {
    const p = await api.createProject(name || "Untitled project");
    await refreshProjects();
    setProjectId(p.id);
    setName("");
  }

  async function upload(files: FileList | File[]) {
    if (!projectId) return;
    setBusy(true);
    try {
      for (const f of Array.from(files)) {
        try {
          await api.uploadFile(projectId, f);
        } catch (e) {
          alert(`Upload failed for ${f.name}: ${(e as Error).message}`);
        }
      }
      setProfile(await api.profile(projectId));
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    if (!projectId) return;
    if (!confirm(`Delete ${name}? This cannot be undone.`)) return;
    try {
      await api.deleteFile(projectId, name);
      setProfile(await api.profile(projectId));
    } catch (e) {
      alert(`Could not delete ${name}: ${(e as Error).message}`);
    }
  }

  async function removeProject(p: Project) {
    if (
      !confirm(
        `Delete project "${p.name}" and ALL of its uploaded files and runs? This cannot be undone.`,
      )
    )
      return;
    try {
      await api.deleteProject(p.id);
      await refreshProjects();
      if (projectId === p.id) setProjectId(""); // clear selection if the current one was removed
    } catch (e) {
      alert(`Could not delete project: ${(e as Error).message}`);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h2>Project &amp; upload</h2>
        <p className="subtitle">
          A project is a folder of your experiment’s files. Create or select one, add your
          data, and FLOW will validate each file as you go.
        </p>
      </header>

      <div className="split">
        <div>
          <section className="card">
            <h3>Select a project</h3>
            {projects.length === 0 ? (
              <p className="muted">No projects yet — create one on the right to begin.</p>
            ) : (
              <ul className="proj-list">
                {projects.map((p) => (
                  <li key={p.id} className={p.id === projectId ? "active" : ""}>
                    <button
                      className="proj-pick"
                      onClick={() => setProjectId(p.id)}
                      aria-pressed={p.id === projectId}
                    >
                      <span className="proj-name">{p.name}</span>
                      <span className="proj-id">{p.id}</span>
                    </button>
                    <button
                      className="ghost"
                      title={`Delete project "${p.name}"`}
                      onClick={() => removeProject(p)}
                    >
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="card">
            <h3>Create a new project</h3>
            <div className="field">
              <input
                placeholder="Project name (e.g. Patient-07 flow run)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && create()}
              />
            </div>
            <div className="actions">
              <button className="primary" onClick={create}>
                Create project
              </button>
            </div>
          </section>

          {projectId && (
            <section className="card">
              <h3>Upload data files</h3>
              <p className="hint">
                Drag &amp; drop your files below. They’re validated on upload and never
                interpreted as analysis instructions.
              </p>
              <div
                className={`dropzone ${drag ? "drag" : ""}`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDrag(false);
                  upload(e.dataTransfer.files);
                }}
              >
                {busy ? <span className="spinner" /> : "Drop files here, or click to browse"}
              </div>
              <input
                ref={inputRef}
                type="file"
                multiple
                style={{ display: "none" }}
                onChange={(e) => e.target.files && upload(e.target.files)}
              />
            </section>
          )}

          {profile && (
            <section className="card">
              <h3>Validation</h3>
              {profile.notes.map((n, i) => (
                <div key={i} className="banner warn">
                  {n}
                </div>
              ))}
              <table>
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Recognized as</th>
                    <th>Rows</th>
                    <th>Columns</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {profile.files.map((f) => (
                    <tr key={f.name}>
                      <td>{f.name}</td>
                      <td>{f.kind}</td>
                      <td>{f.rows ?? "—"}</td>
                      <td className="muted">{f.columns.join(", ") || f.detail}</td>
                      <td>
                        {f.ok ? (
                          <span className="badge ok">ok</span>
                        ) : (
                          <span className="badge err" title={f.detail}>
                            error
                          </span>
                        )}
                      </td>
                      <td>
                        <button
                          className="ghost"
                          title={`Delete ${f.name}`}
                          onClick={() => remove(f.name)}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="actions" style={{ marginTop: 18 }}>
                <button className="primary" onClick={onNext}>
                  Next: configure →
                </button>
              </div>
            </section>
          )}
        </div>

        <aside className="aside">
          <section className="card">
            <div className="card-title">How it works</div>
            <ol className="steps-list">
              <li>Create a project and upload your data files.</li>
              <li>Set the research question and pick a model.</li>
              <li>An automated first-run pipeline preprocesses the data (e.g. gating).</li>
              <li>AI agents review &amp; refine those results, then agree on a consensus.</li>
            </ol>
          </section>

          <section className="card">
            <div className="card-title">Accepted files</div>
            <ul className="reflist">
              {ACCEPTED_FILES.map((f) => (
                <li key={f.name}>
                  <div className="name">{f.name}</div>
                  <div className="desc">{f.desc}</div>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </div>
    </div>
  );
}
