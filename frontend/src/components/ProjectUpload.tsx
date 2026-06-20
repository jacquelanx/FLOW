import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Profile, Project } from "../types";

// Page 1: create/select a project and drag-drop the lab's files. Shows per-file
// validation feedback so non-computational users know what was recognized.
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

  return (
    <div>
      <h2>Project & Upload</h2>

      <div className="card">
        <h3>Select an existing project</h3>
        {projects.length === 0 && <p className="muted">No projects yet — create one below.</p>}
        <div className="row">
          <div>
            <select
              value={projectId || ""}
              onChange={(e) => setProjectId(e.target.value)}
              aria-label="Select project"
            >
              <option value="">— choose —</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.id})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Create a new project</h3>
        <div className="row">
          <div>
            <input
              placeholder="Project name (e.g. Patient-07 flow run)"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <button onClick={create}>Create</button>
        </div>
      </div>

      {projectId && (
        <div className="card">
          <h3>Upload data files</h3>
          <p className="hint">
            Drag &amp; drop metadata.json, flow.csv, alc.csv, your cytometry event table
            or FCS files, and optionally config.yaml / true_lab_results.csv. Files are
            validated on upload — they are never interpreted as analysis instructions.
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
        </div>
      )}

      {profile && (
        <div className="card">
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
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 14 }}>
            <button onClick={onNext}>Next: Config →</button>
          </div>
        </div>
      )}
    </div>
  );
}
