import type { DockerHealth, Project } from "../types";

// App chrome: the wordmark, a project switcher reachable from every page, and the sandbox
// status. Lifting these out of the nav rail means you can change project without first
// navigating back to page 1.
export default function TopBar({
  projects,
  projectId,
  onSelectProject,
  health,
}: {
  projects: Project[];
  projectId: string | null;
  onSelectProject: (id: string) => void;
  health: DockerHealth | null;
}) {
  const current = projects.find((p) => p.id === projectId) ?? null;
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brandmark">FLOW</span>
        <span className="brandtag">Flow cytometry analysis agent</span>
      </div>

      <div className="topbar-field">
        <label htmlFor="topbar-project">Project</label>
        <select
          id="topbar-project"
          className="project-switch"
          value={projectId ?? ""}
          title={current ? `${current.name} · ${current.id}` : "No project selected"}
          onChange={(e) => {
            // The placeholder is not a selectable target — ignore it.
            if (e.target.value) onSelectProject(e.target.value);
          }}
        >
          {(!projectId || projects.length === 0) && (
            <option value="">Select a project…</option>
          )}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
        {current && <span className="topbar-id mono">{current.id}</span>}
      </div>

      <div className="topbar-right">
        <span className="foot-label">Sandbox</span>
        <DockerBadge health={health} />
      </div>
    </header>
  );
}

export function DockerBadge({ health }: { health: DockerHealth | null }) {
  if (!health) return <span className="muted">backend: connecting…</span>;
  if (health.ready) return <span className="badge ok">Docker ready</span>;
  return (
    <span className="badge warn" title={health.notes.join(" ")}>
      Docker not ready
    </span>
  );
}
