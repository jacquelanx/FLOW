import { useEffect, useState } from "react";
import { api } from "./api";
import type { DockerHealth, Project } from "./types";
import ProjectUpload from "./components/ProjectUpload";
import ConfigEditor from "./components/ConfigEditor";
import RunLauncher from "./components/RunLauncher";
import LiveTrajectory from "./components/LiveTrajectory";
import NotebookViewer from "./components/NotebookViewer";
import ResultsDashboard from "./components/ResultsDashboard";
import ArtifactBrowser from "./components/ArtifactBrowser";
import RunHistory from "./components/RunHistory";

type Page =
  | "project"
  | "config"
  | "run"
  | "trajectory"
  | "notebook"
  | "results"
  | "artifacts"
  | "history";

const PAGES: { id: Page; label: string; needsProject?: boolean; needsRun?: boolean }[] = [
  { id: "project", label: "1 · Project & Upload" },
  { id: "config", label: "2 · Config", needsProject: true },
  { id: "run", label: "3 · Launch Run", needsProject: true },
  { id: "trajectory", label: "4 · Live Trajectory", needsRun: true },
  { id: "notebook", label: "Notebook", needsRun: true },
  { id: "results", label: "Results", needsRun: true },
  { id: "artifacts", label: "Artifacts", needsRun: true },
  { id: "history", label: "Run History", needsProject: true },
];

export default function App() {
  const [page, setPage] = useState<Page>(
    () => (localStorage.getItem("flow.page") as Page) || "project",
  );
  const [projectId, setProjectId] = useState<string | null>(
    () => localStorage.getItem("flow.projectId"),
  );
  // The *selected* run (for viewing past runs) is distinct from the *active* run just
  // launched in this session, so a persisted selection never hijacks the launcher.
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    () => localStorage.getItem("flow.selectedRunId"),
  );
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [health, setHealth] = useState<DockerHealth | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  useEffect(() => {
    api.health().then((h) => setHealth(h.docker)).catch(() => setHealth(null));
    api.listProjects().then(setProjects).catch(() => setProjects([]));
  }, []);

  useEffect(() => {
    localStorage.setItem("flow.page", page);
  }, [page]);
  useEffect(() => {
    if (projectId) localStorage.setItem("flow.projectId", projectId);
  }, [projectId]);
  useEffect(() => {
    if (selectedRunId) localStorage.setItem("flow.selectedRunId", selectedRunId);
  }, [selectedRunId]);

  // The run the viewer pages should display: the active one (just launched) wins.
  const viewRunId = activeRunId || selectedRunId;

  function go(p: Page) {
    setPage(p);
  }

  function onRunStarted(rid: string) {
    setActiveRunId(rid);
    setSelectedRunId(rid);
    setPage("trajectory");
  }

  function onSelectRun(rid: string) {
    // Selecting a historical run must NOT advance the launcher; only viewer pages.
    setSelectedRunId(rid);
    setActiveRunId(null);
    setPage("results");
  }

  const refreshProjects = () => api.listProjects().then(setProjects).catch(() => {});

  return (
    <div className="app">
      <nav className="sidebar">
        <h1>FLOW</h1>
        <div className="tag">Finch-faithful analysis agent</div>
        {PAGES.map((p) => {
          const disabled =
            (p.needsProject && !projectId) || (p.needsRun && !viewRunId);
          return (
            <button
              key={p.id}
              className={`navbtn ${page === p.id ? "active" : ""}`}
              disabled={disabled}
              onClick={() => go(p.id)}
            >
              {p.label}
            </button>
          );
        })}
        <div style={{ marginTop: "auto", fontSize: "0.78rem" }}>
          <DockerBadge health={health} />
        </div>
      </nav>

      <main className="main">
        {page === "project" && (
          <ProjectUpload
            projects={projects}
            projectId={projectId}
            setProjectId={(id) => {
              setProjectId(id);
              setActiveRunId(null);
              setSelectedRunId(null);
            }}
            refreshProjects={refreshProjects}
            onNext={() => go("config")}
          />
        )}
        {page === "config" && projectId && (
          <ConfigEditor projectId={projectId} onNext={() => go("run")} />
        )}
        {page === "run" && projectId && (
          <RunLauncher
            projectId={projectId}
            health={health}
            onRunStarted={onRunStarted}
          />
        )}
        {page === "trajectory" && viewRunId && (
          <LiveTrajectory runId={viewRunId} onViewResults={() => go("results")} />
        )}
        {page === "notebook" && viewRunId && <NotebookViewer runId={viewRunId} />}
        {page === "results" && viewRunId && (
          <ResultsDashboard runId={viewRunId} onBrowse={() => go("artifacts")} />
        )}
        {page === "artifacts" && viewRunId && <ArtifactBrowser runId={viewRunId} />}
        {page === "history" && projectId && (
          <RunHistory projectId={projectId} onSelectRun={onSelectRun} />
        )}
        {!projectId && page !== "project" && (
          <div className="banner warn">Select or create a project first.</div>
        )}
      </main>
    </div>
  );
}

function DockerBadge({ health }: { health: DockerHealth | null }) {
  if (!health) return <span className="muted">backend: connecting…</span>;
  if (health.ready)
    return <span className="badge ok">Docker ready</span>;
  return (
    <span className="badge warn" title={health.notes.join(" ")}>
      Docker not ready
    </span>
  );
}
