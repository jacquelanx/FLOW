import { useEffect, useState } from "react";
import { api } from "./api";
import type { DockerHealth, Project } from "./types";
import ProjectUpload from "./components/ProjectUpload";
import ConfigEditor from "./components/ConfigEditor";
import AnalysisEditor from "./components/AnalysisEditor";
import RunLauncher from "./components/RunLauncher";
import LiveTrajectory from "./components/LiveTrajectory";
import NotebookViewer from "./components/NotebookViewer";
import ResultsDashboard from "./components/ResultsDashboard";
import ArtifactBrowser from "./components/ArtifactBrowser";
import RunHistory from "./components/RunHistory";
import TopBar from "./components/TopBar";
import Stepper from "./components/Stepper";
import type { StepId, StepProgress } from "./components/Stepper";

type Page =
  | "project"
  | "config"
  | "analysis"
  | "run"
  | "trajectory"
  | "notebook"
  | "results"
  | "artifacts"
  | "history";

type NavItem = { id: Page; label: string; needsProject?: boolean; needsRun?: boolean };
// Grouped nav: the linear workflow is numbered; the run-output views and history sit under
// their own section labels, so the absence of numbers there reads as intentional.
const NAV_GROUPS: { section?: string; items: NavItem[] }[] = [
  {
    section: "Set up",
    items: [
      { id: "project", label: "1 · Project & Upload" },
      { id: "config", label: "2 · Configure", needsProject: true },
      { id: "analysis", label: "3 · Analysis", needsProject: true },
      { id: "run", label: "4 · Launch Run", needsProject: true },
    ],
  },
  {
    section: "Review run",
    items: [
      { id: "trajectory", label: "Live Trajectory", needsRun: true },
      { id: "notebook", label: "Notebook", needsRun: true },
      { id: "results", label: "Results", needsRun: true },
      { id: "artifacts", label: "Artifacts", needsRun: true },
    ],
  },
  {
    section: "History",
    items: [{ id: "history", label: "Run History", needsProject: true }],
  },
];

// "1 · Project & Upload" → ["1", "Project & Upload"] so the step number can be set as a
// counter chip. Labels without a number are returned whole.
function splitStepLabel(label: string): [string | null, string] {
  const m = label.match(/^(\d+)\s*·\s*(.+)$/);
  return m ? [m[1], m[2]] : [null, label];
}

// The four setup pages the stepper covers.
const STEP_PAGES: StepId[] = ["project", "config", "analysis", "run"];
const isStepPage = (p: Page): p is StepId => (STEP_PAGES as string[]).includes(p);

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
  // Which setup steps are genuinely finished, read from the backend — never inferred from
  // "this page was visited". Re-probed on project change and on every navigation, so
  // saving a question on step 2 lights step 2 up as soon as you move on.
  const [progress, setProgress] = useState<StepProgress>({
    project: false,
    config: false,
    analysis: false,
    run: false,
  });

  useEffect(() => {
    api.health().then((h) => setHealth(h.docker)).catch(() => setHealth(null));
    api.listProjects().then(setProjects).catch(() => setProjects([]));
  }, []);

  useEffect(() => {
    if (!projectId) {
      setProgress({ project: false, config: false, analysis: false, run: false });
      return;
    }
    let active = true;
    Promise.all([
      api.getConfig(projectId).catch(() => null),
      api.getAnalysis(projectId).catch(() => null),
      api.listRuns(projectId).catch(() => []),
    ]).then(([cfg, analysis, runs]) => {
      if (!active) return;
      setProgress({
        project: true,
        config: Boolean(cfg?.config?.question?.trim()),
        // `seeded` means the profile is still FLOW's starting point, not the lab's own.
        analysis: Boolean(analysis && !analysis.seeded),
        run: runs.length > 0,
      });
    });
    return () => {
      active = false;
    };
  }, [projectId, page]);

  useEffect(() => {
    localStorage.setItem("flow.page", page);
  }, [page]);
  useEffect(() => {
    // Clear the stored id when it goes empty, so a deleted/switched project can't reload.
    if (projectId) localStorage.setItem("flow.projectId", projectId);
    else localStorage.removeItem("flow.projectId");
  }, [projectId]);
  useEffect(() => {
    // Clear the stored run when deselected (e.g. on project switch), so a run from one
    // project never reloads while a different project is selected.
    if (selectedRunId) localStorage.setItem("flow.selectedRunId", selectedRunId);
    else localStorage.removeItem("flow.selectedRunId");
  }, [selectedRunId]);

  // The run the viewer pages should display: the active one (just launched) wins.
  const viewRunId = activeRunId || selectedRunId;

  function go(p: Page) {
    setPage(p);
  }

  // Switching project anywhere (page 1 list or the top-bar switcher) must drop the run
  // selection, so a run from one project never shows while another is open.
  function selectProject(id: string) {
    setProjectId(id || null);
    setActiveRunId(null);
    setSelectedRunId(null);
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
    <div className="shell">
      <TopBar
        projects={projects}
        projectId={projectId}
        onSelectProject={selectProject}
        health={health}
      />
      <div className="app">
        <nav className="sidebar">
          {NAV_GROUPS.map((group, gi) => (
            <div key={gi}>
              {group.section && <div className="navsection">{group.section}</div>}
              <div className="navgroup">
                {group.items.map((p) => {
                  const disabled =
                    (p.needsProject && !projectId) || (p.needsRun && !viewRunId);
                  // Say *why* a step is unavailable rather than just greying it out.
                  const why = !disabled
                    ? undefined
                    : p.needsRun && !viewRunId
                      ? "Available once a run has been launched or opened from Run History"
                      : "Select or create a project first";
                  const [num, text] = splitStepLabel(p.label);
                  const done = isStepPage(p.id) && progress[p.id];
                  return (
                    <button
                      key={p.id}
                      className={`navbtn ${page === p.id ? "active" : ""} ${
                        done ? "done" : ""
                      }`}
                      disabled={disabled}
                      title={why ?? p.label}
                      aria-current={page === p.id ? "page" : undefined}
                      onClick={() => go(p.id)}
                    >
                      {num && (
                        <span className="navnum">
                          {done && page !== p.id ? "✓" : num}
                        </span>
                      )}
                      <span className="navlabel">{text}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        <main className="main">
          {isStepPage(page) && (
            <Stepper
              current={page}
              progress={progress}
              hasProject={Boolean(projectId)}
              onGo={(id) => go(id)}
            />
          )}
          {page === "project" && (
            <ProjectUpload
              projects={projects}
              projectId={projectId}
              setProjectId={selectProject}
              refreshProjects={refreshProjects}
              onNext={() => go("config")}
            />
          )}
          {page === "config" && projectId && (
            <ConfigEditor projectId={projectId} onNext={() => go("analysis")} />
          )}
          {page === "analysis" && projectId && (
            <AnalysisEditor projectId={projectId} onNext={() => go("run")} />
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
            <RunHistory
              projectId={projectId}
              onSelectRun={onSelectRun}
              onRunDeleted={(rid) => {
                // If the deleted run was the one being viewed, drop the selection.
                if (selectedRunId === rid) setSelectedRunId(null);
                if (activeRunId === rid) setActiveRunId(null);
                if (localStorage.getItem("flow.selectedRunId") === rid) {
                  localStorage.removeItem("flow.selectedRunId");
                }
              }}
            />
          )}
          {!projectId && page !== "project" && (
            <div className="banner warn">Select or create a project first.</div>
          )}
        </main>
      </div>
    </div>
  );
}
