// Horizontal rail for the four-step setup workflow, shown above every workflow page.
// "Done" is only ever claimed from real state the backend reports (see App's progress
// probe) — never from "you visited this page".

export type StepId = "project" | "config" | "analysis" | "run";

export type StepProgress = Record<StepId, boolean>;

const STEPS: { id: StepId; n: number; title: string; doneHint: string }[] = [
  { id: "project", n: 1, title: "Project & upload", doneHint: "A project is selected" },
  { id: "config", n: 2, title: "Configure", doneHint: "A research question is saved" },
  { id: "analysis", n: 3, title: "Analysis", doneHint: "Your analysis spec is saved" },
  { id: "run", n: 4, title: "Launch run", doneHint: "This project has at least one run" },
];

export default function Stepper({
  current,
  progress,
  hasProject,
  onGo,
}: {
  current: StepId;
  progress: StepProgress;
  hasProject: boolean;
  onGo: (id: StepId) => void;
}) {
  return (
    <nav className="stepper" aria-label="Setup progress">
      {STEPS.map((s) => {
        const active = s.id === current;
        const done = progress[s.id];
        const disabled = s.id !== "project" && !hasProject;
        const state = active ? "You are here" : done ? "Done" : "";
        return (
          <button
            key={s.id}
            className={`stepper-item ${active ? "active" : ""} ${done ? "done" : ""}`}
            disabled={disabled}
            aria-current={active ? "step" : undefined}
            title={
              disabled
                ? "Select or create a project first"
                : done
                  ? s.doneHint
                  : `Go to step ${s.n}: ${s.title}`
            }
            onClick={() => onGo(s.id)}
          >
            <span className="stepnum">{done && !active ? "✓" : s.n}</span>
            <span className="steplabel">
              <span className="steptitle">{s.title}</span>
              {state && <span className="stepstate">{state}</span>}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
