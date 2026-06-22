import type { TrajectorySummary } from "../types";

// Small selector for choosing which trajectory of a consensus run to inspect. Shared by the
// Notebook and Artifacts pages so they always view a single, well-defined trajectory.
export default function TrajectoryPicker({
  trajectories,
  idx,
  onChange,
}: {
  trajectories: TrajectorySummary[];
  idx: number;
  onChange: (idx: number) => void;
}) {
  if (trajectories.length === 0) return null;
  return (
    <div className="card" style={{ display: "flex", alignItems: "center", gap: 14 }}>
      <label htmlFor="trajsel" style={{ margin: 0 }}>
        Trajectory
      </label>
      <select
        id="trajsel"
        style={{ width: "auto", minWidth: 220 }}
        value={idx}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
      >
        {trajectories.map((t) => (
          <option key={t.idx} value={t.idx}>
            #{t.idx} — {t.submitted ? "answered" : t.status}
          </option>
        ))}
      </select>
    </div>
  );
}
