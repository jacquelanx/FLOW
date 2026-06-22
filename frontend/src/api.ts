// Thin typed wrapper over the FLOW backend API. All calls go to /api (proxied in dev).

import type {
  Artifact,
  Batch,
  BatchStatus,
  ConfigResponse,
  DockerHealth,
  Profile,
  Project,
  ProvidersInfo,
  StepRecord,
  TrajectorySummary,
} from "./types";

export interface ConfigForm {
  question: string;
  description?: string;
  provider: string;
  model: string;
  meta_provider?: string;
  meta_model?: string;
  max_steps: number;
  per_cell_timeout?: number;
  per_trajectory_timeout?: number;
  memory?: string;
  cpus?: string;
  pids_limit?: number;
  allow_network?: boolean;
  allow_raw_data_to_model?: boolean;
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async health(): Promise<{ status: string; docker: DockerHealth }> {
    return j(await fetch("/api/health"));
  },
  async providers(): Promise<string[]> {
    const r = await j<{ providers: string[] }>(await fetch("/api/providers"));
    return r.providers;
  },
  async providersInfo(): Promise<ProvidersInfo> {
    return j(await fetch("/api/providers"));
  },
  async listProjects(): Promise<Project[]> {
    const r = await j<{ projects: Project[] }>(await fetch("/api/projects"));
    return r.projects;
  },
  async createProject(name: string): Promise<Project> {
    return j(
      await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }),
    );
  },
  async getProject(pid: string): Promise<{ project: Project; profile: Profile }> {
    return j(await fetch(`/api/projects/${pid}`));
  },
  async profile(pid: string): Promise<Profile> {
    return j(await fetch(`/api/projects/${pid}/profile`));
  },
  async uploadFile(pid: string, file: File): Promise<{ file: any; size: number }> {
    const fd = new FormData();
    fd.append("file", file);
    return j(await fetch(`/api/projects/${pid}/files`, { method: "POST", body: fd }));
  },
  async getConfig(pid: string): Promise<ConfigResponse> {
    return j(await fetch(`/api/projects/${pid}/config`));
  },
  async saveConfig(pid: string, content: string): Promise<{ ok: boolean; error?: string }> {
    return j(
      await fetch(`/api/projects/${pid}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      }),
    );
  },
  async saveConfigForm(
    pid: string,
    form: ConfigForm,
  ): Promise<{ ok: boolean; error?: string; content?: string }> {
    return j(
      await fetch(`/api/projects/${pid}/config/form`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      }),
    );
  },
  async startRun(body: {
    project_id: string;
    question?: string;
    provider: string;
    model: string;
    meta_provider?: string;
    meta_model?: string;
    max_steps?: number;
    n_trajectories: number;
  }): Promise<{ run_id: string; status: string; n_trajectories: number }> {
    return j(
      await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },
  async listRuns(projectId?: string): Promise<Batch[]> {
    const q = projectId ? `?project_id=${projectId}` : "";
    const r = await j<{ runs: Batch[] }>(await fetch(`/api/runs${q}`));
    return r.runs;
  },
  async getRun(rid: string): Promise<BatchStatus> {
    return j(await fetch(`/api/runs/${rid}`));
  },
  async getTrajectories(rid: string): Promise<TrajectorySummary[]> {
    const r = await j<{ trajectories: TrajectorySummary[] }>(
      await fetch(`/api/runs/${rid}/trajectories`),
    );
    return r.trajectories;
  },
  async getSteps(rid: string, idx: number): Promise<{ steps: StepRecord[]; status: string }> {
    return j(await fetch(`/api/runs/${rid}/trajectories/${idx}/steps`));
  },
  async getNotebook(rid: string, idx: number): Promise<any> {
    return j(await fetch(`/api/runs/${rid}/trajectories/${idx}/notebook`));
  },
  async listArtifacts(rid: string, idx: number): Promise<Artifact[]> {
    const r = await j<{ artifacts: Artifact[] }>(
      await fetch(`/api/runs/${rid}/trajectories/${idx}/artifacts`),
    );
    return r.artifacts;
  },
  artifactUrl(rid: string, idx: number, path: string): string {
    return `/api/runs/${rid}/trajectories/${idx}/artifacts/${path}`;
  },
  downloadUrl(rid: string): string {
    return `/api/runs/${rid}/download`;
  },
};
