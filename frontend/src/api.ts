// Thin typed wrapper over the FLOW backend API. All calls go to /api (proxied in dev).

import type {
  Artifact,
  DockerHealth,
  Profile,
  Project,
  Run,
  RunStatus,
  StepRecord,
} from "./types";

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
  async getConfig(pid: string): Promise<{ content: string; exists: boolean }> {
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
  async startRun(body: {
    project_id: string;
    question?: string;
    provider: string;
    model: string;
    max_steps?: number;
  }): Promise<{ run_id: string; status: string }> {
    return j(
      await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },
  async listRuns(projectId?: string): Promise<Run[]> {
    const q = projectId ? `?project_id=${projectId}` : "";
    const r = await j<{ runs: Run[] }>(await fetch(`/api/runs${q}`));
    return r.runs;
  },
  async getRun(rid: string): Promise<RunStatus> {
    return j(await fetch(`/api/runs/${rid}`));
  },
  async getSteps(rid: string): Promise<{ steps: StepRecord[]; status: string }> {
    return j(await fetch(`/api/runs/${rid}/steps`));
  },
  async getNotebook(rid: string): Promise<any> {
    return j(await fetch(`/api/runs/${rid}/notebook`));
  },
  async listArtifacts(rid: string): Promise<Artifact[]> {
    const r = await j<{ artifacts: Artifact[] }>(await fetch(`/api/runs/${rid}/artifacts`));
    return r.artifacts;
  },
  artifactUrl(rid: string, path: string): string {
    return `/api/runs/${rid}/artifacts/${path}`;
  },
  downloadUrl(rid: string): string {
    return `/api/runs/${rid}/download`;
  },
};
