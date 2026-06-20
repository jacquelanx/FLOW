// Shared types mirroring the FastAPI responses.

export interface Project {
  id: string;
  name: string;
  created_at: number;
  path: string;
}

export interface FileReport {
  name: string;
  kind: string;
  ok: boolean;
  rows: number | null;
  columns: string[];
  detail: string;
}

export interface Profile {
  ok: boolean;
  notes: string[];
  files: FileReport[];
}

export interface Run {
  id: string;
  project_id: string;
  question: string;
  provider: string;
  model: string;
  status: string;
  created_at: number;
  finished_at: number | null;
  submitted: number;
  failure_reason: string | null;
  artifact_dir: string;
}

export interface StepRecord {
  step: number;
  tool: string;
  arguments: Record<string, unknown>;
  observation: string;
  done?: boolean;
  reward?: number;
}

export interface RunStatus {
  run: Run;
  live: Record<string, unknown>;
  status: string;
  answer: string;
  run_meta: Record<string, unknown>;
}

export interface Artifact {
  path: string;
  name: string;
  size: number;
  kind: string;
}

export interface ProvidersInfo {
  providers: string[];
  ui_providers: string[];
  catalog: Record<string, string[]>;
  default_models: Record<string, string>;
}

export interface RuntimeCfg {
  provider: string;
  model: string;
  max_steps: number;
  per_cell_timeout: number;
  per_trajectory_timeout: number;
}

export interface SafetyCfg {
  memory: string;
  cpus: string;
  pids_limit: number;
  allow_network: boolean;
  allow_raw_data_to_model: boolean;
}

export interface ParsedConfig {
  question: string;
  dataset: { root: string; description: string };
  runtime: RuntimeCfg;
  safety: SafetyCfg;
}

export interface ConfigResponse {
  content: string;
  exists: boolean;
  config: ParsedConfig;
}

export interface DockerHealth {
  installed: boolean;
  running: boolean;
  image_present: boolean;
  image: string;
  ready: boolean;
  notes: string[];
}
