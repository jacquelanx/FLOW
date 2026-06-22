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

// A "run" in the UI is a consensus batch: N independent trajectories + a meta-analysis.
export interface Batch {
  id: string;
  project_id: string;
  question: string;
  provider: string;
  model: string;
  n_trajectories: number;
  status: string;
  created_at: number;
  finished_at: number | null;
  consensus_ok: number;
  n_submitted: number;
  failure_reason: string | null;
  artifact_dir: string;
}

export interface TrajectorySummary {
  idx: number;
  status: string;
  steps: number;
  submitted: boolean;
  failure_reason: string | null;
  answer: string;
}

export interface Consensus {
  consensus: string | null;
  synthesized: boolean | null;
  n_submitted: number | null;
  n_total: number | null;
  failure_reason: string | null;
}

export interface StepRecord {
  step: number;
  tool: string;
  arguments: Record<string, unknown>;
  observation: string;
  done?: boolean;
  reward?: number;
}

export interface BatchStatus {
  run: Batch;
  status: string;
  phase: string | null;
  trajectories: TrajectorySummary[];
  consensus: Consensus;
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
  meta_provider: string;
  meta_model: string;
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
