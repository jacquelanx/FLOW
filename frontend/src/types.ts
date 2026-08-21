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
  // Null when the project's first-run left no obligations to check — which is NOT the same
  // as zero undischarged, so the UI must not render null as "complete".
  obligations_total: number | null;
  obligations_undischarged: number | null;
}

export interface Consensus {
  consensus: string | null;
  synthesized: boolean | null;
  n_submitted: number | null;
  n_total: number | null;
  failure_reason: string | null;
  n_with_unread_evidence: number | null;
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
  first_run: string;
  max_steps: number;
  per_cell_timeout: number;
  per_trajectory_timeout: number;
}

export interface SafetyCfg {
  memory: string;
  cpus: string;
  pids_limit: number;
  allow_network: boolean;
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

// Biologist-owned analysis specification (compiled to agent guidance).
export interface MarkerDef {
  detector: string;
  marker: string;
  notes: string;
}

export interface GateStep {
  name: string;
  definition: string;
}

export interface AnalysisProfile {
  panel_name: string;
  markers: MarkerDef[];
  gating: GateStep[];
  populations_of_interest: string[];
  comparisons: string[];
  qc_expectations: string;
  desired_outputs: string[];
  biological_context: string;
  plain_language: boolean;
  extra_instructions: string;
}

export interface AnalysisPayload {
  profile: AnalysisProfile;
  guidance_preview: string;
  seeded: boolean;
}
