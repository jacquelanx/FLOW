"""FLOW command-line interface (stdlib argparse — no extra deps).

Commands:
  flow doctor                         Check Docker + the BixBench-env image.
  flow run --data <dir> --question .. Run one trajectory inside Docker.
  flow serve [--port 8000]            Start the FastAPI backend (uvicorn).

``flow run`` builds a config from config.yaml in the data dir (if present), then applies
command-line overrides. It always executes inside Docker; if Docker is missing it fails
with actionable guidance (never runs on the host).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path


def _cmd_doctor(args: argparse.Namespace) -> int:
    from flow.env.docker_runtime import doctor

    rep = doctor(args.image)
    print("FLOW doctor")
    print("-----------")
    print(f"  docker installed : {rep.docker_installed}")
    print(f"  docker running   : {rep.docker_running}")
    print(f"  image '{rep.image}' present : {rep.image_present}")
    if rep.image_digest:
        print(f"  image digest     : {rep.image_digest}")
    for note in rep.notes:
        print(f"  • {note}")
    return 0 if rep.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from flow.config import ConfigError, FlowConfig, load_config

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        return 2

    cfg_path = data_dir / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = load_config(cfg_path)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        if not args.question:
            print("error: no config.yaml found and --question not provided.", file=sys.stderr)
            return 2
        cfg = FlowConfig(question=args.question)

    # CLI overrides.
    if args.question:
        cfg.question = args.question
    if args.provider:
        cfg.runtime.provider = args.provider
    if args.model:
        cfg.runtime.model = args.model
    if args.max_steps:
        cfg.runtime.max_steps = args.max_steps

    run_id = uuid.uuid4().hex[:12]
    run_dir = Path(args.out).resolve() / run_id if args.out else data_dir / "runs" / run_id

    print(f"FLOW run {run_id}")
    print(f"  question : {cfg.question}")
    print(f"  provider : {cfg.runtime.provider} / {cfg.runtime.model}")
    print(f"  data     : {data_dir}")
    print(f"  out      : {run_dir}")
    print("  executing inside Docker (BixBench-env)...\n")

    from flow.env.docker_runtime import DockerUnavailableError
    from flow.runner import run_analysis

    def _on_step(rec) -> None:
        preview = (rec.observation or "").strip().splitlines()
        head = preview[0] if preview else ""
        print(f"  step {rec.step:>2} [{rec.tool}] {head[:100]}")

    try:
        result, meta = run_analysis(
            data_dir=data_dir, config=cfg, run_dir=run_dir, run_id=run_id, image=args.image,
            on_step=_on_step,
        )
    except DockerUnavailableError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 3

    print()
    if result.submitted:
        print("=== ANSWER ===")
        print(result.answer)
        print(f"\nArtifacts: {run_dir}")
        return 0
    print(f"RUN FAILED: {result.failure_reason}", file=sys.stderr)
    print(f"Artifacts (partial): {run_dir}", file=sys.stderr)
    return 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("flow.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flow", description="FLOW — the Finch-faithful data-analysis agent.")
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("doctor", help="Check Docker + the BixBench-env image.")
    pd.add_argument("--image", default="flow-bixbench-env:1.0")
    pd.set_defaults(func=_cmd_doctor)

    pr = sub.add_parser("run", help="Run one trajectory inside Docker.")
    pr.add_argument("--data", required=True, help="Project/data directory.")
    pr.add_argument("--question", help="Research question (overrides config.yaml).")
    pr.add_argument("--provider", help="mock | ollama | gemini | groq | openrouter | deepseek | openai")
    pr.add_argument("--model", help="Model id (e.g. gemini-2.0-flash, qwen2.5-coder).")
    pr.add_argument("--max-steps", type=int, dest="max_steps")
    pr.add_argument("--out", help="Output root for artifacts (default: <data>/runs).")
    pr.add_argument("--image", default="flow-bixbench-env:1.0")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("serve", help="Start the FastAPI backend.")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--reload", action="store_true")
    ps.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
