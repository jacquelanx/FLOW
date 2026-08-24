"""FLOW command-line interface (stdlib argparse — no extra deps).

Commands:
  flow doctor                         Check Docker + the BixBench-env image.
  flow run --data <dir> --question .. Run one trajectory inside Docker.
  flow serve [--port 8000]            Start FLOW (web app + API on one port).

``flow run`` builds a config from config.yaml in the data dir (if present), then applies
command-line overrides. It always executes inside Docker; if Docker is missing it fails
with actionable guidance (never runs on the host).
"""

from __future__ import annotations

import argparse
import sys
import time
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
    if args.meta_provider:
        cfg.runtime.meta_provider = args.meta_provider
    if args.meta_model:
        cfg.runtime.meta_model = args.meta_model
    if args.max_steps:
        cfg.runtime.max_steps = args.max_steps

    n = max(1, int(args.trajectories or 1))
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = Path(args.out).resolve() / batch_id if args.out else data_dir / "runs" / batch_id

    print(f"FLOW run {batch_id}")
    print(f"  question     : {cfg.question}")
    print(f"  provider     : {cfg.runtime.provider} / {cfg.runtime.model}")
    meta_p = cfg.runtime.meta_provider or cfg.runtime.provider
    meta_m = cfg.runtime.meta_model or cfg.runtime.model
    print(f"  consensus    : {meta_p} / {meta_m}")
    print(f"  trajectories : {n}" + (" (consensus meta-analysis)" if n > 1 else ""))
    print(f"  data         : {data_dir}")
    print(f"  out          : {batch_dir}")
    print("  executing inside Docker (BixBench-env)...\n")

    from flow.env.docker_runtime import DockerUnavailableError
    from flow.runner import run_batch

    def _on_traj_status(idx: int, status: str, info: dict) -> None:
        if status == "running":
            print(f"  ── trajectory {idx} started")
        else:
            tag = "submitted" if info.get("submitted") else f"no answer ({status})"
            print(f"  ── trajectory {idx} finished: {tag}")

    def _on_step(idx: int, rec) -> None:
        head = ((rec.observation or "").strip().splitlines() or [""])[0]
        print(f"     t{idx} step {rec.step:>2} [{rec.tool}] {head[:80]}")

    def _on_phase(phase: str) -> None:
        if phase == "synthesizing":
            print("\n  synthesizing consensus from all trajectories...\n")

    try:
        bres = run_batch(
            data_dir=data_dir, config=cfg, batch_dir=batch_dir, n_trajectories=n,
            batch_id=batch_id, image=args.image,
            on_trajectory_status=_on_traj_status, on_step=_on_step, on_phase=_on_phase,
        )
    except DockerUnavailableError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 3

    cons = bres.consensus or {}
    n_sub = sum(1 for t in bres.trajectories if t["submitted"])
    print(f"\nTrajectories: {n_sub}/{n} produced an answer.")
    if cons.get("consensus"):
        label = "CONSENSUS" if cons.get("synthesized") else "CONSENSUS (single trajectory)"
        print(f"\n=== {label} ===")
        print(cons["consensus"])
        print(f"\nArtifacts: {batch_dir}")
        return 0
    print(f"RUN FAILED: {cons.get('failure_reason')}", file=sys.stderr)
    print(f"Artifacts (partial): {batch_dir}", file=sys.stderr)
    return 1


def _server_listening(host: str, port: int) -> bool:
    """True if something accepts TCP connections on host:port."""
    import socket

    target = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((target, port)) == 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import threading
    import webbrowser

    import uvicorn

    from flow.api.webui import web_dir

    url = f"http://localhost:{args.port}"

    # Fail before uvicorn does, so the message names the likely cause rather than
    # printing a bind traceback.
    if _server_listening(args.host, args.port):
        print(f"Port {args.port} is already in use.", file=sys.stderr)
        print(f"  FLOW may already be running — open {url} in your browser.", file=sys.stderr)
        print(f"  Otherwise start on a different port:  flow serve --port {args.port + 10}",
              file=sys.stderr)
        return 1

    built = web_dir()
    # Flushed: uvicorn logs to stderr, and a block-buffered stdout would print
    # this banner after the server's own startup lines.
    print("FLOW is starting.", flush=True)
    if built is None:
        print("  Web app  : NOT BUILT — run 'npm install && npm run build' in frontend/",
              flush=True)
        print(f"  API      : {url}/docs", flush=True)
    else:
        print(f"  Open in your browser: {url}", flush=True)
    print("  Press Ctrl+C to stop.\n", flush=True)

    # Open the browser once the server is accepting connections. --reload spawns a
    # reloader process, so only the child that actually serves should do this.
    if built is not None and args.browser and not args.reload:
        def _open() -> None:
            for _ in range(100):  # up to ~10s
                if _server_listening(args.host, args.port):
                    webbrowser.open(url)
                    return
                time.sleep(0.1)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run("flow.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flow", description="FLOW — the Finch-faithful data-analysis agent.")
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("doctor", help="Check Docker + the BixBench-env image.")
    pd.add_argument("--image", default="flow-bixbench-env:1.0")
    pd.set_defaults(func=_cmd_doctor)

    pr = sub.add_parser(
        "run",
        help="Run N independent trajectories inside Docker + a consensus meta-analysis.",
    )
    pr.add_argument("--data", required=True, help="Project/data directory.")
    pr.add_argument("--question", help="Research question (overrides config.yaml).")
    pr.add_argument("--provider", help="mock | ollama | gemini | groq | openrouter | deepseek | openai")
    pr.add_argument("--model", help="Model id (e.g. gemini-3.6-flash, qwen2.5-coder).")
    pr.add_argument("--meta-provider", dest="meta_provider",
                    help="Provider for the consensus synthesis (default: same as --provider).")
    pr.add_argument("--meta-model", dest="meta_model",
                    help="Model for the consensus synthesis (default: same as --model).")
    pr.add_argument("--max-steps", type=int, dest="max_steps")
    pr.add_argument(
        "--trajectories", type=int, default=1,
        help="Number of independent trajectories to run, then synthesize (default: 1).",
    )
    pr.add_argument("--out", help="Output root for artifacts (default: <data>/runs).")
    pr.add_argument("--image", default="flow-bixbench-env:1.0")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("serve", help="Start FLOW: the web app and the API on one port.")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--reload", action="store_true")
    ps.add_argument("--no-browser", dest="browser", action="store_false",
                    help="Don't open a browser window on start.")
    ps.set_defaults(func=_cmd_serve, browser=True)

    return p


def main(argv: list[str] | None = None) -> int:
    # Load a host-side .env (if present) so provider API keys are available without
    # exporting them every session. Existing environment variables are never overridden.
    from flow.envfile import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
