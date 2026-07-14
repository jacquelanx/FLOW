"""Kernel abstractions: a persistent execution backend for the NotebookEnvironment.

Two implementations share **one** execution code path (``kernel_server.execute_code``):

  * ``DockerKernel`` — the real, safe backend. Launches a long-lived container and
    talks to the kernel server over stdin/stdout. This is what production runs use.
  * ``InProcessKernel`` — a host-side backend used **only by tests** to drive the full
    ReAct loop without Docker. It is never selected by ``flow.runner`` for real runs,
    preserving the "all agent code runs in Docker" guarantee.

Both return the same ``CellResult`` shape so the environment is backend-agnostic.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from flow.env import kernel_server
from flow.env.docker_runtime import (
    DEFAULT_IMAGE,
    DockerLimits,
    build_run_command,
    image_digest,
    require_docker,
)


@dataclass
class CellResult:
    """Structured result of executing one cell (JSON-safe)."""

    stdout: str = ""
    stderr: str = ""
    result: str = ""
    images: list[str] = field(default_factory=list)
    error: bool = False
    traceback: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CellResult":
        return cls(
            stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""),
            result=d.get("result", ""),
            images=list(d.get("images", [])),
            error=bool(d.get("error", False)),
            traceback=d.get("traceback", ""),
        )


class Kernel(Protocol):
    """Minimal persistent-kernel interface."""

    def start(self) -> None: ...

    def execute(self, code: str, timeout: float) -> CellResult: ...

    def shutdown(self) -> None: ...

    @property
    def image_digest(self) -> str | None: ...


class KernelTimeout(RuntimeError):
    """Raised when a single cell exceeds its wall-clock budget."""


class InProcessKernel:
    """Host-side kernel for tests only — NOT used for real agent runs.

    Runs the identical ``execute_code`` used inside the container, against a persistent
    namespace, so notebook state semantics match. No isolation: tests only.
    """

    def __init__(self, plots_dir: str | Path):
        self.plots_dir = str(plots_dir)
        self._ns: dict[str, Any] = {}
        self._digest = "inprocess:test-kernel"

    def start(self) -> None:
        self._ns = {"__name__": "__flow_cell__"}

    def execute(self, code: str, timeout: float = 60.0) -> CellResult:
        # timeout is accepted for interface parity; in-process execution is synchronous.
        out = kernel_server.execute_code(self._ns, code, self.plots_dir)
        return CellResult.from_dict(out)

    def shutdown(self) -> None:
        self._ns = {}

    @property
    def image_digest(self) -> str | None:
        return self._digest


class DockerKernel:
    """Persistent kernel running inside a hardened Docker container.

    A single ``docker run -i`` process hosts the kernel server for the whole trajectory.
    Cells are sent as JSON lines; results are read back up to the sentinel. Per-cell
    timeouts are enforced host-side; on timeout the container is force-killed.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        work_dir: Path,
        container_name: str,
        image: str = DEFAULT_IMAGE,
        limits: DockerLimits | None = None,
        kernel_server_path: Path | None = None,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.container_name = container_name
        self.image = image
        self.limits = limits or DockerLimits()
        self.kernel_server_path = (
            Path(kernel_server_path)
            if kernel_server_path
            else Path(kernel_server.__file__).resolve()
        )
        self._proc: subprocess.Popen[str] | None = None
        self._digest: str | None = None

    def start(self) -> None:
        require_docker(self.image)
        self._digest = image_digest(self.image)
        (self.work_dir / "plots").mkdir(parents=True, exist_ok=True)
        # Mount the first-run scripts dir so the seed cell can invoke them (also baked in).
        try:
            from flow.firstrun import FIRSTRUN_DIR

            first_run_dir = FIRSTRUN_DIR if FIRSTRUN_DIR.exists() else None
        except Exception:
            first_run_dir = None
        cmd = build_run_command(
            image=self.image,
            container_name=self.container_name,
            data_dir=self.data_dir,
            work_dir=self.work_dir,
            kernel_server_path=self.kernel_server_path,
            limits=self.limits,
            first_run_dir=first_run_dir,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._wait_for_ready(timeout=120)

    def _wait_for_ready(self, timeout: float) -> None:
        assert self._proc and self._proc.stdout
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read() if self._proc.stderr else ""
                    raise RuntimeError(f"Kernel container exited at startup: {err}")
                continue
            if line.strip() == "FLOW_KERNEL_READY":
                return
        raise KernelTimeout("Kernel container did not become ready in time.")

    def execute(self, code: str, timeout: float = 120.0) -> CellResult:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Kernel is not running.")
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps({"code": code}) + "\n")
        self._proc.stdin.flush()

        result_box: dict[str, Any] = {}

        def _read() -> None:
            assert self._proc and self._proc.stdout
            payload_line = None
            for line in self._proc.stdout:
                stripped = line.rstrip("\n")
                if stripped == kernel_server.SENTINEL:
                    break
                payload_line = stripped
            if payload_line is not None:
                try:
                    result_box["data"] = json.loads(payload_line)
                except json.JSONDecodeError:
                    result_box["data"] = {
                        "error": True,
                        "traceback": f"Malformed kernel output: {payload_line!r}",
                    }

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            self.shutdown(force=True)
            raise KernelTimeout(
                f"Cell exceeded the per-cell timeout of {timeout:.0f}s and was killed."
            )
        if "data" not in result_box:
            raise RuntimeError("Kernel did not return a result.")
        return CellResult.from_dict(result_box["data"])

    def shutdown(self, force: bool = False) -> None:
        if self._proc is None:
            return
        try:
            if force:
                subprocess.run(
                    ["docker", "kill", self.container_name],
                    capture_output=True,
                    timeout=15,
                )
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        finally:
            self._proc = None

    @property
    def image_digest(self) -> str | None:
        return self._digest
