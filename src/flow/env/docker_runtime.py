"""Docker runtime helpers: command construction, image checks, and the doctor.

This module centralises the **safety policy** for how agent code is sandboxed. Every
container FLOW launches is:
  * read-only dataset mount (``:ro``) + one writable workdir,
  * ``--network none`` by default (dependency installs require explicit opt-in),
  * resource-limited (``--memory``, ``--cpus``, ``--pids-limit``),
  * non-root, ``--cap-drop=ALL``, read-only rootfs with a tmpfs workdir.

LLM API keys never enter the container — provider calls happen host-side.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_IMAGE = "flow-bixbench-env:1.0"


@dataclass
class DockerLimits:
    """Per-container resource and isolation limits."""

    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256
    network: str = "none"  # "none" (default, safe) or "bridge" (opt-in installs)
    read_only_rootfs: bool = True
    workdir_tmpfs_size: str = "1g"
    user: str = "1000:1000"


class DockerUnavailableError(RuntimeError):
    """Raised when Docker or the FLOW image is not available.

    Carries an actionable message — FLOW refuses to run agent code on the host.
    """


def docker_available() -> bool:
    """True if the ``docker`` CLI exists and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    """True if the named image is present locally."""
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=15
        )
        return r.returncode == 0
    except Exception:
        return False


def image_digest(image: str = DEFAULT_IMAGE) -> str | None:
    """Return the image's content digest (for the audit trail), or None."""
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        return None
    return None


def require_docker(image: str = DEFAULT_IMAGE) -> None:
    """Hard-fail with guidance if Docker or the image is missing."""
    if not docker_available():
        raise DockerUnavailableError(
            "Docker is not available. FLOW executes ALL agent code inside Docker and "
            "refuses to run it on the host.\n"
            "  • Install/start Docker Desktop (or the Docker daemon), then re-run.\n"
            "  • Verify with: flow doctor"
        )
    if not image_exists(image):
        raise DockerUnavailableError(
            f"Docker image '{image}' not found. Build it first:\n"
            f"  docker build -t {image} -f docker/Dockerfile .\n"
            "Then verify with: flow doctor"
        )


def build_run_command(
    *,
    image: str,
    container_name: str,
    data_dir: Path,
    work_dir: Path,
    kernel_server_path: Path,
    limits: DockerLimits,
) -> list[str]:
    """Construct the ``docker run`` argv for a persistent kernel container.

    The container runs the kernel server on stdin/stdout (``-i``), so the host keeps a
    long-lived process handle and notebook state persists for the whole trajectory.
    """
    cmd: list[str] = [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        container_name,
        "--network",
        limits.network,
        "--memory",
        limits.memory,
        "--cpus",
        limits.cpus,
        "--pids-limit",
        str(limits.pids_limit),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        limits.user,
        # Read-only dataset mount — agent can never mutate the source data.
        "-v",
        f"{data_dir}:/data:ro",
        # One writable working directory for the notebook + artifacts.
        "-v",
        f"{work_dir}:/work:rw",
        # The kernel server, mounted read-only (also baked into the image as a fallback).
        "-v",
        f"{kernel_server_path}:/flow/kernel_server.py:ro",
        "-w",
        "/work",
        "-e",
        "FLOW_DATA_DIR=/data",
        "-e",
        "FLOW_WORK_DIR=/work",
        "-e",
        "FLOW_PLOTS_DIR=/work/plots",
        "-e",
        "MPLBACKEND=Agg",
        # matplotlib needs a writable config/cache dir; the rootfs is read-only, so point
        # it at the tmpfs. Avoids the "/home/flow/.config: Read-only file system" warning.
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        # Likewise give other tools a writable HOME on the tmpfs.
        "-e",
        "HOME=/tmp",
    ]
    if limits.read_only_rootfs:
        cmd += [
            "--read-only",
            "--tmpfs",
            f"/tmp:size={limits.workdir_tmpfs_size}",
        ]
    cmd += [image, "python", "/flow/kernel_server.py"]
    return cmd


@dataclass
class DoctorReport:
    """Result of ``flow doctor`` — environment readiness for safe execution."""

    docker_installed: bool
    docker_running: bool
    image_present: bool
    image: str
    image_digest: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.docker_installed and self.docker_running and self.image_present


def doctor(image: str = DEFAULT_IMAGE) -> DoctorReport:
    """Diagnose whether FLOW can run agent code safely."""
    installed = shutil.which("docker") is not None
    running = docker_available()
    present = image_exists(image) if running else False
    notes: list[str] = []
    if not installed:
        notes.append("Docker CLI not found on PATH. Install Docker Desktop or engine.")
    elif not running:
        notes.append("Docker daemon is not responding. Start Docker, then re-run.")
    elif not present:
        notes.append(
            f"Image '{image}' missing. Build: docker build -t {image} -f docker/Dockerfile ."
        )
    else:
        notes.append("All checks passed — FLOW can execute agent code safely in Docker.")
    return DoctorReport(
        docker_installed=installed,
        docker_running=running,
        image_present=present,
        image=image,
        image_digest=image_digest(image) if present else None,
        notes=notes,
    )
