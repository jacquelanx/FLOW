"""Safety tests: Docker command policy, traversal refusal, Docker-absent behavior."""

from pathlib import Path

import pytest

from flow.env.docker_runtime import (
    DockerLimits,
    DockerUnavailableError,
    build_run_command,
    require_docker,
)
from flow.validation import UnsafePathError, safe_filename


def _cmd(tmp_path, **limit_kw):
    data = tmp_path / "data"
    work = tmp_path / "work"
    data.mkdir()
    work.mkdir()
    ks = tmp_path / "kernel_server.py"
    ks.write_text("# kernel")
    return build_run_command(
        image="flow-bixbench-env:1.0",
        container_name="flow-test",
        data_dir=data,
        work_dir=work,
        kernel_server_path=ks,
        limits=DockerLimits(**limit_kw),
    )


def test_dataset_mounted_read_only(tmp_path):
    cmd = _cmd(tmp_path)
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    data_mounts = [m for m in mounts if m.endswith(":/data:ro")]
    assert len(data_mounts) == 1, "dataset must be mounted exactly once, read-only"
    # And there is no writable data mount.
    assert not any(m.endswith(":/data:rw") for m in mounts)


def test_default_network_is_none(tmp_path):
    cmd = _cmd(tmp_path)
    idx = cmd.index("--network")
    assert cmd[idx + 1] == "none"


def test_network_opt_in(tmp_path):
    cmd = _cmd(tmp_path, network="bridge")
    idx = cmd.index("--network")
    assert cmd[idx + 1] == "bridge"


def test_resource_limit_flags_present(tmp_path):
    cmd = _cmd(tmp_path)
    for flag in ["--memory", "--cpus", "--pids-limit", "--cap-drop", "--user"]:
        assert flag in cmd, f"missing safety flag {flag}"
    assert "no-new-privileges" in cmd
    assert "--read-only" in cmd
    # matplotlib/tooling get a writable config dir on the tmpfs (read-only rootfs).
    assert "MPLCONFIGDIR=/tmp/matplotlib" in cmd


def test_path_traversal_refused():
    for bad in ["../etc/passwd", "/etc/passwd", "a/b.csv", "..", "foo/../bar", "x\\y"]:
        with pytest.raises(UnsafePathError):
            safe_filename(bad)


def test_safe_filename_accepts_normal_names():
    assert safe_filename("metadata.json") == "metadata.json"
    assert safe_filename("events.csv") == "events.csv"


def test_require_docker_raises_when_absent(monkeypatch):
    import flow.env.docker_runtime as dr

    monkeypatch.setattr(dr, "docker_available", lambda: False)
    with pytest.raises(DockerUnavailableError) as ei:
        require_docker()
    assert "Docker is not available" in str(ei.value)


def test_require_docker_raises_when_image_missing(monkeypatch):
    import flow.env.docker_runtime as dr

    monkeypatch.setattr(dr, "docker_available", lambda: True)
    monkeypatch.setattr(dr, "image_exists", lambda image=dr.DEFAULT_IMAGE: False)
    with pytest.raises(DockerUnavailableError) as ei:
        require_docker()
    assert "not found" in str(ei.value)
