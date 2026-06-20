"""Docker integration test — auto-skipped when Docker or the image is absent.

Exercises the real DockerKernel: persistent state across cells executed INSIDE the
container. This is the only test that touches Docker; everything else uses InProcessKernel.
"""

import shutil
import uuid
from pathlib import Path

import pytest

from flow.env.docker_runtime import DEFAULT_IMAGE, docker_available, image_exists

pytestmark = pytest.mark.skipif(
    not (docker_available() and image_exists(DEFAULT_IMAGE)),
    reason="Docker or flow-bixbench-env:1.0 image not available",
)


@pytest.fixture
def shared_tmp():
    """A temp dir under the repo (./var) so it lives on a Docker-shareable path.

    Docker Desktop on macOS does not share ``/var/folders`` (pytest's default tmp) by
    default, which would make bind mounts appear empty; project dirs under the user's
    home are shared. Using a repo-local dir keeps this test portable.
    """
    base = Path(__file__).resolve().parent.parent / "var" / "_ittest"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex[:8]
    d.mkdir()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_docker_kernel_persistent_state(shared_tmp: Path):
    from flow.env.kernel import DockerKernel

    data = shared_tmp / "data"
    work = shared_tmp / "work"
    data.mkdir()
    work.mkdir()
    (data / "x.txt").write_text("hello")

    kernel = DockerKernel(
        data_dir=data, work_dir=work, container_name="flow-itest"
    )
    kernel.start()
    try:
        r1 = kernel.execute("y = 41")
        assert not r1.error
        r2 = kernel.execute("print(y + 1)")
        assert "42" in r2.stdout
        # Read-only dataset is visible in the container.
        r3 = kernel.execute(
            "import os; print(open(os.path.join(os.environ['FLOW_DATA_DIR'],'x.txt')).read())"
        )
        assert "hello" in r3.stdout
    finally:
        kernel.shutdown()


def test_docker_dataset_is_read_only(shared_tmp: Path):
    from flow.env.kernel import DockerKernel

    data = shared_tmp / "data"
    work = shared_tmp / "work"
    data.mkdir()
    work.mkdir()
    (data / "x.txt").write_text("hello")

    kernel = DockerKernel(data_dir=data, work_dir=work, container_name="flow-itest-ro")
    kernel.start()
    try:
        r = kernel.execute(
            "import os\n"
            "try:\n"
            "    open(os.path.join(os.environ['FLOW_DATA_DIR'],'x.txt'),'w').write('nope')\n"
            "    print('WROTE')\n"
            "except Exception as e:\n"
            "    print('READONLY', type(e).__name__)\n"
        )
        assert "READONLY" in r.stdout
    finally:
        kernel.shutdown()
