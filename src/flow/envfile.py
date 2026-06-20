"""Minimal .env loader (standard library only).

Loads ``KEY=VALUE`` pairs from a host-side ``.env`` file into the process environment so
provider API keys (e.g. GEMINI_API_KEY, GROQ_API_KEY) can live in one gitignored file
instead of being exported every session.

Design choices:
  * **Never overrides** variables already set in the real environment — an explicit
    ``export FOO=...`` or a value passed by the shell always wins over the file.
  * Host-side only. These keys are used for provider calls on the host and are **never**
    passed into the Docker container (the privacy boundary is unchanged).
  * No third-party dependency; this is a deliberately tiny parser.

Lookup order for the file: the ``FLOW_ENV_FILE`` env var if set, else ``.env`` in the
current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] | None = None) -> bool:
    """Load ``.env`` into ``os.environ`` (without overriding existing vars).

    Returns True if a file was found and read, False otherwise.
    """
    p = Path(path or os.environ.get("FLOW_ENV_FILE", ".env"))
    if not p.exists() or not p.is_file():
        return False

    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip a single layer of matching quotes.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        # Real environment wins; the file only fills gaps.
        if key and key not in os.environ:
            os.environ[key] = val
    return True
