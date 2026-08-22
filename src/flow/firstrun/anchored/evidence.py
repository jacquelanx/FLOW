"""Content-addressed canonical first-run bundle."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


BUNDLE_SCHEMA = "flow.first_run_bundle.v1"


class BundleWriteConflict(RuntimeError):
    """Raised rather than replacing an existing, non-identical canonical manifest."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entry(path: Path, root: Path, *, prefix: str) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    entry = {
        "artifact_id": f"{prefix}:{rel}",
        "relative_path": rel,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }
    entry["content_hash_role"] = "deterministic_canonical"
    if path.suffix.lower() == ".json":
        try:
            entry["content_sha256"] = sha256_json(
                _without_volatile_fields(json.loads(path.read_text()))
            )
        except (OSError, ValueError):
            entry["content_sha256"] = entry["sha256"]
    else:
        entry["content_sha256"] = entry["sha256"]
    return entry


def _without_volatile_fields(value: Any) -> Any:
    volatile_keys = {"created_utc", "generated_at", "generated_utc", "timestamp", "run_at"}
    if isinstance(value, dict):
        return {
            key: _without_volatile_fields(item) for key, item in value.items()
            if str(key).lower() not in volatile_keys
        }
    if isinstance(value, list):
        return [_without_volatile_fields(item) for item in value]
    return value


def _content_sha256(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            return sha256_json(_without_volatile_fields(json.loads(path.read_text())))
        except (OSError, ValueError):
            pass
    return sha256_file(path)


def _files(root: Path, *, exclude: set[str] | None = None) -> list[Path]:
    excluded = exclude or set()
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )


def build_first_run_bundle(
    *,
    out_dir: str | Path,
    input_files: Iterable[str | Path],
    source_files: Iterable[str | Path],
    config_files: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Write a deterministic manifest whose content hash excludes volatile timestamps."""
    out_root = Path(out_dir)
    excluded = {"first_run_bundle.json"}
    artifacts = [_entry(path, out_root, prefix="artifact")
                 for path in _files(out_root, exclude=excluded)]

    def external_entries(paths: Iterable[str | Path], prefix: str) -> list[dict[str, Any]]:
        entries = []
        existing = [Path(path) for path in paths if Path(path).is_file()]
        ordered = sorted(existing, key=lambda p: (p.name.lower(), sha256_file(p)))
        for index, raw in enumerate(ordered):
            if not raw.is_file():
                continue
            entries.append({
                "artifact_id": f"{prefix}:{index:04d}:{raw.name}",
                "name": raw.name,
                "sha256": sha256_file(raw),
                "content_sha256": _content_sha256(raw),
                "size_bytes": raw.stat().st_size,
            })
        return entries

    distributions = {}
    for distribution in ("flow-agent", "numpy", "pandas", "matplotlib", "flowkit",
                         "scipy", "scikit-learn", "pydantic"):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = "NOT_INSTALLED"
    container_digest = os.environ.get("FLOW_CONTAINER_DIGEST")
    execution_mode = "container" if container_digest else "local_python_validation"
    content = {
        "schema_version": BUNDLE_SCHEMA,
        "deterministic_seed": 0,
        "canonical_event_exclusion_applied": False,
        "inputs": external_entries(input_files, "input"),
        "sources": external_entries(source_files, "source"),
        "configuration": external_entries(config_files, "config"),
        "runtime": {
            "execution_mode": execution_mode,
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "container_image_digest": container_digest or "NOT_APPLICABLE_LOCAL_VALIDATION",
            "dependencies": distributions,
        },
        "artifacts": artifacts,
    }
    stable_content = copy.deepcopy(content)
    for section in ("inputs", "sources", "configuration"):
        stable_content[section] = [
            {"artifact_id": item["artifact_id"], "name": item["name"],
             "content_sha256": item["content_sha256"]}
            for item in content[section]
        ]
    stable_content["artifacts"] = [
        {
            "artifact_id": item["artifact_id"],
            "relative_path": item["relative_path"],
            "content_hash_role": item["content_hash_role"],
            "content_sha256": item["content_sha256"],
        }
        for item in artifacts
    ]
    content_hash = sha256_json(stable_content)
    bundle = content | {
        "first_run_bundle_sha256": content_hash,
        "content_hash_policy": (
            "hash inputs/source/config/runtime and every canonical artifact including "
            "PDF/HTML/page renders; normalize only declared volatile JSON timestamp fields"
        ),
        "write_policy": "atomic write-once; conflicting existing manifest is fatal",
    }
    target = out_root / "first_run_bundle.json"
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, ValueError) as exc:
            raise BundleWriteConflict(f"existing bundle is unreadable: {target}") from exc
        if existing == bundle:
            return existing
        raise BundleWriteConflict(
            f"refusing to replace non-identical canonical bundle: {target}"
        )
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    temporary.replace(target)
    return bundle
