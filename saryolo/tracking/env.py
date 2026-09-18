"""Environment capture for reproducibility.

The project brief requires every experiment row to record the software and
hardware it ran on. Values that cannot be determined are recorded as ``None``
rather than guessed, so a gap is visible instead of silently wrong.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

__all__ = ["capture_environment", "git_commit", "hash_file", "hash_config", "describe_hardware"]


def _run(cmd: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=15, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit(cwd: str | Path | None = None) -> str | None:
    """Current git commit hash, or ``None`` outside a repository."""
    return _run(["git", "rev-parse", "HEAD"], cwd=cwd)


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """SHA-256 of a file, used to fingerprint dataset manifests and checkpoints."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_config(cfg: dict) -> str:
    """Stable hash of a configuration dict (used to detect config drift)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def describe_hardware() -> dict:
    """GPU/CPU description, if discoverable."""
    info: dict = {"cpu": platform.processor() or platform.machine()}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["gpu_memory_gb"] = (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else None
        )
    except Exception:  # torch missing or a driver problem: record what we know
        info.setdefault("cuda_available", None)
    return info


def capture_environment(cwd: str | Path | None = None) -> dict:
    """Collect the full reproducibility fingerprint for an experiment row."""
    env: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(cwd),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except Exception:
        env["torch"] = None
        env["cuda"] = None
    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except Exception:
        env["ultralytics"] = None
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    env.update(describe_hardware())
    return env
