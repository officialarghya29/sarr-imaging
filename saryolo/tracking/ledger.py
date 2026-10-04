"""Experiment ledger: an append-only record of every run.

Rationale
---------
The paper's tables must be traceable to runs. Each row here pins the experiment
id, configuration, seed, environment and dataset fingerprint, plus the metrics.
The ledger is append-only (JSONL) so a rerun never silently overwrites an
earlier result; CSV is derived for convenience.

Crucially, the ledger only ever records *measured* values. Anything not measured
is stored as ``None``, and downstream table generation renders it as
``TBD`` — there is no code path in this repository that invents a number.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .env import capture_environment, hash_config

__all__ = ["ExperimentRecord", "ExperimentLedger"]


try:  # POSIX advisory locking; the repo's runs are on Linux, Windows falls back below.
    import fcntl

except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]


@contextlib.contextmanager
def _exclusive(path: Path):
    """Serialise writers on a lock file, so concurrent runs cannot corrupt the ledger.

    Training jobs are launched in parallel, and before this lock existed two processes
    appending at the same time raced on the *shared* CSV temp path: one would
    ``os.replace`` it away while the other still expected it, raising
    ``FileNotFoundError`` and killing an otherwise healthy run. A unique temp name per
    process fixes the replace race; this lock also keeps the read-modify-write of the
    derived CSV from interleaving. On platforms without ``fcntl`` it degrades to a
    no-op, because a ledger write must never be the thing that stops a run.
    """
    if fcntl is None:  # pragma: no cover - Windows only
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

METRIC_KEYS = (
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "f1",
    "AP_small",
    "AP_medium",
    "AP_large",
    "params_M",
    "flops_G",
    "fps",
    "latency_ms",
    "gpu_mem_train_GB",
    "model_size_MB",
    "train_minutes",
)


@dataclass
class ExperimentRecord:
    """One experiment run."""

    experiment_id: str
    model: str
    dataset: str
    split_seed: int = 0
    train_seed: int = 0
    status: str = "running"  # running | completed | failed
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_hash: str | None = None
    epochs: int | None = None
    batch: int | None = None
    imgsz: int | None = None
    optimizer: str | None = None
    lr0: float | None = None
    weights: str | None = None
    save_dir: str | None = None
    notes: str = ""
    metrics: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def flatten(self) -> dict:
        """Flat row for CSV export; missing metrics become empty strings."""
        row = {k: v for k, v in asdict(self).items() if k not in ("metrics", "environment", "extra")}
        for key in METRIC_KEYS:
            value = self.metrics.get(key)
            row[key] = "" if value is None else value
        row["gpu"] = self.environment.get("gpu_name") or ""
        row["git_commit"] = self.environment.get("git_commit") or ""
        row["torch"] = self.environment.get("torch") or ""
        row["ultralytics"] = self.environment.get("ultralytics") or ""
        row["environment"] = json.dumps(self.environment, default=str)
        row["extra"] = json.dumps(self.extra, default=str)
        return row


class ExperimentLedger:
    """Append-only experiment log backed by JSONL, with a derived CSV view."""

    def __init__(self, root: str | Path = "results"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "experiments.jsonl"
        self.csv_path = self.root / "experiments.csv"
        self.lock_path = self.root / ".experiments.lock"

    # ------------------------------------------------------------------ writing
    def append(self, record: ExperimentRecord) -> ExperimentRecord:
        """Append a record (JSONL) and refresh the CSV view.

        The whole append is serialised across processes (see :func:`_exclusive`), and the
        JSONL line is written in a single call so a reader never sees a half-written
        record even if the lock is unavailable on this platform.
        """
        line = json.dumps(asdict(record), default=str) + "\n"
        with _exclusive(self.lock_path):
            with self.jsonl_path.open("a") as fh:
                fh.write(line)
                fh.flush()
            self._rewrite_csv()
        return record

    def start(self, experiment_id: str, model: str, dataset: str, config: dict | None = None, **kwargs) -> ExperimentRecord:
        """Create and log a ``running`` record before training begins."""
        record = ExperimentRecord(
            experiment_id=experiment_id,
            model=model,
            dataset=dataset,
            config_hash=hash_config(config or {}),
            environment=capture_environment(),
            **kwargs,
        )
        return self.append(record)

    def finish(self, record: ExperimentRecord, metrics: dict, **kwargs) -> ExperimentRecord:
        """Log the terminal state and measured metrics of a run.

        Only keys that were actually measured are stored; absent metrics stay
        ``None`` so that table generation can mark them as not-yet-run.
        """
        record.status = "completed"
        record.timestamp = datetime.now(timezone.utc).isoformat()
        record.metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        for key, value in kwargs.items():
            setattr(record, key, value)
        return self.append(record)

    def fail(self, record: ExperimentRecord, error: str) -> ExperimentRecord:
        """Mark a run as failed, preserving the error for post-mortem."""
        record.status = "failed"
        record.notes = (record.notes + f" | ERROR: {error}").strip(" |")
        return self.append(record)

    # ------------------------------------------------------------------ reading
    def load(self) -> list[ExperimentRecord]:
        """All records, in append order."""
        if not self.jsonl_path.exists():
            return []
        records = []
        for line in self.jsonl_path.read_text().splitlines():
            if line.strip():
                records.append(ExperimentRecord(**json.loads(line)))
        return records

    def completed(self) -> list[ExperimentRecord]:
        """Only successful runs — the ones eligible to appear in a table."""
        return [r for r in self.load() if r.status == "completed"]

    def find(self, experiment_id: str) -> list[ExperimentRecord]:
        """All records for one experiment id (including failures and reruns)."""
        return [r for r in self.load() if r.experiment_id == experiment_id]

    def best(self, experiment_id: str, metric: str = "mAP50_95"):
        """Best completed record for an experiment by a metric, or None."""
        candidates = [r for r in self.find(experiment_id) if r.status == "completed" and r.metrics.get(metric) is not None]
        return max(candidates, key=lambda r: r.metrics[metric]) if candidates else None

    def _rewrite_csv(self) -> None:
        """Rewrite the CSV view; skipped for very large ledgers.

        The temp file name is unique per process. A single fixed temp path meant that two
        concurrent writers could both open it, so whichever replaced first removed the file
        the other was about to replace -- a crash unrelated to the run itself.
        """
        rows = [r.flatten() for r in self.load()]
        if not rows:
            return
        fields = list(rows[0].keys())
        tmp = self.csv_path.with_name(f"{self.csv_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tmp.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp, self.csv_path)
        finally:
            # Never leave a stray temp file behind, even if the replace failed.
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
