"""Tests for the reproducibility fingerprint recorded with every experiment row.

``saryolo/tracking/env.py`` is what stamps each ledger entry with the software, hardware
and commit it ran on -- the values that are supposed to make a run reproducible. It had no
dedicated tests, and it is the one module where a wrong value is invisible: a fingerprint
that silently returns ``None`` for a field looks exactly like a fingerprint for a machine
that could not report it, and every downstream row inherits the gap. The properties pinned
here are the ones that decide whether the stamp means anything:

* ``hash_file`` is a plain SHA-256 -- checked against ``hashlib`` directly, and across the
  1 MiB read-chunk boundary, so a chunking bug cannot produce a stable-looking wrong digest.
* ``hash_config`` is order-independent (a config's key order is not a fact about the run)
  but value-sensitive, and it does not crash on a non-JSON-native value.
* ``git_commit`` reads *this* repository and returns ``None`` outside a work tree -- it must
  never guess.
* ``capture_environment`` actually carries the version fields, not just their keys.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from saryolo.tracking.env import (
    capture_environment,
    describe_hardware,
    git_commit,
    hash_config,
    hash_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hash_file_is_a_plain_sha256_and_matches_across_the_read_chunk(tmp_path):
    """The digest must be over the whole file regardless of the read-chunk size."""
    data = bytes(range(256)) * 5000  # ~1.28 MB, past the 1 MiB default chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    assert hash_file(path) == hashlib.sha256(data).hexdigest()
    # A chunk that does not divide the file exercises the boundary read; the digest must not
    # move, or a large checkpoint and a small one would be fingerprinted inconsistently.
    assert hash_file(path, chunk=7) == hashlib.sha256(data).hexdigest()


def test_hash_file_returns_none_for_a_missing_path(tmp_path):
    """A path that is not a file is ``None``, not a hash of nothing."""
    assert hash_file(tmp_path / "does-not-exist.bin") is None
    assert hash_file(tmp_path) is None, "a directory is not a file to fingerprint"


def test_hash_config_is_order_independent_but_value_sensitive():
    """Config drift detection only works if key order is not part of the identity."""
    a = {"lr": 0.01, "epochs": 2, "names": ["x", "y"]}
    b = {"names": ["x", "y"], "epochs": 2, "lr": 0.01}
    assert hash_config(a) == hash_config(b), "reordering the keys changed the config hash"
    assert hash_config(a) != hash_config({**a, "epochs": 3}), "a changed value did not change the hash"
    assert len(hash_config(a)) == 16

    # A value that is not JSON-native is stringified rather than raising, so a config carrying
    # a Path still gets a stable fingerprint instead of crashing the run it was recording.
    assert hash_config({"weights": Path("a/b.pt")}) == hash_config({"weights": Path("a/b.pt")})


def test_git_commit_reads_this_repository_and_is_none_outside_one(tmp_path):
    """The commit must be the real HEAD here, and ``None`` where there is no work tree."""
    head = git_commit(REPO_ROOT)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert head == expected
    assert len(head) == 40 and all(c in "0123456789abcdef" for c in head)
    # Outside a repository the field is None -- a guess would make a row look traceable
    # when it is not.
    assert git_commit(tmp_path) is None


def test_describe_hardware_reports_a_cpu_and_a_cuda_verdict():
    """The hardware block must always name a CPU and always state whether CUDA is present."""
    info = describe_hardware()
    assert info["cpu"], "the CPU field is empty, so the row cannot name its host"
    assert "cuda_available" in info
    if not info["cuda_available"]:
        # No CUDA: the GPU fields must be explicitly empty rather than stale.
        assert info.get("gpu_count") == 0
        assert info.get("gpu_name") is None
        assert info.get("gpu_memory_gb") is None


def test_capture_environment_records_the_versions_and_not_only_their_keys():
    """Every field the reproducibility stamp promises must carry a real value here."""
    env = capture_environment(REPO_ROOT)
    assert env["git_commit"] == git_commit(REPO_ROOT)
    assert env["python"].count(".") == 2, f"python version looks wrong: {env['python']!r}"
    assert env["platform"]
    for key in ("torch", "ultralytics", "numpy"):
        assert env[key], f"the environment stamp left {key} empty on a host that has it"
    assert "cuda" in env and env["cuda_available"] in (True, False)
