"""Deployment-contract tests: what a hosted instance needs that a local run does not prove.

Why this file exists
--------------------
`tests/test_app.py` drives the Streamlit script through Streamlit's in-process harness, which
proves the *script* is right. It does not prove the deployed *entry point* starts. A missing
dependency, a malformed `.streamlit/config.toml`, an incompatible pinned runtime, or a committed
checkpoint that no longer matches its published checksum all pass that harness and fail on a host —
and Streamlit Community Cloud re-deploys on every push to the connected branch, so a broken commit
reaches the public URL with nothing in the way.

These checks are that gate. Each compares an artefact the host consumes (the entry point, the
config, the runtime pin, the committed weight) against the code or document that defines it, so the
repository cannot state one thing and ship another. The last one measures the demo's real memory
envelope in a **fresh interpreter**, because a process that has already imported the entire test
suite cannot answer "how much does a container need for this app?".

Nothing here needs a dataset download: the workload uses a synthetic image, which is the point of
measuring an envelope rather than an accuracy.
"""

from __future__ import annotations

import hashlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app.py"
CONFIG = REPO_ROOT / ".streamlit" / "config.toml"
RUNTIME = REPO_ROOT / "runtime.txt"
WEIGHTS = REPO_ROOT / "weights" / "sarvo_ssac001.pt"
WEIGHTS_DOC = REPO_ROOT / "weights" / "README.md"
DEPLOY_DOC = REPO_ROOT / "docs" / "DEPLOYMENT.md"

#: The repository's own operating budget for the demo, in MB. This is a *measured envelope*, not a
#: platform specification -- the hosting platform's own limit is not something this repository can
#: verify, so it states the budget it requires and holds itself to it. The measured peak is far
#: below it; the point of the check is to catch a gross regression, not to run close to the edge.
MEMORY_BUDGET_MB = 1536

#: Runs in a fresh interpreter: import the real web stack, load the committed checkpoint, run the
#: demo's workload, and report the process high-water RSS. `ru_maxrss` is KiB on Linux and bytes on
#: macOS, so the divisor is chosen at run time rather than guessed.
_MEMORY_SCRIPT = r'''
import resource
import sys

import numpy as np
import streamlit  # noqa: F401  (the demo's dependency, so the figure includes the web stack)

from saryolo.inference import load_detector, predict_image

weights = "weights/sarvo_ssac001.pt"
model = load_detector(weights, device="cpu")
image = np.zeros((640, 640, 3), np.uint8)
image[100:300, 100:300] = 90  # structure, so nothing short-circuits on a constant field
for _ in range(2):
    predict_image(model, image, imgsz=640, conf=0.25, device="cpu", weights=weights)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("PEAK_RSS_MB=%.1f" % (peak / 1024.0 if sys.platform != "darwin" else peak / 1024.0 / 1024.0))
'''


def _free_port() -> int:
    """An ephemeral port, so the check cannot collide with anything else on the machine."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ------------------------------------------------------------------ artefacts vs their definitions
def test_the_configured_upload_limit_is_the_limit_the_code_enforces():
    """The documented 8 MB must be the enforced 8 MB, in the config *and* in the code.

    These two numbers live in different files and are edited at different times. If they drift, the
    app either accepts a file the platform already refused, or advertises a limit it does not apply
    -- and the guide a reader follows would be wrong about which.
    """
    import tomllib

    from saryolo.inference import MAX_UPLOAD_BYTES

    config = tomllib.loads(CONFIG.read_text())
    limit_mb = config["server"]["maxUploadSize"]
    assert limit_mb == MAX_UPLOAD_BYTES // 1024**2, (
        f".streamlit/config.toml allows {limit_mb} MB but the code refuses above "
        f"{MAX_UPLOAD_BYTES / 1024**2:.0f} MB"
    )
    # A non-interactive host needs headless, and a research demo should send no telemetry.
    assert config["server"]["headless"] is True
    assert config["browser"]["gatherUsageStats"] is False


def test_the_pinned_runtime_satisfies_the_python_floor_the_package_declares():
    """A host installs from the pin; it must be a version the package says it supports."""
    import tomllib

    pinned = RUNTIME.read_text().strip()
    assert re.fullmatch(r"python-3\.\d+", pinned), f"unexpected runtime pin: {pinned!r}"
    pinned_version = tuple(int(part) for part in pinned.removeprefix("python-").split("."))

    requires = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", requires)
    assert floor, f"pyproject declares no Python floor: {requires!r}"
    assert pinned_version >= (int(floor.group(1)), int(floor.group(2))), (
        f"runtime.txt pins 3.{pinned_version[1]} but the package requires {requires}"
    )


def test_the_committed_checkpoint_matches_the_checksum_and_size_the_repository_publishes():
    """The binary in git must be the measured one, and its published note must describe it.

    A committed weight is the one artefact whose contents nobody reads in review, so the checksum
    is the only thing tying the file people download to the file that produced the published
    numbers. Reading the expected value *from the document* (rather than repeating it here) means
    the note and the artefact cannot disagree without this failing.
    """
    assert WEIGHTS.is_file(), f"the committed checkpoint is missing: {WEIGHTS}"
    doc = WEIGHTS_DOC.read_text()

    documented_sha = re.search(r"\| SHA-256 \| `([0-9a-f]{64})` \|", doc)
    assert documented_sha, "weights/README.md states no SHA-256"
    actual_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    assert actual_sha == documented_sha.group(1), (
        f"committed checkpoint hashes to {actual_sha} but weights/README.md publishes "
        f"{documented_sha.group(1)}"
    )

    documented_size = re.search(r"\| Size \| ([\d,]+) bytes", doc)
    assert documented_size, "weights/README.md states no size"
    expected_bytes = int(documented_size.group(1).replace(",", ""))
    assert WEIGHTS.stat().st_size == expected_bytes, (
        f"checkpoint is {WEIGHTS.stat().st_size} bytes but the note says {expected_bytes}"
    )


def test_the_deployment_guide_states_the_memory_budget_this_suite_enforces():
    """The budget asserted below must be the budget the guide tells a deployer to expect."""
    assert f"{MEMORY_BUDGET_MB} MB" in DEPLOY_DOC.read_text(), (
        f"docs/DEPLOYMENT.md does not state the {MEMORY_BUDGET_MB} MB budget the suite enforces"
    )


# --------------------------------------------------------------------- the hosted entry point
def test_the_streamlit_entry_point_boots_and_answers_its_health_check(tmp_path):
    """The real server must start, become healthy, and serve the page -- not just run in-process.

    This is the failure a hosted deploy actually hits: the stack resolves, the config parses, the
    server binds. Streamlit's in-process harness bypasses all three. The check is deliberately
    *not* skipped when the server fails to come up, because "it did not start" is exactly the
    defect it exists to catch.
    """
    port = _free_port()
    log = tmp_path / "streamlit-server.log"
    with log.open("w") as sink:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(APP),
                "--server.headless", "true",
                "--server.port", str(port),
                "--server.fileWatcherType", "none",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=REPO_ROOT, stdout=sink, stderr=subprocess.STDOUT, text=True,
        )
    try:
        healthy = False
        deadline = time.time() + 150
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=3
                ) as response:
                    if response.status == 200 and response.read().strip() == b"ok":
                        healthy = True
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        assert healthy, (
            f"the entry point never became healthy (server exit={proc.poll()}); log:\n"
            + log.read_text()[-2000:]
        )
        # Health alone would pass for a server that answers but cannot render, so fetch the shell.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=15) as response:
            body = response.read()
            assert response.status == 200
            assert b"<title>" in body, "the server answered without serving a page"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    # A process that ignores SIGTERM would leave a port squatted on the host; assert it is gone.
    assert proc.poll() is not None, "the server did not exit after SIGTERM"


# ------------------------------------------------------------------------ the memory envelope
def test_a_fresh_interpreter_serving_the_demo_stays_inside_the_documented_memory_budget():
    """The demo's real envelope, measured where it is meaningful: in a process that is only the demo.

    Measuring inside the test process would report the whole suite's accumulated memory and would
    answer a different question. A fresh interpreter is what a container actually runs, and the
    dominant term turns out to be importing the deep-learning stack rather than the 6.77 MB
    checkpoint -- which is what decides whether a small container is enough.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _MEMORY_SCRIPT],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        "the demo workload failed in a fresh interpreter:\n" + proc.stderr[-2000:]
    )
    measured = re.search(r"PEAK_RSS_MB=([\d.]+)", proc.stdout)
    assert measured, "the workload printed no measurement:\n" + proc.stdout[-2000:]
    peak_mb = float(measured.group(1))
    # Non-vacuity: a measurement of ~0 would mean the workload never ran.
    assert peak_mb > 100.0, f"peak RSS {peak_mb} MB is implausibly small; the workload did not run"
    assert peak_mb < MEMORY_BUDGET_MB, (
        f"the demo peaked at {peak_mb:.0f} MB, over the documented {MEMORY_BUDGET_MB} MB budget"
    )
