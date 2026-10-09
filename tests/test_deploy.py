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
repository cannot state one thing and ship another.

Getting the memory figure right took two attempts, and both failures are pinned here:

* the envelope must come from a primitive that **describes the measured process**, not the one that
  forked it (`ru_maxrss` survives `execve`, so a child inherits its parent's high-water mark — the
  first version of this file reported the *test suite's* footprint and passed alone while failing in
  the full suite);
* the budget must be keyed by the **framework build**, because importing the CUDA-enabled wheel
  costs far more than the CPU-only one and a host resolves `torch` to the former.

Nothing here needs a dataset download: the workload uses a synthetic image, which is the point of
measuring an envelope rather than an accuracy.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app.py"
CONFIG = REPO_ROOT / ".streamlit" / "config.toml"
RUNTIME = REPO_ROOT / "runtime.txt"
WEIGHTS = REPO_ROOT / "weights" / "sarvo_ssac001.pt"
WEIGHTS_DOC = REPO_ROOT / "weights" / "README.md"
DEPLOY_DOC = REPO_ROOT / "docs" / "DEPLOYMENT.md"

#: Peak-RSS budget in MB, keyed by framework build.
#:
#: * ``cpu``  — **measured** on this development host (CPU-only wheel) against the documented
#:   1024 MB requirement, from the workload below in a fresh interpreter.
#: * ``cuda`` — the stated requirement for a host that resolves ``torch`` to the CUDA-enabled
#:   wheel, which is what a `pip install -r requirements.txt` does on Linux. That wheel is **not
#:   installed on this development host**, so the ceiling is stated rather than measured here and
#:   it is CI — which does install it — that enforces it.
#:
#: The budget is keyed rather than single-valued because a budget that ignores the wheel is right
#: on one machine and wrong on the host that matters: the first version of this was set from the
#: CPU wheel alone and failed on CI. The development host's limitation is stated here rather than
#: papered over with a number that was never measured on the wheel it names.
MEMORY_BUDGET_MB: dict[str, int] = {"cpu": 1024, "cuda": 3072}

#: Peak-RSS helper embedded in the scripts below. Kept as a string so the workload and the
#: regression test measure with exactly the same primitive instead of two similar-looking ones.
_PEAK_RSS_HELPER = r'''
import sys

def _peak_rss_mb():
    # Peak resident set of *this* process image, plus which primitive produced it.
    #
    # `resource.getrusage(RUSAGE_SELF).ru_maxrss` is deliberately not the primary source: it lives
    # in the task struct and survives execve, so a child forked from a large parent starts its
    # high-water mark at the parent's. Measured directly: the same trivial child reported 11 MB
    # from a small parent and 911 MB from one holding 900 MB. `VmHWM` is per-address-space and is
    # reset when exec installs the new image, so it describes the process that actually ran.
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0, "VmHWM"  # kB -> MB
    except OSError:
        pass
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / 1024.0 / 1024.0, "ru_maxrss"
    return raw / 1024.0, "ru_maxrss"
'''

#: The demo's workload, in a fresh interpreter: import the real web stack, load the committed
#: checkpoint, run two 640 px inferences, and report the envelope.
_MEMORY_SCRIPT = (
    _PEAK_RSS_HELPER
    + r'''
import numpy as np
import streamlit  # noqa: F401  (the demo's dependency, so the figure includes the web stack)
import torch

from saryolo.inference import load_detector, predict_image

# Which framework build produced the figure; the envelope depends on it more than on the model.
print("TORCH_BUILD=%s" % ("cuda" if torch.version.cuda else "cpu"))

weights = "weights/sarvo_ssac001.pt"
model = load_detector(weights, device="cpu")
image = np.zeros((640, 640, 3), np.uint8)
image[100:300, 100:300] = 90  # structure, so nothing short-circuits on a constant field
for _ in range(2):
    predict_image(model, image, imgsz=640, conf=0.25, device="cpu", weights=weights)

peak_mb, source = _peak_rss_mb()
print("PEAK_RSS_SOURCE=%s" % source)
print("PEAK_RSS_MB=%.1f" % peak_mb)
import resource
print("RU_MAXRSS_MB=%.1f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
'''
)

#: The same two primitives, in a process that imports nothing heavy, so the regression test below
#: can isolate the *measuring* from the *measured*.
_TRIVIAL_PEAK_SCRIPT = (
    _PEAK_RSS_HELPER
    + r'''
import resource
peak_mb, source = _peak_rss_mb()
print("PEAK_RSS_SOURCE=%s" % source)
print("VMHWM_MB=%.1f" % peak_mb)
print("RU_MAXRSS_MB=%.1f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
'''
)

#: Allocates a known amount of memory in *its own* process, then forks the trivial child. Used to
#: vary the parent's size deterministically, independent of how large the pytest process happens
#: to be when the check runs. The child source is passed through the environment to avoid quoting
#: a script inside a script.
_SPAWNER_SCRIPT = r'''
import os
import subprocess
import sys

ballast_mb = int(os.environ.get("BALLAST_MB", "0"))
if ballast_mb:
    ballast = bytearray(ballast_mb * 1024 * 1024)
    for index in range(0, len(ballast), 4096):  # touch every page, so it is really resident
        ballast[index] = 1
child = subprocess.run(
    [sys.executable, "-c", os.environ["CHILD_SCRIPT"]],
    capture_output=True, text=True, check=True,
)
sys.stdout.write(child.stdout)
'''


def _free_port() -> int:
    """An ephemeral port, so the check cannot collide with anything else on the machine."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _budget_for(build: str) -> int:
    """The documented budget for a framework build, refusing an undocumented one rather than guessing."""
    assert build in MEMORY_BUDGET_MB, (
        f"no documented memory budget for a {build!r} framework build; measure the envelope and "
        f"record it before enforcing it (documented: {sorted(MEMORY_BUDGET_MB)})"
    )
    return MEMORY_BUDGET_MB[build]


def _assert_inside_budget(build: str, peak_mb: float) -> None:
    """The envelope guard itself, separated so each build's boundary can be exercised directly."""
    budget = _budget_for(build)
    assert peak_mb > 100.0, f"peak RSS {peak_mb} MB is implausibly small; the workload did not run"
    assert peak_mb < budget, (
        f"the demo peaked at {peak_mb:.0f} MB on a {build} framework build, over the documented "
        f"{budget} MB budget"
    )


def _spawn_reported_peaks(ballast_mb: int) -> dict[str, float]:
    """Run the trivial child from a parent holding ``ballast_mb`` MB, and report both primitives."""
    proc = subprocess.run(
        [sys.executable, "-c", _SPAWNER_SCRIPT],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "BALLAST_MB": str(ballast_mb), "CHILD_SCRIPT": _TRIVIAL_PEAK_SCRIPT},
    )
    assert proc.returncode == 0, f"the measuring harness failed:\n{proc.stderr[-1000:]}"
    found = dict(re.findall(r"(VMHWM_MB|RU_MAXRSS_MB)=([\d.]+)", proc.stdout))
    assert set(found) == {"VMHWM_MB", "RU_MAXRSS_MB"}, (
        "the harness did not report both primitives, so the comparison would be vacuous:\n"
        + proc.stdout
    )
    return {key: float(value) for key, value in found.items()}


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
        f"runtime.txt pins {pinned} but the package requires {requires}"
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


def test_the_deployment_guide_states_every_memory_budget_this_suite_enforces():
    """Every budget the suite enforces must be a budget the guide tells a deployer to expect.

    Enforcing a number the documentation does not state is how a deployer ends up planning against
    the wrong envelope -- which is not hypothetical here: the first version of this budget was set
    from one wheel, unstated in the guide, and was wrong for the wheel a host installs.
    """
    guide = DEPLOY_DOC.read_text()
    missing = [f"{mb} MB" for mb in MEMORY_BUDGET_MB.values() if f"{mb} MB" not in guide]
    assert not missing, f"docs/DEPLOYMENT.md does not state the enforced budget(s): {missing}"


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
def test_the_envelope_measurement_describes_the_child_not_the_parent_that_forked_it():
    """Pin the primitive: the peak must describe the measured process, not the measuring one.

    This is the defect that shipped first. The envelope was read from `ru_maxrss`, which is carried
    across `execve`, so the reported peak was the *test process's* footprint: the check passed when
    run alone (small parent) and failed inside the full suite (large parent), while the number was
    always describing the wrong process. The assertion is deliberately two-sided -- the chosen
    primitive must agree across parents of different sizes, *and* the rejected one must visibly
    disagree -- so it cannot pass by measuring nothing.
    """
    small = _spawn_reported_peaks(0)
    large = _spawn_reported_peaks(400)

    drift = abs(large["VMHWM_MB"] - small["VMHWM_MB"])
    assert drift < 64, (
        f"the parent-independent peak moved with the parent's size "
        f"({small['VMHWM_MB']:.0f} MB -> {large['VMHWM_MB']:.0f} MB); the primitive is wrong"
    )
    inherited = large["RU_MAXRSS_MB"] - small["RU_MAXRSS_MB"]
    assert inherited > 200, (
        f"`ru_maxrss` did not inherit the parent's 400 MB footprint as expected "
        f"({small['RU_MAXRSS_MB']:.0f} MB -> {large['RU_MAXRSS_MB']:.0f} MB), so the check above "
        "proves nothing about parent-independence"
    )


def test_a_fresh_interpreter_serving_the_demo_stays_inside_the_documented_memory_budget():
    """The demo's real envelope, measured where it is meaningful: in a process that is only the demo.

    Measuring inside the test process would report the whole suite's accumulated memory and would
    answer a different question. A fresh interpreter is what a container actually runs, and the
    dominant term turns out to be importing the deep-learning stack rather than the 6.77 MB
    checkpoint -- which is what decides whether a small container is enough.
    """
    import torch

    proc = subprocess.run(
        [sys.executable, "-c", _MEMORY_SCRIPT],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        "the demo workload failed in a fresh interpreter:\n" + proc.stderr[-2000:]
    )
    measured = re.search(r"PEAK_RSS_MB=([\d.]+)", proc.stdout)
    reported_build = re.search(r"TORCH_BUILD=(\w+)", proc.stdout)
    source = re.search(r"PEAK_RSS_SOURCE=(\w+)", proc.stdout)
    assert measured and reported_build and source, (
        "the workload printed no measurement:\n" + proc.stdout[-2000:]
    )

    # The figure must come from the parent-independent primitive. Asserting *which* one is what
    # keeps a future edit from quietly reverting to the one this file exists to avoid.
    expected_source = "VmHWM" if Path("/proc/self/status").is_file() else "ru_maxrss"
    assert source.group(1) == expected_source, (
        f"the envelope was measured with {source.group(1)} instead of {expected_source}"
    )

    # The subprocess is this interpreter, so the wheel it reports must be the one this process has.
    # A mismatch would mean the figure came from something other than the demo's own environment --
    # and since the budget is keyed by the build, that would silently test the wrong boundary.
    local_build = "cuda" if torch.version.cuda else "cpu"
    assert reported_build.group(1) == local_build, (
        f"the subprocess reported a {reported_build.group(1)} build but this interpreter is "
        f"{local_build}"
    )
    _assert_inside_budget(local_build, float(measured.group(1)))


def test_the_budget_check_accepts_a_realistic_envelope_and_rejects_a_regression():
    """Each build's boundary is exercised directly, because only one wheel is installed here.

    The ``cuda`` budget cannot be reached by running the workload on this host, and left untested
    it would be exercised only by the platform it is supposed to gate. The recorded envelopes stand
    in for the missing wheel, and the assertion that a gross regression is still caught keeps the
    guard from being widened into decoration.
    """
    for build, observed in (("cpu", 460.0), ("cuda", 1800.0)):
        _assert_inside_budget(build, observed)  # a realistic envelope must pass
        _assert_inside_budget(build, float(_budget_for(build) - 1.0))  # just inside must pass
        with pytest.raises(AssertionError, match="over the documented"):
            _assert_inside_budget(build, float(_budget_for(build)))  # the boundary itself must fail
    # A gross regression must still be caught, or the guard would be widened into decoration.
    with pytest.raises(AssertionError, match="over the documented"):
        _assert_inside_budget("cuda", 8192.0)
    with pytest.raises(AssertionError, match="implausibly small"):
        _assert_inside_budget("cpu", 1.0)
    with pytest.raises(AssertionError, match="no documented memory budget"):
        _assert_inside_budget("tpu", 500.0)
