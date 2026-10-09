#!/usr/bin/env python
"""Verify a deployed SARVO demo over HTTP, and say plainly what HTTP cannot check.

Why this exists
---------------
"Deployed" and "working" are different claims, and the difference is invisible from a build log:
a host can serve an application shell while the Python script raises on its first run, and a green
build says nothing either way. So the deployment is checked against the running service rather
than against the pipeline that produced it.

What this establishes, and what it cannot
-----------------------------------------
HTTP reaches the server, so this can establish that the host is up, that the thing answering is
genuinely the Streamlit runtime, and that nothing is a 404 or a sleeping 503. It **cannot**
establish that `app.py` executes: interaction in Streamlit travels over a websocket, and a failure
inside the script appears there, not in the document. This script therefore prints that remaining
check as UNVERIFIED instead of implying it passed, and points at the browser table in
`docs/DEPLOYMENT.md` that does cover it. Claiming a script-level pass from an HTTP 200 is exactly
the overclaim this repository's rules exist to prevent.

Usage:
    python scripts/verify_deploy.py --url https://<app>.streamlit.app
    python scripts/verify_deploy.py --url http://127.0.0.1:8501        # a local instance
    python scripts/verify_deploy.py --url https://<app>.streamlit.app --json

Exit code is 1 when any check fails, so it can gate a release; 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

#: The shell Streamlit serves. Both markers were read off a running instance rather than guessed,
#: and both are present in the document the server returns for `/`.
SHELL_MARKERS = ('id="root"', "streamlit")
#: Where Streamlit exposes its runtime configuration; its shape (a JSON object naming
#: ``allowedOrigins``) is what distinguishes the runtime from any other web server that happens to
#: answer on the same port.
HOST_CONFIG_PATH = "/_stcore/host-config"
HEALTH_PATH = "/_stcore/health"

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"


@dataclass
class Report:
    """Collected rows, printed as a table and returned as JSON on request."""

    url: str
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.rows.append((name, status, detail))

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [row for row in self.rows if row[1] == FAIL]

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "ok": not self.failed,
            "checks": [{"check": n, "status": s, "detail": d} for n, s, d in self.rows],
        }


def _get(url: str, timeout: float) -> tuple[int, bytes, dict[str, str]]:
    """Fetch a URL, returning (status, body, headers). Raises only on *transport* failure.

    Header names are lower-cased on the way out. HTTP header names are case-insensitive but a
    plain ``dict(response.headers)`` preserves whatever case the server used, and servers differ:
    Streamlit sends ``content-type`` while a stdlib test server sends ``Content-type``. A
    case-sensitive lookup therefore passes on one and fails on the other, which is a bug that
    reads as the server's fault.

    An HTTP *error status* is returned rather than raised. The host answered, it just did not
    succeed, and the difference matters: on Community Cloud a 303 means the app is not reachable
    anonymously, which is a different problem with a different fix from a host that is down.
    Conflating them (the first version of this reported a 303 as "no response") sends the reader to
    the wrong diagnosis.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "sarvo-deploy-verify"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = {name.lower(): value for name, value in response.headers.items()}
            return response.status, response.read(), headers
    except urllib.error.HTTPError as exc:
        headers = {name.lower(): value for name, value in (exc.headers or {}).items()}
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 - the body is diagnostic only, never required
            body = b""
        return exc.code, body, headers


def _is_local(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def verify(url: str, timeout: float = 60.0) -> Report:
    """Run every check the transport allows, and record what it cannot see."""
    report = Report(url=url)
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    base = f"{parsed.scheme}://{parsed.netloc}"

    # ---------------------------------------------------------------- transport-level expectations
    if parsed.scheme == "https":
        report.add("scheme", PASS, "served over HTTPS")
    elif _is_local(host):
        report.add("scheme", PASS, f"{parsed.scheme} accepted for a local instance ({host})")
    else:
        report.add(
            "scheme", FAIL, f"{parsed.scheme or 'no'} scheme on a public host; a deploy must be HTTPS"
        )

    # --------------------------------------------------------------------------- is anything there
    try:
        started = time.perf_counter()
        status, body, headers = _get(base + "/", timeout)
        first_byte_ms = (time.perf_counter() - started) * 1000.0
    except (urllib.error.URLError, OSError) as exc:
        report.add("reachable", FAIL, f"no response from {base}: {exc}")
        # Everything below needs a response; record them as unverified rather than as failures, so
        # one unreachable host is not reported as five separate problems.
        for name in ("health endpoint", "runtime identity", "application shell"):
            report.add(name, UNVERIFIED, "not reachable")
        _add_script_reminder(report)
        return report
    if status == 200:
        report.add("reachable", PASS, f"HTTP 200 in {first_byte_ms:.0f} ms")
    elif 300 <= status < 400:
        report.add(
            "reachable",
            FAIL,
            f"HTTP {status} redirect to {headers.get('location', 'an unknown target')!r}; on "
            "Streamlit Community Cloud this means the app is not reachable anonymously -- it may "
            "not exist at this name, or it is private",
        )
    elif status == 404:
        report.add("reachable", FAIL, "HTTP 404: nothing is deployed at this path (check the URL)")
    elif status in {502, 503, 504}:
        report.add("reachable", FAIL, f"HTTP {status}: the host is up but the app is unavailable")
    else:
        report.add("reachable", FAIL, f"HTTP {status} from GET /")

    if status != 200:
        # Nothing behind a redirect, a 404 or an error page can be verified, and listing three more
        # failures that share this single cause would bury it. Record them as unverified instead.
        for name in ("health endpoint", "runtime identity", "application shell"):
            report.add(name, UNVERIFIED, f"not verifiable: GET / returned HTTP {status}")
        _add_script_reminder(report)
        return report

    # ------------------------------------------------------------------- /_stcore/health == "ok"
    try:
        health_status, health_body, _ = _get(base + HEALTH_PATH, timeout)
        body_text = health_body.decode("utf-8", "replace").strip()
        if health_status == 200 and body_text == "ok":
            report.add("health endpoint", PASS, f"{HEALTH_PATH} -> 200 {body_text!r}")
        else:
            report.add("health endpoint", FAIL, f"{HEALTH_PATH} -> {health_status} {body_text!r}")
    except (urllib.error.URLError, OSError) as exc:
        report.add("health endpoint", FAIL, f"{HEALTH_PATH} unreachable: {exc}")

    # --------------------------------------------------------------- is it really the Streamlit runtime
    try:
        config_status, config_body, _ = _get(base + HOST_CONFIG_PATH, timeout)
        config = json.loads(config_body.decode("utf-8", "replace"))
        if config_status == 200 and "allowedOrigins" in config:
            origins = len(config.get("allowedOrigins") or [])
            report.add("runtime identity", PASS, f"{HOST_CONFIG_PATH} names {origins} allowed origins")
        else:
            report.add(
                "runtime identity", FAIL, f"{HOST_CONFIG_PATH} -> {config_status}, unexpected shape"
            )
    except json.JSONDecodeError:
        report.add("runtime identity", FAIL, f"{HOST_CONFIG_PATH} did not return JSON")
    except (urllib.error.URLError, OSError) as exc:
        report.add("runtime identity", FAIL, f"{HOST_CONFIG_PATH} unreachable: {exc}")

    # ------------------------------------------------------------------ the shell the user receives
    if "text/html" not in headers.get("content-type", ""):
        report.add("application shell", FAIL, f"GET / is not HTML: {headers.get('content-type')!r}")
    else:
        text = body.decode("utf-8", "replace").lower()
        missing = [marker for marker in SHELL_MARKERS if marker.lower() not in text]
        if missing:
            report.add(
                "application shell",
                FAIL,
                f"GET / is HTML but does not look like the runtime (missing {missing})",
            )
        else:
            report.add("application shell", PASS, f"{len(body)} bytes of application shell")

    _add_script_reminder(report)
    return report


def _add_script_reminder(report: Report) -> None:
    """Record the thing HTTP cannot see, so a passing report is not read as a working model."""
    report.add(
        "inference runs in-app",
        UNVERIFIED,
        "interaction is websocket-driven; verify in a browser with the table in docs/DEPLOYMENT.md",
    )


def _print_report(report: Report) -> None:
    width = max(len(name) for name, _status, _detail in report.rows)
    print(f"Verifying {report.url}\n")
    for name, status, detail in report.rows:
        print(f"  {status:<10} {name:<{width}}  {detail}")
    print()
    if report.failed:
        print(f"{len(report.failed)} check(s) FAILED. A deployed app must pass every check.")
    else:
        print(
            "All HTTP-level checks passed. This does NOT confirm that the model runs: run the "
            "browser table\nin docs/DEPLOYMENT.md to establish that."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a deployed SARVO demo over HTTP.")
    parser.add_argument("--url", required=True, help="the deployed base URL, e.g. https://x.streamlit.app")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    try:
        report = verify(args.url, timeout=args.timeout)
    except socket.gaierror as exc:
        print(f"FAIL  DNS lookup failed for {args.url}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
