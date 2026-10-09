# Deployment — running the SARVO demo locally and hosted

This document is the operational half of the project: how to get the detector in front of a
user, on this machine and on a host. The research record lives in
[`../paper/RESULTS.md`](../paper/RESULTS.md),
[`claim_evidence_audit.md`](claim_evidence_audit.md) and
[`release_readiness.md`](../reports/release_readiness.md); nothing here changes a measured
number.

The guiding rule is that **the demo must never show a prediction that was not computed**.
There is no fallback model, no bundled sample output, and no cached example prediction: with
no checkpoint the application says so and disables detection.

---

## 1. What the system is

Three layers, in dependency order. Each layer is usable without the one above it, which is why
the same forward pass can be reached from a script, the CLI, or the interface.

| Layer | Path | Role |
| --- | --- | --- |
| Inference pipeline | `saryolo/inference.py` | validate an upload, load a checkpoint, run one forward pass, return one documented result type |
| Command line | `saryolo/cli.py` → `predict` | the same pipeline with file in / JSON out, for scripts and servers |
| Web interface | `app.py` | a thin Streamlit layer over the pipeline; it draws and reports, it does not compute |

`app.py` deliberately contains no model wiring. The box a user sees is drawn by
`saryolo.inference.annotate_image` from the detections `saryolo.inference.predict_image`
returned, and `tests/test_app.py` asserts the displayed detections equal a direct pipeline call
on the same chip.

---

## 2. Model weights

The trained checkpoint is a **training artefact and is not committed by default** — the
repository's `.gitignore` excludes `results/**`. `saryolo.inference.resolve_weights` therefore
looks in this order:

1. an explicit path (`saryolo predict --weights …`, or the sidebar field);
2. the `SARVO_WEIGHTS` environment variable;
3. `weights/sarvo_ssac001.pt`, then `results/runs/SSAC-001/weights/best.pt`.

A path that is *configured* is returned even when the file is missing, so the error message
names the path that was actually asked for. "Nothing configured" and "configured but missing"
are reported as different problems, because they have different fixes.

### Getting a checkpoint

**Option A — train one locally (CPU-feasible).** The pilot arm this demo was verified against:

```bash
python -m saryolo train --exp configs/exp/SSAC-001_hrsid_ssac.yaml
# writes results/runs/SSAC-001/weights/best.pt
```

**Option B — point at an existing checkpoint.** Set the variable and start the app:

```bash
export SARVO_WEIGHTS=/absolute/path/to/best.pt
streamlit run app.py
```

**Option C — commit the weights so a hosted instance has them with no configuration.** Copy
the 6.77 MB checkpoint to the documented location, which `resolve_weights` already discovers:

```bash
mkdir -p weights
cp results/runs/SSAC-001/weights/best.pt weights/sarvo_ssac001.pt
```

Option C is what makes a Streamlit Community Cloud deployment self-contained: the platform
clones the repository, so a file present in git is present in the running app. The file is well
inside GitHub's 100 MB per-file limit and Streamlit's repository-size guidance. If the artefact
is instead kept out of git, the app on the host will correctly report that no checkpoint is
configured — it will not fabricate output.

The checkpoint `results/runs/SSAC-001/weights/best.pt` is **3,396,329 parameters**, 6.77 MB on
disk, one class (`ship`), trained on the 200/60/60 HRSID subset at 320 px for 40 epochs on CPU.

---

## 3. Local setup and run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Then either entry point:

```bash
# Command line: one image in, JSON out (no web stack needed)
.venv/bin/python -m saryolo predict \
    --weights results/runs/SSAC-001/weights/best.pt \
    --source datasets/processed/hrsid_real/images/val/P0001_1200_2000_8400_9200.jpg \
    --imgsz 640 --conf 0.25 --out /tmp/prediction.json

# Web interface
.venv/bin/python -m streamlit run app.py
```

`streamlit run app.py` prints a local URL (by default `http://localhost:8501`).

---

## 4. Testing

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
python scripts/check_chart_layout.py
```

The inference and application tests skip with a **named reason** when the checkpoint or the
HRSID subset is absent, so the suite runs on a clean machine; a skip is reported as "not run
here", never as a pass. The current measured state is recorded in
`reports/release_readiness.md`.

---

## 5. Hosted deployment (Streamlit Community Cloud)

`https://share.streamlit.io` builds the repository, installs `requirements.txt`, and runs the
entry point.

| Setting | Value |
| --- | --- |
| Repository | this repository, branch `main` |
| Main file path | `app.py` |
| Python version | 3.12, pinned by `runtime.txt` |
| Dependencies | `requirements.txt` (includes `streamlit>=1.30.0`) |
| Runtime configuration | `.streamlit/config.toml` (upload limit 8 MB, headless, telemetry off) |
| Secrets / environment | `SARVO_WEIGHTS` — only needed if the weights are **not** committed (Option C in §2) |

Streamlit reads `.streamlit/config.toml` from the repository, so the 8 MB upload limit there
matches `saryolo.inference.MAX_UPLOAD_BYTES` and an oversized file is refused before it is read
into memory.

### Resource envelope

The app is designed to fit a small free-tier container. Measured locally on CPU:

| Quantity | Value |
| --- | --- |
| Parameters | 3.40 M |
| Checkpoint on disk | 6.77 MB |
| Peak process RSS while profiling | recorded per row in `latency_peak_rss_mb` (see `docs/assets/facts.json`) |
| Warm single-image latency, CPU, 640 px | ~70 ms (first call of a session includes warm-up) |
| Cold start | checkpoint load into memory; cached per `(path, mtime)` for the session |

The model is cached with `st.cache_resource` keyed on the checkpoint's modification time, so
weights are loaded once per session and a re-trained file is picked up rather than served stale.

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| "No checkpoint is configured" | neither `SARVO_WEIGHTS` nor a default location supplied one | set the secret, or use Option C in §2 |
| "No checkpoint at `<path>`" | a configured path does not exist on the host | correct the path; the message names it |
| "could not load the checkpoint" | the file is not a detector checkpoint | re-export or re-train; the loader never substitutes a fresh model |
| "the file is N MB, over the 8 MB limit" | upload too large | downscale or re-encode |
| "could not be decoded as an image" | corrupt or mislabelled file | re-export as PNG/JPG |
| App starts but shows no boxes | no detections above the threshold | lower the confidence slider; an empty result is a valid pilot result |

### Current deployment status

**Not deployed from this machine.** Publishing requires an interactive Streamlit Community
Cloud account authorised against this GitHub account, which is an action only the repository
owner can take. Everything the host needs — entry point, dependency file, runtime config,
weight-resolution contract, and the tests above — is in place; the deployment step itself is
the outstanding item, and no public URL is claimed here.

---

## 6. Optional front end (to be built after the Streamlit version is deployed and verified)

A separate static front end would submit an image to a Python backend that hosts the model, over
an HTTPS contract. Two constraints are worth writing down before any of it is built:

* the PyTorch model cannot run inside a static front-end deployment — it needs a Python-runtime
  backend;
* the API must return the same `InferenceResult` shape the CLI already prints, so a front end
  cannot invent a second representation of a prediction.

No front end or API service has been built, no endpoint exists, and no third-party host has been
contacted.

---

## 7. Known limitations of the deployed system

These are properties of the *model*, not of the deployment, and the interface states them to
the user:

* **Pilot scale.** One dataset (HRSID), a 200/60/60 subset, one class (ship), 320 px training,
  CPU only. mAP50:95 = 0.30.
* **The adaptive-allocation accuracy claim is withdrawn** after a three-seed check; the
  negative is reported in `paper/RESULTS.md`.
* **The efficiency result is scale-bound** — sparse execution is slower at 320 px and faster at
  640 px.
* **No cross-sensor, cross-resolution, or leave-one-source-out result** exists.
* **GPU paths are unexecuted on this host**; no CUDA device is available.
* Detections are the raw model output at the chosen threshold; low-confidence boxes are common
  at this scale, and that is the model rather than the display.

Full detail: [`claim_evidence_audit.md`](claim_evidence_audit.md),
[`release_readiness.md`](../reports/release_readiness.md),
[`defect_log.md`](../reports/defect_log.md).
