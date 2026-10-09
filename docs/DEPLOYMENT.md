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

The demo's checkpoint **is committed**, at `weights/sarvo_ssac001.pt` — deliberately, and as the
only weight in the repository, so a hosted instance has one with no configuration. Other
training artefacts are not committed: `.gitignore` excludes `results/**` and ignores `*.pt`, with
a single narrow negation for that file. Provenance and checksum are in
[`../weights/README.md`](../weights/README.md).

`saryolo.inference.resolve_weights` therefore looks in this order:

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

**Option C — use the committed weights (the default, and what a hosted deploy relies on).**
`weights/sarvo_ssac001.pt` is in the repository, so a Streamlit Community Cloud instance gets it
by cloning — no secret, no upload, no environment variable:

```bash
sha256sum weights/sarvo_ssac001.pt   # c8128733...45a98a5, per weights/README.md
```

To refresh it from a re-trained arm:

```bash
cp results/runs/SSAC-001/weights/best.pt weights/sarvo_ssac001.pt
```

The file is well inside GitHub's 100 MB per-file limit and Streamlit's repository-size guidance.
If the artefact were instead kept out of git, the app on the host would correctly report that no
checkpoint is configured — it would not fabricate output.

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
| Secrets / environment | **none required** — the checkpoint is committed at `weights/sarvo_ssac001.pt`. Set `SARVO_WEIGHTS` only to serve a different checkpoint |

Streamlit reads `.streamlit/config.toml` from the repository, so the 8 MB upload limit there
matches `saryolo.inference.MAX_UPLOAD_BYTES` and an oversized file is refused before it is read
into memory.

### Resource envelope

The app is designed to fit a small free-tier container. **Measured on this host** (CPU, no GPU),
resident set of one process with `streamlit` imported, the checkpoint loaded, and one 640 px
inference run:

| Quantity | Value |
| --- | --- |
| Parameters | 3.40 M |
| Checkpoint on disk | 6.77 MB |
| RSS after importing the stack | 269 MB |
| … after loading the checkpoint | 293 MB (+25 MB) |
| … during one 640 px inference | 443 MB (+150 MB, the activations) |
| **Peak process RSS (high-water)** | **446 MB** |
| Warm single-image latency, CPU, 640 px | ~70 ms; the first call of a session includes warm-up (~0.9 s cold here) |

Two caveats, stated rather than implied: the dominant term is the *import* of the deep-learning
stack, not the model, so the figure moves with the library version; and this was measured on the
16-core CPU host above rather than inside a hosted container, so it is an envelope and not a
platform guarantee.

**The budget this repository requires is 1536 MB of peak RSS for one session**, leaving the
measured figure roughly 3× of headroom for a second concurrent session's image buffers. That
number is not decorative: `tests/test_deploy.py` re-measures the envelope in a *fresh interpreter*
and fails if it crosses it, and it currently reports ~452 MB — agreeing with the 446 MB above,
which was measured the same way but with a decoded HRSID chip instead of a synthetic one. The
hosting platform's own limit is not something this repository can verify, so the guide states the
requirement it can hold itself to instead of quoting a specification it has not tested against.

The model is cached with `st.cache_resource` keyed on the checkpoint's modification time, so
weights are loaded once per session and a re-trained file is picked up rather than served stale.

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| "No checkpoint is configured" | neither `SARVO_WEIGHTS` nor a default location supplied one — e.g. the committed weights are missing from the checkout | restore `weights/sarvo_ssac001.pt`, or set `SARVO_WEIGHTS` |
| "No checkpoint at `<path>`" | a configured path does not exist on the host | correct the path; the message names it |
| "could not load the checkpoint" | the file is not a detector checkpoint | re-export or re-train; the loader never substitutes a fresh model |
| "the file is N MB, over the 8 MB limit" | upload too large | downscale or re-encode |
| "could not be decoded as an image" | corrupt or mislabelled file | re-export as PNG/JPG |
| App starts but shows no boxes | no detections above the threshold | lower the confidence slider; an empty result is a valid pilot result |

### Publishing on push

There is **no CLI, token, or API that creates a Community Cloud app** — the first deployment is
necessarily an interactive sign-in, and no workflow file can substitute for it. What *is*
platform behaviour is the update path: once the app exists, Streamlit sets a webhook on the
connected repository and **every push to the connected branch re-deploys it automatically**
(Streamlit's own description: "any time you do a git push, your app will update immediately").

That makes correctness *before* the push the only control that matters, which is why the deploy
contract is enforced by the test suite rather than by a deploy workflow:

| Guard (`tests/test_deploy.py`) | The hosted failure it prevents |
| --- | --- |
| The entry point boots and answers `/_stcore/health` | a missing dependency or malformed config shipping straight to the public URL |
| The config's upload limit equals `MAX_UPLOAD_BYTES` | the app advertising a limit it does not enforce |
| `runtime.txt` satisfies the declared Python floor | an incompatible runtime pinned for the host |
| The committed checkpoint matches its published SHA-256 and size | a swapped or stale binary serving different weights than the numbers describe |
| A fresh interpreter stays inside the 1536 MB budget | a memory blow-up on a small container |

### Step-by-step publish (owner action)

Publishing requires an interactive sign-in that cannot be performed on the owner's behalf, so
this is the exact path for the repository owner. Nothing in the repository needs to change first.

1. Sign in at `https://share.streamlit.io` **with GitHub**, and authorise the app for this
   repository (private-account permissions are not needed; the repository is public).
2. **Create app → Deploy a public app from GitHub.**
3. Set exactly these three fields:

   | Field | Value |
   | --- | --- |
   | Repository | `officialarghya29/sarr-imaging` |
   | Branch | `main` |
   | Main file path | `app.py` |

4. Open **Advanced settings** and confirm **Python 3.12** (matches `runtime.txt`). No secrets are
   required — the checkpoint is committed.
5. **Deploy** and wait for the build to finish. A non-zero-exit build shows its log in the UI;
   the most common cause is a missing dependency, which `requirements.txt` covers.

### Verifying the live app

Run every line of this on the **deployed** URL, not locally. Each check names what failure it
would catch.

| # | Do this | Pass looks like |
| --- | --- | --- |
| 1 | Open the URL | The title renders and the model panel shows Parameters **3.40 M**, Checkpoint **6.77 MB**, Device **cpu**, Classes **ship** — proving the committed weights loaded |
| 2 | Upload a SAR image and press **Run Detection** | Two images (input, annotated) and a Detections/Latency/Throughput row. A chip from the HRSID val split is the cheapest check; the app's local-sample picker will be absent on the host, because the dataset is not committed |
| 3 | Cross-check one detection against the pipeline | `.venv/bin/python -m saryolo predict --weights weights/sarvo_ssac001.pt --source <same file> --imgsz 640` gives the same count and confidences (the confidence field has 4 dp) |
| 4 | Upload a `.gif`, then a file over 8 MB | "That file cannot be used: … not a supported image type", and "over the 8 MB limit" respectively, each with a Recovery note — no detection control appears |
| 5 | Type a path that does not exist into the sidebar | "No checkpoint at `<path>`" plus Recovery guidance, and no **Run Detection** button |
| 6 | Reload the page | The app returns to state 1 without re-uploading — weights are cached per `(path, mtime)` |
| 7 | Watch the app's resource panel while running a few images | Latency is stated in ms and the app does not restart — a memory-crash shows as a restart, and the local peak was 446 MB |

If every row passes, record the URL in this file and in the README. If a row fails, the failure is
reported as a failed check rather than omitted; rows 4 and 5 in particular are the ones a demo
typically skips and a reviewer typically tries.

### Current deployment status

**Not deployed from this machine.** Publishing requires an interactive Streamlit Community
Cloud account authorised against this GitHub account, which is an action only the repository
owner can take. Everything the host needs — entry point, dependency file, runtime config,
committed weights, and the tests above — is in place; the deployment step itself is the
outstanding item, and no public URL is claimed here. Verify with the table above before
recording one.

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
