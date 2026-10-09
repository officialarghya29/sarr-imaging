# Release readiness

The master directive (section 6) requires each readiness requirement to be reported as
**passed**, **failed**, or **untested**, with the evidence for the verdict. This file is that
record, and it is deliberately conservative: a requirement is only marked *passed* when a test
or measurement in this repository exercises it on this machine, and everything not covered is
listed as a gap rather than implied to work.

The machine the checks below were run on is a CPU-only host (no CUDA device): a 16-core AMD
Ryzen 7 8840HS with 30 GB RAM, Python 3.12, Ultralytics 8.4.x, PyTorch CPU build. That is the
**minimum supported configuration**, and it is the environment the automated suite and CI run in.

## Requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| **Functional correctness** | **Passed** | Every architecture constructs and forwards (`tests/test_arch.py`); the CLI's subcommands respond and refuse bad input (`tests/test_cli.py`); the model, data layer, aug, loss, and evaluation modules each have dedicated tests. Full result in [`reproduction_status.md`](reproduction_status.md). |
| **Automated testing** | **Passed** | The full suite passes on CPU with no dataset download (`pytest -q`), and CI re-runs lint + the suite on Python 3.12 for every push. Composition: `docs/assets/tests.svg`. |
| **Training stability** | **Passed** | A miniature, complete training cycle runs end to end in `tests/test_conditioning_smoke.py` (metadata-bearing dataset, validation forward asserted) and `tests/test_peft.py` (a LoRA arm trains and writes a usable checkpoint); the loss paths are exercised in `tests/test_losses.py`. |
| **Inference** | **Passed** | Prediction outputs and formats are validated by the evaluator tests (`tests/test_metrics.py`), single- and batch-image loading for timing (`tests/test_efficiency.py`, `tests/test_efficiency_real.py`), and the CLI `eval` / `efficiency` paths (`tests/test_cli.py`). The single-image path is tested end to end as well: real chips produce boxes inside the image at the stated threshold, the CLI's JSON equals a direct pipeline call, and the demo displays exactly what the pipeline returns (`tests/test_inference.py`, `tests/test_app.py`). |
| **Numerical stability** | **Passed** | Finiteness is asserted across the architecture, front end, and loss tests (`torch.isfinite` in `tests/test_arch.py`, `tests/test_cfar_frontend.py`, `tests/test_losses.py`), the training loss **refuses** a non-finite value instead of backpropagating it (`tests/test_reliability.py`; defect D-08), and the inference pipeline refuses a non-finite detection rather than reporting it (`tests/test_inference.py`; defect D-11). Both are in [`defect_log.md`](defect_log.md). |
| **Resource stability** | **Passed (CPU)** | Repeated inference is asserted deterministic and flat in resident-set growth over 20 repeats, and the profiler records the process peak RSS next to every latency (`tests/test_reliability.py`). The demo's whole-process envelope is measured in a fresh interpreter and gated per framework build (`tests/test_deploy.py`). Peak GPU memory is untested — see the gaps below. |
| **Recovery** | **Passed** | Checkpoint writing and restoration are covered by `tests/test_peft.py::test_the_save_time_merge_makes_a_loadable_graph_and_restores_the_adapters` and `::test_a_lora_arm_end_to_end_trains_adapters_and_writes_a_usable_checkpoint`; corrupt or incompatible checkpoints are refused rather than half-loaded (`tests/test_reliability.py`, `tests/test_init.py`). |
| **Performance** | **Passed** | Parameters, FLOPs, latency, throughput, checkpoint size, and CPU peak memory are measured under a recorded protocol by `saryolo.evaluation.efficiency`; the measured values and their protocol are in [`reproduction_status.md`](reproduction_status.md) and `docs/assets/facts.json`. |
| **Low-resource compatibility** | **Passed (CPU) / Untested (GPU)** | The minimum configuration is CPU-only and is what the suite and every pilot run use; no component requires a GPU (`saryolo` imports and runs without CUDA). GPU execution is untested here because no CUDA device is available. |
| **Reproducibility** | **Passed** | Dependencies are declared in `requirements.txt`; the environment and the exact setup / train / eval / inference commands are documented in [`../README.md`](../README.md) and `docs/DATASETS.md`; generated figures are byte-stable across runs (fixed matplotlib hash salt, no timestamp); CI reproduces lint + test on a clean runner. |

## Application and deployment

The deployment workflow's own stages, reported the same way. "Verified" means the step was
actually executed on this machine and its result observed.

| Stage | Status | Evidence |
| --- | --- | --- |
| Single-image inference pipeline | **Verified** | `saryolo/inference.py`; real detections from the SSAC-001 checkpoint on held-out HRSID chips, boxes inside the image at the stated threshold (`tests/test_inference.py`). |
| Command-line entry point | **Verified** | `saryolo predict` prints and writes the same JSON a direct pipeline call produces (`tests/test_cli.py`). |
| Streamlit application | **Verified locally** | `streamlit run app.py` starts, loads the checkpoint, and completes upload → Run Detection → display; the shown detections equal a direct pipeline call (`tests/test_app.py`). Startup, missing-checkpoint, unloadable-checkpoint, and unusable-upload paths are all covered. |
| Model caching and resource control | **Verified** | The checkpoint is cached per `(path, mtime)` and reused for the session; uploads are size-, type-, and dimension-checked before a decode; the profiler records CPU peak RSS. |
| Deploy contract (what a host needs) | **Verified** | `tests/test_deploy.py` — the real `streamlit run app.py` entry point boots and answers `/_stcore/health` with a served page; the configured upload limit equals the code's; `runtime.txt` satisfies the declared Python floor; the committed checkpoint matches its published SHA-256 and size; a fresh interpreter serving the demo stays inside its build's budget (measured 451 MB against a 1024 MB ceiling on this host's CPU-only wheel; the 3072 MB ceiling for the CUDA wheel a hosted install resolves to is enforced by CI, since that wheel cannot be installed here). |
| Hosted public demo | **Untested — blocked** | Publishing requires an interactive Streamlit Community Cloud account authorised against this GitHub account, which only the repository owner can do; no workflow or token can create the app. No public URL is claimed. Everything the host needs is in place and documented in [`../docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md). |
| Optional separate front end | **Not started** | Depends on the Streamlit version being deployed and verified first; nothing has been built or hosted. |

## Not covered here — the honest gaps

These are **untested** or **blocked**, not passed. They are carried in `../paper/OUTLINE.md`'s
`blocked-on-run` list as well.

| Gap | Why it is not tested here |
| --- | --- |
| **GPU execution** (CUDA kernels, AMP, mixed precision, `torch.cuda.max_memory_allocated`) | No CUDA device on this host. The code paths are written and guarded by device checks, but they have not been executed. |
| **Out-of-memory behaviour** | Cannot be forced deterministically on a shared CPU host without destabilising it; the mechanism is exercised only indirectly through the memory-growth check. |
| **Full-release training, multi-seed at scale, cross-sensor / cross-resolution (LOSO) on real data** | Compute- and data-bound; the pilot on a 200/60/60 HRSID subset is the largest run performed. |
| **Second architecture (RT-DETR) training end to end** | The arm builds and is shape-tested (`tests/test_rtdetr_arm.py`), but no trained result exists. |
| **The hosted demo** | Requires interactive account authorisation against this GitHub account; cannot be performed from this environment. The application is verified locally instead. |
| **Cold-start and steady-state memory inside a real container** | The envelope is measured in a fresh interpreter with a parent-independent primitive and gated per framework build (`tests/test_deploy.py`, defect D-12), but on a 30 GB CPU host — not inside a container whose cgroup limit enforces it, and not with a second concurrent session. The CUDA-wheel ceiling is a stated requirement, enforced by CI rather than measured on this host. |

## How to reproduce these checks

```bash
.venv/bin/python -m ruff check .                       # lint
.venv/bin/python -m pytest -q                          # full suite, CPU, no download
.venv/bin/python scripts/check_chart_layout.py         # generated-figure layout lint
```

The suite is the evidence for every *passed* row above; where a test is skipped for a missing
dataset or GPU, the skip is explicit and named, so an absent resource reads as "not run here"
rather than as a pass.
