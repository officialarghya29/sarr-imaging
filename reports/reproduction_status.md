# Reproduction status — what can and cannot be reproduced on this machine

**Date:** 2026-10-03
**Rule applied:** a result is "reproduced" only if the command was actually executed and
its output observed. Installing code, or intending to run it, is not reproduction.

---

## 1. Reproducible now, without a GPU or a dataset

Every item below was run and observed in this environment.

| What | Command | Observed result |
| --- | --- | --- |
| Test suite | `.venv/bin/python -m pytest -q` | **366 passed** |
| Baseline parity with stock YOLO11 | `pytest tests/test_arch.py -k baseline` | exact published counts (n: 2,624,080; s: 9,458,752 at 80 classes) |
| All variants build and forward | `pytest tests/test_arch.py -k every_variant` | 92 variants pass |
| Module identity at init | `pytest tests/test_arch.py -k identity` | `max\|f(x)−x\| = 0.0e+00` for all |
| Architecture cost benchmark | `python -m saryolo bench --variants v2_full v2_lite_s ...` | see §2 |
| Efficiency profiling | `pytest tests/test_efficiency.py` | 17 passed |
| Paper tables generate | `python -m saryolo assets` | 18 files written; every accuracy cell `TBD` |
| README charts regenerate | `python scripts/make_readme_assets.py` | 9 SVGs + `facts.json`, byte-reproducible |

## 2. Measured cost (the only quantitative results that exist)

Profiled on this machine with a one-class head. These are real measurements.

| Variant | Params (M) | GFLOPs @640² |
| --- | ---: | ---: |
| `baseline_s` (stock YOLO11s) | 9.428 | 21.67 |
| `v2_full` (reference) | 16.230 | 55.68 |
| `v2_full_p35_s` | 15.808 | 38.76 |
| **`v2_lite_s` (SARVO-Lite)** | **11.016** | **32.55** |
| `v2_lite_p35_s` | 10.858 | 24.24 |
| `v2_lite_cond_s` | 11.093 | 32.55 |
| `v2_lite_n` | 3.043 | 11.71 |
| `v2_lite_m` | 22.541 | 97.18 |

Reproduce with:

```bash
.venv/bin/python -m saryolo bench --variants baseline_s v2_full v2_full_p35_s \
    v2_lite_s v2_lite_p35_s v2_lite_cond_s v2_lite_n v2_lite_m
```

## 3. NOT reproducible on this machine

| What | Why | What is needed |
| --- | --- | --- |
| Any accuracy number (mAP50, mAP50:95, AP_small) | no GPU, no real dataset | GPU + SSDD (pilot) per `docs/RUNBOOK_SSDD.md` |
| Cross-sensor LOSO result | same | GPU + a multi-source dataset (SARDet-100K or SSDD+HRSID) |
| Cross-resolution result | same, **plus** HRSID ships no per-chip resolution mapping | GPU + a *sourced* HRSID scene sidecar |
| Robustness sweep | needs a trained checkpoint | GPU run first |
| Representation-probe diagnosis | needs a trained checkpoint | GPU run first |
| RT-DETR cross-architecture result | training-loop integration unfinished | integration + GPU run |
| LoRA / adaptation comparison | baseline not implemented | build it (no GPU needed), then a GPU run |

## 4. The existing ledger contains no research result

`results/experiments.jsonl` holds 14 rows, all `SMOKE-*` runs: 2 epochs at 128 px on
**synthetic Gamma-speckle** data, reporting `mAP50 = 0.0000`. They exist to prove the
pipeline runs end to end. They are **not** results and must never appear in a table. The
README and the table generators both exclude `SMOKE-*` from experiment counts.

## 5. Status summary

| Category | Reproduced here | Blocked |
| --- | --- | --- |
| Infrastructure correctness | ✅ all | — |
| Architecture cost | ✅ all | — |
| Accuracy / generalisation | ❌ none | GPU + dataset |
| Adaptation comparison (LoRA) | ❌ none | baseline build + GPU |

The honest one-line status: **the instrument is verified; the measurement has not been
taken.**
