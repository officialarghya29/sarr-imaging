# Reproduction status — what can and cannot be reproduced on this machine

**Date:** 2026-10-03
**Rule applied:** a result is "reproduced" only if the command was actually executed and
its output observed. Installing code, or intending to run it, is not reproduction.

---

## 1. Reproducible now, without a GPU or a dataset

Every item below was run and observed in this environment.

| What | Command | Observed result |
| --- | --- | --- |
| Test suite | `.venv/bin/python -m pytest -q` | **596 passed** |
| Baseline parity with stock YOLO11 | `pytest tests/test_arch.py -k baseline` | exact published counts (n: 2,624,080; s: 9,458,752 at 80 classes) |
| All variants build and forward | `pytest tests/test_arch.py -k every_variant` | 110 variants pass |
| Module identity at init | `pytest tests/test_arch.py -k identity` | `max\|f(x)−x\| = 0.0e+00` for all |
| Architecture cost benchmark | `python -m saryolo bench --variants v2_full v2_lite_s ...` | see §2 |
| Efficiency profiling | `pytest tests/test_efficiency.py` | 29 passed |
| Paper tables generate | `python -m saryolo assets` | 20 files written (10 tables × md/tex); every *benchmark* accuracy cell `TBD`, the real-data pilot table complete |
| README charts regenerate | `python scripts/make_readme_assets.py` | 10 SVGs + PNG twins + `facts.json`, byte-reproducible |
| Real-input efficiency | `python -m saryolo efficiency --real --runs 3 …` | 5 checkpoints profiled; front end costs 3.0× baseline latency |

## 2. Measured cost (architecture, no training required)

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

## 3. Measured on real SAR data (pilot)

Three arms, one real dataset, one schedule. Every number below was produced by a run on this
machine and written into `results/experiments.jsonl` by the run itself; `docs/assets/facts.json`
is generated from that ledger, and the README's copy of the table is checked against the facts by
a test, so a number cannot be typed in by hand and survive.

**Data.** The official **HRSID** release (SAR ship, horizontal boxes), fetched from the
`dronefreak/HRSID` mirror in YOLO format and assembled into a subset: **200 train / 60 val / 60
test** images. The full release is 5,604 images. This is a pilot on a fraction of it.

**Schedule.** 40 epochs, 320 px, batch 4, seed 0, `deterministic`, one class, CPU only
(no GPU on this machine). The arms share all of it, so a row differs from its neighbours
only in what is trained.

| Arm | What is trained | mAP50 | mAP50:95 | Precision | Recall | Params (M) | GFLOPs@320 | FPS | Train (min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL-001 | stock YOLO11n, every parameter | 0.5706 | 0.3012 | 0.9240 | 0.5380 | 2.590 | 1.613 | 61.80 | 6.95 |
| REAL-002 | SARVO-Lite (s) — full v2 minus the two heaviest compute slots | 0.5724 | 0.3287 | 0.8014 | 0.5263 | 11.016 | 8.106 | 12.39 | 36.22 |
| REAL-003 | YOLO11n + rank-8 LoRA adapter (0.27 M adapter params, 10.1 % of the model trainable) | 0.5781 | 0.2984 | 0.8682 | 0.5556 | 2.590 | 1.613 | 65.57 | 8.84 |
| REAL-004 | SARVO prototype: ratio-space CFAR front end (learned per-pixel gain) | 0.5691 | 0.3151 | 0.9104 | 0.5346 | 2.590 | 1.764 | 36.70 | 12.47 |
| REAL-005 | Prototype control: same statistic, fixed threshold, 0 learnable params | 0.4952 | 0.2465 | 0.7931 | 0.4737 | 2.590 | 1.728 | 44.14 | 7.31 |
| REAL-006 | Prototype control: matched-cost conv stem, params identical to REAL-004 | 0.5654 | 0.3021 | 0.9244 | 0.5146 | 2.590 | 1.649 | 44.87 | 10.03 |

**The proposal's own falsification test, extended past the first seed.** The arm (`REAL-004`)
was then repeated at seeds 1 and 2 alongside the baseline (`MSEED-B1`/`MSEED-C1`, `MSEED-B2`/`MSEED-C2`,
same schedule, `train_seed` only changing). mAP50:95, paired:

| Seed | Baseline | CFAR front end | Paired Δ |
| ---: | ---: | ---: | ---: |
| 0 | 0.3012 | 0.3151 | +0.0139 |
| 1 | 0.2804 | 0.3036 | +0.0232 |
| 2 | 0.2871 | 0.3036 | +0.0165 |
| mean ± std | 0.2895 ± 0.0106 | 0.3074 ± 0.0066 | **+0.0179** |

The front end leads the baseline on mAP50:95 at **all three seeds**, and the matched-cost conv
control (`REAL-006`) sits below it (0.3021), so the gain is not the 217 extra parameters. A
five-corruption robustness sweep (`saryolo robustness`, `results/robustness/REAL-001` and
`REAL-004`) is directionally consistent: mean relative mAP50:95 loss under speckle −15.6 %
(baseline) vs −10.7 % (prototype), under low contrast −32.0 % vs −13.1 %. The effect is small
and pilot-scale, and the front end costs ~40 % of CPU throughput; recorded as *survives both
falsifiers it named, no paper-grade benefit demonstrated*.

**SAR-appearance augmentation (`AUG-001…004`).** Four arms on the augmented HRSID splits (one or
two corruption draws per training image; clean val/test), to test whether training under the SAR
degradation model widens the front end's lead:

| Arm | Train images | mAP50 | mAP50:95 | Precision | Recall | Train (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AUG-001 · baseline, SAR-augmented x1 | 400 | 0.6060 | 0.3571 | 0.9548 | 0.5556 | 12.71 |
| AUG-002 · CFAR front end, SAR-augmented x1 | 400 | 0.6049 | 0.3638 | 0.9493 | 0.5478 | 24.36 |
| AUG-003 · baseline, SAR-augmented x2 | 600 | 0.6278 | 0.3898 | 0.9418 | 0.5848 | 16.50 |
| AUG-004 · CFAR front end, SAR-augmented x2 | 600 | 0.6160 | 0.3914 | 0.9224 | 0.5789 | 30.85 |
| AUG-005 · baseline, SAR-augmented x2, seed 1 | 600 | 0.6194 | 0.3892 | 0.9375 | 0.5673 | 35.14 |
| AUG-006 · baseline, SAR-augmented x2, seed 2 | 600 | 0.6137 | 0.3728 | 0.9168 | 0.5848 | 35.18 |

Augmentation is the largest accuracy movement measured here, and it keeps helping as it
strengthens — the baseline rises 0.3012 → 0.3571 → 0.3898 mAP50:95 (+29.4 % relative). But the
prototype's lead **shrinks monotonically** as augmentation grows (+0.0139 → +0.0067 → +0.0016),
so the hypothesis is **not supported**: the corruption model gives both arms much of the
robustness the statistic provided. See `paper/RESULTS.md` §3.

The two-view augmentation gain was then seed-checked (`AUG-005`/`AUG-006`, paired against
`MSEED-B1`/`MSEED-B2`): **+0.0886 / +0.1089 / +0.0857** mAP50:95 at seeds 0/1/2, mean
**+0.0944 ± 0.0126** — positive at every seed and ~9× the seed spread of either arm
(0.3839 ± 0.0097 augmented vs 0.2895 ± 0.0106 clean).

Real-input cost (`saryolo efficiency --real --runs 3`, imgsz 320, batch 8; median of three
blocks, measured back-to-back in one quiet interval): baseline 113.1 FPS / 8.85 ms,
`SARVO-Lite (s)` 13.7 FPS / 72.88 ms, front end 37.2 FPS / 26.90 ms, augmented baseline
101.2 FPS / 9.89 ms. The front end's **3.0×** latency penalty is a latency cost, not compute
(+9 % FLOPs).

Reproduce with:

```bash
python -m saryolo train --exp configs/exp/REAL-001_hrsid_baseline.yaml
python -m saryolo train --exp configs/exp/REAL-002_hrsid_v2_lite.yaml
python -m saryolo train --exp configs/exp/REAL-003_hrsid_lora_r8.yaml
python -m saryolo train --exp configs/exp/REAL-004_hrsid_cfar.yaml
python -m saryolo train --exp configs/exp/REAL-005_hrsid_cfar_fixed.yaml
python -m saryolo train --exp configs/exp/REAL-006_hrsid_cfar_conv.yaml
python -m saryolo augment --data configs/datasets/hrsid_real.yaml --views 1 \
    --out datasets/processed/hrsid_real_aug        # then copy val/test from hrsid_real
python -m saryolo train --exp configs/exp/AUG-001_hrsid_aug_baseline.yaml
python -m saryolo train --exp configs/exp/AUG-002_hrsid_aug_cfar.yaml
python -m saryolo augment --data configs/datasets/hrsid_real.yaml --views 2 --out datasets/processed/hrsid_real_aug2
python -m saryolo train --exp configs/exp/AUG-003_hrsid_aug2_baseline.yaml
python -m saryolo train --exp configs/exp/AUG-004_hrsid_aug2_cfar.yaml
python -m saryolo train --exp configs/exp/AUG-005_hrsid_aug2_baseline_s1.yaml
python -m saryolo train --exp configs/exp/AUG-006_hrsid_aug2_baseline_s2.yaml
python -m saryolo efficiency --weights results/runs/REAL-004/weights/best.pt \
    --imgsz 320 --real --data configs/datasets/hrsid_real.yaml --batch 8 --runs 3 \
    --out results/efficiency/REAL-004
python scripts/make_readme_assets.py     # regenerates docs/assets/facts.json + the charts
python scripts/make_real_figures.py      # regenerates the qualitative detection panel
```

**What the pilot does not support.** Nothing here is a cross-sensor, cross-resolution or
leave-one-source-out result: HRSID is one source, and the subset is small. The primary
comparison is three seeds on one 200-image subset, which is a noise check and not validation.
The ladder ablations (SFE, SFM, SAA, AMF, P2, conditioning) are still `TBD` and still need a GPU.

**What the LoRA arm measured, and where the adapter is.** The arm trains a low-rank update to
the wrapped convolutions plus the detection head; the wrapped base weights are frozen. The
checkpoint that the run writes is the *adapted* model with the update folded into the base
weights, because an Ultralytics checkpoint loader fuses convolutions with the following
BatchNorm and cannot load a wrapper (measured: `'LoRALayer' object has no attribute 'weight'`,
in Ultralytics' own end-of-run validation of the file it had just written). That is why its
parameter count in the table equals the baseline's: the parameter-efficiency is in *what was
optimised*, which the ledger records separately (`lora_params`, `fraction_trainable`,
`n_wrapped_layers`).

## 4. `BLOCKED-ON-RUN`: not reproducible on this machine

Everything in this section is **not yet** measured for a stated reason, and each row names what
would unblock it. The rule is that a blocked item is written down rather than omitted: an absent
row reads as a finished one.

| What | Why | What is needed |
| --- | --- | --- |
| **The paper's accuracy table** (full SSDD/HRSID/SARDet-100K, the ablation ladder, AP_small) | the pilot in §3 is a *subset* on a CPU; the ladder needs the full benchmark | GPU + `docs/RUNBOOK_SSDD.md` |
| Cross-sensor LOSO result | same | GPU + a multi-source dataset (SARDet-100K or SSDD+HRSID) |
| Cross-resolution result | same, **plus** HRSID ships no per-chip resolution mapping | GPU + a *sourced* HRSID scene sidecar |
| Robustness sweep | needs a checkpoint trained on the full schedule | GPU run first |
| Representation-probe diagnosis | same | GPU run first |
| RT-DETR cross-architecture result | training-loop integration unfinished | integration + GPU run |

## 5. The ledger

`results/experiments.jsonl` holds the pilot runs of §3 plus a set of `SMOKE-*` runs: 2 epochs at
128 px on **synthetic Gamma-speckle** data. The smoke runs exist to prove the pipeline runs end
to end, they are **not** results, and the generator that counts experiments excludes them by id.
One more id is deliberately present and deliberately empty of meaning: `SMOKE-LORA`, the arm
`tests/test_peft.py` trains on synthetic data to prove the adapter path end to end.

### Rows that were removed, and why

The first version of this ledger held two real-data rows that were deleted rather than kept,
because keeping them would have been the most expensive kind of dishonesty -- a plausible number
for something that never happened:

* **`REAL-001` measured on the 87/9/0 subset.** Superseded: the subset it trained on was later
  enlarged to 200/60/60, so its number is not comparable with anything measured since.
* **`REAL-003` recorded as a parameter-efficient arm.** Its ledger entry claimed a rank-8 adapter
  over 87 wrapped layers and a 10 % trainable budget. Inspecting the checkpoint showed **zero**
  `lora_` tensors: Ultralytics' loader rebuilds a model from its config inside the trainer, so the
  adapter had been injected into a facade that was thrown away before the first step, and the run
  was a full fine-tune of the baseline graph. The tell is in the numbers: that run's mAP is
  *bit-identical* to `REAL-001`'s (0.3011883083461638), which is exactly what a deterministic
  full fine-tune of the same graph, data, seed and schedule produces. The adapter is now applied
  by the trainer that trains, and the row was measured again (below).

Both removals are recorded here because an append-only ledger whose deletions are invisible would
be a worse record than one that keeps its mistakes: the mistake is documented, the fix is in
`saryolo/training/trainer.py` with the failure named in the comment, and the regression test that
would have caught it is `tests/test_peft.py::test_a_lora_arm_end_to_end_trains_adapters_and_writes_a_usable_checkpoint`.

## 6. Status summary

| Category | Reproduced here | Blocked |
| --- | --- | --- |
| Infrastructure correctness | ✅ all | — |
| Architecture cost | ✅ all | — |
| Accuracy on real SAR | ✅ pilot, 3 arms, subset of HRSID | full benchmark + ladder |
| Adaptation comparison (LoRA) | ✅ pilot, one rank | rank/scale sweep |
| Generalisation (LOSO, cross-sensor, cross-resolution) | ❌ none | GPU + multi-source dataset |

The honest one-line status: **the instrument is verified, and a real pilot has now been measured
on a subset -- which is a beginning, not the paper's result.**
