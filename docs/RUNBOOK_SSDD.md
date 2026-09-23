# Runbook: the first measured number (SSDD)

The single most important thing this repository still lacks is **one real accuracy
number**. This runbook produces it on the smallest pilot dataset, end to end, and
the go/no-go below decides whether the paper's central claim lives or changes.

## Why SSDD first

SSDD is the cheapest honest start (registry: `ssdd`, 1,160 images, one `ship`
class, VOC boxes, 512 px):

- a full ablation grid fits inside one free Colab GPU session;
- the official release states per-image **inshore/offshore** scenery tags and
  mixed Gaofen-3 / Sentinel-1/RadarSat-2 acquisition — enough structure for both
  a cross-scenery probe and a conditioning pilot;
- the cs231n mirror vendors it with on-disk verification counts (1,160 images,
  four label variants), so there is no scraping step.

HRSID (5,604 images, 0.5/1/3 m, Sentinel-1B/TerraSAR-X/TanDEM-X) is the second
rung: it is where the conditioning adapter first has a real three-way resolution
and sensor spread. SARDet-100K is the benchmark tier and comes last.

## Stage 1 — Acquire and prepare (CPU)

```bash
# Download SSDD (official repo, or the cs231n mirror which vendors it) to ./raw/SSDD
python scripts/prepare_dataset.py --dataset ssdd --raw ./raw/SSDD
# -> datasets/processed/ssdd/{images,labels}/{train,val,test} + data.yaml
```

The converter emits horizontal 5-field boxes. SSDD ships four label variants; use
`BBox_SSDD` (horizontal boxes) — the OBB variants produce 9-field rows that
`check-data` will refuse.

## Stage 2 — Audit before any GPU minute (CPU, mandatory)

```bash
python -m saryolo.cli check-data --dataset datasets/processed/ssdd --classes ship
python -m saryolo.cli stats      --dataset datasets/processed/ssdd --classes ship --name ssdd
```

`check-data` exits non-zero on any structural problem. Do not book GPU time
before it is clean: a corrupted image or an unreadable label discovered after
training means a wasted run and a ledger row that cannot be compared with
anything.

## Stage 3 — Cross-scenery folds (the failure measurement)

SSDD tags every image as inshore or offshore. Scenery is *not* sensor, but it is
the strongest shift available inside one dataset, and the machinery takes any
stated grouping key:

```bash
python -m saryolo.cli loso --images datasets/processed/ssdd/images \
    --rule regex --pattern '^(.*)_[0-9]+' --data configs/datasets/ssdd.yaml \
    --min-test-images 30 --out datasets/splits/ssdd_scenery
```

**State the pattern from the archive's own naming, and verify the grouping
summary before proceeding**: two groups (inshore / offshore), no unmatched
images, both ≥ `--min-test-images`. If the pattern is wrong the command refuses
rather than building one group — that refusal is the guard doing its job.

If SSDD's filenames do not encode scenery in your release, use the release's own
split files as a sidecar mapping (`--rule sidecar`) — never a guessed pattern.

## Stage 4 — Acquisition metadata (CPU)

```bash
python -m saryolo.cli metadata --images datasets/processed/ssdd/images \
    --sidecar ssdd_acquisition.csv --out datasets/metadata_ssdd.json
```

The sidecar is a per-image CSV you build from the archive's own documentation
(columns: `image`, plus `sensor`, `resolution_m`, `polarization`, `band`,
`incidence_deg` where known; unknown → blank, never guessed). For SSDD the
sensor column comes from the release's stated sensor mix; `write_acquisition_metadata`
in `saryolo/data/convert.py` builds this table automatically when the dataset has
a sourced profile — use that path where available.

## Stage 5 — Baseline + LOSO training (GPU)

```bash
# In-domain reference
python -m saryolo.cli train --exp configs/exp/EXP-001_baseline.yaml

# One fold to prove the pipeline, then all folds
python -m saryolo.cli train --exp datasets/splits/ssdd_scenery/<fold>/EXP-901.yaml  # if generated
# else train directly on the fold config:
python -m saryolo.cli eval --weights runs/EXP-001/weights/best.pt \
    --data datasets/splits/ssdd_scenery/<fold>/eval_holdout.yaml
```

The fold's `eval_holdout.yaml` points `val` at the held-out scenery — **that** is
the number to report. `data.yaml`'s `val` shares sources with training and exists
for early stopping only.

## Stage 6 — Conditioning arms (GPU, after the failure is measured)

Only run these once Stage 5 shows a measurable held-out drop. If in-domain and
held-out numbers are close, there is no failure to fix and the conditioning
claim is unmotivated — which is itself a finding.

```bash
python scripts/make_exp_configs.py --dataset ssdd   # regenerate with the fold datasets
# control vs the transferable arm:
python -m saryolo.cli train --exp configs/exp/EXP-331_conditioning_none.yaml
python -m saryolo.cli train --exp configs/exp/EXP-334_conditioning_film_sensor_resolution.yaml
```

`cond_continuous` (film-physical-only) is the arm that can reach an unseen
acquisition, because it reads only physically-ordered fields. If only the
sensor-embedding arm helps, the method specialises to known sensors — report
that honestly.

## Stage 7 — Representation probe (CPU, on the trained checkpoint)

```bash
python -m saryolo.cli probe --weights runs/EXP-001/weights/best.pt \
    --data datasets/processed/ssdd/data.yaml --field sensor --split val --out results/probes/ssdd
```

Reads (§14 of the master plan): a `sensor` probe far above chance with a weak
`class` probe says the representation is appearance-dominated — the evidence that
justifies the conditioning adapter, and later the invariant branch.

## Decision gate

| Result | Meaning | Action |
| --- | --- | --- |
| In-domain high, held-out low | **The failure exists** — proceed as planned | Stage 6 conditioning arms, then representation probe on both models |
| In-domain ≈ held-out | No measurable scenery/sensor shift at this scale | Move to HRSID's three resolutions before concluding; SSDD may be too homogeneous |
| Probe: sensor ≫ chance, class weak | Hypothesis supported at the representation level | Justifies the invariant branch (§15–19) — build it only now |
| Probe: sensor ≈ chance | Hypothesis not supported on this data | **Do not build the invariant branch.** Reframe: the paper becomes the benchmark + failure analysis |

## Cost envelope

| Stage | Compute | Wall time (T4-class) |
| --- | --- | --- |
| 1–4 | CPU | < 1 h |
| 5, baseline ×1 | 1 GPU session | 1–2 h at 512 px, batch 16 |
| 5, all folds | k GPU sessions | k × baseline |
| 6, two conditioning arms | 2 GPU sessions | ≈ baseline each (<0.5% param overhead) |
| 7 | CPU | minutes |

## Failure modes already guarded against

- a grouping rule that keys on `train/val/test` is refused (in-domain wearing a
  cross-source label);
- images the rule cannot key are refused by default (silent test-set shrinkage);
- an evaluation split with zero ground truth raises before inference;
- a held-out source below `--min-test-images` is refused (noise as evidence);
- the ledger records init stage, seed and config hash, so a number can always be
  traced to the run that produced it.
