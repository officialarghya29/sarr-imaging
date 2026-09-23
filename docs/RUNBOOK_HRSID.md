# Runbook: the cross-resolution pilot (HRSID)

Where SSDD (`docs/RUNBOOK_SSDD.md`) answers *"does the detector degrade across
scenery at all?"*, HRSID is the first dataset where the **resolution** axis is
real: the official release states three resolutions — **0.5 m, 1 m and 3 m** —
across **Sentinel-1B, TerraSAR-X and TanDEM-X** acquisitions. That is exactly the
spread the cross-resolution protocol (Experiment D) and the conditioning arms
were built to measure.

## Why HRSID is the resolution pilot

- **Three stated resolutions** span a 6× range — the same spread the SARDet-100K
  sources span, in a dataset one-fortieth the size.
- **Three stated resolutions and a scene list**: the release names its sensors
  and resolution tiers, which is exactly the structure the cross-resolution
  protocol needs — once a per-image mapping is supplied through a sidecar, the
  `sensor` and `resolution` fields are *stated values*, not inferred ones.
- **Two pilot roles at once**: the false-alarm probe (the 400 background-only
  images shipped separately) and the conditioning experiments share one dataset.

## Stage 1 — Acquire and prepare (CPU)

```bash
# Official repo (chaozhong2010/HRSID) or the cs231n mirror, which vendors the
# full JPG release plus COCO splits (train 3,642 / test 1,962)
python scripts/prepare_dataset.py --dataset hrsid --raw ./raw/HRSID
```

The COCO converter emits the standard `images/<split>` / `labels/<split>` layout;
5,604 images, single `ship` class.

## Stage 2 — Audit (CPU, mandatory)

```bash
python -m saryolo.cli check-data --dataset datasets/processed/hrsid --classes ship
python -m saryolo.cli stats      --dataset datasets/processed/hrsid --classes ship --name hrsid
```

HRSID chips are cut from a few large scenes, so a random chip split **leaks**:
near-duplicate chips of the same acquisition land on both sides. The leakage
report exists for exactly this — run it before trusting any number:

```bash
# After Stage 3 produces folds, verify no chip crosses splits:
python -m saryolo.cli loso --images datasets/processed/hrsid/images \
    --rule <see Stage 3> --leakage ...
```

## Stage 3 — Acquisition metadata: verified values only (CPU)

HRSID's public release states three resolutions (0.5/1/3 m) and several sensors,
but does **not** ship a verified scene-to-chip mapping, so the registry's HRSID
profile intentionally carries no per-image sensor or resolution values. The
command still works — it is the route for any sidecar you can source — but a
profile-only table is refused:

```bash
# Refused: no documented acquisition values matched; nothing is invented.
python -m saryolo.cli metadata --images datasets/processed/hrsid/images \
    --dataset hrsid --out datasets/metadata_hrsid.json
```

To build a usable table, source per-image values from the archive itself (its
scene list or release notes) into a CSV/JSON sidecar keyed by image stem:

```bash
python -m saryolo.cli metadata --images datasets/processed/hrsid/images \
    --dataset hrsid --sidecar hrsid_per_image.csv --out datasets/metadata_hrsid.json
```

Every value in the table is traceable: the table's `source` field records the
profile plus the sidecar used, and fields the archive does not state per image
stay null. A range or a sensor mix is a per-dataset summary, not a per-image
label, and it is never converted into one — a fabricated midpoint would create
false resolution bins and quietly poison the cross-resolution claim.

## Stage 4 — Cross-resolution folds (Experiment D)

```bash
python -m saryolo.cli loso --images datasets/processed/hrsid/images \
    --rule resolution --metadata datasets/metadata_hrsid.json \
    --edges 0.75,2.0 --data configs/datasets/hrsid.yaml \
    --min-test-images 30 --out datasets/splits/hrsid_crossres
```

With edges `0.75,2.0` the three stated tiers fall into three bins:
`resolution_m<=0.75` (0.5 m), `0.75<resolution_m<=2.0` (1 m), `resolution_m>2.0`
(3 m) — one held-out fold each. The edges are a modelling choice and are
deliberately required on the command line; state them from the dataset's own
resolution tiers, never leave them defaulted.

The fold guard that matters here: if every image carried the same resolution
value, the command **refuses** with "single bin" — a cross-resolution run over
one resolution would report a number measuring nothing of the kind.

## Stage 5 — Train and measure (GPU)

```bash
# In-domain reference (one seed, then EXP-012's seeds for the final table)
python -m saryolo.cli train --exp configs/exp/EXP-001_baseline.yaml

# Each fold: train on two resolution bins, evaluate on the held-out one
for fold in datasets/splits/hrsid_crossres/*/; do
    python -m saryolo.cli train --data "$fold/data.yaml"  --name "$(basename "$fold")"
    python -m saryolo.cli eval  --weights "runs/$(basename "$fold")/weights/best.pt" \
                                --data "$fold/eval_holdout.yaml"
done
```

`eval_holdout.yaml` points `val` at the held-out resolution bin — **that** is
the reported number. Report scale-wise AP alongside the overall figure: a
resolution shift that hides in mAP50 may be visible in `AP_small`, where the
paper's claim lives.

## Stage 6 — Conditioning arms on the resolution axis (GPU)

Only after Stage 5 shows a measurable cross-resolution drop:

```bash
python -m saryolo.cli train --exp configs/exp/EXP-331_conditioning_none.yaml          # control
python -m saryolo.cli train --exp configs/exp/EXP-333_conditioning_film_resolution.yaml  # continuous field
```

`film:resolution` is the arm the cross-resolution claim rests on: resolution is
a continuous physical field, so it transfers to a bin *and* a sensor never seen
in training. If only the categorical sensor arm helps, the method specialises to
known sensors — report that honestly, per the SSDD runbook's decision gate.

## Decision gate

| Result | Meaning | Action |
| --- | --- | --- |
| Baseline: high in-bin, low held-out-bin | The cross-resolution failure exists | Run the conditioning arms (Stage 6) |
| `film:resolution` recovers part of the drop on unseen bins | The central claim holds on the resolution axis | Extend to SARDet-100K sources; start the paper's Table 4 |
| Probe (Stage 7) shows resolution predictable ≫ chance at every level | Resolution is *entangled in the representation*, not just the head | Strengthens the §14 evidence chain |
| No measurable cross-resolution drop | HRSID's spread is too easy at 800 px | Test at 512 px input (harder); if still flat, the failure claim narrows to sensors — say so |

## Cost envelope

| Stage | Compute | Wall time (T4-class) |
| --- | --- | --- |
| 1–4 | CPU | < 1 h (COCO conversion is the slow part) |
| 5, baseline ×1 | 1 GPU session | 3–5 h at 800 px, batch 8 |
| 5, 3 folds | 3 GPU sessions | ≈ baseline each |
| 6, two arms | 2 GPU sessions | ≈ baseline each |
| 7 (probe, `saryolo probe --dataset hrsid`) | CPU | minutes |

## Relation to the other runbooks

- **SSDD** (`docs/RUNBOOK_SSDD.md`): first measured number; scenery-axis failure
  probe; smallest cost.
- **This file**: resolution-axis failure; the Experiment D protocol on real
  stated tiers; first dataset where `film:resolution` vs `film:sensor` decides
  which mechanism the paper leads with.
- **SARDet-100K** (next): source-level LOSO with the ten-source profile — the
  benchmark-tier headline experiment, only after both pilots support the claim.
