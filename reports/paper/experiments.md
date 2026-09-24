# 4 Experiments — skeleton

**Already fixable now** (protocol is code, not results):

## 4.1 Datasets
- SSDD (primary, source axis) — split/fold protocol in `docs/RUNBOOK_SSDD.md`.
- HRSID (secondary) — resolution axis requires a sourced sidecar
  (`docs/RUNBOOK_HRSID.md`); **no per-image resolution is invented**.
- SARDet-100K (scale-up, pending download).

## 4.2 Protocol
- Leave-one-source-out folds, generated once and committed
  (`saryolo/data/groups.py`); a fold refuses a random split.
- Identical train/val/test across every compared arm; the ledger pins
  `config_hash` + environment per run.
- Multi-seed: mean ± sample std, best/worst (EXP-012 arms).

## 4.3 Metrics
- mAP50, mAP50:95, precision, recall, AP-small/medium/large (COCO protocol),
  params (M), GFLOPs, FPS, latency.
- Generalisation gap = in-domain − LOSO, reported per source.

## 4.4 Controls (what a reviewer will ask for)
- Capacity-matched arm (`v2_full_p35_s`) — parameters are never a confound.
- Metadata withholding curve: all → sensor → sensor+resolution → none.
- Baseline reproduces stock YOLO11 bit-exactly (tested).

## 4.5 Implementation details
- Ultralytics 8.4.155, PyTorch 2.x, imgsz 512 first, AMP, fixed seeds 0/1/2,
  100 epochs + patience 30 for full arms (per generated EXP configs).

**BLOCKED-ON-RUN:** 4.6 (hardware actually used), 4.7 (any measured number).
