# Committed model weights

`sarvo_ssac001.pt` is the one checkpoint this repository commits, and it is here for a specific
reason: a hosted instance clones the repository and has no other way to obtain a training
artefact, because `results/**` is excluded from git. Without it the deployed demo would correctly
report that no checkpoint is configured — and show nothing.

Everything else is still ignored. The applicable `.gitignore` rules are `*.pt` plus this single
negation, so exactly this file can be committed and no other weight accidentally can.

## What the file is

| | |
| --- | --- |
| Arm | `SSAC-001` — the Scatter-Selective Adaptive Computation arm (`configs/exp/SSAC-001_hrsid_ssac.yaml`) |
| Parameters | 3,396,329 |
| Size | 7,096,137 bytes (6.77 MB) |
| Classes | one (`ship`) |
| Trained on | the 200/60/60 HRSID subset, 320 px, 40 epochs, batch 4, seed 0, CPU only |
| Measured mAP50 / mAP50:95 | 0.5626 / 0.3089 (pilot — see `paper/RESULTS.md`) |
| SHA-256 | `c81287334beba49c27b13682fe1120ab5c025743dc1a8fe41c5c7f13f45a98a5` |

Verify the committed bytes are the measured ones:

```bash
sha256sum weights/sarvo_ssac001.pt
```

## What it is not

This is a **pilot** checkpoint on a subset of one dataset. It is not a benchmark result, and its
accuracy claim for the adaptive allocation is withdrawn (`docs/claim_evidence_audit.md`). Report
it with the caveats that accompany every number in `paper/RESULTS.md`.

## Reproducing it

```bash
python -m saryolo train --exp configs/exp/SSAC-001_hrsid_ssac.yaml
# writes results/runs/SSAC-001/weights/best.pt; copy it here to refresh the committed artefact
```

The demo resolves weights in the order given in `docs/DEPLOYMENT.md`: an explicit path, then
`SARVO_WEIGHTS`, then this file, then `results/runs/SSAC-001/weights/best.pt`.
