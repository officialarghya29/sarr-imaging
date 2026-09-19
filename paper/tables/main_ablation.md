**Main ablation. Each row adds exactly one component over the row above, so every delta is attributable to that component alone. The clutter row is a mode change on the speckle slot rather than an added module.**

| Model | SFE | Clutter | Attention | AMF | P2 head | SAR loss | Prior | Frequency | Context | Refine | mAP50 | mAP50:95 | Params (M) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO baseline |  |  |  |  |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + SFE | \checkmark |  |  |  |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + Speckle | \checkmark |  |  |  |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + Attention | \checkmark |  | \checkmark |  |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + AMF | \checkmark |  | \checkmark | \checkmark |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + Small head | \checkmark |  | \checkmark | \checkmark | \checkmark |  |  |  |  |  | TBD | TBD | TBD | TBD |
| Full (v1) | \checkmark |  | \checkmark | \checkmark | \checkmark | \checkmark |  |  |  |  | TBD | TBD | TBD | TBD |
| + Clutter-aware | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark |  |  |  |  | TBD | TBD | TBD | TBD |
| + Target prior | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark |  |  |  | TBD | TBD | TBD | TBD |
| + Spatial-frequency | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark |  |  | TBD | TBD | TBD | TBD |
| + Context | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark |  | TBD | TBD | TBD | TBD |
| Full (v2) | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | TBD | TBD | TBD | TBD |

`TBD` = not yet measured (run the corresponding experiment).
