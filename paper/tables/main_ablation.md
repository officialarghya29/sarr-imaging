**Main ablation. Each row adds exactly one component, so every delta is attributable to that component alone.**

| Model | SFE | Speckle | Attention | AMF | P2 head | SAR loss | mAP50 | mAP50:95 | Params (M) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO baseline |  |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + SFE | \checkmark |  |  |  |  |  | TBD | TBD | TBD | TBD |
| + Speckle | \checkmark | \checkmark |  |  |  |  | TBD | TBD | TBD | TBD |
| + Attention | \checkmark | \checkmark | \checkmark |  |  |  | TBD | TBD | TBD | TBD |
| + AMF | \checkmark | \checkmark | \checkmark | \checkmark |  |  | TBD | TBD | TBD | TBD |
| + Small head | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark |  | TBD | TBD | TBD | TBD |
| Full | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | \checkmark | TBD | TBD | TBD | TBD |

`TBD` = not yet measured (run the corresponding experiment).
