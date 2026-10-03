# Research questions

Each question is stated so it can be **answered negatively**, names the experiment that
answers it, and names the control that would make a positive answer unconvincing. A
question whose experiment has not run is marked `OPEN`; none is marked answered.

---

## RQ1 — Does the baseline actually degrade across acquisition sources?

**Question.** Under a guarded leave-one-source-out split, does a stock detector's mAP drop
measurably relative to in-domain, and does the drop concentrate in particular sources?

**Why it gates everything.** If the baseline does *not* degrade, there is no problem to
solve and the conditioning adapter is unmotivated. This is the first number the project
must produce.

**Experiment.** `EXP-001` (baseline) under LOSO folds from `saryolo/data/groups.py`,
compared against the in-domain `EXP-001` number.

**Control.** A random-split run of the same baseline, to separate "source shift" from "a
smaller training set". The fold machinery refuses a random split for the *claim*; the
random split is run only as the contrast.

**Status.** `OPEN` — **BLOCKED-ON-RUN** (no GPU, no real dataset).

---

## RQ2 — Is the degradation a *representation* problem, or a data-volume problem?

**Question.** On frozen features, does a linear probe recover *sensor/source* far above
chance while a probe for *class* does not?

**Why it matters.** A positive sensor probe is evidence that acquisition appearance is
entangled with the representation — the exact failure the adapter addresses. A negative
result means the failure is elsewhere (fewer examples, not confounded features) and the
adapter should not be built.

**Experiment.** `saryolo.evaluation.probes` on the trained `EXP-001` checkpoint:
probe accuracy per level, within-class centroid drift across acquisition groups, linear
CKA between acquisition groups.

**Control.** Probes are compared against the **chance rate implied by the group count**,
not against 1.0 — a probe at 0.9 with four groups is informative; at 0.9 with 90 groups it
is not.

**Status.** `OPEN` — tooling built and tested; needs a trained checkpoint.

---

## RQ3 — Does acquisition conditioning recover part of the cross-source gap?

**Question.** Does `cond_continuous` — reading only resolution, band and incidence, the
fields that exist for *any* sensor — beat the `v2_full` control under LOSO, at <0.5 %
parameter overhead?

**Why it is the headline.** This is the one arm that can transfer to a sensor with no
embedding row, which is what the cross-sensor claim requires.

**Experiment.** `EXP-331…338` under the same LOSO folds, with `v2_full` as the control.

**Control.** `cond_sensor` (categorical only). If it succeeds where `cond_continuous`
fails, the honest conclusion is that the method specialises to *known* sensors — a
negative result the slot is instrumented to detect, not to hide.

**Status.** `OPEN` — **BLOCKED-ON-RUN**.

---

## RQ4 — Does the model still work when metadata is missing?

**Question.** When acquisition fields are withheld at encode time, how far does detection
degrade — and does a withheld field behave exactly like a never-recorded one?

**Why it matters.** A deployment often has no metadata table. A method that requires it is
not deployable; a method that silently assumes `resolution = 0.1 m` is worse.

**Experiment.** `metadata_fields` restriction in the data config (or `loso
--metadata-fields`), swept from all fields → sensor only → sensor+resolution → none.

**Control.** A withheld field is encoded *identically* to a never-recorded field (value,
availability flag and categorical id all masked) — pinned by `tests/test_field_mask.py`,
so the study measures real degradation rather than a fabricated value.

**Status.** `OPEN` — masking semantics built and tested; the degradation curve needs a run.

---

## RQ5 — Is the conditioning mechanism architecture-general?

**Question.** Does the same adapter improve cross-source detection when inserted into an
RT-DETR graph rather than a YOLO graph?

**Why it matters.** "One architecture" is a standard reviewer objection; a second family
turns a single result into a mechanism claim.

**Experiment.** `SARYOLORTDetectionModel` under LOSO, compared against stock RT-DETR.

**Control.** Gate-closed RT-DETR must be bit-identical to stock at init (already verified),
so any difference is learned.

**Status.** `OPEN` — graph integration verified; **training-loop integration unfinished**,
so no number can exist yet.

---

## RQ6 — Is the efficiency claim real, and at what operating point?

**Question.** Can the detector reach a materially lower compute cost at a *stated* accuracy
point, compared with both its own reference and the published field?

**Why it is answerable now.** Cost is measurable without a GPU.

**Experiment.** `EXP-501…505`; the frontier table combines measured cost with *cited*
published baselines kept in a separate section.

**Control.** The light arms must drop **only** the two declared compute slots and must be
strictly cheaper than their **scale-matched** reference — both asserted by test, so the
saving can never come from silently deleting a physical prior.

**Status.** **Cost answered and measured** (11.016 M / 32.55 GFLOPs vs 16.230 M /
55.68 GFLOPs at scale `s`). The *accuracy-at-cost* half is `OPEN` — **BLOCKED-ON-RUN**.

---

## RQ7 — Does the method hold in the low-label regime?

**Question.** At 1 %, 5 %, 10 %, 25 % and 50 % of the labelled training data, does
conditioning help more than it does at 100 %?

**Why it matters.** This is master-context Direction E, and it is where a
*representation* fix should pay off most: with fewer examples, a model leans harder on
acquisition-specific appearance.

**Experiment.** A data-fraction sweep with identical sampling across arms.

**Control.** The same sampling seed and the same folds across every fraction.

**Status.** `OPEN` — not implemented.

---

## Priority

| RQ | Blocks | Needs | Order |
| --- | --- | --- | --- |
| RQ1 | everything | GPU + dataset | 1 |
| RQ2 | whether to build the invariant branch | GPU (checkpoint) | 2 |
| RQ3 | the headline claim | GPU + folds | 3 |
| RQ4 | the deployability claim | GPU + folds | 4 |
| RQ6 | the efficiency claim's accuracy half | GPU + folds | 5 |
| RQ7 | the data-efficiency claim | GPU + fractions | 6 |
| RQ5 | architecture generality | integration + GPU | 7 |

The order is set by what unblocks the most, not by what is most interesting. RQ1 is a
single baseline run and it gates the entire project.
