---
title: "Forest Pulse — Engineering Decisions: Evaluation Strategy"
project: forest-pulse
scope: Evaluation methodology, metric selection, label quality strategy
date: 2026-04-07
version: 1.0
status: Academic Documentation
phases_documented: "1–11"
author: Jordi Catafal
---

# Engineering Decisions — Evaluation Strategy

This document records the evaluation strategy decisions that define how model quality is measured in Forest Pulse. The central challenge: measuring quality honestly in the absence of real ground truth during early phases.

---

## EVAL-001: LiDAR-Verified Metric as the "Honest" Evaluation Standard

**Status:** Accepted  
**Date:** 2026-04-07  
**Commit:** `77833df`  
**File:** `autoresearch/eval_lidar.py`

### Context

After Phase 7, the project had:
- RF-DETR mAP50 = 0.904 (on self-labeled validation set)
- LiDAR point clouds that could provide physically-verified ground truth

The mAP50 of 0.904 was measured against the model's own training labels — a feedback loop. The model could not disagree with itself. Meanwhile, manual inspection showed real recall was only ~20–30% in dense canopy. The metric was **biased by design**, and Phase 11 (auto-research optimization) could not be trusted if it optimized toward a self-referential signal.

### Options Considered

| Option | Why Considered | Rejected Because |
|--------|---------------|-----------------|
| **LiDAR CHM local-maximum filtering** | Physical truth, not self-referential; standard forestry technique | **Selected** |
| Expand gold set with Roboflow | Human labeling is authoritative | Hard to scale; cannot reproduce overnight |
| Copernicus GLO-30 DSM | Free global DSM | 30 m resolution = too coarse for individual trees |
| Teacher model (OAM-TCD pretrained) | Transfer signal as external reference | Domain mismatch (US deciduous vs Mediterranean) |

### Decision

Implement `autoresearch/eval_lidar.py` that:
1. Derives ground-truth tree positions from CHM via local-maximum filtering (standard forestry technique).
2. Matches RF-DETR detections to truth positions using greedy nearest-neighbor within 2 m tolerance.
3. Reports precision, recall, F1 as independent metrics (micro-averaged across patches).

**Key parameters:**
- Height threshold: 5.0 m (consistent with `filter_by_height`; excludes shrubs)
- Local-max window: 3.0 m radius (~6 m diameter, typical Mediterranean crown)
- Smoothing: Gaussian sigma 1.0 px on 0.5 m CHM
- Matching tolerance: 2.0 m (Spanish Forest Inventory convention)
- Aggregation: micro-averaging (pool counts across patches before computing ratios)

### Rationale

- **Physically grounded:** errors come from canopy structure (merged crowns), not from training on self-labels.
- **Standard forestry technique:** published in lidR, Spanish Forest Inventory, USFS standard workflows.
- **Greedy matching:** <1% different from Hungarian matching at 2 m tolerance and 5+ m tree spacing. Deterministic, no optimization library dependency.
- **Enables Phase 11:** optimizing this metric = finding more real trees, not fitting self-labels.
- **Documents the gap:** inflated 0.904 vs honest ~30% recall documents the crown merging problem.

### Consequences

- **Recall ~30%** because tree-top detection misses merged crowns in dense uniform canopy. This is a *feature, not a bug* — it documents the hard problem and forces the project toward the ensemble approach (detect → height filter → classifier).
- Metric tied to LiDAR availability; areas without LAZ coverage cannot use this approach.
- Local-max filtering may produce >1 peak per irregular crown — documented limitation.
- `eval_lidar.py` becomes the PRIMARY optimization target for Phase 11.

---

## EVAL-002: eval.py Locked During Auto-Research Harness Runs

**Status:** Accepted  
**Date:** 2026-04-06 (implicit during Phase 2 implementation)  
**Commits:** `d2ba322`, `74961d0`  
**File:** `autoresearch/eval.py`

### Context

The auto-research harness (Phase 11) optimizes RF-DETR hyperparameters overnight in an agentic loop: modify `train.py` → train → measure → keep or revert → repeat. If the agent could also modify `eval.py`, it could "improve" by changing the benchmark rather than the model — e.g., lowering the IoU threshold or swapping the validation set. This violates the fundamental purpose of evaluation.

### Options Considered

| Option | Rejected Because |
|--------|-----------------|
| **Lock eval.py (architectural)** | **Selected** |
| Trust the agent to behave ethically | Even good agents fail when incentives are wrong |
| Remote measurement server | Adds infrastructure; defeats fast iteration |
| Cryptographic hash check | Overly complex; architectural lock is simpler |

### Decision

**eval.py is immutable during harness runs.**

Enforcement mechanisms:
1. **Explicit docstring warning** — `"""LOCKED evaluation script — DO NOT MODIFY during harness runs."""`
2. **Separate modules** — eval_gold.py and eval_lidar.py in separate files with independent logic.
3. **Hardcoded data paths** — VAL_DIR and VAL_ANNOT never change.
4. **Machine-readable output** — always prints `val_map50: X.XXXX` to stdout; harness greps this exact format.

The agent may **only modify `train.py`**, which defines the search surface: backbone, learning rate, batch size, epochs.

### Rationale

- Creates a clear boundary: `train.py` = editable (optimization target); `eval.py` = ground truth.
- Analogous to tournament rules: scoring function is fixed before competition starts.
- Simplest enforcement is architectural (separate locked file) rather than computational (hashing).
- Scales to multi-agent setups: multiple agents all optimize toward the same evaluation module.

### Consequences

- **Enables:** multi-round auto-research runs with trustable optimization signal.
- **Limitation:** eval.py itself is now obsolete for honest evaluation (EVAL-001). Phase 11 should migrate to eval_lidar.py as the optimization target.
- The docstring is a social contract, not a cryptographic guarantee — relies on agent discipline.

---

## EVAL-003: Gold Evaluation Set Strategy

**Status:** Accepted (Partially Superseded by EVAL-001)  
**Date:** 2026-04-06  
**Commit:** `6f1cccc`  
**Files:** `autoresearch/eval_gold.py`, `data/montseny/eval_gold/`

### Context

After Phase 2, the project had RF-DETR at mAP50=0.904 on self-labels, and no independent ground truth. A small human-verified set was created as a sanity check, before LiDAR integration was planned.

### Decision

Curate **20 diverse patches** across all 8 elevation zones with manual COCO annotations via Roboflow. Implement `eval_gold.py` reporting `gold_map50: X.XXXX`.

Patch selection criteria via `prepare_gold_eval.py`: all 8 zones represented, varied canopy density, varied illumination/shadow conditions.

### Rationale

- 20 patches is a sweet spot: ~200–500 annotations, annotatable in one afternoon (~2–4 hours).
- Human-verified annotations are >95% confident — highest-quality signal available.
- Serves as **mid-development sanity check**: if gold_map50 drops unexpectedly, something is wrong.
- Documents gap between inflated eval.py mAP50 and honest performance.

### Consequences

- **Static size (20 patches):** higher variance than larger evaluation sets.
- **Superseded by EVAL-001:** LiDAR metric scales to any number of patches automatically, with no human bias.
- Gold set remains useful for sanity checks and spot verification.

---

## EVAL-004: mAP50 as Primary Detection Metric

**Status:** Accepted  
**Date:** 2026-04-06 (implicit in eval.py implementation)  
**Files:** `autoresearch/eval.py`, `eval_gold.py`, `eval_lidar.py`

### Context

Multiple detection metrics exist. The project needed a metric that is standard, computable on sparse detections, and explainable to forestry stakeholders.

### Options Considered

| Metric | Rejected Because |
|--------|-----------------|
| **mAP50 (IoU ≥ 0.50)** | **Selected** |
| mAP75 | Too strict for crown boundaries (annotator disagreement often exceeds 0.25 IoU) |
| COCO mAP (0.50:0.95:0.05) | 10x computation; overkill for forest monitoring |
| Precision & Recall at fixed threshold | Threshold-dependent; mAP integrates across all operating points |
| Custom crown area overlap | Out of scope for detection; health scoring is a separate concern |

### Decision

**mAP50 (IoU ≥ 0.50)** via `supervision.metrics.MeanAveragePrecision` across all evaluation modules.

### Rationale

- **Industry standard:** COCO, Pascal VOC, KITTI all use mAP50. Results directly comparable to published tree detection papers.
- **Forgiving of small errors:** Tree crowns are irregular; annotator disagreement is often >25% bbox error. IoU 0.50 allows ~25% bbox mismatch.
- **Confidence-agnostic:** Integrates precision-recall curve across all operating points.
- **Already in dependencies:** supervision library provides this out of the box.

### Consequences

- mAP50 hides localization errors (0.51 IoU counts as TP even if crown boundary is imprecise).
- Scale-insensitive: large and small trees weighted equally by IoU — generally desirable for forestry.
- Direct comparison with academic literature on tree crown detection.

---

## EVAL-005: Multi-Round Self-Training Annotation Strategy

**Status:** Accepted  
**Date:** 2026-04-06  
**Commit:** `7f873ff`  
**Files:** `scripts/self_train.py`, `data/montseny/annotations_round_*.json`

### Context

Starting with 10,376 DeepForest weak labels (~60% precision), the project needed a mechanism to iteratively improve training label quality without manual annotation.

### Decision

**Three-round self-training loop** via `scripts/self_train.py`:

```
Round 0: DeepForest on 800 patches → annotations_raw.json (27K labels)
Round 1: Train RF-DETR → re-label at 0.5 confidence → annotations_round_1.json (~20K)
Round 2: Train RF-DETR → re-label at 0.5 confidence → annotations_round_2.json (~18K)
Round 3: Train RF-DETR → re-label at 0.5 confidence → annotations_round_3.json (~17K)
```

Each round: re-label all patches → prepare dataset → train from scratch (10 epochs) → measure quality → save checkpoint and annotations.

### Rationale

- **Domain adaptation:** RF-DETR trained on Montseny data outperforms DeepForest on Catalan forests from round 1.
- **Confidence as filter:** 0.5 confidence preserves diversity while filtering ambiguous detections.
- **Discrete rounds:** enable archiving labeled datasets per round (reproducibility + audit trail).
- **Not circular:** fresh model per round prevents overfitting to one model's biases; gold set sanity check.

### Consequences

- 37% noise reduction: 27K raw → 17K clean labels.
- RF-DETR on round_3 labels achieves mAP50 ≈ 0.85–0.90 (vs 0.4 on raw labels).
- **Metric still inflated:** evaluated on self-generated labels. Phase 8 (LiDAR measurement) reveals real recall is ~30%.
- Computational cost: ~4 hours total on Apple MPS for 3 rounds.

---

## EVAL-006: Bootstrap with Weak Labels Before Manual Annotation or LiDAR

**Status:** Accepted  
**Date:** 2026-04-05  
**Commit:** `6659f1b`

### Context

Three labeling approaches were available: weak labels (DeepForest), manual annotation (Roboflow), and LiDAR auto-labels (Phase 7). Which to use first?

### Decision

Use **DeepForest weak labels first** for all 800 patches.

**Ordering rationale:**
- Manual annotation: 2 weeks + €1000 — blocks iteration while awaiting annotators.
- LiDAR auto-labels: not planned until Phase 7 (3–4 weeks of infrastructure).
- **Weak labels:** available in minutes; enables immediate iteration; discovers real bottlenecks while infrastructure is built in parallel.

### Rationale

- **Speed & iteration:** By day 1, 27K annotations and a working RF-DETR model. By day 2, self-training had improved labels. LiDAR infrastructure built in parallel without blocking training.
- **Reversible:** manual annotation or LiDAR could be added later. In fact, all three coexist (weak labels → gold set → LiDAR).
- **Accelerated learning:** discovering mAP50=0.85 on self-labels is meaningless led to EVAL-001 as a better design.

### Consequences

- **mAP50 inflated:** evaluating on self-generated labels masks that real recall is ~30%.
- **Timeline acceleration:** by the time LiDAR measurement was ready (April 7), the team had already validated self-training and built the necessary infrastructure.

---

## Decision Timeline

```
April 5: EVAL-006 (weak labels) ← start Phase 3
April 5–6: EVAL-005 (self-training × 3) ← Phase 3 completion  
April 6: EVAL-004 (mAP50), EVAL-002 (lock eval.py), EVAL-003 (gold set) ← Phase 4
April 7: EVAL-001 (LiDAR metric) ← Phase 7 completion, unlocks Phase 11
```

---

## Summary Table

| Decision | Date | Status | Key Constraint | Consequence |
|----------|------|--------|----------------|-------------|
| EVAL-001 | Apr 7 | Accepted | CHM local-max filtering, 5 m threshold, 2 m match tolerance | PRIMARY metric for Phase 11 |
| EVAL-002 | Apr 6 | Accepted | Agent must never modify eval.py | Prevents metric gaming |
| EVAL-003 | Apr 6 | Accepted (Superseded) | 20 human-annotated patches | Sanity check; superseded by LiDAR |
| EVAL-004 | Apr 6 | Accepted | IoU ≥ 0.50, supervision MeanAveragePrecision | Standard metric, comparable to literature |
| EVAL-005 | Apr 6 | Accepted | 3 rounds, 0.5 confidence relabeling | 37% label noise reduction |
| EVAL-006 | Apr 5 | Accepted | Weak labels before manual/LiDAR | Fast iteration; discovers real bottlenecks |

---

## Implications for Phase 11 (Auto-Research)

The decision sequence enables honest optimization:

```
Agent workflow per round:
  1. Modify only autoresearch/train.py (backbone, LR, batch size, epochs)
  2. Run training
  3. Measure with eval_lidar.py (PRIMARY: physically grounded)
  4. Measure with eval_gold.py (sanity check: human-verified)
  5. NOTE: eval.py is LOCKED — never touched
  6. Keep if lidar metric improves; revert otherwise
```

---

## Design Patterns for Future Projects

1. **Weak labels → self-training → honest metric** (EVAL-005 + EVAL-006 + EVAL-001): applicable to any domain with cheap weak annotations and expensive ground truth.
2. **Locked evaluation file** (EVAL-002): standard practice for any agentic optimization loop.
3. **Multiple evaluation tracks** (EVAL-003 + EVAL-001): sanity check + primary metric provides defense in depth.
4. **Domain-specific metric parameters** (EVAL-004): choose IoU threshold appropriate to the target domain.
