# Implementation Plan: more-data-clean-eval

**Version:** 1
**Date:** 2026-04-06
**Based on:** SPEC.md + OBSERVE.md

## Summary

Add 5 new ICGC zones to the ZONES dict in download_montseny.py (data change only, no logic changes). Create eval_gold.py as a separate evaluation module for 20 human-annotated patches. Prepare a gold eval directory with selected patches + instructions for the user to annotate in Roboflow.

## Files to Modify

| File | Change | Rationale |
|---|---|---|
| `scripts/download_montseny.py` | Add 5 new entries to ZONES dict | More geographic diversity across Montseny |
| `scripts/tile_orthophoto.py` | Increase DEFAULT_MAX_PATCHES from 300 to 800 | 8 zones need more budget (100/zone) |

## Files to Create

| File | Purpose |
|---|---|
| `autoresearch/eval_gold.py` | Evaluate against human-annotated gold set (~60 lines) |
| `scripts/prepare_gold_eval.py` | Select 20 diverse patches + create empty COCO template (~50 lines) |

## Implementation Details

### 1. New ICGC Zones (download_montseny.py)

5 new zones covering different parts of Montseny not yet sampled:

| Zone key | Location | Elevation | Forest type | Why |
|---|---|---|---|---|
| `ne_slopes` | NE Montseny (Sant Marçal area) | 900-1200m | Beech + oak transition | Different aspect than existing mid |
| `sw_valley` | SW Montseny (Tagamanent area) | 400-700m | Holm oak, dry slope | Drier microclimate than low zone |
| `se_ridge` | SE Montseny (above Sant Celoni) | 600-900m | Cork oak + pine | Different species mix |
| `nw_plateau` | NW Montseny (Espinelves area) | 700-1000m | Mixed deciduous | Humid Atlantic influence |
| `summit` | Central ridge (Matagalls area) | 1400-1700m | Subalpine meadow + sparse beech | High altitude edge case |

Each zone = 3km × 3km = 9 tiles. 5 zones × 9 = 45 new tiles + existing 27 = 72 total.

### 2. Gold Evaluation Set

**eval_gold.py** — Separate from locked eval.py:
- Reads from `data/montseny/eval_gold/images/` + `annotations.json`
- Same mAP50 metric (supervision.metrics.MeanAveragePrecision)
- Same API: `evaluate_gold(checkpoint_path) -> float`
- Prints `gold_map50: X.XXXX` (machine-readable)

**prepare_gold_eval.py** — Run once:
- Select 20 patches from diverse zones (2-3 per zone)
- Copy them to `data/montseny/eval_gold/images/`
- Create empty `annotations.json` (COCO structure, images registered, zero annotations)
- Print instructions: "Upload to Roboflow, annotate, export COCO, replace annotations.json"

### 3. tile_orthophoto.py change

One line: `DEFAULT_MAX_PATCHES = 800` (was 300). With 8 zones → 100 per zone.

## Tests to Write

| Done Criterion # | Test | Type |
|---|---|---|
| 1 | `test_zones_count` — verify ZONES has >= 8 entries | unit |
| 3 | `test_eval_gold_with_mock` — verify eval_gold returns float mAP50 | unit |

## Approach Alternatives

| Alternative | Rejected Because |
|---|---|
| Modify eval.py to accept `--gold` flag | eval.py is LOCKED — can't modify |
| Download ALL of Montseny (301 km²) | Overkill — 8 zones (72 km²) is sufficient diversity |
| Auto-generate gold labels from self-trained model | Defeats the purpose — gold must be human-verified |

## Risks

- New zone coordinates might miss forest (hit roads/towns). Mitigation: verify against ICGC viewer.
- 45 new tiles = ~2.3 GB download. Mitigation: tiles are git-ignored, only patches committed.
- User might not annotate the 20 gold images. Mitigation: eval_gold.py warns if annotations.json has 0 annotations.

## Estimated Scope

- 2 modified files (1 line each in tile_orthophoto.py, ~30 lines in download_montseny.py)
- 2 new files (~110 lines total)
- Complexity: low
