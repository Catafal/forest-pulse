# Codebase Observation: more-data-clean-eval

**Date:** 2026-04-06
**GitNexus available:** no

## Relevant Files

- `scripts/download_montseny.py` — ZONES dict (3 entries), download_zone() creates `data/montseny/raw/{zone}/`. CLI `--zones` auto-discovers from ZONES.keys(). Just add entries to expand.
- `scripts/tile_orthophoto.py` — Auto-discovers zone dirs via `RAW_DIR.iterdir()`. Budget: `max_patches // len(zones)`. With 8 zones at default 300 → 37/zone. Need to increase `--max-patches`.
- `scripts/bootstrap_annotations.py` — Reads `data/montseny/patches/*.jpg`. No zone awareness. Works with any patch count.
- `scripts/self_train.py` — Same: reads `data/montseny/patches/`. Works with any count.
- `autoresearch/eval.py` — LOCKED. Hardcoded: `VAL_DIR = data/rfdetr/valid`, `VAL_ANNOT = VAL_DIR/_annotations.coco.json`. Cannot modify paths or add flags inside this file.
- `scripts/prepare_rfdetr_dataset.py` — `prepare_from_single_json()` splits 80/20. Could create gold split too.

## Key Findings

1. **Adding zones = just expand ZONES dict.** No code changes. tile/bootstrap/self_train auto-discover.
2. **eval.py is truly locked** — can't add `--eval-gold` flag there. Solution: create `autoresearch/eval_gold.py` as a separate evaluation module that reads from `data/montseny/eval_gold/`.
3. **Patch budget issue:** 8 zones at default max_patches=300 → only 37 per zone. Need `--max-patches 800` to get ~100 per zone.
4. **Gold eval set needs:** 20 patches selected from diverse zones + empty COCO JSON template for user to fill after Roboflow annotation.

## Conditional Skill Routing

- [ ] /plan-eng-review — not applicable (small scope)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable
