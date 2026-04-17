# Archived Scripts

These scripts were used during the development of Forest Pulse
(Phases 1-9.5) but are **no longer part of the production pipeline**.
They are retained for reproducibility and historical reference.

The current production entry point is:

```bash
python scripts/inventory_montseny.py --detector lidar-first
```

## What's here

| Script | Original phase | Why archived |
|---|---|---|
| `demo.py` | Phase 1 | Superseded by `full_pipeline_demo.py` |
| `apply_filters_demo.py` | Phase 4 | Uses old `filter_by_height`, not the current `lidar_tree_top_filter` |
| `self_train.py` | Phase 2 | RF-DETR self-training loop; RF-DETR is no longer on the production critical path |
| `teacher_annotations.py` | Phase 2 | Annotation generation for self-training |
| `bootstrap_annotations.py` | Phase 3 | DeepForest weak label generation |
| `prepare_rfdetr_dataset.py` | Phase 3 | COCO dataset preparation for RF-DETR training |
| `prepare_sample.py` | Phase 3 | OAM-TCD sample data prep |
| `prepare_gold_eval.py` | Phase 3 | Gold evaluation set preparation |
| `train_classifier.py` | Phase 9/9.5b | Trains the (now-superseded) sklearn tree classifier; replaced by `forest_pulse.species` (Phase 12a) |
| `sam2_smoke_test.py` | Phase 6 | One-shot SAM2 verification |
| `lidar_smoke_test.py` | Phase 7 | One-shot LiDAR pipeline verification |

## Still-active scripts (in `scripts/`)

| Script | Purpose |
|---|---|
| `inventory_montseny.py` | **Production driver** — full park inventory |
| `full_pipeline_demo.py` | Single-patch demo with all stages logged |
| `run_lidar_eval.py` | Phase 8-10 LiDAR-verified F1 evaluation |
| `sweep_confidence.py` | Phase 10a-c confidence × filter experiment |
| `probe_lidar_peaks.py` | Phase 11a LiDAR peak validation probe |
| `download_montseny.py` | ICGC orthophoto download |
| `download_lidar.py` | ICGC LAZ tile download |
| `download_data.py` | Generic data download helper |
| `tile_orthophoto.py` | GeoTIFF → 640×640 patch tiling |
