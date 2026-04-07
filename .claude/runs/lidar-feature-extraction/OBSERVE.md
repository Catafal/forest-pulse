# Codebase Observation: lidar-feature-extraction

**Date:** 2026-04-07
**GitNexus available:** no

## External research (verified, live-tested URLs)

Full report: `.claude/research/icgc_laz_urls.md`.

### ICGC LAZ URL pattern (live-tested, 200 OK)

```
https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
  full10km{ID10K}/lidar-territorial-v3r1-full1km{ID1K}-2021-2023.laz
```

### Tile ID encoding (from the authoritative GeoJSON tile index)

```python
easting_km  = int(x) // 1000          # e.g. 450000 → 450
northing_km = int(y) // 1000          # e.g. 4625000 → 4625
ID1K  = f"{easting_km:03d}{(northing_km - 4000):03d}"
        # (450, 4625) → "450625"
ID10K = f"{(easting_km // 10):02d}{((northing_km // 10) - 400):02d}"
        # (450, 4625) → "4562"
```

**Critical gotcha**: the 3-digit Y component is `northing_km - 4000`, not
the raw last 3 digits. A naive `northing_km % 1000` would give "625" by
coincidence here, but would break for tiles where northing is e.g. 4005
(correct suffix "005", naive "005" — same here) vs 4715 (correct "715",
naive "715" — same) … it actually works out identically for this range
but the intent matters for clarity and for tiles outside the typical
UTM-31N range.

### Verified single URL for Montseny (x=450000, y=4625000)

```
https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
  full10km4562/lidar-territorial-v3r1-full1km450625-2021-2023.laz
```
200 OK, Content-Length ~645 MB, Accept-Ranges: bytes, Last-Modified Nov 2025.

### Tile index (authoritative, can be used as fallback)

```
https://datacloud.icgc.cat/datacloud/lidar-territorial/json/
  lidar-territorial-tall.json
```
6.56 MB GeoJSON FeatureCollection, EPSG:25831, each feature is a 1 km
cell polygon with `properties.ID1K` / `properties.ID10K`. Useful as a
validation step if the naive encoding ever produces a 404.

### LAZ file characteristics

- Format: raw LAZ (not zipped despite the `laz_unzip` path name)
- Classification: standard ASPRS LAS 1.4 codes
  - 2 = ground
  - 3/4/5 = low/medium/high vegetation
  - 6 = building
  - 9 = water
- Density: 8+ pts/m² minimum across Catalonia (empirically 10-25 pts/m² in
  dense forest like Montseny)
- File size: 400 MB – 1 GB per 1 km tile compressed
- Multi-return: yes (up to 4 returns per pulse); intensity values included

### laspy status

`laspy` is NOT currently installed in the venv. Must add to
`pyproject.toml` under the training/extras section and `pip install -e`.
Needs `laspy[lazrs]` for LAZ decompression support.

## Codebase — current lidar.py state

Already has:
- `LiDARFeatures` **not yet defined** (the SPEC introduces it as a new dataclass)
- `CrownFilter`-like constants at top (`DEFAULT_HEIGHT_THRESHOLD = 5.0`)
- `fetch_chm_for_patch()` — stub that raises `NotImplementedError` with
  an explanation. We REPLACE this with a working implementation.
- `filter_by_height(detections, chm_path, bounds, size, threshold, aggregation)`
  — **already works correctly** on any CHM raster. We just feed it a
  real CHM now. **Zero changes to this function.**
- `_pixel_bbox_to_geo()` — coordinate transform, reused.
- `_sample_raster_agg()` — windowed raster read with aggregation, reused.
- `_strip_rfdetr_metadata()` — reused.

## Codebase — other integration points

- `src/forest_pulse/georef.py` — takes `detections`, optional `health_scores`.
  Gains an optional `lidar_features: list[LiDARFeatures] | None = None`.
  When provided, adds 7 new columns to the GeoDataFrame. Backward compatible.
- `scripts/download_lidar.py` (new) — downloader CLI for Montseny zones.
- `scripts/lidar_smoke_test.py` (new) — end-to-end verification script.
- `tests/test_lidar_features.py` (new) — unit tests with synthetic point clouds
  (mock laspy.read to avoid needing a real LAZ file in CI).
- `.gitignore` — add `data/montseny/lidar/` (LAZ cache directory).
- `pyproject.toml` — add `laspy[lazrs]>=2.5.0` to a `[lidar]` extra.

## Key patterns to follow

- **Lazy imports for heavy deps**: `import laspy` inside functions, like
  `rfdetr` in `detect.py` and `Sam2Model` in `segment.py`. Missing dep →
  clear `ImportError` with install instructions.
- **Module-level cache**: small dict keyed by path for already-loaded LAZ
  files within a run (optional — LAZ read is fast-ish, probably not needed).
- **Pure functions**: inputs → outputs, no hidden state. Tests exercise
  pure helpers with synthetic data.
- **Backward compatibility**: every existing call site keeps working
  without changes. New features are opt-in via new parameters.

## Architectural constraints

- Max 200 lines per function, 1000 per file. The LAZ-specific code will
  push `lidar.py` toward 800-900 lines, approaching the 1000-line limit.
  Acceptable, but no frivolous additions.
- All imports at top (except heavy lazy ones inside functions).
- Type hints on public functions + Google-style docstrings.
- Comments explain WHY, not WHAT.

## Risks surfaced during OBSERVE

1. **`laspy[lazrs]` install**: `lazrs` is a Rust-backed decoder;
   `pip install` pulls a prebuilt wheel on macOS arm64 → should be fast.
   If prebuilt wheel is missing we fall back to `laszip` (C++, heavier).
2. **Tile boundary trees**: a tree spanning two 1 km LAZ tiles is rare at
   25 cm patch scale but possible. MVP ignores this (pick the tile
   containing the patch center). Worst case: edge-of-patch tree gets
   slightly truncated features. Acceptable.
3. **Temporal mismatch**: LiDAR 2021-2023, RGB ~2024. For 5 m height
   threshold, irrelevant. For future species work, introduces ~30% noise
   but not a dealbreaker.
4. **Memory**: loading a ~645 MB LAZ file into memory uses ~2-3 GB RAM
   after decompression. Fine on M4 Pro 48 GB; watch on Mac Mini 24 GB.
5. **First-run download time**: ~645 MB × network bandwidth. On 50 Mbps
   that's ~100 s. Document expected timing in the smoke test.

## Conditional skill routing

- [ ] /plan-eng-review — not applicable (scope is focused, contracts clear)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable
