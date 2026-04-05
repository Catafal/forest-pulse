# Tech Stack — Forest Pulse

All versions are pinned. No "latest".

## Core Runtime

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| Python | >= 3.10 | Runtime | PSF |
| torch | >= 2.2.0 | Deep learning framework | BSD |
| torchvision | >= 0.17.0 | Vision utilities | BSD |
| supervision | >= 0.25.0 | Visualization, annotation, tracking | MIT |
| numpy | >= 1.24.0 | Numerical computing | BSD |
| Pillow | >= 10.0.0 | Image I/O | HPND |
| geopandas | >= 1.0.0 | GIS data manipulation | BSD |
| shapely | >= 2.0.0 | Geometric operations | BSD |
| pyproj | >= 3.6.0 | Coordinate transformations | MIT |
| piexif | >= 1.1.3 | EXIF metadata parsing | MIT |

## Training (optional)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| rfdetr | >= 1.0.0 | RF-DETR detection model (DINOv2 backbone) | Apache 2.0 |
| datasets | >= 2.18.0 | HuggingFace dataset loading | Apache 2.0 |
| albumentations | >= 1.4.0 | Image augmentations | MIT |

## Label Bootstrapping (one-time use)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| deepforest | >= 2.0.0 | Pre-trained tree crown detection for generating weak labels | MIT |

## Development

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| pytest | >= 8.0.0 | Testing | MIT |
| ruff | >= 0.4.0 | Linting + formatting | MIT |
| jupyter | >= 1.0.0 | Notebooks | BSD |
| matplotlib | >= 3.8.0 | Plotting in notebooks | PSF |

## Training Compute

| Option | Specs | Cost |
|--------|-------|------|
| Google Colab (free) | T4 GPU, 15GB VRAM | Free |
| Google Colab Pro | A100 GPU, 40GB VRAM | ~$10/mo |
| Apple MPS (local) | M-series GPU | Free (already owned) |

## Datasets

| Dataset | Size | Source | License |
|---------|------|--------|---------|
| OAM-TCD | 280K trees, 5K images | `restor/tcd` on HuggingFace | CC-BY-4.0 |
| BAMFORESTS | 27K trees, European | DLR website | CC-BY-4.0 |
| Swedish Forest Damages | 102K bboxes + health labels | LILA BC | Open |
| NeonTreeEvaluation | 6K eval + 10K train | Zenodo #5914554 | Open |
| SelvaBox | 83K tropical crowns | `CanopyRS/SelvaBox` on HuggingFace | CC-BY-4.0 |

## Aerial Imagery Sources (Free, No Drone)

| Source | Resolution | Coverage | Access |
|--------|-----------|----------|--------|
| ICGC Catalunya | 10-25 cm | All Catalonia | icgc.cat (free download) |
| PNOA Spain | 10-25 cm | All Spain | Google Earth Engine / datos.gob.es |
| OpenAerialMap | Variable | Global | openaerialmap.org |
