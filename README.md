# Forest Pulse

**Aerial imagery tree detection, health scoring, and temporal change monitoring.**

Detect individual trees, assess their health from RGB imagery alone, and track changes over time — no drone required. Works with free public aerial orthophotos.

<!-- TODO: Add demo GIF/image here once Phase 1 is complete -->
<!-- ![Forest Pulse Demo](docs/assets/demo.png) -->

## Features

- **Tree Crown Detection** — Individual tree detection using RF-DETR (DINOv2 backbone) or DeepForest
- **RGB Health Scoring** — Classify trees as healthy/stressed/dead using GRVI and Excess Green indices — no multispectral imagery needed
- **GIS-Ready Output** — GeoJSON export with GPS coordinates, loadable directly in QGIS/ArcGIS
- **Temporal Change Detection** — Compare two time periods to identify tree loss, crown decline, and health progression
- **Auto-Research Harness** — Karpathy-style overnight fine-tuning optimization to find the best model config for your specific forest

## Quick Start

```bash
# Install from source (PyPI release coming in Phase 6)
git clone https://github.com/jordicatafal/forest-pulse.git
cd forest-pulse
pip install -e .
```

```python
import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees
from forest_pulse.health import score_health
from forest_pulse.visualize import annotate_trees

# Load aerial image
image = np.array(Image.open("aerial_image.tif"))

# Detect trees in aerial imagery
detections = detect_trees(image)

# Score health of each detected tree
health = score_health(image, detections)

# Visualize with health color coding
annotated = annotate_trees(image, detections, health)
```

<!-- Colab notebook coming in Phase 1 -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jordicatafal/forest-pulse/blob/main/notebooks/01_quickstart.ipynb) -->

## No Drone Needed

Forest Pulse works with free public aerial imagery:

| Source | Resolution | Coverage | Access |
|--------|-----------|----------|--------|
| **ICGC Catalunya** | 10-25 cm | All Catalonia | [icgc.cat](https://www.icgc.cat/en/Citizens/Download/Aerial-and-satellite-images) |
| **PNOA Spain** | 10-25 cm | All Spain | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/Spain_PNOA_PNOA10) |
| **OpenAerialMap** | Variable | Global | [openaerialmap.org](https://openaerialmap.org) |

## Architecture

```
Aerial Image → detect.py → health.py → georef.py → export.py → GeoJSON
                  ↓
             visualize.py (annotated image with health colors)
                  ↓
             temporal.py (change report between time periods)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed pipeline design and module contracts.

## Training Data

All training data is publicly available under permissive licenses:

| Dataset | Trees | Source | License |
|---------|-------|--------|---------|
| [OAM-TCD](https://huggingface.co/datasets/restor/tcd) | 280,000 | HuggingFace | CC-BY-4.0 |
| [SelvaBox](https://huggingface.co/datasets/CanopyRS/SelvaBox) | 83,000 | HuggingFace | CC-BY-4.0 |
| [BAMFORESTS](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/bamforests) | 27,160 | DLR | CC-BY-4.0 |
| [Swedish Forest Damages](https://lila.science/datasets/forest-damages-larch-casebearer/) | 102,000 | LILA BC | Open |

## Auto-Research: Overnight Fine-Tuning

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an AI agent iterates on model configurations overnight while you sleep.

```
eval.py    → LOCKED metric (mAP50 on validation set)
train.py   → EDITABLE config (backbone, LR, augmentations)
program.md → Agent instructions (edit, run, keep/revert, never stop)
```

After 8 hours: ~32 experiments tried, empirically validated configuration in git history.

See [autoresearch/README.md](autoresearch/README.md) for details.

## Development

```bash
# Clone and install
git clone https://github.com/jordicatafal/forest-pulse.git
cd forest-pulse
pip install -e ".[dev,train,notebooks]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Download sample data
python scripts/download_data.py --dataset oam-tcd --sample
```

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | MVP detection + visualization | In progress |
| 2 | RF-DETR training + auto-research | Planned |
| 3 | Health classification | Planned |
| 4 | Georeferencing + GIS export | Planned |
| 5 | Temporal change detection | Planned |
| 6 | Polish + release | Planned |

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed breakdown.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow (ICLR 2026)
- [Supervision](https://github.com/roboflow/supervision) by Piotr Skalski
- [DeepForest](https://github.com/weecology/DeepForest) by Weecology
- [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
- [OAM-TCD](https://huggingface.co/datasets/restor/tcd) by Restor Foundation (NeurIPS 2024)
