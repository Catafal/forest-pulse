"""Download datasets for Forest Pulse training and evaluation.

Supports downloading from HuggingFace, Zenodo, and direct URLs.
All data is saved to the data/ directory (git-ignored).

Usage:
    python scripts/download_data.py --sample-image
    python scripts/download_data.py --dataset oam-tcd --sample
    python scripts/download_data.py --dataset neon-eval
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# DeepForest ships a sample NEON aerial image in its package.
# This URL points to the same image hosted on GitHub for standalone download.
SAMPLE_IMAGE_URL = (
    "https://raw.githubusercontent.com/weecology/DeepForest"
    "/main/deepforest/data/OSBS_029.png"
)
SAMPLE_IMAGE_NAME = "OSBS_029.png"

DATASETS = {
    "oam-tcd": {
        "name": "OAM-TCD (280K trees, global)",
        "source": "huggingface",
        "hf_id": "restor/tcd",
        "license": "CC-BY-4.0",
    },
    "selvabox": {
        "name": "SelvaBox (83K tropical crowns)",
        "source": "huggingface",
        "hf_id": "CanopyRS/SelvaBox",
        "license": "CC-BY-4.0",
    },
    "neon-eval": {
        "name": "NeonTreeEvaluation (benchmark)",
        "source": "zenodo",
        "zenodo_id": "5914554",
        "license": "Open",
    },
    "swedish-damages": {
        "name": "Swedish Forest Damages (102K bboxes + health labels)",
        "source": "lila",
        "url": "https://lila.science/datasets/forest-damages-larch-casebearer/",
        "license": "Open",
    },
}


def download_sample_image() -> Path:
    """Download a single NEON aerial forest image for demo purposes.

    Uses the OSBS_029 image from DeepForest's test fixtures — a well-known
    aerial RGB tile of a mixed forest at Ordway-Swisher Biological Station.

    Returns:
        Path to the downloaded image.
    """
    output_dir = DATA_DIR / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / SAMPLE_IMAGE_NAME

    if output_path.exists():
        logger.info("Sample image already exists: %s", output_path)
        return output_path

    logger.info("Downloading sample image from DeepForest repo...")
    urllib.request.urlretrieve(SAMPLE_IMAGE_URL, output_path)
    logger.info("Saved to: %s", output_path)
    return output_path


def download_huggingface(hf_id: str, output_dir: Path, sample: bool = False):
    """Download a dataset from HuggingFace Hub."""
    # TODO: Implement full HuggingFace download (Phase 2)
    print(f"[Phase 2] To download {hf_id}, run:")
    print(f'  python -c "from datasets import load_dataset; '
          f"ds = load_dataset('{hf_id}'"
          f'{", streaming=True" if sample else ""})'
          f'"')


def download_zenodo(zenodo_id: str, output_dir: Path):
    """Download a dataset from Zenodo by record ID."""
    # TODO: Implement Zenodo download (Phase 2)
    print(f"[Phase 2] Download manually from: https://zenodo.org/records/{zenodo_id}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Download Forest Pulse datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to download.",
    )
    parser.add_argument("--sample-image", action="store_true", help="Download a single demo image.")
    parser.add_argument("--sample", action="store_true", help="Download only a small sample.")
    parser.add_argument("--full", action="store_true", help="Download the full dataset.")
    args = parser.parse_args()

    if args.sample_image:
        path = download_sample_image()
        print(f"\nSample image ready: {path}")
        print(f"Run demo:  python scripts/demo.py --image {path}")
        return

    if not args.dataset:
        parser.error("Provide --dataset or --sample-image")

    ds_info = DATASETS[args.dataset]
    output_dir = DATA_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {ds_info['name']}")
    print(f"License: {ds_info['license']}")
    print(f"Output:  {output_dir}")

    if ds_info["source"] == "huggingface":
        download_huggingface(ds_info["hf_id"], output_dir, sample=args.sample)
    elif ds_info["source"] == "zenodo":
        download_zenodo(ds_info["zenodo_id"], output_dir)
    else:
        print(f"Manual download: {ds_info.get('url', 'see TECH_STACK.md')}")


if __name__ == "__main__":
    main()
