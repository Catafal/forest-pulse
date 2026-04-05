"""Download datasets for Forest Pulse training and evaluation.

Supports downloading from HuggingFace, Zenodo, and LILA BC.
All data is saved to the data/ directory (git-ignored).

Usage:
    python scripts/download_data.py --dataset oam-tcd --sample
    python scripts/download_data.py --dataset oam-tcd --full
    python scripts/download_data.py --dataset neon-eval
    python scripts/download_data.py --dataset swedish-damages
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


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


def download_huggingface(hf_id: str, output_dir: Path, sample: bool = False):
    """Download a dataset from HuggingFace Hub."""
    # TODO: Implement HuggingFace download
    # from datasets import load_dataset
    # ds = load_dataset(hf_id, split="train", streaming=sample)
    raise NotImplementedError(f"HuggingFace download not yet implemented for {hf_id}")


def download_zenodo(zenodo_id: str, output_dir: Path):
    """Download a dataset from Zenodo by record ID."""
    # TODO: Implement Zenodo download
    raise NotImplementedError(f"Zenodo download not yet implemented for {zenodo_id}")


def main():
    parser = argparse.ArgumentParser(description="Download Forest Pulse datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to download.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download only a small sample (for development).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full dataset (large, needs disk space).",
    )
    args = parser.parse_args()

    ds_info = DATASETS[args.dataset]
    output_dir = DATA_DIR / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {ds_info['name']}")
    print(f"License: {ds_info['license']}")
    print(f"Output: {output_dir}")

    if ds_info["source"] == "huggingface":
        download_huggingface(ds_info["hf_id"], output_dir, sample=args.sample)
    elif ds_info["source"] == "zenodo":
        download_zenodo(ds_info["zenodo_id"], output_dir)
    else:
        print(f"Manual download required: {ds_info.get('url', 'see TECH_STACK.md')}")


if __name__ == "__main__":
    main()
