"""
Visualize crop images segmented by image area.

Example:
    python visualize_crop_area_segments.py \
        --input-dir three_crop_output \
        --output area_segments.png \
        --bin-size 500 \
        --samples-per-bin 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class CropInfo:
    path: Path
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bin crop images by area and create a visualization."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("three_crop_output"),
        help="Directory containing crop images (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("three_crop_output_area_segments.png"),
        help="Path to save the visualization image.",
    )
    parser.add_argument("--bin-size", type=int, default=500, help="Area bin size in pixels^2.")
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=2,
        help="How many random images to show for each non-empty bin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample preview images.",
    )
    return parser.parse_args()


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def collect_crops(input_dir: Path) -> list[CropInfo]:
    crops: list[CropInfo] = []
    for path in iter_image_paths(input_dir):
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        crops.append(CropInfo(path=path, width=w, height=h))
    return crops


def bin_label(bin_idx: int, bin_size: int) -> str:
    low = bin_idx * bin_size
    high = low + bin_size
    return f"{low:,}-{high:,} px^2"


def make_visualization(
    crops: list[CropInfo],
    output_path: Path,
    bin_size: int,
    samples_per_bin: int,
    seed: int,
) -> None:
    areas = np.array([c.area for c in crops], dtype=np.int64)
    rng = np.random.default_rng(seed)
    bin_ids = areas // bin_size
    unique_bins = sorted(set(int(v) for v in bin_ids))

    occupied_bins: list[tuple[int, list[int]]] = []
    for bin_idx in unique_bins:
        members = [i for i, b in enumerate(bin_ids) if int(b) == bin_idx]
        if members:
            occupied_bins.append((bin_idx, members))

    if not occupied_bins:
        raise RuntimeError("No non-empty bins found.")

    nrows = len(occupied_bins)
    ncols = samples_per_bin
    fig = plt.figure(figsize=(2.9 * ncols, 2.7 * nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35)

    for row_idx, (bin_idx, members) in enumerate(occupied_bins):
        k = min(samples_per_bin, len(members))
        chosen = rng.choice(members, size=k, replace=False)

        for col_idx in range(samples_per_bin):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.axis("off")
            if col_idx < len(chosen):
                crop = crops[int(chosen[col_idx])]
                image = cv2.imread(str(crop.path))
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
            if col_idx == 0:
                ax.set_title(
                    f"Bin {row_idx + 1}: {bin_label(bin_idx, bin_size)} | n={len(members)}",
                    fontsize=10,
                    loc="left",
                )

    fig.suptitle(f"Crop Area Bins ({len(crops)} images total)", fontsize=14, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bin_size < 1:
        raise ValueError("--bin-size must be >= 1")
    if args.samples_per_bin < 1:
        raise ValueError("--samples-per-bin must be >= 1")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    crops = collect_crops(args.input_dir)
    if not crops:
        raise RuntimeError(
            f"No readable images found under {args.input_dir}. "
            "Check the path and file extensions."
        )

    make_visualization(
        crops=crops,
        output_path=args.output,
        bin_size=args.bin_size,
        samples_per_bin=args.samples_per_bin,
        seed=args.seed,
    )
    print(f"Saved area-binned visualization to: {args.output}")


if __name__ == "__main__":
    main()
