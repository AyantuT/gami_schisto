"""
dataset.py – GeoTIFF tile dataset for binary vegetation classification.

Supports two labelling strategies:
  1. pixel_majority  – reads the raster, takes the mode of non-zero pixels.
     Works perfectly for your exported ArcGIS tiles where each file is pure
     class-1 or class-2 (num_unique == 1).
  2. filename_prefix – parses the label from the filename via a regex group.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# rasterio is the standard choice for geospatial rasters; fall back to
# tifffile if rasterio is unavailable (e.g. no GDAL in the environment).
try:
    import rasterio
    from rasterio.enums import Resampling
    _RASTERIO = True
except ImportError:
    import tifffile
    _RASTERIO = False
    warnings.warn("rasterio not found – falling back to tifffile (no CRS/projection support).")


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_tile_rasterio(path: str, bands: List[int], size: int,
                        percentile_clip: float) -> np.ndarray:
    """Return (C, H, W) float32 array normalised to [0, 1]."""
    with rasterio.open(path) as src:
        out_shape = (len(bands), size, size)
        data = src.read(bands, out_shape=out_shape,
                        resampling=Resampling.bilinear).astype(np.float32)
    return _normalise(data, percentile_clip)


def _read_tile_tifffile(path: str, bands: List[int], size: int,
                        percentile_clip: float) -> np.ndarray:
    import cv2
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, ...]          # (1, H, W)
    elif img.ndim == 3 and img.shape[2] <= 8:
        img = img.transpose(2, 0, 1)       # (C, H, W)
    selected = img[[b - 1 for b in bands if b <= img.shape[0]]]
    # resize each band
    resized = np.stack([
        cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
        for b in selected
    ])
    return _normalise(resized, percentile_clip)


def _normalise(data: np.ndarray, percentile_clip: float) -> np.ndarray:
    """Per-channel percentile clip then scale to [0, 1]."""
    out = np.zeros_like(data)
    for c in range(data.shape[0]):
        lo = np.percentile(data[c], percentile_clip)
        hi = np.percentile(data[c], 100 - percentile_clip)
        if hi > lo:
            out[c] = np.clip((data[c] - lo) / (hi - lo), 0, 1)
        else:
            out[c] = 0.0
    return out


def _majority_label_rasterio(path: str) -> int:
    with rasterio.open(path) as src:
        data = src.read(1)
    vals, counts = np.unique(data[data > 0], return_counts=True)
    if len(vals) == 0:
        raise ValueError(f"No non-zero pixels in {path}")
    return int(vals[counts.argmax()])


def _majority_label_tifffile(path: str) -> int:
    img = tifffile.imread(path)
    flat = img.ravel() if img.ndim <= 3 else img[..., 0].ravel()
    nonzero = flat[flat > 0]
    if len(nonzero) == 0:
        raise ValueError(f"No non-zero pixels in {path}")
    vals, counts = np.unique(nonzero, return_counts=True)
    return int(vals[counts.argmax()])


# ── main dataset ──────────────────────────────────────────────────────────────

class VegetationDataset(Dataset):
    """
    Parameters
    ----------
    root_dir        : directory containing .tif files
    class_map       : dict mapping pixel value → class name  {1: "emergent", 2: "submergent"}
    label_strategy  : "pixel_majority" | "filename_prefix"
    label_regex     : only used for filename_prefix (e.g. r"class(\\d+)_")
    bands           : list of 1-indexed band numbers to read
    image_size      : output spatial size (square)
    percentile_clip : normalisation percentile (e.g. 2 clips bottom/top 2 %)
    transform       : torchvision transforms applied to the tensor
    """

    def __init__(
        self,
        root_dir: str,
        class_map: Dict[int, str],
        label_strategy: str = "pixel_majority",
        label_regex: Optional[str] = None,
        bands: List[int] = [1, 2, 3],
        image_size: int = 224,
        percentile_clip: float = 2.0,
        transform=None,
        file_list: Optional[List[str]] = None,
    ):
        self.root = Path(root_dir)
        self.class_map = class_map                          # {pixel_val: name}
        self.name_to_idx = {v: i for i, v in enumerate(sorted(class_map.values()))}
        self.idx_to_name = {i: v for v, i in self.name_to_idx.items()}
        self.label_strategy = label_strategy
        self.label_regex = re.compile(label_regex) if label_regex else None
        self.bands = bands
        self.image_size = image_size
        self.percentile_clip = percentile_clip
        self.transform = transform

        # Discover files
        if file_list is not None:
            self.files = [self.root / f for f in file_list]
        else:
            self.files = sorted(self.root.glob("*.tif")) + sorted(self.root.glob("*.tiff"))

        # Pre-compute labels (fast for pure-class tiles)
        self.labels = self._compute_labels()

    # ── label extraction ──────────────────────────────────────────────────────

    def _compute_labels(self) -> List[int]:
        labels = []
        for p in self.files:
            lbl = self._get_label(p)
            labels.append(lbl)
        return labels

    def _get_label(self, path: Path) -> int:
        if self.label_strategy == "filename_prefix":
            if self.label_regex is None:
                raise ValueError("label_regex must be set when using filename_prefix strategy")
            m = self.label_regex.search(path.name)
            if not m:
                raise ValueError(f"Regex did not match filename: {path.name}")
            pixel_val = int(m.group(1))
        else:  # pixel_majority
            if _RASTERIO:
                pixel_val = _majority_label_rasterio(str(path))
            else:
                pixel_val = _majority_label_tifffile(str(path))

        if pixel_val not in self.class_map:
            raise ValueError(f"Pixel value {pixel_val} not in class_map: {self.class_map}")
        class_name = self.class_map[pixel_val]
        return self.name_to_idx[class_name]

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> torch.Tensor:
        if _RASTERIO:
            arr = _read_tile_rasterio(str(path), self.bands,
                                      self.image_size, self.percentile_clip)
        else:
            arr = _read_tile_tifffile(str(path), self.bands,
                                      self.image_size, self.percentile_clip)
        return torch.from_numpy(arr)  # (C, H, W) float32 in [0,1]

    # ── dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self._load_image(self.files[idx])
        lbl = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced datasets."""
        counts = np.bincount(self.labels, minlength=len(self.class_map))
        weights = 1.0 / (counts + 1e-6)
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)

    def summary(self):
        counts = np.bincount(self.labels, minlength=len(self.class_map))
        print(f"Dataset: {len(self.files)} tiles  |  root: {self.root}")
        for i, name in self.idx_to_name.items():
            print(f"  [{i}] {name:20s}  {counts[i]:5d} tiles")
