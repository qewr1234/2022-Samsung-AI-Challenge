"""Dataset utilities for the 2022 Samsung AI Challenge (3D Metrology).

Expected layout (relative to ``--data-root``)::

    train/SEM/*.png              SEM images, several per depth map (e.g. xxx_itr0.png)
    train/Depth/*.png            ground-truth depth maps (xxx.png)
    simulation_data/Case_N/SEM   same convention, synthetic
    simulation_data/Case_N/Depth
    test/SEM/*.png               test SEM images, no labels

A SEM file is paired with the depth file of the same stem once a trailing
``_itr<N>`` suffix is removed.  If no such suffix exists the stems must match
exactly.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_ITR_RE = re.compile(r"_itr\d+$")


def depth_key(sem_path: Path) -> str:
    """Return the depth stem a SEM file belongs to (``abc_itr2`` -> ``abc``)."""
    return _ITR_RE.sub("", sem_path.stem)


def load_gray(path: Path) -> np.ndarray:
    """Load an image as float32 array in [0, 1], shape (H, W)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    return arr / 255.0


def collect_pairs(sem_dir: Path, depth_dir: Path) -> list[tuple[Path, Path]]:
    """Match every SEM image to its depth map. Raises if any SEM is unmatched."""
    depth_by_stem = {p.stem: p for p in sorted(depth_dir.glob("*.png"))}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for sem in sorted(sem_dir.glob("*.png")):
        key = depth_key(sem)
        d = depth_by_stem.get(key) or depth_by_stem.get(sem.stem)
        if d is None:
            missing.append(sem.name)
        else:
            pairs.append((sem, d))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} SEM files in {sem_dir} have no depth map, e.g. {missing[:3]}"
        )
    if not pairs:
        raise FileNotFoundError(f"No SEM/Depth pairs found under {sem_dir}")
    return pairs


def split_by_depth(
    pairs: list[tuple[Path, Path]], val_frac: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split so that all SEM views of one depth map land in the same fold."""
    keys = sorted({d.stem for _, d in pairs})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac))) if val_frac > 0 else 0
    val_keys = set(keys[:n_val])
    train = [p for p in pairs if p[1].stem not in val_keys]
    val = [p for p in pairs if p[1].stem in val_keys]
    return train, val


class SemDepthDataset(Dataset):
    """Yields (sem, depth) tensors of shape (1, H, W) in [0, 1]."""

    def __init__(self, pairs: list[tuple[Path, Path]], augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sem_path, depth_path = self.pairs[idx]
        sem = load_gray(sem_path)
        depth = load_gray(depth_path)
        if self.augment:
            # Flips keep the SEM/depth correspondence valid.
            if random.random() < 0.5:
                sem, depth = sem[:, ::-1], depth[:, ::-1]
            if random.random() < 0.5:
                sem, depth = sem[::-1, :], depth[::-1, :]
        sem_t = torch.from_numpy(np.ascontiguousarray(sem))[None]
        depth_t = torch.from_numpy(np.ascontiguousarray(depth))[None]
        return sem_t, depth_t


class SemOnlyDataset(Dataset):
    """Test-time dataset: yields (sem tensor, file name)."""

    def __init__(self, sem_dir: Path):
        self.paths = sorted(Path(sem_dir).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No png files in {sem_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        return torch.from_numpy(load_gray(p))[None], p.name
