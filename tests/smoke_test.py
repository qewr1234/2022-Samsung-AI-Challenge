"""End-to-end smoke test on synthetic data. Runs on CPU in well under a minute."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
H, W = 72, 48


def make_split(base: Path, n_depth: int, n_itr: int, rng: np.random.Generator) -> None:
    (base / "SEM").mkdir(parents=True)
    (base / "Depth").mkdir(parents=True)
    for i in range(n_depth):
        depth = rng.integers(0, 256, (H, W), dtype=np.uint8)
        Image.fromarray(depth, "L").save(base / "Depth" / f"img_{i:04d}.png")
        for k in range(n_itr):
            sem = np.clip(depth.astype(np.int16) + rng.integers(-30, 30, (H, W)), 0, 255).astype(np.uint8)
            Image.fromarray(sem, "L").save(base / "SEM" / f"img_{i:04d}_itr{k}.png")


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        make_split(data / "train", n_depth=20, n_itr=4, rng=rng)
        make_split(data / "simulation_data" / "Case_1", n_depth=5, n_itr=1, rng=rng)
        (data / "test" / "SEM").mkdir(parents=True)
        for i in range(7):
            Image.fromarray(rng.integers(0, 256, (H, W), dtype=np.uint8), "L").save(
                data / "test" / "SEM" / f"test_{i:04d}.png")

        out_dir = Path(tmp) / "run"
        subprocess.run([sys.executable, "train.py", "--data-root", str(data), "--out-dir", str(out_dir),
                        "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--sim-cases", "1",
                        "--base-channels", "8"], cwd=ROOT, check=True)
        assert (out_dir / "best.pt").exists()

        sub = Path(tmp) / "submission.zip"
        subprocess.run([sys.executable, "predict.py", "--data-root", str(data), "--checkpoint", str(out_dir / "best.pt"),
                        "--out", str(sub), "--png-dir", str(Path(tmp) / "pngs"), "--num-workers", "0", "--tta"],
                       cwd=ROOT, check=True)
        with zipfile.ZipFile(sub) as zf:
            names = sorted(zf.namelist())
        assert names == [f"test_{i:04d}.png" for i in range(7)], names
        with zipfile.ZipFile(sub) as zf, zf.open(names[0]) as f:
            arr = np.asarray(Image.open(f))
        assert arr.shape == (H, W) and arr.dtype == np.uint8, (arr.shape, arr.dtype)
    print("smoke test passed")


if __name__ == "__main__":
    main()
