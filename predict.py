"""Predict depth maps for test/SEM and build the submission zip.

Example:
    python predict.py --data-root ./data --checkpoint runs/baseline/best.pt --out submission.zip

Each test SEM image produces one depth png with the same file name, stored
flat inside the zip (the competition's sample_submission layout).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SemOnlyDataset
from src.model import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("submission.zip"))
    p.add_argument("--png-dir", type=Path, default=Path("submission"), help="where individual pngs are written")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="average predictions over horizontal/vertical flips")
    return p.parse_args()


@torch.no_grad()
def predict_batch(model: torch.nn.Module, sem: torch.Tensor, tta: bool) -> torch.Tensor:
    pred = model(sem)
    if tta:
        pred = pred + model(sem.flip(-1)).flip(-1)
        pred = pred + model(sem.flip(-2)).flip(-2)
        pred = pred + model(sem.flip(-1, -2)).flip(-1, -2)
        pred = pred / 4.0
    return pred


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNet(base=ckpt.get("base_channels", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, val RMSE {ckpt.get('val_rmse', float('nan')):.3f})")

    ds = SemOnlyDataset(args.data_root / "test" / "SEM")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    args.png_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for sem, names in tqdm(loader, desc="predict"):
        pred = predict_batch(model, sem.to(device), args.tta)
        pred = (pred.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        for arr, name in zip(pred[:, 0], names):
            out_path = args.png_dir / name
            Image.fromarray(arr, mode="L").save(out_path)
            written.append(out_path)

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in written:
            zf.write(p, arcname=p.name)
    print(f"wrote {len(written)} depth maps -> {args.out}")


if __name__ == "__main__":
    main()
