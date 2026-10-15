"""Train the SEM -> depth baseline.

Example (RTX 4060 Ti 8GB):
    python train.py --data-root ./data --epochs 30 --batch-size 64 --amp

Validation RMSE is reported on the 0-255 pixel scale, the same scale the
competition uses.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data import SemDepthDataset, collect_pairs, split_by_depth
from src.model import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="folder containing train/, test/, simulation_data/")
    p.add_argument("--out-dir", type=Path, default=Path("runs/baseline"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32, help="U-Net width; 32 fits easily in 8GB")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--sim-cases", type=str, default="", help="comma list of simulation cases to add, e.g. 1,2,3,4")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision (recommended on GPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="debug: use only the first N training pairs")
    return p.parse_args()


def build_datasets(args: argparse.Namespace):
    root = args.data_root
    pairs = collect_pairs(root / "train" / "SEM", root / "train" / "Depth")
    if args.limit:
        pairs = pairs[: args.limit]
    train_pairs, val_pairs = split_by_depth(pairs, args.val_frac, args.seed)

    train_sets = [SemDepthDataset(train_pairs, augment=True)]
    for case in [c.strip() for c in args.sim_cases.split(",") if c.strip()]:
        case_dir = root / "simulation_data" / f"Case_{case}"
        sim_pairs = collect_pairs(case_dir / "SEM", case_dir / "Depth")
        if args.limit:
            sim_pairs = sim_pairs[: args.limit]
        train_sets.append(SemDepthDataset(sim_pairs, augment=True))
        print(f"added simulation Case_{case}: {len(sim_pairs)} pairs")

    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    val_ds = SemDepthDataset(val_pairs, augment=False)
    print(f"train pairs: {len(train_ds)}  val pairs: {len(val_ds)}")
    return train_ds, val_ds


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return RMSE on the 0-255 scale."""
    model.eval()
    se_sum = 0.0
    n = 0
    for sem, depth in loader:
        sem, depth = sem.to(device), depth.to(device)
        pred = model(sem)
        se_sum += F.mse_loss(pred * 255.0, depth * 255.0, reduction="sum").item()
        n += depth.numel()
    return math.sqrt(se_sum / max(n, 1))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"device: {device}  amp: {use_amp}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    train_ds, val_ds = build_datasets(args)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = UNet(base=args.base_channels).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for sem, depth in pbar:
            sem, depth = sem.to(device, non_blocking=True), depth.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(sem)
                loss = F.mse_loss(pred, depth)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        train_rmse = math.sqrt(loss_sum / len(train_loader)) * 255.0
        val_rmse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})
        improved = val_rmse < best_rmse
        if improved:
            best_rmse = val_rmse
            torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                        "val_rmse": val_rmse, "epoch": epoch}, args.out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "base_channels": args.base_channels,
                    "val_rmse": val_rmse, "epoch": epoch}, args.out_dir / "last.pt")
        print(f"epoch {epoch:3d}  train RMSE {train_rmse:7.3f}  val RMSE {val_rmse:7.3f}"
              f"{'  *' if improved else ''}  ({time.time() - t0:.0f}s)")
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"best val RMSE: {best_rmse:.3f}  -> {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
