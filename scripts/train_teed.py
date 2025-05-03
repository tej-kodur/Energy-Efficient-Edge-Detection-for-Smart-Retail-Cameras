#!/usr/bin/env python3
"""
Train TEED ultra‑tiny model:
  1. warm‑up on BIPED (10 epochs)
  2. fine‑tune on RPC  (12 epochs)
  3. save best checkpoint to results/checkpoints/teed_best.pth
"""
import argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.data import BIPED, RPC
from src.models import TEEDNet


def train(model, loader, epochs, lr, device):
    opt = optim.AdamW(model.parameters(), lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    for ep in range(1, epochs + 1):
        tot = 0
        model.train()
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            outs = model(x)  # final map
            loss = loss_fn(outs, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0)
        print(f"epoch {ep:02d}/{epochs}  loss {tot/len(loader.dataset):.4f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TEEDNet().to(device)

    # datasets
    biped = BIPED(args.biped_root, "train")
    rpc   = RPC(args.rpc_ids, args.rpc_img_root, args.rpc_edge_root, train=True)
    dl_biped = DataLoader(biped, batch_size=32, shuffle=True, num_workers=2)
    dl_rpc   = DataLoader(rpc,   batch_size=16, shuffle=True, num_workers=2)

    train(model, dl_biped, epochs=10, lr=1e-3, device=device)
    train(model, dl_rpc,   epochs=12, lr=1e-3, device=device)

    Path(args.ckpt.parent).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    print("✓ saved", args.ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--biped_root", required=True)
    ap.add_argument("--rpc_ids", required=True)
    ap.add_argument("--rpc_img_root", required=True)
    ap.add_argument("--rpc_edge_root", required=True)
    ap.add_argument("--ckpt", default="results/checkpoints/teed_best.pth", type=Path)
    main(ap.parse_args())
