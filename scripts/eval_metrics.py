#!/usr/bin/env python3
"""
Evaluate TEED & PiDiNet checkpoints on RPC validation split.
Outputs:
  results/metrics.csv
  results/figs/sample_grid.png
  results/figs/tradeoff.png
"""
import argparse, json, random, cv2, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
from src.data import RPC
from src.models import TEEDNet, PiDiNetLite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def f1_iou(pred, gt):
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return float(f1), float(iou)


def eval_model(model, loader, thr):
    model.eval(); f1s=[]; ious=[]
    with torch.no_grad():
        for x, y in loader:
            o = model(x.to(device))
            p = torch.sigmoid(o).cpu() > thr
            for pp, yy in zip(p, y.bool()):
                f, i = f1_iou(pp.squeeze(), yy.squeeze())
                f1s.append(f); ious.append(i)
    return np.mean(f1s), np.mean(ious)


def main(args):
    # ─ datasets
    rpc_val = RPC(args.rpc_ids, args.rpc_img_root, args.rpc_edge_root, train=False)
    val_loader = DataLoader(rpc_val, batch_size=8)

    # ─ load models
    teed = TEEDNet(args.teed_ckpt).to(device)
    pidi = PiDiNetLite(args.pidi_ckpt).to(device)

    rows = []
    for name, mdl, thr in [("TEED", teed, 0.07), ("PiDiNet-Lite", pidi, 0.08)]:
        f1, iou = eval_model(mdl, val_loader, thr)
        macs, params = profile(mdl, inputs=(torch.randn(1,3,256,256).to(device),))
        rows.append(dict(Model=name, F1=f1, IoU=iou, Params=params, GFLOPs=macs/1e9))

    # Canny baseline on 400 val frames
    f1s=[];ious=[]
    for uid in random.sample(rpc_val.ids, 400):
        img = cv2.imread(str(rpc_val.img_root/f"{uid}.jpg"), cv2.IMREAD_GRAYSCALE)
        gt  = np.load(rpc_val.edge_root/f"{uid.replace('/','___')}_edge.npy")>0
        pred= cv2.Canny(cv2.resize(img,(256,256)),50,150)>0
        f,i = f1_iou(pred, gt); f1s.append(f); ious.append(i)
    rows.append(dict(Model="Canny", F1=np.mean(f1s), IoU=np.mean(ious), Params=0, GFLOPs=0))

    # ─ save CSV
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True); Path("results/figs").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/metrics.csv", index=False)
    print(df)

    # ─ trade‑off plot
    plt.figure(figsize=(4.5,4))
    plt.scatter(df.GFLOPs, df.F1)
    for _,r in df.iterrows():
        plt.text(r.GFLOPs, r.F1, r.Model, fontsize=8, ha='right', va='bottom')
    plt.xlabel("GFLOPs @256²"); plt.ylabel("F‑score"); plt.title("Accuracy vs Compute")
    plt.grid(); plt.tight_layout()
    plt.savefig("results/figs/tradeoff.png", dpi=200)

    # ─ qualitative grid (3 samples)
    samples = random.sample(rpc_val.ids, 3)
    fig,axs=plt.subplots(3,4,figsize=(9,6))
    for row,uid in enumerate(samples):
        rgb = cv2.imread(str(rpc_val.img_root/f"{uid}.jpg"))
        gt  = np.load(rpc_val.edge_root/f"{uid.replace('/','___')}_edge.npy")
        pred= torch.sigmoid(teed(torch.tensor(cv2.resize(rgb,(256,256)))
                          .permute(2,0,1).unsqueeze(0).float().to(device)/255.)).cpu().numpy()[0,0]>0.07
        overlay = cv2.addWeighted(cv2.resize(rgb,(256,256)),0.8,
                                  cv2.cvtColor((pred*255).astype(np.uint8),cv2.COLOR_GRAY2BGR),0.2,0)
        for col,mat,title,cmap in [
            (0,rgb,"RGB",None),
            (1,gt,"GT edge",'gray'),
            (2,pred*255,"TEED pred",'gray'),
            (3,overlay,"Overlay",None)]:
            axs[row,col].imshow(mat,cmap=cmap); axs[row,col].axis('off'); axs[row,col].set_title(title,fontsize=9)
    plt.tight_layout(); plt.savefig("results/figs/sample_grid.png", dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rpc_ids", required=True)
    p.add_argument("--rpc_img_root", required=True)
    p.add_argument("--rpc_edge_root", required=True)
    p.add_argument("--teed_ckpt", default="results/checkpoints/teed_best.pth")
    p.add_argument("--pidi_ckpt", default="results/checkpoints/pidinet_best.pth")
    main(p.parse_args())
