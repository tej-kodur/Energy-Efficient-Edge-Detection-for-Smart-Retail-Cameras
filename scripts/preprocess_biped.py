#!/usr/bin/env python3
"""
Build 256x256 edge‑tensor pairs for the BIPED dataset.

Usage:
  python scripts/preprocess_biped.py \
        --raw_dir data/biped \
        --out_dir data/biped/edges_256
"""
import argparse, cv2, glob, os
from pathlib import Path
import numpy as np
from tqdm import tqdm

def main(raw_dir: str, out_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(glob.glob(str(raw/"imgs"/"train"/"**"/*.png"), recursive=True))
    edge_paths= sorted(glob.glob(str(raw/"edges"/"edge_maps"/"train"/"**"/*.png"), recursive=True))
    for img_p,edge_p in tqdm(zip(img_paths, edge_paths), total=len(img_paths)):
        img  = cv2.imread(img_p, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_p, cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (256,256))
        edge = cv2.resize(edge, (256,256))
        np.save(out/f"{Path(img_p).stem}_img.npy",  img.astype(np.uint8))
        np.save(out/f"{Path(edge_p).stem}_edge.npy",edge > 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.raw_dir, args.out_dir)
