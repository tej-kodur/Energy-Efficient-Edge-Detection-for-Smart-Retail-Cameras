#!/usr/bin/env python3
"""
Convert RPC COCO annotations into 3‑pixel contour labels (.npy).

Example:
  python scripts/preprocess_rpc.py \
      --json instances_val2019.json instances_test2019.json \
      --img_root data/rpc \
      --out_dir  data/rpc/edges_256 \
      --ids_file data/rpc/rpc_ids.txt
"""
import argparse, json, random, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

def mask_union(coco, img_id):
    info = coco.loadImgs(img_id)[0]
    h, w = info["height"], info["width"]
    full = np.zeros((h, w), np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        try:
            full |= coco.annToMask(ann)
        except Exception:
            x, y, bw, bh = map(int, ann["bbox"])
            full[y : y + bh, x : x + bw] = 1
    return full, info["file_name"]

def main(json_files, img_root, out_dir, ids_file):
    img_root = Path(img_root)
    out_dir  = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for jf in json_files:
        subset = Path(jf).stem.split("_")[1]  # val2019 / test2019
        coco = COCO(jf)
        for img_id in tqdm(coco.getImgIds(), desc=subset):
            union, name = mask_union(coco, img_id)
            edge = cv2.Canny(union * 255, 50, 150)
            edge = cv2.dilate(edge, None, 1)
            edge = cv2.resize(edge, (256, 256), cv2.INTER_NEAREST)
            stem = Path(name).stem
            uid = f"{subset}/{stem}"
            np.save(out_dir / f"{uid.replace('/','___')}_edge.npy", edge)
            ids.append(uid)
    Path(ids_file).write_text("\n".join(ids))
    print("Finished:", len(ids), "edge maps")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ids_file", required=True)
    args = ap.parse_args()
    main(args.json, args.img_root, args.out_dir, args.ids_file)
