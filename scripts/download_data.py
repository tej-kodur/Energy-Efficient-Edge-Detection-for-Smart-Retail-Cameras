#!/usr/bin/env python3
"""
Download BIPED and RPC datasets directly from Kaggle Hub into ./data/.

Usage:
    python scripts/download_data.py     # downloads both
    python scripts/download_data.py --dataset biped
"""

import argparse, kagglehub, shutil, os, tarfile
from pathlib import Path


def download_kaggle(dataset_id: str, dest: Path):
    print(f"⤵  Downloading {dataset_id} …")
    tmp_path = kagglehub.dataset_download(dataset_id)
    tmp_path = Path(tmp_path)
    # kagglehub returns a directory; contents may be zipped/tarred
    for item in tmp_path.iterdir():
        if item.suffix in (".zip", ".tar", ".gz", ".tgz"):
            print("• extracting", item.name)
            to_extract = dest / dataset_id.split("/")[-1]
            to_extract.mkdir(parents=True, exist_ok=True)
            if item.suffix == ".zip":
                shutil.unpack_archive(item, to_extract)
            else:  # tar variations
                with tarfile.open(item) as tar:
                    tar.extractall(to_extract)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), dest / item.name)
    print("✓ finished", dataset_id)


def main():
    ROOT = Path(__file__).resolve().parent.parent / "data"
    dsets = {
        "biped":  "xavysp/biped",
        "rpc":    "diyer22/retail-product-checkout-dataset",
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dsets.keys(), default="all")
    args = ap.parse_args()

    if args.dataset == "all":
        for d in dsets.values():
            download_kaggle(d, ROOT)
    else:
        download_kaggle(dsets[args.dataset], ROOT)


if __name__ == "__main__":
    main()
