from pathlib import Path
import cv2, numpy as np, torch, random
from torch.utils.data import Dataset


class RPC(Dataset):
    """
    Retail‑Product‑Checkout checkout dataset with pre‑computed contour edges.
    """

    def __init__(
        self,
        id_file: str,
        img_root: str,
        edge_root: str,
        train: bool = True,
        split: float = 0.9,
        shuffle: bool = True,
    ):
        ids = Path(id_file).read_text().splitlines()
        if shuffle:
            random.Random(42).shuffle(ids)
        cut = int(len(ids) * split)
        self.ids = ids[:cut] if train else ids[cut:]
        self.img_root = Path(img_root)
        self.edge_root = Path(edge_root)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]  # e.g. val2019/XXXX
        img = cv2.imread(str(self.img_root / f"{uid}.jpg"), cv2.IMREAD_COLOR)
        edge = np.load(self.edge_root / f"{uid.replace('/','___')}_edge.npy")
        img = cv2.resize(img, (256, 256))
        edge = cv2.resize(edge, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        edge = torch.tensor(edge / 255.0).unsqueeze(0).float()
        return img, edge
