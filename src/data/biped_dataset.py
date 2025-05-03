from pathlib import Path
import glob, cv2, torch
from torch.utils.data import Dataset


class BIPED(Dataset):
    """
    Loads BIPED edge‑detection dataset.
    Expected layout:
      data/biped/imgs/<split>/rgbr/real/*.png
      data/biped/edges/edge_maps/<split>/rgbr/real/*.png
    """

    def __init__(self, root: str, split: str = "train"):
        assert split in ("train", "test")
        root = Path(root)
        imgs_glob = root / "imgs" / split
        edges_glob = root / "edges" / "edge_maps" / split

        self.img_paths = sorted(
            glob.glob(str(imgs_glob / "**" / "*.png"), recursive=True)
            + glob.glob(str(imgs_glob / "**" / "*.jpg"), recursive=True)
        )
        self.edge_paths = sorted(
            glob.glob(str(edges_glob / "**" / "*.png"), recursive=True)
            + glob.glob(str(edges_glob / "**" / "*.jpg"), recursive=True)
        )
        assert len(self.img_paths) == len(
            self.edge_paths
        ), "Image/edge count mismatch"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        edge = cv2.imread(self.edge_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        edge = cv2.resize(edge, (256, 256))
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        edge = torch.tensor(edge / 255.0).unsqueeze(0).float()
        return img, edge
