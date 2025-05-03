"""
Thin wrapper that exposes TEED as a clean nn.Module
and hides the repo‑specific import gymnastics.
"""
from pathlib import Path
import sys, torch.nn as nn

# ensure TEED repo is importable
_REPO = Path(__file__).resolve().parent.parent.parent / "external" / "TEED"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
try:
    from TEED.ted import TED
except ModuleNotFoundError:
    from ted import TED  # fallback if repo cloned at project root


class TEEDNet(nn.Module):
    def __init__(self, pretrained: str | None = None):
        super().__init__()
        self.net = TED()
        if pretrained:
            self.net.load_state_dict(torch.load(pretrained, map_location="cpu"))

    def forward(self, x):
        outs = self.net(x)  # list[Tensor]
        return outs[-1]  # final side‑map  (1×H×W)
