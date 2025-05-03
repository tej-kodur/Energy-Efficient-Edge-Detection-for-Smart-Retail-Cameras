"""
Wrapper for PiDiNet tiny‑L (carv4).
"""
from pathlib import Path
import sys, torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent / "external" / "pidinet"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models import pidinet as _pidinet


class Args:  # minimal arg namespace replicating repo CLI
    config = "carv4"
    dil = False
    sa = False
    se = False
    classes = 1
    in_ch = 3


class PiDiNetLite(nn.Module):
    def __init__(self, pretrained: str | None = None):
        super().__init__()
        self.net = _pidinet(Args)
        if pretrained:
            self.net.load_state_dict(torch.load(pretrained, map_location="cpu"))

    def forward(self, x):
        outs = self.net(x)  # list[Tensor]
        return outs[-1]
