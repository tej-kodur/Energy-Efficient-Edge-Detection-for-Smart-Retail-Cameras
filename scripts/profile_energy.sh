#!/usr/bin/env bash
# Usage: ./scripts/profile_energy.sh weights.pth TEED|PIDINET
# Requires nvidia-smi and a single GPU

CKPT=$1
NAME=$2
LOG=results/energy_${NAME}.csv
DUMMY=tmp_dummy.py

python - <<'PY' "$CKPT" "$NAME" "$DUMMY"
import torch, time, sys, subprocess
from src.models import TEEDNet, PiDiNetLite
device=torch.device("cuda")
ckpt=sys.argv[1]; name=sys.argv[2]; dummy=sys.argv[3]
net = TEEDNet(ckpt) if name=="TEED" else PiDiNetLite(ckpt)
net.to(device).eval()
x=torch.randn(16,3,256,256, device=device)
with torch.no_grad():
    for _ in range(20): net(x); torch.cuda.synchronize()
open(dummy,"w").write("pass")
PY

nvidia-smi dmon -s pucv -d 100 -o TD > "$LOG" &
PID=$!
python $DUMMY > /dev/null
sleep 1
kill $PID
rm $DUMMY
echo "Power log saved to $LOG"
