"""Can the policy net learn the rollout teacher's target choices from its own features? (DECISION_LOG §219)
Pure behaviour cloning (TEACH_BC's loss alone) of the fork-point net on 80% of the demos (split by demo); on the other 20%:
does it pick the teacher's target where the teacher overrode the original policy ("override" rows), and keep the original
choice where it agreed ("agree" rows)? Train accuracy rising while held-out override accuracy stays ~0 = memorising.
   ENVFILE=<learner_submit.sh> .venv/bin/python dev/bc_probe.py <async_last.pt> <demo dir> [steps] [lr]
(the RL3_* settings are read from ENVFILE so the net and losses are the learner's; demos of other policies are skipped)"""
import glob, os, re, sys
from pathlib import Path

for k, v in re.findall(r"\b(RL3_[A-Z0-9_]+|TEACH_BC|LAUNCH)=([^,\s]+)", open(os.environ["ENVFILE"]).read()):
    os.environ.setdefault(k, v)
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev import async_ppo as A
from dev.rl3_net import Net3

ck, ddir = sys.argv[1], sys.argv[2]
steps, lr = int(sys.argv[3]) if len(sys.argv) > 3 else 3000, float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
torch.manual_seed(0); rng = np.random.default_rng(0)
net = Net3(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["net"])
sizes, tg = A.dims(True), (lambda X: {"tgt": X[:, 0].long()})
D = [d for d in (dict(np.load(f)) for f in sorted(glob.glob(f"{ddir}/[0-9]*.npz"))) if "rl3_policy_280225306" in str(d["policy"])]
test = set(rng.permutation(len(D))[: len(D) // 5].tolist())
rows = [(d["obs"][i], d["act"][i], j in test) for j, d in enumerate(D) for i in range(len(d["act"])) if d["act"][i, 1] != 3]
O = torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32)
Aa = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32)
te = torch.tensor([r[2] for r in rows]); lab = Aa[:, 0].long()


def stats():
    with torch.no_grad():
        tl = net(O, **tg(Aa))[0][:, : sizes[0]]
    p = torch.softmax(tl, -1); hit = tl.argmax(1) == lab
    return hit, -(p * torch.log(p + 1e-12)).sum(1)


hit0, _ = stats(); over = ~hit0  # the original policy's choice != the teacher's
print(f"{len(D)} demos ({len(test)} held out), {len(rows)} flight rows, override rows {int(over.sum())} "
      f"(train {int((over & ~te).sum())}, test {int((over & te).sum())})")
opt = torch.optim.Adam(net.parameters(), lr=lr)
tr = torch.nonzero(~te)[:, 0]
for k in range(steps + 1):
    if k % 500 == 0:
        hit, H = stats()
        f = lambda m: f"{hit[m].float().mean():.2f}"  # noqa: E731
        print(f"step {k:5d}  train: override {f(over & ~te)} agree {f(~over & ~te)}   test: override {f(over & te)} "
              f"agree {f(~over & te)}   entropy train {H[~te].mean():.3f} test {H[te].mean():.3f}", flush=True)
    b = tr[torch.randint(len(tr), (256,))]
    loss = A.sil_loss(net, O[b], Aa[b], torch.zeros(len(b)), sizes, tg, bc=True)
    opt.zero_grad(); loss.backward(); opt.step()
