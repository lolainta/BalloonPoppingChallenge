"""Time one PPO-sized update (5760 samples x 4 epochs, minibatch 1440) of the RL2 attention net on a device."""
import sys, time
sys.path.insert(0, ".")
import torch
from dev.rl2_net import Net
from BalloonPoppingGymEnv.agents.rl2_agent import N_OBS, F, G_DIM, N_MAX
dev = sys.argv[1]
torch.set_num_threads(8)
net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-4)
X = torch.randn(5760, N_OBS, device=dev); X.view(5760, -1)[:, G_DIM:].view(5760, N_MAX, F)[..., 14] = (torch.rand(5760, N_MAX, device=dev) < 0.6).float()
def upd():
    for _ in range(4):
        for j in torch.randperm(5760, device=dev).split(1440):
            tl, sl, v = net(X[j]); loss = tl.logsumexp(1).mean() + sl.logsumexp(1).mean() + v.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
upd(); t0 = time.time(); upd(); print(f"{dev}: one PPO update {time.time() - t0:.2f}s", flush=True)
with torch.no_grad():
    t0 = time.time()
    for _ in range(20): net(X[:90])
    if dev == "cuda": torch.cuda.synchronize()
    print(f"{dev}: rollout forward (90 obs) {1000 * (time.time() - t0) / 20:.1f} ms", flush=True)
