"""Debug: policy's masked-current prediction vs its actual pick after the pop (one seed)."""
import contextlib, io, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chain_free as CF  # noqa: E402


class Dbg(CF.FreeChain):
    def decide(self, obs, action):
        st = obs["balloon_status"][:, 0]; t = float(obs["simulation_time"])
        popped_now = self.cmd is not None and int((st == 2).sum()) > self.n_popped
        if popped_now:
            x = self.features(obs); tl = self.forward(x)[0]
            print(f"t={t:.2f} POP (cur was {self.cmd[0]}, st={st[self.cmd[0]]}) pending={self._pending} pick={action[0]} top5 now {list(np.argsort(-tl)[:5])} "
                  f"prev top5 masked {self._prev_top}")
        super().decide(obs, action)
        if self.cmd is not None:
            x = self.features(obs); cur = self.cmd[0]
            x[CF.G_DIM + cur * CF.F + 14] = 0.0
            tl = self.forward(x)[0]; self._prev_top = list(np.argsort(-tl)[:5])


params, given = CF.C.rl2_env.load_scenario_parameters(1)
env = CF.C.rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, _ = env.reset(seed=20000)
ag = Dbg(given, mode=0, policy_path=CF.C.CK, sl_path=CF.C.CK.replace("rl3_policy", "rl2_sl")); ag._prev_top = None
done = False
while not done:
    obs, r, term, trunc, inf = env.step(ag.get_action(obs)); done = term or trunc
