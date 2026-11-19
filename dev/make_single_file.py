"""Build a self-contained RL3 agent file for submission: gnc_agent + rl2_agent + rl3_agent merged, policy
weights embedded (base64 npz). The leaderboard JSON packs only agent_module_path, so the whole agent has to
live in one file. Only depends on numpy + the official BaseAgent / get_initial_attitude.

.venv/bin/python dev/make_single_file.py runs/ppo8/rl3_policy_702400.npz BalloonPoppingGymEnv/agents/bpc_rl3_agent.py [SL.npz]
(pass the SL reach model the checkpoint was trained/evaluated with, e.g. runs/ppo9/rl2_sl_<step>.npz)
--kw "ho_keep=1.0 redecide=2" bakes default agent kwargs into the file (an explicit agent_kwargs entry still wins);
values parse as in dev/sweep_gnc.py. Refused: ens and *_path (the file has one embedded policy / SL), unknown names.
--ens a.npz,b.npz (§508) embeds ensemble members (rl3_agent ens=, same order; their SL must equal SL.npz), e.g. with
--kw "ens_agg=prob".
"""
import argparse
import base64
import re
import sys
from pathlib import Path

AG = Path(__file__).resolve().parents[1] / "BalloonPoppingGymEnv/agents"


def body(src, drop_imports):
    """Source without its module docstring and without the given import lines."""
    src = re.sub(r'\A""".*?"""\n', "", src, count=1, flags=re.S)
    return "\n".join(l for l in src.splitlines() if not any(l.startswith(d) for d in drop_imports)) + "\n"


def bake(items, srcs):
    """["k=v", ...] -> {k: float or str} (dev/run_seeds.parse_kw); refuses what the single file cannot honour."""
    kw = {}
    for it in items:
        k, v = it.split("=")
        try:
            kw[k] = float(v)
        except ValueError:
            kw[k] = v
        if k == "ens" or k.endswith("_path"):
            sys.exit(f"--kw {k}: the single file embeds exactly one policy and SL model")
        if not re.search(rf'kw\.get\(\s*"{k}"', srcs):
            sys.exit(f"--kw {k}: not an agent kwarg (no kw.get(\"{k}\") in the inlined sources)")
    return kw


def main(weights, out, sl=None, launch_rule=None, kw=(), ens=()):
    gnc = body((AG / "gnc_agent.py").read_text(), [])
    rl2 = body((AG / "rl2_agent.py").read_text(), ["from pathlib import Path", "import numpy as np",
                                                    "from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent",
                                                    "from BalloonPoppingGymEnv.agents.dp_plan import"])
    dpp = body((AG / "dp_plan.py").read_text(), ["import numpy as np"])  # DP planner, inlined before RL2
    # the file-based SL loading is removed: the reach model (if given) is embedded below (_SL_B64) instead
    rl2 = re.sub(r'        sl_path = Path\(kw\.get\("sl_path".*?self\.sl = load_sl\(sl_path\)\n', "", rl2, flags=re.S)
    assert "load_sl" not in rl2, "SL block not removed"
    rl3 = body((AG / "rl3_agent.py").read_text(), ["from pathlib import Path", "import numpy as np",
                                                    "from BalloonPoppingGymEnv.agents import rl2_agent as v2",
                                                    "from BalloonPoppingGymEnv.agents.rl2_agent import",
                                                    "from BalloonPoppingGymEnv.agents.gnc_agent import"])
    # RL3's own globals would shadow RL2's: rename them, and point v2.* at the RL2 names
    rl3 = re.sub(r"\bG_DIM\b", "G3_DIM", rl3)
    rl3 = re.sub(r"\bN_OBS\b", "N3_OBS", rl3)
    rl3 = rl3.replace("v2.G3_DIM", "G_DIM")
    assert "v2." not in rl3, "unresolved v2 reference"
    if launch_rule:  # default launch rule baked into the file (the eval cfg's agent_kwargs can still override it)
        o = 'self.launch_rule = kw.get("launch_rule", "argmax")'
        assert o in rl3, "launch_rule default not found"
        rl3 = rl3.replace(o, f'self.launch_rule = kw.get("launch_rule", "{launch_rule}")')
    baked = bake(kw, gnc + dpp + rl2 + rl3)
    if launch_rule and "launch_rule" in baked:
        sys.exit("give launch_rule either positionally or in --kw, not both")
    o = '        kw.setdefault("policy_path", Path(__file__).with_name("rl3_policy.npz"))\n'
    assert o in rl3, "policy_path default not found"
    rl3 = rl3.replace(o, ("        kw = {**_BAKED_KW, **kw}  # baked defaults (make_single_file --kw); explicit kwargs win\n"
                          if baked else "") + '        kw.setdefault("policy_path", "__embedded__")\n')
    rl3 = rl3.replace("        super().__init__(given_parameters, **kw)\n",
                      "        super().__init__(given_parameters, **kw)\n"
                      "        if self.net is None and kw.get(\"use_policy\", 1):\n"
                      "            self.net = _embedded_weights()\n"
                      "        if _SL_B64 and kw.get(\"use_sl\", 1):  # learned reach model the policy was trained with\n"
                      "            self.sl = load_sl(io.BytesIO(base64.b64decode(_SL_B64)))\n", 1)
    if ens:  # ensemble members: same checks as rl3_agent's ens= loader, then appended after its (empty) file loop
        import numpy as np
        w0, s0 = np.load(weights), Path(sl).read_bytes() if sl else None
        for p in ens:
            m, sib = np.load(p), Path(p).with_name(Path(p).name.replace("rl3_policy", "rl2_sl"))
            assert set(m.files) == set(w0.files) and all(m[k].shape == w0[k].shape for k in m.files), f"--ens {p}: architecture differs"
            assert all(np.array_equal(m[k], w0[k]) for k in m.files if w0[k].ndim == 0), f"--ens {p}: config differs"
            assert s0 is not None and sib.is_file() and sib.read_bytes() == s0, f"--ens {p}: SL {sib} missing or != {sl}"
        o = "            self.ens.append(n)\n"
        assert rl3.count(o) == 1, "ens loader not found"
        rl3 = rl3.replace(o, o + "        self.ens += [_embedded_member(i) for i in range(len(_ENS_B64))]  # make_single_file --ens\n")
    ens_src = (f"_ENS_B64 = {[base64.b64encode(Path(p).read_bytes()).decode() for p in ens]!r}  # ensemble members, ens= order\n\n\n"
               "def _embedded_member(i):\n"
               "    return {k: v for k, v in np.load(io.BytesIO(base64.b64decode(_ENS_B64[i]))).items()}\n\n\n") if ens else ""
    slsrc =body((AG / "rl2_sl.py").read_text(), ["import numpy as np"]) if sl else ""
    sl_b64 = base64.b64encode(Path(sl).read_bytes()).decode() if sl else ""
    b64 = base64.b64encode(Path(weights).read_bytes()).decode()
    text = f'''"""BPC 2026 guidance agent (single file): RL3 = attention policy over all balloons that picks the target, the
approach speed level and the launch time; GNC (navigation, ZEM guidance, attitude control) flies it.

Uses only observations and given_parameters. Policy weights are embedded below (trained offline with PPO,
see the team's training code). Generated by dev/make_single_file.py from {Path(weights).name}{"".join(" + " + Path(p).name for p in ens)}.
"""
import base64
import io
from pathlib import Path

import numpy as np

_WEIGHTS_B64 = "{b64}"
_SL_B64 = "{sl_b64}"  # empty = no learned reach model
{f"_BAKED_KW = {baked!r}{chr(10)}" if baked else ""}

def _embedded_weights():
    return {{k: v for k, v in np.load(io.BytesIO(base64.b64decode(_WEIGHTS_B64))).items()}}


{ens_src}# ============================================================================ learned reach model (rl2_sl.py)
{slsrc}
# ============================================================================ GNC base (gnc_agent.py)
{gnc}
# ============================================================================ DP planner (dp_plan.py)
{dpp}
# ============================================================================ RL2 (rl2_agent.py)
{rl2}
# ============================================================================ RL3 (rl3_agent.py)
{rl3}'''
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text) / 1e3:.0f} kB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("out"); ap.add_argument("sl", nargs="?"); ap.add_argument("launch_rule", nargs="?")
    ap.add_argument("--kw", default="", help='baked default kwargs, e.g. "ho_keep=1.0 redecide=2"')
    ap.add_argument("--ens", default="", help="ensemble members to embed (rl3_agent ens=), comma-separated npz")
    a = ap.parse_args()
    main(a.weights, a.out, a.sl, a.launch_rule, a.kw.split(), [p for p in a.ens.split(",") if p])
