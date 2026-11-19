"""Did the 'hold' variant engage?  One free CoolBa run: logged t_go and commanded f in the last 0.6 s of burn, and the closest
approach to the chased target after burnout, for base vs hold."""
import contextlib, io, os, sys
from pathlib import Path
os.environ["SLACK"] = "inf"
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v_hold as vh  # noqa: E402

def go(variant, name="10_CoolBa"):
    info = vh.rt.replay_info(str(vh.ROOT.parent / "lb_replays" / f"{name}.json"))
    params, given = vh.rl2_env.load_scenario_parameters(1)
    env = vh.rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = vh.mixin(vh.RL3Agent, variant)(given, policy_path=vh.CK, sl_path=vh.CK.replace("rl3_policy", "rl2_sl"))
    done, rows, tl = False, [], None
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        t = float(obs["simulation_time"]); R = np.asarray(inf["rocket_states"], float)
        if tl is None and np.isfinite(R[0]): tl = t
        dbg = ag.debug or {}
        if tl is not None and 29.3 <= t - tl <= 31.5:
            tg = dbg.get("target"); last = getattr(ag, "_hold_last", None)
            b = obs["balloon_states"][int(tg if tg is not None else (last[0] if last else 0))]
            rows.append((t - tl, None if tg is None else int(tg), None if last is None else round(last[1], 2),
                         np.round(dbg.get("f", [np.nan] * 3), 1), np.linalg.norm(b[:3] - R[:3])))
    return rows

for v in ("base", "hold"):
    rows = go(v)
    print(f"--- {v}")
    for r in rows[::10][:12]:
        print(f"  t+{r[0]:.2f} tgt {r[1]} tgo {r[2]} f {r[3]} dist {r[4]:.2f}")
    print(f"  min dist to chased balloon t+29.3..31.5: {min(r[4] for r in rows):.2f} m")
