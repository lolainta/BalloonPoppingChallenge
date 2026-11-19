"""Dev: record one run to an mp4 (scenario 1, or SCENARIO=4 etc.).

[SCENARIO=1] [AGENT=gnc|beam|rl2|rl3|cur] [HOUR=h] uv run python dev/record.py SEED OUT.mp4 [k=v ...]   (kwargs as in run_seeds)
AGENT=cur = the finale single file (bpc_s40_fb_cur.py); HOUR = scenario-40 launch hour (environment.date[3]).
OUT.pkl instead of OUT.mp4: simulate only and dump the frames (cluster has no ffmpeg) -> render later with OUT.pkl OUT.mp4.
"""

import contextlib
import io
import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

SCENARIO = int(os.environ.get("SCENARIO", 1))
FRAME_EVERY = 10  # sim steps per frame (0.1 s) -> 2x real time at 20 fps


def simulate(seed):
    params, given = load_scenario_parameters(SCENARIO)
    if os.environ.get("HOUR"):
        params["environment"]["date"][3] = int(os.environ["HOUR"])
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=seed)
    from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
    from BalloonPoppingGymEnv.agents.rl2_agent import RL2Agent
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from BalloonPoppingGymEnv.agents.bpc_s40_fb_cur import RL3Agent as CurAgent
    def val(k, v):  # numbers, except paths and string options (e.g. launch_rule=median)
        if k.endswith("_path"):
            return v
        try:
            return float(v)
        except ValueError:
            return v
    kw = {k: val(k, v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
    agent = {"gnc": GNCAgent, "beam": BeamAgent, "rl2": RL2Agent, "rl3": RL3Agent, "cur": CurAgent}[os.environ.get("AGENT", "gnc")](given, **kw)
    if kw.get("sl_path"):
        from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
        agent.sl = load_sl(kw["sl_path"])
    frames, pops = [], []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(agent.get_action(obs))
        step += 1
        t = obs["simulation_time"]
        if reward:
            pops.append((t, info["rocket_states"][:3].copy()))
        if step % FRAME_EVERY == 0 or terminated or truncated:
            frames.append(
                (t, info["rocket_states"][:3].copy(), obs["balloon_states"][:, :3].copy(),
                 obs["balloon_status"][:, 0].copy(), info["popped_count"])
            )
    return frames, pops, params["environment"]["elevation"]


def dump(seed, out):  # ponytail: pickle of (frames, pops, elev); one file per episode
    frames, pops, elev = simulate(seed)
    pickle.dump((frames, pops, elev), open(out, "wb"))
    print(f"seed {seed}: popped {frames[-1][4]} -> {out}")


def render(seed, out, src=None):
    if src:
        frames, pops, elev = pickle.load(open(src, "rb"))
    else:
        frames, pops, elev = simulate(seed)
    launch_i = next(i for i, f in enumerate(frames) if not np.isnan(f[1][0]))
    end_t = frames[-1][0]
    frames = [f for f in frames if f[0] >= frames[launch_i][0] - 3]  # 3 s before launch

    rocket = np.array([f[1] for f in frames])
    ok = ~np.isnan(rocket[:, 0])
    pts = np.vstack([rocket[ok]] + [f[2][f[3] >= 1] for f in frames[:: max(len(frames) // 20, 1)]])
    lo, hi = pts.min(0) - 10, pts.max(0) + 10
    lo[2] = elev

    # presentation style (11/6, finale talk): 16:9 dark, pop flashes, HUD counter; the camera orbits slowly
    bg = "#0b1020"
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=bg)
    ax = fig.add_subplot(projection="3d", facecolor=bg)
    ax.set_position([0.0, -0.05, 1.0, 1.1])
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (hi[2] - lo[2]) * 1.2))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0.03)), axis._axinfo["grid"].update(color=(1, 1, 1, 0.08), linewidth=0.6)
        axis.set_tick_params(colors="#8892b0", labelsize=8)
    ax.set_xlabel("East (m)", color="#8892b0"), ax.set_ylabel("North (m)", color="#8892b0"), ax.set_zlabel("Altitude ASL (m)", color="#8892b0")
    released = ax.scatter([], [], [], c="#ff6ec7", s=14, alpha=0.75, depthshade=False)
    flash = ax.scatter([], [], [], c="#ffd166", s=[], marker="*", alpha=0.95, depthshade=False)
    popped = ax.scatter([], [], [], c="#ff5c5c", s=40, marker="x", alpha=0.6, depthshade=False)
    (trail,) = ax.plot([], [], [], "-", color="#5ce1ff", lw=1.8)
    (head,) = ax.plot([], [], [], "^", color="white", ms=9)
    hud = fig.text(0.04, 0.90, "", color="white", fontsize=34, fontweight="bold", family="DejaVu Sans")
    sub = fig.text(0.04, 0.85, "", color="#8892b0", fontsize=16)
    tag = fig.text(0.96, 0.05, f"scenario {SCENARIO}  seed {seed}  hour {os.environ.get('HOUR', '-')}  {os.environ.get('AGENT', 'gnc')}",
                   color="#8892b0", fontsize=11, ha="right")
    if os.environ.get("NOHUD"):  # clean background plate (title slide)
        hud.set_alpha(0), sub.set_alpha(0), tag.set_alpha(0)
    pop_xyz = np.array([p for _, p in pops]) if pops else np.zeros((0, 3))
    pop_t = np.array([t for t, _ in pops])
    launch_t = frames[launch_i][0] if launch_i < len(frames) else frames[0][0]

    def update(i):
        t, pos, bpos, status, count = frames[i]
        up = bpos[status == 1]
        released._offsets3d = (up[:, 0], up[:, 1], up[:, 2])
        age = t - pop_t
        fl = (age >= 0) & (age < 1.5)
        shown = pop_xyz[fl]
        flash._offsets3d = (shown[:, 0], shown[:, 1], shown[:, 2])
        flash.set_sizes(400 * (1 - age[fl] / 1.5) ** 2 + 40)
        old = pop_xyz[age >= 1.5]
        popped._offsets3d = (old[:, 0], old[:, 1], old[:, 2])
        r = rocket[: i + 1][ok[: i + 1]]
        if len(r):
            trail.set_data_3d(r[:, 0], r[:, 1], r[:, 2])
            head.set_data_3d(r[-1:, 0], r[-1:, 1], r[-1:, 2])
        ax.view_init(elev=20, azim=-70 + 50 * i / max(len(frames) - 1, 1))
        hud.set_text(f"{count} popped")
        sub.set_text(f"t = {t:5.1f} s" + ("   (launch in %.1f s)" % (launch_t - t) if t < launch_t else ""))
        return released, flash, popped, trail, head, hud, sub

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out, writer=FFMpegWriter(fps=20, bitrate=6000), savefig_kwargs={"facecolor": bg})
    print(f"seed {seed}: popped {frames[-1][4]}, flight ended t={end_t:.1f}s, pops at {np.round(pop_t, 2).tolist()} -> {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from dev.seed_registry import record
    if sys.argv[2].endswith(".pkl") and len(sys.argv) > 3 and sys.argv[3].endswith(".mp4"):  # render a dump, no simulation
        render(int(sys.argv[1]), sys.argv[3], src=sys.argv[2])
        sys.exit()
    record([int(sys.argv[1])], "record", f"AGENT={os.environ.get('AGENT', 'gnc')} {' '.join(sys.argv[3:])} -> {sys.argv[2]}", SCENARIO)
    (dump if sys.argv[2].endswith(".pkl") else render)(int(sys.argv[1]), sys.argv[2])
