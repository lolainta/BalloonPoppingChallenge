"""Terminal rocket lander game on the official simulator (no balloons, no targets).

Launch, fly around, and touch down gently before the 30 s burn runs out.

    uv run python dev/game.py [scenario]   # 1 = competition physics (default), 0 = no wind, 3 = + actuator lag

Keys: SPACE launch | w/s throttle | arrows tilt (E/W, N/S) | c center tilt | r restart | q quit
The autopilot only holds the tilt you ask for; where the rocket goes is up to you.
"""

import contextlib
import curses
import io
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, quat_to_R  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

FPS = 20
MAX_TILT = 35.0  # deg
SOFT_LANDING = 3.0  # m/s


SCENARIO = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def new_game():
    params, given = load_scenario_parameters(SCENARIO)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=0)
    return env, GNCAgent(given), params


def draw(scr, st):
    scr.erase()
    h, w = scr.getmaxyx()
    plot_h, plot_w = h - 7, w - 2
    x_span = max(150.0, abs(st["pos"][0]) * 1.2)
    z_top = max(200.0, st["alt"] * 1.2)

    def cell(x, z):
        c = int((x + x_span) / (2 * x_span) * (plot_w - 1))
        r = plot_h - 1 - int(z / z_top * (plot_h - 1))
        return r, c

    ground_r, pad_c = cell(0, 0)
    scr.addstr(ground_r + 1, 0, "_" * (w - 1))
    scr.addstr(ground_r + 1, max(pad_c - 2, 0), "[==]")
    for x, z in st["trail"][-400::4]:
        r, c = cell(x, z)
        if 0 <= r < plot_h and 0 <= c < plot_w:
            scr.addstr(r, c, ".")
    r, c = cell(st["pos"][0], st["alt"])
    nose_e = st["nose"][0]
    glyph = "^" if abs(nose_e) < 0.2 else ("/" if nose_e > 0 else "\\")
    if 0 <= r < plot_h and 0 <= c < plot_w:
        scr.addstr(r, c, glyph, curses.A_BOLD)

    v = st["vel"]
    info = (
        f"t={st['t']:5.1f}s  fuel={st['fuel']:4.1f}s  thr={st['throttle']:.2f}  "
        f"tilt E={st['tilt'][0]:+5.1f} N={st['tilt'][1]:+5.1f}deg"
    )
    info2 = (
        f"alt={st['alt']:6.1f}m  x(E)={st['pos'][0]:+7.1f}  y(N)={st['pos'][1]:+7.1f}  "
        f"vz={v[2]:+6.1f}  vh={np.hypot(v[0], v[1]):5.1f} m/s  [scenario {SCENARIO}]"
    )
    scr.addstr(h - 5, 0, info[: w - 1])
    scr.addstr(h - 4, 0, info2[: w - 1])
    scr.addstr(h - 3, 0, st["msg"][: w - 1], curses.A_BOLD)
    scr.addstr(h - 2, 0, "SPACE launch  w/s throttle  arrows tilt  c center  r restart  q quit"[: w - 1])
    scr.refresh()


def play(scr):
    curses.curs_set(0)
    scr.nodelay(True)
    while True:
        scr.erase()
        scr.addstr(0, 0, f"Loading scenario {SCENARIO} (wind field from the competition data)...")
        scr.refresh()
        env, pilot, params = new_game()
        burn = params["rocket"]["motor"]["burn_time"]
        dt = params["simulation"]["time_step"]
        steps = max(1, round(1 / FPS / dt))
        launched_at, throttle, tilt = None, 1.0, np.zeros(2)
        st = {"t": 0.0, "fuel": burn, "throttle": throttle, "tilt": tilt, "alt": 0.0,
              "pos": np.zeros(3), "vel": np.zeros(3), "nose": np.array([0, 0, 1.0]),
              "trail": [], "msg": "Press SPACE to launch"}
        terminated = truncated = False
        launch = False
        while not (terminated or truncated):
            frame_start = time.monotonic()
            key = scr.getch()
            while key != -1:
                if key == ord("q"):
                    return
                if key == ord("r"):
                    truncated = True
                if key == ord(" "):
                    launch = True
                if key == ord("w"):
                    throttle = min(throttle + 0.05, 1.0)
                if key == ord("s"):
                    throttle = max(throttle - 0.05, 0.0)
                if key == curses.KEY_RIGHT:
                    tilt[0] = min(tilt[0] + 5, MAX_TILT)
                if key == curses.KEY_LEFT:
                    tilt[0] = max(tilt[0] - 5, -MAX_TILT)
                if key == curses.KEY_UP:
                    tilt[1] = min(tilt[1] + 5, MAX_TILT)
                if key == curses.KEY_DOWN:
                    tilt[1] = max(tilt[1] - 5, -MAX_TILT)
                if key == ord("c"):
                    tilt[:] = 0
                key = scr.getch()
            if truncated:
                break

            for _ in range(steps):
                s = env._rocket_states  # game display only, not a competition agent
                tvc, roll = np.zeros(2), 0.0
                if launched_at is not None and not np.isnan(s[6]):
                    pilot.q = s[6:10]
                    te, tn = np.radians(tilt)
                    want = np.array([np.sin(te), np.sin(tn), 1.0])
                    tvc, roll = pilot._attitude_control(want, s[10:13])
                action = {
                    "launch": launch,
                    "launch_inclination_heading": np.array([90.0, 0.0]),
                    "tvc": tvc,
                    "roll": float(roll),
                    "throttle": throttle,
                }
                obs, _, terminated, truncated, info = env.step(action)
                if launch and launched_at is None:
                    launched_at = obs["simulation_time"]
                if terminated or truncated:
                    break

            s = info["rocket_states"]
            st["t"] = obs["simulation_time"]
            st["throttle"], st["tilt"] = throttle, tilt
            if launched_at is not None and not np.isnan(s[0]):
                st["pos"], st["vel"] = s[:3], s[3:6]
                st["alt"] = s[2] - params["environment"]["elevation"]
                st["nose"] = quat_to_R(s[6:10])[:, 2]
                st["trail"].append((s[0], st["alt"]))
                st["fuel"] = max(burn - (st["t"] - launched_at), 0.0)
                st["msg"] = "BURNOUT - falling!" if st["fuel"] == 0 else "Flying..."
            draw(scr, st)
            time.sleep(max(0.0, 1 / FPS - (time.monotonic() - frame_start)))

        if terminated:
            speed = np.linalg.norm(st["vel"])
            st["msg"] = (
                f"Touchdown at {speed:.1f} m/s, {np.hypot(*st['pos'][:2]):.0f} m from pad - "
                + ("SOFT LANDING!" if speed <= SOFT_LANDING else "CRASH")
                + "   (r = again, q = quit)"
            )
            draw(scr, st)
            scr.nodelay(False)
            while (key := scr.getch()) not in (ord("r"), ord("q")):
                pass
            scr.nodelay(True)
            if key == ord("q"):
                return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    curses.wrapper(play)
