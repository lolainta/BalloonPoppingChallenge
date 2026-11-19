"""Run the official evaluate.main() on one seed without editing scenario_4_parameters.yaml (other processes share it):
load_scenario_parameters is wrapped to set scenario.random_seed; the env class is subclassed only to time reset and
note the last pop time. Everything else is the unmodified evaluator path.
.venv/bin/python dev/qual_check/eval_seed.py SEED   (from the repo root)
(finale: CFG=<secret-free cfg yaml> SC=40 [HOUR=h: the launch hour, which the finale draws before evaluation]; any fresh seed)"""
import sys, time, os
T_PROC = time.time()
sys.path.insert(0, os.getcwd())
from BalloonPoppingGymEnv.evaluation import evaluate as ev
from dev.seed_registry import record

seed = int(sys.argv[1])
CFG, SC = os.environ.get("CFG", "dev/qual_check/qual_cfg.yaml"), int(os.environ.get("SC", 4))
HOUR = os.environ.get("HOUR")  # scenario 40: environment.date[3], not in the given parameters
assert 10000 <= seed <= 10003 or "CFG" in os.environ
record([seed], "evaluate.py", f"qual_check eval_seed.py: {CFG} hour {HOUR}, secret-free cfg", SC)
orig_load, Env = ev.load_scenario_parameters, ev.BalloonPoppingEnv
stats = {}


def load(n):
    p, g = orig_load(n)
    p["scenario"]["random_seed"] = seed
    if HOUR is not None:
        p["environment"]["date"][3] = int(HOUR)
    return p, g


class TimedEnv(Env):
    def reset(self, *a, **k):
        stats["reset_start"] = time.time()
        out = super().reset(*a, **k)
        stats["reset_end"] = time.time()
        return out

    def step(self, action):
        o, r, te, tr, info = super().step(action)
        if r > 0:
            stats["last_pop"] = float(o["simulation_time"])
        if te or tr:
            stats["episode_end"] = time.time(); stats["pops"] = info["popped_count"]; stats["t_end"] = float(o["simulation_time"])
        return o, r, te, tr, info


ev.load_scenario_parameters, ev.BalloonPoppingEnv = load, TimedEnv
t_main = time.time()
ev.main([CFG])
t_done = time.time()
cpu = time.process_time()
print(f"RESULT seed {seed} hour {HOUR} pops {stats['pops']} last_pop {stats.get('last_pop')} sim_end {stats['t_end']:.2f}s | "
      f"reset {stats['reset_end'] - stats['reset_start']:.1f}s episode {stats['episode_end'] - stats['reset_end']:.1f}s "
      f"reset->end {stats['episode_end'] - stats['reset_start']:.1f}s main() {t_done - t_main:.1f}s "
      f"process {t_done - T_PROC:.1f}s cpu {cpu:.1f}s margin_vs_600 {600 - (stats['episode_end'] - stats['reset_start']):.0f}s", flush=True)
