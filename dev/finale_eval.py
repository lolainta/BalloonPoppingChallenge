"""TASTI finale (11/8, scenario #40; DECISION_LOG §504): one official evaluation per drawn seed, at the drawn launch hour.
    .venv/bin/python dev/finale_eval.py SEED HOUR              (or: uv run python dev/finale_eval.py SEED HOUR)
    .venv/bin/python dev/finale_eval.py SEED HOUR --cfg CFG    (your own eval cfg instead, e.g. one that packs an upload file)
1. A seed in our record (data/seed_registry/used_seeds.csv and the newest for_organizers list, where present) or in
   [100000, 10000000) (pre-9/29 training draws there were not all logged) is refused: ask for a redraw. --force runs it.
2. Writes the seed and the hour into scenario_40_parameters.yaml (scenario.random_seed, environment.date[3]: the official
   way; the file is put back afterwards) and reads them back the way evaluate.py loads it. The hour is not a given
   parameter in scenario 40, so the agent never sees it.
3. Runs the unmodified `python BalloonPoppingGymEnv/evaluation/evaluate.py CFG` with this checkout first on PYTHONPATH.
   Default CFG = dev/qual_check/final_cfg.yaml with agent_module_path = --agent (secret-free, leaderboard_submission
   false). A --cfg must say scenario_number: 40.
4. Prints pops, the time of the last pop (the tie-break, from the saved trajectory) and the wall time against the 600 s
   limit; records the seed (dev/seed_registry.py) and appends a row to logs/finale_eval.tsv.
"""
import argparse, glob, gzip, hashlib, json, os, re, socket, subprocess, sys, time
from datetime import datetime, timezone
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
Y = "BalloonPoppingGymEnv/envs/scenario_parameters/scenario_40_parameters.yaml"
REG = os.environ.get("SEED_REGISTRY") or "data/seed_registry/used_seeds.csv"  # where dev/seed_registry.py writes
KNOWN = {"851bd0dbe0bd2fe5a8b8dfafadcfca80": "finale file pqp (probability ensemble), official 16/16 (§512)",
         "78ca384e54b34170198a804cf3d0b586": "previous finale file phe, official 16/16 (§500)",
         "a5fcaf325f2525ab7dd1430176dc75e9": "fallback cur"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("seed", type=int)
ap.add_argument("hour", type=int, choices=range(24), metavar="HOUR", help="launch hour 0-23 (UTC) as drawn")
g = ap.add_mutually_exclusive_group()
g.add_argument("--agent", default="BalloonPoppingGymEnv/agents/bpc_s40_fin_pqp.py")  # §512 (was bpc_s40_fin_phe.py)
g.add_argument("--cfg")
ap.add_argument("--force", action="store_true", help="run a seed the check refuses (the organisers' call)")
a = ap.parse_args()
if re.fullmatch(r"login\d+|cp\d+", socket.gethostname().split(".")[0]) and ROOT == os.path.realpath(os.path.expanduser("~/bpc")):
    sys.exit("ABORT: the cluster's shared ~/bpc (running jobs read this yaml): use a private copy")

why, lists = [], sorted(glob.glob("data/seed_registry/for_organizers/*.txt.gz"))
if os.path.exists(REG) and any(l.split(",", 1)[0] == str(a.seed) for l in open(REG)):
    why.append(f"in {REG}")
if lists and any(l.strip() == str(a.seed) for l in gzip.open(lists[-1], "rt")):
    why.append(f"in {lists[-1]}")
if 100000 <= a.seed < 10000000:
    why.append("in [100000, 10000000), where pre-9/29 training seeds were not all logged")
if not os.path.exists(REG) and not lists:
    print("WARNING: no seed record in this checkout; the organisers check against the list we sent")
if why:
    print(f"SEED {a.seed}: {'; '.join(why)} -> ask the organisers for a redraw" + ("" if a.force else " (--force runs it)"))
    if not a.force:
        sys.exit(1)

os.makedirs("logs", exist_ok=True)
cfg = a.cfg or f"logs/finale_eval_{a.seed}.cfg.yaml"
if not a.cfg:
    src = open("dev/qual_check/final_cfg.yaml", encoding="utf-8").read()
    src = re.sub(r"(?m)^agent_module_path: .*", lambda m: f"agent_module_path: {a.agent}", src)
    src = re.sub(r"(?m)^agent_name: .*", lambda m: f'agent_name: "{os.path.basename(a.agent)} (finale, seed {a.seed})"', src)
    open(cfg, "w", encoding="utf-8").write(src)
c = yaml.safe_load(open(cfg, encoding="utf-8-sig"))
agent = c["agent_module_path"]
if c.get("scenario_number") != 40 or (not a.cfg and agent != a.agent) or not os.path.isfile(agent):
    sys.exit(f"ABORT: {cfg}: scenario_number {c.get('scenario_number')} (needs 40), agent_module_path {agent} (must exist)")
md5 = hashlib.md5(open(agent, "rb").read()).hexdigest()
print(f"agent {agent}  md5 {md5}  ({KNOWN.get(md5, 'not a file we verified')})")
try:
    from dev.seed_registry import record  # fcntl: not on Windows
    record([a.seed], "finale_eval.py", f"official evaluate.py, hour {a.hour}, {cfg}, agent md5 {md5}", 40)
except ImportError as e:
    print(f"WARNING: seed not recorded ({e})")

orig = open(Y, "rb").read()
txt, n1 = re.subn(r"(?m)^(  random_seed:)[^#\r\n]*", lambda m: f"{m[1]} {a.seed} ", orig.decode("utf-8-sig"))
txt, n2 = re.subn(r"(?m)^(  date: \[[^,\]]*,[^,\]]*,[^,\]]*,)[^\]]*\]", lambda m: f"{m[1]} {a.hour}]", txt)
log = f"logs/finale_eval_{a.seed}.log"
try:
    with open(Y, "w", encoding="utf-8", newline="") as f:
        f.write(txt)
    p = yaml.safe_load(open(Y, encoding="utf-8-sig"))
    if not (n1 == n2 == 1 and p["scenario"]["random_seed"] == a.seed and p["environment"]["date"][3] == a.hour):
        sys.exit(f"ABORT: could not set the seed / hour in {Y} ({n1} {n2})")
    print(f"{Y}: random_seed {p['scenario']['random_seed']}, date {p['environment']['date']} (UTC)")
    cmd = [sys.executable, "BalloonPoppingGymEnv/evaluation/evaluate.py", cfg]
    print("$", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.time()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as pr, open(log, "w") as f:
        for line in pr.stdout:
            sys.stdout.write(line); f.write(line)
    rc, wall = pr.returncode, time.time() - t0
finally:
    open(Y, "wb").write(orig)

out = open(log, encoding="utf-8", errors="replace").read()
tot = re.findall(r"Total reward: (\d+)", out)
pops = int(tot[-1]) if tot else None
tj = max((f for f in glob.glob("BalloonPoppingGymEnv/evaluation/results/*_trajectory.json") if os.path.getmtime(f) >= t0),
         key=os.path.getmtime, default=None)
last = None
if tj:
    steps = json.load(open(tj))
    n = [sum(s == 2 for s in st["balloon_status"]) for st in steps]  # 2 = popped
    if n and n[-1]:
        last = next(st["time"] for st, k in zip(steps, n) if k == n[-1])
    if n and n[-1] != pops:
        print(f"WARNING: the trajectory has {n[-1]} pops, the log says {pops}")
lp = "-" if last is None else f"{last:.2f} s"
hit = "   WALL-TIME LIMIT HIT" if "Reached max wall time" in out else ""
print(f"\n==== seed {a.seed}  hour {a.hour}:  pops {pops}   last pop at {lp}   wall {wall:.0f} s (whole run; limit 600 s){hit}"
      f"   exit {rc} ====\nlog {log}   trajectory {tj}")
T = "logs/finale_eval.tsv"
new = not os.path.exists(T)
with open(T, "a") as f:
    if new:
        f.write("utc\tseed\thour\tpops\tlast_pop_s\twall_s\texit\tagent\tmd5\tcfg\tlog\ttrajectory\n")
    f.write("\t".join(map(str, [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), a.seed, a.hour, pops, last,
                                f"{wall:.1f}", rc, agent, md5, cfg, log, tj])) + "\n")
sys.exit(rc)
