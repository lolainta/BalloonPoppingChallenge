"""Leaderboard candidates: collect single-episode results from every source, rank them (pops desc, last pop asc),
copy the weights of the top ones to runs/candidates/ and re-run them deterministically to prove they reproduce.

Sources: per-seed deterministic checkpoint evals (runs/*/det_eval_<step>.txt), seed-search / eval logs
(logs/eval_<job>_<task>.log), and stochastic training episodes (runs/*/best_episodes.csv, snapshot weights).
Output: runs/leaderboard_candidates.csv (verified_* columns = deterministic re-run from the copied weights).

.venv/bin/python dev/candidates.py [--loop 300] [--top 15] [--procs 8]
"""
import argparse, csv, glob, hashlib, os, re, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("BPC_ROOT", Path.home() / "bpc"))  # BPC_ROOT: local mirror (dev/local_services.sh)
OUT = ROOT / "runs/leaderboard_candidates.csv"
SAFE = ROOT / "runs/candidates"
COLS = ["pops", "last_pop", "source", "agent", "kw", "seed", "verified_pops", "verified_last_pop", "weights_copy"]
MIN_POPS = 11  # only leaderboard-relevant results are kept, secured and verified (user: record >= 11 only)
LINE = re.compile(r"^seed\s+(\d+)\s+popped\s+(\d+)\s+last_pop\s+(\S+)")


def agent_for(kw, header_agent=None):
    if header_agent:
        return header_agent
    return "rl2" if "policy_path" in kw else ("beam" if "leg_dt" in kw else "gnc")


def collect():
    rows = []
    for f in glob.glob(str(ROOT / "runs/*/det_eval_*.txt")):  # deterministic checkpoint evals
        run, step = Path(f).parent.name, Path(f).stem.split("_")[-1]
        kw = f"policy_path=runs/{run}/rl2_policy_{step}.npz use_sl=0"  # files written before the kw header existed
        lines = open(f).read().splitlines()
        if lines and lines[0].startswith("kw "):
            kw = lines[0][3:].strip()
        ag = lines[1].split()[1] if len(lines) > 1 and lines[1].startswith("agent ") else "rl2"
        for line in lines:
            m = LINE.match(line)
            if m:
                rows.append(dict(pops=int(m[2]), last_pop=m[3], source=f"det_eval {run}@{step}", agent=ag, kw=kw, seed=m[1]))
    for f in glob.glob(str(ROOT / "logs/eval_*_*.log")):  # eval.sbatch logs (seed searches, held-out sweeps)
        text = open(f, errors="ignore").read().splitlines()
        if not text or not text[0].startswith("variant"):
            continue
        kw = text[0].split(":", 1)[1].strip()
        ha = text[1].split()[1] if len(text) > 1 and text[1].startswith("agent ") else None
        job = Path(f).stem.split("_")[1]
        for line in text:
            m = LINE.match(line)
            if m:
                rows.append(dict(pops=int(m[2]), last_pop=m[3], source=f"eval job {job}", agent=agent_for(kw, ha), kw=kw, seed=m[1]))
    for f in []:  # stochastic training episodes are not reproducible: excluded (user decision 2026-09-24)
        run = Path(f).parent.name
        for r in csv.DictReader(open(f)):
            kw = ""
            if r.get("snapshot"):
                sl = f"runs/{run}/{r['snapshot'].replace('.npz', '_sl.npz')}"
                kw = f"policy_path=runs/{run}/{r['snapshot']} " + (f"sl_path={sl}" if (ROOT / sl).exists() else "use_sl=0")
            rows.append(dict(pops=int(float(r["pops"])), last_pop=r["last_pop"], source=f"training {run}@{r['step']} (stochastic)",
                             agent="rl2", kw=kw, seed=r["seed"]))
    return rows


def lp(r):
    try:
        return float(r["last_pop"])
    except (TypeError, ValueError):
        return 1e9


def secure_weights(r):
    """Copy the weight files a candidate needs into runs/candidates/ and point its kw at the copies."""
    SAFE.mkdir(parents=True, exist_ok=True)
    kw, copies = r["kw"], []
    m = re.search(r"policy_path=(\S+)", kw)
    if m and (ROOT / m[1]).exists():
        src = ROOT / m[1]
        dst = SAFE / f"{src.parent.name}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        kw = kw.replace(m[1], str(dst.relative_to(ROOT)))
        copies.append(str(dst.relative_to(ROOT)))
    m = re.search(r"sl_path=(\S+)", kw)
    if m and (ROOT / m[1]).exists():
        src = ROOT / m[1]
        dst = SAFE / f"{src.parent.name}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        kw = kw.replace(m[1], str(dst.relative_to(ROOT)))
        copies.append(str(dst.relative_to(ROOT)))
    if r["agent"] == "beam":  # Beam depends on the learned leg model file: pin it by content hash
        src = ROOT / "BalloonPoppingGymEnv/agents/leg_model.npz"
        h = hashlib.md5(src.read_bytes()).hexdigest()[:8]
        dst = SAFE / f"leg_model_{h}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
        copies.append(str(dst.relative_to(ROOT)))
    return kw, " ".join(copies)


def verify(r):
    if not r["kw"] and r["agent"] == "rl2":
        return "no-weights", "", ""
    kw, copies = secure_weights(r)
    env = dict(os.environ, AGENT=r["agent"], OMP_NUM_THREADS="1")
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "dev/run_seeds.py", "--seeds", r["seed"], "--kw", *kw.split()],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200).stdout
    for line in out.splitlines():
        m = LINE.match(line)
        if m:
            return m[2], m[3], copies
    return "error", "", copies


def step(top, procs):
    old = {}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            old[(r["source"], r["agent"], r["kw"], r["seed"])] = r
    rows = []
    for r in collect():
        k = (r["source"], r["agent"], r["kw"], r["seed"])
        prev = old.get(k, {})
        r.update(verified_pops=prev.get("verified_pops", ""), verified_last_pop=prev.get("verified_last_pop", ""),
                 weights_copy=prev.get("weights_copy", ""))
        rows.append(r)
    rows = [r for r in rows if r["pops"] >= MIN_POPS]
    rows.sort(key=lambda r: (-r["pops"], lp(r)))
    todo = [r for r in rows[:top] if not r["verified_pops"]]
    with ThreadPoolExecutor(procs) as ex:
        for r, (vp, vl, cp) in zip(todo, ex.map(verify, todo)):
            r.update(verified_pops=vp, verified_last_pop=vl, weights_copy=cp)
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows[:500])
    tmp.replace(OUT)
    print(time.strftime("%T"), f"{len(rows)} results, verified {len(todo)} new; best: "
          f"{rows[0]['pops']} pops seed {rows[0]['seed']} ({rows[0]['source']})" if rows else "no results", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = run once")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    while True:
        step(a.top, a.procs)
        if not a.loop:
            break
        time.sleep(a.loop)
