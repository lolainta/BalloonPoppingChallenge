"""Submission selection at scale: every candidate flies the same worlds (one reset per seed, dev/sweep_gnc.py), seeds split
into chunks = Slurm array tasks (dev/qual_eval.sbatch, 16 cores each). Files in logs/qual_TAG/: cands.txt, cfgs.txt,
meta.json, part_<task>.csv, task_<task>.log.
  .venv/bin/python dev/qual_select.py submit --cands C.txt --seeds 30000-31023 --chunk 32 --tag sel1 [--qos devq_low]
      [--exclude cp14] [--throttle 20]       (C.txt lines: "name policy.npz sl.npz [k=v ...]", '#' comments)
  .venv/bin/python dev/qual_select.py report --tag sel1 [--ref name]
report: missing / truncated rows, per-candidate mean +- SE and paired diff vs ref (seeds that have every candidate),
ranking, and the per-chunk means (drift between chunks). Pure stdlib + numpy.
"""
import argparse, csv, glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_cands(path):
    out = []
    for l in Path(path).read_text().splitlines():
        f = l.split("#")[0].split()
        if not f:
            continue
        if len(f) < 3 or any("=" in x for x in f[:3]) or any("=" not in x for x in f[3:]):
            sys.exit(f"bad candidate line {l!r}: name policy.npz sl.npz [k=v ...]")
        out.append((f[0], " ".join([f"policy_path={f[1]}", f"sl_path={f[2]}"] + f[3:])))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate candidate names {names}")
    return out


def submit(a):
    d = ROOT / f"logs/qual_{a.tag}"
    if (d / "meta.json").exists():
        sys.exit(f"{d} exists: pick a new --tag")
    cands = read_cands(a.cands)
    for _, c in cands:
        for p in c.split()[:2]:
            if not (ROOT / p.split("=", 1)[1]).is_file():  # relative to the repo root (or absolute)
                sys.exit(f"missing {p}")
    lo, hi = map(int, a.seeds.split("-")); n = -(-(hi - lo + 1) // a.chunk)
    d.mkdir(parents=True)
    (d / "cands.txt").write_text(Path(a.cands).read_text())
    (d / "cfgs.txt").write_text("".join(c + "\n" for _, c in cands))
    (d / "meta.json").write_text(json.dumps(dict(lo=lo, hi=hi, chunk=a.chunk, names=[x for x, _ in cands],
                                                 cfgs=[c for _, c in cands]), indent=1))
    cmd = ["sbatch", f"--array=0-{n - 1}" + (f"%{a.throttle}" if a.throttle else ""), f"--qos={a.qos}",
           f"-J", f"bpc-qual-{a.tag}", "-o", f"logs/qual_{a.tag}/task_%a.log",
           f"--export=ALL,QTAG={a.tag},QLO={lo},QHI={hi},QCHUNK={a.chunk}"] + \
          ([f"--exclude={a.exclude}"] if a.exclude else []) + ["dev/qual_eval.sbatch"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"{len(cands)} candidates x seeds {lo}-{hi} = {n} tasks of {a.chunk} seeds -> {d}")


def report(a):
    d = ROOT / f"logs/qual_{a.tag}"
    m = json.loads((d / "meta.json").read_text())
    names, cfgs, lo, hi, ch = m["names"], m["cfgs"], m["lo"], m["hi"], m["chunk"]
    K, seeds = len(names), list(range(lo, hi + 1))
    rows, trunc, bad, dup = {}, [], [], []
    parts = sorted(glob.glob(str(d / "part_*.csv")))
    for p in parts:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                s, j = int(r["seed"]), int(r["cfg"])
                if j >= K or r["config"] != cfgs[j] or not lo <= s <= hi:
                    bad.append(f"{os.path.basename(p)} seed {s} cfg {j}"); continue
                if (s, j) in rows:
                    dup.append(f"seed {s} {names[j]}")
                rows.setdefault((s, j), (float(r["pops"]), float(r["last_pop"]) if r["last_pop"] else np.nan))
                if r["truncated"] == "1":
                    trunc.append(f"seed {s} {names[j]}")
    print(f"tag {a.tag}: {K} candidates, seeds {lo}-{hi} ({len(seeds)}), {len(parts)} parts, {len(rows)} rows")
    if dup:
        print(f"DUPLICATE {len(dup)} rows (first one kept): " + ", ".join(dup))
    if bad:
        print(f"IGNORED {len(bad)} rows not matching meta.json: " + ", ".join(bad))
    miss = {s: [names[j] for j in range(K) if (s, j) not in rows] for s in seeds}
    gone = [s for s in seeds if len(miss[s]) == K]
    part = [f"seed {s}: {' '.join(miss[s])}" for s in seeds if 0 < len(miss[s]) < K]
    if gone or part:
        print(f"MISSING: {len(gone)} seeds without any row" + (f" ({fmt_seeds(gone)})" if gone else "")
              + "".join(f"\n  {x}" for x in part))
    if trunc:
        print(f"TRUNCATED by max_wall_time ({len(trunc)}): " + ", ".join(trunc))
    S = [s for s in seeds if not miss[s]]
    if not S:
        sys.exit("no seed has every candidate")
    P = np.array([[rows[s, j][0] for j in range(K)] for s in S])
    T = np.array([[rows[s, j][1] for j in range(K)] for s in S])
    ref = names.index(a.ref) if a.ref else 0
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")  # noqa: E731
    w = max(len(n) for n in names)
    print(f"\n{len(S)} complete seeds; paired diff vs {names[ref]}; ranked by mean pops")
    for j in sorted(range(K), key=lambda j: -P[:, j].mean()):
        d_ = P[:, j] - P[:, ref]
        print(f"  {names[j]:{w}s}  mean {P[:, j].mean():.3f} +- {se(P[:, j]):.3f}  diff {d_.mean():+.3f} +- {se(d_):.3f}"
              f"  (+{(d_ > 0).sum()} ={(d_ == 0).sum()} -{(d_ < 0).sum()})  last_pop {np.nanmean(T[:, j]):.1f}s")
    print(f"\nper chunk of {ch} seeds (complete seeds only): mean pops")
    print(f"  {'chunk':12s} {'n':>3s}  " + "  ".join(f"{n[:10]:>10s}" for n in names))
    c = (np.array(S) - lo) // ch
    for k in np.unique(c):
        q = P[c == k]
        print(f"  {lo + k * ch:5d}-{min(hi, lo + k * ch + ch - 1):<6d} {len(q):3d}  " + "  ".join(f"{x:10.3f}" for x in q.mean(0)))


def fmt_seeds(s):
    out, a = [], s[0]
    for x, y in zip(s, s[1:] + [None]):
        if y != x + 1:
            out.append(f"{a}" if a == x else f"{a}-{x}"); a = y
    return ", ".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit"); s.add_argument("--cands", required=True); s.add_argument("--seeds", required=True)
    s.add_argument("--chunk", type=int, required=True); s.add_argument("--tag", required=True)
    s.add_argument("--qos", default="devq_low"); s.add_argument("--exclude"); s.add_argument("--throttle", type=int)
    r = sp.add_parser("report"); r.add_argument("--tag", required=True); r.add_argument("--ref")
    a = ap.parse_args()
    submit(a) if a.cmd == "submit" else report(a)
