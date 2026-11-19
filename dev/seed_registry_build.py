"""Reconstruct every env seed our agent has met so far (organisers' request 2026-09-29: seeds used in training,
fine-tuning or testing are not drawn for our qualification / final evaluations; over-reporting is harmless,
under-reporting is the risk). Forward recording is dev/seed_registry.py + runs/<run>/episode_seeds.csv; this script
rebuilds the history from what is on disk, locally and on the cluster.

    .venv/bin/python dev/seed_registry_build.py            # rsync cluster text files, scan, write outputs
    .venv/bin/python dev/seed_registry_build.py --no-sync  # reuse the last cluster mirror

Outputs: data/seed_registry/history_seeds.csv (seed, category, scenario, n_files, source_file = first file seen)
         data/seed_registry/history_summary.md
The cluster is only read (rsync pull, find, one numpy read of the 'seed' arrays of data npz files). Never reads
BalloonPoppingGymEnv/evaluation/configs/; submission JSONs are only json-parsed for their random_seed.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BPC = REPO.parent  # ~/bpc
OUT = REPO / "data/seed_registry"
SCR = OUT / "_cluster_mirror"
HOST = "user@login"  # login02 refuses our logins since 2026-10-08
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST]
MAX_SPAN = 20000  # a "seeds A-B" range wider than this is reported, not expanded (e.g. the training draw range)
TRAIN_LO, TRAIN_HI = 100_000, 10_000_000
TEXT_EXT = {".txt", ".log", ".out", ".csv", ".json", ".jsonl", ".meta", ".md", ".sh", ".sbatch", ".py", ".spec", ""}
SKIP_DIRS = {".venv", ".venv-gpu", "__pycache__", "ActiveRocketPy", ".git", "_cluster_mirror", "node_modules",
             "t1data", "s4data", "launchdata", "_home"}
BIG = 30_000_000  # larger text files: only their random_seed (trajectory / replay JSONs)

# ---------------------------------------------------------------- cluster mirror


def sync():
    SCR.mkdir(parents=True, exist_ok=True)
    flt = ["--exclude=/BalloonPoppingGymEnv/", "--exclude=configs/", "--exclude=/.venv*/", "--exclude=/ActiveRocketPy/",
           "--exclude=/tests/", "--exclude=t1data*/", "--exclude=s4data/", "--exclude=__pycache__/",
           "--include=/logs/***", "--include=/runs/", "--include=/dev/", "--include=/data/"]
    flt += ["--include=/runs/**/", "--include=/dev/**/", "--include=/data/**/"]
    flt += [f"--include=*{e}" for e in (".txt", ".csv", ".log", ".out", ".json", ".jsonl", ".meta", ".md", ".sh", ".sbatch",
                                         ".py")]
    flt += ["--include=worker_env", "--include=workers.spec*", "--include=next_env/*", "--include=*seeds*",
            "--include=teach_run*", "--include=next_gen_env", "--exclude=*"]
    subprocess.run(["rsync", "-a", "--prune-empty-dirs", "-e", "ssh -o BatchMode=yes", *flt, f"{HOST}:bpc/", f"{SCR}/"],
                   check=True)
    shutil_rm(SCR / "BalloonPoppingGymEnv")  # belt and braces: never keep anything from the package dir
    home = SCR / "_home"; home.mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{HOST}:.bash_history", f"{home}/"], check=False)
    # binaries stay on the cluster: read their seed arrays there (numpy loads one key of an npz lazily)
    remote_py = r'''
import glob, os, numpy as np
for f in sorted(glob.glob("data/**/*.npz", recursive=True)):
    try:
        d = np.load(f)
        if "seed" in d.files: s = np.unique(np.asarray(d["seed"]).ravel())
        elif "pops" in d.files and np.asarray(d["pops"]).ndim == 2: s = np.unique(np.asarray(d["pops"])[:, 0])
        else: continue
        print(f + "\t" + ",".join(str(int(x)) for x in s))
    except Exception as e:
        print(f + "\tERR " + type(e).__name__)
for f in sorted(glob.glob("runs/*/demos*/*.npz")):
    print(f + "\tNAME")
'''
    r = subprocess.run(SSH + ["cd ~/bpc && .venv/bin/python -"], input=remote_py, capture_output=True, text=True, check=True)
    (SCR / "_remote_npz.tsv").write_text(r.stdout)


def shutil_rm(p):
    import shutil
    shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------- extraction

R_LINE = re.compile(r"(?im)^\s*seed[\s:=#]+(\d+)")
R_KV = re.compile(r"(?i)(?<![a-z_])seed(?!s)[_ ]?(?:=|:|#|\s)\s*(\d+)(?![\d.\-])")
R_ATT = re.compile(r"(?i)(?<![a-z])seed(\d{2,})(?!\d)")
R_RANGE = re.compile(r"(?i)(?:seeds?|種子)[^\d\n]{0,16}?(\d+)\s*(?:-|–|\.\.|~)\s*(\d+)")
R_LINE_RANGE = re.compile(r"(?<![\d.])(\d{3,})\s*[-–]\s*(\d{3,})(?![\d.])")
R_SEEDWORD = re.compile(r"(?i)seed|種子")
R_LIST = [re.compile(r"(?i)--seeds[= ]+([\d][\d,]*)"), re.compile(r"(?i)SEED_LIST=([\d][\d, ]*)"),
          re.compile(r"(?i)seeds\s*=\s*([\d][\d,]{3,})"), re.compile(r"(?i)seeds?\s*[:=]?\s*\[([\d,\s]+)\]")]
R_CMD = [re.compile(r"record\.py\s+(\d+)"), re.compile(r"search_teacher\.py\s+\S+\s+\S+\s+([\d ]+)")]
R_SCEN = re.compile(r"(?im)(?:^scenario\s+|--scenario[= ]|\bscenario[= ]|SCENARIO=|RL3_SCEN=|'scenario': )(\d)")
R_COUNT_ONLY = re.compile(r"(?im)^scenario \d base .* seeds (\d+)$")


class Registry:
    def __init__(self):
        self.rows = {}  # (seed, cat, scen) -> [n_files, first_source]
        self.wide = []  # (lo, hi, source)
        self.notes = defaultdict(list)

    def add(self, seeds, cat, src, scen=""):
        for s in set(seeds):
            if not 0 <= s < 2**32:
                continue
            k = (s, cat, str(scen))
            if k in self.rows:
                self.rows[k][0] += 1
            else:
                self.rows[k] = [1, src]

    def seeds(self, cat=None):
        return {k[0] for k in self.rows if cat is None or k[1] == cat}


def ranges_in(text, src, reg):
    out = []
    spans = [(m.group(1), m.group(2)) for m in R_RANGE.finditer(text)]
    for line in text.splitlines():
        if R_SEEDWORD.search(line) and len(line) < 2000:
            spans += R_LINE_RANGE.findall(line)
    for a, b in spans:
        lo, hi = int(a), int(b)
        if hi < lo or (len(a) > 4 and len(b) <= 2):  # "10000-3" style fragments
            continue
        if hi - lo > MAX_SPAN:
            reg.wide.append((lo, hi, src)); continue
        out += range(lo, hi + 1)
    return out


def text_seeds(text, src, reg):
    s = [int(x) for x in R_LINE.findall(text)] + [int(x) for x in R_KV.findall(text)] + [int(x) for x in R_ATT.findall(text)]
    for r in R_LIST + R_CMD:
        for m in r.findall(text):
            s += [int(x) for x in re.split(r"[,\s]+", m) if x.isdigit()]
    return s + ranges_in(text, src, reg)


def json_seeds(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "seed" in k.lower() and not k.lower().startswith(("pseed", "rng")):
                if isinstance(v, bool):
                    pass
                elif isinstance(v, int):
                    out.append(v)
                elif isinstance(v, list):
                    out += [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
                elif isinstance(v, str):
                    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", v)
                    if m and (not m.group(2) or int(m.group(2)) - int(m.group(1)) <= MAX_SPAN):
                        out += range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1)
            json_seeds(v, out)
    elif isinstance(obj, list):
        for v in obj:
            json_seeds(v, out)
    return out


def random_seed_only(path):
    """Submission / trajectory / replay JSON: the random_seed value(s), nothing else is kept or printed."""
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return []
    out = []

    def walk(o, depth=0):
        if depth > 3:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "random_seed" and isinstance(v, int):
                    out.append(v)
                elif isinstance(v, (dict, list)) and k != "trajectory":
                    walk(v, depth + 1)
        elif isinstance(o, list) and o and isinstance(o[0], dict) and depth < 2:
            for v in o[:50]:
                walk(v, depth + 1)
    walk(d)
    return out


def default_start(name):
    """First seed of the argparse default of the tool that writes this kind of log (dev/sweep_gnc.py 10000-10095,
    dev/dp_fidelity.py 14900-14915, dev/gnc2_eob/sweep_eob.py 30000-30191)."""
    if name.startswith("eob"):
        return 30000
    if name.startswith("dp_"):
        return 14900
    if name.startswith(("sw", "robust", "avg_cmp")) or "_cmp" in name or name.startswith("line"):
        return 10000
    return None


def scenario_of(rel, text):
    found = set(R_SCEN.findall(text[:200_000])) if text else set()
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        return "mixed"
    name = rel.lower()
    if re.search(r"(^|[/_])s4", name) or "_s4" in name:
        return "4"
    if R_LINE.search(text or "") and "popped" in (text or "")[:5000]:
        return "1"  # dev/run_seeds.py output without a scenario marker: its default, scenario 1
    return ""


def category_of(origin, rel):
    p, name = rel.lower(), Path(rel).name.lower()
    if origin == "decision_log":
        return "decision_log"
    if origin in ("fish_history", "bash_history", "claude_transcript"):
        return "shell_history"
    if name == "used_seeds.csv":
        return "live_registry"
    if name in ("best_episodes.csv", "episode_seeds.csv"):
        return "finetune" if re.search(r"runs/tune", p) else "training_logged"
    if re.search(r"runs/tune[^/]*/seeds[^/]*\.txt$", p) or name in ("worker_env",) or name.startswith("workers.spec"):
        return "finetune"
    if name.startswith(("teach_", "local_teacher")) or "demos" in p or "/teacher" in p or "search_teacher" in name:
        return "search_teacher"
    if "lb_replays" in p or name.startswith("lbseed_") or name.endswith("_submission.json") or "trajectory" in name \
            or "lb_lastpop" in name or "auto_submit" in p or "official_checks" in p or "single_file" in name:
        return "leaderboard"
    if name == "leaderboard_candidates.csv":
        return "candidates"
    if p.endswith(".mp4"):
        return "videos"
    if re.match(r"(legs|bc)_\d+\.(log|npz)$", name) or p.startswith(("data/legs", "data/r2/", "data/r3/", "data/bc/")):
        return "sl_data_collection"
    m = re.match(r"eval_(.+)_\d+\.log$", name)
    if (m and not m.group(1).isdigit()) or name.startswith(("hunt_", "seed_hunt")) \
            or re.search(r"(^|/)(runs/)?(hunt[^/]*|floor_seeds|easy_seeds)\.txt$", p):
        return "hunts"
    return "evaluation"


def walk_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not (d == "configs" and dp.endswith("evaluation"))]
        if dp.endswith(os.path.join("evaluation", "configs")):
            continue
        for fn in fns:
            yield Path(dp) / fn


def scan_tree(reg, origin, root, prefix=""):
    root = Path(root)
    if root.is_file():
        files = [root]; root = root.parent
    else:
        files = walk_files(root)
    for f in files:
        rel = prefix + str(f.relative_to(root))
        if rel.startswith("data/seed_registry/history") or "/evaluation/configs/" in "/" + rel \
                or rel in ("dev/seed_registry_build.py", "dev/seed_registry.py") or f.name.startswith("DECISION_LOG"):
            continue
        ext = f.suffix.lower()
        name = f.name.lower()
        src = f"{origin}:{rel}"
        if ext == ".mp4":
            parts = f.stem.split("_")
            s = parts[2] if parts[0] == "cand" and len(parts) > 2 else next((x for x in reversed(parts) if x.isdigit()), None)
            if s:
                reg.add([int(s)], "videos", src, "4" if parts[0] == "s4" else "1")
            continue
        if ext == ".npz":
            if re.match(r"(rl\d_policy|rl2_sl|live_policy|rl2_policy|init_|bc_only|runs_bc)", name) or "candidates" in rel:
                continue
            cat = category_of(origin, rel)
            if re.fullmatch(r"\d+\.npz", name):  # teacher demos: <seed>.npz
                reg.add([int(f.stem)], cat, src); continue
            try:
                d = np.load(f)
                if "seed" in d.files:
                    s = np.asarray(d["seed"]).ravel()
                elif "pops" in d.files and np.asarray(d["pops"]).ndim == 2:
                    s = np.asarray(d["pops"])[:, 0]
                else:
                    m = re.search(r"seed(\d+)", name)
                    s = [int(m.group(1))] if m else []
                reg.add([int(x) for x in np.unique(s)], cat, src, scenario_of(rel, ""))
            except Exception:
                pass
            continue
        if ext not in TEXT_EXT and not name.startswith("workers.spec"):
            continue
        try:
            size = f.stat().st_size
        except OSError:
            continue
        cat = category_of(origin, rel)
        if ext == ".json" and ("submission" in name or "trajectory" in name or "lb_replays" in rel or size > BIG):
            rs = random_seed_only(f)
            if not rs and "trajectory" in name and not re.search(r"seed\d+", name):
                reg.notes["traj_noseed"].append(src)
            m = re.search(r"seed(\d+)", name)
            reg.add(rs + ([int(m.group(1))] if m else []), cat, src, "1" if cat == "leaderboard" else "")
            continue
        if size > BIG:
            reg.notes["skipped_big"].append(src); continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        scen = scenario_of(rel, text)
        if name in ("best_episodes.csv", "episode_seeds.csv") and scen in ("", "mixed"):  # training: RL3_SCEN, default 1
            env = f.parent / "worker_env"
            scen = "4" if env.exists() and "RL3_SCEN=4" in env.read_text(errors="replace") else "1"
        seeds = []
        if ext == ".csv":
            rows = list(csv.reader(text.splitlines()))
            if rows:
                cols = [i for i, h in enumerate(rows[0]) if h.strip().lower() in ("seed", "random_seed")]
                if cols:
                    sc = [i for i, h in enumerate(rows[0]) if h.strip().lower() == "scenario"]
                    by = defaultdict(list)
                    for r in rows[1:]:
                        for i in cols:
                            if i < len(r) and r[i].strip().isdigit():
                                by[r[sc[0]] if sc and sc[0] < len(r) else scen].append(int(r[i]))
                    for k, v in by.items():
                        reg.add(v, cat, src, k)
                    continue
        if ext in (".json", ".jsonl"):
            objs = []
            try:
                objs = [json.loads(text)] if ext == ".json" else [json.loads(x) for x in text.splitlines() if x.strip()]
            except Exception:
                pass
            seeds += json_seeds(objs, [])
        if re.fullmatch(r"[\d,\s]+", text or "x") and re.search(r"seed|hunt|floor|easy", name):
            seeds += [int(x) for x in re.split(r"[,\s]+", text) if x]  # plain seed-list file
        seeds += text_seeds(text, src, reg)
        if not seeds and R_COUNT_ONLY.search(text):
            counts = R_COUNT_ONLY.findall(text)
            reg.notes["count_only"].append((src, counts))
            lo = default_start(name)  # the command line is lost: assume the tool's default first seed
            if lo is not None:
                n = -(-max(int(c) for c in counts) // 16) * 16  # 191 -> 192 (crashed resets are not counted)
                seeds = list(range(lo, lo + n))
                reg.notes["assumed"].append((src, lo, lo + n - 1))
        reg.add(seeds, cat, src, scen)


def scan_histories(reg):
    fish = Path.home() / ".local/share/fish/fish_history"
    if fish.exists():
        cmds = [l[7:] for l in fish.read_text(errors="replace").splitlines() if l.startswith("- cmd: ")]
        txt = "\n".join(c for c in cmds if re.search(r"bpc|Balloon|dev/|\.venv/bin/python", c))
        reg.add(text_seeds(txt, "fish_history", reg), "shell_history", "local:~/.local/share/fish/fish_history")
    for f in (Path.home() / ".bash_history", SCR / "_home/.bash_history"):
        if f.exists():
            o = "cluster:~/.bash_history" if SCR in f.parents else "local:~/.bash_history"
            reg.add(text_seeds(f.read_text(errors="replace"), o, reg), "shell_history", o)
    # Claude Code session transcripts of this project: every Bash command and its output (ad hoc run_seeds calls
    # whose output was never saved to a file only exist here)
    for f in glob.glob(str(Path.home() / ".claude/projects/*/**/*.jsonl"), recursive=True):
        chunks = []
        with open(f, errors="replace") as fh:
            for line in fh:
                if "seed" not in line.lower() and "種子" not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                c = (d.get("message") or {}).get("content")
                for part in c if isinstance(c, list) else []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "tool_use" and part.get("name") == "Bash":
                        chunks.append(str((part.get("input") or {}).get("command", "")))
                    elif part.get("type") == "tool_result":
                        cc = part.get("content")
                        chunks.append(cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc or []
                                                                                if isinstance(x, dict)))
        src = "local:" + f.replace(str(Path.home()), "~")
        reg.add(text_seeds("\n".join(chunks), src, reg), "shell_history", src)


def scan_remote_npz(reg):
    p = SCR / "_remote_npz.tsv"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        path, _, val = line.partition("\t")
        cat = category_of("cluster", path)
        if val == "NAME":
            m = re.match(r"\d+", Path(path).stem)  # 2026-11-06: "1062031_dt.npz" variants -> the leading seed number
            if m:
                reg.add([int(m[0])], cat, f"cluster:{path}")
        elif val and not val.startswith("ERR"):
            reg.add([int(x) for x in val.split(",")], cat, f"cluster:{path}",
                    "1" if cat == "sl_data_collection" else "")


def teacher_job_seeds(reg):
    """dev/search_teacher.sbatch drew its seeds with random.Random(SLURM_JOB_ID): regenerate the full list (200 = the
    most any job drew; the sequence is a prefix, so fewer launch-time branches only mean a shorter prefix)."""
    for f in (SCR / "logs").glob("teach_*.log"):
        job = f.stem.split("_")[1]
        if job.isdigit():
            r = random.Random(int(job))
            reg.add([r.randint(100000, 9999999) for _ in range(200)], "search_teacher", f"cluster:logs/{f.name} (regenerated)", "1")


def local_hunt_seeds(reg):
    """The 2026-09-25 local hunt (dev/local_seed_hunt.sh on /tmp/hunt_local.txt, since deleted) lost the per-seed output
    of 5 chunks that crashed in a reset: regenerate its seed file exactly (the command is in the session transcript:
    random.Random(20260925), 25,000 draws excluding runs/hunt_seeds.txt and 10000-10095; [:20000] = runs/hunt_big.txt,
    which this check reproduces; [20000:] = hunt_local.txt) and list all of it."""
    old_f, big_f = SCR / "runs/hunt_seeds.txt", SCR / "runs/hunt_big.txt"
    if not (old_f.exists() and big_f.exists()):
        return
    old = set(int(x) for x in old_f.read_text().replace(",", " ").split())
    r, out, seen = random.Random(20260925), [], set()
    while len(out) < 25000:
        x = r.randint(100000, 9999999)
        if x not in old and not (10000 <= x <= 10095) and x not in seen:
            out.append(x); seen.add(x)
    big = [int(x) for x in big_f.read_text().replace(",", " ").split()]
    if out[:20000] == big:  # regenerated exactly
        reg.add(out[20000:], "hunts", "local:/tmp/hunt_local.txt (regenerated; 2026-09-25 local hunt)", "1")
    else:
        print("WARNING: could not regenerate hunt_local.txt (hunt_big.txt mismatch)")


# ---------------------------------------------------------------- training-episode estimate


def training_estimate():
    """Episodes drawn by training workers per run, from progress.csv (steps = policy decisions) and status.json
    (decisions / episode from the learner's dec_per_s and eps_per_h). Cluster mirror first, local runs not on it."""
    runs = {}
    for base, origin in ((SCR / "runs", "cluster"), (REPO / "runs", "local")):
        for pc in sorted(base.glob("*/progress.csv")):
            run = pc.parent.name
            if run in runs:
                continue
            steps, prev = 0, None
            for row in csv.reader(pc.read_text().splitlines()):
                try:
                    s = int(float(row[0]))
                except (ValueError, IndexError):
                    continue
                steps += (s - prev) if prev is not None and s > prev else (0 if prev is not None else min(s, 19200))
                prev = s
            dpe, src = 60.0, "default"
            st = pc.parent / "status.json"
            if st.exists():
                try:
                    d = json.loads(st.read_text())
                    if d.get("eps_per_h") and d.get("dec_per_s"):
                        dpe, src = 3600 * d["dec_per_s"] / d["eps_per_h"], "status.json"
                except Exception:
                    pass
            scen = "4" if run.startswith("s4") else "1"
            env = pc.parent / "worker_env"
            if env.exists() and "RL3_SCEN=4" in env.read_text():
                scen = "4"
            seedlist = "tune" in run
            runs[run] = dict(origin=origin, steps=steps, dpe=dpe, dpe_src=src, eps=int(steps / dpe), scen=scen,
                             seedlist=seedlist)
    return runs


# ---------------------------------------------------------------- report


def contiguous(seeds, min_len):
    s = sorted(seeds); out = []; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        if j - i + 1 >= min_len:
            out.append((s[i], s[j]))
        i = j + 1
    return out


def fmt_ranges(rs):
    return ", ".join(f"{a}-{b} ({b - a + 1})" for a, b in rs)


def write(reg, runs):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "history_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "category", "scenario", "n_files", "source_file"])
        for (s, c, sc), (n, src) in sorted(reg.rows.items()):
            w.writerow([s, c, sc, n, src])
    allS = reg.seeds()
    cats = sorted({k[1] for k in reg.rows})
    core = {k[0] for k in reg.rows if k[1] not in ("decision_log", "shell_history")}  # files, not narratives
    L = ["# Seed history (reconstructed)", "",
         f"Generated by `dev/seed_registry_build.py`. Rows: {len(reg.rows)} in `history_seeds.csv` "
         "(one row per seed x category x scenario; `source_file` = first file it was found in, `n_files` = how many).", "",
         f"**Total distinct seeds: {len(allS)}** (min {min(allS)}, max {max(allS)}); "
         f"{sum(1 for s in allS if s < TRAIN_LO)} below {TRAIN_LO}, {sum(1 for s in allS if TRAIN_LO <= s < TRAIN_HI)} "
         f"inside the training draw range [{TRAIN_LO}, {TRAIN_HI}), {sum(1 for s in allS if s >= TRAIN_HI)} above it.", "",
         "## Distinct seeds per category", "", "| category | distinct seeds | min | max | files |", "|---|---:|---:|---:|---:|"]
    for c in cats:
        S = reg.seeds(c)
        nf = len({v[1] for k, v in reg.rows.items() if k[1] == c})
        L.append(f"| {c} | {len(S)} | {min(S)} | {max(S)} | >= {nf} |")
    L += ["", "## Distinct seeds per scenario", "", "(scenario '' = not stated in the source; 'mixed' = the file names "
          "several; a seed can appear under several scenarios)", "", "| scenario | distinct seeds |", "|---|---:|"]
    for sc in sorted({k[2] for k in reg.rows}):
        L.append(f"| {sc or '(unknown)'} | {len({k[0] for k in reg.rows if k[2] == sc})} |")
    big = contiguous(allS, 24)
    L += ["", "## Fully covered contiguous ranges (>= 24 consecutive seeds)", "",
          f"{len(big)} ranges, {sum(b - a + 1 for a, b in big)} seeds: " + fmt_ranges(big), ""]
    dl = reg.seeds("decision_log")
    miss = sorted(dl - core)
    L += ["## Cross-check against dev/DECISION_LOG.md", "",
          f"{len(dl)} distinct seeds (ranges expanded) mentioned in the decision log (and its copies); {len(dl & core)} also "
          "found in data / log files (shell history and session transcripts not counted). "
          f"{len(miss)} only in the log (kept in the registry anyway): "
          + (fmt_ranges(contiguous(miss, 2)) + " | singles: " + ", ".join(map(str, [s for s in miss if not any(a <= s <= b for a, b in contiguous(miss, 2))][:200]))
             if miss else "none"), ""]
    reg.wide = [w for w in reg.wide if w[0] >= 1000 and "/.claude/" not in w[2]]  # "seed 192 ... 5266218" fragments
    wide = sorted({(a, b) for a, b, _ in reg.wide})
    L += ["## Wide ranges seen but not expanded", "",
          f"Ranges wider than {MAX_SPAN} seeds are not listed seed by seed (they are draw ranges, not enumerations):", ""]
    for a, b in wide[:40]:
        srcs = sorted({s for x, y, s in reg.wide if (x, y) == (a, b)})
        L.append(f"- {a}-{b} ({len(srcs)} files, e.g. `{srcs[0]}`)")
    L += ["", "## Training episodes that were never logged", "",
          "Training workers (dev/async_ppo.py, dev/rl2_env.py / rl3_env.py `GuidanceEnv(seed_lo=100000, "
          "seed_hi=10000000)`; older dev/rl_env.py the same) draw episode seeds uniformly with `np_random.integers(100000, "
          "10000000)`, 10 per prefetch batch (seeds whose reset crashes are skipped). Before 2026-09-29 only episodes "
          "with many pops went to `best_episodes.csv`; the rest were never written anywhere. Fine-tuning runs (tune*) "
          "drew only from their `SEED_LIST` file, which is listed in full (category finetune).", "",
          "| run | host | trained decisions | decisions/episode | est. episodes | scenario | seed source |",
          "|---|---|---:|---:|---:|---|---|"]
    tot = tot_rand = 0
    for run, d in sorted(runs.items()):
        L.append(f"| {run} | {d['origin']} | {d['steps']} | {d['dpe']:.0f} ({d['dpe_src']}) | {d['eps']} | {d['scen']} | "
                 f"{'SEED_LIST (listed)' if d['seedlist'] else 'uniform [100000, 10000000)'} |")
        tot += d["eps"]; tot_rand += 0 if d["seedlist"] else d["eps"]
    logged = len(reg.seeds("training_logged"))
    frac = 1 - (1 - 1 / (TRAIN_HI - TRAIN_LO)) ** tot_rand
    L += ["", f"Estimated total training episodes: ~{tot:,} (of which ~{tot_rand:,} with uniform random seeds; runs "
          f"branched from each other, so steps are counted per run from progress.csv deltas). Logged distinct training "
          f"seeds: {logged}. With ~{tot_rand:,} uniform draws from 9.9 M values, roughly {frac:.1%} of the range "
          f"[{TRAIN_LO}, {TRAIN_HI}) (~{int(frac * (TRAIN_HI - TRAIN_LO)):,} seeds) was met at least once in training and "
          "cannot be listed. The decisions/episode figure comes from the learner's last status.json (default 60 when "
          "missing), so the episode counts are estimates (+-30%).", "",
          "Other seeds that cannot be reconstructed exactly:", "",
          "- search-teacher episodes run on the local host (dev/local_teacher.sh, unseeded `random.Random()`): only "
          "the seeds printed in logs/local_teacher.log or still present as demo files are listed. Cluster teacher "
          "jobs (dev/search_teacher.sbatch) seeded `random.Random(SLURM_JOB_ID)`: their full seed lists were "
          "regenerated from the job ids of logs/teach_<job>.log.",
          "- evaluations that only printed aggregates (dev/sweep_gnc.py, dev/dp_fidelity.py and "
          f"dev/gnc2_eob/sweep_eob.py print 'seeds N' = a count): {len(reg.notes['count_only'])} files had no explicit "
          "range. Their exact command lines are mostly lost; the builder assumes the tool's default first seed "
          "(10000 / 14900 / 30000) and the logged count rounded up to 16, which is how these tools were run "
          "(ranges also recovered from shell history, Claude session transcripts and the DECISION_LOG are included). "
          "Logs deleted outright cannot be recovered at all.",
          "- seeds of balloon-world resets that crashed (negative balloon mass) in training workers.",
          "- official-evaluator runs (BalloonPoppingGymEnv/evaluation/evaluate.py) take their seed from the scenario "
          "config, which this script must not open; their *_trajectory.json files hold no seed. Runs that produced a "
          "submission JSON are covered by its random_seed; the other local trajectory files "
          f"({len(reg.notes['traj_noseed'])}) are listed below and their seeds are only known if they match a "
          "submission or a config seed mentioned elsewhere.", ""]
    if reg.notes["traj_noseed"]:
        L += ["Trajectory files without a seed: " + ", ".join(f"`{x}`" for x in reg.notes["traj_noseed"][:40]), ""]
    if reg.notes["count_only"]:
        L += ["Files with a seed count but no explicit seed list, and the range assumed for them:", ""]
        asm = {s: (a, b) for s, a, b in reg.notes["assumed"]}
        L += [f"- `{s}`: count {', '.join(c)} -> " + (f"{asm[s][0]}-{asm[s][1]}" if s in asm else "not assumed")
              for s, c in reg.notes["count_only"][:80]]
        L.append("")
    lb = sorted(reg.seeds("leaderboard"))
    ours = sorted({k[0] for k, v in reg.rows.items() if k[1] == "leaderboard" and v[1].endswith("_submission.json")})
    L += ["## Leaderboard", "", f"Our submissions' random_seed values: {', '.join(map(str, ours)) or 'none found'}.", "",
          f"{len(lb)} distinct leaderboard-related seeds in total (our submissions, the public replays we re-flew, "
          "the public launch-time table of all teams, leaderboard seed tests): " + ", ".join(map(str, lb[:150]))
          + (" ..." if len(lb) > 150 else ""), ""]
    lowS = sorted(s for s in allS if s < TRAIN_LO)
    highS = sorted(s for s in allS if s >= TRAIN_HI)
    L += ["## Suggested statement for the organisers", "",
          "> Our training episodes drew environment seeds uniformly at random from [100000, 10000000) "
          f"(about {tot_rand / 1e6:.1f} million episodes, so roughly {frac:.0%} of that range was met at least once; "
          "only part of those draws was logged). Fine-tuning runs used only the fixed seeds listed. Every seed we used "
          "outside training (evaluation, seed searches, data collection, teacher search, candidates, videos, "
          f"leaderboard tests) is listed in the attached file ({len(allS):,} distinct seeds). The simplest safe choice "
          "is to draw evaluation seeds >= 10000000 that are not in the list; a seed inside [100000, 10000000) that is "
          f"not in the list still has about a {frac:.0%} chance of having appeared in one of our training episodes.", "",
          f"Seeds below {TRAIN_LO} in the list: {len(lowS)} (covered runs >= 96: "
          f"{fmt_ranges(contiguous(lowS, 96))}). Seeds >= {TRAIN_HI} in the list: {len(highS)} "
          f"({', '.join(map(str, highS[:40]))}; mostly numbers from shell history / other teams' public seeds, kept "
          "because over-reporting is harmless).", ""]
    if reg.notes["skipped_big"]:
        L += ["Skipped (too large to scan as text): " + ", ".join(reg.notes["skipped_big"][:20]), ""]
    (OUT / "history_summary.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    reg = Registry()
    scan_tree(reg, "cluster", SCR)
    scan_tree(reg, "local", REPO)
    for extra in ("data", "analysis", "reports", "auto_submit", "review_2026-09-29", "qualification_source",
                  "lb_replays", "videos", "lb_lastpop_cache.json", "dashboard.log"):
        p = BPC / extra
        if p.exists():
            scan_tree(reg, "local", p, prefix=(extra + "/") if p.is_dir() else "")
    scan_remote_npz(reg)
    teacher_job_seeds(reg)
    local_hunt_seeds(reg)
    scan_histories(reg)
    dl = REPO / "dev/DECISION_LOG.md"
    reg.add(text_seeds(dl.read_text(), "decision_log", reg), "decision_log", "local:dev/DECISION_LOG.md")
    runs = training_estimate()
    write(reg, runs)
    print(f"{len(reg.seeds())} distinct seeds, {len(reg.rows)} rows -> {OUT}/history_seeds.csv, history_summary.md")


if __name__ == "__main__":
    sys.exit(main())
