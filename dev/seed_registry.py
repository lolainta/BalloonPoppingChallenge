"""Record every env seed our agent meets (organisers' request 2026-09-29: seeds we used in training, fine-tuning
or testing are not drawn for our qualification / final evaluations; over-reporting is harmless).

    from dev.seed_registry import record
    record(seeds, "run_seeds", scenario=4, note="policy_path=...")

Appends one row per seed (seed, scenario, tool, note, host, utc) to data/seed_registry/used_seeds.csv: one os.write
under an exclusive flock, three attempts (DECISION_LOG §390: two sweep jobs on one node registered in the same second
and one whole append vanished silently). Never raises: logging must not break experiments; a failure goes to stderr.
Training episodes are logged per run by dev/async_ppo.py in runs/<run>/episode_seeds.csv instead.
"""
import csv, fcntl, io, os, socket, sys, time
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(os.environ.get("SEED_REGISTRY") or Path(__file__).resolve().parents[1] / "data/seed_registry/used_seeds.csv")
HEADER = ["seed", "scenario", "tool", "note", "host", "utc"]


def _rows(rows):
    buf = io.StringIO(); csv.writer(buf).writerows(rows); return buf.getvalue()


def record(seeds, tool, note="", scenario=""):
    try:
        host, now = socket.gethostname().split(".")[0], datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = _rows([int(s), scenario, tool, note, host, now] for s in seeds)
        PATH.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                fd = os.open(PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    data = (("" if os.fstat(fd).st_size else _rows([HEADER])) + body).encode()
                    n = os.write(fd, data)
                    if n != len(data):
                        raise OSError(f"short write {n}/{len(data)}")
                finally:
                    os.close(fd)  # also releases the lock
                return
            except OSError as e:
                err = e; time.sleep(1 + attempt)
        print(f"seed_registry: giving up after 3 attempts: {err!r}", file=sys.stderr)
    except Exception as e:
        print(f"seed_registry: {e!r}", file=sys.stderr)


if __name__ == "__main__":  # self-check: 8 processes appending 96 rows each at once must all land
    import multiprocessing as mp, subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, SEED_REGISTRY=f"{d}/r.csv")
        code = "import sys; sys.path.insert(0, '.'); from dev.seed_registry import record; record(range(int(sys.argv[1]), int(sys.argv[1]) + 96), 'test', 'x,y', 40)"
        ps = [subprocess.Popen([sys.executable, "-c", code, str(i * 100)], env=env, cwd=Path(__file__).resolve().parents[1]) for i in range(8)]
        assert all(p.wait() == 0 for p in ps)
        rows = list(csv.reader(open(f"{d}/r.csv")))
        assert rows[0] == HEADER and len(rows) == 1 + 8 * 96 and all(len(r) == 6 for r in rows), len(rows)
        assert sorted(int(r[0]) for r in rows[1:]) == sorted(i * 100 + k for i in range(8) for k in range(96))
    print("seed_registry self-check ok")
