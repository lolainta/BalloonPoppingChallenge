#!/usr/bin/env python3
"""DECISION_LOG §390: after a sweep, make sure every planned seed is in the registry (one probe's record() vanished when
two jobs on one node registered in the same second). Usage: seed_backfill.py SCENARIO NOTE LO-HI [LO-HI ...]"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev.seed_registry import PATH, record  # noqa: E402

sc, note, ranges = sys.argv[1], sys.argv[2], sys.argv[3:]
planned = {s for r in ranges for s in range(int(r.split("-")[0]), int(r.split("-")[1]) + 1)}
have = {int(r[0]) for r in csv.reader(open(PATH)) if r and r[0].isdigit()} if PATH.exists() else set()
missing = sorted(planned - have)
if missing:
    record(missing, "sweep_gnc", f"backfill ({note})", sc)
    print(f"seed registry: {len(missing)} of {len(planned)} planned seeds were missing -> backfilled ({missing[0]}..{missing[-1]})")
else:
    print(f"seed registry: all {len(planned)} planned seeds registered")
