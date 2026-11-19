"""Was this seed ever used by our agent? (qualification 2026-10-10: a drawn seed we used in training, tuning or testing is redrawn)
    .venv/bin/python dev/seed_check.py 123456 [789 ...]
Looks in data/seed_registry/history_seeds.csv (dev/seed_registry_build.py: everything reconstructed from disk, local + cluster)
and the live registries written since (data/seed_registry/used_seeds.csv here and the cluster copy in the build's mirror).
Prints USED with where it was found, or NOT USED. Rebuild the history first if training ran since (seed_registry_build.py)."""
import csv
import sys
from pathlib import Path

REG = Path(__file__).resolve().parents[1] / "data/seed_registry"
SOURCES = [REG / "history_seeds.csv", REG / "used_seeds.csv", *sorted(REG.glob("_cluster_mirror/**/used_seeds.csv"))]


def main(seeds):
    want, hits = {int(s) for s in seeds}, {}
    for f in SOURCES:
        if not f.is_file():
            continue
        with open(f, newline="") as fh:
            for r in csv.reader(fh):
                if r and r[0].isdigit() and int(r[0]) in want:
                    hits.setdefault(int(r[0]), []).append(f"{f.relative_to(REG)}: {','.join(r[1:4])}")
    print("checked: " + ", ".join(str(f.relative_to(REG)) for f in SOURCES if f.is_file()))
    for s in sorted(want):
        print(f"{s}: USED ({len(hits[s])} rows) e.g. {hits[s][0][:150]}" if s in hits else f"{s}: NOT USED")


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
