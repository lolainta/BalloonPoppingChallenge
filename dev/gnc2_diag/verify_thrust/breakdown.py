"""Verifier: where our throttle loss goes (speed level, time-to-pop, rate limit, gimbal cos, clamp), and end-of-hop
utilization per POP (not per run) for both teams."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from common import *
from util import run, a_max

def pops_from_S(d):
    S = d["S"]; t = d["t"]
    out = []
    for b in range(S.shape[1]):
        k = np.flatnonzero(S[:, b] == 2)
        if k.size: out.append((t[k[0]], b, k[0]))
    return sorted(out)

def main():
  lines = []
  for kind in ("ours_strict", "ours_free"):
      acc = dict(gim=0, thr=0, rate=0, rate_steps=0, n=0, clamp_steps=0, clamp_loss=0)
      lvl = {}
      for n in NAMES:
          r = run(kind, n); d, x, m = r["d"], r["x"], r["m"]
          A = d["A"]; thr = A[:, 3]; g = np.radians(A[:, :2]); cg = np.sqrt(np.clip(1 - np.sin(g[:, 0])**2 - np.sin(g[:, 1])**2, 0, 1))
          am = r["am_all"]
          F = d["F"]; Fm = np.linalg.norm(F, axis=1); want = np.clip(Fm / d["a_thrust"], 0, 1)
          acc["gim"] += (thr * (1 - cg) * am)[m].sum() * DT
          acc["thr"] += ((1 - thr) * am)[m].sum() * DT
          rl = np.isfinite(want) & (want - thr > 0.005)
          acc["rate"] += (np.where(np.isfinite(want), np.maximum(want - thr, 0), 0) * am)[m].sum() * DT
          acc["rate_steps"] += (np.isfinite(want) & (np.abs(want - thr) > 0.005))[m].sum(); acc["n"] += m.sum()
          cl = np.isfinite(Fm) & (F[:, 2] < 1.0 + 1e-6 + 0.05) & (Fm < 0.98 * d["a_thrust"])
          acc["clamp_steps"] += cl[m].sum(); acc["clamp_loss"] += ((1 - thr) * am)[m & cl].sum() * DT
          st = d["STRETCH"]
          for L in np.unique(st[m]):
              k = m & (st == L)
              e = lvl.setdefault(float(L), [0, 0.0, 0.0]); e[0] += k.sum(); e[1] += thr[k].sum(); e[2] += ((1 - thr) * am)[k].sum() * DT
      N = acc["n"]
      lines.append(f"{kind}: per run lost: throttle {acc['thr']/11:.1f}, gimbal-cos {acc['gim']/11:.2f}, rate-limit {acc['rate']/11:.2f} m/s "
                   f"(steps |want-thr|>0.005: {acc['rate_steps']/N:.3f}); clamp-like steps (F_z<1.05 & |F|<0.98a) {acc['clamp_steps']/N:.3f}, loss {acc['clamp_loss']/11:.2f}")
      for L, (k, s, lo) in sorted(lvl.items()):
          lines.append(f"    stretch {L}: share {k/N:.3f} mean thr {s/max(k,1):.3f} lost {lo/11:.1f} m/s/run")

  # end-of-hop utilization, per pop, both teams (same estimator)
  lines.append("\nend-of-hop: U_ax by time-to-next-pop (pops in burn); lost impulse per POP and per run")
  bins = [(0, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5)]
  for kind in ("their", "ours_strict", "ours_free"):
      tot = {b: [0.0, 0, 0.0] for b in bins}; npop = 0; after_last = 0.0
      for n in NAMES:
          r = run(kind, n); d, x, m = r["d"], r["x"], r["m"]
          P = pops_from_S(d)
          tp = np.array([p[0] for p in P])
          t = x["t"] - 0.5 * DT
          idx = np.searchsorted(tp, t)  # next pop after t
          ttp = np.where(idx < len(tp), tp[np.minimum(idx, len(tp) - 1)] - t, np.inf)
          U, am = r["U_all"], r["am_all"]
          lost = np.maximum(am * (1 - U), 0) * DT
          npop += int(((tp - x["L"]) < BURN).sum())
          after_last += lost[m & ~np.isfinite(ttp)].sum()
          for b in bins:
              k = m & (ttp >= b[0]) & (ttp < b[1])
              tot[b][0] += U[k].sum(); tot[b][1] += k.sum(); tot[b][2] += lost[k].sum()
      s = f"  {kind:12s} pops in burn {npop}: "
      s += " ".join(f"[{b[0]},{b[1]}) U={tot[b][0]/max(tot[b][1],1):.3f}" for b in bins)
      l15 = sum(tot[b][2] for b in bins[:4])
      s += f"\n  {'':12s} lost in last 1.5 s: {l15/11:.1f} m/s/run = {l15/npop:.2f} m/s per pop; time share of last-1.5s window {sum(tot[b][1] for b in bins[:4])*DT/npop:.2f} s/pop; lost after last in-burn pop {after_last/11:.1f} m/s/run"
      lines.append(s)
  txt = "\n".join(lines); print(txt)
  (ROOT / "dev/gnc2_diag/verify_thrust/breakdown.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
