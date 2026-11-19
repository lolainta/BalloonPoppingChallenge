"""V3: per-hop pointing vs hindsight constant-force direction; pop-state metrics; first hop; throttle proxy."""
import numpy as np, json
from base import *

def hoplist(d):
    """(ts, te, b, is_first) for pops within [L, L+BURN]."""
    out, prev = [], d["L"]
    for tp, b in d["pops"]:
        if tp > d["L"] + BURN: break
        out.append((prev, tp, int(b), prev == d["L"])); prev = tp
    return out

def a_full_tab():
    xs, ys = [], []
    for n in NAMES:
        d = get("ours_strict", n)
        az = (d["f"] * d["zb"]).sum(1)
        m = d["burnmask"] & (d["A"][:, 3] >= 0.99) & (np.hypot(d["A"][:, 0], d["A"][:, 1]) < 3)
        m[:d["i0"] + 100] = False; m[-3:] = False
        # throttle rate-limited: require throttle cmd >=0.99 for the previous 0.5 s
        thr = d["A"][:, 3]; ok = np.array([thr[max(0, k - 50):k + 1].min() >= 0.99 for k in range(len(thr))])
        m &= ok
        xs.append(d["t"][m] - d["L"]); ys.append(az[m])
    x, y = np.concatenate(xs), np.concatenate(ys)
    grid = np.arange(0, 31, 1.0)
    med = np.array([np.median(y[np.abs(x - g) < 0.5]) if (np.abs(x - g) < 0.5).sum() > 20 else np.nan for g in grid])
    ok = np.isfinite(med)
    return grid[ok], med[ok]

if __name__ == "__main__":
    gx, gy = a_full_tab()
    print("a_full(t since launch):", " ".join(f"{a:.0f}:{b:.2f}" for a, b in zip(gx[::3], gy[::3])))
    afull = lambda s: np.interp(s, gx, gy)
    out = {}
    for kind in ("their", "ours_strict", "ours_free"):
        H, P, FIRST, THR = [], [], [], []
        for n in NAMES:
            d = get(kind, n); t = d["t"]
            az = (d["f"] * d["zb"]).sum(1)
            b = d["burnmask"].copy(); b[:d["i0"] + 50] = False; b[-3:] = False
            THR.append(az[b] / afull(t[b] - d["L"]))
            hl = hoplist(d)
            for j, (ts, te, bb, first) in enumerate(hl):
                i_s, i_e = ti(t, ts), ti(t, te)
                pb = d["B"][i_e - 1, bb, :3].astype(float)  # balloon pos at the last step before popped
                k = np.arange(i_s + 1, i_e)
                tau = te - t[k]
                k = k[tau >= 0.3]; tau = te - t[k]
                if len(k) < 5: continue
                fh = 2 * (pb - d["p"][k] - d["v"][k] * tau[:, None]) / tau[:, None] ** 2 - G
                th = angd(d["zb"][k], fh); tt = t[k] - ts
                fb = lambda x: float(tt[np.argmax(th < x)]) if (th < x).any() else np.nan
                rec = dict(name=n, hop=te - ts, first=first, th0=float(th[0]), t20=fb(20), t10=fb(10))
                (FIRST if first else H).append(rec)
            # pop state for each pop that has a next pop within the burn
            for (ts, te, bb, first), (ts2, te2, b2, _) in zip(hl[:-1], hl[1:]):
                i = ti(t, te)
                tau = te2 - te
                pb2 = d["B"][ti(t, te2) - 1, b2, :3].astype(float)
                fn = 2 * (pb2 - d["p"][i] - d["v"][i] * tau) / tau ** 2 - G
                r = d["B"][i, b2, :3].astype(float) - d["p"][i]
                vrel = d["v"][i] - d["B"][i, b2, 3:].astype(float)
                P.append(dict(name=n, first=first, w=float(d["wxy"][i]), nose_next=float(angd(d["zb"][i], fn)),
                              vel_next=float(angd(d["v"][i], pb2 - d["p"][i])), close=float(vrel @ r / np.linalg.norm(r)),
                              speed=float(np.linalg.norm(d["v"][i])), tau=tau))
        big = [h for h in H if h["th0"] > 30]
        med = lambda L, k: float(np.nanmedian([h[k] for h in L]))
        print(f"\n[{kind}] non-first hops n={len(H)} hop med {med(H,'hop'):.2f} mean {np.mean([h['hop'] for h in H]):.2f} | th0 med {med(H,'th0'):.1f} | t20 med {med(H,'t20'):.2f} | t10 med {med(H,'t10'):.2f}")
        print(f"   big (th0>30) {len(big)}/{len(H)} = {len(big)/len(H):.2f}: t10 med {med(big,'t10'):.2f} (IQR {np.nanpercentile([h['t10'] for h in big],25):.2f}-{np.nanpercentile([h['t10'] for h in big],75):.2f}), th0 med {med(big,'th0'):.1f}, rate {np.nanmedian([(h['th0']-10)/h['t10'] for h in big]):.1f} deg/s")
        print(f"   first hop n={len(FIRST)} dur med {med(FIRST,'hop'):.2f} th0 med {med(FIRST,'th0'):.1f} t10 med {med(FIRST,'t10'):.2f}")
        Pn = [p for p in P if True]
        print(f"   pop state (n={len(Pn)}): |w| med {med(Pn,'w'):.2f} | nose vs next dir {med(Pn,'nose_next'):.1f} | vel vs next {med(Pn,'vel_next'):.1f} | closing {med(Pn,'close'):.1f} m/s | speed {med(Pn,'speed'):.1f}")
        Pnf = [p for p in P if not p["first"]]
        print(f"   pop state excl pop#1 (n={len(Pnf)}): |w| med {med(Pnf,'w'):.2f} | nose vs next {med(Pnf,'nose_next'):.1f} | vel vs next {med(Pnf,'vel_next'):.1f} | closing {med(Pnf,'close'):.1f}")
        THR = np.concatenate(THR)
        print(f"   throttle proxy mean {THR.mean():.3f} frac>=0.97 {np.mean(THR>=0.97):.3f}")
        out[kind] = dict(H=H, P=P, FIRST=FIRST)
    json.dump(out, open("v3_out.json", "w"), default=float)
