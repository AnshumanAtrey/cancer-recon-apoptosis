#!/usr/bin/env python3
"""
RUNG 14 — the MECHANISM ARENA: many cancer-killing strategies x many regimes, one run (laptop/Colab, no GPU).

WHY THIS EXISTS
---------------
Up to RUNG-13 we drilled ONE mechanism at a time (recognition axes; the bystander death wave). Anshuman's call:
stop hand-tuning a single model -- instead build ONE broad testbed that runs MANY death/propagation strategies
across MANY regimes (time horizon, dose, recognition leak q_n, priming) in a single shot, so the RUN tells us
which strategy HITS (clears tumour AND spares normal), which is CLOSE, and which is FAR. You run it once; the
leaderboard ranks them.

WHAT IT COMPARES (all share the same validated EARM-burst death effector from RUNG-13)
-------------------------------------------------------------------------------------
RIGOROUS dynamical arms (real mechanisms, modelled honestly):
  per_cell    : each recognised cell triggered independently (NO propagation). Leak ~ q_n LINEAR. The baseline.
  wave        : RUNG-13 bystander death wave -- dead cell emits ~CP, neighbour transduces (gated). Bounded relay.
  quorum      : cell dies only where LOCAL DENSITY of recognised cells exceeds a threshold (cancer is clonal/
                dense; scattered normal false-positives lack quorum). Shriya's "recognise neighbours" literally.
  diffusible  : dead cell releases a DIFFUSING death factor (reaction-diffusion). Models GDEPT (HSV-TK/GCV,
                cytosine-deaminase/5-FC) -- the bystander route that does NOT need gap junctions (RUNG-12P/A
                found tumours barely gap-couple).
  oncolytic   : the death signal SELF-AMPLIFIES (replicates) only in tumour-permeable cells before triggering --
                an oncolytic-virus / self-amplifying-RNA analogue. Super-critical in tumour, dies out in normal.
  alt_death   : a fraction of tumour is APOPTOSIS-RESISTANT (huge brake); a SECOND, brake-independent effector
                (ferroptosis/pyroptosis analogue) reroutes them. Tests Shriya 6.3 resistance head-on.
  combo       : wave + a BH3-mimetic that LOWERS the apoptotic threshold in tumour ("prime then push").

FUTURE / PHYSICS arms (TOY cartoons, clearly flagged -- here so the concept is visible, NOT validated biology):
  oncotripsy  : ultrasound at the cancer cell's resonant frequency (cancer is mechanically softer -> shifted f0).
                Population Lorentzian-resonance model. Real test = mechanics/wet-lab (Heyden-Ortiz 2016).
  ttfields    : alternating EM field disrupts the mitotic spindle (FDA-approved for glioblastoma, Optune).
                Population model on division-rate differential. Real test = physics/clinic.

WHAT THE OUTPUT MEANS (honestly)
--------------------------------
For each (mechanism, regime): tumour_killed, normal_killed, and SAFE&EFFECTIVE = (tumour>=80% AND normal<=1%).
The leaderboard = fraction of regimes each mechanism is safe&effective in. This says which strategy CONCEPT is
robust in silico across conditions -- NOT that it cures cancer. Every arm's recognition->effector coupling is a
PROXY parameter; mapping it to a real molecular efficiency (the agonism/transduction/delivery residual) is the
wet-lab handoff named since RUNG-1. Toy physics arms are cartoons for triage, not models of the real physics.

USAGE
  python scripts/39_mechanism_arena.py            # full arena -> JSON leaderboard + figure
  python scripts/39_mechanism_arena.py quick      # small/fast subset (for a first look)
  python scripts/39_mechanism_arena.py selftest   # logic checks (tiny lattice, numpy only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung14_arena"
RESULT_JSON = OUT_DIR / "rung14_arena.json"
FIGURE_PNG = OUT_DIR / "rung14_arena.png"

# validated EARM-burst death effector (mirrors RUNG-13 scripts/38)
CELL = dict(ai=0.35, kfb=0.5, d8=0.03, kc=25.0, ki=20.0, d3=0.5, kIon=1.0, kseq=60.0, IAP0=1.2, kp=40.0)
COMMIT = 0.5
SAFE_TUMOUR = 0.80     # cleared
SAFE_NORMAL = 0.01     # spared

RIGOROUS = ["per_cell", "wave", "quorum", "diffusible", "oncolytic", "alt_death", "combo"]
TOY = ["oncotripsy", "ttfields"]


# --------------------------------------------------------------------------- helpers
def _neigh_mean(A):
    s = np.zeros_like(A); cnt = np.zeros_like(A)
    s[1:, :] += A[:-1, :]; cnt[1:, :] += 1
    s[:-1, :] += A[1:, :]; cnt[:-1, :] += 1
    s[:, 1:] += A[:, :-1]; cnt[:, 1:] += 1
    s[:, :-1] += A[:, 1:]; cnt[:, :-1] += 1
    return s / cnt


def _neigh_sum(A):
    s = np.zeros_like(A)
    s[1:, :] += A[:-1, :]; s[:-1, :] += A[1:, :]
    s[:, 1:] += A[:, :-1]; s[:, :-1] += A[:, 1:]
    return s


def _deg(L):
    d = np.full((L, L), 4.0)
    d[0, :] -= 1; d[-1, :] -= 1; d[:, 0] -= 1; d[:, -1] -= 1
    return d


def _density(g, iters=3):
    """local recognised-cell density (smooth g over a few neighbourhoods)."""
    rho = g.copy()
    for _ in range(iters):
        rho = _neigh_mean(rho)
    return rho


# --------------------------------------------------------------------------- the unified spatial simulator
def sim_spatial(mech, L=61, r=17, T=80.0, dt=0.02, q_t=0.95, q_n=0.10, strength=1.0,
                seed_frac=0.03, rng=None, p=CELL):
    """One realisation. Returns dict(tumour_killed, normal_killed). Mechanisms differ in how input S is set."""
    if rng is None:
        rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:L, 0:L]
    cx = L // 2
    is_t = ((xx - cx) ** 2 + (yy - cx) ** 2) <= r * r
    g = np.where(rng.random((L, L)) < np.where(is_t, q_t, q_n), 1.0, 0.0)
    deg = _deg(L)

    # priming / threshold modulation per mechanism
    iap_mult = np.ones((L, L))
    if mech == "combo":                       # BH3-mimetic lowers tumour apoptotic brake
        iap_mult = np.where(is_t, 0.4, 1.0)
    IAP0 = p["IAP0"] * iap_mult

    C8a = np.zeros((L, L)); C3a = np.zeros((L, L)); IAP = IAP0.copy(); CP = np.zeros((L, L))
    CPa = np.zeros((L, L))                     # second (brake-independent) effector for alt_death
    D = np.zeros((L, L))                       # diffusible death factor
    V = np.zeros((L, L))                       # self-amplifying agent

    # apoptosis-resistant tumour fraction (for alt_death): apoptosis-INCOMPETENT subclone (caspase/effector
    # blocked -> kc=0), so the apoptotic channel cannot kill them; only the brake-independent alt effector can.
    resistant = np.zeros((L, L), bool)
    if mech == "alt_death":
        resistant = is_t & (rng.random((L, L)) < 0.5)

    # seeds (propagating mechanisms)
    if mech in ("wave", "diffusible", "oncolytic", "alt_death", "combo"):
        seed = is_t & (rng.random((L, L)) < seed_frac) & (g > 0)
        CP[seed] = 0.9
        if mech == "oncolytic":
            V[seed] = 0.5

    ai, kfb, d8, kc, ki, d3, kIon, kseq, kp = (p["ai"], p["kfb"], p["d8"], p["kc"], p["ki"],
                                               p["d3"], p["kIon"], p["kseq"], p["kp"])
    kc_cell = np.where(resistant, 0.0, kc)     # resistant cells can't run apoptosis (only alt effector kills them)
    rho = _density(g) if mech == "quorum" else None
    quorum_theta = 0.6

    def cell_S():
        if mech == "per_cell":
            return strength * g
        if mech in ("wave", "combo"):
            return strength * g * _neigh_mean(CP)
        if mech == "quorum":
            return strength * g * (rho >= quorum_theta)
        if mech == "diffusible":
            return strength * g * D
        if mech == "oncolytic":
            return strength * g * V
        if mech == "alt_death":
            return strength * g * _neigh_mean(np.maximum(CP, CPa))
        return strength * g

    def cell_deriv(C8a, C3a, IAP, CP, S):
        return ((ai * S + kfb * C3a) * (1 - C8a) - d8 * C8a,
                kc_cell * C8a * (1 - C3a) - ki * C3a * IAP - d3 * C3a,
                kIon * (IAP0 - IAP) - kseq * C3a * IAP,
                kp * (C3a * C3a) * (1 - CP))

    n = int(T / dt)
    for _ in range(n):
        S = cell_S()
        st = (C8a, C3a, IAP, CP)
        d = cell_deriv(*st, S)
        s2 = tuple(a + 0.5 * dt * b for a, b in zip(st, d)); d2 = cell_deriv(*s2, S)
        s3 = tuple(a + 0.5 * dt * b for a, b in zip(st, d2)); d3_ = cell_deriv(*s3, S)
        s4 = tuple(a + dt * b for a, b in zip(st, d3_)); d4 = cell_deriv(*s4, S)
        C8a, C3a, IAP, CP = (np.clip(a + (dt / 6) * (b + 2 * x + 2 * z + w), 0.0, None)
                             for a, b, x, z, w in zip(st, d, d2, d3_, d4))
        # aux fields (explicit Euler)
        if mech == "diffusible":
            lap = _neigh_sum(D) - deg * D
            D = np.clip(D + dt * (0.15 * lap - 0.2 * D + 1.0 * CP), 0.0, None)
        elif mech == "oncolytic":
            lap = _neigh_sum(V) - deg * V
            V = np.clip(V + dt * (0.05 * lap + 1.5 * g * V * (1 - V) - 0.1 * V), 0.0, None)
        elif mech == "alt_death":
            # brake-independent effector: reroutes cells reached by the wave, ignores IAP (ferroptosis/pyroptosis)
            S_alt = strength * g * _neigh_mean(np.maximum(CP, CPa))
            CPa = np.clip(CPa + dt * (3.0 * np.minimum(S_alt, 1.0) * (1 - CPa)), 0.0, None)

    dead = CP >= COMMIT
    if mech == "alt_death":
        dead = dead | (CPa >= COMMIT)
    return {"tumour_killed": float((dead & is_t).sum() / is_t.sum()),
            "normal_killed": float((dead & ~is_t).sum() / (~is_t).sum()) if (~is_t).any() else 0.0}


# --------------------------------------------------------------------------- toy physics population models
def sim_oncotripsy(n=20000, q_t=0.95, q_n=0.10, rng=None, Q=12.0, amp=1.6, thresh=1.0):
    """TOY: cancer cells mechanically softer -> lower resonant freq. Sweep drive freq; report best selectivity."""
    if rng is None:
        rng = np.random.default_rng(0)
    nt = n // 2
    f_t = rng.normal(0.7, 0.06, nt)            # cancer resonance (a.u.)
    f_n = rng.normal(1.00, 0.06, n - nt)       # normal resonance
    best = {"tumour_killed": 0.0, "normal_killed": 1.0, "score": -1}
    for f in np.linspace(0.5, 1.2, 36):
        def resp(f0):
            bw = f0 / (2 * Q)
            return amp / (1 + ((f - f0) / bw) ** 2)
        tk = float((resp(f_t) >= thresh).mean()); nk = float((resp(f_n) >= thresh).mean())
        score = tk - nk
        if score > best["score"]:
            best = {"drive_freq": round(float(f), 3), "tumour_killed": round(tk, 3),
                    "normal_killed": round(nk, 3), "score": round(score, 3)}
    return best


def sim_ttfields(q_t=0.95, q_n=0.10, field=1.0, T=80.0, div_t=0.04, div_n=0.004):
    """TOY: alternating EM disrupts mitosis; tumour divides faster -> more affected. Quiescent normal spared."""
    tk = 1 - np.exp(-field * div_t * T)
    nk = 1 - np.exp(-field * div_n * T)
    return {"tumour_killed": round(float(tk), 3), "normal_killed": round(float(nk), 3)}


# --------------------------------------------------------------------------- the arena
def regimes(quick=False):
    Ts = [80.0] if quick else [30.0, 80.0]
    qns = [0.05, 0.30] if quick else [0.05, 0.20, 0.50]
    strengths = [1.0] if quick else [0.5, 1.0]
    out = []
    for T in Ts:
        for qn in qns:
            for s in strengths:
                out.append({"T": T, "q_n": qn, "strength": s})
    return out


def run_arena(quick=False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L = 41 if quick else 61
    r = 12 if quick else 17
    regs = regimes(quick)
    print(f"[rung14] arena: {len(RIGOROUS)} rigorous + {len(TOY)} toy mechanisms x {len(regs)} regimes "
          f"(L={L}, {'quick' if quick else 'full'})")

    results = {}
    for mech in RIGOROUS:
        rows = []
        for reg in regs:
            o = sim_spatial(mech, L=L, r=r, T=reg["T"], q_n=reg["q_n"], strength=reg["strength"],
                            rng=np.random.default_rng(101))
            safe = (o["tumour_killed"] >= SAFE_TUMOUR) and (o["normal_killed"] <= SAFE_NORMAL)
            rows.append({**reg, "tumour_killed": round(o["tumour_killed"], 4),
                         "normal_killed": round(o["normal_killed"], 4), "safe_effective": bool(safe)})
        nsafe = sum(r["safe_effective"] for r in rows)
        results[mech] = {"kind": "rigorous", "regimes": rows, "safe_fraction": round(nsafe / len(rows), 3),
                         "best": max(rows, key=lambda r: (r["safe_effective"], r["tumour_killed"] - 5 * r["normal_killed"]))}
        print(f"  [{mech:11}] safe&effective in {nsafe}/{len(rows)} regimes "
              f"(best: tumour {results[mech]['best']['tumour_killed']:.2f} / normal {results[mech]['best']['normal_killed']:.3f})")

    # toy physics (single representative each)
    onc = sim_oncotripsy(rng=np.random.default_rng(5))
    tt = sim_ttfields()
    results["oncotripsy"] = {"kind": "toy_future", "result": onc,
                             "note": "TOY mechanical-resonance cartoon; real test = mechanics/wet-lab (Heyden-Ortiz 2016)"}
    results["ttfields"] = {"kind": "toy_future", "result": tt,
                           "note": "TOY mitosis-disruption cartoon; real = FDA Optune physics/clinic"}
    print(f"  [oncotripsy ] (TOY) best selectivity: tumour {onc['tumour_killed']:.2f} / normal {onc['normal_killed']:.3f}")
    print(f"  [ttfields   ] (TOY) tumour {tt['tumour_killed']:.2f} / normal {tt['normal_killed']:.3f}")

    # leaderboard (rigorous only)
    board = sorted([(m, results[m]["safe_fraction"]) for m in RIGOROUS], key=lambda x: -x[1])

    out = {
        "tag": "rung14_mechanism_arena",
        "question": "Across many regimes (time, dose, recognition leak q_n), which cancer-killing STRATEGY "
                    "clears tumour while sparing normal -- which hits, which is close, which is far?",
        "death_effector": "validated EARM-burst switch (RUNG-13 / scripts/38), shared by all dynamical arms",
        "safe_effective_def": {"tumour_killed_min": SAFE_TUMOUR, "normal_killed_max": SAFE_NORMAL},
        "regimes_tested": regs,
        "mechanisms": results,
        "leaderboard_rigorous": [{"mechanism": m, "safe_fraction": f} for m, f in board],
        "INTERPRETATION_MAP": {
            "hits": "safe_fraction high across regimes -> robust strategy concept worth wet-lab prioritisation.",
            "close": "safe in some regimes only -> works in a window; needs the right dose/time/leak (tune).",
            "far": "safe_fraction ~0 -> concept does not contain itself in silico; deprioritise or redesign."},
        "DECISIVE": "",
        "CEILING": "Every arm's recognition->effector coupling is a PROXY parameter; mapping it to a real "
                   "molecular/delivery efficiency is the wet-lab residual (agonism, since RUNG-1). Dynamical arms "
                   "share one reduced death model (RUNG-13), 2D, no immune/real-diffusion-geometry. TOY physics "
                   "arms (oncotripsy/ttfields) are cartoons for triage, NOT validated physics. Leaderboard ranks "
                   "in-silico containment robustness, NOT clinical efficacy.",
    }
    winners = [m for m, f in board if f >= 0.5]
    out["DECISIVE"] = (f"In-silico containment leaderboard (safe&effective fraction): "
                       + ", ".join(f"{m}={f:.2f}" for m, f in board)
                       + f". Strategies that HIT across >=50% of regimes: {winners or 'none -- all need tuning'}. "
                       f"per_cell (no propagation) is the baseline whose normal-leak scales LINEARLY with q_n; "
                       f"propagation/quorum/gated arms that beat it convert that into bounded leak. Toy physics "
                       f"arms reported separately (future). Coupling/delivery efficiency is the wet-lab residual.")

    def _jd(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        return str(o)

    RESULT_JSON.write_text(json.dumps(out, indent=2, default=_jd))
    print(f"\n[rung14] wrote {RESULT_JSON}")
    print(f"  LEADERBOARD: " + " > ".join(f"{m}({f:.2f})" for m, f in board))
    print(f"\n  DECISIVE: {out['DECISIVE']}")
    _make_figure(out)
    return 0


def _make_figure(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung14] matplotlib unavailable ({e})")
        return
    regs = out["regimes_tested"]
    reg_labels = [f"T{int(r['T'])}|qn{r['q_n']}|s{r['strength']}" for r in regs]
    fig, ax = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.3, 1]})

    # heatmap: mechanism x regime, value = tumour_killed - 5*normal_killed (safe&effective outlined)
    M = np.zeros((len(RIGOROUS), len(regs)))
    safe = np.zeros_like(M, bool)
    for i, m in enumerate(RIGOROUS):
        for j, row in enumerate(out["mechanisms"][m]["regimes"]):
            M[i, j] = row["tumour_killed"] - 5 * row["normal_killed"]
            safe[i, j] = row["safe_effective"]
    im = ax[0].imshow(M, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(regs))); ax[0].set_xticklabels(reg_labels, rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(RIGOROUS))); ax[0].set_yticklabels(RIGOROUS, fontsize=9)
    for i in range(len(RIGOROUS)):
        for j in range(len(regs)):
            if safe[i, j]:
                ax[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
    ax[0].set_title("score = tumour_killed - 5*normal_killed  (black box = SAFE&EFFECTIVE)\n"
                    "rows=strategy, cols=regime (time|q_n leak|dose)")
    fig.colorbar(im, ax=ax[0], label="score")

    board = out["leaderboard_rigorous"]
    names = [b["mechanism"] for b in board]; fracs = [b["safe_fraction"] for b in board]
    ax[1].barh(range(len(names))[::-1], fracs, color="#2E7D32")
    ax[1].set_yticks(range(len(names))[::-1]); ax[1].set_yticklabels(names, fontsize=9)
    ax[1].set_xlabel("fraction of regimes safe & effective"); ax[1].set_xlim(0, 1)
    ax[1].set_title("LEADERBOARD\n(which strategy hits)")
    for i, f in enumerate(fracs):
        ax[1].text(f + 0.02, len(names) - 1 - i, f"{f:.2f}", va="center", fontsize=8)

    fig.suptitle("RUNG-14 mechanism arena — which cancer-killing strategy contains itself in silico across regimes "
                 "(coupling/delivery = wet-lab residual; toy physics arms separate)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PNG, dpi=120)
    print(f"[rung14] wrote {FIGURE_PNG}")


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    kw = dict(L=31, r=9, T=50.0, dt=0.03, rng=np.random.default_rng(0))
    # every rigorous mechanism runs and returns sane numbers
    for m in RIGOROUS:
        o = sim_spatial(m, q_t=0.95, q_n=0.10, strength=1.0, **kw)
        check(f"{m}: returns valid kill fractions in [0,1]",
              0 <= o["tumour_killed"] <= 1 and 0 <= o["normal_killed"] <= 1)

    # per_cell leak scales with q_n (linear baseline)
    lo = sim_spatial("per_cell", q_t=0.95, q_n=0.05, strength=1.0, **kw)["normal_killed"]
    hi = sim_spatial("per_cell", q_t=0.95, q_n=0.50, strength=1.0, **kw)["normal_killed"]
    check("per_cell normal-leak rises with q_n (linear baseline)", hi > lo)

    # quorum spares scattered normal false-positives even at high q_n (the advantage)
    qn_hi = sim_spatial("quorum", q_t=0.95, q_n=0.50, strength=1.0, **kw)["normal_killed"]
    check("quorum spares normal at high q_n better than per_cell", qn_hi < hi)

    # wave clears tumour at low q_n
    w = sim_spatial("wave", q_t=0.95, q_n=0.05, strength=1.0, **kw)
    check("wave clears tumour at low q_n (>0.5)", w["tumour_killed"] > 0.5)

    # alt_death rescues clearance vs apoptosis-only when tumour is half-resistant
    a = sim_spatial("alt_death", q_t=0.95, q_n=0.05, strength=1.0, **kw)["tumour_killed"]
    check("alt_death achieves tumour clearance with resistant fraction (>0.4)", a > 0.4)

    # toy arms run
    onc = sim_oncotripsy(n=4000, rng=np.random.default_rng(1)); tt = sim_ttfields()
    check("oncotripsy toy returns selectivity", onc["tumour_killed"] >= onc["normal_killed"])
    check("ttfields toy: tumour killed > normal killed", tt["tumour_killed"] > tt["normal_killed"])

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUNG-14 mechanism arena")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "quick", "selftest"])
    args = ap.parse_args()
    if args.mode == "selftest":
        sys.exit(selftest())
    sys.exit(run_arena(quick=(args.mode == "quick")))
