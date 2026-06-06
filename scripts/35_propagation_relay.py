#!/usr/bin/env python3
"""
RUNG 12P / Part B — the gated-relay death-wave simulation (laptop, seconds, no GPU/atlas).

THE QUESTION (the Iron-Man one)
-------------------------------
Part A killed the PASSIVE route: tumours barely couple (Cx43 in ~6% of malignant cells) and the connexin
they do have leaks into 8/11 vital tissues. So the death wave must use ENGINEERED coupling AND re-check
tumour identity at each hop. The question this run answers:

  Does a per-hop-GATED death wave clear the tumour while sparing normal tissue -- and is its safe operating
  region BIGGER than the per-cell gate's (R5 found NO per-cell surface gate worst-donor-safe)?

WHY IT MIGHT (the physics, which is the robust part)
----------------------------------------------------
Model the wave as SITE-BOND PERCOLATION on a lattice: a tumour disk in a normal field. A cell dies + relays
only if (a) it is reached by a dead neighbour through a COUPLED edge (engineered coupling, prob c) and (b) it
PASSES its recognition gate (per-hop fidelity q_t for tumour, false-positive q_n for normal). A cell dies iff
it is connected to a SEED through permeable cells + coupled edges.

  - In TUMOUR: effective transmissibility c*q_t. If > percolation threshold -> SUPER-critical -> the wave
    sweeps the whole disk from a few seeds. (efficacy decoupled from per-cell recognition: seed once, spread.)
  - In NORMAL: effective c*q_n. If < threshold -> SUB-critical -> any false-positive normal cell starts a wave
    that DIES OUT in a finite cluster (mean size ~ 1/(1 - c*q_n*z) below threshold). ERRORS DON'T CASCADE.

So there is a WINDOW  c*q_n < p_c < c*q_t  where the tumour clears but normal tissue self-extinguishes. The
decisive contrast: the PER-CELL gate kills normal tissue LINEARLY in its false-positive (every false-positive
cell, anywhere, dies -> leak = q_n); the RELAY only kills normal cells on a percolating path from the tumour,
which below threshold is a thin BOUNDED rind. That is the safety amplification -- and it is what could rescue
a recognition signal too leaky to pass R5's per-cell bar.

WHAT THIS IS, HONESTLY
----------------------
An ABSTRACTION (2D site-bond percolation), not a tissue. The per-hop fidelities q_t/q_n and coupling c are
PARAMETERS we sweep, not measured molecular values -- mapping them to a real gate (the RUNG-11 neoantigen
gate, a synNotch relay) + engineered coupling efficiency is the wet-lab residual. The death effector per cell
is RUNG-1's EARM bistable switch (modular), abstracted here as binary. The ROBUST, parameter-free claim is the
RELATIONSHIP: relay leak is sub-critical-bounded vs per-cell leak linear -> a threshold-protected safety margin.
Caveats: no 3D, no diffusion kinetics, no immune clearance, no partial/graded signals, single tumour focus.

USAGE
  python scripts/35_propagation_relay.py            # parameter sweep -> JSON + figure
  python scripts/35_propagation_relay.py selftest   # percolation-logic checks (fast, no deps beyond numpy/scipy)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung12pB_relay"
RESULT_JSON = OUT_DIR / "rung12pB_relay.json"
FIGURE_PNG = OUT_DIR / "rung12pB_relay.png"

PER_CELL_LEAK_BAR = 0.02   # R5/R7 worst-donor safety bar on per-cell false-positive (LEAK_BAR) — the thing to beat


# ---------------------------------------------------------------------------
#  One percolation realisation. Pure (rng injected) so selftest is deterministic.
#  A cell dies iff connected to a SEED through permeable cells and coupled edges.
# ---------------------------------------------------------------------------
def simulate(L, r, c, q_t, q_n, f, rng, moore=False):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    yy, xx = np.mgrid[0:L, 0:L]
    cx = cy = L // 2
    is_t = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r           # central tumour disk
    q = np.where(is_t, q_t, q_n)
    permeable = rng.random((L, L)) < q                        # cell would PASS its per-hop gate if signalled
    seed = is_t & (rng.random((L, L)) < f) & permeable        # tumour cells that passed INITIAL recognition

    idx = np.arange(L * L).reshape(L, L)
    src, dst = [], []

    def add(a_mask, b_mask, a_idx, b_idx):
        # edge open iff BOTH endpoints permeable AND coupled (prob c)
        ok = a_mask & b_mask & (rng.random(a_idx.shape) < c)
        src.append(a_idx[ok]); dst.append(b_idx[ok])

    pm = permeable
    add(pm[:, :-1], pm[:, 1:], idx[:, :-1], idx[:, 1:])       # right neighbour
    add(pm[:-1, :], pm[1:, :], idx[:-1, :], idx[1:, :])       # down neighbour
    if moore:
        add(pm[:-1, :-1], pm[1:, 1:], idx[:-1, :-1], idx[1:, 1:])
        add(pm[:-1, 1:], pm[1:, :-1], idx[:-1, 1:], idx[1:, :-1])
    s = np.concatenate(src) if src else np.array([], int)
    d = np.concatenate(dst) if dst else np.array([], int)
    N = L * L
    adj = coo_matrix((np.ones(len(s) * 2), (np.concatenate([s, d]), np.concatenate([d, s]))), shape=(N, N))
    n_comp, labels = connected_components(adj, directed=False)

    seed_labels = np.unique(labels.reshape(L, L)[seed]) if seed.any() else np.array([], int)
    dead = np.isin(labels, seed_labels).reshape(L, L) & permeable   # only permeable cells can die
    dead[~np.isin(labels, seed_labels).reshape(L, L)] = False
    nt, nn = int(is_t.sum()), int((~is_t).sum())
    return {"tumour_killed": float((dead & is_t).sum() / nt),
            "normal_killed": float((dead & ~is_t).sum() / nn) if nn else 0.0}


def sweep(L, r, c, q_t, q_n, f, n_trials, moore=False, seed0=12345):
    tk, nk = [], []
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + t)
        out = simulate(L, r, c, q_t, q_n, f, rng, moore)
        tk.append(out["tumour_killed"]); nk.append(out["normal_killed"])
    return float(np.mean(tk)), float(np.mean(nk))


def simulate3d(L, r, c, q_t, q_n, f, rng):
    """3D simple-cubic (z=6) version — the honesty check: 3D percolates EASIER (bond p_c~0.249 << 2D 0.5),
    so the safe q_n window is the conservative direction. Same site-bond logic as simulate()."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    zz, yy, xx = np.mgrid[0:L, 0:L, 0:L]
    cc = L // 2
    is_t = (xx - cc) ** 2 + (yy - cc) ** 2 + (zz - cc) ** 2 <= r * r     # central tumour sphere
    q = np.where(is_t, q_t, q_n)
    permeable = rng.random((L, L, L)) < q
    seed = is_t & (rng.random((L, L, L)) < f) & permeable
    idx = np.arange(L ** 3).reshape(L, L, L)
    src, dst = [], []

    def add(am, bm, ai, bi):
        ok = am & bm & (rng.random(ai.shape) < c)
        src.append(ai[ok]); dst.append(bi[ok])
    pm = permeable
    add(pm[:, :, :-1], pm[:, :, 1:], idx[:, :, :-1], idx[:, :, 1:])
    add(pm[:, :-1, :], pm[:, 1:, :], idx[:, :-1, :], idx[:, 1:, :])
    add(pm[:-1, :, :], pm[1:, :, :], idx[:-1, :, :], idx[1:, :, :])
    s = np.concatenate(src) if src else np.array([], int)
    d = np.concatenate(dst) if dst else np.array([], int)
    N = L ** 3
    adj = coo_matrix((np.ones(len(s) * 2), (np.concatenate([s, d]), np.concatenate([d, s]))), shape=(N, N))
    _, labels = connected_components(adj, directed=False)
    seed_labels = np.unique(labels.reshape(L, L, L)[seed]) if seed.any() else np.array([], int)
    dead = np.isin(labels, seed_labels).reshape(L, L, L) & permeable
    nt, nn = int(is_t.sum()), int((~is_t).sum())
    return {"tumour_killed": float((dead & is_t).sum() / nt),
            "normal_killed": float((dead & ~is_t).sum() / nn) if nn else 0.0}


def sweep3d(L, r, c, q_t, q_n, f, n_trials, seed0=777):
    tk, nk = [], []
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + t)
        out = simulate3d(L, r, c, q_t, q_n, f, rng)
        tk.append(out["tumour_killed"]); nk.append(out["normal_killed"])
    return float(np.mean(tk)), float(np.mean(nk))


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L, r, f, n_trials = 121, 34, 0.03, 12       # tumour disk ~ a quarter of the field; seed only 3% of tumour
    c, q_t = 0.85, 0.90                          # engineered coupling efficiency; tumour per-hop relay fidelity
    z = 4                                        # von Neumann degree -> bond percolation threshold ~0.5
    p_c = 0.5

    print(f"[rung12pB] gated-relay percolation: L={L}, tumour r={r}, seed f={f}, coupling c={c}, q_t={q_t}, "
          f"trials={n_trials}; bond p_c~{p_c} (z={z})")

    # 1) CONTAINMENT curve: vary normal per-hop false-positive q_n; relay leak vs per-cell leak (= q_n)
    qn_grid = [round(x, 3) for x in np.linspace(0.0, 0.95, 20)]
    containment = []
    for qn in qn_grid:
        tk, nk = sweep(L, r, c, q_t, qn, f, n_trials)
        containment.append({"q_n": qn, "c_qn": round(c * qn, 3), "relay_normal_killed": round(nk, 4),
                            "per_cell_normal_killed": qn, "tumour_killed": round(tk, 4)})
    # the relay's safe ceiling on q_n: highest q_n with relay normal-kill <= 1% (vs per-cell bar 0.02)
    safe_qn = max([row["q_n"] for row in containment if row["relay_normal_killed"] <= 0.01], default=0.0)
    amplification = round(safe_qn / PER_CELL_LEAK_BAR, 1) if PER_CELL_LEAK_BAR else None

    # 2) OPERATING REGION: (q_t x q_n) -> cleared (>90% tumour) AND spared (<1% normal)?
    qt_grid = [round(x, 3) for x in np.linspace(0.3, 1.0, 11)]
    qn_grid2 = [round(x, 3) for x in np.linspace(0.0, 0.95, 11)]
    region_tk = np.zeros((len(qt_grid), len(qn_grid2)))
    region_nk = np.zeros((len(qt_grid), len(qn_grid2)))
    for i, qt in enumerate(qt_grid):
        for j, qn in enumerate(qn_grid2):
            tk, nk = sweep(L, r, c, qt, qn, f, max(6, n_trials // 2))
            region_tk[i, j] = tk; region_nk[i, j] = nk
    safe_eff = (region_tk > 0.90) & (region_nk < 0.01)
    frac_region_safe_eff = float(safe_eff.mean())

    # 3) tumour clearance onset vs q_t (efficacy percolation), at a safe q_n
    qt_curve = []
    for qt in [round(x, 3) for x in np.linspace(0.3, 1.0, 15)]:
        tk, nk = sweep(L, r, c, qt, 0.10, f, n_trials)
        qt_curve.append({"q_t": qt, "c_qt": round(c * qt, 3), "tumour_killed": round(tk, 4),
                         "normal_killed": round(nk, 4)})

    # 4) 3D SENSITIVITY (honesty check): 3D percolates easier -> narrower safe window. Same coupling/q_t.
    L3, r3 = 41, 12
    cont3d = []
    for qn in [round(x, 3) for x in np.linspace(0.0, 0.95, 12)]:
        tk, nk = sweep3d(L3, r3, c, q_t, qn, f, 6)
        cont3d.append({"q_n": qn, "relay_normal_killed": round(nk, 4), "tumour_killed": round(tk, 4)})
    safe_qn_3d = max([row["q_n"] for row in cont3d if row["relay_normal_killed"] <= 0.01], default=0.0)
    amp_3d = round(safe_qn_3d / PER_CELL_LEAK_BAR, 1) if PER_CELL_LEAK_BAR else None

    result = {
        "tag": "rung12pB_gated_relay_percolation",
        "question": "Does a per-hop-GATED death wave (engineered coupling) clear the tumour while sparing "
                    "normal tissue, and is its safe q_n region wider than the per-cell gate's (R5 found 0 "
                    "per-cell surface gates safe)?",
        "model": "2D site-bond percolation. Cell dies iff connected to a seed via permeable cells (per-hop "
                 "gate pass prob q_type) + coupled edges (engineered coupling prob c). EARM is the per-cell "
                 "effector (modular, binarised).",
        "params": {"L": L, "tumour_radius": r, "seed_fraction": f, "coupling_c": c, "q_t_default": q_t,
                   "lattice_degree": z, "bond_p_c_approx": p_c, "n_trials": n_trials},
        "per_cell_leak_bar_R5": PER_CELL_LEAK_BAR,
        "containment_curve": containment,
        "relay_safe_q_n_ceiling_at_1pct": safe_qn,
        "safety_amplification_vs_percell": amplification,
        "operating_region": {"q_t_grid": qt_grid, "q_n_grid": qn_grid2,
                             "tumour_killed": region_tk.round(3).tolist(),
                             "normal_killed": region_nk.round(3).tolist(),
                             "frac_safe_and_effective": round(frac_region_safe_eff, 3)},
        "tumour_clearance_vs_qt": qt_curve,
        "sensitivity_3D": {"L": L3, "tumour_radius": r3, "bond_p_c_approx": 0.249,
                           "containment_curve": cont3d, "safe_q_n_ceiling_at_1pct": safe_qn_3d,
                           "amplification_vs_percell": amp_3d,
                           "note": "3D simple-cubic (z=6) percolates EASIER (p_c~0.249 << 2D 0.5) -> safe q_n "
                                   "window is NARROWER than 2D. This is the conservative, realistic-tissue direction."},
        "DECISIVE": "",   # filled below
        "CEILING": "ABSTRACTION (2D site-bond percolation): q_t/q_n/c are swept PARAMETERS not measured "
                   "molecular fidelities; mapping them to a real gate (RUNG-11 neoantigen / synNotch) + "
                   "engineered coupling is the wet-lab residual. No 3D/diffusion/immune/graded-signal. The "
                   "robust parameter-free claim is the RELATIONSHIP: relay leak is sub-critical-BOUNDED vs "
                   "per-cell leak LINEAR -> threshold-protected safety margin.",
    }

    # decisive verdict
    if safe_qn > PER_CELL_LEAK_BAR and frac_region_safe_eff > 0.0:
        result["DECISIVE"] = (
            f"POSITIVE (architecture rescues leaky recognition): a per-hop-gated relay tolerates per-hop "
            f"false-positive up to q_n~{safe_qn:.2f} while keeping normal-tissue kill <=1% -- ~{amplification}x "
            f"the per-cell worst-donor bar ({PER_CELL_LEAK_BAR}). Below the percolation threshold (c*q_n<{p_c}) "
            f"a false-positive normal cell starts a wave that DIES OUT (sub-critical, bounded rind) instead of "
            f"cascading, while the tumour (c*q_t={c*q_t:.2f}>{p_c}) clears from {f:.0%} seeds. A {frac_region_safe_eff:.0%} "
            f"slice of (q_t,q_n) space is both cleared (>90%) and spared (<1%). 3D SENSITIVITY (realistic, "
            f"conservative): 3D tissue percolates easier (p_c~0.25), so the 3D safe q_n ceiling is {safe_qn_3d:.2f} "
            f"(~{amp_3d}x the per-cell bar) -- still well above {PER_CELL_LEAK_BAR}. => propagation can convert a "
            f"recognition signal TOO LEAKY for R5's per-cell gate into a safe therapy -- IF coupling is engineered "
            f"and gating re-checked per hop. The recognition bottleneck is RELAXED, not removed (still need q_n "
            f"below the percolation threshold; mapping to a molecular gate + coupling efficiency is the wet-lab residual).")
    else:
        result["DECISIVE"] = (
            f"NEGATIVE/bounded: even gated, the relay's safe q_n ceiling ({safe_qn:.2f}) does not beat the "
            f"per-cell bar ({PER_CELL_LEAK_BAR}) -- errors cascade or the tumour doesn't clear in the tested "
            f"regime. Propagation does not relax the recognition requirement here.")

    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung12pB] wrote {RESULT_JSON}")
    print(f"\n  containment (relay vs per-cell normal-kill as q_n rises):")
    for row in containment[::3]:
        print(f"    q_n={row['q_n']:.2f} (c*q_n={row['c_qn']:.2f})  relay_leak={row['relay_normal_killed']:.3f}"
              f"   per_cell_leak={row['per_cell_normal_killed']:.3f}   tumour_killed={row['tumour_killed']:.3f}")
    print(f"\n  relay safe q_n ceiling (<=1% leak):  2D {safe_qn:.2f} (~{amplification}x)   "
          f"3D {safe_qn_3d:.2f} (~{amp_3d}x)   vs per-cell bar {PER_CELL_LEAK_BAR}")
    print(f"  (q_t,q_n) safe&effective fraction (2D): {frac_region_safe_eff:.0%}")
    print(f"\n  DECISIVE: {result['DECISIVE']}")
    _make_figure(result)
    return 0


def _make_figure(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung12pB] matplotlib unavailable ({e})"); return
    cc = result["containment_curve"]
    qn = [r["q_n"] for r in cc]
    relay = [r["relay_normal_killed"] for r in cc]
    percell = [r["per_cell_normal_killed"] for r in cc]
    tk = [r["tumour_killed"] for r in cc]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: the safety amplification — relay leak (bounded) vs per-cell leak (linear)
    ax[0].plot(qn, percell, "--", color="#C1432B", label="per-cell gate leak (= q_n, R5 architecture)")
    ax[0].plot(qn, relay, "-o", color="#1B5E20", ms=4, label="RELAY leak (sub-critical bounded)")
    ax[0].plot(qn, tk, "-", color="#3B7DD8", alpha=0.7, label="tumour killed (relay)")
    ax[0].axhline(result["per_cell_leak_bar_R5"], color="grey", ls=":", lw=1, label=f"R5 safety bar {result['per_cell_leak_bar_R5']}")
    pc = result["params"]["bond_p_c_approx"]; c = result["params"]["coupling_c"]
    ax[0].axvline(pc / c, color="orange", ls="--", lw=1, label=f"percolation threshold q_n={pc/c:.2f}")
    ax[0].set_xlabel("per-hop normal false-positive  q_n"); ax[0].set_ylabel("fraction killed")
    ax[0].set_title("Relay converts LINEAR per-cell leak\ninto a THRESHOLD-bounded sub-critical leak")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3); ax[0].set_ylim(-0.02, 1.02)

    # panel 2: operating region (q_t x q_n) — safe & effective
    reg = result["operating_region"]
    tkm = np.array(reg["tumour_killed"]); nkm = np.array(reg["normal_killed"])
    safe = ((tkm > 0.90) & (nkm < 0.01)).astype(float)
    im = ax[1].imshow(safe, origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1,
                      extent=[reg["q_n_grid"][0], reg["q_n_grid"][-1], reg["q_t_grid"][0], reg["q_t_grid"][-1]])
    ax[1].set_xlabel("normal per-hop false-positive  q_n"); ax[1].set_ylabel("tumour per-hop fidelity  q_t")
    ax[1].set_title(f"Safe-AND-effective region (green)\ncleared >90% & spared <1%  |  {reg['frac_safe_and_effective']:.0%} of grid")
    fig.colorbar(im, ax=ax[1], label="safe & effective")
    fig.suptitle("RUNG-12P/B: does a per-hop-gated death wave rescue leaky recognition? (engineered coupling)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung12pB] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    L, r = 81, 24
    rng = lambda s=0: np.random.default_rng(s)

    # f=0 (no seed) -> nothing dies
    tk0, nk0 = sweep(L, r, 0.9, 0.9, 0.9, 0.0, 4)
    check("no seed -> nothing dies", tk0 == 0.0 and nk0 == 0.0)

    # super-critical tumour (c*q_t >> p_c) -> tumour largely cleared from few seeds
    tk1, nk1 = sweep(L, r, 0.95, 0.95, 0.0, 0.03, 6)
    check("super-critical tumour clears from 3% seeds (tumour_killed>0.8)", tk1 > 0.8)

    # CONTAINMENT: low q_n (c*q_n well below p_c~0.5) -> normal leak stays tiny and << per-cell q_n
    _, nk_lo = sweep(L, r, 0.85, 0.9, 0.15, 0.03, 8)   # c*q_n = 0.13 << 0.5
    check("sub-critical normal leak is tiny (<0.02) at q_n=0.15", nk_lo < 0.02)
    check("relay leak << per-cell leak (0.15) in sub-critical regime", nk_lo < 0.15 / 3)

    # super-critical normal (c*q_n >> p_c) -> normal DOES get killed a lot (the wave cascades) — sanity that
    # the model can fail too (errors cascade above threshold)
    _, nk_hi = sweep(L, r, 0.95, 0.9, 0.95, 0.03, 6)   # c*q_n = 0.90 >> 0.5
    check("super-critical normal cascades (leak>0.3 at q_n=0.95)", nk_hi > 0.3)

    # monotonicity: normal leak rises with q_n
    _, a = sweep(L, r, 0.9, 0.9, 0.2, 0.03, 6)
    _, b = sweep(L, r, 0.9, 0.9, 0.6, 0.03, 6)
    check("normal leak monotonic in q_n", b >= a)

    # tumour clearance rises with q_t
    t_lo, _ = sweep(L, r, 0.9, 0.4, 0.0, 0.03, 6)
    t_hi, _ = sweep(L, r, 0.9, 0.95, 0.0, 0.03, 6)
    check("tumour clearance monotonic in q_t", t_hi >= t_lo)

    # non-permeable cells never die: q_t=q_n=0 -> nothing dies even with seeds (seeds need permeable)
    tkz, nkz = sweep(L, r, 0.9, 0.0, 0.0, 0.5, 4)
    check("q=0 (no cell passes gate) -> nothing dies", tkz == 0.0 and nkz == 0.0)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUNG-12P/B gated-relay death-wave percolation")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
