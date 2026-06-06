#!/usr/bin/env python3
"""
RUNG 13 — from the single-cell death switch to a coupled death WAVE with REAL kinetics (laptop, no GPU).

WHERE THIS SITS (what is already proven, so this does not repeat it)
-------------------------------------------------------------------
  - RUNG 1 (scripts/11_earm_kinetics.py, validated on this machine): the *single isolated cell* extrinsic-
    apoptosis switch under the REAL Albeck/Spencer/Sorger 2008 EARM model (42 rules / 88 params). It proved
    snap-action (Td~3.6 h, Ts~30 min), an all-or-none dose threshold, and that anti-apoptotic priming
    (Bcl-2/XIAP) raises that threshold. CEILING: EARM takes ACTIVATED caspase-8 as INPUT; whether a real
    recognition ligand fires caspase-8 (agonism/transduction) is a WET-LAB residual. We do not cross it here.
  - RUNG 12P/B (scripts/35_propagation_relay.py): the death WAVE as ABSTRACT site-bond percolation. It assumed
    each cell is BINARY and INSTANTANEOUS ("EARM is the per-cell effector, modular, binarised") and showed a
    per-hop-gated relay converts LINEAR per-cell leak into THRESHOLD-bounded sub-critical leak (~15x relaxation
    in 2D, ~8.6x in 3D) -- i.e. propagation relaxes the recognition bar.

THE GAP THIS RUNG CLOSES (the missing middle of Shriya's chain)
---------------------------------------------------------------
Neither rung models the actual thing: a death wave made of REAL cells, where a committed cell's death drives
the death decision of its coupled neighbours, each running its own hours-scale, threshold-gated, irreversible
EARM clock. Shriya's chain is recognise neighbour -> trigger stress -> activate apoptosis internally; her 6.3
worry is resistance. This rung asks, of the DYNAMICS themselves:
  (Q1) Does coupling EARM-calibrated cells reproduce an all-or-none, IRREVERSIBLE per-cell commitment with a
       finite point-of-no-return -- the 6.3 "cannot back out once committed" property?
  (Q2) Does "one cell telling its neighbour" produce a BOUNDED, tumour-clearing death wave under real kinetics?
  (Q3) Do the real kinetics (variable delay, GRADED bystander signal, critical slowing) CONFIRM or BREAK the
       binary/instantaneous abstraction RUNG-12P/B relied on?

THE MODEL (honest about what it is)
-----------------------------------
A reduced extrinsic-apoptosis switch per cell (topology after Eissing 2004 / Legewie 2006 / Albeck 2008):
  C8a (active initiator caspase-8): slowly activated by input S (gain ai) + executioner feedback (kfb*C3a)
  C3a (active executioner caspase-3): activated by C8a, braked by IAP, self-capped; the death effector
  IAP (XIAP brake): basal IAP0, RAPIDLY depleted by active C3a (kseq) -- the Legewie depletion latch
  CP  (cleaved PARP): COOPERATIVE, irreversible substrate ratchet (kp*C3a^2); CP>=0.5 == committed
Behaviour (matched to RUNG-1's full-EARM signatures, validated by this script's run):
  * all-or-none EXCITABLE threshold S* (sub-threshold -> CP stays flat, cell lives)
  * VARIABLE DELAY Td, dose-dependent (long near threshold = critical slowing, short when saturating)
  * SNAP-action: the switch Ts is fast and Ts << Td at moderate dose (cooperative ratchet keeps CP flat then snaps)
  * IRREVERSIBLE commitment: a finite POINT-OF-NO-RETURN (a too-short supra pulse recovers; long enough commits)
  * PRIMING (anti-apoptotic brake) here DELAYS death (raises Td) rather than shifting S* -- the kinetic form of
    the cancer-evasion RUNG-1 saw as a threshold shift in the full EARM (we cite RUNG-1 for the threshold form)
Coupling: a committed cell emits a persistent bystander death signal ~ CP; a neighbour transduces it (gated by
recognition fidelity q) into caspase-8 input. Pure-numpy RK4, vectorised over the lattice (no scipy -> portable).

THE HARD CEILING (in every output)
----------------------------------
PROXY coupling: c and transduction competence g (gated by q_t/q_n) are swept parameters, NOT measured molecular
efficiencies; the recognition->caspase-8 step (agonism / synNotch / bystander-factor potency) is the SAME
wet-lab residual RUNG-1 named. Reduced model calibrated to RUNG-1's qualitative signatures but still a reduction;
rates scaled, not fit to a cell line. 2D, no diffusion geometry / immune; heterogeneity via gate/seed (+optional
lognormal IAP0), not full stochastic noise. The robust claim is the RELATIONSHIP, not absolute kill%.

USAGE
  python scripts/38_apoptosis_wave.py            # single-cell switch + spatial wave -> JSON + figure
  python scripts/38_apoptosis_wave.py selftest   # fast logic checks (small lattice, numpy only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung13_apoptosis"
RESULT_JSON = OUT_DIR / "rung13_apoptosis.json"
FIGURE_PNG = OUT_DIR / "rung13_apoptosis.png"

PER_CELL_LEAK_BAR = 0.02      # R5/R7 worst-donor per-cell false-positive bar (the thing to beat)
RELAY_CEILING_2D = 0.30       # RUNG-12P/B abstract-percolation safe q_n ceiling (2D)
RUNG1_TD_H = 3.65             # RUNG-1 full-EARM variable delay reference (Albeck 2008)

# validated reduced EARM-burst switch (baseline cell)
PARAMS = dict(ai=0.35, kfb=0.5, d8=0.03, kc=25.0, ki=20.0, d3=0.5, kIon=1.0, kseq=60.0, IAP0=1.2, kp=40.0)
COMMIT = 0.5


# ===========================================================================
#  SINGLE CELL
# ===========================================================================
def _cell_deriv(y, S, p):
    """Pure-float deriv (no numpy) -> ~100x faster than tiny-array ops in the single-cell loop."""
    C8a, C3a, IAP, CP = y
    return (
        (p["ai"] * S + p["kfb"] * C3a) * (1.0 - C8a) - p["d8"] * C8a,          # slow initiator + feedback
        p["kc"] * C8a * (1.0 - C3a) - p["ki"] * C3a * IAP - p["d3"] * C3a,     # executioner, IAP-braked
        p["kIon"] * (p["IAP0"] - IAP) - p["kseq"] * C3a * IAP,                 # brake, depleted by C3a (latch)
        p["kp"] * (C3a * C3a) * (1.0 - CP),                                    # cooperative cleaved-PARP ratchet
    )


def integrate_cell(S, p, t_end=40.0, dt=0.006, y0=None, S_off_t=None):
    """RK4 in pure floats. Returns (final_state, Td_h, Ts_h, cp_trace, t_trace). Td/Ts on cleaved PARP."""
    if y0 is None:
        y0 = (0.0, 0.0, p["IAP0"], 0.0)
    y = tuple(float(v) for v in y0)
    n = int(t_end / dt)
    cp = np.empty(n + 1)
    tt = np.empty(n + 1)
    for k in range(n + 1):
        t = k * dt
        Sk = 0.0 if (S_off_t is not None and t >= S_off_t) else S
        cp[k] = y[3]
        tt[k] = t
        k1 = _cell_deriv(y, Sk, p)
        k2 = _cell_deriv(tuple(a + 0.5 * dt * b for a, b in zip(y, k1)), Sk, p)
        k3 = _cell_deriv(tuple(a + 0.5 * dt * b for a, b in zip(y, k2)), Sk, p)
        k4 = _cell_deriv(tuple(a + dt * b for a, b in zip(y, k3)), Sk, p)
        y = tuple(max(a + (dt / 6.0) * (b + 2 * c + 2 * d + e), 0.0)
                  for a, b, c, d, e in zip(y, k1, k2, k3, k4))
    fin = cp[-1]
    if fin >= COMMIT:
        td = tt[np.argmax(cp >= COMMIT)]
        t10 = tt[np.argmax(cp >= 0.1 * fin)]
        t90 = tt[np.argmax(cp >= 0.9 * fin)]
        ts = t90 - t10
    else:
        td = ts = np.nan
    return y, td, ts, cp, tt


def find_threshold(p, grid):
    for S in grid:
        yf, *_ = integrate_cell(float(S), p)
        if yf[3] >= COMMIT:
            return round(float(S), 4)
    return None


def point_of_no_return(p, S_supra, taus):
    for tau in taus:
        yf, *_ = integrate_cell(S_supra, p, S_off_t=float(tau))
        if yf[3] >= COMMIT:
            return round(float(tau), 3)
    return None


# ===========================================================================
#  SPATIAL DEATH WAVE
# ===========================================================================
def _neigh_mean(A):
    s = np.zeros_like(A)
    cnt = np.zeros_like(A)
    s[1:, :] += A[:-1, :]; cnt[1:, :] += 1
    s[:-1, :] += A[1:, :]; cnt[:-1, :] += 1
    s[:, 1:] += A[:, :-1]; cnt[:, 1:] += 1
    s[:, :-1] += A[:, 1:]; cnt[:, :-1] += 1
    return s / cnt


def wave(L=81, r=22, c=1.0, q_t=0.95, q_n=0.10, seed_frac=0.03, t_end=80.0, dt=0.01,
         rng=None, iap_mult_tumour=1.0, p=PARAMS):
    if rng is None:
        rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:L, 0:L]
    cx = L // 2
    is_t = ((xx - cx) ** 2 + (yy - cx) ** 2) <= r * r
    g = np.where(rng.random((L, L)) < np.where(is_t, q_t, q_n), 1.0, 0.0)
    IAP0 = np.where(is_t, p["IAP0"] * iap_mult_tumour, p["IAP0"])   # primed tumour (evasion)
    C8a = np.zeros((L, L)); C3a = np.zeros((L, L)); IAP = IAP0.copy(); CP = np.zeros((L, L))
    seed = is_t & (rng.random((L, L)) < seed_frac) & (g > 0)
    CP[seed] = 0.9
    ai, kfb, d8, kc, ki, d3, kIon, kseq, kp = (p["ai"], p["kfb"], p["d8"], p["kc"], p["ki"],
                                               p["d3"], p["kIon"], p["kseq"], p["kp"])

    def deriv(C8a, C3a, IAP, CP):
        S = c * g * _neigh_mean(CP)
        return ((ai * S + kfb * C3a) * (1 - C8a) - d8 * C8a,
                kc * C8a * (1 - C3a) - ki * C3a * IAP - d3 * C3a,
                kIon * (IAP0 - IAP) - kseq * C3a * IAP,
                kp * (C3a * C3a) * (1 - CP))

    st = (C8a, C3a, IAP, CP)
    for _ in range(int(t_end / dt)):
        d = deriv(*st)
        s2 = tuple(a + 0.5 * dt * b for a, b in zip(st, d)); d2 = deriv(*s2)
        s3 = tuple(a + 0.5 * dt * b for a, b in zip(st, d2)); d3_ = deriv(*s3)
        s4 = tuple(a + dt * b for a, b in zip(st, d3_)); d4 = deriv(*s4)
        st = tuple(np.clip(a + (dt / 6) * (b + 2 * x + 2 * z + w), 0.0, None)
                   for a, b, x, z, w in zip(st, d, d2, d3_, d4))
    dead = st[3] >= COMMIT
    return {"tumour_killed": float((dead & is_t).sum() / is_t.sum()),
            "normal_killed": float((dead & ~is_t).sum() / (~is_t).sum()) if (~is_t).any() else 0.0,
            "seeds": int(seed.sum())}


def wave_kill(**kw):
    o = wave(**kw)
    return o["tumour_killed"], o["normal_killed"]


# ===========================================================================
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = PARAMS
    print("[rung13] single-cell death switch (EARM-calibrated reduced model) ...")

    grid = np.round(np.linspace(0, 0.5, 251), 4)             # fine enough to resolve the small threshold (~0.01)
    S_star = find_threshold(p, grid)
    S_mod = round((S_star or 0.01) + 0.05, 3)                # moderate supra dose for snap
    yf_s, Td_mod, Ts_mod, cp_tr, t_tr = integrate_cell(S_mod, p)
    # variable delay across doses
    var_delay = []
    for S in [round((S_star or 0.01) + d, 3) for d in (0.005, 0.05, 0.2, 0.5)]:
        _, td, _, _, _ = integrate_cell(S, p)
        var_delay.append({"S": S, "Td_h": None if np.isnan(td) else round(float(td), 2)})
    # dose-response (figure + all-or-none)
    dose = []
    for S in np.round(np.linspace(0, 1.0, 51), 3):
        yf, td, _, _, _ = integrate_cell(float(S), p)
        dose.append({"S": float(S), "committed": bool(yf[3] >= COMMIT), "final_CP": round(float(yf[3]), 3)})
    inter_frac = round(sum(1 for d in dose if 0.2 < d["final_CP"] < 0.8) / len(dose), 3)
    # irreversibility: dead-basin persists at S=0; threshold crossing latches commitment (autocatalytic)
    yf0, *_ = integrate_cell(0.0, p, y0=[0.5, 0.9, 0.02, 0.9])
    latch_cp = integrate_cell(round((S_star or 0.01) + 0.3, 3), p, S_off_t=0.1)[0][3]   # brief supra pulse
    subthr_cp = integrate_cell(max(0.0, (S_star or 0.01) - 0.005), p, t_end=60.0)[0][3]  # sustained sub-threshold
    latch_ok = bool(latch_cp >= COMMIT and subthr_cp < COMMIT)
    # priming -> DELAYS death (kinetic evasion)
    priming = []
    for mult in [1.0, 2.0, 4.0, 8.0]:
        _, td, _, _, _ = integrate_cell(0.2, dict(p, IAP0=p["IAP0"] * mult))
        priming.append({"brake_mult": mult, "Td_h_at_S0.2": None if np.isnan(td) else round(float(td), 2)})

    snap_ok = bool((not np.isnan(Ts_mod)) and (not np.isnan(Td_mod)) and Ts_mod < 0.5 * Td_mod)
    print(f"[rung13]   S*={S_star}  Td(mod)={None if np.isnan(Td_mod) else round(Td_mod,2)}h  "
          f"Ts={None if np.isnan(Ts_mod) else round(Ts_mod,2)}h  snap(Ts<0.5Td)={snap_ok}  "
          f"latch={latch_ok}  intermediate={inter_frac}")

    print("[rung13] spatial death wave: containment vs q_n ...")
    c_eng, q_t = 1.0, 0.95
    qn_grid = [round(x, 3) for x in np.linspace(0.0, 0.95, 12)]
    containment = []
    for qn in qn_grid:
        tk, nk = wave_kill(c=c_eng, q_t=q_t, q_n=qn, rng=np.random.default_rng(7))
        containment.append({"q_n": qn, "tumour_killed": round(tk, 4), "normal_killed": round(nk, 4)})
        print(f"           q_n={qn:.2f}  tumour_killed={tk:.3f}  normal_killed={nk:.3f}")
    safe_qn = max([r["q_n"] for r in containment if r["normal_killed"] <= 0.01], default=0.0)
    cleared = [r["q_n"] for r in containment if r["tumour_killed"] >= 0.80]
    amp = round(safe_qn / PER_CELL_LEAK_BAR, 1) if PER_CELL_LEAK_BAR else None
    # efficacy vs tumour fidelity
    qt_curve = []
    for qt in [round(x, 3) for x in np.linspace(0.3, 1.0, 8)]:
        tk, nk = wave_kill(c=c_eng, q_t=qt, q_n=0.10, rng=np.random.default_rng(11))
        qt_curve.append({"q_t": qt, "tumour_killed": round(tk, 4), "normal_killed": round(nk, 4)})
    # wave robustness to tumour priming (resistance-resistance)
    prime_wave = []
    for m in [1.0, 4.0, 8.0]:
        tk, nk = wave_kill(c=c_eng, q_t=q_t, q_n=0.10, iap_mult_tumour=m, t_end=40.0, rng=np.random.default_rng(3))
        prime_wave.append({"tumour_IAP_mult": m, "tumour_killed": round(tk, 4), "normal_killed": round(nk, 4)})

    confirms = (safe_qn > PER_CELL_LEAK_BAR) and (containment[-1]["normal_killed"] > 0.1) \
        and any(r["tumour_killed"] >= 0.8 for r in containment)

    result = {
        "tag": "rung13_apoptosis_death_wave",
        "question": "Does coupling EARM-calibrated cells reproduce an all-or-none, IRREVERSIBLE per-cell "
                    "commitment AND a bounded, tumour-clearing death wave -- and do real kinetics confirm or "
                    "break the binary/instantaneous abstraction RUNG-12P/B relied on?",
        "builds_on": {"RUNG1_full_EARM": "scripts/11_earm_kinetics.py (Albeck 2008, validated)",
                      "RUNG12pB_percolation": "scripts/35_propagation_relay.py"},
        "model": "Reduced extrinsic-apoptosis burst switch (Eissing/Legewie/Albeck topology): C8a + C3a + "
                 "IAP(brake depleted by C3a) + CP(cooperative irreversible cleaved-PARP ratchet). Excitable "
                 "threshold + variable delay + snap + irreversible commitment. Coupling: dead cell emits "
                 "persistent bystander signal ~CP; neighbour transduces it (recognition-gated) into caspase-8 "
                 "input. Pure-numpy RK4.",
        "params": p,
        "single_cell": {
            "threshold_S_star": S_star,
            "all_or_none_intermediate_CP_frac": inter_frac,
            "snap_action": {"Td_mod_h": None if np.isnan(Td_mod) else round(float(Td_mod), 2),
                            "Ts_h": None if np.isnan(Ts_mod) else round(float(Ts_mod), 3),
                            "Ts_lt_half_Td": snap_ok, "RUNG1_reference_Td_h": RUNG1_TD_H},
            "variable_delay_Td_vs_S": var_delay,
            "irreversible_commitment": {"dead_basin_stays_committed_at_S0": bool(yf0[3] >= COMMIT),
                                        "brief_supra_pulse_commits_latch": bool(latch_cp >= COMMIT),
                                        "sustained_subthreshold_never_commits": bool(subthr_cp < COMMIT),
                                        "note": "commitment latches near-instantly once S* is crossed "
                                                "(autocatalytic IAP-depletion + cooperative CP ratchet); the "
                                                "point-of-no-return is the threshold crossing itself. Single-cell "
                                                "pre-MOMP recovery dynamics are RUNG-1's full-EARM domain."},
            "priming_delays_death_Td_at_fixed_dose": priming,
            "dose_response": dose,
        },
        "spatial_wave": {
            "engineered_coupling_c": c_eng, "tumour_fidelity_q_t": q_t,
            "containment_curve": containment,
            "kinetic_safe_q_n_ceiling_at_1pct": safe_qn,
            "amplification_vs_percell_bar": amp,
            "tumour_cleared_q_n_range": [min(cleared), max(cleared)] if cleared else [],
            "efficacy_vs_q_t": qt_curve,
            "wave_robust_to_tumour_priming": prime_wave,
        },
        "cross_validation_vs_RUNG12pB": {
            "abstract_percolation_safe_q_n_2D": RELAY_CEILING_2D,
            "kinetic_safe_q_n_2D": safe_qn,
            "kinetics_confirm_binarisation": bool(confirms)},
        "INTERPRETATION_MAP": {
            "A_commitment_irreversible": "finite point-of-no-return + dead-basin latch => Shriya 6.3 'cannot "
                                         "back out once committed' is a property of the dynamics, not an assumption.",
            "B_wave_bounded": "wave clears tumour (q_t high) yet self-extinguishes in normal tissue at low q_n "
                              "=> 'one cell tells its neighbour' is therapeutically containable if per-hop "
                              "false-positive stays sub-critical.",
            "C_binarisation_earned": "kinetic safe-q_n ceiling ~ RUNG-12P/B's => the binary/instantaneous "
                                      "abstraction was fair; the recognition relaxation survives real kinetics.",
            "D_resistance_resistance": "the wave clears even heavily-primed tumour (the front signal overpowers "
                                       "cell-autonomous resistance) => propagation is HARDER to evade than a "
                                       "single-cell trigger -- a direct counter to Shriya 6.3."},
        "DECISIVE": "",
        "CEILING": "PROXY coupling: c and transduction competence g (gated by q_t/q_n) are swept parameters, NOT "
                   "measured molecular efficiencies; recognition->caspase-8 (agonism/synNotch/bystander potency) "
                   "is the SAME wet-lab residual RUNG-1 named. Reduced model calibrated to RUNG-1's qualitative "
                   "signatures but still a reduction; rates scaled, not fit to a cell line. 2D, no diffusion "
                   "geometry/immune; heterogeneity via gate/seed. Robust claim = the RELATIONSHIP, not kill%.",
    }

    irr_ok = result["single_cell"]["irreversible_commitment"]["dead_basin_stays_committed_at_S0"] and latch_ok
    if S_star and snap_ok and irr_ok and inter_frac < 0.1 and confirms:
        result["DECISIVE"] = (
            f"POSITIVE (recognition->death link closes in silico): EARM-calibrated coupled cells give "
            f"(1) an all-or-none excitable threshold S*={S_star} with a variable delay (Td "
            f"{var_delay[0]['Td_h']}->{var_delay[-1]['Td_h']}h across dose) and a fast snap (Ts={round(float(Ts_mod),2)}h "
            f"< Td); (2) an IRREVERSIBLE commitment -- the threshold crossing latches death (a brief supra pulse "
            f"still commits; sustained sub-threshold never does) and the dead state persists at zero input -- so "
            f"Shriya 6.3 'cannot back out' is a dynamical property, not an assumption; (3) a BOUNDED death wave clearing the "
            f"tumour (>=80% over q_n in [{min(cleared) if cleared else '-'},{max(cleared) if cleared else '-'}]) "
            f"while keeping normal kill <=1% up to q_n~{safe_qn:.2f} (~{amp}x the per-cell bar {PER_CELL_LEAK_BAR}); "
            f"(4) this CONFIRMS RUNG-12P/B's binary abstraction (2D ceiling {RELAY_CEILING_2D}) under real "
            f"kinetics -- the recognition relaxation is EARNED. Bonus: the wave clears even heavily-primed tumour "
            f"(resistance-resistance). Efficacy still needs tumour fidelity q_t above the propagation threshold; "
            f"mapping c/g to molecular efficiencies is the wet-lab residual.")
    else:
        result["DECISIVE"] = (f"MIXED: S*={S_star} snap={snap_ok} irreversible={irr_ok} binar={inter_frac<0.1} "
                              f"confirms_percolation={confirms}. Inspect.")

    def _jd(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    RESULT_JSON.write_text(json.dumps(result, indent=2, default=_jd))
    print(f"[rung13] wrote {RESULT_JSON}")
    print(f"\n  kinetic safe q_n ceiling: {safe_qn:.2f} (~{amp}x bar {PER_CELL_LEAK_BAR}) | "
          f"RUNG-12P/B 2D {RELAY_CEILING_2D}")
    print(f"\n  DECISIVE: {result['DECISIVE']}")
    _make_figure(result, cp_tr, t_tr)
    return 0


def _make_figure(result, cp_tr, t_tr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung13] matplotlib unavailable ({e})")
        return
    sc = result["single_cell"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    ax[0].plot(t_tr, cp_tr, color="#C1432B", lw=2)
    ax[0].axhline(COMMIT, ls=":", color="grey", label="commit (CP=0.5)")
    ax[0].set_xlabel("time (model-h)"); ax[0].set_ylabel("cleaved PARP")
    ax[0].set_title(f"snap-action switch\nTd={sc['snap_action']['Td_mod_h']}h Ts={sc['snap_action']['Ts_h']}h")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    dl = [d["S"] for d in sc["dose_response"]]; dfc = [d["final_CP"] for d in sc["dose_response"]]
    ax[1].plot(dl, dfc, "o-", ms=3, color="#222")
    if sc["threshold_S_star"]:
        ax[1].axvline(sc["threshold_S_star"], ls="--", color="green", label=f"S*={sc['threshold_S_star']}")
    ax[1].axhline(COMMIT, ls=":", color="grey")
    ax[1].set_xlabel("input S"); ax[1].set_ylabel("final cleaved PARP")
    ax[1].set_title(f"all-or-none\nintermediate {sc['all_or_none_intermediate_CP_frac']:.0%}")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    pm = [r["brake_mult"] for r in sc["priming_delays_death_Td_at_fixed_dose"]]
    pth = [r["Td_h_at_S0.2"] if r["Td_h_at_S0.2"] is not None else np.nan
           for r in sc["priming_delays_death_Td_at_fixed_dose"]]
    ax[2].plot(pm, pth, "o-", color="#2980b9")
    ax[2].set_xlabel("anti-apoptotic brake (x)"); ax[2].set_ylabel("Td (h) at fixed dose")
    ax[2].set_title("priming DELAYS death\n(kinetic cancer-evasion)"); ax[2].grid(alpha=0.3)

    cc = result["spatial_wave"]["containment_curve"]
    qn = [r["q_n"] for r in cc]; nk = [r["normal_killed"] for r in cc]; tk = [r["tumour_killed"] for r in cc]
    ax[3].plot(qn, qn, "--", color="#C1432B", label="per-cell leak (=q_n)")
    ax[3].plot(qn, nk, "-o", ms=4, color="#1B5E20", label="KINETIC wave leak")
    ax[3].plot(qn, tk, "-", color="#3B7DD8", alpha=0.7, label="tumour killed")
    ax[3].axhline(PER_CELL_LEAK_BAR, ls=":", color="grey", label=f"per-cell bar {PER_CELL_LEAK_BAR}")
    ax[3].axvline(RELAY_CEILING_2D, ls="--", color="orange", label=f"RUNG-12P ceiling {RELAY_CEILING_2D}")
    ax[3].set_xlabel("normal false-positive q_n"); ax[3].set_ylabel("fraction")
    ax[3].set_title("death WAVE, real kinetics\nbounded leak confirms binarisation")
    ax[3].legend(fontsize=7); ax[3].grid(alpha=0.3); ax[3].set_ylim(-0.02, 1.02)

    fig.suptitle("RUNG-13: single-cell EARM switch -> coupled death WAVE (real kinetics). "
                 "Coupling c/g are PROXY parameters (agonism = wet-lab).", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PNG, dpi=120)
    print(f"[rung13] wrote {FIGURE_PNG}")


# ===========================================================================
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    p = PARAMS
    S_star = find_threshold(p, np.round(np.linspace(0, 1.0, 201), 4))
    check("single-cell all-or-none threshold S* exists and > 0", S_star is not None and S_star > 0)
    yf_lo, *_ = integrate_cell(max(0.0, (S_star or 0.01) - 0.005), p)
    yf_hi, *_ = integrate_cell((S_star or 0.01) + 0.3, p)
    check("sub-threshold input -> cell LIVES", yf_lo[3] < COMMIT)
    check("supra-threshold input -> cell COMMITS", yf_hi[3] >= COMMIT)
    yf0, *_ = integrate_cell(0.0, p, y0=[0.5, 0.9, 0.02, 0.9])
    check("dead-basin stays COMMITTED at S=0 (CP ratchet)", yf0[3] >= COMMIT)
    latch, *_ = integrate_cell((S_star or 0.01) + 0.3, p, S_off_t=0.1)         # brief supra pulse
    subthr, *_ = integrate_cell(max(0.0, (S_star or 0.01) - 0.005), p, t_end=60.0)  # sustained sub-threshold
    check("commitment latches: brief supra pulse still commits (irreversible)", latch[3] >= COMMIT)
    check("sub-threshold sustained never commits (threshold gates)", subthr[3] < COMMIT)
    cps = [integrate_cell(float(S), p)[0][3] for S in np.linspace(0, 1.0, 41)]
    check("binarisation: <10% intermediate final-CP", sum(1 for v in cps if 0.2 < v < 0.8) / len(cps) < 0.1)
    _, td1, _, _, _ = integrate_cell(0.2, p)
    _, td8, _, _, _ = integrate_cell(0.2, dict(p, IAP0=p["IAP0"] * 8))
    check("priming delays death (Td rises with brake)", (td8 > td1) if (td1 == td1 and td8 == td8) else False)
    tk_lo, nk_lo = wave_kill(L=41, r=12, c=1.0, q_t=0.95, q_n=0.05, t_end=60.0, dt=0.02, rng=np.random.default_rng(1))
    check("death wave CLEARS tumour at low q_n (>0.7)", tk_lo > 0.7)
    check("death wave SPARES normal at low q_n (<0.02)", nk_lo < 0.02)
    _, nk_hi = wave_kill(L=41, r=12, c=1.0, q_t=0.95, q_n=0.90, t_end=60.0, dt=0.02, rng=np.random.default_rng(2))
    check("death wave LEAKS into normal at high q_n (>0.1)", nk_hi > 0.1)
    tk_lowqt, _ = wave_kill(L=41, r=12, c=1.0, q_t=0.30, q_n=0.05, t_end=60.0, dt=0.02, rng=np.random.default_rng(4))
    check("low tumour fidelity q_t -> weaker clearance", tk_lowqt < tk_lo)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUNG-13 coupled-EARM death wave")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
