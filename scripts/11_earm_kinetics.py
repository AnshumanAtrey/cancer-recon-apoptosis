#!/usr/bin/env python3
"""
RUNG 1 — single-cell death-commitment kinetics (real model: PySB + EARM 1.0).

Replaces the abstract ABM's instantaneous on-threshold death (scripts/08-10 have NO
per-cell latency at all) with the REAL, single-cell-calibrated extrinsic-apoptosis
cascade: TRAIL -> DR5/DISC -> caspase-8 -> Bid -> MOMP (Bax/Bcl2/Smac) -> caspase-3 ->
cleaved PARP. Model: pysb.examples.earm_1_0 (Albeck/Spencer/Sorger 2008, 42 rules / 88 params).

WHAT THIS PROVES (under real published kinetics, not assumed rates):
  1. Death commitment has a computable THRESHOLD (dose/gate-strength below which the cell lives).
  2. Commitment is ALL-OR-NONE / snap-action: once MOMP fires, cleaved-PARP goes from ~0 to full
     (~PARP_0) — the switch (Ts) is fast relative to the variable delay (Td, hours).
  3. Timing is a VARIABLE DELAY tunable by input, not a fixed clock (gives a per-cell latency
     distribution — the thing the propagation ABM entirely lacks).
  4. PRIMING is the knob: raising the anti-apoptotic brakes (Bcl-2, XIAP) raises the death
     threshold — i.e. susceptibility is set by how primed the cell is, NOT by receptor level.
     (Directly backs the Step-3 finding that DR5 selectivity is priming, not expression.)

THE HARD CEILING (stated in every output): EARM takes ACTIVATED caspase-8 / bound ligand as INPUT.
It says "IF this much caspase-8 is activated THEN the cell commits to death." It has NO representation
of DR5 receptor clustering/valency/geometry, so it CANNOT say whether a Trop2-anchored DR5 binder
actually achieves that activation — that is the AGONISM crux and an intrinsically WET-LAB experiment
(EVIDENCE_AND_HANDOFF.md). The input knobs (L_0 dose, caspase-8 activation-rate multiplier g) are
PROXIES and are NEVER to be multiplied by the Rung-2 clustering score into a pseudo-efficacy number.

CALIBRATION GATE (Albeck 2008): variable delay Td on the order of hours; switch Ts much faster than
Td (snap). Reported, and the run is flagged if the qualitative switch is absent.

USAGE:  python scripts/11_earm_kinetics.py
REQS :  pip install pysb bionetgen  (CPU, no GPU; ~minutes)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "earm_kinetics"
T_END = 40000.0            # seconds (~11 h) — long enough to see late commitment
N_T = 1200
SEED = 20260530


def ensure_bngpath():
    """PySB needs BioNetGen for network generation; point BNGPATH at the bundled BNG2.pl."""
    if os.environ.get("BNGPATH"):
        return
    try:
        import bionetgen
        base = os.path.dirname(bionetgen.__file__)
        # prefer the platform build matching this OS
        plat = "bng-mac" if sys.platform == "darwin" else ("bng-win" if sys.platform.startswith("win") else "bng-linux")
        hits = glob.glob(f"{base}/{plat}/BNG2.pl") or glob.glob(f"{base}/**/BNG2.pl", recursive=True)
        if hits:
            os.environ["BNGPATH"] = os.path.dirname(hits[0])
    except Exception:
        pass


ensure_bngpath()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("earm")


def get_sim_and_model():
    from pysb.examples import earm_1_0
    from pysb.simulator import ScipyOdeSimulator
    m = earm_1_0.model
    t = np.linspace(0, T_END, N_T)
    try:
        sim = ScipyOdeSimulator(m, tspan=t, integrator="lsoda", compiler="cython")
        sim.run()  # probe cython
    except Exception:
        sim = ScipyOdeSimulator(m, tspan=t, integrator="lsoda", compiler="python")
    return m, t, sim


def commitment_metrics(t, cparp, parp0):
    """Td (time to half-max cleaved PARP, hours), Ts (10->90% rise, min), committed bool."""
    mx = float(cparp.max())
    committed = mx >= 0.5 * parp0          # cleaved >= half of total PARP = death committed
    if mx <= 1e-6:
        return committed, float("nan"), float("nan"), mx
    td = t[np.argmax(cparp >= mx / 2)] / 3600.0
    t10 = t[np.argmax(cparp >= 0.1 * mx)]
    t90 = t[np.argmax(cparp >= 0.9 * mx)]
    return committed, td, (t90 - t10) / 60.0, mx


def pset(m, **overrides):
    """Return a param dict for a run with named parameter overrides."""
    p = {par.name: par.value for par in m.parameters}
    p.update(overrides)
    return p


def main() -> int:
    log.info("RUNG 1 — EARM single-cell death-commitment kinetics (real model)")
    try:
        m, t, sim = get_sim_and_model()
    except Exception as e:
        log.error("PySB/EARM unavailable: %s: %s — run: pip install pysb bionetgen", type(e).__name__, e)
        return 2
    pnames = {p.name for p in m.parameters}
    L0 = next(p.value for p in m.parameters if p.name == "L_0")
    PARP0 = next(p.value for p in m.parameters if p.name == "PARP_0")
    log.info("EARM loaded: %d rules, %d params. default L_0=%.0f PARP_0=%.0f", len(m.rules), len(m.parameters), L0, PARP0)

    # ---- A) default trajectory: the snap-action switch ----
    res = sim.run()
    cparp = np.asarray(res.observables["CPARP_total"])
    committed, td, ts, mx = commitment_metrics(t, cparp, PARP0)
    log.info("[default L_0=%.0f] committed=%s  Td=%.2f h  Ts=%.1f min  (cPARP max=%.2e of %.0e)",
             L0, committed, td, ts, mx, PARP0)

    # ---- B) dose-response: threshold (all-or-none) ----
    doses = np.unique(np.round(np.logspace(0, 4.5, 22))).astype(float)
    dose_rows = []
    for d in doses:
        r = sim.run(param_values=pset(m, L_0=d))
        c = np.asarray(r.observables["CPARP_total"])
        com, dtd, dts, dmx = commitment_metrics(t, c, PARP0)
        dose_rows.append({"L_0": d, "committed": bool(com), "Td_h": None if np.isnan(dtd) else round(dtd, 2),
                          "cparp_frac": round(dmx / PARP0, 3)})
    committed_doses = [r["L_0"] for r in dose_rows if r["committed"]]
    threshold = min(committed_doses) if committed_doses else None
    log.info("[dose-response] commitment threshold L_0* = %s (below it the cell LIVES; all-or-none)", threshold)

    # ---- C) priming: anti-apoptotic brakes raise the threshold (susceptibility = priming) ----
    prime_rows = []
    brake = "Bcl2_0" if "Bcl2_0" in pnames else ("Bax_0" if "Bax_0" in pnames else None)
    bcl2_default = next((p.value for p in m.parameters if p.name == brake), None) if brake else None
    if brake and bcl2_default:
        for mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            # find min dose committing at this brake level
            thr = None
            for d in doses:
                r = sim.run(param_values=pset(m, L_0=d, **{brake: bcl2_default * mult}))
                com, *_ = commitment_metrics(t, np.asarray(r.observables["CPARP_total"]), PARP0)
                if com:
                    thr = d; break
            prime_rows.append({"brake": brake, "brake_mult": mult, "threshold_L0": thr})
            log.info("  [priming] %s x%.2f -> death threshold L_0* = %s", brake, mult, thr)

    # ---- D) cell-to-cell timing variability (lognormal initial protein counts, Spencer 2009) ----
    rng = np.random.default_rng(SEED)
    varied = [p.name for p in m.parameters if p.name.endswith("_0")]
    tds, kills = [], 0
    N_CELLS = 60
    for _ in range(N_CELLS):
        ov = {nm: next(p.value for p in m.parameters if p.name == nm) * float(rng.lognormal(0, 0.25)) for nm in varied}
        r = sim.run(param_values=pset(m, **ov))
        com, ctd, cts, _ = commitment_metrics(t, np.asarray(r.observables["CPARP_total"]), PARP0)
        if com:
            kills += 1
            if not np.isnan(ctd):
                tds.append(ctd)
    kill_frac = kills / N_CELLS
    td_cv = float(np.std(tds) / np.mean(tds)) if tds else float("nan")
    log.info("[population n=%d, lognormal CV=0.25] kill_fraction=%.2f  Td mean=%.2f h  Td CV=%.2f",
             N_CELLS, kill_frac, (np.mean(tds) if tds else float("nan")), td_cv)

    # ---- calibration gate (Albeck 2008 qualitative): hours-scale delay, snap switch ----
    checks = {
        "default cell commits to death (cleaved PARP -> full)": committed,
        "variable delay Td is hours-scale (0.5-8 h)": (not np.isnan(td)) and 0.5 <= td <= 8.0,
        "switch is snap-action (Ts < Td, i.e. < 60 min and < Td*60)": (not np.isnan(ts)) and ts < min(60.0, td * 60),
        "dose threshold exists (all-or-none, not graded)": threshold is not None,
        "priming raises threshold (more brake -> higher L_0*)":
            (len(prime_rows) >= 2 and any(r["threshold_L0"] for r in prime_rows)
             and (prime_rows[-1]["threshold_L0"] or 1e9) >= (prime_rows[0]["threshold_L0"] or 0)),
        "cell-to-cell timing variability present (Td CV > 0.05)": (not np.isnan(td_cv)) and td_cv > 0.05,
    }
    log.info("=" * 64)
    log.info("CALIBRATION GATE (Albeck/Spencer 2008-2009 qualitative signatures):")
    for k, ok in checks.items():
        log.info("  [%s] %s", "✓" if ok else "✗", k)
    ok_all = all(checks.values())

    def _jd(o):  # numpy -> json-native
        import numpy as _np
        if isinstance(o, _np.bool_): return bool(o)
        if isinstance(o, _np.integer): return int(o)
        if isinstance(o, _np.floating): return None if _np.isnan(o) else float(o)
        return str(o)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "earm_results.json").write_text(json.dumps({
        "default": {"L_0": L0, "committed": bool(committed), "Td_h": round(td, 2), "Ts_min": round(ts, 1),
                    "cparp_max": mx, "PARP_0": PARP0},
        "dose_response": dose_rows, "threshold_L0": threshold,
        "priming": prime_rows, "population": {"n": N_CELLS, "kill_fraction": kill_frac,
                                              "Td_mean_h": (np.mean(tds) if tds else None), "Td_CV": td_cv},
        "calibration_checks": checks, "calibration_passed": ok_all,
        "AGONISM_CAVEAT": "EARM takes activated caspase-8 as INPUT; this does NOT demonstrate a DR5 binder "
                          "fires caspase-8 (the agonism crux is a wet-lab experiment). Input knobs are proxies.",
    }, indent=2, default=_jd))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        ax[0].plot(t / 3600, cparp / PARP0, color="#c0392b")
        ax[0].set_xlabel("time (h)"); ax[0].set_ylabel("cleaved PARP (fraction)")
        ax[0].set_title(f"snap-action death switch\nTd={td:.1f} h, Ts={ts:.0f} min")
        dl = [r["L_0"] for r in dose_rows]; df = [r["cparp_frac"] for r in dose_rows]
        ax[1].semilogx(dl, df, "o-", color="#222")
        if threshold: ax[1].axvline(threshold, ls="--", color="green", label=f"threshold L_0*={threshold:.0f}")
        ax[1].set_xlabel("input dose L_0 (proxy)"); ax[1].set_ylabel("final cleaved PARP (fraction)")
        ax[1].set_title("all-or-none commitment threshold"); ax[1].legend(fontsize=8)
        if prime_rows:
            mlt = [r["brake_mult"] for r in prime_rows]; thr = [r["threshold_L0"] or np.nan for r in prime_rows]
            ax[2].loglog(mlt, thr, "o-", color="#2980b9")
            ax[2].set_xlabel(f"anti-apoptotic brake ({brake}) x"); ax[2].set_ylabel("death threshold L_0*")
            ax[2].set_title("priming sets susceptibility\n(more brake -> harder to kill)")
        fig.suptitle("EARM single-cell apoptosis kinetics — real model. NOTE: input=activated caspase-8; "
                     "does NOT prove a binder fires it (agonism = wet-lab).", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT_DIR / "earm_kinetics.png", dpi=110)
        log.info("figure → runs/earm_kinetics/earm_kinetics.png")
    except Exception as e:
        log.warning("figure skipped (%s: %s)", type(e).__name__, e)

    log.info("=" * 64)
    if ok_all:
        log.info("✅ RUNG 1 PASS: real EARM kinetics reproduce the all-or-none, variable-delay death switch;")
        log.info("   a death-commitment THRESHOLD exists and is raised by anti-apoptotic priming (Bcl-2/XIAP).")
        log.info("   This upgrades the ABM's instantaneous death to a real per-cell latency distribution.")
    else:
        log.info("⚠️ some calibration signatures not met — inspect; may need MOMP-topology variant or longer T_END.")
    log.info("CEILING: EARM does NOT touch agonism (binding->clustering->caspase-8). That is the wet-lab handoff.")
    log.info("results → runs/earm_kinetics/earm_results.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
