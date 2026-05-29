#!/usr/bin/env python3
"""
Metastasis-clearing hybrid — solve the gap sim #1/#2 surfaced: a contact-propagating death wave
clears a CONNECTED tumour but cannot jump to disconnected metastatic foci.

The fix (and how a real systemic therapy actually works): SCOUT + AMPLIFY.
  - SCOUT (systemic): the recognition agent circulates and triggers death in a SMALL fraction of
    cancer cells ANYWHERE (recognition-gated → only antigen+ cancer), so every focus — including
    disconnected metastases — gets at least one seed.
  - AMPLIFY (local): the recognition-gated self-propagating wave then clears each focus locally,
    so you don't need to saturate every cell with drug.

Compares three strategies on a tissue of PRIMARY tumour + scattered metastatic foci:
  local_only    — seed the primary only, propagation on  → clears primary, METASTASES SURVIVE (the gap)
  systemic_only — seed each cancer cell w.p. p, NO propagation → must hit ~every cell (needs huge dose)
  hybrid        — low systemic seed p + propagation → clears ALL foci at a fraction of the dose

FALSIFIABLE PREDICTION: hybrid reaches >99% total clearance at a p_seed FAR below systemic_only's
(amplification = reach + dose economy), at ~0 healthy death (gated). If hybrid does NOT clear the
metastases that local_only misses, OR needs as much dose as systemic_only, the hybrid thesis is wrong.

CAVEAT: abstract agent-based model, assumed parameters — tests the LOGIC of scout+amplify, not kinetics.

USAGE:  python scripts/10_metastasis_hybrid.py
REQS :  numpy (matplotlib optional). CPU, seconds.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "metastasis_hybrid"

GRID = 161
SEED = 20260530
DEATH_STEPS = 200
HEALTHY, CANCER, DEAD = 1, 2, 3
NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
P_SWEEP = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.99]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("methybrid")


def neighbour_count(mask):
    acc = np.zeros(mask.shape, np.int16)
    m = mask.astype(np.int16)
    for dy, dx in NEI8:
        acc += np.roll(np.roll(m, dy, 0), dx, 1)
    return acc


def build_tissue():
    state = np.full((GRID, GRID), HEALTHY, np.int8)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    c = GRID // 2
    state[(yy - c) ** 2 + (xx - c) ** 2 <= 16 ** 2] = CANCER          # primary tumour
    mets = [(35, 35), (35, 126), (126, 35), (126, 126), (30, 80), (130, 90), (80, 30)]
    for my, mx in mets:                                               # disconnected metastatic foci
        state[(yy - my) ** 2 + (xx - mx) ** 2 <= 5 ** 2] = CANCER
    return state, (c, c)


def run(state0, p_seed, propagate, seed_primary_only, primary_center, rng):
    state = state0.copy()
    cancer0 = int((state == CANCER).sum())
    healthy0 = int((state == HEALTHY).sum())
    if seed_primary_only:
        cy, cx = primary_center
        if state[cy, cx] == CANCER:
            state[cy, cx] = DEAD
    else:
        seed = (state == CANCER) & (rng.random((GRID, GRID)) < p_seed)   # systemic, recognition-gated
        state[seed] = DEAD
    dose = int((state == DEAD).sum())          # cells initially triggered = "drug dose" proxy
    for t in range(DEATH_STEPS):
        if not propagate:
            break
        dead = state == DEAD
        dying = (neighbour_count(dead) > 0) & (state == CANCER)        # gated wave: kills cancer only
        if not dying.any():
            break
        state[dying] = DEAD
    cleared = 1 - int((state == CANCER).sum()) / max(cancer0, 1)
    healthy_killed = (healthy0 - int((state == HEALTHY).sum())) / max(healthy0, 1)
    return {"cleared": cleared, "healthy_killed": healthy_killed, "dose_frac": dose / max(cancer0, 1)}


def min_p_for_clearance(curve, target=0.99):
    for p, c in curve:
        if c >= target:
            return p
    return None


def main() -> int:
    log.info("metastasis hybrid — scout(systemic) + amplify(gated propagation) vs the gap")
    state0, pc = build_tissue()
    rng = np.random.default_rng(SEED)

    # local_only: seed the primary only, propagate
    loc = run(state0, 0.0, propagate=True, seed_primary_only=True, primary_center=pc, rng=np.random.default_rng(SEED))
    log.info("[local_only]  cleared=%.1f%%  healthy=%.2f%%  (primary cleared, metastases survive)",
             100 * loc["cleared"], 100 * loc["healthy_killed"])

    # sweep p for systemic_only (no propagation) and hybrid (propagation)
    sys_curve, hyb_curve, rows = [], [], []
    for p in P_SWEEP:
        s = run(state0, p, propagate=False, seed_primary_only=False, primary_center=pc, rng=np.random.default_rng(SEED))
        h = run(state0, p, propagate=True,  seed_primary_only=False, primary_center=pc, rng=np.random.default_rng(SEED))
        sys_curve.append((p, s["cleared"])); hyb_curve.append((p, h["cleared"]))
        rows.append({"p_seed": p, "systemic_cleared": round(s["cleared"], 3), "hybrid_cleared": round(h["cleared"], 3),
                     "systemic_dose": round(s["dose_frac"], 3), "hybrid_dose": round(h["dose_frac"], 3),
                     "hybrid_healthy": round(h["healthy_killed"], 4)})
        log.info("  p_seed=%.3f | systemic cleared=%.1f%% | hybrid cleared=%.1f%% (dose=%.1f%%, healthy=%.2f%%)",
                 p, 100 * s["cleared"], 100 * h["cleared"], 100 * h["dose_frac"], 100 * h["healthy_killed"])

    p_sys = min_p_for_clearance(sys_curve)
    p_hyb = min_p_for_clearance(hyb_curve)
    log.info("=" * 64)
    log.info("min p_seed for >99%% clearance:  systemic_only=%s  hybrid=%s", p_sys, p_hyb)
    economy = (p_sys / p_hyb) if (p_sys and p_hyb) else None
    if economy:
        log.info("→ hybrid clears disseminated disease at %.0fx LESS dose than pure-systemic (amplification)", economy)

    checks = {
        "local_only leaves metastases (cleared < 90%)": loc["cleared"] < 0.90,
        "hybrid clears everything (>99%) at low dose (p_seed <= 0.1)":
            (p_hyb is not None and p_hyb <= 0.1),
        "systemic_only needs near-saturating dose (p_seed >= 0.5 for >99%)":
            (p_sys is None or p_sys >= 0.5),
        "hybrid spares healthy (<1%)": all(r["hybrid_healthy"] < 0.01 for r in rows),
        "hybrid >> systemic dose economy (>=5x)": (economy is not None and economy >= 5),
    }
    for k, ok in checks.items():
        log.info("  [%s] %s", "✓" if ok else "✗", k)
    supported = all(checks.values())
    log.info("=" * 64)
    if supported:
        log.info("✅ HYBRID SOLVES THE METASTASIS GAP (in-model): low systemic 'scout' seeding reaches every")
        log.info("   focus, gated propagation amplifies locally → disseminated disease cleared at a fraction")
        log.info("   of the dose, healthy spared. Local-only alone leaves metastases; systemic-only alone")
        log.info("   needs to hit ~every cell. Scout+amplify is the regime that clears AND is dose-economical.")
    else:
        log.info("⚠️ not cleanly supported under these params — inspect which check failed.")
    log.info("CAVEAT: abstract ABM, assumed parameters — tests the LOGIC, not real tumour/PK kinetics.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metastasis_hybrid_results.json").write_text(json.dumps(
        {"local_only": loc, "sweep": rows, "min_p_systemic": p_sys, "min_p_hybrid": p_hyb,
         "dose_economy_x": economy, "checks": checks, "supported": supported}, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ps = [p for p, _ in sys_curve]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ps, [100 * c for _, c in sys_curve], "o-", color="#888", label="systemic only (no amplification)")
        ax.plot(ps, [100 * c for _, c in hyb_curve], "o-", color="#c0392b", label="hybrid (scout + gated amplify)")
        ax.axhline(99, ls=":", color="green", label="99% cleared")
        ax.axhline(100 * loc["cleared"], ls="--", color="#3498db", label=f"local only ({100*loc['cleared']:.0f}%)")
        ax.set_xscale("log"); ax.set_xlabel("systemic seed probability p_seed (≈ dose)"); ax.set_ylabel("% tumour cleared")
        ax.set_title("Hybrid scout+amplify clears disseminated disease at a fraction of the dose")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT_DIR / "metastasis_hybrid.png", dpi=110)
        log.info("figure → runs/metastasis_hybrid/metastasis_hybrid.png")
    except Exception as e:
        log.warning("figure skipped (%s: %s)", type(e).__name__, e)
    log.info("results → runs/metastasis_hybrid/metastasis_hybrid_results.json")
    return 0 if supported else 1


if __name__ == "__main__":
    sys.exit(main())
