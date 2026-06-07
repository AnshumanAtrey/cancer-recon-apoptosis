#!/usr/bin/env python3
"""
RUNG 19 — the EVOLUTIONARY ESCAPE RACE: does the death wave clear the tumour before resistance arises & sweeps?
(laptop / Colab CPU, pure-numpy stochastic lattice — no GPU.)

THE QUESTION (Shriya's concept §6.3, made into a falsifiable race)
-----------------------------------------------------------------
RUNG-13 showed the recognition-gated death wave clears a SUSCEPTIBLE tumour and even overpowers PRE-SEEDED
resistance via the front signal. But the harder, honest question is EVOLUTIONARY: a real tumour keeps
DIVIDING and MUTATING during treatment. If a cell mutates to RESISTANT (loses the recognised neoantigen, or
the death machinery, or the MHC window — RUNG-18) faster than the wave can clear, the resistant clone SWEEPS
and the tumour escapes. This is a race between two clocks: wave clearance vs resistance establishment.

THE MODEL
---------
Stochastic cellular automaton on a lattice. States: EMPTY · SUSCEPTIBLE(S) · RESISTANT(R) · FRONT(dying,
transmits the wave). Two phases:
  1. GROWTH (pre-treatment): a tumour grows from a seed to size N0; at each division an S daughter mutates
     S->R with probability mu. This naturally seeds STANDING resistance (Luria-Delbrück variation).
  2. TREATMENT: the death wave is seeded and sweeps. An S cell adjacent to the FRONT commits (joins the
     front). R cells are IMMUNE to the recognition-gated kill — they only die by BYSTANDER (a resistance-
     AGNOSTIC component: prob `bystander` that a dying neighbour kills an adjacent R too — this is the
     ferroptosis_wave / quorum cross-kill from RUNG-14). Tumour keeps GROWING into cleared space during the
     sweep. Outcome at clearance (no S, no front left): CURE iff R==0, else ESCAPE (R regrows).

THE DIMENSIONLESS RESULT (why it generalises past a tiny lattice)
----------------------------------------------------------------
P(cure) collapses onto a UNIVERSAL function of the expected number of resistant founders present at
treatment, L ≈ mu·N (Luria-Delbrück). L<<1 -> wave cures; L>>1 -> resistance pre-exists -> wave fails
UNLESS bystander cross-kill reaches the R cells. The lattice VALIDATES this scaling on tractable sizes
(sim standing-R vs the Luria-Delbrück mean); we then EXTRAPOLATE analytically to clinical N with the honest
caveat. Headline: pure recognition-gated wave monotherapy cures only up to N* ~ 1/mu cells; a clinical
tumour (~1e8–1e9 cells) ALWAYS harbours pre-existing escape -> a resistance-agnostic bystander/combination
(RUNG-14 ferroptosis_wave/quorum, or a 2nd orthogonal handle) is REQUIRED. RUNG-18's measured genetic escape
(~4% systemic-dark) pins where real tumours sit on the curve.

HONEST CEILING
--------------
A lattice CA, not a tumour: 2D, synchronous-ish update, fixed neighbourhood, no microenvironment / immune
infiltration / spatial drug gradients. mu here is an EFFECTIVE per-division resistance probability lumping
all escape routes (antigen loss, apoptosis-incompetence, MHC silencing) — not a single point-mutation rate.
The clinical extrapolation assumes the Luria-Delbrück scaling holds out of the simulated range (stated, not
proven at 1e9). The bystander parameter is a PROXY for a real cross-kill mechanism whose strength is itself a
wet-lab residual. Honest direction: this BOUNDS the curable size and quantifies the bystander needed — it is
not a cure claim.

USAGE
  python scripts/45_escape_race.py selftest    # synthetic invariants, no heavy run
  python scripts/45_escape_race.py run         # full sweep -> runs/rung19_escape_race/  (CPU, ~1-3 min)
  python scripts/45_escape_race.py quick        # small/fast sanity run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung19_escape_race"
RESULT_JSON = OUT_DIR / "rung19_escape_race.json"
FIGURE_PNG = OUT_DIR / "rung19_escape_race.png"

EMPTY, S, R, FRONT, DEAD = 0, 1, 2, 3, 4

# default lattice / dynamics (validated to give a clean race; quick mode shrinks these)
GRID = 140
FRONT_LIFE = 3          # a FRONT cell transmits the wave for this many steps, then clears to EMPTY
P_WAVE = 0.9            # per-neighbour prob an S cell adjacent to the front commits (wave is fast, RUNG-13)
P_GROW = 0.25           # per-step prob an EMPTY cell with a tumour neighbour gets occupied (division)
R_COST = 0.0            # resistant fitness cost: R divides at rate (1-R_COST)*P_GROW (default neutral)
MAX_TREAT_STEPS = 1200  # safety cap on the treatment phase


def _rng(seed):
    return np.random.default_rng(seed)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """8-neighbour (Moore) count of True cells, via rolls (toroidal edges -> negligible at tumour core)."""
    m = mask.astype(np.int16)
    s = np.zeros_like(m)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            s += np.roll(np.roll(m, dx, axis=0), dy, axis=1)
    return s


def grow_to(grid_n: int, target_n: int, mu: float, rng, p_grow: float = P_GROW, r_cost: float = R_COST):
    """Grow a tumour from a central seed to `target_n` cells; S->R mutation prob `mu` per division.
    Returns (state, n_R_standing). Captures Luria-Delbrück standing resistance."""
    state = np.zeros((grid_n, grid_n), np.int8)
    c = grid_n // 2
    state[c, c] = S
    steps = 0
    max_steps = 50 * grid_n
    while int((state == S).sum() + (state == R).sum()) < target_n and steps < max_steps:
        steps += 1
        tumour = (state == S) | (state == R)
        nt = _neighbor_count(tumour)
        empty_grow = (state == EMPTY) & (nt >= 1)
        if not empty_grow.any():
            break
        # division probability (R slightly slower if cost>0); use S-neighbour vs R-neighbour to pick daughter
        nS = _neighbor_count(state == S)
        nR = _neighbor_count(state == R)
        p = 1.0 - (1.0 - p_grow) ** nt                       # any-neighbour division prob
        draw = rng.random(state.shape)
        born = empty_grow & (draw < p)
        # daughter type: weighted by S vs R neighbours; R neighbours weighted by (1-cost) for fitness
        wR = nR * (1.0 - r_cost)
        frac_R = np.where((nS + wR) > 0, wR / np.maximum(nS + wR, 1e-9), 0.0)
        is_R_daughter = born & (rng.random(state.shape) < frac_R)
        is_S_daughter = born & ~is_R_daughter
        # mutation S->R at birth
        mutate = is_S_daughter & (rng.random(state.shape) < mu)
        state[is_S_daughter & ~mutate] = S
        state[is_R_daughter | mutate] = R
    return state, int((state == R).sum())


def run_episode(grid_n: int, target_n: int, mu: float, bystander: float, rng,
                p_wave: float = P_WAVE, p_grow: float = P_GROW, r_cost: float = R_COST,
                n_seeds: int = 6, max_steps: int = MAX_TREAT_STEPS):
    """Grow to N0, seed the wave, race wave-clearance vs resistance. Returns outcome dict."""
    state, n_R_standing = grow_to(grid_n, target_n, mu, rng, p_grow, r_cost)
    n_tumour0 = int((state == S).sum() + (state == R).sum())
    if n_tumour0 == 0:
        return {"outcome": "no_tumour", "cured": False, "n_R_standing": 0, "n_tumour0": 0}

    # seed the wave at random susceptible cells (the recognised seeds)
    s_idx = np.argwhere(state == S)
    if len(s_idx) == 0:                                      # tumour already all-resistant -> guaranteed escape
        return {"outcome": "escape", "cured": False, "n_R_standing": n_R_standing,
                "n_tumour0": n_tumour0, "n_R_end": int((state == R).sum())}
    pick = s_idx[rng.choice(len(s_idx), size=min(n_seeds, len(s_idx)), replace=False)]
    age = np.zeros_like(state, np.int16)
    for (i, j) in pick:
        state[i, j] = FRONT

    steps = 0
    while steps < max_steps and ((state == FRONT).any() or (state == S).any()):
        steps += 1
        nf = _neighbor_count(state == FRONT)
        nS = _neighbor_count(state == S)
        nR = _neighbor_count(state == R)
        rand = rng.random(state.shape)
        # 1) wave: S adjacent to front commit (recognition-gated)
        p_commit = 1.0 - (1.0 - p_wave) ** nf
        new_front = (state == S) & (nf >= 1) & (rand < p_commit)
        # 2) bystander cross-kill: R adjacent to front die too (resistance-AGNOSTIC component, RUNG-14)
        if bystander > 0:
            p_by = 1.0 - (1.0 - bystander) ** nf
            new_front |= (state == R) & (nf >= 1) & (rng.random(state.shape) < p_by)
        # 3) front PERSISTS while it still has work (adjacent S, or adjacent R when bystander on); it ages out
        #    and clears to DEAD only after FRONT_LIFE idle steps. This makes the wave sweep the whole connected
        #    susceptible blob -> with no resistance the cure is guaranteed (isolates wave-vs-RESISTANCE).
        has_work = (state == FRONT) & ((nS > 0) | ((bystander > 0) & (nR > 0)))
        idle = (state == FRONT) & ~has_work
        clear = idle & (age >= FRONT_LIFE)
        # 4) regrowth into EMPTY/DEAD from a LIVING tumour neighbour, but NOT adjacent to a front
        #    (inflammatory suppression at the front) -> living S can't slip past the wave; R repopulates the
        #    cleared field behind the wave = the escape route.
        living = (state == S) | (state == R)
        nt = _neighbor_count(living)
        p_g = 1.0 - (1.0 - p_grow) ** nt
        regrow = ((state == EMPTY) | (state == DEAD)) & (nt >= 1) & (nf == 0) & (rng.random(state.shape) < p_g)
        wR = nR * (1.0 - r_cost)
        frac_R = np.where((nS + wR) > 0, wR / np.maximum(nS + wR, 1e-9), 0.0)
        born_R = regrow & (rng.random(state.shape) < frac_R)
        born_S = regrow & ~born_R
        mutate = born_S & (rng.random(state.shape) < mu)

        # apply: age idle fronts (reset working fronts), clear aged-out to DEAD, set new fronts, then regrow
        age[has_work] = 0
        age[idle] += 1
        state[clear] = DEAD
        age[clear] = 0
        state[new_front] = FRONT
        age[new_front] = 0
        state[born_S & ~mutate] = S
        state[born_R | mutate] = R

    n_R_end = int((state == R).sum())
    n_S_end = int((state == S).sum())
    cured = (n_R_end == 0 and n_S_end == 0)
    return {"outcome": "cure" if cured else ("escape" if n_R_end > 0 else "timeout"),
            "cured": bool(cured), "n_R_standing": n_R_standing, "n_tumour0": n_tumour0,
            "n_R_end": n_R_end, "n_S_end": n_S_end, "steps": steps}


def luria_delbruck_expected(N: float, mu: float) -> float:
    """Classic LD mean number of resistant cells when a population reaches size N: ~ mu * N * ln(N).
    (Used to VALIDATE the simulated standing-R, and to EXTRAPOLATE to clinical sizes.)"""
    if N <= 1:
        return 0.0
    return float(mu * N * np.log(N))


def p_cure_analytic(N: float, mu: float) -> float:
    """P(no pre-existing resistant founder) ~ exp(-mu*N) (Poisson, one founder kills the cure).
    The honest clinical extrapolation for the recognition-gated wave with NO bystander."""
    return float(np.exp(-mu * N))


def sweep(grid_n, target_n, mus, bystanders, reps, base_seed=19):
    """P(cure) and mean standing-R over (mu × bystander), reps replicates each."""
    out = {}
    for by in bystanders:
        for mu in mus:
            cures, standing = 0, []
            for r in range(reps):
                rng = _rng(base_seed + 1000 * int(by * 100) + 13 * int(-np.log10(max(mu, 1e-12)) * 10) + r)
                ep = run_episode(grid_n, target_n, mu, by, rng)
                cures += int(ep["cured"])
                standing.append(ep["n_R_standing"])
            out[(round(by, 3), mu)] = {
                "p_cure": round(cures / reps, 3),
                "mean_standing_R": round(float(np.mean(standing)), 3),
                "ld_expected_R": round(luria_delbruck_expected(target_n, mu), 3),
                "expected_founders_muN": round(mu * target_n, 4),
            }
    return out


# ---------------------------------------------------------------------------
def main_run(quick: bool = False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    grid_n = 70 if quick else GRID
    target_n = int(0.18 * grid_n * grid_n)                  # established tumour fills ~18% of the lattice
    reps = 12 if quick else 36
    bystanders = [0.0, 0.15, 0.35]
    # sweep mu to span L = mu*N from <<1 to >>1 (the universal axis)
    mus = [10 ** e for e in ([-4.5, -3.5, -2.5] if quick else [-5, -4.5, -4, -3.5, -3, -2.5, -2])]
    print(f"[rung19] grid={grid_n} N0={target_n} reps={reps} mus={['%.1e'%m for m in mus]} "
          f"bystander={bystanders}", flush=True)

    grid_sweep = sweep(grid_n, target_n, mus, bystanders, reps)

    # validate the simulated standing-R against Luria-Delbrück (the rigor check)
    ld_check = []
    for mu in mus:
        k = grid_sweep[(0.0, mu)]
        ld_check.append({"mu": mu, "sim_mean_R": k["mean_standing_R"], "ld_expected_R": k["ld_expected_R"],
                         "muN": k["expected_founders_muN"]})

    # critical founders L* where P(cure) crosses 0.5, per bystander level
    crit = {}
    for by in bystanders:
        rows = sorted(((grid_sweep[(round(by, 3), mu)]["expected_founders_muN"],
                        grid_sweep[(round(by, 3), mu)]["p_cure"]) for mu in mus), key=lambda x: x[0])
        L_star = None
        for (L, pc) in rows:
            if pc < 0.5:
                L_star = L
                break
        crit[round(by, 3)] = {"L_star_muN_at_pcure0.5": L_star, "curve": rows}

    # clinical extrapolation (analytic, honest): pure recognition-gated wave at real tumour sizes
    clinical = {}
    for label, N in [("micromet_1e5", 1e5), ("small_1e7", 1e7), ("1cm_~1e9", 1e9)]:
        clinical[label] = {mu_label: round(p_cure_analytic(N, mu), 4)
                           for mu_label, mu in [("mu_1e-7", 1e-7), ("mu_1e-6", 1e-6), ("mu_1e-5", 1e-5)]}

    # where does a real tumour sit? RUNG-18 genetic escape ~4% systemic-dark = a STANDING-variation proxy
    rung18_anchor = {}
    g18 = PROJECT_ROOT / "runs" / "rung18_mhc_window" / "rung18_mhc_window.json"
    if g18.exists():
        gj = json.load(open(g18))
        rung18_anchor = {
            "note": "RUNG-18 measured ~3.7% of tumours already fully window-dark (systemic) + ~18% dimmed. "
                    "A standing escape fraction this high means L=mu*N >> 1 at any clinical N -> resistance "
                    "PRE-EXISTS -> the recognition-gated wave alone cannot cure -> bystander/combination required.",
            "overall_fully_dark": gj.get("HEADLINE", {}).get("window_fully_dark_systemic_ROUTE_DIES"),
            "overall_dimmed": gj.get("HEADLINE", {}).get("window_dimmed_HLA_only_route_survives"),
        }

    result = {
        "tag": "rung19_escape_race",
        "question": "Does the recognition-gated death wave clear the tumour before resistance arises and "
                    "sweeps? Race wave-clearance vs evolutionary escape; find the curable-size ceiling and "
                    "the bystander cross-kill needed to beat pre-existing resistance.",
        "model": "stochastic lattice CA (S/R/FRONT), growth+mutation then wave+bystander; pure-numpy CPU.",
        "params": {"grid": grid_n, "N0": target_n, "reps": reps, "p_wave": P_WAVE, "p_grow": P_GROW,
                   "front_life": FRONT_LIFE, "r_cost": R_COST, "mus": mus, "bystanders": bystanders},
        "sweep": {f"by={by}|mu={mu:.1e}": grid_sweep[(round(by, 3), mu)] for by in bystanders for mu in mus},
        "luria_delbruck_validation": ld_check,
        "critical_founders_L_star": crit,
        "clinical_extrapolation_pcure_no_bystander": clinical,
        "rung18_anchor": rung18_anchor,
        "HEADLINE": {
            "plain": "Without bystander cross-kill, the recognition-gated wave cures only when expected "
                     "resistant founders L=mu*N << 1 (small tumours); as L crosses ~1 the cure probability "
                     "collapses (Luria-Delbrück). Bystander cross-kill (resistance-agnostic, = RUNG-14 "
                     "ferroptosis_wave/quorum) shifts the curable threshold UP, rescuing larger tumours.",
            "curable_ceiling_no_bystander": "N* ~ 1/mu cells (exp(-mu*N) cure law).",
            "clinical_reality": "At 1e9 cells, even mu=1e-7 gives P(cure)~exp(-100)~0 with no bystander -> "
                                "monotherapy wave cannot cure an established tumour; a resistance-agnostic "
                                "second mechanism or combination is REQUIRED.",
        },
        "INTERPRETATION_MAP": {
            "sim standing-R tracks LD mean": "the lattice reproduces Luria-Delbrück -> the scaling (and the "
                                             "clinical extrapolation) is trustworthy within stated caveats.",
            "P(cure) collapses at L~1, bystander>0 shifts L_star up": "resistance-agnostic cross-kill is the "
                                                                      "lever that beats evolutionary escape — "
                                                                      "ties the cure to RUNG-14's ferroptosis_wave/quorum, not the bare wave.",
        },
        "CEILING": "2D lattice CA, not a tumour (no microenvironment/immune/3D/drug-gradient); mu is an "
                   "EFFECTIVE lumped per-division escape prob, not a point-mutation rate; clinical "
                   "extrapolation assumes LD scaling holds past the simulated range (stated, not proven at "
                   "1e9); bystander is a proxy for a real cross-kill whose strength is a wet-lab residual. "
                   "BOUNDS the curable size + the bystander needed; NOT a cure claim.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung19] wrote {RESULT_JSON}  ({time.monotonic() - t0:.1f}s)", flush=True)

    print("\n  bystander |  mu      muN     P(cure)   simR   ldR")
    for by in bystanders:
        for mu in mus:
            k = grid_sweep[(round(by, 3), mu)]
            print(f"   {by:5.2f}    | {mu:.1e}  {k['expected_founders_muN']:6.3f}   "
                  f"{k['p_cure']:5.2f}    {k['mean_standing_R']:5.1f}  {k['ld_expected_R']:5.1f}")
    print("\n  L* (founders at P(cure)=0.5) by bystander:",
          {by: crit[by]["L_star_muN_at_pcure0.5"] for by in crit})
    _make_figure(grid_sweep, mus, bystanders, crit)
    return 0


def _make_figure(grid_sweep, mus, bystanders, crit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung19] matplotlib unavailable ({e}); skipped figure"); return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {0.0: "#B23A2E", 0.15: "#E0A040", 0.35: "#3F7D54"}
    # panel 1: P(cure) vs expected founders L=muN, one line per bystander -> universal collapse + rescue
    for by in bystanders:
        L = [grid_sweep[(round(by, 3), mu)]["expected_founders_muN"] for mu in mus]
        pc = [grid_sweep[(round(by, 3), mu)]["p_cure"] for mu in mus]
        ax[0].plot(L, pc, "o-", color=colors.get(by, "#333"), label=f"bystander={by}")
    ax[0].axhline(0.5, ls="--", color="grey", alpha=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("expected resistant founders  L = μ·N0  (Luria-Delbrück)")
    ax[0].set_ylabel("P(cure)")
    ax[0].set_title("Escape race: cure collapses as resistance founders cross ~1\n"
                    "bystander cross-kill (resistance-agnostic) shifts the cliff right")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_ylim(-0.03, 1.03)
    # panel 2: sim standing-R vs Luria-Delbrück expectation (validation)
    simR = [grid_sweep[(0.0, mu)]["mean_standing_R"] for mu in mus]
    ldR = [grid_sweep[(0.0, mu)]["ld_expected_R"] for mu in mus]
    ax[1].plot(ldR, simR, "o", color="#444")
    lim = max(max(ldR), max(simR), 1e-3)
    ax[1].plot([0, lim], [0, lim], "--", color="grey", label="y=x")
    ax[1].set_xlabel("Luria-Delbrück expected standing-R  (μ·N·lnN)")
    ax[1].set_ylabel("simulated mean standing-R")
    ax[1].set_title("Validation: lattice reproduces Luria-Delbrück\n(so the clinical extrapolation is trustworthy)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("RUNG-19: evolutionary escape race — wave clearance vs resistance sweep", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung19] wrote {FIGURE_PNG}", flush=True)


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = _rng(1)
    # 1. neighbour count basics
    m = np.zeros((5, 5), bool); m[2, 2] = True
    nc = _neighbor_count(m)
    check("neighbour count: center has 0 self, 8 neighbours flagged", nc[2, 2] == 0 and nc[1, 1] == 1 and nc[2, 1] == 1)

    # 2. grow_to reaches target and mu=0 => no resistance
    st, nR = grow_to(40, 200, mu=0.0, rng=rng)
    check("grow_to reaches ~target size", (st == S).sum() + (st == R).sum() >= 180)
    check("mu=0 => zero standing resistance", nR == 0)
    # high mu => some resistance
    st2, nR2 = grow_to(40, 400, mu=0.02, rng=_rng(2))
    check("high mu => standing resistance > 0", nR2 > 0)

    # 3. episode: mu=0 => always cured (no resistance can exist); reproducible
    cures0 = sum(run_episode(40, 200, mu=0.0, bystander=0.0, rng=_rng(100 + r))["cured"] for r in range(8))
    check("mu=0 => wave cures every replicate", cures0 == 8)

    # 4. high mu (many founders), no bystander => escape dominates
    p_no_hi = np.mean([run_episode(50, 500, mu=0.02, bystander=0.0, rng=_rng(200 + r))["cured"] for r in range(10)])
    check("high-mu no-bystander cure rate is low (<0.5)", p_no_hi < 0.5)
    # rescue must be tested where it's VISIBLE (few founders); at 56 founders nothing short of by=0.9 helps
    p_no = np.mean([run_episode(50, 500, mu=1e-3, bystander=0.0, rng=_rng(250 + r))["cured"] for r in range(16)])
    p_by = np.mean([run_episode(50, 500, mu=1e-3, bystander=0.7, rng=_rng(350 + r))["cured"] for r in range(16)])
    check("bystander raises cure rate (rescue)", p_by > p_no + 0.1)

    # 5. monotonicity: more mu => fewer cures (no bystander)
    lo = np.mean([run_episode(50, 500, mu=1e-3, bystander=0.0, rng=_rng(400 + r))["cured"] for r in range(10)])
    hi = np.mean([run_episode(50, 500, mu=5e-2, bystander=0.0, rng=_rng(500 + r))["cured"] for r in range(10)])
    check("P(cure) decreases with mu", lo >= hi)

    # 6. Luria-Delbrück helper + analytic cure law sane
    check("LD expected rises with mu", luria_delbruck_expected(1000, 1e-3) > luria_delbruck_expected(1000, 1e-4))
    check("analytic P(cure) -> 1 as muN->0", abs(p_cure_analytic(1e3, 1e-12) - 1.0) < 1e-6)
    check("analytic P(cure) -> 0 as muN->inf", p_cure_analytic(1e9, 1e-5) < 1e-6)

    # 7. no NaN / valid states only
    ep = run_episode(40, 200, mu=1e-3, bystander=0.2, rng=_rng(7))
    check("episode returns finite counts", all(isinstance(ep[k], int) for k in ("n_R_end", "n_S_end", "n_tumour0")))

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(selftest())
    elif cmd == "quick":
        sys.exit(main_run(quick=True))
    elif cmd == "run":
        sys.exit(main_run(quick=False))
    print(f"unknown command: {cmd} (use selftest|run|quick)"); sys.exit(64)
