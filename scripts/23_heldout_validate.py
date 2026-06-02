#!/usr/bin/env python3
"""
RUNG 5 — HELD-OUT VALIDATION: the winner's-curse / look-elsewhere rigor layer, PROVEN ON SYNTHETIC
GROUND TRUTH before it is ever pointed at the real surfaceome (the same discipline scripts/20 and
scripts/22 use: validate the METHOD on known answers first).

When an optimizer sweeps thousands of surface genes x {AND, AND-NOT}, the BEST gate's apparent
selectivity is inflated by multiple comparisons. With enough genes, *some* gate clears the bars by
chance. A method that reports that gate as a "discovery" has reward-hacked itself. This module adds the
three guards the adversarial review demanded and proves each catches a planted failure:

  GUARD A  TARGET-DECOY FAMILY-MAX FDR  (decoy family size == true N).
           Decoys = the SAME gate family scored on a panel whose tumour-vs-(non-vital-normal) labels are
           permuted (exclusivity destroyed; the per-cell vital-safety audit held fixed). A "discovery"
           must beat the (1-alpha) quantile of the DECOY FAMILY-MAX coverage null — not just a per-gate
           threshold. This is the look-elsewhere correction: with N genes you WILL find a good-looking
           gate by chance; the family-max null calibrates how good is good enough.
  GUARD B  DONOR-LABEL-PERMUTATION NULL.
           Shuffle donor labels within normal vital cells and recompute the MAX-over-donor vital leak.
           A gate whose apparent cleanliness depends on the true donor partition (not on real biology)
           is flagged: its observed worst-donor leak should sit inside the permuted null.
  GUARD C  CLUSTER-BOOTSTRAP WINNER'S-CURSE SHRINKAGE.
           Resample whole DONORS (never cells), re-select the winning gate inside each resample, and
           measure it on the OUT-OF-BAG donors. The in-bag-minus-OOB gap is the optimism. Coverage is
           corrected DOWN, leak corrected UP. A gate certified "safe" only by in-sample optimism flips.

HONEST NOTE (a real finding, surfaced not hidden): the RUNG-5 vital-leak estimator is already the
MAX-over-donor Jeffreys UPPER bound, which is intrinsically winner's-curse-resistant for the SAFETY
axis — so vital-leak shrinkage is small by design. The curse bites hardest on the COVERAGE we MAXIMISE
and on the pooled non-vital (strict/regen) leak we MINIMISE; those are where this module's correction
does the work, and where a naive search would over-claim.

CEILING: this validates the STATISTICS, not biology. mRNA != surface protein; a surviving gate is the
best NEXT wet-lab experiment, never a cure or a safety proof. "No gate survives FDR + shrinkage" is a
first-class outcome. Recognition-selectivity is a separate axis, never multiplied with RUNG-1/2/3.

USAGE: python scripts/23_heldout_validate.py     (runs the synthetic validation harness; CPU, ~seconds)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung5_logicgate"

# import the optimizer core (which itself imports scripts/18 as opt.lg)
spec = importlib.util.spec_from_file_location("opt", PROJECT_ROOT / "scripts" / "22_gate_optimizer.py")
opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)
lg = opt.lg

# ---- frozen validation params (would live in manifest_PREREG.yaml for the real run) ----
SEED = 20260601
ALPHA = 0.05          # family-max discovery level AND target per-gate FDR
N_PERM = 120          # decoy permutations (each re-scores the FULL family => family size preserved)
N_BOOT = 150          # cluster-bootstrap resamples over donors
COV_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ============================================================================================
#  gate family + label permutations
# ============================================================================================
def and_pair_family(genes):
    """All 2-literal AND gates over `genes` (the search family; size == C(len,2))."""
    return [{"pos": [a, b], "neg": []} for a, b in combinations(genes, 2)]


def score_family(panel, family, required_vital):
    # VECTORISED scorer (bit-equivalent to per-gate score_gate; validated) — required for real-scale FDR
    # (family x N_PERM over 1M+ cells x 1000s of donors is intractable with the per-gate Python loop).
    return opt.score_gates_vec(panel, family, required_vital)


def permute_exclusivity_preserve_vital(panel, rng):
    """DECOY: destroy tumour-vs-normal EXCLUSIVITY while holding the per-cell vital-safety audit FIXED.
    Vital cells stay normal (and powered); among all NON-vital cells (tumour + non-vital normal) the
    compartment label is shuffled. Under this null no gate is genuinely tumour-exclusive, yet a gate can
    still be wrongly declared SELECTIVE by chance coverage on the random 'tumour' subset — exactly the
    look-elsewhere false positive we are FDR-controlling. Returns a new Panel (same shape)."""
    comp = panel.compartment.copy()
    vital = np.isin(panel.cell_type, list(lg.VITAL_NONREGEN)) & (comp == "normal")
    idx = np.where(~vital)[0]
    comp[idx] = comp[idx][rng.permutation(idx.size)]
    return lg.Panel(panel.counts, panel.genes, panel.cell_type, panel.tissue, comp, donor=panel.donor)


def permute_donor_within_vital(panel, rng):
    """NULL for GUARD B: shuffle donor labels among normal vital cells only (breaks the real per-donor
    partition; expression + compartment + cell_type untouched)."""
    don = panel.donor.copy()
    vital = np.isin(panel.cell_type, list(lg.VITAL_NONREGEN)) & (panel.compartment == "normal")
    idx = np.where(vital)[0]
    don[idx] = don[idx][rng.permutation(idx.size)]
    return lg.Panel(panel.counts, panel.genes, panel.cell_type, panel.tissue, panel.compartment, donor=don)


# ============================================================================================
#  GUARD A — target-decoy family-MAX FDR
# ============================================================================================
def familymax_fdr(panel, family, required_vital, rng, n_perm=N_PERM):
    """Returns the family-max discovery threshold + per-cov-grid empirical FDR.

    true_safe_cov  : coverage of every TRUE gate that is `safe` (verdict SELECTIVE or SAFE-LOW-COVERAGE)
    decoy_max_cov  : per permutation, the MAX coverage among decoy gates that are `safe`  (the family-max
                     null: best apparent safe-coverage achievable by chance over the WHOLE family)
    threshold      : (1-alpha) quantile of decoy_max_cov  -> a TRUE gate is a DISCOVERY iff cov > threshold
    fdr_curve      : at each coverage c, E_perm[# decoy safe gates with cov>=c] / max(1,# true safe gates>=c)
    """
    true_scores = score_family(panel, family, required_vital)
    true_safe = [r for r in true_scores if r["safe"]]
    true_safe_cov = np.array([r["coverage"] for r in true_safe]) if true_safe else np.array([])

    decoy_max_safe_cov = np.zeros(n_perm)          # family-max coverage among SAFE decoys (discovery null)
    decoy_cov_mat = np.zeros((n_perm, len(family)))  # ALL-gate coverage (for the look-elsewhere order stat)
    decoy_safe_cov_all = []                         # pooled SAFE decoy coverages (per-gate FDR curve)
    for p in range(n_perm):
        d = permute_exclusivity_preserve_vital(panel, rng)
        ds_all = score_family(d, family, required_vital)
        decoy_cov_mat[p] = [r["coverage"] for r in ds_all]
        safe_cov = [r["coverage"] for r in ds_all if r["safe"]]
        decoy_max_safe_cov[p] = max(safe_cov) if safe_cov else 0.0
        decoy_safe_cov_all.extend(safe_cov)
    decoy_safe_cov_all = np.array(decoy_safe_cov_all)

    threshold = float(np.quantile(decoy_max_safe_cov, 1.0 - ALPHA))
    discoveries = [r for r in true_safe if r["coverage"] > threshold]

    # look-elsewhere order statistic (why family-max is needed, independent of the safety filter):
    #   per-gate null  = the TYPICAL single gate's 95th-pctile coverage (median over genes of each gene's p95)
    #   family-max null= 95th pctile of the per-permutation MAX coverage over the whole family
    # family-max strictly exceeds per-gate when N>1 with variance -> a per-gate cutoff under-controls.
    pergate_p95_each = np.quantile(decoy_cov_mat, 0.95, axis=0)          # one p95 per gene
    pergate_cov_p95 = float(np.median(pergate_p95_each))
    familymax_cov_p95 = float(np.quantile(decoy_cov_mat.max(axis=1), 0.95))

    fdr_curve = {}
    for c in COV_GRID:
        n_true = int((true_safe_cov >= c).sum())
        exp_decoy = float((decoy_safe_cov_all >= c).sum()) / n_perm   # expected # decoy safe gates >= c
        fdr_curve[float(c)] = round(exp_decoy / max(1, n_true), 3)

    return {
        "n_family": len(family),
        "n_true_safe": len(true_safe),
        "familymax_threshold_cov": round(threshold, 3),
        "n_discoveries": len(discoveries),
        "discoveries": [r["gate"] for r in discoveries],
        "best_true_safe_cov": round(float(true_safe_cov.max()), 3) if true_safe_cov.size else 0.0,
        "decoy_max_safe_cov_mean": round(float(decoy_max_safe_cov.mean()), 3),
        "pergate_cov_p95": round(pergate_cov_p95, 3),
        "familymax_cov_p95": round(familymax_cov_p95, 3),
        "lookelsewhere_inflation": round(familymax_cov_p95 - pergate_cov_p95, 3),
        "fdr_curve": fdr_curve,
    }


# ============================================================================================
#  GUARD B — donor-permutation null on the worst-donor vital leak
# ============================================================================================
def donor_permutation_null(panel, gate, required_vital, rng, n_perm=N_PERM):
    # obs computed by the SAME (unrounded) estimator as the null, so a uniformly-clean gate gives obs==null
    # exactly (p==1.0); comparing a rounded obs to an unrounded null would spuriously flag clean gates.
    fire0 = opt.gate_fire(panel, gate["pos"], gate["neg"])
    obs, _, _ = opt._vital_leak_max_over_donor(panel, fire0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        d = permute_donor_within_vital(panel, rng)
        fire = opt.gate_fire(d, gate["pos"], gate["neg"])
        null[i], _, _ = opt._vital_leak_max_over_donor(d, fire)
    # one-sided p: how often a RANDOM donor partition concentrates leak >= the observed worst donor.
    # high p  -> the worst-donor concentration is NOT special to the real partition (generic sampling);
    # low p   -> the leak lives in a SPECIFIC real donor (a lethal patient a pooled view would erase).
    p = float((null >= obs - 1e-9).mean())
    return {"obs_vital_leak": round(float(obs), 4), "null_mean": round(float(null.mean()), 4),
            "null_p95": round(float(np.quantile(null, 0.95)), 4),
            "p_value": round(p, 3),
            "donor_structure_dependent": p < ALPHA}


# ============================================================================================
#  GUARD C — cluster-bootstrap winner's-curse shrinkage
# ============================================================================================
def _eval_gate(panel, gate, required_vital):
    r = opt.score_gate(panel, gate["pos"], gate["neg"], required_vital)
    return r["coverage"], r["vital_leak"], r["strict_leak"], r["regen_leak"], r["safe"]


def bootstrap_winners_curse(panel, family, required_vital, rng, n_boot=N_BOOT):
    """Cluster-bootstrap over DONORS. In each resample: re-select the winning gate by the lexicographic
    rank_key (the same objective the optimizer uses), then measure that gate on the OUT-OF-BAG donors.
    Optimism = mean(in-bag - OOB). Coverage corrected DOWN, leaks corrected UP. Reports whether the
    full-data winner stays SAFE after the pessimistic (corrected) leaks are applied."""
    donors = np.array(sorted(set(panel.donor)))
    # full-data winner (what a naive run would report)
    full = sorted(score_family(panel, family, required_vital), key=opt.rank_key)
    winner = {"pos": full[0]["pos"], "neg": full[0]["neg"]} if full else None
    if winner is None:
        return None
    full_r = full[0]

    opt_cov, opt_strict, opt_regen, opt_vital = [], [], [], []
    for _ in range(n_boot):
        draw = rng.choice(donors, size=donors.size, replace=True)
        inbag = set(draw.tolist())
        oob = set(donors.tolist()) - inbag
        if not oob:
            continue
        pin = opt.subset(panel, inbag)
        poob = opt.subset(panel, oob)
        # need tumour cells in both partitions to define coverage; skip degenerate draws
        if not (pin.compartment == "tumour").any() or not (poob.compartment == "tumour").any():
            continue
        sel = sorted(score_family(pin, family, required_vital), key=opt.rank_key)
        if not sel:
            continue
        g = {"pos": sel[0]["pos"], "neg": sel[0]["neg"]}
        cin, vin, sin, rin, _ = _eval_gate(pin, g, required_vital)
        cout, vout, sout, rout, _ = _eval_gate(poob, g, required_vital)
        opt_cov.append(cin - cout)         # coverage optimistic HIGH in-bag -> positive optimism
        opt_strict.append(sout - sin)      # leak optimistic LOW in-bag -> OOB higher -> positive optimism
        opt_regen.append(rout - rin)
        opt_vital.append(vout - vin)

    def m(a): return float(np.mean(a)) if a else 0.0
    optimism = {"coverage": m(opt_cov), "strict_leak": m(opt_strict),
                "regen_leak": m(opt_regen), "vital_leak": m(opt_vital)}

    corrected = {
        "coverage": round(max(0.0, full_r["coverage"] - max(0.0, optimism["coverage"])), 3),
        "vital_leak": round(full_r["vital_leak"] + max(0.0, optimism["vital_leak"]), 3),
        "strict_leak": round(full_r["strict_leak"] + max(0.0, optimism["strict_leak"]), 3),
        "regen_leak": round(full_r["regen_leak"] + max(0.0, optimism["regen_leak"]), 3),
    }
    still_safe = (corrected["vital_leak"] <= opt.LEAK_BAR and corrected["strict_leak"] <= opt.LEAK_BAR
                  and corrected["regen_leak"] <= opt.REGEN_BAR)
    still_selective = still_safe and corrected["coverage"] >= opt.COV_BAR
    return {
        "winner_gate": full_r["gate"],
        "naive": {"coverage": full_r["coverage"], "vital_leak": full_r["vital_leak"],
                  "strict_leak": full_r["strict_leak"], "regen_leak": full_r["regen_leak"],
                  "safe": full_r["safe"], "selective": full_r["selective"]},
        "optimism": {k: round(v, 3) for k, v in optimism.items()},
        "corrected": corrected,
        "still_safe_after_shrinkage": bool(still_safe),
        "still_selective_after_shrinkage": bool(still_selective),
    }


# ============================================================================================
#  SYNTHETIC GROUND-TRUTH PANELS  (we know the answer -> we can validate the method)
# ============================================================================================
def _panel_from_blocks(blocks):
    """blocks: list of (cell_type, tissue, compartment, donor, lambdas, n). Returns a Panel."""
    cc, ct, ts, comp, don = [], [], [], [], []
    rng_local = np.random.default_rng(SEED)
    for cell_type, tissue, compartment, donor, lams, n in blocks:
        cc.append(np.column_stack([rng_local.poisson(l, n) for l in lams]))
        ct += [cell_type] * n; ts += [tissue] * n; comp += [compartment] * n; don += [donor] * n
    return np.vstack(cc), np.array(ct), np.array(ts), np.array(comp), np.array(don)


HI, MID, LOWDET, ABSENT = 6.0, 2.2, 0.55, 0.01
NPER, NTUM = 320, 420
D_NORM, D_TUM = 10, 4


def build_signal_panel():
    """One planted CLEAN gate (POS1 AND POS2: co-positive ONLY on tumour, absent on all normal incl
    vital) among M decoy NOISE genes (low everywhere) + a BROAD gene (high on normal too)."""
    M = 6
    genes = ["POS1", "POS2"] + [f"NOISE{i}" for i in range(M)] + ["BROAD"]

    def lam(role):
        v = [ABSENT, ABSENT] + [LOWDET] * M + [MID]              # default normal profile
        if role == "tumour":
            v = [HI, HI] + [LOWDET] * M + [MID]                   # POS1,POS2 co-positive only here
        if role == "vital":
            v = [ABSENT, ABSENT] + [LOWDET] * M + [MID]
        return v

    blocks = []
    for d in range(D_NORM):
        dn = f"ds::n{d}"
        blocks.append(("cardiomyocyte", "heart", "normal", dn, lam("vital"), NPER))
        blocks.append(("hepatocyte", "liver", "normal", dn, lam("normal"), NPER))
        blocks.append(("normal_epithelium", "lung", "normal", dn, lam("normal"), NPER))
    for d in range(D_TUM):
        blocks.append(("tumour_malignant", "tumour", "tumour", f"tds::t{d}", lam("tumour"), NTUM))
    counts, ct, ts, comp, don = _panel_from_blocks(blocks)
    return lg.Panel(counts, genes, ct, ts, comp, donor=don), genes


def build_null_panel():
    """NO gene pair is tumour-exclusive: POS1 high on tumour but POS2 ABSENT on tumour (never co-positive);
    everything else marginal. There is NO true selective gate -> nothing should survive family-max."""
    M = 6
    genes = ["POS1", "POS2"] + [f"NOISE{i}" for i in range(M)] + ["BROAD"]

    def lam(role):
        if role == "tumour":
            return [HI, ABSENT] + [LOWDET] * M + [MID]            # POS1 high, POS2 absent -> no co-positivity
        return [MID, ABSENT] + [LOWDET] * M + [MID]               # POS1 also mid on normal -> not exclusive

    blocks = []
    for d in range(D_NORM):
        dn = f"ds::n{d}"
        blocks.append(("cardiomyocyte", "heart", "normal", dn, lam("vital"), NPER))
        blocks.append(("hepatocyte", "liver", "normal", dn, lam("normal"), NPER))
        blocks.append(("normal_epithelium", "lung", "normal", dn, lam("normal"), NPER))
    for d in range(D_TUM):
        blocks.append(("tumour_malignant", "tumour", "tumour", f"tds::t{d}", lam("tumour"), NTUM))
    counts, ct, ts, comp, don = _panel_from_blocks(blocks)
    return lg.Panel(counts, genes, ct, ts, comp, donor=don), genes


WC_TUM_DONORS = 6
WC_NTUM = 200


def build_winnerscurse_panel():
    """Isolate the winner's curse on COVERAGE (where it genuinely lives — the MAXIMISED, donor-heterogeneous
    quantity), holding safety trivially clean so the only axis in play is coverage. M partner genes, each
    HI on the tumour cells of EXACTLY ONE of 6 tumour donors and ABSENT everywhere else (and absent on all
    normal -> every gate is SAFE). So DRV AND P_i covers ~1/6 ~ 0.16 of the tumour — just above COV_BAR by
    construction. The optimizer MAXIMISES coverage, so it picks whichever single-donor gate sampled
    luckiest; that gate is SELECTIVE in-sample but its coverage lives in donors that fall OUT-OF-BAG under
    cluster-bootstrap -> the corrected coverage drops BELOW COV_BAR and the winner is honestly DEMOTED
    (SELECTIVE -> SAFE-LOW-COVERAGE). This is the donor-held-out discipline the whole project rests on."""
    M = 12
    genes = ["DRV"] + [f"P{i}" for i in range(M)]

    def tumour_lam(donor_idx):
        v = [HI]                                   # DRV: activator, HI on all tumour
        for i in range(M):
            v.append(HI if (i % WC_TUM_DONORS) == donor_idx else ABSENT)  # P_i fires only in its 1 donor
        return v

    clean = [ABSENT] * (M + 1)                     # all normal: DRV + partners absent -> always safe
    blocks = []
    for d in range(D_NORM):
        blocks.append(("cardiomyocyte", "heart", "normal", f"ds::n{d}", clean, NPER))
        blocks.append(("normal_epithelium", "lung", "normal", f"ds::n{d}", clean, NPER))
    for d in range(WC_TUM_DONORS):
        blocks.append(("tumour_malignant", "tumour", "tumour", f"tds::t{d}", tumour_lam(d), WC_NTUM))
    counts, ct, ts, comp, don = _panel_from_blocks(blocks)
    return lg.Panel(counts, genes, ct, ts, comp, donor=don), genes


# ============================================================================================
#  VALIDATION HARNESS
# ============================================================================================
def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def main() -> int:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    req = {"cardiomyocyte"}
    print("=" * 92)
    print("RUNG 5 — HELD-OUT VALIDATION (family-max FDR + donor-permutation null + winner's-curse shrinkage)")
    print("       validated on synthetic ground truth BEFORE any real surfaceome run")
    print("=" * 92)

    # ---------- SIGNAL panel: planted clean gate must SURVIVE FDR ----------
    sig, sgenes = build_signal_panel()
    sig_fam = and_pair_family(sgenes)
    print(f"\n[SIGNAL]  {sig.counts.shape[0]} cells, {len(set(sig.donor))} donors, family={len(sig_fam)} AND-pairs "
          f"(planted clean gate = POS1 AND POS2)")
    sig_fdr = familymax_fdr(sig, sig_fam, req, rng)
    print(f"  family-max threshold cov={sig_fdr['familymax_threshold_cov']:.2f} (p95 of decoy family-max); "
          f"best true safe cov={sig_fdr['best_true_safe_cov']:.2f}")
    print(f"  discoveries (beat family-max): {sig_fdr['discoveries']}")
    sig_planted_survives = any(set(g.split(" AND ")) == {"POS1", "POS2"} for g in sig_fdr["discoveries"])

    # ---------- NULL panel: NOTHING should survive ----------
    nul, ngenes = build_null_panel()
    nul_fam = and_pair_family(ngenes)
    print(f"\n[NULL]    {nul.counts.shape[0]} cells, {len(set(nul.donor))} donors, family={len(nul_fam)} AND-pairs "
          f"(NO truly exclusive gate exists)")
    nul_fdr = familymax_fdr(nul, nul_fam, req, rng)
    print(f"  family-max threshold cov={nul_fdr['familymax_threshold_cov']:.2f}; "
          f"best true safe cov={nul_fdr['best_true_safe_cov']:.2f}")
    print(f"  discoveries (beat family-max): {nul_fdr['discoveries'] or 'NONE'}")
    null_no_discovery = nul_fdr["n_discoveries"] == 0

    # ---------- family-max NECESSITY: look-elsewhere order statistic (independent of the safety filter) ----------
    # the 95th-pctile of the FAMILY-MAX coverage null exceeds a typical single gate's 95th-pctile coverage:
    # a per-gate cutoff calibrated to one gene is anti-conservative across a family -> family-max is required.
    le = nul_fdr["lookelsewhere_inflation"]
    print(f"  look-elsewhere: family-max cov p95={nul_fdr['familymax_cov_p95']:.2f} vs typical per-gate "
          f"p95={nul_fdr['pergate_cov_p95']:.2f}  -> inflation {le:+.2f} "
          f"({'CONFIRMED' if le > 0.02 else 'absent'}; family-max correction is {'needed' if le > 0.02 else 'moot'})")
    familymax_needed = le > 0.02

    # ---------- GUARD B: donor-permutation null — BOTH error directions ----------
    # (i) the planted CLEAN gate is uniformly ~0 on vital -> NOT donor-structure-dependent (must be p~1).
    dperm = donor_permutation_null(sig, {"pos": ["POS1", "POS2"], "neg": []}, req, rng)
    # (ii) a donor-CONCENTRATED lethal gate (scripts/22 ATTACK-3: 8% in ONE donor's heart) MUST be flagged.
    atk_panel = opt._build_attack_panel(np.random.default_rng(20260530))
    dperm_atk = donor_permutation_null(atk_panel, {"pos": ["ACT", "ATK3"], "neg": []}, {"cardiomyocyte"}, rng)
    print(f"\n[GUARD B] clean gate : obs vital_leak={dperm['obs_vital_leak']:.4f}, null mean={dperm['null_mean']:.4f}, "
          f"p={dperm['p_value']:.2f} -> donor-dependent={dperm['donor_structure_dependent']}")
    print(f"[GUARD B] ATK3 gate  : obs vital_leak={dperm_atk['obs_vital_leak']:.4f}, null mean={dperm_atk['null_mean']:.4f}, "
          f"p={dperm_atk['p_value']:.2f} -> donor-dependent={dperm_atk['donor_structure_dependent']} (lethal donor concentration)")
    donor_null_ok = (not dperm["donor_structure_dependent"]) and dperm_atk["donor_structure_dependent"]

    # ---------- GUARD C: winner's-curse shrinkage flips an over-claimed winner ----------
    wc, wgenes = build_winnerscurse_panel()
    wc_fam = and_pair_family(wgenes)
    print(f"\n[GUARD C] winner's-curse panel: {wc.counts.shape[0]} cells, {len(set(wc.donor))} donors, "
          f"family={len(wc_fam)} AND-pairs (each partner covers ONE tumour donor ~1/6~0.16, clean on normal)")
    boot = bootstrap_winners_curse(wc, wc_fam, req, rng)
    if boot:
        print(f"  naive winner   : {boot['winner_gate']:24s} cov={boot['naive']['coverage']:.2f} "
              f"vital={boot['naive']['vital_leak']:.3f} selective={boot['naive']['selective']} "
              f"(coverage lives in ONE donor)")
        print(f"  optimism (in-bag - OOB): cov={boot['optimism']['coverage']:+.3f}  "
              f"(that donor falls out-of-bag -> held-out coverage collapses)")
        print(f"  corrected cov  : {boot['corrected']['coverage']:.2f}  "
              f"(COV_BAR={opt.COV_BAR}) -> still_selective={boot['still_selective_after_shrinkage']}")
    coverage_optimism_positive = bool(boot and boot["optimism"]["coverage"] > 0)
    shrinkage_demotes = bool(boot and boot["naive"]["selective"] and not boot["still_selective_after_shrinkage"])

    # ---------- assertions / checks ----------
    decoy_family_size_eq = (sig_fdr["n_family"] == len(sig_fam)) and (nul_fdr["n_family"] == len(nul_fam))

    # audit stats/F1+F3 FIX: the decoy null must be NON-DEGENERATE (it was identically 0 before — decoys were
    # impossible-to-pass because the permutation relabelled real tumour cells 'normal'; score_gate now excludes
    # malignant cell types from the normal-leak audit, so decoys CAN be safe-with-coverage). Two checks prove
    # the FDR now does real rejection work (not vacuous): the threshold is positive, and it REJECTS some safe
    # gates (sub-threshold coverage gates that the old threshold=0 wrongly admitted).
    decoy_nondegenerate = sig_fdr["familymax_threshold_cov"] > 0.05
    fdr_rejects_some = sig_fdr["n_discoveries"] < sig_fdr["n_true_safe"]   # not everything passes (vs old threshold=0)
    noise_rejected = not any("NOISE" in g for g in sig_fdr["discoveries"])  # low-cov chance pairs excluded

    checks = {
        "SIGNAL: planted clean gate (POS1 AND POS2) SURVIVES family-max FDR": sig_planted_survives,
        "SIGNAL: decoy null is NON-DEGENERATE (family-max threshold > 0.05, not the old vacuous 0)": decoy_nondegenerate,
        "SIGNAL: family-max FDR REJECTS sub-threshold safe gates (discoveries < safe; noise pairs excluded)": fdr_rejects_some and noise_rejected,
        "NULL: NO gate survives family-max (no false discovery)": null_no_discovery,
        "family-max NECESSARY: family-max cov-null p95 > per-gate p95 (look-elsewhere)": familymax_needed,
        "decoy family size == true family size N": decoy_family_size_eq,
        "GUARD B: clean gate NOT flagged AND donor-concentrated lethal gate IS flagged": donor_null_ok,
        "GUARD C: coverage optimism (in-bag - OOB) > 0 (winner's curse exists)": coverage_optimism_positive,
        "GUARD C: shrinkage DEMOTES an over-claimed winner (selective -> not)": shrinkage_demotes,
    }
    print("\n" + "=" * 92)
    print("VALIDATION CHECKS:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    ok = all(checks.values())
    print("=" * 92)

    sha = _git_sha()

    def _jd(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        return str(o)

    (OUT_DIR / "heldout_validation.json").write_text(json.dumps({
        "frozen_git_sha": sha, "seed": SEED, "alpha": ALPHA, "n_perm": N_PERM, "n_boot": N_BOOT,
        "signal_fdr": sig_fdr, "null_fdr": nul_fdr,
        "donor_permutation_null_clean": dperm, "donor_permutation_null_attack": dperm_atk,
        "winners_curse_bootstrap": boot,
        "checks": checks, "validated": ok,
        "HARD_RULE": "ranking is lexicographic (no-multiply); decoy family size == true N; the leak we TRUST "
                     "is the winner's-curse-corrected (pessimistic) one, never the optimistic in-sample one.",
        "HONEST_NOTE": "the vital-leak estimator is already a max-over-donor Jeffreys UPPER bound, intrinsically "
                       "winner's-curse-resistant; the curse bites the MAXIMISED coverage and the pooled "
                       "non-vital leak, which is where the bootstrap correction does its work.",
        "CEILING": "validates the STATISTICS on synthetic ground truth, not biology. mRNA!=surface protein; a "
                   "surviving gate is the best NEXT wet-lab experiment, not a cure or a safety proof. 'No gate "
                   "survives FDR + shrinkage' is a first-class outcome. Real surfaceome run = Colab (scripts/17).",
    }, indent=2, default=_jd))
    print("results -> runs/rung5_logicgate/heldout_validation.json")

    _figure(sig_fdr, nul_fdr, boot, ok)
    print("=" * 92)
    print("CEILING: this is the look-elsewhere / winner's-curse rigor LAYER, proven on known answers. It does")
    print("NOT find a gate; it makes the eventual real-data verdict honest. mRNA!=protein; recognition is a")
    print("separate axis, never multiplied with RUNG-1/2/3.")
    return 0 if ok else 1


def _figure(sig_fdr, nul_fdr, boot, ok):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        # 1: FDR curves
        cs = sorted(sig_fdr["fdr_curve"].keys())
        ax[0, 0].plot(cs, [sig_fdr["fdr_curve"][c] for c in cs], "-o", ms=3, color="#27ae60", label="SIGNAL panel")
        ax[0, 0].plot(cs, [nul_fdr["fdr_curve"][c] for c in cs], "-o", ms=3, color="#c0392b", label="NULL panel")
        ax[0, 0].axhline(ALPHA, ls="--", color="grey", lw=0.8, label=f"alpha={ALPHA}")
        ax[0, 0].set_xlabel("tumour coverage threshold c"); ax[0, 0].set_ylabel("empirical FDR")
        ax[0, 0].set_title("target-decoy empirical FDR vs coverage"); ax[0, 0].legend(fontsize=8)
        # 2: family-max threshold vs best true
        ax[0, 1].bar(["SIGNAL\nbest true", "SIGNAL\nfamily-max\nthreshold", "NULL\nbest true", "NULL\nfamily-max\nthreshold"],
                     [sig_fdr["best_true_safe_cov"], sig_fdr["familymax_threshold_cov"],
                      nul_fdr["best_true_safe_cov"], nul_fdr["familymax_threshold_cov"]],
                     color=["#27ae60", "#7f8c8d", "#c0392b", "#7f8c8d"])
        ax[0, 1].set_ylabel("tumour coverage")
        ax[0, 1].set_title("discovery = best-true ABOVE family-max null\n(SIGNAL clears; NULL does not)")
        # 3: look-elsewhere — family-max coverage null exceeds a typical single-gate null
        ax[1, 0].bar(["typical per-gate\ncov p95 (NULL)", "family-max\ncov p95 (NULL)"],
                     [nul_fdr["pergate_cov_p95"], nul_fdr["familymax_cov_p95"]],
                     color=["#2980b9", "#e67e22"])
        ax[1, 0].set_ylabel("decoy coverage 95th pctile")
        ax[1, 0].set_title(f"look-elsewhere effect (+{nul_fdr['lookelsewhere_inflation']:.2f}):\n"
                           "best-of-family beats any single gene\n-> a per-gate cutoff under-controls")
        # 4: winner's-curse shrinkage (coverage)
        ax[1, 1].axis("off")
        if boot:
            t = (f"WINNER'S-CURSE SHRINKAGE (coverage)\n\nwinner: {boot['winner_gate']}\n"
                 "coverage lives in ONE tumour donor\n\n"
                 f"{'':16s}{'naive':>8s}{'corrected':>11s}\n"
                 f"tumour coverage {boot['naive']['coverage']:6.2f}{boot['corrected']['coverage']:11.2f}\n"
                 f"vital leak      {boot['naive']['vital_leak']:6.3f}{boot['corrected']['vital_leak']:11.3f}\n\n"
                 f"COV_BAR = {opt.COV_BAR}\n"
                 f"naive selective:        {boot['naive']['selective']}\n"
                 f"selective after\nshrinkage:              {boot['still_selective_after_shrinkage']}\n\n"
                 "the held-out donor erases the luck;\nthe coverage we TRUST is the corrected one.")
            ax[1, 1].text(0.02, 0.98, t, va="top", ha="left", fontsize=9.5, family="monospace")
        fig.suptitle(f"RUNG 5 held-out validation: family-max FDR + winner's-curse shrinkage  "
                     f"[{'VALIDATED' if ok else 'FAILED'}]", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT_DIR / "rung5_heldout_validation.png", dpi=110)
        print("figure -> runs/rung5_logicgate/rung5_heldout_validation.png")
    except Exception as e:
        print(f"figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
