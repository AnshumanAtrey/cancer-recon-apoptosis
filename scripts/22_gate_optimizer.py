#!/usr/bin/env python3
"""
RUNG 5 — full-surfaceome logic-gate OPTIMIZER, with the anti-reward-hacking guards the adversarial review
demanded, PROVEN ON SYNTHETIC PLANTED ATTACKS before any real Census run (the spec's hard prerequisite).

An optimizer over thousands of surface genes x {AND, AND-NOT} logic is the single biggest reward-hacking
risk in this project: it will happily report a confident-but-FALSE "safe" gate by exploiting (a) donor
POOLING — averaging away the one patient whose tumour genetics make 8% of their heart cells lethal; (b)
scRNA DROPOUT — picking near-undetectable antigens whose co-positivity is dropout-deflated to look clean.
This module builds the HARD guards against both and proves them on planted attacks:

  FIX-1  vital leak = MAX over donors of the Jeffreys UPPER bound (NEVER the donor-pooled estimate).
  FIX-3  AND-arm dropout-deflation guard: if any positive arm is near-undetectable, vital safety is UNCERTAIN
         (the leak estimate is on the wrong, dropout-deflated quantity — cannot certify safe).
  fail-closed vital: a gate that leaks into ANY non-regenerating vital type is NOT BORN (fitness = -inf);
         no amount of tumour coverage buys back a vital hit (never subtracted, never multiplied).
  lexicographic objective: rank = (not selective, not safe, worst_leak, -coverage) — NEVER a scalar
         coverage - lambda*leak (a scalar lets the search arbitrage safety for coverage).
  three-partition DONOR split: DISCOVERY builds, HELDOUT-SELECT drives early-stop/pruning, HELDOUT-REPORT
         is never touched until final scoring (a validation set used for early-stopping is not a test set).

HONEST CEILING: a top gate is a TRANSCRIPT-LEVEL hypothesis (mRNA != surface protein; co-positivity needs
CITE-seq), NOT a cure or a safety proof. "No clean gate exists in this space" is a first-class outcome.
Recognition-selectivity is a separate axis — never multiplied with RUNG-1/2/3 or escape-durability.

USAGE: python scripts/22_gate_optimizer.py        (runs the synthetic attack harness; CPU, seconds)
The real-data Census run (donor-aware fetch over the full surfaceome) is the Colab handoff (scripts 23/24
add FDR + the HLA-LOH population column; this file establishes the un-gameable scoring + search core).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("lg", PROJECT_ROOT / "scripts" / "18_logicgate_search.py")
lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)

# ---- frozen bars (would live in manifest_PREREG.yaml for the real run) ----
K = 2
LEAK_BAR = 0.02
REGEN_BAR = 0.15
COV_BAR = 0.15
AND_DETECT_FLOOR = 0.05          # an AND arm below this max-detection cannot certify vital-safe (Fix-3)
MIN_DONORS = 3                   # >= this many powered held-out donors per vital type, else UNCERTAIN (Fix-2)
MIN_VITAL_CELLS_PER_DONOR = 185  # per-donor floor (Jeffreys UB at 0 co-positives <= leak_bar) (Fix-2)
BEAM_WIDTH = 8
MAX_LITERALS = 3


def positive(panel, gene, k=K):
    return panel.counts[:, panel.genes.index(gene)] >= k


def gate_fire(panel, pos_genes, neg_genes, k=K):
    """Boolean fire per cell: ALL pos_genes positive AND NO neg_genes positive."""
    fire = np.ones(panel.counts.shape[0], bool)
    for g in pos_genes:
        fire &= positive(panel, g, k)
    for g in neg_genes:
        fire &= ~positive(panel, g, k)
    return fire


def arm_max_detect(panel, gene, k=K):
    """Max detection fraction of `gene` over any cell type with >=20 cells (Fix-3 dropout guard)."""
    best = 0.0
    for ct in np.unique(panel.cell_type):
        m = panel.cell_type == ct
        if m.sum() >= 20:
            best = max(best, float(positive(panel, gene, k)[m].mean()))
    return best


def arm_max_detect_in(panel, gene, sub_mask, k=K):
    """Max detection of `gene` over cell types WITHIN sub_mask (>=20 cells). Used for the AND-NOT blocker
    falsifiability guard: a NOT-arm only protects NORMAL cells if the blocker is detectable IN NORMAL tissue
    (detection only in tumour does not make the NOT credible) — audit guards/F2."""
    best = 0.0
    if not sub_mask.any():
        return best
    for ct in np.unique(panel.cell_type[sub_mask]):
        m = sub_mask & (panel.cell_type == ct)
        if m.sum() >= 20:
            best = max(best, float(positive(panel, gene, k)[m].mean()))
    return best


def _vital_leak_max_over_donor(panel, fire):
    """FIX-1: vital leak = MAX over (vital cell type, donor) of the Jeffreys UPPER bound — never pooled.
    Returns (vital_leak, worst_label, audited_ok_per_type). A donor counts only if powered (>= per-donor
    floor); a vital type needs >= MIN_DONORS powered donors or it is UNAUDITED (fail-closed)."""
    norm = panel.compartment == "normal"
    donors = panel.donor if panel.donor is not None else np.array(["_one"] * len(panel.cell_type))
    vital_leak, worst = 0.0, None
    audited = {}
    for ct in np.unique(panel.cell_type[norm]):
        if ct not in lg.VITAL_NONREGEN:
            continue
        powered_donors = 0
        for d in np.unique(donors[norm & (panel.cell_type == ct)]):
            m = norm & (panel.cell_type == ct) & (donors == d)
            n = int(m.sum())
            if n < MIN_VITAL_CELLS_PER_DONOR:
                continue                       # a blind look, not a clean look — does not count
            powered_donors += 1
            ub = lg.jeffreys_upper(int(fire[m].sum()), n)
            if ub > vital_leak:
                vital_leak, worst = ub, f"{ct}@{d}"
        audited[ct] = powered_donors >= MIN_DONORS
    return vital_leak, worst, audited


def _normal_leaks(panel, fire, required_vital):
    """WORST-DONOR (not pooled) Jeffreys UPPER-bound leak for EVERY normal cell type — audit guards/F1: the
    anti-pooling guard must NOT be limited to the vital set. Per type: max over POWERED donors (>=
    MIN_VITAL_CELLS_PER_DONOR cells) of the per-donor Jeffreys UB; if NO donor of the type is powered, fall
    back to the POOLED UB and RECORD the type in pooled_fallback (honest: a donor-private leak there could be
    averaged away — a known, REPORTED limitation, never silent). MALIGNANT cell types (those present in the
    tumour compartment) are EXCLUDED from the normal-leak audit — audit stats/F1: a decoy label permutation
    can relabel real tumour cells 'normal'; their high expression would otherwise pollute the strict-leak and
    force every decoy unsafe (collapsing the FDR null). No-op on real data (no malignant cell is compartment
    'normal'). Returns vital/strict/regen worst-donor leaks + per-vital-type audit flags + pooled_fallback."""
    norm = panel.compartment == "normal"
    donors = panel.donor if panel.donor is not None else np.array(["_one"] * len(panel.cell_type))
    malignant_types = set(np.unique(panel.cell_type[panel.compartment == "tumour"]).tolist())
    vital_leak, vital_worst, strict_leak, regen_leak = 0.0, None, 0.0, 0.0
    audited, pooled_fallback = {}, []
    for ct in np.unique(panel.cell_type[norm]):
        if ct in malignant_types:
            continue
        ct_mask = norm & (panel.cell_type == ct)
        best_ub, best_lab, powered = 0.0, None, 0
        for d in np.unique(donors[ct_mask]):
            m = ct_mask & (donors == d)
            n = int(m.sum())
            if n < MIN_VITAL_CELLS_PER_DONOR:
                continue
            powered += 1
            ub = lg.jeffreys_upper(int(fire[m].sum()), n)
            if ub > best_ub:
                best_ub, best_lab = ub, f"{ct}@{d}"
        if powered == 0:
            tot = int(ct_mask.sum())
            if tot < 20:
                continue                       # too few cells to assess at all
            best_ub, best_lab = lg.jeffreys_upper(int(fire[ct_mask].sum()), tot), f"{ct}(pooled)"
            pooled_fallback.append(ct)         # could not do worst-donor (under-sampled per donor) -> flagged
        if ct in lg.VITAL_NONREGEN:
            audited[ct] = powered >= MIN_DONORS
            if best_ub > vital_leak:
                vital_leak, vital_worst = best_ub, best_lab
        elif ct in lg.REGEN_TYPES:
            regen_leak = max(regen_leak, best_ub)
        else:
            strict_leak = max(strict_leak, best_ub)
    return {"vital_leak": vital_leak, "vital_worst": vital_worst, "strict_leak": strict_leak,
            "regen_leak": regen_leak, "audited": audited, "pooled_fallback": pooled_fallback}


def score_gate(panel, pos_genes, neg_genes, required_vital):
    """Donor-aware, fail-closed, dropout-guarded score. Returns the verdict dict. fitness=-inf if NOT BORN.
    WORST-DONOR (never pooled) leak for ALL normal types (guards/F1); malignant cells excluded from the
    normal-leak audit (stats/F1); AND-NOT blocker falsifiability guard (guards/F2)."""
    fire = gate_fire(panel, pos_genes, neg_genes)
    is_tum = panel.compartment == "tumour"
    coverage = float(fire[is_tum].mean()) if is_tum.any() else 0.0

    L = _normal_leaks(panel, fire, required_vital)
    vital_leak, vital_worst = L["vital_leak"], L["vital_worst"]
    strict_leak, regen_leak = L["strict_leak"], L["regen_leak"]
    unaudited = {ct for ct in required_vital if not L["audited"].get(ct, False)}

    # dropout guards: every POSITIVE arm robustly detectable somewhere; every AND-NOT blocker robustly
    # detectable IN NORMAL tissue (else the NOT cannot protect normals — its '~positive' is dropout, not absence).
    norm = panel.compartment == "normal"
    underdetected = [g for g in pos_genes if arm_max_detect(panel, g) < AND_DETECT_FLOOR]
    unfalsifiable_neg = [g for g in neg_genes if arm_max_detect_in(panel, g, norm) < AND_DETECT_FLOOR]

    # verdict (safety FIRST, independent of coverage; fail-closed)
    if vital_leak > LEAK_BAR:
        verdict = f"NON-SELECTIVE (vital leak {vital_leak:.3f} @ {vital_worst} — FORBID)"
    elif strict_leak > LEAK_BAR:
        verdict = f"NON-SELECTIVE (non-regen normal leak {strict_leak:.3f})"
    elif regen_leak > REGEN_BAR:
        verdict = f"NON-SELECTIVE (regen ceiling {regen_leak:.3f})"
    elif underdetected:
        verdict = f"UNCERTAIN (AND-arm near-undetectable, dropout-deflated: {underdetected})"
    elif unfalsifiable_neg:
        verdict = f"UNCERTAIN (AND-NOT blocker not detectable in normal tissue, unfalsifiable: {unfalsifiable_neg})"
    elif unaudited:
        verdict = f"UNCERTAIN (vital types under-powered/unaudited: {sorted(unaudited)} — FAIL-CLOSED)"
    else:
        verdict = "SAFE-LOW-COVERAGE" if coverage < COV_BAR else "SELECTIVE"

    safe = verdict in ("SELECTIVE", "SAFE-LOW-COVERAGE")
    selective = verdict == "SELECTIVE"
    return {"pos": list(pos_genes), "neg": list(neg_genes),
            "gate": " AND ".join(pos_genes) + ("".join(f" AND-NOT {g}" for g in neg_genes)),
            "coverage": round(coverage, 3), "vital_leak": round(vital_leak, 3), "vital_worst": vital_worst,
            "strict_leak": round(strict_leak, 3), "regen_leak": round(regen_leak, 3),
            "pooled_fallback": L["pooled_fallback"],
            "safe": safe, "selective": selective, "verdict": verdict,
            "transcript_only": True, "protein_copositivity_status": "NO_SINGLECELL_PROTEIN_DATA"}


def _beta_ppf_vec(k, n, alpha=0.05):
    """Vectorised Jeffreys UPPER bound over arrays (elementwise-identical to lg.jeffreys_upper): n<=0 -> 1.0
    (fail-closed), k>=n -> 1.0, else beta.ppf(1-alpha, k+0.5, n-k+0.5)."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    out = np.ones(n.shape, float)
    m = (n > 0) & (k < n)
    if m.any():
        from scipy.stats import beta
        out[m] = beta.ppf(1 - alpha, k[m] + 0.5, n[m] - k[m] + 0.5)
    return out


def score_gates_vec(panel, gates, required_vital):
    """VECTORISED equivalent of [score_gate(panel, g['pos'], g['neg'], required_vital) for g in gates] —
    bit-equivalent on coverage/leaks/verdict (validated batch==per-gate), but tractable at real scale
    (M cells x thousands of donors x large family x perms). Precomputes per-(cell_type,donor) groups + the
    per-gene positive columns ONCE; each gate is then one bincount + a Jeffreys bound on only the NONZERO
    groups (zero-fire groups reuse a precomputed UB). Same worst-donor / fail-closed / dropout / AND-NOT
    semantics as score_gate (it shares the constants and lg.VITAL_NONREGEN / REGEN_TYPES classification)."""
    ct = np.asarray(panel.cell_type); comp = np.asarray(panel.compartment)
    donors = np.asarray(panel.donor) if panel.donor is not None else np.array(["_one"] * len(ct))
    is_tum = comp == "tumour"; ntum = int(is_tum.sum())
    malignant = set(np.unique(ct[is_tum]).tolist()) if is_tum.any() else set()
    norm = comp == "normal"
    normv = norm & ~np.isin(ct, list(malignant)) if malignant else norm           # exclude malignant from leak
    nv_idx = np.where(normv)[0]
    # groups over normal-valid cells = (cell_type, donor)
    gkeys = np.array([f"{a}\x01{b}" for a, b in zip(ct[nv_idx], donors[nv_idx])]) if nv_idx.size else np.array([], object)
    uniq, ginv = (np.unique(gkeys, return_inverse=True) if gkeys.size else (np.array([], object), np.array([], int)))
    ng = len(uniq)
    gtot = np.bincount(ginv, minlength=ng).astype(float) if ng else np.array([])
    gtype = np.array([u.split("\x01")[0] for u in uniq]) if ng else np.array([], object)
    utypes, tinv = (np.unique(gtype, return_inverse=True) if ng else (np.array([], object), np.array([], int)))
    is_vital_t = np.isin(utypes, list(lg.VITAL_NONREGEN)) if len(utypes) else np.array([], bool)
    is_regen_t = np.isin(utypes, list(lg.REGEN_TYPES)) if len(utypes) else np.array([], bool)
    gpowered = gtot >= MIN_VITAL_CELLS_PER_DONOR if ng else np.array([], bool)
    UB0 = _beta_ppf_vec(np.zeros(ng), gtot) if ng else np.array([])               # k=0 fast-path per group
    used = sorted({g for gate in gates for g in gate["pos"] + gate["neg"]})
    P = {g: (panel.counts[:, panel.genes.index(g)] >= K) for g in used}
    Pnv = {g: P[g][nv_idx] for g in used}
    det_all = {g: arm_max_detect(panel, g) for g in used}
    det_norm = {g: arm_max_detect_in(panel, g, norm) for g in used}

    out = []
    for gate in gates:
        pos, neg = list(gate["pos"]), list(gate["neg"])
        fire_t = np.ones(ntum, bool) if ntum else np.zeros(0, bool)
        if ntum:
            tnz = np.where(is_tum)[0]
            for g in pos: fire_t &= P[g][tnz]
            for g in neg: fire_t &= ~P[g][tnz]
        coverage = float(fire_t.mean()) if ntum else 0.0
        firenv = np.ones(nv_idx.size, bool)
        for g in pos: firenv &= Pnv[g]
        for g in neg: firenv &= ~Pnv[g]
        gk = np.bincount(ginv, weights=firenv, minlength=ng) if ng else np.array([])
        gub = UB0.copy() if ng else np.array([])
        nz = gk > 0 if ng else np.array([], bool)
        if ng and nz.any():
            gub[nz] = _beta_ppf_vec(gk[nz], gtot[nz])
        type_leak = np.zeros(len(utypes)); type_pw = np.zeros(len(utypes))
        type_k = np.zeros(len(utypes)); type_n = np.zeros(len(utypes))
        if ng:
            np.add.at(type_k, tinv, gk); np.add.at(type_n, tinv, gtot)
            if gpowered.any():
                np.maximum.at(type_leak, tinv[gpowered], gub[gpowered])
                np.add.at(type_pw, tinv[gpowered], 1.0)
        pooled_fallback = []
        for i in range(len(utypes)):
            if type_pw[i] == 0:                       # no powered donor -> pooled fallback (or skip if <20 cells)
                if type_n[i] < 20:
                    type_leak[i] = 0.0
                else:
                    type_leak[i] = float(_beta_ppf_vec(np.array([type_k[i]]), np.array([type_n[i]]))[0])
                    pooled_fallback.append(str(utypes[i]))
        vmask = is_vital_t; rmask = is_regen_t; smask = (~is_vital_t) & (~is_regen_t) if len(utypes) else np.array([], bool)
        vital_leak = float(type_leak[vmask].max()) if vmask.any() else 0.0
        regen_leak = float(type_leak[rmask].max()) if rmask.any() else 0.0
        strict_leak = float(type_leak[smask].max()) if smask.any() else 0.0
        # worst (vital type, donor) label = the powered vital group with the max UB (matches score_gate's
        # "ct@donor"); fall back to the worst vital TYPE label when only a pooled estimate exists.
        vgrp = (is_vital_t[tinv] & gpowered) if ng else np.array([], bool)
        if ng and vgrp.any():
            wi = np.where(vgrp)[0][int(np.argmax(gub[vgrp]))]
            vital_worst = str(uniq[wi]).replace("\x01", "@")
        else:
            vital_worst = str(utypes[vmask][int(np.argmax(type_leak[vmask]))]) if vmask.any() else None
        audited = {str(utypes[i]): bool(type_pw[i] >= MIN_DONORS) for i in range(len(utypes)) if is_vital_t[i]}
        unaudited = {c for c in required_vital if not audited.get(c, False)}
        underdetected = [g for g in pos if det_all[g] < AND_DETECT_FLOOR]
        unfalsifiable_neg = [g for g in neg if det_norm[g] < AND_DETECT_FLOOR]
        if vital_leak > LEAK_BAR:
            verdict = f"NON-SELECTIVE (vital leak {vital_leak:.3f} @ {vital_worst} — FORBID)"
        elif strict_leak > LEAK_BAR:
            verdict = f"NON-SELECTIVE (non-regen normal leak {strict_leak:.3f})"
        elif regen_leak > REGEN_BAR:
            verdict = f"NON-SELECTIVE (regen ceiling {regen_leak:.3f})"
        elif underdetected:
            verdict = f"UNCERTAIN (AND-arm near-undetectable, dropout-deflated: {underdetected})"
        elif unfalsifiable_neg:
            verdict = f"UNCERTAIN (AND-NOT blocker not detectable in normal tissue, unfalsifiable: {unfalsifiable_neg})"
        elif unaudited:
            verdict = f"UNCERTAIN (vital types under-powered/unaudited: {sorted(unaudited)} — FAIL-CLOSED)"
        else:
            verdict = "SAFE-LOW-COVERAGE" if coverage < COV_BAR else "SELECTIVE"
        safe = verdict in ("SELECTIVE", "SAFE-LOW-COVERAGE")
        out.append({"pos": pos, "neg": neg,
                    "gate": " AND ".join(pos) + ("".join(f" AND-NOT {g}" for g in neg)),
                    "coverage": round(coverage, 3), "vital_leak": round(vital_leak, 3), "vital_worst": vital_worst,
                    "strict_leak": round(strict_leak, 3), "regen_leak": round(regen_leak, 3),
                    "pooled_fallback": pooled_fallback, "safe": safe, "selective": verdict == "SELECTIVE",
                    "verdict": verdict, "transcript_only": True,
                    "protein_copositivity_status": "NO_SINGLECELL_PROTEIN_DATA"})
    return out


def rank_key(r):
    """LEXICOGRAPHIC — never a scalar combination of leak and coverage (no-multiply invariant)."""
    return (not r["selective"], not r["safe"], r["vital_leak"], r["strict_leak"], -r["coverage"])


def fitness_inf_if_unsafe(r):
    """fail-closed: an unsafe gate is NOT BORN (fitness = -inf); coverage NEVER buys back a vital hit."""
    if not r["safe"]:
        return float("-inf")
    return r["coverage"]


def assert_no_multiply(rank_callable):
    """FIX-5b: prove the ranking path is lexicographic (a tuple), not a scalar leak/coverage product.
    Raises if a scalarized objective is passed."""
    probe = {"selective": True, "safe": True, "vital_leak": 0.01, "strict_leak": 0.0, "coverage": 0.5}
    out = rank_callable(probe)
    if not isinstance(out, tuple):
        raise AssertionError("HARD RULE: ranking must be a LEXICOGRAPHIC TUPLE, never a scalar "
                             "combination of leak and coverage (a scalar lets the search trade safety for coverage).")


def optimize(disc, sel, activators, partners, required_vital, max_literals=MAX_LITERALS, beam=BEAM_WIDTH):
    """Greedy seed + bounded beam. Seed from activators by tumour COVERAGE (an activator alone is EXPECTED
    unsafe — the whole point of AND-gating is that adding a partner restores safety). ALL keep/prune
    decisions read HELDOUT-SELECT via the LEXICOGRAPHIC rank_key (safe-ness first, then coverage), so the
    rigorous 'improve, pull back on negatives' = keep a literal only if it strictly improves the held-out
    rank tuple (turning unsafe->safe is the biggest improvement). Final SELECTIVE gates fall out at the top."""
    assert_no_multiply(rank_key)
    frontier = [{"pos": [a], "neg": []} for a in activators]   # seed structure (unsafe-alone is fine)
    best = list(frontier)
    for _ in range(max_literals - 1):
        cand = []
        for gate in frontier:
            rk_parent = rank_key(score_gate(sel, gate["pos"], gate["neg"], required_vital))
            for g in partners:
                if g in gate["pos"] or g in gate["neg"]:
                    continue
                for key in ("pos", "neg"):
                    ng = {"pos": gate["pos"] + ([g] if key == "pos" else []),
                          "neg": gate["neg"] + ([g] if key == "neg" else [])}
                    rk = rank_key(score_gate(sel, ng["pos"], ng["neg"], required_vital))
                    if rk < rk_parent:          # PULL BACK: keep only strict held-out improvements
                        cand.append((ng, rk))
        if not cand:
            break
        seen, dedup = set(), []
        for ng, rk in sorted(cand, key=lambda x: x[1]):
            kk = (frozenset(ng["pos"]), frozenset(ng["neg"]))
            if kk not in seen:
                seen.add(kk); dedup.append(ng)
        frontier = dedup[:beam]
        best += frontier
    out, seen = [], set()
    for gate in best:
        kk = (frozenset(gate["pos"]), frozenset(gate["neg"]))
        if kk not in seen:
            seen.add(kk); out.append(gate)
    return out


def three_partition(donors, seed=20260530):
    """Split DONORS (never cells) into DISCOVERY / SELECT / REPORT — pairwise disjoint by donor."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(donors)))
    rng.shuffle(uniq)
    n = len(uniq)
    a, b = n // 3, 2 * n // 3
    return set(uniq[:a]), set(uniq[a:b]), set(uniq[b:])


def subset(panel, donor_set):
    m = np.array([d in donor_set for d in panel.donor])
    return lg.Panel(panel.counts[m], panel.genes, panel.cell_type[m], panel.tissue[m], panel.compartment[m],
                    donor=panel.donor[m])


# =====================================================================================================
#  SYNTHETIC ATTACK HARNESS — prove the guards catch the reward-hacks BEFORE trusting the real Census run
# =====================================================================================================
def _build_attack_panel(rng):
    """A donor-resolved synthetic panel with: a planted CLEAN gate (tumour-only), and two reward-hack traps:
    ATTACK-3 (donor-pooling): gate clean when donors pooled, but ONE donor's cardiomyocytes are 8% double+;
    ATTACK-1 (dropout-deflation): a gate truly double-positive on cardiomyocytes but its partner is
    near-undetectable so the naive co-positivity reads ~0 (false-safe)."""
    HI, MID, ABSENT, UNDETECT, LOWDETECT = 6.0, 2.2, 0.01, 0.03, 0.5
    #  ABSENT (P>=2 ~5e-5, truly off) | UNDETECT (~4e-4, below the 5% detection floor) | LOWDETECT (~9% +)
    genes = ["ACT", "CLEAN", "ATK3", "ATK1_undetect", "DECOY"]
    #         activator(broad), clean tumour-only partner, attack-3 partner, attack-1 undetectable, lung-epi decoy
    blocks_c, ct, ts, comp, don = [], [], [], [], []
    NPER = 220   # per (cell_type, donor) > MIN_VITAL_CELLS_PER_DONOR; UB(0/220)~0.0135 < leak_bar -> powered+clean

    def add(cell_type, tissue, compartment, donor, lams, n=NPER):
        blocks_c.append(np.column_stack([rng.poisson(l, n) for l in lams]))
        ct.extend([cell_type] * n); ts.extend([tissue] * n); comp.extend([compartment] * n); don.extend([donor] * n)

    for d in range(10):   # 10 normal donors
        dn = f"ds::donor{d}"
        # ACT = broadly-expressed activator (HI on heart -> UNSAFE ALONE; the AND must rescue it).
        # CLEAN/ATK3/DECOY truly ABSENT on normal heart. ATK3 EXCEPT donor0 where it is LOWDETECT (~8-9%
        # of donor0's cardiomyocytes become ACT+ATK3 double-positive = the lethal donor pooling would erase).
        # ATK1_undetect is UNDETECT everywhere (below the dropout floor) -> its 'safety' is unfalsifiable.
        add("cardiomyocyte", "heart", "normal", dn,
            [HI, ABSENT, (LOWDETECT if d == 0 else ABSENT), UNDETECT, ABSENT])
        add("hepatocyte", "liver", "normal", dn, [MID, ABSENT, ABSENT, UNDETECT, ABSENT])
        # lung normal epithelium: ACT + DECOY both present -> ACT AND DECOY LEAKS here (DECOY is a bad partner)
        add("normal_epithelium", "lung", "normal", dn, [MID, ABSENT, ABSENT, UNDETECT, MID])
    # tumour (malignant): ACT+CLEAN co-positive = the real gate; ATK3 also high (ATK3 gate has coverage too);
    # ATK1_undetect UNDETECT (so even its coverage is dropout-deflated); DECOY MID.
    for d in range(4):
        add("tumour_malignant", "tumour", "tumour", f"tds::tdonor{d}", [HI, HI, HI, UNDETECT, MID], n=300)
    counts = np.vstack(blocks_c)
    return lg.Panel(counts, genes, np.array(ct), np.array(ts), np.array(comp), donor=np.array(don))


def _attack_harness():
    rng = np.random.default_rng(20260530)
    panel = _build_attack_panel(rng)
    required_vital = {"cardiomyocyte"}
    print("=" * 84)
    print("RUNG 5 — gate-optimizer GUARD validation on synthetic planted attacks")
    print("=" * 84)
    print(f"panel: {panel.counts.shape[0]} cells, {len(set(panel.donor))} donors, "
          f"vital={sorted(set(panel.cell_type) & lg.VITAL_NONREGEN)}")

    # control: the planted CLEAN gate (ACT AND CLEAN) — tumour-only -> should be SELECTIVE
    clean = score_gate(panel, ["ACT", "CLEAN"], [], required_vital)
    # ATTACK-3: ACT AND ATK3 — pooled cardiomyocyte leak tiny, but donor0 is 8% double+ -> max-over-donor catches it
    atk3 = score_gate(panel, ["ACT", "ATK3"], [], required_vital)
    # what a NAIVE donor-POOLED scorer would say about ATK3 (the bug we are guarding against):
    norm = panel.compartment == "normal"; cm = norm & (panel.cell_type == "cardiomyocyte")
    fire3 = gate_fire(panel, ["ACT", "ATK3"], [])
    pooled_ub = lg.jeffreys_upper(int(fire3[cm].sum()), int(cm.sum()))
    # ATTACK-1: ACT AND ATK1_undetect — partner near-undetectable -> dropout-deflation guard -> UNCERTAIN
    atk1 = score_gate(panel, ["ACT", "ATK1_undetect"], [], required_vital)
    # ATTACK-4 (audit guards/F2): CLEAN AND-NOT ATK1_undetect — the POS arm (CLEAN) is tumour-only so this gate
    # would read SELECTIVE, EXCEPT the blocker is undetectable in normal tissue, so the NOT cannot be shown to
    # protect normals (its '~positive' is dropout, not proven absence) -> must be UNCERTAIN (isolates the guard).
    atk4 = score_gate(panel, ["CLEAN"], ["ATK1_undetect"], required_vital)

    print("-" * 84)
    print(f"[planted CLEAN]  ACT AND CLEAN        cov={clean['coverage']:.2f} vital={clean['vital_leak']:.3f} -> {clean['verdict']}")
    print(f"[ATTACK-3 pool]  ACT AND ATK3         pooled-cardiomyocyte UB={pooled_ub:.3f} (a naive scorer calls this SAFE)")
    print(f"[ATTACK-3 fix]   ACT AND ATK3         max-over-donor vital={atk3['vital_leak']:.3f}@{atk3['vital_worst']} -> {atk3['verdict']}")
    print(f"[ATTACK-1 drop]  ACT AND ATK1_undetect cov={atk1['coverage']:.2f} vital={atk1['vital_leak']:.3f} -> {atk1['verdict']}")
    print(f"[ATTACK-4 NOT ]  CLEAN AND-NOT ATK1_undetect (undetectable blocker) -> {atk4['verdict']}")

    # run the optimizer and confirm it picks the clean gate, not the attacks
    disc_d, sel_d, rep_d = three_partition(panel.donor)
    # ensure tumour donors present in all splits for coverage scoring (tumour donors are separate keys)
    tum_d = set(d for d, c in zip(panel.donor, panel.compartment) if c == "tumour")
    disc = subset(panel, disc_d | tum_d); sel = subset(panel, sel_d | tum_d)
    shortlist = optimize(disc, sel, activators=["ACT"], partners=["CLEAN", "ATK3", "ATK1_undetect", "DECOY"],
                         required_vital=required_vital)
    # FINAL safety certification uses ALL donors (max-over-donor is conservative — more donors only ever
    # REVEAL more leak; the search used disc/select without peeking, the verdict uses the full evidence).
    scored = sorted([score_gate(panel, g["pos"], g["neg"], required_vital) for g in shortlist], key=rank_key)
    print("-" * 84)
    print(f"optimizer shortlist ({len(shortlist)} gates; search on DISCOVERY/SELECT donors, certified on ALL donors):")
    for r in scored[:6]:
        print(f"  {r['gate']:28s} cov={r['coverage']:.2f} vital={r['vital_leak']:.3f} -> {r['verdict']}")
    top = scored[0] if scored else None

    checks = {
        "planted CLEAN gate scores SELECTIVE": clean["selective"],
        "ATTACK-3 (donor-pooling): naive pooled UB looks SAFE (<bar)": pooled_ub <= LEAK_BAR,
        "ATTACK-3 FIX: max-over-donor catches the lethal donor -> NON-SELECTIVE": not atk3["safe"] and atk3["vital_leak"] > LEAK_BAR,
        "ATTACK-1 (dropout-deflation): near-undetectable arm -> UNCERTAIN (not SELECTIVE)": (not atk1["selective"]) and atk1["verdict"].startswith("UNCERTAIN"),
        "ATTACK-4 (AND-NOT falsifiability): undetectable blocker -> UNCERTAIN (not SELECTIVE)": (not atk4["selective"]) and atk4["verdict"].startswith("UNCERTAIN"),
        "fail-closed: unsafe gate fitness == -inf": fitness_inf_if_unsafe(atk3) == float("-inf"),
        "objective is lexicographic (no-multiply): scalarized rank raises": _scalarized_raises(),
        "optimizer top gate is the planted CLEAN gate (not an attack)": top is not None and set(top["pos"]) == {"ACT", "CLEAN"} and top["selective"],
    }
    print("=" * 84)
    print("GUARD CHECKS:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    ok = all(checks.values())
    print("=" * 84)
    print("CEILING: transcript-level hypothesis; mRNA!=protein (CITE-seq confirms co-positivity); a top gate")
    print("is the best NEXT wet-lab experiment, not a cure or a safety proof. Real Census run = Colab (next).")
    return 0 if ok else 1


def _scalarized_raises():
    try:
        assert_no_multiply(lambda r: r["coverage"] - 5.0 * r["vital_leak"])   # a forbidden scalar objective
        return False
    except AssertionError:
        return True


if __name__ == "__main__":
    sys.exit(_attack_harness())
