#!/usr/bin/env python3
"""
RUNG 4 / Step-5 — logic-gate recognition SELECTIVITY ENGINE.

Solves the wall Step-3 + RUNG-3 keep hitting: NO single tumour-exclusive antigen exists (every Step-2
activator — HER2, Trop2, MUC1, ... — FAILED the scripts/07 vital-parenchyma safety audit). The fix is a
LOGIC GATE: kill only a cell that shows antigen A AND B together (or A AND-NOT a 'safe' marker), so a
normal cell carrying only one badge is spared. This engine scores such gates by the single most dangerous
question, computed at TRUE SINGLE-CELL resolution:

   is there ANY normal cell — especially in heart / brain / kidney — that wears BOTH badges at once?

CRUX (adversary-caught): selectivity MUST be single-cell. Two antigens both present "in the liver" tells
you nothing — if A is on hepatocytes and B on endothelium, the AND-gate is SAFE; if one hepatocyte has
BOTH, it is LETHAL. Bulk / pseudobulk co-expression hides exactly the cell that kills the patient. This
engine computes per-cell co-positivity and reports the bulk-vs-single-cell gap explicitly.

HONEST CEILING: a HYPOTHESIS generator over mRNA, not proof. mRNA != surface protein (single-cell
r~0.1-0.4); 'absent' can be scRNA dropout, not true absence; co-localisation != a functional circuit that
clusters DR5 to fire caspase-8 (wet-lab). Recognition-selectivity is a SEPARATE FOURTH axis — NEVER
multiplied into RUNG-1 (death-timing) / RUNG-2 (clustering, refuted) / RUNG-3 (tissue dynamics). It is
also never multiplied into escape-durability: tightening AND-selectivity WIDENS the escape surface, and
they are reported side by side.

This engine is data-agnostic: it takes a per-cell counts matrix + cell_type/tissue/compartment labels
(plain numpy/pandas — no anndata), so it runs identically on the synthetic biological-ground-truth
benchmark (scripts/20, local) and on real CELLxGENE single-cell atlases (scripts/17, Colab).
"""
from __future__ import annotations

import numpy as np

# Non-regenerating vital parenchyma — co-expression here FORBIDS a gate. These tissues do not regenerate,
# so a single double-positive cell is potentially lethal (heart/brain/kidney/pancreas-islet/adrenal/muscle).
VITAL_NONREGEN = {"cardiomyocyte", "neuron", "kidney_tubule", "kidney_podocyte",
                  "pancreatic_islet", "adrenal_cortical", "skeletal_myocyte"}
# Regenerating tissue — single-cell co-expression TOLERATED up to a higher (but finite) ceiling: you can
# survive transient loss of liver/gut/marrow/epithelium, but not wholesale denudation.
REGEN_TYPES = {"hepatocyte", "marrow_hsc", "enterocyte", "keratinocyte", "liver_endothelial",
               "normal_epithelium"}


class Panel:
    """Per-cell counts matrix + labels (plain class — importlib-safe). counts:(n_cells,n_genes) int UMI;
    genes:list; cell_type/tissue/compartment:(n_cells,) str arrays ('normal'|'tumour')."""
    def __init__(self, counts, genes, cell_type, tissue, compartment):
        self.counts = counts; self.genes = list(genes)
        self.cell_type = cell_type; self.tissue = tissue; self.compartment = compartment

    def gidx(self, name):
        return self.genes.index(name)

    def positive(self, gene, k):
        """Per-cell POSITIVE iff UMI >= k (k>=2 suppresses ambient/background; a 0 is 'undetected')."""
        return self.counts[:, self.gidx(gene)] >= k

    def detect_frac(self, gene, k):
        return float(self.positive(gene, k).mean())


def gate_fire(panel: Panel, A, B, logic, k):
    """Per-cell boolean: does the gate fire in this cell? logic in {SINGLE, AND, AND_NOT}."""
    a = panel.positive(A, k)
    if logic == "SINGLE":
        return a
    b = panel.positive(B, k)
    if logic == "AND":
        return a & b
    if logic == "AND_NOT":
        return a & ~b
    raise ValueError(logic)


def not_arm_falsifiable(panel: Panel, B, k, min_detect=0.10):
    """An AND-NOT 'safe' call rests on B being ABSENT in normals — but a single-cell zero is 'undetected',
    not 'proven absent'. The NOT is falsifiable only if B is robustly DETECTABLE somewhere (the assay can
    see it). Returns (ok, max_detect_in_any_celltype)."""
    best = 0.0
    for ct in np.unique(panel.cell_type):
        m = panel.cell_type == ct
        if m.sum() >= 20:
            best = max(best, float(panel.positive(B, k)[m].mean()))
    return best >= min_detect, round(best, 3)


def jeffreys_upper(k, n, alpha=0.05):
    """One-sided upper (1-alpha) confidence bound on a binomial rate k/n (Jeffreys interval). A 0/40 group
    has point-rate 0 but upper bound ~0.07 — so 'we saw no leak' is NOT credited as 'there is no leak'.
    Vital safety is gated on THIS bound, not the point estimate (the dangerous error is the false zero)."""
    if n <= 0:
        return 1.0   # no cells observed -> maximal uncertainty (fail-closed)
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, k + 0.5, n - k + 0.5)) if k < n else 1.0
    except Exception:
        return min(1.0, (k + 1.96 ** 2 / 2) / (n + 1.96 ** 2) + 1.96 / (n + 4) * 0.5)  # crude fallback


def _decide(tumour_coverage, vital_leak, strict_leak, regen_leak, not_ok, logic, leak_bar, regen_bar, cov_bar,
            unaudited_vital=None):
    """The TIERED safety verdict — single source of truth, shared by score_gate and score_gates_batch so
    they CANNOT diverge. Strict near-zero bar on non-regenerating tissue (FORBID); a finite ceiling on
    regenerating tissue. FAIL-CLOSED: if a required vital type was never adequately captured, the gate is
    NOT certifiable vital-safe — 'we never looked at the heart' must NOT read as 'the heart is clean'.
    Leak inputs are UPPER confidence bounds, so a false zero from dropout/undersampling cannot pass."""
    if tumour_coverage < cov_bar:
        return "NON-SELECTIVE (insufficient tumour coverage)"
    if vital_leak > leak_bar:
        return "NON-SELECTIVE (vital parenchyma co-expression — non-regenerating tissue FORBID)"
    if strict_leak > leak_bar:
        return "NON-SELECTIVE (non-regenerating normal leak)"
    if regen_leak > regen_bar:
        return "NON-SELECTIVE (regenerating-tissue leak exceeds the recovery ceiling)"
    if logic == "AND_NOT" and not not_ok:
        return "UNCERTAIN (NOT-arm dropout-unfalsifiable: blocker not robustly detectable)"
    if unaudited_vital:
        return f"UNCERTAIN (cannot certify vital-safe: {sorted(unaudited_vital)} not adequately captured — FAIL-CLOSED)"
    return "SELECTIVE" + (" (regen leak tolerated)" if regen_leak > leak_bar else "")


def score_gate(panel: Panel, A, B=None, logic="SINGLE", k=2, leak_bar=0.02, regen_bar=0.15, cov_bar=0.30,
               required_vital=None):
    """Score one gate. Leak metrics are UPPER confidence bounds (Jeffreys) so a false zero from dropout/
    undersampling cannot pass the vital bar. FAIL-CLOSED: required_vital types that were never adequately
    captured make the gate UNCERTAIN (not vital-safe). NEVER a single fused number."""
    fire = gate_fire(panel, A, B, logic, k)
    is_tum = panel.compartment == "tumour"
    tumour_coverage = float(fire[is_tum].mean()) if is_tum.any() else 0.0

    # per (cell_type, tissue) NORMAL group -> (k co-positive, n cells); leak = UPPER bound of k/n
    norm = panel.compartment == "normal"
    groups = {}
    for ct in np.unique(panel.cell_type[norm]):
        for ts in np.unique(panel.tissue[norm & (panel.cell_type == ct)]):
            m = norm & (panel.cell_type == ct) & (panel.tissue == ts)
            n = int(m.sum())
            if n >= 20:
                groups[(ct, ts)] = (int(fire[m].sum()), n)
    leaks = {g: jeffreys_upper(kk, nn) for g, (kk, nn) in groups.items()}     # UPPER-bound leak per group
    worst_leak = max(leaks.values(), default=0.0); worst_group = max(leaks, key=leaks.get) if leaks else None
    vit = {g: v for g, v in leaks.items() if g[0] in VITAL_NONREGEN}
    vital_leak = max(vit.values(), default=0.0); vital_group = max(vit, key=vit.get) if vit else None
    regen_leak = max((v for g, v in leaks.items() if g[0] in REGEN_TYPES), default=0.0)
    strict = {g: v for g, v in leaks.items() if g[0] not in REGEN_TYPES}
    strict_leak = max(strict.values(), default=0.0); strict_group = max(strict, key=strict.get) if strict else None
    audited_vital = {g[0] for g in groups if g[0] in VITAL_NONREGEN}
    unaudited_vital = (set(required_vital) - audited_vital) if required_vital else set()

    pseudobulk_leak = 0.0
    if logic in ("AND", "AND_NOT"):
        for ts in np.unique(panel.tissue[norm]):
            m = norm & (panel.tissue == ts)
            if m.sum() >= 20:
                a_present = panel.positive(A, k)[m].mean()
                b = panel.positive(B, k)[m]
                b_term = (1 - b.mean()) if logic == "AND_NOT" else b.mean()
                pseudobulk_leak = max(pseudobulk_leak, float(min(a_present, b_term)))

    not_ok, not_detect = (True, None)
    if logic == "AND_NOT":
        not_ok, not_detect = not_arm_falsifiable(panel, B, k)

    verdict = _decide(tumour_coverage, vital_leak, strict_leak, regen_leak, not_ok, logic,
                      leak_bar, regen_bar, cov_bar, unaudited_vital)

    return {
        "gate": f"{A}" + (f" {logic} {B}" if logic != "SINGLE" else " (single)"),
        "logic": logic, "A": A, "B": B, "k": k,
        "tumour_coverage": round(tumour_coverage, 3),
        "worst_normal_leak": round(worst_leak, 3),   # UPPER bound
        "worst_group": f"{worst_group[0]}@{worst_group[1]}" if worst_group else None,
        "vital_leak": round(vital_leak, 3),
        "vital_group": f"{vital_group[0]}@{vital_group[1]}" if vital_group else None,
        "strict_leak": round(strict_leak, 3),
        "strict_group": f"{strict_group[0]}@{strict_group[1]}" if strict_group else None,
        "regen_leak": round(regen_leak, 3),
        "pseudobulk_leak": round(pseudobulk_leak, 3),
        "bulk_trap_gap": round(pseudobulk_leak - worst_leak, 3),
        "not_arm_falsifiable": not_ok, "not_arm_detect": not_detect,
        "audited_vital": sorted(audited_vital), "unaudited_vital": sorted(unaudited_vital),
        "leak_is_upper_bound": True,
        "verdict": verdict, "selective": verdict.startswith("SELECTIVE"),
    }


def score_gates_batch(panel: Panel, specs, k=2, leak_bar=0.02, regen_bar=0.15, cov_bar=0.30,
                      not_min_detect=0.10, required_vital=None, progress=None):
    """Score MANY gates fast: precompute per-cell-type/tissue group ids + per-gene positives ONCE, then
    each gate is a couple of boolean ops + one np.bincount (milliseconds), instead of re-scanning every
    group per gate. Returns the SAME dicts as score_gate (verdict via the shared _decide), so results are
    identical — verified against score_gate in scripts/20. `specs` = list of (A, B, logic). `progress`
    = optional callback(i, n, result) for live logging."""
    import pandas as pd
    norm = panel.compartment == "normal"
    tum = panel.compartment == "tumour"
    # --- precompute group ids over NORMAL cells (only groups with >=20 cells, matching score_gate) ---
    gkey = pd.Series([f"{c}\t{t}" for c, t in zip(panel.cell_type, panel.tissue)])
    gid = np.full(len(panel.cell_type), -1, dtype=np.int64)
    glabels, gvital, gregen, gsize = [], [], [], []
    for key in pd.unique(gkey[norm]):
        ct, ts = key.split("\t")
        m = norm.copy() & (gkey.values == key)
        if m.sum() < 20:
            continue
        idx = len(glabels)
        gid[m] = idx
        glabels.append((ct, ts)); gvital.append(ct in VITAL_NONREGEN)
        gregen.append(ct in REGEN_TYPES); gsize.append(int(m.sum()))
    gvital = np.array(gvital); gregen = np.array(gregen); gsize = np.array(gsize); ng = len(glabels)
    # --- precompute per-gene positives, per-gene max detection (for NOT falsifiability), per-tissue marginals ---
    pos = {g: panel.positive(g, k) for g in panel.genes}
    cts = np.unique(panel.cell_type)
    gene_max_detect = {g: max((pos[g][panel.cell_type == c].mean() for c in cts
                               if (panel.cell_type == c).sum() >= 20), default=0.0) for g in panel.genes}
    tissues_norm = [t for t in np.unique(panel.tissue[norm])]
    tmask = {t: (norm & (panel.tissue == t)) for t in tissues_norm}

    # leak per group = UPPER confidence bound of (co-positive count / group size), vectorised
    try:
        from scipy.stats import beta as _beta
    except Exception:
        _beta = None

    def grp_upper(fire_norm):
        if not ng:
            return np.zeros(0)
        sel = gid[fire_norm]
        cnt = np.bincount(sel[sel >= 0], minlength=ng).astype(float)
        if _beta is None:
            return cnt / np.maximum(gsize, 1)
        ub = _beta.ppf(0.95, cnt + 0.5, gsize - cnt + 0.5)
        return np.where(cnt >= gsize, 1.0, np.nan_to_num(ub, nan=1.0))

    audited_vital = {glabels[j][0] for j in range(ng) if gvital[j]}
    unaudited_vital = (set(required_vital) - audited_vital) if required_vital else set()

    out = []
    for i, (A, B, logic) in enumerate(specs):
        a = pos[A]
        fire = a if logic == "SINGLE" else (a & pos[B] if logic == "AND" else a & ~pos[B])
        tumour_coverage = float(fire[tum].mean()) if tum.any() else 0.0
        fr = grp_upper(fire & norm)
        worst_leak = float(fr.max()) if ng else 0.0
        wi = int(fr.argmax()) if ng else -1
        vital_leak = float(fr[gvital].max()) if gvital.any() else 0.0
        vi = int(np.where(gvital)[0][fr[gvital].argmax()]) if gvital.any() else -1
        strict_leak = float(fr[~gregen].max()) if (~gregen).any() else 0.0
        si = int(np.where(~gregen)[0][fr[~gregen].argmax()]) if (~gregen).any() else -1
        regen_leak = float(fr[gregen].max()) if gregen.any() else 0.0
        pseudobulk = 0.0
        if logic in ("AND", "AND_NOT"):
            for t in tissues_norm:
                m = tmask[t]
                if m.sum() >= 20:
                    ap = pos[A][m].mean(); bt = (1 - pos[B][m].mean()) if logic == "AND_NOT" else pos[B][m].mean()
                    pseudobulk = max(pseudobulk, float(min(ap, bt)))
        not_ok, not_detect = (True, None)
        if logic == "AND_NOT":
            not_detect = round(gene_max_detect[B], 3); not_ok = gene_max_detect[B] >= not_min_detect
        verdict = _decide(tumour_coverage, vital_leak, strict_leak, regen_leak, not_ok, logic,
                          leak_bar, regen_bar, cov_bar, unaudited_vital)
        r = {"gate": f"{A}" + (f" {logic} {B}" if logic != "SINGLE" else " (single)"),
             "logic": logic, "A": A, "B": B, "k": k,
             "tumour_coverage": round(tumour_coverage, 3),
             "worst_normal_leak": round(worst_leak, 3),
             "worst_group": f"{glabels[wi][0]}@{glabels[wi][1]}" if wi >= 0 else None,
             "vital_leak": round(vital_leak, 3),
             "vital_group": f"{glabels[vi][0]}@{glabels[vi][1]}" if vi >= 0 else None,
             "strict_leak": round(strict_leak, 3),
             "strict_group": f"{glabels[si][0]}@{glabels[si][1]}" if si >= 0 else None,
             "regen_leak": round(regen_leak, 3),
             "pseudobulk_leak": round(pseudobulk, 3),
             "bulk_trap_gap": round(pseudobulk - worst_leak, 3),
             "not_arm_falsifiable": not_ok, "not_arm_detect": not_detect,
             "audited_vital": sorted(audited_vital), "unaudited_vital": sorted(unaudited_vital),
             "leak_is_upper_bound": True,
             "verdict": verdict, "selective": verdict.startswith("SELECTIVE")}
        out.append(r)
        if progress:
            progress(i + 1, len(specs), r)
    return out


def scrambled_null(panel: Panel, A, B, logic, k, n_perm=200, seed=20260530):
    """Permute cell-type labels and re-score; if a 'selective' gate's low leak is REPRODUCED by scrambled
    labels, the selectivity is an artifact -> VOID (the control that voided the RUNG-2 headline)."""
    rng = np.random.default_rng(seed)
    obs = score_gate(panel, A, B, logic, k)["worst_normal_leak"]
    ge = 0
    for _ in range(n_perm):
        p = Panel(panel.counts, panel.genes, rng.permutation(panel.cell_type), panel.tissue, panel.compartment)
        if score_gate(p, A, B, logic, k)["worst_normal_leak"] <= obs:
            ge += 1
    # fraction of scrambles that look AS SELECTIVE (<=obs leak); high => selectivity is label-independent => suspicious
    return {"obs_leak": obs, "frac_scramble_as_selective": round((ge + 1) / (n_perm + 1), 3)}


def escape_durability(retain_A_per_div=0.02, retain_B_per_div=0.02, n_div=40):
    """SEPARATE axis from selectivity (never multiplied). An AND gate dies if a subclone loses EITHER
    antigen, so it decays ~twice as fast as a single-antigen gate. For a CONTACT death-wave (RUNG-3b) an
    antigen-negative escaper is UNREACHABLE -> strictly worse than a CAR. Returns coverage curves + the
    AND-vs-single durability DELTA. Two numbers, never a product."""
    t = np.arange(n_div + 1)
    rA = (1 - retain_A_per_div) ** t   # fraction still antigen-A+
    rB = (1 - retain_B_per_div) ** t
    cov_single = rA
    cov_and = rA * rB
    cov_or = 1 - (1 - rA) * (1 - rB)

    def half_life(cov):
        below = np.where(cov <= 0.5)[0]
        return int(below[0]) if below.size else n_div + 1
    return {
        "divisions": t.tolist(),
        "coverage_single": [round(x, 3) for x in cov_single],
        "coverage_AND": [round(x, 3) for x in cov_and],
        "coverage_OR": [round(x, 3) for x in cov_or],
        "half_life_single_div": half_life(cov_single),
        "half_life_AND_div": half_life(cov_and),
        "durability_delta_div": half_life(cov_single) - half_life(cov_and),
        "contact_penalty": "antigen-negative escaper is UNREACHABLE by the RUNG-3b contact death-wave -> "
                           "AND-gate escape cost is strictly worse than a (bystander-capable) CAR.",
    }


# HARD RULE (mirrors scripts/15 line ~329): recognition-selectivity is a SEPARATE FOURTH axis.
MULTIPLY_RECOGNITION_WITH_OTHER_AXES = False


def assert_no_multiply():
    assert MULTIPLY_RECOGNITION_WITH_OTHER_AXES is False, (
        "HARD RULE: recognition-selectivity is NEVER multiplied/fused with RUNG-1/2/3 or with "
        "escape-durability. Report four axes side by side; selectivity and durability are in tension.")


if __name__ == "__main__":
    assert_no_multiply()
    print("logicgate engine OK — import from scripts/20 (local benchmark) or scripts/17 (Colab atlases).")
    print("escape-durability self-check:", {k: v for k, v in escape_durability().items()
                                            if k.startswith("half_life") or k.startswith("durability")})
