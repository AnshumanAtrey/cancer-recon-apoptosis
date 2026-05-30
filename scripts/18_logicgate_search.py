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

# Non-regenerating vital parenchyma — co-expression here FORBIDS a gate (heart/brain/kidney).
# (Cell-type names align with the HPA vital-parenchyma set used in scripts/07.)
VITAL_NONREGEN = {"cardiomyocyte", "neuron", "kidney_tubule", "kidney_podocyte"}
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


def score_gate(panel: Panel, A, B=None, logic="SINGLE", k=2, leak_bar=0.02, regen_bar=0.15, cov_bar=0.30):
    """Score one gate. Returns coverage, worst normal leak (vital broken out), bulk-vs-single-cell gap,
    NOT-falsifiability, and a verdict under the TIERED safety rule (strict near-zero bar on
    non-regenerating tissue incl. heart/brain/kidney; a higher but finite ceiling on regenerating tissue).
    NEVER a single fused number."""
    fire = gate_fire(panel, A, B, logic, k)
    is_tum = panel.compartment == "tumour"
    tumour_coverage = float(fire[is_tum].mean()) if is_tum.any() else 0.0

    # per (cell_type, tissue) NORMAL group leak = fraction of that group's cells the gate would kill
    groups = {}
    norm = panel.compartment == "normal"
    for ct in np.unique(panel.cell_type[norm]):
        for ts in np.unique(panel.tissue[norm & (panel.cell_type == ct)]):
            m = norm & (panel.cell_type == ct) & (panel.tissue == ts)
            if m.sum() >= 20:
                groups[(ct, ts)] = float(fire[m].mean())
    worst_leak = max(groups.values(), default=0.0)
    worst_group = max(groups, key=groups.get) if groups else None
    vital_leak = max((v for (ct, ts), v in groups.items() if ct in VITAL_NONREGEN), default=0.0)
    vital_group = max(((g, v) for g, v in groups.items() if g[0] in VITAL_NONREGEN),
                      key=lambda x: x[1], default=(None, 0.0))[0]
    regen_leak = max((v for (ct, ts), v in groups.items() if ct in REGEN_TYPES), default=0.0)
    # strict bucket = every normal group that is NOT regenerating (vital + any other non-regen normal)
    strict_leak = max((v for (ct, ts), v in groups.items() if ct not in REGEN_TYPES), default=0.0)
    strict_group = max(((g, v) for g, v in groups.items() if g[0] not in REGEN_TYPES),
                       key=lambda x: x[1], default=(None, 0.0))[0]

    # BULK TRAP: a pseudobulk method asks "are A and B both present in this tissue?" (marginal), which
    # FALSELY flags co-expression when A and B sit on DIFFERENT cells. Report the gap.
    pseudobulk_leak = 0.0
    if logic in ("AND", "AND_NOT"):
        for ts in np.unique(panel.tissue[norm]):
            m = norm & (panel.tissue == ts)
            if m.sum() >= 20:
                a_present = panel.positive(A, k)[m].mean()
                b = panel.positive(B, k)[m]
                b_term = (1 - b.mean()) if logic == "AND_NOT" else b.mean()
                pseudobulk_leak = max(pseudobulk_leak, float(min(a_present, b_term)))  # marginal AND proxy

    not_ok, not_detect = (True, None)
    if logic == "AND_NOT":
        not_ok, not_detect = not_arm_falsifiable(panel, B, k)

    # TIERED safety verdict: strict near-zero bar on non-regenerating tissue (heart/brain/kidney FORBID);
    # a higher but finite ceiling on regenerating tissue (tolerate transient loss, not denudation).
    if tumour_coverage < cov_bar:
        verdict = "NON-SELECTIVE (insufficient tumour coverage)"
    elif vital_leak > leak_bar:
        verdict = "NON-SELECTIVE (vital parenchyma co-expression — heart/brain/kidney FORBID)"
    elif strict_leak > leak_bar:
        verdict = "NON-SELECTIVE (non-regenerating normal leak)"
    elif regen_leak > regen_bar:
        verdict = "NON-SELECTIVE (regenerating-tissue leak exceeds the recovery ceiling)"
    elif logic == "AND_NOT" and not not_ok:
        verdict = "UNCERTAIN (NOT-arm dropout-unfalsifiable: blocker not robustly detectable)"
    else:
        verdict = "SELECTIVE" + (" (regen leak tolerated)" if regen_leak > leak_bar else "")

    return {
        "gate": f"{A}" + (f" {logic} {B}" if logic != "SINGLE" else " (single)"),
        "logic": logic, "A": A, "B": B, "k": k,
        "tumour_coverage": round(tumour_coverage, 3),
        "worst_normal_leak": round(worst_leak, 3),
        "worst_group": f"{worst_group[0]}@{worst_group[1]}" if worst_group else None,
        "vital_leak": round(vital_leak, 3),
        "vital_group": f"{vital_group[0]}@{vital_group[1]}" if vital_group else None,
        "strict_leak": round(strict_leak, 3),
        "strict_group": f"{strict_group[0]}@{strict_group[1]}" if strict_group else None,
        "regen_leak": round(regen_leak, 3),
        "pseudobulk_leak": round(pseudobulk_leak, 3),
        "bulk_trap_gap": round(pseudobulk_leak - worst_leak, 3),  # >0 => bulk would falsely condemn a safe gate
        "not_arm_falsifiable": not_ok, "not_arm_detect": not_detect,
        "verdict": verdict, "selective": verdict.startswith("SELECTIVE"),
    }


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
