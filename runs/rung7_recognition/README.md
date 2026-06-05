# RUNG 7 — the AND-NOT recognition-gate DISCRIMINATION model

**Question RUNG-6 left open:** RUNG-6 *counted* how many patients have a usable genetic gate. It never asked
the literal recognition question — **does the Tmod AND-NOT gate actually separate a cancer cell from a healthy
one, and how does it fail?** This models that (bar #2 of the gate ladder) and couples it to the apoptosis
commit (RUNG-1 / EARM essence), giving the first end-to-end in-silico *recognise → commit to apoptosis* demo.

## The model
`activator = Hill(antigen density)` **AND-NOT** `blocker = Hill(HLA density)` → kill-license → **all-or-none
threshold commit** (the hard limit of the EARM bistable switch; the full EARM ODE is RUNG-1/`scripts/11`, *not*
re-coupled here — this is discrimination geometry, not a dynamical apoptosis sim). Tumour: antigen-high, HLA
lost (clonal) or retained (subclonal). Normal: antigen mostly low + a leak tail, HLA high except a
downregulated "HLA-low" fraction. Parameters are literature-grounded (low-affinity CAR antigen-density
threshold; MHC-I ~71k/cell; LIR-1 blocker); per-cell distributions are **illustrative** (real joint spread
needs the atlas → Colab).

## Result (`rung7_gate_discrimination.json`, `.png`) — audited & corrected
Baseline: TPR 48%, FPR **1.4%**, **ROC-AUC 0.88** (rank-exact). Two findings, with sensitivity sweeps:

1. **Safety scales with the normal-tissue HLA-low fraction, not LOH frequency.** Off-tumour toxicity floor =
   `P(HLA-low) × P(normal antigen > kill-cutoff)` = 5% × 30% = **1.5% ≈ measured 1.4%** (reconciles). FPR
   tracks the HLA-low rate 1:1 (0%→0%, 5%→1.4%, 20%→5.8%), robust across activator thresholds.
   **Honest caveat:** that "safety flows through the blocker" and "FPR is linear in HLA-low" are **true *by
   construction*** (the blocker is the only NOT arm) — *not* discoveries. What the model genuinely *computes*
   is (a) the proportionality constant (~0.30, set by the normal-antigen leak), (b) the **orthogonality** of
   the safety vs efficacy axes, (c) robustness to the activator threshold. The headline FPR is **conditional
   on the 5% HLA-low fraction — the one load-bearing, unsourced-for-normal-tissue parameter.**
2. **Efficacy is bounded by the clonal-LOH fraction.** TPR rises 19%→96% as clonal LOH goes 0.2→1.0 —
   subclonal HLA-retaining tumour cells keep the blocker on and escape. RUNG-6's clonal haircut, live.

**Two failure modes** = the recognition problem: **(A) false-kill** (normal cells downregulating the sensed
allele); **(B) escape** (antigen-low or subclonal HLA-retaining tumour cells).

## Honest framing
A **mechanistic circuit model**, not a measurement. Structure (orthogonal safety/efficacy bounds; two failure
routes) is parameter-robust; exact %s are parameter-dependent (sweeps included). *binding ≠ agonism* is the
wet-lab residual. **Audited by an independent agent** (2026-06): fixed a coarse-grid AUC bug (0.75→0.88),
fixed the floor formula (wrong antigen cutoff), demoted "safety carried entirely by the blocker" from finding
to by-construction, and corrected "bistable" → threshold commit.

**The next measurable thing it points to:** normal-tissue HLA heterogeneity — the real, unsourced safety
constraint — measurable on the scRNA atlas (Colab). Selftest 10/10. Log in `runs/logs/`.
