# cancer-recon-apoptosis — Honest Project Assessment

## Will it succeed?

Four tiers of "success" with very different odds:

| Tier | Definition | Odds |
|---|---|---|
| **1. Publishable paper** | Ship a paper like PharmaRL — infra contribution + held-out eval + open artifacts | **~85%** |
| **2. Plausible candidates** | Trained policy emits designs that predicted metrics say should work | **~50%** |
| **3. Wet-lab validation** | Someone synthesizes a candidate and it actually kills cancer cells in a dish | **~5%** (needs wet-lab partner) |
| **4. Real treatment** | Becomes a drug humans take | **<1%** (decade horizon) |

**Targeting Tier 1 with a credible Tier 2 argument.** Beyond that needs a wet-lab — separate paper later.

## What it will and won't prove

### WILL prove
- An RL policy with a *specificity-aware oracle* (cancer-receptor binding minus normal-receptor binding) produces better designs than baseline language-model sampling.
- The open-source structure-prediction stack (Boltz-2 + Protenix + CellChat) is sufficient infrastructure for cancer-target design — no Google dependency needed.
- Cancer-cell-restricted ligand-receptor pairs can be systematically discovered and computationally engineered for.
- A reusable environment any future team can plug a new policy/oracle into.
- With the simulation tier (Phase 4): an end-to-end in silico pipeline from sequence design → predicted apoptosis cascade → predicted bystander propagation → ADMET filtering produces measurably better survival rates than baseline.

### WON'T prove
- Any designed molecule actually kills cancer cells in a Petri dish (no wet-lab).
- The bystander cascade actually propagates through real tumors (needs animal model).
- It's better than chemo / immunotherapy (needs clinical data).

Same honest scope as PharmaRL. **Infrastructure + reference recipe + held-out delta = real contribution.** Don't promise more.

---

## Risk audit

### 1. Oracle saturation (HIGH)
Boltz-2 might not separate cancer-receptor binding from normal-receptor binding well, because receptor sequences are 95%+ identical between cancer and normal variants.
**Mitigation:** Lean on differential *expression* (CellChat) as the specificity axis, not just binding affinity. Use cancer cell lines where the receptor is overexpressed, not necessarily structurally unique.

### 2. Cascade not captured at oracle level (MEDIUM)
Single-binding rewards don't model gap-junction propagation.
**Mitigation:** Phase 4 (PhysiCell+PhysiBoSS) explicitly models the cascade. Scope the paper to "design of bystander-trigger ligands" not "modeling of bystander cascade."

### 3. Reward hacking (MEDIUM)
Designs that score great on Boltz-2 but are biological nonsense (Renz 2020 documents this for QSAR/QED).
**Mitigation:** ESM-3 perplexity as foldability sanity check (same role KL anchor plays in PharmaRL). Filter out unrealistic candidates pre-evaluation.

### 4. Compute cost (MEDIUM)
Boltz-2 takes 30s–2min per prediction; RL loop with thousands of rollouts gets expensive.
**Mitigation:** Distill Boltz-2 into a cheaper surrogate after warmup, or cache predictions aggressively. Single A10G GPU budget like PharmaRL.

### 5. Baker Lab Oct 2025 close prior art (MEDIUM)
Their "De Novo Protein Design Targeting Oncogenic Interfaces" is structurally similar work.
**Mitigation:** Differentiate clearly on (a) RL loop they don't have, (b) bystander/cell-cell framing they don't have, (c) cancer-vs-normal specificity reward they don't have, (d) end-to-end simulation tier they don't have.

### 6. Geopolitical (LOW)
Chinese tool stack (Protenix/HelixFold-3) could face export restrictions.
**Mitigation:** Use Boltz-2 (MIT lab + Recursion) as primary; Chinese tools as backup. Multiple oracles makes the project resilient.

### 7. M2 8GB local compute (LOW)
Boltz-2 full model needs ~16GB RAM and GPU. Local M2 8GB is for development only.
**Mitigation:** Cloud GPU (A10G/A100, same as PharmaRL) for training + heavy inference. Local development uses ESM-2 small variants and HF Space remote calls.

---

## Day-1 kill criteria

**If Step 1 (Boltz-2 smoke test) shows DR5+DR5B and DR5+scrambled gap < 1 kcal/mol after proper setup**, the binding-affinity-only oracle is too noisy. Pivot options:
- Switch to absolute binding free energy (ABFE) via OpenFE + Boltz-2 ensemble (Recursion Oct 2025 pipeline).
- Use AlphaFold 3 Server as primary oracle with Boltz-2 as cross-check.
- Add explicit MD-refinement step (OpenMM 5ns simulation post-Boltz-2 prediction).

**If pivot still doesn't separate signal from noise**, reconsider the project — the design problem may be fundamentally harder than PharmaRL.

---

## What this project is NOT

- **Not a cure for cancer.** Computational design pipeline only.
- **Not a wet-lab paper.** No mouse, no Petri dish, no IRB.
- **Not a novel-biology paper.** All underlying biology (bystander effect, DR5, apoptosis cascade) is known. Contribution is computational integration.
- **Not a drug discovery paper in the pharma sense.** No IND, no clinical pathway. Designs are concept-stage.

---

## What this project IS

- An **open computational substrate** for a new class of design problems: cancer-specific cell-cell signaling.
- A **reference recipe** for combining structure-prediction + RL + multi-scale biological simulation.
- A **stress test** of the open-source biology AI stack (Boltz-2, ESM-3, PhysiCell, ADMET-AI) at hackathon-scale compute.
- A **follow-up vehicle** to PharmaRL — same authors, same infra muscles, new domain.
