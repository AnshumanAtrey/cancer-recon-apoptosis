# Evidence ceiling & wet-lab handoff — what this project can and cannot claim

Fact-checked (adversarially verified, May 2026). This file keeps us honest: it defines the
project's real ceiling so we never drift into "in-silico discovery that skips the bench."

## The hard truth (verified)

- **Wet-lab cannot be skipped.** Zero fully AI-designed (de novo) drugs are approved by any
  regulator as of May 2026; no therapeutic has ever reached patients by pure computation.
  `skipped_wetlab = FALSE` for 100% of documented AI/in-silico drug examples.
- **The "virtual cell" is aspirational (~a decade out).** *How to build the virtual cell with AI*
  (Bunne, Quake et al., Cell 2024) is a roadmap, not a system. Foundation models (scGPT,
  scFoundation, GEARS) do **not** beat simple linear baselines on unseen perturbations
  (Ahlmann-Eltze, Huber & Anders, *Nature Methods* 2025). Arc's "State" beats linear baselines on
  genetic knockdowns but not consistently across all metrics; drug response is harder. Mechanistic
  whole-cell models exist only for the simplest organism (Karr 2012, *M. genitalium*, 401 genes).
- **AI de-risks chemistry, not biology.** AI-discovered drugs: Phase I ~80–90% (vs ~40–65% historic),
  Phase II ~**40%** — same as traditional (Jayatunga et al., *Drug Discovery Today* 2024; small sample).
- **Track record:** best case = rentosertib/INS018_055 (Insilico, IPF) — Phase IIa positive
  (*Nature Medicine* 2025), Phase III *company-stated on track* for H2 2026, **not approved, not
  initiated**. Only AI-touched *approved* molecule = baricitinib (COVID, 2022) — **repurposing** of an
  existing drug (AI generated the hypothesis only; efficacy from conventional RCTs), not de novo design.
  Cautionary failures: DSP-1181 (Exscientia/Sumitomo, discontinued ~Phase I), BEN-2293 (BenevolentAI,
  missed Phase IIa efficacy 2023), REC-994 (Recursion, discontinued May 2025).

## What we CAN trust in-silico (act on)

| signal | trust | use |
|---|---|---|
| AlphaFold2/3 structure (ordered domains) | Tier A | starting model. Caveat: one static conformer; no physics; fails IDRs/novel folds |
| Boltz-2 affinity (~0.62 Pearson; preprint) | Tier B | **ranking only** ("A tighter than B"), never a calibrated Kd; small-molecule head only |
| Our two-axis interface filter (iPLDDT≥0.70 ∧ iPAE≤15Å) | proxy | correct mitigation for confident-non-binder failure we hit; still needs wet-lab |
| Computational **NO/FAIL** (unfoldable, no-bind, hERG+ AUROC~0.94) | strong | use as a **negative filter** to deprioritize — defensible |
| PySB/EARM, PhysiCell | logic | **mechanism prototyping** + parameter-sensitivity; hypothesis generation, not efficacy |

## What we CANNOT claim (intrinsically experimental)

- **Binding → function (agonism).** No score distinguishes a DR5-**clustering** agonist from an inert
  binder. **This is our scientific crux and the #1 wet-lab experiment.**
- Cancer-vs-healthy **selectivity** in real tissue.
- Cell-level **efficacy**; whether the apoptosis wave **stays contained** vs runs into healthy tissue.
- In-vivo toxicity, immunogenicity, PK/PD, delivery, patient heterogeneity.
- That the **biology is correct in a human** (the Phase II wall — where AI helps ~0).
- A "discovery" or "validated therapy" from simulation alone.

## The project's honest ceiling = three deliverables

1. A mechanistically-coherent **novel hypothesis** (recognition-gated self-propagating apoptosis),
   characterized in PySB/PhysiCell with stated assumptions.
2. A **de-risked, prioritized shortlist** (structure + Boltz ranking + two-axis filter + ADMET
   negative-filter), explicitly labeled as proxies.
3. A **concrete wet-lab handoff**: specific constructs, cell lines, and assays — centred on the
   cheapest experiment that tests the **agonism crux** (DR5 clustering / caspase-8 activation),
   plus selectivity (cancer vs normal) and wave-containment.

## The path to "real" (reachable, not blocked)

Computation is the **funnel**, the bench is the **proof** — exactly how Insilico/Baker operate. The
bench is accessible without owning a lab: **cloud labs (Emerald Cloud Lab, Strateos) run real
experiments via code**; the FDA's April 2025 NAMs roadmap pushes in-silico + organ-on-chip. So the
handoff is a real next step, not a dead end. Framing as "de-risking + hypothesis + defined handoff"
is credible and grant-survivable; "in-silico discovery that avoids the lab" is false and gets rejected.
