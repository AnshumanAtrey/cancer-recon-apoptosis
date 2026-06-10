# RUNG-27a — Reality-grounding the predicted neoantigen handles

**Question.** Our whole recognition arc rests on *predicted* presentation (MHCflurry) and a *predicted*
TCR-recognition propensity. CAPSTONE.md names two of these as irreducible wet-lab residuals:
(#1) is the peptide *really* presented / a real epitope? and (#2) does a real cognate **TCR** exist
(the MAGE-A3 question)? This rung asks two curated *experimental* databases whether our exact handles
have already been seen — converting those residuals from assumptions into **measured facts** where the
data exists, and an honest *predicted-only* where it does not.

**Data (public, not committed — re-downloadable; see `scripts/53_reality_grounding.py`):**
- **VDJdb** 2026-06-03 — 182,794 class-I human TCR–epitope pairs (real receptors).
- **IEDB** epitope_full v3 — 2,671,330 catalogued epitope records (real assays).
- Handles: the 32 structure-certified handles from `runs/rung12_pmhc/rung12_manifest.json`.

Matcher selftested 15/15 (known-true KRAS-G12D + viral controls match; scrambles + wrong-gene hits do not).

## Result (allele-aware — a TCR recognises a peptide-**MHC complex**, not a bare peptide)

| verdict | n | meaning |
|---|---|---|
| **GROUNDED_TCR** | **4** | real cognate TCR exists for **this exact pMHC** (allele matches) — strongest de-risk |
| GROUNDED_EPITOPE_OTHER_ALLELE | 8 | peptide is a real TCR-validated neoantigen, but on a **different restriction** than we predicted |
| GROUNDED_EPITOPE | 6 | IEDB-catalogued / presented, but no TCR on record |
| GROUNDED_REGISTER | 5 | a TCR exists for a register-variant of the same driver mutation |
| PREDICTED_ONLY | 9 | not yet in these DBs (DBs are incomplete → **not** a negative) |

**GROUNDED_TCR (real cognate TCR for the exact pMHC):**
- `KRAS_G12D_A1101` — VVVGADGVGK / A\*11:01 (14 TCRs)
- `KRAS_G12D_C0802` — GADGVGKSAL / C\*08:02 (7 TCRs)
- `KRAS_G12V_A0301` — VVVGAVGVGK / A\*03:01 (15 TCRs)
- `KRAS_G12V_A1101` — VVVGAVGVGK / A\*11:01 (15 TCRs)

**Reality disciplines the prediction.** 8 KRAS handles we predicted on alleles like A\*01:01 / A\*26:01 /
B\*35:01 carry the *real* peptide but the *real* TCRs are restricted to A\*11:01 / C\*08:02 — i.e. our
broader multi-allele MHCflurry calls are optimism not yet validated (the same way structure disciplined
RUNG-12).

**Safety, now measured not assumed.** The WT self-peptide VVVGAGGVGK is itself gene-confirmed in VDJdb on
A\*11:01 (2 TCRs) — so for `KRAS_G12D_A1101` and `KRAS_G12V_A1101` a real TCR exists against the *self*
peptide on the same restriction → the mut-vs-WT discrimination is a genuine, data-grounded crux (the
MAGE-A3 failure mode), not a hypothetical.

## The unifying insight — this tells us *where de novo design is needed*

- **KRAS-G12D / A\*11:01** and **KRAS-G12V** already have natural cognate TCRs → designing a binder de novo
  there is **redundant** (nature solved it; the engineering task is TCR-T manufacturing, not discovery).
- **IDH1-R132H** (RUNG-26 target #1) is `PREDICTED_ONLY` on every allele → **no natural class-I TCR exists**
  → de novo design is exactly the right tool. *The binder run on account #1 is validated by this.*
- **BRAF-V600E** (melanoma's ~50% driver) is `GROUNDED_EPITOPE` (real, IEDB-catalogued, presented) but has
  **no catalogued TCR** → de novo design fills a real gap → **RUNG-26 target #2** (`notebooks/binder_design_braf_colab.ipynb`).

## Ceiling (stated, not papered over)
- A MATCH = a real receptor/epitope exists (strong de-risk). **NO match = the DB is incomplete, reported as
  PREDICTED_ONLY, never as a negative.** VDJdb/IEDB are biased toward well-studied (esp. viral) epitopes.
- `antigen.gene` confirmation guards coincidental sequence hits; register-core hits are weaker than exact.
- A WT match = the germline self-peptide is also recognised = cross-reactivity *context*, not proof of harm.

*Result JSON: `grounding.json`. Script + selftest: `scripts/53_reality_grounding.py`.*
