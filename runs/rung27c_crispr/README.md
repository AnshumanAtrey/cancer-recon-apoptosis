# RUNG-27c — CRISPR rescue for the wobble drivers (testing RUNG-27b's DNA-sensable assertion)

**Why.** RUNG-27b set `dna_sensable = True` for every driver ("CRISPR has no wobble"). That was an
*assertion*. RUNG-27c **tests it**: for each G>A-wobble driver the RNA toehold can't discriminate, it scans
the real CDS sequence for a SpCas9 (NGG) allele-specific guide — either a **PAM-creating** SNV (mutant makes
an NGG the WT lacks = perfect allele specificity) or an existing PAM with the **SNV in the PAM-proximal seed**
(WT then carries a seed mismatch that collapses Cas9 activity) — and **designs the actual guide**.

Selftest 6/6 (PAM-create, seed, antisense, and no-PAM cases all validated). The first run returned a false
0/7 because the window was too short to hold a 20-nt guide; caught by the selftest, fixed to a ±32-nt window.

## Result: 5/7 wobble drivers are CRISPR-rescuable (SpCas9-NGG)

| driver | mechanism | seed pos (from PAM) | 20-nt guide | strand |
|---|---|---|---|---|
| **KRAS-G12D** | SEED | **1** (deepest — ideal) | `CTTGTGGTAGTTGGAGCTGA` | + |
| KRAS-G13D | SEED | 4 | `GTAGTTGGAGCTGGTGACGT` | + |
| IDH1-R132H | SEED | 10 (seed edge) | `ATCATAGGTCATCATGCTTA` | + |
| PIK3CA-E545K | SEED | 4 | `TCTCTCTGAAATCACTAAGC` | + |
| TP53-R248Q | SEED | 2 | `GCATGGGCGGCATGAACCAG` | + |
| TP53-R175H | **none** | — | — | — |
| TP53-R273H | DISTAL | 16 (too far) | — | — |

**Biologically consistent:** every rescue is a SEED guide, *zero* PAM-creating — because G>A transitions
*destroy* a G (can't make a new `GG`), so allele specificity comes from a nearby existing PAM, not PAM
creation. KRAS-G12D landing at seed position 1 (the base immediately PAM-adjacent) is the strongest possible
single-mismatch discrimination.

## What this does to the story (rule-5 refinement)
RUNG-27b's blanket "all DNA-sensable" was **optimistic**: with SpCas9-NGG, **5/7 (71%)** of the wobble drivers
are cleanly addressable, not 7/7. The two TP53 hotspots (R175H, R273H) need a relaxed-PAM Cas (SpCas9-NG),
Cas12a (TTTV), or intron-aware genomic context to recover a PAM. So RUNG-27b's per-cancer RNA+DNA coverage is
slightly optimistic where it leaned on those two TP53 hotspots — but TP53-R248Q rescues, and the alternative
Cas enzymes very likely close the gap. **The DNA rescue is real and designed — just not 100% with one enzyme.**

The flagship drivers the whole arc kept hitting — **KRAS-G12D and IDH1-R132H** (the ones RNA toehold and
de novo binder both failed on) — now have **concrete allele-specific guides** for an MHC-free autonomous
self-destruct.

## Ceiling (rule 3/5)
- Context is **CDS-local** (Ensembl MANE, U→T); true targeting includes introns within ±32 bp of exon-edge
  hotspots → SEED calls near an exon edge need genomic confirmation; PAM-creating calls are robust.
- A PAM + seed position is **necessary, not sufficient** — real on-/off-target cutting is a wet-lab residual.
- SpCas9-NGG only here; SpCas9-NG / Cas12a would raise addressability (noted, not yet scanned).

*Result: `rung27c_crispr.json` (guides per driver). Script + selftest: `scripts/55_crispr_rescue.py`.*
