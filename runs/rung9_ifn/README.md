# RUNG 9 — does IFN rescue the blocker? (HLA-A vs the interferon program)

**The question that decides whether RUNG-8's safety hole is real in vivo.** RUNG-8 found the Tmod blocker's
antigen (HLA-A) is low *at rest* in immune-privileged vital tissue (cardiac conduction, cardiomyocytes,
neurons) → the blocker fails there. But CAR-T therapy induces **IFN-γ, which upregulates MHC-I**. So: when the
interferon program is ON in those cells, is HLA-A turned back on (blocker rescued)?

## How to run
Open **`notebooks/rung9_ifn_inducibility_colab.ipynb`** in Colab (**CPU runtime**), run top to bottom.
Outputs: `rung9_ifn_inducibility.json` + `rung9_ifn.png`. Bundle filed with
`python scripts/archive_colab_run.py --commit`.

## What it computes
Per vital cell type, fetch HLA-A/B/C + a core ISG panel (STAT1, IRF1, GBP1, IFIT1/3, ISG15, MX1, OAS1, TAP1,
IRF7) + a housekeeping panel (ACTB, GAPDH, RPL13A, TMSB4X). Per cell: **ISG-score** (#ISGs detected = IFN
activity) and **HK-score** (#housekeeping detected = a depth proxy). Stratify into **IFN-low** vs **IFN-high**
and compare HLA-A-low (LOWER = among MHC-I-detected, depth-controlled; UPPER = all cells), among
HLA-measuring datasets only (RUNG-8's fix).

**The honesty control:** IFN-high cells are also deeper-sequenced, which alone would raise HLA-A. We report
HK-score per stratum and set `hk_matched` — a "rescue" is only believed when the two strata have similar depth.

## Deliverable
`ifn_rescue_delta_lower` per immune-privileged type (HLA-A-low drop from IFN-low → IFN-high). A large,
**depth-matched** drop ⇒ IFN re-arms the blocker ⇒ RUNG-8's hole shrinks in the inflamed therapeutic context.
A small drop ⇒ the hole is real even under inflammation ⇒ the genetic NOT-gate is unsafe in heart/brain. The
JSON `VERDICT` field states the call.

## First real run (2026-06, `colab_runs/20260605T204104Z_72eac5a/`) — and the honest read
699,530 cells, 11 datasets dropped as non-HLA-measuring. **The controls did their job:** in every two-stratum
tissue (10/10) IFN-high cells have *lower* HLA-A-low (direction consistent with IFN inducing MHC-I), and in the
one **depth-matched** tissue — pancreatic islet — IFN rescues strongly (17%→1%). **But for the tissues that
matter the test is INCONCLUSIVE:** cardiac conduction had only **9 IFN-high cells** in the entire resting atlas
(immune-privileged tissue barely runs IFN at baseline), and cardiomyocyte/neuron comparisons are
**depth-confounded** (`hk_matched=False` — IFN-high cells are just deeper-sequenced). So: **IFN does upregulate
HLA-A where it is active, but whether *therapeutic* IFN reaches and re-arms the blocker in heart/brain is a
WET-LAB question the resting atlas cannot settle.** (The v1 auto-VERDICT one-liner overstated this as "hole is
real"; the verdict logic was corrected to report INCONCLUSIVE + the directional/depth-matched evidence. A free
re-run of Cell 5 regenerates the corrected VERDICT string; the per-type data was already correct.)

Next refinement if needed: **depth-matched subsampling** (match IFN-high/low cells on housekeeping score before
comparing) to get a clean immune-privileged answer; or IFN-stimulated / surface-protein data (wet lab).

## Honest ceiling
mRNA ≠ surface protein; ISG-score is a proxy for IFN exposure (resting atlas has few IFN-high cells → that
stratum can be small/noisy, n reported); depth confound controlled by housekeeping genes but not perfectly
(`hk_matched`); HLA-A gene not the A\*02 allele; atlas IFN-high cells reflect the donor's inflammation, not the
exact CAR-T IFN dose. Definitive answer needs IFN-stimulated tissue / surface-protein assay (wet lab).

## Provenance
`scripts/30_hla_ifn_inducibility.py` (selftest 9/9, incl. the depth-confound flag). Census `2024-07-01`.
Reuses RUNG-8's resumable Drive-tile + heartbeat + Arrow-dictionary memory-safe machinery. Logs in `runs/logs/`.
