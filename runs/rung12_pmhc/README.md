# RUNG 12 — pMHC structural discriminability: measure β, certify relay targets

Closes the loop opened by the RUNG-12P bridge. That bridge showed a gated relay unlocks RUNG-11's
`tcr_dependent` neoantigen handles **if** the TCR's mut-vs-WT cross-bind `β` is low — but `β` was *swept*.
This rung **measures** a per-handle `β` and re-runs the bridge with it → a **ranked, certified target list**.

## How β is measured (discriminability D = 1 − β)
Probabilistic-OR of independent discrimination mechanisms (D high if *any* fires):
- **M — MHC-level** (from RUNG-11 NetMHCpan): WT binds much worse / not presented → WT pMHC rarely on the
  surface. `clean` tier → M = 1. *(robust, fold-independent)*
- **E·P — TCR-level**: the mutated residue is solvent-**E**xposed (RSA from the ESMFold-bound conformation)
  **and** physicochemically different (**P** = charge/volume/hydropathy delta). *(P robust; E best-effort)*
- **Z — sequence**: ESM-2 embedding distance between mutant and WT peptide, normalised across handles. *(robust)*

Then `q_n = presentation_factor(wt_rank) · β` → **per-cell-safe** (q_n ≤ 0.02) / **relay-safe** (q_n ≤ 0.17,
the RUNG-12P/B 3D ceiling). Re-running coverage with the measured β gives the certified usable target set.

## STRUCTURAL REDO — ColabFold (`notebooks/rung12_structure_colab.ipynb`)
After the ESMFold attempt failed (below), the structural arm was redone with **ColabFold / AlphaFold2-multimer**
(MSA-guided → actually docks the peptide; license-free; GPU). It folds the HLA α1α2 groove + peptide for the
**non-`clean` handles only** (clean → β=0 regardless of structure; 24 of 32 here), measures the mutated
residue's **TCR-facing exposure (RSA)**, and feeds the measured `E` into the β scoring. The RSA analysis was
**validated against a real HLA-A\*02:01 crystal (1HHK)**: it correctly reads buried anchors (P2/P9 ≈ 0.02) vs
TCR-facing residues (P4–6 ≈ 0.5). Run that notebook on **T4 GPU**, Run all; ~1.5–2.5 h, resumable.

---

## (Superseded) first attempt — `notebooks/rung12_pmhc_colab.ipynb` (ESMFold)
Open it, **Runtime → Run all**, same Google account as before. Stages:
1. **prep** — selects the top ~32 handles by prevalence (surfaces IDH1 R132H/glioma, KRAS, BRAF V600E),
   fetches HLA α1α2 grooves from IPD-IMGT/HLA, writes `groove:peptide` ESMFold inputs. *(validated locally)*
2. **ESM-2 embeddings** → per-handle Z. *(robust core — no MSA server)*
3. **ESMFold structures** → bound pMHC PDBs → RSA exposure E. **Best-effort**: ESMFold's openfold extras
   often fail to build on Colab and its peptide docking is the soft part; if it fails, the run still produces
   a valid β from M + P + Z (E falls back to a position prior). Resumable per handle.
4. **analyze** → `rung12_pmhc.json` + `rung12_pmhc.png` (ranked targets + measured-β bridge coverage).
Bundle with `python scripts/archive_colab_run.py --commit`.

## Result — STRUCTURE-CERTIFIED (real T4 run `b416b0b`, ColabFold 24/32 folded)
The structural arm **executed**: ColabFold/AF2-multimer folded **24 of 32** handles (all non-`clean`; the 8
clean ones are β=0 regardless), and the **measured TCR-facing exposure `E`** (RSA from the real poses) replaced
the position prior for every one. (The earlier ESMFold attempt failed to build — superseded.) Most poses are
confident (pep pLDDT 70–95); a few are low (47–54 → their exposure is less certain, flagged).

**The real poses make the answer SHARPER and more honest — and temper the bridge's optimism hard:**
- **9 per-cell-safe, 11 relay-safe, only 2 unlocked by the relay.** Most `tcr_dependent` handles fall to
  **risky** once the measured exposure + binding differential are in — the WT pMHC is genuinely cross-reactive.
- The **robustly-safe core is the `clean` handles** (WT not presented, structure-independent): they carry
  **PDAC 26%, glioma 22% (IDH1 R132H), melanoma 11%, CRC 11%**.
- **Structure *promoted* one handle:** `KRAS-G12D/C*08:01` (`C0501`) — the pose shows the G→D **buried**
  (E=0.04, pLDDT 78) *and* the WT binds far worse (M=0.85) → MHC-level discrimination → **per-cell-safe**
  (q_n=0.018). A genuinely clean target the sequence-only pass underrated.
- **Relay unlock survives only for melanoma: 10.8% → 19.1%** (BRAF V600E on A\*68:01 & C\*06:02 — exposed
  enough + big V→E change → relay-safe but not per-cell-safe). Elsewhere the marginal unlock is ~0.
- **The bridge's optimistic glioma/IDH1-R132H relay unlock did NOT survive structure:** those handles' mutated
  residue is buried with no binding differential (E low, M≈0) → genuinely hard → **risky**. Honest downgrade.
- **KRAS-G12D/C\*08:02** (the *proven* clinical TCR target) flags **risky** by the generic proxy (E=0.03 buried,
  q_n=0.30). That's the key honest limitation: a generic β **cannot** capture a specifically-engineered exquisite
  TCR — and the clinical win for this exact handle took years of dedicated TCR discovery. **"Risky by proxy" ≠
  "impossible" — it means "needs an exceptional TCR,"** which is precisely what reality required. The pipeline is
  a *prioritiser*, not a verdict.

**Takeaway:** structure didn't inflate the story — it disciplined it. The safe, generalisable targets are the
`clean` handles (tumour-exclusive *and* WT-off-MHC); the relay buys a real but modest melanoma extension; and
the famously hard `tcr_dependent` handles are correctly flagged hard. That is the honest map a wet lab needs.

## Honest ceiling
ColabFold/AF2-multimer docks pMHC-I reasonably (it gets the peptide register/anchors right — validated vs the
1HHK crystal), but a 1-residue mut/WT change barely moves the backbone, so the signal is the mutated residue's
**exposure**, not RMSD; low-pLDDT poses (a few here, 47–54) make their `E` less certain. **β is a proxy, not a
measured TCR Kd** — a generic structural+binding+physicochem β *cannot* model a specifically-engineered exquisite
TCR (see KRAS-G12D/C\*08:02, clinically real yet proxy-risky). A top-ranked handle is a **prioritised hypothesis
for wet-lab TCR isolation**, not a validated target. Inherits RUNG-11 (population frequencies, mRNA→presentation)
and RUNG-12P (percolation-abstraction relay ceiling) caveats.

## Provenance
`scripts/37_pmhc_discriminability.py` (selftest 16/16; `prep` validated end-to-end against IPD-IMGT/HLA —
32/32 grooves). ESM-2 (`fair-esm`), ESMFold (best-effort), Biopython SASA. HLA grooves cached in
`data/refs/{A,B,C}_prot.fasta`. Consumes `runs/rung11_neoantigen/…` + `runs/rung12pB_relay/…`. **Next:** the
top-ranked relay-safe handles become the wet-lab shortlist; β can later be replaced by measured TCR affinities.
