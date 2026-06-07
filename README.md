# cancer-recog-apoptosis — Cancer-Cell Recognition-Triggered Apoptosis

Computational design pipeline for cancer-specific signaling ligands that make cancer cells recognize their cancer neighbors as abnormal and trigger apoptosis — operationalizing Shriya Rai's conceptual proposal.

Builds on Shriya Rai's conceptual paper *"Inducing Self-Destruction in Cancer Cells Through Altered Cellular Communication Signaling"* (Apr 2026) and the methodological pattern from PharmaRL (Anshuman et al., Apr 2026, Meta PyTorch OpenEnv Hackathon).

**Shriya's core concept (her own words, §3):** *"introducing or modifying signaling pathways that allow cancer cells to recognize neighboring cancer cells as abnormal, trigger intracellular stress or recognition mechanisms, and ultimately activate apoptosis internally."*

The bystander apoptosis cascade is one engineerable mechanism we use; the broader project name reflects her recognition-triggered framing, not just the bystander angle.

---

## Status

- [x] Phase 0 — Project scaffolding
- [ ] Step 1 — Boltz-2 oracle smoke test (this week)
- [ ] Step 2 — CellChat cancer-restricted target shortlist
- [ ] Step 3 — Specificity differential audit
- [ ] (rest of 19-step plan — see [PLAN.md](docs/PLAN.md))

---

## Quick map

```
cancer-treatment/
├── README.md                       this file
├── CLAUDE.md                       project working rules
├── THESIS.md                       core thesis / running synthesis
├── Conceptual Research(1).pdf      Shriya's source paper
├── docs/                           PLAN, ASSESSMENT, EVIDENCE_AND_HANDOFF, RELATED_WORK + methodology/
├── src/
│   ├── oracles/                    Boltz-2, ESM, composite reward
│   ├── env/                        OpenEnv-spec environment (Phase 2)
│   ├── policy/                     GRPO + LoRA recipe (Phase 2)
│   └── simulation/                 PySB + PhysiCell + ADMET (Phase 4)
├── scripts/
│   ├── 01_boltz_smoketest.py       cloud-ready full Step-1 test
│   ├── 01_local_smoketest.py       M2-compatible ESM-2 proxy version
│   ├── 02_cellchat_targets.R       Step 2 (later)
│   └── 03_specificity_check.py     Step 3 (later)
├── data/
│   ├── sequences/                  reference FASTAs (DR5, DR5-B, scrambled)
│   ├── targets/                    cancer-restricted L-R shortlist (Phase 0)
│   └── refs/                       calibration complexes
├── runs/                           training + eval outputs (gitignored)
├── notebooks/                      exploration
├── requirements.txt                local-runnable deps (M2-compatible)
└── requirements-cloud.txt          heavy deps for cloud GPU
```

---

## Quickstart (M2 / local development)

```bash
cd /Users/atrey/Desktop/code/side-claude-income/research-papers/cancer-treatment

# install lightweight local deps (no GPU required)
pip install -r requirements.txt

# run the M2-compatible smoke test (ESM-2 small variant)
python scripts/01_local_smoketest.py
```

Expected output: cosine-distance / interface-contact metric showing DR5+DR5-B has signal vs DR5+scrambled.

## Quickstart (cloud GPU — A10G or A100)

```bash
# install full deps
pip install -r requirements-cloud.txt

# run the real Boltz-2 oracle test
python scripts/01_boltz_smoketest.py
```

Expected output: ΔG(DR5+DR5-B) − ΔG(DR5+scrambled) ≥ 2 kcal/mol. If not, see docs/ASSESSMENT.md kill criteria.

---

## Reading order if joining the project

1. [Conceptual Research(1).pdf](Conceptual%20Research(1).pdf) — Shriya's idea, 8 pages
2. [PLAN.md](docs/PLAN.md) — 19 steps, what we're building, why
3. [ASSESSMENT.md](docs/ASSESSMENT.md) — honest odds, risks, day-1 kill criteria
4. [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — 16 papers we cite
5. [scripts/01_local_smoketest.py](scripts/01_local_smoketest.py) — runs today

---

## Headline contributions (target paper)

1. First OpenEnv-native chemistry/biology environment for cancer-specific ligand design with a *specificity-aware oracle* (cancer-receptor binding minus normal-receptor binding).
2. Reference GRPO recipe ported from PharmaRL to peptide/protein design space with held-out cancer-type generalization.
3. End-to-end in silico evaluation pipeline: design → binding → apoptosis cascade (PySB+EARM) → tissue bystander propagation (PhysiCell+PhysiBoSS) → ADMET filter. Reports cumulative survival rate as the load-bearing metric.

---

## 🧬 Hypothesis catalog — every way we can make cancer die (living list)

The running arsenal. We do **not** marry one mechanism — we sweep many and let the data say which *hits*. Each
entry: the idea (one line), the biological basis, how we test it **with the tools we already have** (atlas /
sequence / structure / simulation — no wet lab), and status. Many of the dynamical ones run head-to-head in the
**mechanism arena** (`scripts/39_mechanism_arena.py`, `notebooks/mechanism_arena_colab.ipynb`) which ranks them
by *safe & effective across regimes* (clears tumour ≥80%, spares normal ≤1%). **RUNG-15**
(`scripts/40_atlas_mechanism_map.py`) then maps the winners onto **real per-cancer addressability** — neoantigen
axis (instant, from RUNG-12 handles) + surface axis (CELLxGENE Census).

> **Leaderboard (RUNG-14, 9 strategies × 12 regimes):** `quorum` HITS (8/12) · `wave`/`alt_death`/`combo`/`ferroptosis_wave`
> CLOSE (4/12) · `synthetic_lethal` SAFE-but-coverage-limited (12/12 safe, <80% kill alone) · `per_cell`/`diffusible`/`oncolytic`
> FAR (leak at high q_n). **RUNG-15 map:** propagation unlocks melanoma (+8%) & CRC; clean neoantigens (PDAC 26%, glioma 22%)
> need no propagation; quorum's headroom needs surface markers (→ the Census arm).
>
> **RUNG-16 (clonal neoantigen burden, MHCflurry):** the *shared hotspot* ceiling (RUNG-11) is bounded (3–20%), but with
> the **full per-patient clonal mutation repertoire**, the fraction of patients with ≥1 clean tumour-exclusive handle to
> **seed the wave** is *much* higher for high-mutation cancers — at the conservative 1–5% neoantigen yield: **MSI-CRC 99–100%,
> melanoma 81–100%, NSCLC/bladder 64–99%**, down to **PDAC 20–68%, breast 18–64%** (TMB-graded). Caveat: this is **personalised**
> addressability (per-patient neoantigen ID + effector), not off-the-shelf; presentation ≠ killing (TCR = wet-lab residual).
>
> **RUNG-17 (binding axis — does a T-cell recognise the clean handles?, MHCflurry+BLOSUM):** scored every clean neoantigen
> handle for TCR-recognition propensity (agretopicity ⊕ foreignness ⊕ TCR-contact hydrophobicity). **Key finding: safety and
> immunogenicity ALIGN** — a clean (tumour-exclusive) handle is *automatically* high on agretopicity (the dominant validated
> immunogenicity driver), so all 29 clean handles sit HIGH/MED propensity. Axis validated: the famous *hard* clinical targets
> (KRAS-G12D/TP53-R175H — needed engineered TCRs) correctly fall **below** the clean median (clean DAI 1.06 vs hard-clinical 0.69).
> Top screen priorities: CTNNB1-S37F, EGFR-L858R, TP53-R248Q. Caveat: PREDICTED propensity, a prioritisation for TCR discovery —
> **not** proof a receptor exists (MAGE-A3: high prediction, fatal cross-reactivity).
>
> **RUNG-15 census (real CELLxGENE, 25 tumour antigens): DECISIVE NEGATIVE — 0/25 single surface markers are safe under any mechanism. Specificity
> anti-correlation: tumour-high markers (EPCAM/TROP2/MUC1/HER2/EGFR) leak into vital tissue (q_n 0.8–1.0); vital-clean
> markers (CD20/PSMA/CD19) are barely tumour-expressed (q_t <0.05). → tumour-exclusivity must come from MUTATION
> (neoantigen) or COMBINATORIAL AND-NOT logic, not a single shared self-antigen. (Pan-cancer pooled; a per-cancer marker may still win.)**
>
> **RUNG-18 (does the cancer cell keep its MHC-I "display window" ON? — 6,319 real WGS tumours, Martínez-Jiménez 2023):** the
> immune route silently assumed the window is lit; this MEASURES it and GRADES the failure. **Across all tumours: window genetically
> INTACT 77.9% · DIMMED (HLA-LOH/mut — one allele lost, *route survives*, still presents on the rest) 18.4% · fully DARK (systemic
> B2M/TAP/NLRC5 — *route dies*, no surface MHC-I at all) only 3.7%** (IFN-blind, can't re-light: 4.0%). In our route cancers the
> full-dark fraction stays **<10% everywhere** (melanoma 8.4% — highest, it's the most immune-edited; NSCLC 6.0%, CRC 5.7%, bladder
> 3.1%). Driver of the dark cases is **B2M** (mut+del = 112/232), exactly as biology predicts. **Verdict: the immune route is
> GENETICALLY viable — the window is mostly there.** Caveat (load-bearing): GENETIC only — epigenetic/transcriptional MHC-I silencing
> is NOT captured, so 3.7% is a **FLOOR** on fully-dark, not the total; patient-level not clonal. The reversible (epigenetic) arm is
> RUNG-9 territory; the next test is **RUNG-18b — single-cell HLA/B2M *transcription* in malignant cells (Colab)**. Where the window
> is genetically dark *and* IFN-blind, only an MHC-independent killer (NK-engager / **Shriya's original autonomous self-destruct**) works.

**Status legend** — ✅ built + tested · 🟢 testable now with our tools · 🔮 future (physics/delivery, kept safe)
**The one rule:** every "kill" claim is a HYPOTHESIS with a stated wet-lab residual. β / kill% are proxies, never verdicts.

### Tier A — Recognition: *which* cell to kill (the address)
- **A1 Expression logic** — gate on which genes are ON/OFF (surfaceome AND/AND-NOT). *Basis: targeted therapy.* `Test:` CELLxGENE addressability. **Status: ✅ RUNG 5–10b — bounded (HLA-LOH ceiling ~14–28%, deployed gates 3–6%).**
- **A2 Neoantigen / sequence** — *is this protein MUTATED?* Tumour-exclusive by construction. *Basis: MANA/neoantigen immunology.* `Test:` MHCflurry presentation. **Status: ✅ RUNG-11 — out-reaches expression; oracle 3/3; GOLD handles beat the ceiling.**
- **A3 Structure / pMHC** — does the mutation actually face the TCR on the MHC groove? `Test:` AlphaFold2-multimer + measured RSA. **Status: ✅ RUNG-12 — 24/32 folded; clean handles certified; honest negatives kept.**
- **A4 Multi-input AND-NOT / HLA-LOH NOT-gate** — fire only on (markerX AND NOT markerY); use tumour's own MHC loss as a NOT signal. `Test:` atlas addressability. **Status: ✅ RUNG-6.**

### Tier B — Self-recognition & propagation: *one cell tells its neighbour* (Shriya's core + my idea)
- **B1 Bystander death wave** — seed a few, let death spread cell→cell; per-hop recognition-gated. *Basis: HSV-TK bystander, synNotch.* `Test:` coupled-EARM lattice. **Status: ✅ RUNG-13 — bounded wave, confirms percolation, resistance-resistant.**
- **B2 Quorum / density gate** — die only where the *local density* of recognised cells is high (cancer is clonal & dense; scattered normal false-positives lack quorum). *Basis: bacterial quorum sensing ported to mammalian synthetic circuits.* `Test:` arena `quorum`. **Status: 🟢 arena — currently the LEADER (spares isolated false-positives entirely).**
- **B3 Diffusible-factor relay (GDEPT)** — dying cell releases a *diffusing* death factor (HSV-TK/GCV, cytosine-deaminase/5-FC). *Basis: the bystander route that does NOT need gap junctions — relevant because tumours barely gap-couple (RUNG-12P/A, Cx43 in ~6%).* `Test:` arena `diffusible` (reaction-diffusion). **Status: 🟢 arena — leaks at high q_n unless gated tighter.**
- **B4 Oncolytic self-amplifying signal** — the death trigger *replicates* in tumour cells before firing (oncolytic-virus / self-amplifying-RNA analogue) → super-critical spread in tumour, dies out in normal. `Test:` arena `oncolytic`. **Status: 🟢 arena.**
- **B5 Contact synNotch relay chain** — juxtacrine hand-off: cell fires only on direct contact with a recognised neighbour. `Test:` arena (wave variant). **Status: 🟢.**

### Tier C — Alternative death modalities: *beat apoptosis-resistance* (Shriya §6.3, JJK "Domain that ignores your cursed technique")
- **C1 Ferroptosis** — force iron-dependent lipid peroxidation (GPX4 inhibition). *Basis: apoptosis-resistant cancers stay ferroptosis-sensitive; ferroptosis propagates cell-to-cell (Riegman 2020).* `Test:` arena `ferroptosis_wave` + atlas GPX4/SLC7A11/ACSL4. **Status: ✅ arena `ferroptosis_wave` (brake-independent propagating death, immune to apoptotic resistance — CLOSE, 4/12) + 🟢 atlas.**
- **C2 Pyroptosis** — gasdermin pore (GSDMD/GSDME), inflammatory, converts "cold" tumours. `Test:` atlas + arena `alt_death`. **Status: 🟢.**
- **C3 Necroptosis** — RIPK3/MLKL route for caspase-8-deficient tumours. `Test:` atlas. **Status: 🟢.**
- **C4 Cuproptosis** — copper-driven death via lipoylated TCA enzymes (FDX1). `Test:` atlas FDX1/lipoylation. **Status: 🟢.**
- **C5 Death-pathway addressability map** — *which* death route is intact-in-tumour but safe-in-normal, per cancer type. `Test:` meta-scan across CELLxGENE. **Status: 🟢 (high-value next port).**
- **C-arena alt_death** — reroute the apoptosis-incompetent subclone to a brake-independent effector. **Status: ✅ in arena — rescues clearance when half the tumour is apoptosis-proof.**

### Tier D — Lower the death threshold: *prime, then push*
- **D1 BH3-mimetic threshold-lowering** — BCL2/MCL1/BCL-XL dependence makes tumours "primed to die"; lower S* selectively (venetoclax logic). `Test:` atlas dependence + arena `combo`. **Status: 🟢 / ✅ arena `combo`.**
- **D2 In-silico dynamic BH3 profiling** — rank which cells sit closest to the apoptotic threshold. `Test:` RUNG-13 single-cell model + atlas priming signature. **Status: 🟢.**

### Tier E — Synthetic lethality / collateral vulnerability (kill what the tumour *lost*)
- **E1 Synthetic-lethal paralog pairs** — tumour lost gene A → addicted to paralog B (MTAP-del → PRMT5; BRCA → PARP). `Test:` arena `synthetic_lethal` + atlas co-loss/dependence. **Status: ✅ arena `synthetic_lethal` (SAFE-by-construction q_n≈0, but coverage-limited → combine with a propagation arm) + 🟢 atlas.**
- **E2 Collateral passenger-deletion dependence** — deletions next to tumour suppressors create unique dependence (ENO1-del → ENO2). `Test:` atlas. **Status: 🟢.**
- **E3 Non-oncogene addiction** — stress/proteostasis dependencies unique to the transformed state. `Test:` atlas. **Status: 🟢.**

### Tier F — Metabolic & microenvironment gates (the tumour's own niche betrays it)
- **F1 Warburg / glycolysis-gated prodrug** — activate payload only under high-lactate/low-pH. `Test:` atlas glycolytic enzymes + pH-gate sim. **Status: 🟢.**
- **F2 Glutamine addiction (GLS)** — `Test:` atlas. **Status: 🟢.**
- **F3 pH-gated payload** — acidic tumour microenvironment as the trigger. `Test:` spatial reaction-diffusion sim. **Status: 🟢.**
- **F4 Hypoxia-gated (HIF)** — fire only in the hypoxic core. `Test:` atlas + spatial O₂ sim. **Status: 🟢.**

### Tier G — Restore the off-switch / withdraw the on-switch
- **G1 Mutant-p53 reactivation / refolding** — small-molecule re-fold of mutant p53 (APR-246/eprenetapopt logic). `Test:` **AlphaFold mutant-p53 ± stabiliser, ΔΔG** — structure tool we already run. **Status: 🟢 (clean structure experiment).**
- **G2 Oncogene-addiction withdrawal** — block the addicted oncogene (KRAS) → intrinsic apoptosis. `Test:` atlas dependency. **Status: 🟢.**
- **G3 MDM2 inhibition** — free up WT p53 in p53-WT tumours (nutlin logic). `Test:` atlas p53 status. **Status: 🟢.**

### Tier H — Catastrophe induction (push the broken cell off the cliff)
- **H1 Replication-stress catastrophe** — WEE1/CHK1/ATR inhibition forces high-rep-stress tumours into mitotic catastrophe; checkpoint-intact normal cells survive. `Test:` atlas proliferation/checkpoint signature. **Status: 🟢.**
- **H2 Forced premature mitosis** — drive the cell into mitosis before it's ready. `Test:` atlas. **Status: 🟢.**

### Tier I — Immunogenic eradication (call the hunters)
- **I1 MHC-I re-induction** — restore antigen presentation in immune-evading tumours (IFN-inducibility). `Test:` atlas. **Status: ✅ RUNG-9 (inducibility mapped).**
- **I2 MHC-I window status** — *is the display window even ON in the cancer cells?* The load-bearing assumption of the whole immune route. `Test:` 6,319 WGS tumours (genetic immune escape). **Status: ✅ RUNG-18 — window genetically intact 78% / dimmed 18% / fully-dark only 3.7%; route GENETICALLY viable. Epigenetic-silencing complement = RUNG-18b (Colab).**
- **I2 Neoantigen vaccine / CAR target addressability** — `Test:` sequence + atlas. **Status: 🟢 (overlaps A2).**
- **I3 Immunogenic cell death (ICD)** — pick a death mode that *alerts* the immune system (calreticulin exposure). `Test:` atlas + arena death-mode tagging. **Status: 🟢.**

### Tier Z — 🔮 FUTURE: physics & delivery (kept safe, test later — "Gojo doesn't need a scalpel")
- **Z1 Oncotripsy (sound)** — ultrasound at the cancer cell's *resonant frequency*; cancer is mechanically softer → shifted f₀ → selective mechanical lysis. *Basis: Heyden & Ortiz 2016.* `Test:` toy resonance model in arena (`oncotripsy`); real = mechanics/wet-lab. **Status: 🔮 toy arm built.**
- **Z2 TTFields (EM)** — alternating ~200 kHz field disrupts the mitotic spindle (charged tubulin); dividing tumour hit, quiescent normal spared. *Basis: FDA-approved Optune for glioblastoma.* `Test:` toy division-rate model (`ttfields`). **Status: 🔮 toy arm built.**
- **Z3 Plasmonic photothermal** — gold nanoparticles + laser → local heat ablation. **Status: 🔮.**
- **Z4 Magnetic hyperthermia** — magnetic nanoparticles + AC field → local heat. **Status: 🔮.**
- **Z5 Nanorobotic nanometer-precision delivery** — deliver *any* payload exactly where each cancer cell is, decoupling the kill mechanism from self-propagation (if the wave under-reaches, the robot carries it). **Status: 🔮 (the delivery layer that makes every Tier-A–D payload land).**
- **Z6 Hybrids** — wave-payload + nanorobotic seeding · prime (D1) + push (B1) + reroute resistant (C) · sound/EM softening + bio payload. **Status: 🔮 (combination space).**

> **How to add to this list:** drop the hypothesis here with a `Test:` line naming which tool answers it
> (atlas / sequence / structure / arena). If it's a death/propagation dynamic, wire it as a new arm in
> `scripts/39_mechanism_arena.py` and it joins the leaderboard automatically.

---

## Honest scope

This is an infrastructure + methods paper. We design molecules computationally and predict their behavior through cascaded simulators. We do **not** synthesize, test in cells, or claim therapeutic outcomes. Wet-lab validation is the follow-up paper.

See docs/ASSESSMENT.md for explicit success-tier odds.
