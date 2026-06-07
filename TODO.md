# TODO — cancer-recog-apoptosis: next experiments

Living roadmap of what to run next. Every item is a falsifiable, in-silico test that runs on the M2 or free
Colab. Honest negatives are first-class. See `README.md` for the full hypothesis catalog and finished rungs.

**Legend:** `[ ]` open · `[x]` done · **P0** do next · **P1** soon · **P2** later/refinement · **FUT** future-safe (physics/delivery)
**Where:** 💻 M2 laptop (CPU) · ☁️ Colab CPU · ⚡ Colab GPU

---

## Done so far (recap — see README for detail)
- [x] RUNG 5–10b — surface logic gates bounded; single + combinatorial surface AND-NOT can't spare vital tissue
- [x] RUNG-11/12 — neoantigen route + AlphaFold pMHC structure (clean handles certified)
- [x] RUNG-13/14 — death wave validated + mechanism arena (quorum leads, ferroptosis/wave close)
- [x] RUNG-15/16/17 — atlas×mechanism map; clonal neoantigen burden (high-TMB broadly seedable); binding-axis immunogenicity (safety↔immunogenicity align)
- [x] RUNG-18/18b — **MHC window status**: genetically intact ~78% / dimmed ~18% / fully-dark ~4%; expression silencing adds ~2× (lung ~13% dark) → window broadly ON, genetics under-counts ~2×
- [x] RUNG-19 — evolutionary escape race: cure collapses at L=μ·N₀≈1; bystander cross-kill shifts curable size ~10×; clinical tumours NEED a resistance-agnostic 2nd mechanism
- [x] RUNG-20 (Boltz 2b) — Boltz confirms mutants present on MHC; can't discriminate mut-vs-WT by binding (saturated confidence) → discrimination is TCR-level, needs pose-RSA

---

## Recognition windows — normal cell vs cancer cell (the candidate handles)

Every row is a way to tell a cancer cell from a normal one — i.e. a potential recognition "window".
Single surface windows leak (proven). Inside windows are richer/more specific. Power = STACKING with AND-logic.

| # | Feature (plain) | Normal cell | Cancer cell | Where read | Status |
|---|---|---|---|---|---|
| 1 | **Mutations** (neoantigens = mutated protein pieces) | almost none | many, unique | inside → shown on MHC window | ✅ main handle (RUNG-11/16/17) |
| 2 | **MHC window** (display shelf) | on | mostly on, sometimes dark | surface | ✅ measured ~85–90% on (RUNG-18/18b) |
| 3 | **Surface marker proteins** (HER2/EGFR/EpCAM) | low | high — but also on organs | surface | ✅ tested → **leaks** (RUNG-15) |
| 4 | **Sugar coating** (glycans: Tn, sialyl-Tn) | normal | abnormal | surface | ❌ leak-test candidate |
| 5 | **Glucose use** (Warburg: guzzles sugar, spits acid) | calm, uses O₂ | fast, makes lactate | inside + environment | ❌ open |
| 6 | **Self-destruct machinery** (apoptosis) | intact | disabled (BCL-2 high) = weakness | inside | listed (BH3 mimetics) |
| 7 | **"Don't eat me" signal** (CD47) | low | high (hides from immune) | surface | ❌ leak-test candidate |
| 8 | **Chromosome count** (DNA amount, aneuploidy) | correct (46) | wrong, unstable | inside | ❌ open |
| 9 | **Telomerase** (immortality enzyme) | off in adult cells | switched on | inside | ❌ open |
| 10 | **Division rate** (Ki-67, replication stress) | rests | always dividing | inside | ❌ open |
| 11 | **Membrane lipid flip** (phosphatidylserine outside) | tucked inside | flipped out (stress) | surface | ❌ leak-test candidate |
| 12 | **Gene dependency** (synthetic-lethal, e.g. MTAP loss) | has both copies | lost one → addicted to partner | inside | ✅ tested (RUNG-14) |
| 13 | **Microenvironment** (pH, oxygen) | normal | acidic, hypoxic | around the cell | listed (Tier F) |
| 14 | **Physical** (stiffness, size, charge) | normal | softer/stiffer, depolarised | physical | FUT (oncotripsy/TTFields) |

---

## P0 — do next (runnable now)

- [ ] **NEW-WINDOW LEAK TEST** (the table → atlas). ☁️ Colab Census. For the top new candidate windows —
      **glycan-synthesis genes (#4)**, **CD47 / "don't eat me" (#7)**, **phosphatidylserine-flip / metabolic
      genes (#11/#5)** — measure normal-organ leakage exactly like RUNG-15 did for surface markers
      (worst-donor vital-tissue expression q_n). Output: which new windows are clean enough to STACK.
      *Reuse:* RUNG-15/34 Census loader (`scripts/40` census mode, `scripts/34` `find_leak_channels`).
      *Answers:* do any inside/surface windows beat the surface-marker leak ceiling? → the new clean handles.

- [ ] **COMBINATORIAL AND-LOGIC RECOGNITION** (stack windows for precision). 💻 M2. Take the windows that
      pass the leak test + the neoantigen window, model 2–3-input AND gates, compute per-patient
      addressability vs false-positive on vital tissue. *Answers:* does stacking windows push normal-cell
      false-positives toward zero while keeping tumour coverage? (the precision multiplier).

- [x] **CROSS-KILL CLEARANCE — T-cell + NK + bystander combined** ✅ RUNG-21 (T+NK+wave cures 100% across measured escapee 4-13%; all 3 layers load-bearing; NK necessity shown by evasion collapse 1.00→0.07) (RUNG-19's required 2nd mechanism). 💻 M2,
      extend the escape-race lattice. Real tumours already carry MHC-dark "escapee" cells (RUNG-18/18b), so a
      T-cell-only therapy *cannot* clear them — RUNG-19 proved the bare wave needs a **resistance-agnostic
      cross-kill**. Model two killers together: **T-cells** kill MHC+ cells (recognition-gated) + **NK cells**
      (*Natural Killer = immune cells that attack cells with LOW/missing MHC — the opposite trigger to T-cells*)
      kill the MHC-dark escapees + bystander death wave. *Answers:* does T-cell + NK + wave together clear the
      tumour INCLUDING the MHC-loss escapees, where T-cell-alone fails? Quantify the cross-kill strength needed
      vs the measured escapee fraction (RUNG-18/18b). **This is the operational answer to "the immune route's ~4–13% blind spot."**

---

## P1 — soon

- [ ] **CAPSTONE SYNTHESIS** (the honest full story). 💻 M2 / writeup. Assemble the 3-stage chain end-to-end:
      recognition (neoantigen, surface dead) → binding (safety↔immunogenicity align) → apoptosis (wave) +
      window-status (RUNG-18/18b) + escape race (RUNG-19, needs cross-kill at clinical size). Ranked target
      shortlist + named wet-lab residuals. **Fully warranted now** — positive at every stage with stated bounds.

- [ ] **BOLTZ POSE-RSA REFINEMENT** (RUNG-20 done right). ⚡ Colab GPU. Boltz confirmed presentation but
      interface-pLDDT is the wrong ruler for TCR-level discrimination. Parse the Boltz CIF for the mutated
      residue's solvent exposure (RSA) + physicochemical change — the proper structural discrimination metric
      (matches RUNG-12's ESMFold approach, stronger model). *Reuse:* `scripts/37` `analyze_pdb` RSA logic.

- [ ] **RUNG-18b EXTEND to melanoma + bladder**. ⚡ Colab Census. They were absent in Census 2024-07-01 under
      those disease labels → try a newer Census version / specific melanoma & bladder scRNA datasets, or
      epithelial-fallback selection. Completes the route-cancer window-silencing picture (only lung is strong now).

- [ ] **PER-CANCER SURFACE AND-NOT PAIRS**. ☁️ Colab Census. RUNG-15 was pan-cancer pooled; a marker clean+high
      in ONE cancer (e.g. PSMA/prostate) could win per-cancer. Test 2-marker AND-NOT pairs per cancer type.

---

## P1 — the 4 immunotherapy escape obstacles → a runnable test for each

Real tumours fight back four ways. Each maps to an in-silico test with our tools. (No single fix beats all
four — the answer is a LAYERED, multi-target, personalised system; that layered design is the capstone target.)

- [ ] **Obstacle 1 — Tumour turns DOWN its MHC window** (T-cells go blind). *Approaches:* interferon to
      re-light the window; NK cells (kill low-MHC); restore antigen presentation.
      **Our test:** ✅ partly done — RUNG-9 (IFN re-induction in tissue) + RUNG-18/18b (how often dark). **Add:**
      **forced-presentation / re-induction in TUMOUR cells** — does IFN / epigenetic de-repression recover the
      transcriptionally-silenced (reversible) escapees from RUNG-18b? ☁️ Census + model. Plus the **NK arm**
      (see Cross-Kill, P0) for the permanently-dark core.
- [ ] **Obstacle 2 — Tumour stops displaying the target neoantigen** (the flag disappears). *Approaches:*
      target MULTIPLE neoantigens at once; target ESSENTIAL (driver/clonal) mutations that can't be dropped
      without killing the cell; monitor evolution.
      **Our test:** **MULTI-TARGET escape simulation** 💻 — extend RUNG-16/19: P(escape) when targeting 1 vs 3
      vs 5 handles simultaneously (each independent target the tumour must lose ALL of to escape). **+
      ESSENTIALITY ranking** — rank our clean handles by driver-status × clonality (RUNG-16 clonal burden) →
      the "can't-be-lost" targets. *Answers:* how many independent essential targets drive escape probability to ~0?
- [ ] **Obstacle 3 — Immunosuppressive microenvironment** (T-cells arrive but get exhausted/switched off).
      *Approaches:* checkpoint inhibitors (pembrolizumab/nivolumab — *drugs that release the brakes on
      exhausted T-cells*); engineer T-cells to resist suppression; fix the microenvironment.
      **Our test:** 💻 model as a **kill-efficiency / T-cell-exhaustion parameter** in the escape-race & arena —
      how much does suppression have to drop clearance before the wave/cross-kill fails? (bounds how much
      checkpoint-release is "enough"). Microenvironment biology is largely beyond our atlas tools → note honestly.
- [ ] **Obstacle 4 — Tumour evolves NEW escape mutations under pressure** (moving target). *Approaches:*
      attack several independent targets; combine therapies; adaptive/personalised therapy that updates.
      **Our test:** **ADAPTIVE-THERAPY simulation** 💻 — extend RUNG-19: does re-targeting as the tumour evolves
      (vs a fixed single target) beat escape? Compare fixed-1-target vs adaptive-multi-target clearance.

- [ ] **IMMUNOPEPTIDOMICS CONFIRMATION** (is the peptide REALLY on the surface?). ☁️ data lookup. The honest
      gap in RUNG-11/16/17 is *predicted* presentation. Cross-check our top clean handles against **real
      mass-spec-detected peptides** (*immunopeptidomics = directly measuring which peptides sit on MHC*) in the
      public **HLA Ligand Atlas / IEDB**. *Answers:* are any of our predicted handles actually observed on MHC
      in real samples? (turns a prediction into a measurement where data exists).

- [ ] **THE "IDEAL LAYERED SYSTEM"** (the multi-layer defence, as a synthesis) — fold into the **Capstone**:
      sequence tumour → multiple ESSENTIAL neoantigens → multi-target T-cells → **NK cross-kill for MHC-loss
      escapees** → block suppression → monitor evolution & update. Each layer covers another's blind spot;
      our rungs supply the quantitative bound for each layer. *(Cancer is not one disease — solid vs blood differ;
      keep results per-cancer, not "one universal cure".)*

## P2 — catalog tiers not yet tested (from README hypothesis catalog)

- [ ] **C — Alternative death pathway addressability map**. ☁️ Census. Which tumours are wired for
      ferroptosis / pyroptosis / necroptosis / cuproptosis (the brake-free deaths that beat apoptosis-resistance
      & the escape race). Maps RUNG-14 `ferroptosis_wave` onto real per-cancer dependency.
- [ ] **D — BH3-mimetic dependence**. ☁️ Census. Per-tumour BCL-2/MCL-1 reliance → where lowering the
      apoptosis threshold (venetoclax-style) sensitises the recognition-gated wave.
- [ ] **E — Synthetic lethality deeper** (MTAP–PRMT5, ENO1–ENO2 collateral-deletion). ☁️ Census addressability.
- [ ] **F — Metabolic / microenvironment gates** (Warburg, pH, hypoxia as #5/#13 windows). 💻/☁️.
- [ ] **G — p53 refold** (AlphaFold ΔΔG: can a mutant p53 be folded back to active?). ⚡ GPU.
- [ ] **H — Replication-stress / mitotic-catastrophe** gating (#10 window). 💻/☁️.

---

## P2 — DELIVERY & "the medicine" (the BIO half of the syringe / injection / wave-as-medicine ideas — RUNNABLE)

These are the bio-deliverable parts of Anshuman's earlier syringe / AI-injection / "wave in a one-time shot"
hypotheses. The *physics* of delivery (robot, sound, EM) is future-safe below; the *biology* is testable now.

- [ ] **WAVE-AS-A-ONE-TIME-INJECTION — delivery threshold**. 💻 M2, extend RUNG-13/19 wave sim. Anshuman's idea:
      put the death-wave trigger in a single injection (mRNA-LNP / oncolytic virus / peptide) that seeds a few
      cells and lets the wave do the rest. RUNG-13 showed *one* seed can suffice in principle — **quantify it:**
      sweep the SEED FRACTION (what % of tumour cells the injection actually reaches) and find the minimum
      delivery needed for the wave to take over and clear the tumour, vs tumour size & q_n. *Answers:* how good
      does the injection have to be? (decouples "kill mechanism" from "delivery", as Anshuman framed it).
- [ ] **DESIGN THE MEDICINE — personalised neoantigen vaccine construct (in-silico)**. 💻/☁️. For one cancer
      (melanoma), output the actual ranked peptide/mRNA SEQUENCES a lab would synthesise (the thing you'd order
      from Twist/GenScript), built from RUNG-16/17 clean handles + RUNG-20 presentation. *Answers:* produces a
      synthesis-ready blueprint + a publishable proof-of-concept — the closest legitimate "make the medicine"
      step without a wet lab. (NOT synthesize-and-inject — that's the clinical ladder, MAGE-A3 = the warning.)
- [ ] **HYBRID: delivery + wave** (the bio part of "combine wave apoptosis with robotic injection"). 💻. Model
      delivery that UNDER-reaches (robot/injection hits only part of the tumour) → does the self-propagating
      wave close the gap the delivery missed? Quantifies how wave + partial-delivery beat either alone.

## FUT — future-safe (pure PHYSICS / hardware; need lab, kept — full list in README Tier Z, Z1–Z6)
- [ ] **Soundwave at the cancer cell's frequency** (oncotripsy, Z1) — ultrasound resonance; toy arm in RUNG-14
- [ ] **Electromagnetic-wave attack** (TTFields, Z2) — alternating field disrupts division; toy arm in RUNG-14
- [ ] **Photothermal (gold+laser, Z3) · Magnetic hyperthermia (Z4)** — local heat ablation
- [ ] **Nanorobotic nm-precision delivery** (Z5) — robot carries any payload exactly to each cancer cell
- [ ] **Hybrids** (Z6) — sound/EM softening + bio payload · prime + push + reroute-resistant combos

---

## Standing rules (don't violate)
- Honest negatives are first-class; β / kill% / propensities are PROXIES, never verdicts; state the wet-lab residual.
- Validate (selftest) before spending compute. No subagent swarms — run the experiment.
- Commit validated results to the public repo; keep `memory/` (unpublished collaborator IP) PRIVATE/gitignored.
