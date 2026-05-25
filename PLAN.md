# cancer-recon-apoptosis — 19-Step Project Plan

Builds on Shriya Rai's *"Inducing Self-Destruction in Cancer Cells Through Altered Cellular Communication Signaling"* (conceptual paper, Apr 2026). Operationalises the hypothesis as a computational design pipeline.

**Working hypothesis:** Cancer cells can be made to induce apoptosis in neighboring cancer cells via engineered ligands that (a) bind cancer-specific receptors with high affinity, (b) avoid normal-tissue homologs, (c) activate the apoptosis cascade, (d) propagate through gap junctions / paracrine signaling.

---

## PHASE 0 — Validate before you build (Week 1)

### Step 1 — Boltz-2 positive/negative control test
**Tech:** Run Boltz-2 binding prediction on the known DR5–DR5-B complex (positive) and DR5–scrambled-peptide (negative); confirm ≥2 kcal/mol gap.
**Layman:** Before trusting our scoring tool, test it on a known answer. Give it a real cancer-killing protein and a random one — does it tell them apart? If yes, the tool works. If no, switch tools before wasting weeks.

### Step 2 — Cancer-restricted target discovery via CellChat v2
**Tech:** Pull pan-cancer scRNA-seq from CELLxGENE; run CellChat v2 across 5+ cancer types; output ranked list of ligand-receptor pairs enriched in cancer vs matched normal.
**Layman:** Scan tens of thousands of cells from public databases. Find "conversations" that happen between cancer cells but not between normal cells. Those become our shortlist of safe targets.

### Step 3 — Specificity differential audit
**Tech:** For top-10 receptor candidates, predict structures of cancer variant vs healthy-tissue homolog with Boltz-2 and Protenix (cross-check); compute ΔΔG of a reference ligand against each.
**Layman:** For each cancer target, check its "twin" on healthy cells. If they look nearly identical, that target is dangerous — cross out. Keep targets where cancer and healthy versions differ enough to design around.

---

## PHASE 1 — Build the scoring engine (Week 2)

### Step 4 — Composite reward function
**Tech:** Convex combination of: Boltz-2 ΔG to cancer target (+), Boltz-2 ΔG to normal homolog (−penalty), ESM-3 perplexity (foldability), length penalty, ProteinMPNN recovery score (sequence realism).
**Layman:** Build a scorecard. Points for sticking to cancer, points lost for sticking to healthy cells, points lost for being too long or unrealistic. Higher score = better candidate.

### Step 5 — Oracle calibration against published agonists
**Tech:** Score 5–10 known TRAIL/DR5-B variants and BCL2 inhibitors; verify they rank in the top 10–20% of randomly sampled peptides.
**Layman:** Make sure the scorecard agrees with reality. Score real cancer-killing molecules — they should rank near the top. If they don't, the scoring is broken and we fix it now.

### Step 6 — Wrap as OpenEnv FastAPI HTTP service
**Tech:** Package the oracle behind `POST /step`, `POST /reset`, `GET /state` endpoints; deploy to a Hugging Face Space; export typed client.
**Layman:** Turn the scorecard into a website any AI can query. Same plumbing as PharmaRL. Other researchers can plug in their own AI later.

---

## PHASE 2 — Train the design AI (Weeks 3–4)

### Step 7 — OpenEnv environment definition
**Tech:** Action schema: `ADD_RESIDUE`, `SUBSTITUTE_RESIDUE`, `MUTATE_REGION`, `TERMINATE`. Observation: current sequence, target receptor, valid-action set, per-component oracle breakdown.
**Layman:** Define the rules of the game the AI plays. Each turn it can add an amino acid, swap one, mutate a region, or submit its final design.

### Step 8 — Llama-3.2-3B + LoRA + GRPO recipe (port from PharmaRL)
**Tech:** Base policy: Llama-3.2-3B-Instruct in 4-bit; LoRA r=16, α=32; GRPO group size G=8, KL β=0.04, lr=5e-6, AdamW. Optional: swap base to ESM-3 if compute allows.
**Layman:** Reuse the same training setup that worked for PharmaRL. Same algorithm, same model, pointed at the new problem.

### Step 9 — Curriculum training run (300–500 steps)
**Tech:** Three-tier curriculum: trivial (binding only) → easy (binding + foldability) → hard (binding + foldability + specificity penalty). Track parse rate, mean group reward, KL-to-reference.
**Layman:** Teach the AI in stages. First just stick to cancer. Then stick and fold properly. Then stick AND avoid healthy cells. Slow build-up so it actually learns.

---

## PHASE 3 — Molecular-level evaluation (Week 5)

### Step 10 — Held-out target generalization
**Tech:** Train on K cancer L-R pairs (e.g., breast, glioblastoma); evaluate on held-out cancer type (e.g., colorectal). n=30 episodes per condition.
**Layman:** Train on most cancers, hide one. Test on the hidden one. If it still does well, the AI actually learned something general.

### Step 11 — Hand-engineered baseline comparison
**Tech:** Compare trained policy against (a) random sequence sampler, (b) untrained Llama base, (c) BindCraft single-shot binder, (d) DR5-B reference variant.
**Layman:** Race the AI against the dumb version (random), untrained version, and the best human-designed protein already known. Did our AI win? By how much?

### Step 12 — Specificity-reward ablation
**Tech:** Re-train WITHOUT the cancer-minus-normal penalty term; report with-vs-without delta.
**Layman:** Run training twice — with and without the "don't hit healthy cells" rule. Show the difference. This is the paper's load-bearing claim.

---

## PHASE 4 — Biological simulation (Weeks 6–7)

### Step 13 — Apoptosis cascade simulation (PySB + EARM)
**Tech:** For top-K designs, plug predicted binding strength into the Extrinsic Apoptosis Reaction Model as upstream caspase-8 input. Simulate caspase-3 activation over 24h. Report fraction crossing the apoptosis threshold.
**Layman:** Use a published "death-machinery model" of a cell. Plug our designed signal into the front end. The model tells us: yes, the signal triggers the death machinery, or no, it stalls.

### Step 14 — Tissue-level bystander simulation (PhysiCell + PhysiBoSS)
**Tech:** Build 3D virtual tumor (~10K cancer cells + ~2K healthy cells with gap junctions). Inject designed ligand at t=0 in a subset. Simulate 48h. Measure: % cancer dead, % healthy dead, cascade propagation velocity, radius.
**Layman:** Build a SimCity tumor. Inject the designed death message in 100 cells. Hit play for two virtual days. Does death spread through the tumor? Does it leak to healthy cells?

### Step 15 — DepMap + GTEx selectivity heatmap
**Tech:** Predict susceptibility = (receptor expression × predicted ligand affinity) across 1,800 cancer cell lines (DepMap) and 50 normal tissue types (GTEx). Output: tissue × target heatmap.
**Layman:** For ~2,000 cancer types AND ~50 healthy tissues, predict whether our signal would hit them. Cancers should light up; healthy tissues stay dark. The heatmap sells the paper.

### Step 16 — ADMET + immunogenicity panel
**Tech:** Run ADMET-AI (41 predictors) on top designs; run NetMHCpan for MHC-binding prediction (immune-rejection risk). Filter red-flagged candidates.
**Layman:** Final filter. Will the body absorb our drug? Will the immune system attack it as foreign? Standard pharma checks.

### Step 17 — End-to-end pipeline ablation
**Tech:** Re-run the full pipeline (Steps 13–16) with untrained base versus GRPO-trained model. Count designs that pass ALL four filters in each condition. Report multiplicative survival rate.
**Layman:** Run the whole pipeline twice. Once with our trained AI. Once with no training. Count survivors. The trained version should win by a lot. That's the headline number.

---

## PHASE 5 — Publish (Weeks 8–9)

### Step 18 — Paper draft using PharmaRL template
**Tech:** Standard sections: intro / related work / env spec / methodology / molecular results / biological-simulation results / limitations / future work. Explicit differentiation paragraph from Baker Lab Oct 2025 paper.
**Layman:** Reuse our PharmaRL paper's structure. Swap in new content. Be clear about how this differs from closest existing work.

### Step 19 — Open-source artifacts + arXiv submission
**Tech:** Live HF Space (env), HF Hub (trained adapter), GitHub repo (code + simulation configs), Zenodo DOI (archived record). Submit arXiv: cs.LG primary; q-bio.BM, q-bio.QM, physics.bio-ph cross-list.
**Layman:** Put everything online for free — the AI, the website, the code, the data. Get an arXiv link. Anyone in the world can use it, build on it, or check our work.

---

## Calendar

| Week | Phase | Week-end deliverable |
|---|---|---|
| 1 | 0 — Foundation | Boltz-2 calibrated; cancer-restricted target shortlist |
| 2 | 1 — Oracle | Scorecard calibrated against known agonists; HF Space live |
| 3–4 | 2 — Training | First trained checkpoint with non-trivial reward delta |
| 5 | 3 — Molecular eval | Held-out numbers + baseline comparison + ablation table |
| 6 | 4a — Simulation | EARM apoptosis + PhysiCell tissue results for top-K designs |
| 7 | 4b — Simulation | DepMap heatmap + ADMET panel + end-to-end ablation |
| 8 | 5a — Writing | Paper draft complete |
| 9 | 5b — Publish | arXiv submitted, all artifacts public |

**Solo: 9 weeks.** With one collaborator: 5–6 weeks. With Shriya as biology co-author for apoptosis-pathway interpretation: 6–7 weeks with stronger biological framing.

---

## The single highest-leverage move

**Step 1.** Six hours of work. If Boltz-2 can't separate the known cancer-killing peptide from random noise by ≥2 kcal/mol on Day 1, the rest of the project pivots — switch to AlphaFold 3 Server, use ABFE (Recursion Oct 2025 pipeline atop Boltz-2), or rethink the oracle. Don't write Step 2 until Step 1 returns the right number.
