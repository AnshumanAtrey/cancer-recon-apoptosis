# cancer-recon-apoptosis — Cancer-Cell Recognition-Triggered Apoptosis

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
- [ ] (rest of 19-step plan — see [PLAN.md](PLAN.md))

---

## Quick map

```
cancer-treatment/
├── README.md                       this file
├── PLAN.md                         the 19-step roadmap with tech + layman steps
├── ASSESSMENT.md                   honest success odds + risk audit
├── Conceptual Research(1).pdf      Shriya's source paper
├── docs/                           supplementary planning docs
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

Expected output: ΔG(DR5+DR5-B) − ΔG(DR5+scrambled) ≥ 2 kcal/mol. If not, see ASSESSMENT.md kill criteria.

---

## Reading order if joining the project

1. [Conceptual Research(1).pdf](Conceptual%20Research(1).pdf) — Shriya's idea, 8 pages
2. [PLAN.md](PLAN.md) — 19 steps, what we're building, why
3. [ASSESSMENT.md](ASSESSMENT.md) — honest odds, risks, day-1 kill criteria
4. [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — 16 papers we cite
5. [scripts/01_local_smoketest.py](scripts/01_local_smoketest.py) — runs today

---

## Headline contributions (target paper)

1. First OpenEnv-native chemistry/biology environment for cancer-specific ligand design with a *specificity-aware oracle* (cancer-receptor binding minus normal-receptor binding).
2. Reference GRPO recipe ported from PharmaRL to peptide/protein design space with held-out cancer-type generalization.
3. End-to-end in silico evaluation pipeline: design → binding → apoptosis cascade (PySB+EARM) → tissue bystander propagation (PhysiCell+PhysiBoSS) → ADMET filter. Reports cumulative survival rate as the load-bearing metric.

---

## Honest scope

This is an infrastructure + methods paper. We design molecules computationally and predict their behavior through cascaded simulators. We do **not** synthesize, test in cells, or claim therapeutic outcomes. Wet-lab validation is the follow-up paper.

See ASSESSMENT.md for explicit success-tier odds.
