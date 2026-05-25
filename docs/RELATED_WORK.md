# cancer-recon-apoptosis — Related Work

The 16 papers that justify, ground, or constrain this project. Grouped by role.

---

## Group A — Biological foundation (proves the idea is real, not speculative)

### 1. Connexin-43 in Cancer: Above and Beyond Gap Junctions (2024)
Comprehensive review of Cx43's dual role — gap junction-mediated bystander apoptosis AND non-canonical C-terminal / hemichannel signaling.
**Use:** Cite when arguing the *mechanism* by which an engineered ligand could propagate apoptosis through cancer cells.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11674308/

### 2. Cascade-Targeting Apoptosis via TRAIL Bystander + Mitochondrial Photodamage (Nano Letters, 2025)
Nanoagent expresses TRAIL on tumor cells which induce DR5-mediated apoptosis in cancer neighbors. Direct in-vivo proof-of-concept.
**Use:** Closest existing therapeutic to Shriya's hypothesis. Cite as motivation; differentiate ours via *computational design* rather than wet-lab discovery.
https://pubs.acs.org/doi/10.1021/acs.nanolett.5c00878

### 3. Radiation-induced Bystander Signalling in Cancer Therapy (Nature Reviews Cancer)
Foundational review on cell-cell apoptosis propagation via gap junctions, NO, conditioned medium.
**Use:** Required reference for any paper claiming to engineer the bystander effect.
https://www.nature.com/articles/nrc2603

---

## Group B — Structure / affinity oracle stack

### 4. AlphaFold 3 — Accurate Structure Prediction of Biomolecular Interactions (Abramson et al., Nature 2024)
All-atom complex prediction including proteins, ligands, DNA, RNA, ions.
**Use:** Ground-truth benchmark only (server-only access). Cross-validate Boltz-2 outputs against it on selected complexes.
https://pubmed.ncbi.nlm.nih.gov/38718835/

### 5. Boltz-2 — Towards Accurate and Efficient Binding Affinity Prediction (bioRxiv, June 2025) ← PRIMARY ORACLE
First open model approaching FEP-quality affinity prediction at 1000× lower cost. MIT-licensed.
**Use:** Our oracle's binding-affinity signal. Both terms of the specificity reward (cancer binding +, normal binding −).
https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1

### 6. Protenix — ByteDance's Comprehensive AlphaFold3 Reproduction (bioRxiv, Jan 2025)
ByteDance PyTorch AF3 reproduction, Apache 2.0. v1 released Feb 2026 with AF3-level performance.
**Use:** Geopolitical-hedge oracle. Cross-validate critical predictions. Defensible "sovereign stack" framing for India/iDEX context.
https://www.biorxiv.org/content/10.1101/2025.01.08.631967v1

### 7. Technical Report of HelixFold3 (Baidu, arXiv Aug 2024)
PaddleHelix AF3 replication, accuracy comparable to AF3 on conventional ligands/nucleic acids/proteins.
**Use:** Second Chinese option. Backup if Boltz-2 / Protenix unavailable.
https://arxiv.org/abs/2408.16975

### 8. ESM3 — Simulating 500M Years of Evolution with a Language Model (Science 2024)
EvolutionaryScale's 98B multimodal protein LM reasoning over sequence + structure + function.
**Use:** Candidate base for the RL policy (alternative to Llama-3.2-3B). Or as the foldability/perplexity prior in the composite reward.
https://www.science.org/doi/10.1126/science.ads0018

---

## Group C — Design tools (the policy's action space)

### 9. De Novo Design of Protein Structure and Function with RFdiffusion (Watson et al., Nature 2023)
Diffusion-based backbone generation with sub-Angstrom control over binding geometry.
**Use:** Optional pre-conditioning step — generate plausible backbones for the policy to refine, or seed initial sequence templates.
https://www.nature.com/articles/s41586-023-06415-8

### 10. One-Shot Design of Functional Protein Binders with BindCraft (Nature, 2025)
End-to-end AF2-weight-leveraging pipeline, 10–100% experimental success rates without high-throughput screening.
**Use:** Competing pipeline. Argue our value-add: (a) RL loop, (b) specificity reward, (c) end-to-end simulation tier.
https://www.nature.com/articles/s41586-025-09429-6

### 11. PepINVENT — Generative Peptide Design Beyond Natural Amino Acids (2025)
RL-guided peptide generation with non-natural amino acid support.
**Use:** Closest "RL for peptide design" prior art. Reference for our methodology section. PharmaRL-of-peptides analog.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12002334/

---

## Group D — ⚠ Closest prior art (read first; differentiate explicitly)

### 12. De Novo Protein Design Enables Targeting of Intractable Oncogenic Interfaces (Baker Lab, bioRxiv Oct 2025)
RFdiffusion + ProteinMPNN binders against "undruggable" oncogenic protein-protein interfaces.
**Differentiation we own:** (a) RL loop they don't have; (b) bystander/cell-cell framing they don't have; (c) explicit cancer-vs-normal specificity reward; (d) end-to-end biological simulation tier (PySB + PhysiCell + ADMET).
https://www.biorxiv.org/content/10.1101/2025.10.22.683953v1.full.pdf

### 13. Computationally Designed High-Specificity Inhibitors of BCL2 Pro-Survival Proteins
Designed three-helix bundles with pM-nM affinity and >300× specificity to individual BCL2 family members.
**Use:** Cite for "cancer-pathway-specific designed proteins work and specificity is achievable." Apoptosis-side molecular target evidence.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5127641/

### 14. SynNotch CAR-T Cell — When Synthetic Biology and Immunology Meet Again (Frontiers Immunology 2025)
2025 review of Wendell Lim / Kole Roybal logic-gated cell therapy with AND/OR/NOT gates.
**Use:** Conceptual ancestor — "engineering cell-cell communication for cancer." Position our work as the next-step ligand-design layer.
https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2025.1545270/full

---

## Group E — Cancer-specific target discovery (Phase 0)

### 15. CellChat v2 for Cell-Cell Communication from scRNA-seq and Spatial Transcriptomics (Nature Protocols 2024)
Standard tool for inferring L-R communication networks from single-cell data; spatial support.
**Use:** Step 2 — discover cancer-restricted L-R pairs as our training targets. R package, callable via rpy2 or CSV export.
https://www.nature.com/articles/s41596-024-01045-4

### 16. PriorCCI — Interpretable DL for Key Ligand-Receptor Interactions Between Specific Cell Types (PMC 2025)
DL framework for prioritizing cancer-vs-other-cell L-R interactions.
**Use:** Solves the *specificity* problem at the target-discovery stage. Stacks on top of CellChat to rank-filter targets.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345837/

---

## If you read only 5 (in order)

1. **#12 Baker Lab Oct 2025** — competing approach; we differentiate against this paper specifically.
2. **#5 Boltz-2** — the oracle.
3. **#11 PepINVENT** — closest RL-for-peptide-design prior art.
4. **#2 TRAIL Bystander Nano Letters 2025** — the existing wet-lab proof-of-concept.
5. **#1 Cx43 Review 2024** — the mechanism we'd hijack.

That set tells you whether the contribution claim has white space. Current read: YES — nobody sits at the intersection of all four.
