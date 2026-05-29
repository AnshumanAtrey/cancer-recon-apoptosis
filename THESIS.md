# Target Thesis — Hybrid recon-apoptosis (locked 2026-05-29)

## Decision

The project designs **recognition ligands that (1) bind a cancer-enriched cell-surface
receptor for specificity and (2) trigger apoptosis** — a tumour-antigen × death-receptor
design. This operationalises both halves of Shriya Rai's concept (*recognition* +
*self-destruction via altered cell signalling*).

Chosen over the two single-mechanism alternatives:
- **Over-expressed-receptor only** (just bind HER2-like badges) — only the recognition
  half; no killing; least differentiated ("another binder generator").
- **DR5 death-receptor only** (just press the self-destruct switch) — strong death
  mechanism but weak cancer-specificity (DR5 is on normal cells too; Step 2 confirmed it
  is not transcriptionally cancer-restricted), and it discards the Step-2 shortlist.

## The design object

A **bispecific recognition ligand** with two functional ends:

| end | job | source of evidence |
|---|---|---|
| **anchor** | bind a cancer-OVER-EXPRESSED surface receptor → specificity | Step 2 shortlist: HER2/ERBB2, ERBB3, EPHB4, MUC1, SDC1, CD74, ITGB4 … |
| **trigger** | engage/cluster a death receptor → apoptosis | Step 1: DR5 (TNFRSF10B), validated modelable interface |

Mechanistic note: DR5 kills by **clustering**. Anchoring to an over-expressed tumour
antigen drives local DR5 clustering on cancer cells specifically — death is conditional
on the cancer badge being present. This is a real therapeutic class (tumour-antigen ×
DR5-agonist bispecifics) and maps directly onto Shriya's "recognise neighbour → trigger
self-destruct," including the bystander framing (a cancer-deployed/locally-acting ligand
that recognises cancer badges on neighbours and triggers their apoptosis).

## How this reframes the remaining steps

- **Step 2c (LIANA communication, optional/supporting):** annotate which shortlisted
  badges are part of tumour-enriched ligand-receptor *communication* — supports the
  cell-cell recognition / bystander narrative; not on the critical path.
- **Step 3 (specificity audit):** for the top anchor candidates (HER2 first), compare
  cancer vs NORMAL-tissue expression (GTEx/normal scRNA) + structural homolog check, to
  pick anchors with the safest cancer-vs-normal margin. DR5 is fixed as the trigger.
- **Step 4 (reward function):** composite, scored with the Step-1 two-axis Boltz oracle
  (interface pLDDT ∧ interface PAE) on BOTH ends:
  - `+` predicted binding to the cancer anchor (specificity)
  - `+` predicted DR5 engagement (death trigger)
  - `−` predicted binding to the anchor's normal-tissue homolog (avoid healthy cells)
  - `−` foldability / length / realism penalties
- **Steps 5–17:** unchanged in structure; the "ligand" is now this bispecific recon-
  apoptosis design, evaluated end-to-end (apoptosis cascade, bystander tissue sim, DepMap/
  GTEx selectivity, ADMET/immunogenicity).

## Anchor locked (2026-05-29, after Step 3): Trop2 (TACSTD2)

Step 3's specificity audit (STEP3_METHODOLOGY.md) FAILed all 10 Step-2 over-expressed
candidates on safety (broad expression and/or vital parenchyma) — a real, benchmark-
calibrated finding that cancer over-expression ≠ therapeutic window. The anchor therefore
pivots to a **tissue-restricted, audit-PASSing, clinically-validated** antigen:

**ANCHOR = Trop2 / TACSTD2.** Rationale: PASSes the audit (tissue-restricted, not in vital
parenchyma); ADC-approved (sacituzumab govitecan) in **breast + lung** — covers 2 of our 3
tumour types; single-pass TM with a well-characterised ectodomain (cysteine-rich + thyroglobulin
type-1 domain; structures available) → modelable by Boltz. **Key asset for Step 4:** the
sacituzumab Fab is a *known Trop2 binder* → it is the positive control for the anchor-binding
oracle, exactly as TRAIL→DR5 anchored Step 1. Anti-target for specificity: the paralog EPCAM
(TACSTD1) — the binder must hit Trop2, not EPCAM.

**TRIGGER (unchanged) = DR5 / TNFRSF10B.**

So the design object is now concrete: a bispecific that binds **Trop2** (cancer anchor) and
clusters **DR5** (apoptosis trigger). Step 4 reward (two-axis Boltz oracle, the Step-1 method):
`+ binds Trop2 ECD`, `+ engages DR5`, `− binds EPCAM` (paralog off-target), `− foldability/length`.

## One-line project framing

> Engineer recognition ligands that bind cancer-enriched surface receptors and trigger
> apoptosis — operationalising Shriya Rai's cancer self-recognition + self-destruction
> concept as a tumour-antigen × death-receptor bispecific, designed by RL against a
> Boltz-2 two-axis interface oracle.
