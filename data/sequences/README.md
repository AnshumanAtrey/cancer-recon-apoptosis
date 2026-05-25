# Reference Sequences for Smoke Tests

These FASTA files hold the canonical sequences used by `scripts/01_*_smoketest.py`.

| File | What it is | Why we have it |
|---|---|---|
| `dr5_ecd_human.fasta` | DR5 (TNFRSF10B) extracellular domain | The cancer-cell receptor we want our designed ligand to BIND |
| `trail_dr5_binding_loop.fasta` | Short TRAIL fragment known to bind DR5 | Positive control — should score HIGH on our oracle |
| `scrambled_control.fasta` | Same-length random shuffle of the TRAIL fragment | Negative control — should score LOW on our oracle |
| `dr4_ecd_human.fasta` | DR4 (TNFRSF10A) extracellular domain | "Normal homolog" stand-in for specificity-differential checks (Step 3). DR4 is the closest paralog of DR5; selective ligands need to discriminate. |

## Smoke test logic

The smoke test (`scripts/01_boltz_smoketest.py`) asks:
- Does `Oracle(DR5_ECD, TRAIL_loop) > Oracle(DR5_ECD, Scrambled)` by a meaningful margin (≥2 kcal/mol for binding affinity)?

If YES → the oracle gives usable signal, proceed to Step 2.
If NO → see [ASSESSMENT.md](../../ASSESSMENT.md) Day-1 kill criteria.

## Sequence sources

- **DR5 ECD:** UniProt O14763 (TNFRSF10B), residues 56–183 (cysteine-rich extracellular ligand-binding region).
- **DR4 ECD:** UniProt O00220 (TNFRSF10A), analogous region. Used as the "normal homolog" stand-in even though DR4 is also cancer-relevant — in the real Step 3 we'll use proper normal-tissue-restricted homologs from differential-expression data.
- **TRAIL loop:** Residues 130-149 of human TRAIL (TNFSF10, UniProt P50591) — the AA loop that contacts DR5 in the bound complex (PDB: 1D2Q, 1DU3).
- **Scrambled:** Random permutation of the TRAIL-loop residues, preserving composition.

## Verification before commit

Before going to cloud-GPU Step 1, verify each FASTA via:
- BLAST against UniProt — should hit canonical entry
- Length check vs. known domain boundaries
