# Step 1 — Local Plumbing Smoke Test Results

**Date run:** 2026-05-23 (project: `cancer-recon-apoptosis`)
**Host:** Apple M2, 8 GB RAM, macOS Darwin 25.3.0
**Device:** MPS (Apple Silicon GPU backend)
**Script:** `scripts/01_local_smoketest.py`
**Model:** `facebook/esm2_t6_8M_UR50D` (8M params, ~30 MB)
**Status:** ✅ Plumbing OK (workflow validated) — ❌ NOT a real Step-1 PASS

---

## Numbers

| Pair | Cosine similarity | Note |
|---|---|---|
| TRAIL_loop ↔ DR5_ECD | **+0.8100** | Positive control vs target |
| Scrambled ↔ DR5_ECD | **+0.7914** | Negative control vs target |
| **Margin (pos − neg) on DR5** | **+0.0185** | Tiny — see interpretation |
| TRAIL_loop ↔ DR4_ECD | +0.8429 | Specificity diagnostic |
| Scrambled ↔ DR4_ECD | +0.8200 | Specificity diagnostic |

Raw: [`results.json`](results.json)

---

## What this test actually proved

1. **The local pipeline is wired correctly.** Sequences load from FASTA, ESM-2 downloads, MPS backend works on M2, embeddings compute, JSON persists.
2. **The directionality is right.** TRAIL loop has *higher* similarity to DR5 than the scrambled control does. Small margin, but in the expected direction.
3. **ESM-2-8M sequence-embedding cosine is NOT a binding-affinity oracle.** The all-pairs cosine sits around 0.79–0.84 — most of that is ESM-2's residue-composition prior, not binding-relevant signal. The +0.0185 margin between positive and negative is noise-floor adjacent.

## What this test did NOT prove

- **Nothing about real binding affinity.** Cosine of mean-pooled token embeddings does not approximate ΔG.
- **Nothing about Boltz-2 viability.** Boltz-2 uses iterative diffusion-refined complex prediction with an affinity head trained on experimental Ki/Kd data. Wholly different signal.
- **Nothing about cancer-vs-normal specificity.** The TRAIL_loop ↔ DR4 cosine (+0.843) is actually higher than TRAIL_loop ↔ DR5 (+0.810). That is a known artifact of small protein-LM embeddings on short peptides — paralogs cluster — not a specificity claim.

## Verdict for the workflow

Plumbing test → **PASS**. The local development loop works: write a FASTA, run a script, get a JSON of metrics. We can iterate locally now without burning cloud-GPU minutes.

The real Step-1 decision still gates on:
- `scripts/01_boltz_smoketest.py` running on a cloud A10G / A100
- Boltz-2 producing a ΔG separation ≥ 2 kcal/mol between DR5+TRAIL_loop and DR5+scrambled
- See [`../../ASSESSMENT.md`](../../ASSESSMENT.md) Day-1 kill criteria for pivot paths if it doesn't.

---

## Immediate next steps

1. **Provision a cloud GPU** (RunPod / Lambda / Modal / AWS A10G). Same compute envelope as PharmaRL — single A10G suffices for the smoke test.
2. **`pip install -r requirements-cloud.txt`** in that environment.
3. **Run `python scripts/01_boltz_smoketest.py`**. First run downloads Boltz-2 weights (~10 GB) and the MSA-server backend; budget 30 min before the actual prediction starts.
4. **Verify the ΔG gap ≥ 2 kcal/mol.**
5. If PASS → proceed to Step 2 (CellChat cancer-restricted target shortlist).
6. If FAIL → see ASSESSMENT pivot options (AF3 Server, ABFE, MD refinement).

## What we know about the sequences themselves

| Sequence | Length | Sanity check |
|---|---|---|
| DR5_ECD (UniProt O14763 res 56-183) | 128 aa | ✓ matches CRD region from TNFRSF10B |
| TRAIL DR5-binding loop (UniProt P50591 res 130-149) | 20 aa | ✓ within the AA-loop that contacts DR5 in PDB 1D2Q |
| Scrambled control | 20 aa | ✓ same composition as TRAIL loop, randomly permuted |
| DR4_ECD homolog (UniProt O00220) | 129 aa | ✓ analogous to DR5 ECD |

Before running on cloud, **BLAST verification recommended** to confirm canonical UniProt entries — see `data/sequences/README.md`.

## Caveats logged

- The MPS backend reported `MISSING` weights for `pooler.dense` — that's expected: we instantiated `AutoModel` (not `AutoModelForMaskedLM`), so the masked-LM head was dropped. Pooler was initialized fresh but is unused since we use mean-pooling instead.
- ESM-2-8M is a *very* small variant. For better local proxies, swap to `esm2_t12_35M_UR50D` (~150 MB) or `esm2_t30_150M_UR50D` (~600 MB). Both still fit on 8 GB M2.
- For a stronger local proxy that gets closer to "real" binding intuition, use **ESMFold** to predict complex structures and compute interface contact area. That needs ~10 GB RAM, marginal on M2 — better to use cloud directly.
