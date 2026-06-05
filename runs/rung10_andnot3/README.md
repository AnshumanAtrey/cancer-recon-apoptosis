# RUNG 10 / arm (a) — the GPU 3-input SURFACE AND-NOT sweep

**The confirmatory run RUNG-6 promised.** RUNG-5 found 0/1000 single + 2-input surface gates worst-donor-safe.
This asks the one case RUNG-5 didn't exhaustively cover: **does any 3-input SURFACE AND-NOT gate**
(`posA AND posB AND-NOT negC`) close the addressability gap — or does a negative prove the NOT-slot **must** be
a genetic-loss signal (HLA-LOH), not a surface marker?

## How to run
Open **`notebooks/rung10_andnot3_colab.ipynb`** in Colab, **Runtime → Change runtime type → T4 GPU** (this rung
is compute-bound — the GPU genuinely matters), **Run all**. Outputs: `rung10_andnot3.json` + `rung10_andnot3.png`.
Bundle filed with `python scripts/archive_colab_run.py --commit`.

## Design (maximal reuse — no new scorer, no new GPU code)
- **Panel:** the EXACT RUNG-5 atlas panel via `scripts/25`'s cached loaders. Cell 2 points `LOGICGATE_CACHE` at
  the same Drive path RUNG-5 used → `.r5.normal/.r5.tumour` load **instantly** if you ran RUNG-5 on that
  account; otherwise it re-fetches (resumable, per-tissue Drive tiles).
- **Gates:** 3-input AND-NOT family `{pos:[A,B], neg:[C]}` — A,B = top tumour-expressed surface genes, C =
  broadly-normal surface markers. Pruned (`R10_TOP_POS=120 × C(·,2) × R10_N_NEG=40`) to fit a T4 session; the
  pruning is stated, not silent (the full C(682,2)×682 ≈ 1.6×10⁸ is ~44 days CPU).
- **Scorer:** `scripts/22 opt.score_gates_vec` — the **same audited** worst-donor / Jeffreys-UB /
  fail-closed-vital / AND-NOT scorer RUNG-5 used, with its existing CuPy GPU path (`R5_GPU=1`). Identical
  semantics ⇒ a valid apples-to-apples extension of RUNG-5.

## Engineering
- **Resumable** per-batch checkpoints (`rung10_sweep_ckpt.json` on Drive) — a disconnect resumes mid-sweep.
- **Foreground heartbeat** + per-batch progress.
- **GPU earns its place** here (5×10⁵+ gates × 1.26M cells); clean CPU fallback if no GPU (much slower).

## Honest ceiling
Transcript-level (mRNA ≠ surface protein). A surviving gate is a **HYPOTHESIS** needing the full
FDR/permutation/bootstrap rigor (`scripts/23`) + wet-lab agonism, **not a cure**. Expected result is a
negative that strengthens the genetic-NOT-gate thesis (RUNG-6/7/8). selftest 9/9.

## Provenance
`scripts/31_andnot3_surface_sweep.py`. Reuses `scripts/25` panel + `scripts/22` scorer (both audited). Census
`2024-07-01`. Logs in `runs/logs/`; immutable per-run archive in `colab_runs/`.
