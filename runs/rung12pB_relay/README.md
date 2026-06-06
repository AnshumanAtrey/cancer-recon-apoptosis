# RUNG 12P / Part B — the gated-relay death-wave simulation (the Iron-Man question)

**Setup.** Part A killed the *passive* route (tumours barely couple — Cx43 in ~6% of malignant cells — and
that Cx43 leaks into 8/11 vital tissues). So the death wave must use **engineered coupling** and **re-check
tumour identity at each hop**. Part B asks the payoff question:

> Does a per-hop-gated death wave clear the tumour while sparing normal tissue — and is its safe operating
> region **bigger than the per-cell gate's**, the one R5 found *no* surface gate could pass?

## Model
2D (and 3D sensitivity) **site-bond percolation**. A cell dies + relays iff it is connected to a **seed**
through (a) **coupled** edges (engineered coupling, prob `c`) and (b) **permeable** cells — cells that pass
their per-hop recognition gate (true-positive `q_t` in tumour, false-positive `q_n` in normal). The per-cell
death effector is RUNG-1's EARM bistable switch (modular; binarised here).

- **Tumour:** effective transmissibility `c·q_t`. Above threshold → super-critical → the wave sweeps the whole
  disk from a few seeds (efficacy *decoupled* from per-cell recognition: seed once, spread).
- **Normal:** effective `c·q_n`. Below threshold → sub-critical → any false-positive normal cell starts a wave
  that **dies out in a bounded rind**. Errors don't cascade.

## Result — POSITIVE: the architecture rescues leaky recognition
- **Relay converts a *linear* leak into a *threshold-bounded* one.** A per-cell gate kills normal tissue
  linearly in its false-positive (every false-positive cell, anywhere, dies → leak = `q_n`). The relay only
  kills normal cells on a percolating path from the tumour — below threshold, a thin bounded rind. At
  **q_n = 0.30** the relay leaks **0.8%** of normal tissue while a per-cell gate would kill **30%**.
- **Safe per-hop false-positive ceiling (≤1% normal kill):** **2D q_n ≈ 0.30 (~15× the R5 per-cell bar of
  0.02); 3D q_n ≈ 0.17 (~8.6×)** — the conservative, realistic-tissue number, still *well* above 0.02.
- **Tumour clears ~89%** from just a **3% seed** throughout (super-critical), and a 7% slice of the (q_t, q_n)
  grid is both cleared (>90%) and spared (<1%).

**Bottom line:** propagation can convert a recognition signal **too leaky for R5's per-cell gate** into a safe
therapy — *if* coupling is engineered and gating is re-checked per hop. **The recognition bottleneck is
RELAXED, not removed** (q_n must stay below the percolation threshold). This unifies the project: the RUNG-11
`tcr_dependent` neoantigen handles that were *too risky* for a per-cell TCR-T (WT cross-reactivity ≈ per-cell
false-positive) become *usable* as a per-hop relay gate — because an ~8–15× leakier gate is now tolerable.

## Honest ceiling
An **abstraction** (2D/3D site-bond percolation): `q_t`, `q_n`, `c` are **swept parameters**, not measured
molecular fidelities. The robust, parameter-free claim is the **relationship** (relay leak sub-critical-bounded
vs per-cell linear → a threshold-protected margin), not the exact ×-factor (which depends on `c`, the 1% leak
criterion, and `p_c`). Mapping `q` to a real gate (the RUNG-11 neoantigen gate / a synNotch relay) and `c` to
an engineered coupling efficiency is the **wet-lab residual**. No diffusion kinetics, immune clearance,
graded signals, or multi-focal disease. 3D narrows the window (modelled); finer geometry/kinetics could narrow
it further.

## Provenance
`scripts/35_propagation_relay.py` (selftest 8/8). NumPy + SciPy `connected_components`. Laptop, seconds, no
GPU/atlas. Outputs: `rung12pB_relay.json`, `rung12pB_relay.png`.
