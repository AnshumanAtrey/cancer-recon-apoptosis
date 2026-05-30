# RUNG 3 — real-physics tissue: does the recognition-gated death wave clear tumour AND spare healthy?

**One line:** a pure-Python reaction-diffusion + agent tissue sim where **healthy-cell death is DERIVED**
from a real diffusing death-effector field crossing the RUNG-1 commitment threshold (or from contact with
a Trop2+ neighbour) — so the abstract ABM's "0% healthy killed" can actually **break**. It establishes a
physics-conditioned **safety design-envelope**, NOT patient efficacy.

This rung was designed + adversarially critiqued **before** coding. The critiques caught a deep flaw and
it is fixed in the build (below).

---

## The result (real, locally run, all 8 methodology-integrity checks pass)

**Two honest findings:**

1. **Propagation needs a CONTACT mechanism.** A freely-diffusing, death-*released* effector is
   latency-limited (~one cell-layer per death-generation, set by RUNG-1's ~3.7 h commitment delay) and
   **fizzles across all tested ranges** (L_eff 10→200 µm, high dose) — it clears <10% of even a 150 µm
   focus in a viable timeframe. A soluble "secrete-and-sweep" killer doesn't propagate.

2. **Safety needs a TUMOUR-EXCLUSIVE badge.** The contact/juxtacrine wave clears the tumour (~99%) — but
   it spares healthy tissue **only when the recognition antigen is tumour-restricted**:

   | Trop2+ fraction on *normal* epithelium | healthy killed | outcome |
   |---|---|---|
   | 0% (tumour-exclusive ideal) | 0% | **clear + spare** |
   | 10% | 1% | clear + spare |
   | 30% | 5% | **leaks** |
   | 50% | **45%** | **leaks badly** |

   Because **real Trop2 is broadly expressed on normal epithelium** (our own Step-3 finding), the realistic
   case is the leaky one. The death wave percolates into Trop2+ normal tissue (see the figure's top-right panel).

**Verdict:** the recognition-gated death wave is viable **only** as *contact modality* **+** *tumour-exclusive
recognition*. Trop2 alone is not tumour-exclusive → this independently reproduces the Step-3 conclusion that a
single antigen lacks a clean window and **combinatorial logic-gating** (require 2 antigens) is needed. Two of
our results now converge from different scales.

---

## Why you can trust it (the three adversarial fixes)

The critiques caught that the original design was **rigged** — and the fixes are in the code:

1. **The contact arm was tautologically safe** (healthy cells hard-coded Trop2-negative = the exact
   `killable=(state==CANCER)` artifact that made scripts/08-10's "0% healthy" unfalsifiable). **Fixed:**
   the Trop2+ *healthy* fraction is a swept axis, plus a shed-fragment soluble leak — so the contact arm
   **can** kill healthy, and does (45% at 50% Trop2+). Healthy-kill is derived from the mechanism, never a boolean.
2. **The concentration scale was a circular free knob** (healthy-kill = field/threshold ratio, threshold
   defined via the field). **Fixed:** the scale is anchored to a physical observable — molecules/voxel →
   nM via Avogadro and the voxel volume — and the threshold to ~2× the TRAIL EC50 (~1.5 nM). The verdict is
   robust across a ±3× threshold sweep (the contact-clean case stays safe; right panel).
3. **The RUNG-1 latency could be silently re-fit.** **Fixed:** the kinetics are read from
   `runs/earm_kinetics/earm_results.json` (Td_mean 3.774 h, CV 0.155, threshold table); the lognormal
   sampler-mean is matched to the recorded mean; the latency is **dose-coupled** (threshold-edge cells get
   the slow ~10 h delay, saturated cells ~3.1 h, from the RUNG-1 dose→Td table). A latency-OFF control
   (instantaneous death) differs measurably (t_end 14 h vs 1 h) → RUNG-1 is mechanically load-bearing, not decorative.

Plus: a **field-solver oracle check** (point release spreads as ⟨r²⟩ = 4Dt within 5%, mass conserved <1%);
real units throughout (µm, h, nM); and the **no-multiply HARD RULE** asserted in code (the RUNG-1 latency is
never multiplied by the RUNG-2 — refuted — clustering score; three separate axes: death-timing / recognition
gate / clustering).

---

## How it's built

`scripts/15_tissue_rd.py` — the engine: explicit finite-difference diffusion (zero-flux BC) + analytic
decay/receptor-sink reaction; agent cells as numpy state arrays; both **soluble** (field-derived kill on any
DR5+ cell) and **contact** (juxtacrine, Trop2-gated, commitment-propagated) modality arms; RUNG-1 dose-coupled
latency from file; the Gaussian oracle self-check; the no-multiply assert. Reproduce:
`python scripts/15_tissue_rd.py` (CPU, ~45 s). Colab mirror: `notebooks/rung3_tissue_rd_colab.ipynb`.

---

## Your wet-lab directive, answered per assay

The sim **prioritizes which design to pursue** (contact modality + tumour-exclusive logic gate); it does
**not** replace the experiments. Per assay (in-silico proxy | irreducible residual | cheapest student path):

| wet-lab assay | what RUNG 3 contributes | irreducible residual | cheapest real path |
|---|---|---|---|
| **Bystander-containment co-culture** (Trop2+/Trop2− spheroid: does death stay in the tumour compartment?) | this sim newly **motivates** it as the #2 assay — quantifies the leak it predicts | real membrane diffusion, actual normal-epithelium Trop2 levels, 3-D geometry | a Trop2+/Trop2− organoid/spheroid co-culture at a university core or low-cost CRO (*cost UNVERIFIED*) |
| **Caspase-8 firing** (Caspase-Glo 8) — the agonism crux | none (the sim **assumes** ignition) | whether a Trop2-anchored DR5 binder fires caspase-8 at all | Emerald Cloud Lab / India CRO (*access UNVERIFIED*) |
| **Graded-cluster ignition-size assay** | predicts a critical ignition focus; sim gives the hypothesis | real tissue ignition threshold | spheroids of graded size (university core) |

**Bottom line:** RUNG 3 converts "the wave clears everything and spares everyone" (an unfalsifiable ABM
artifact) into an **earned design constraint** — contact modality + tumour-exclusive (logic-gated)
recognition — and a sharpened, prioritized wet-lab handoff.

---

## What RUNG 3 does NOT claim

- **Not patient efficacy.** It is a safety/dynamics **regime map** over assumed, literature-borrowed
  parameters (D, half-life, density, threshold) — a sensitivity landscape, not a calibrated prediction.
- **Not agonism.** It assumes ignition is possible; whether a binder fires caspase-8 is the wet-lab crux.
- **2-D primary.** A 3-D slab would leak more (surface-to-volume); the Trop2+ leak finding is conservative in 2-D.
- The RUNG-1 latency is **never** multiplied by the RUNG-2 clustering score — separate axes, separate ceilings.

See `EVIDENCE_AND_HANDOFF.md` for the project-wide evidence ceiling.
