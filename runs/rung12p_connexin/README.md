# RUNG 12P / Part A — the death-wave CONTAINMENT screen (propagation arm)

**The idea (Anshuman's).** Instead of every therapeutic agent independently recognising every tumour cell —
the per-cell wall R5→R11 mapped — engineer ONE cancer cell to recognise it's cancer, kill itself, and
**propagate** the apoptosis signal to its neighbours. A death wave that spreads cell-to-cell through the
tumour. Recognition then only has to succeed at a few **seed** cells; the wave does the rest. This
**decouples killing from per-cell recognition** and could bypass the whole addressability ceiling — *if the
wave is contained to the tumour.*

## The make-or-break question (asked of the atlas, not guessed)
A **passive** death wave travels through **gap junctions (connexins)** — the established bystander-effect
channel (HSV-TK/ganciclovir). It is contained only if a connexin is expressed in **tumour** (wave
propagates) but **absent in vital normal tissue** (wave can't enter heart/liver/brain). So:

> Is there ANY connexin worst-donor **VITAL-LOW across ALL vital cell types** (a containable channel) that is
> also **TUMOUR-expressed** (usable)?

- **No connexin vital-silent everywhere → every channel leaks into some vital tissue → a passive
  gap-junctional death wave CANNOT be contained → the relay MUST be RECOGNITION-GATED per hop** (synNotch-style
  AND-gate). Decisive; motivates Part B. *(Expected: heart = Cx43-rich, liver = Cx32/Cx26-rich.)*
- **Vital-silent + tumour-expressed connexin exists → candidate passive containable channel** (surprise +) →
  route to Part B percolation sim.

## How to run (Colab, CPU, no GPU)
Open `notebooks/rung12p_connexin_colab.ipynb`, **Run all** (same Google account as RUNG-5 so the tumour cache
is reused). Resumable Drive tiles + heartbeat; ~27 connexin/HK genes × vital cells over 9 vital tissues.
Bundle with `python scripts/archive_colab_run.py --commit`.

## The RUNG-8 trap, handled
CELLxGENE returns 0 for both "measured & zero" and "gene not measured here." RUNG-10b hunted HIGH (dropout
deflates → safe). This hunts **LOW** (vital-silent), where dropout/unmeasured would FAKE a containable channel
— the exact RUNG-8 v1 artifact. **Fix: a housekeeping depth control** — a connexin counts as "off" in a cell
only among **deep** cells (≥1 of ACTB/GAPDH/RPL13A/MALAT1 detected). Shallow cells can't vote "silent."
"Leak" / "silent" use the **worst-case (top-expressing) donor** (never-pooled safety ethos): a channel
expressed in even one patient's heart is a leak risk for that patient.

## Honest ceiling
mRNA ≠ functional gap-junction coupling (a connexin transcript ≠ an open channel); the HK-deep filter
mitigates but doesn't erase dropout, and LOW calls are the anti-conservative direction so that control is
load-bearing; tumour-side connexin coverage comes from the cached RUNG-5 surfaceome panel — connexins absent
from that panel are reported "unknown," not "absent." Scope: passive connexin/pannexin channels (doesn't rule
out engineered gated relays — that's Part B).

## Result (real run, 11 vital cell types across 9 tissues)
**DECISIVE NEGATIVE — a passive gap-junctional death wave cannot be tumour-contained, for two independent reasons:**
1. **The tumour barely couples.** Its most-expressed gap-junction connexin, **Cx43/GJA1, is in only 5.6% of
   malignant cells** and *no* connexin clears 10% — the textbook **loss of gap-junctional communication in
   cancer**. A passive wave can't even spread *through* the tumour.
2. **The one channel it has leaks.** Cx43 is worst-case-donor-high in **8 of 11 vital cell types** (neuron 82%,
   lung 70%, vascular endothelium 68%, cardiomyocyte 31%, kidney 41%).

The only connexins that are vital-silent everywhere (GJA8/Cx50, GJA10/Cx62, GJD3, GJD4) are **eye-restricted
lens/retina channels absent from the tumour surfaceome** — biologically irrelevant as a tumour coupling channel
(their tumour status is formally *unmeasured*, not measured-absent — an honest residual).

**Implication → Part B design is now sharper:** propagation must be **(a) recognition-gated per hop** *and*
**(b) use ENGINEERED coupling** (synNotch / engineered ligand–receptor such as TRAIL-relay), **not native gap
junctions** — because the tumour doesn't natively couple. The passive bystander route is dead.

## Provenance
`scripts/34_connexin_containment.py` (selftest 11/11). Reuses RUNG-10b/RUNG-8 Census machinery (`scripts/32`,
`scripts/30`, `scripts/25`). Census `2024-07-01`. Outputs: `rung12p_connexin.json`, `rung12p_connexin.png`.
Next: **Part B** — the percolation/relay simulation (does per-hop gating with a *leakier* recognition than R5
required still clear the tumour without a normal-tissue death wave — i.e. do errors cascade?).
