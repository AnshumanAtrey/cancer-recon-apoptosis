# CAPSTONE — the honest, end-to-end synthesis (RUNG 1–22)

*What we set out to test, what 22 in-silico rungs actually found, and where the irreducible wet-lab line sits.*
Every number below is a **prediction or simulation**, not biology. Honest negatives are kept as first-class.
This synthesises the public rungs in `README.md`; the source concept is Shriya Rai's (private).

---

## 0. The question

Shriya's concept, in one line: *a cancer cell should recognise itself (or a cancer neighbour) as abnormal and
trigger its own self-destruct (apoptosis), and that death should spread.* We asked: **does this chain hold up,
even on a computer, before any lab — and where exactly does it break?** We split it into the three stages she
named — **recognition → binding → killing/apoptosis** — and tested each to destruction.

---

## 1. The three-stage chain — what each stage gave

| Stage | Question | Result | Honest residual |
|---|---|---|---|
| **Recognition** | Is there a tag on cancer and on *nothing* healthy? | **Surface route DEAD** (every surface marker leaks into a vital organ — RUNG-5/15, 0/25 safe). **Mutation route WORKS** (neoantigens are tumour-exclusive — RUNG-11). High-mutation cancers carry ≥1 clean clonal handle in most patients (RUNG-16: MSI-CRC 99%, melanoma 81–100%, … PDAC 20–68%). | personalised (per-patient), not off-the-shelf |
| **Binding** | Will a T-cell actually grip the tag? | **Safety ↔ immunogenicity ALIGN** (RUNG-17): a tumour-exclusive handle is automatically high on agretopicity (the dominant recognition driver) — being safe *is* being recognisable. Boltz-2 independently confirms the mutants present on MHC (RUNG-20). | predicted propensity, not a proven TCR (MAGE-A3 warning) |
| **Apoptosis / spread** | If one cell dies, does death spread and clear the tumour without spilling? | **Death wave validated** (RUNG-13): snaps on, irreversible, spreads, stays contained. Arena: quorum/wave/ferroptosis lead (RUNG-14). | coupling/delivery = wet-lab |

**Stage verdict:** all three stages hold up in-silico, each with a stated residual. The chain is *coherent*.

---

## 2. The reality check that nearly broke it — and didn't

The immune route silently assumes the cancer keeps its **MHC "display window"** lit. We refused to assume and
measured it:

- **RUNG-18 (genetics, 6,319 tumours):** window intact **78%** · dimmed (one allele, still works) **18%** ·
  fully dark (route dies) only **~4%**.
- **RUNG-18b (expression, 50,719 real lung cancer cells):** transcriptionally dark **12.6%** vs 0.5% in an
  immune-cell control (metric validated) — **~2× the genetic number.** So genetics *under-counts* window-loss,
  but the window is still **lit in ~85–90%** of cancer cells, and the extra loss is the *reversible* (IFN /
  epigenetic) kind.

**Verdict:** the load-bearing assumption is broadly valid — not universal (a small permanently-dark core
exists), and genetics alone under-counts ~2×. We corrected our own confidence with data instead of assuming.

---

## 3. The hard problem — evolution — and the two-route answer

**RUNG-19 (escape race):** a single-target recognition-gated wave **cannot cure an established tumour.** Cure
collapses once expected resistant founders L = μ·N₀ cross ~1; a 1 cm tumour (~10⁹ cells) is always past that.
So the bare wave is necessary but **not sufficient.** This is the honest pivot — and it has exactly **two
escape routes**, each closed by a different layer:

```
                         ESCAPE ROUTE                          CLOSED BY            EVIDENCE
  Route 1: lose the neoantigen target (antigen loss)   →   multi-target / essential   RUNG-22
  Route 2: lose MHC entirely (window goes dark)        →   NK cross-kill + wave        RUNG-21
```

- **RUNG-21 (cross-kill):** a layered **T + NK + bystander wave** cures **100% across the measured escapee
  range (4–13%)**. The tumour is *trapped*: keep MHC → T-cells kill; drop MHC → NK kills ("missing-self").
  **All three layers are load-bearing** — remove NK and cure collapses 1.00→0.07 on the dark escapees.
- **RUNG-22 (multi-target):** escape collapses **exponentially** with the number of independent targets —
  K=3 → escape ~0 at clinical size; **one essential (un-losable) driver → escape-proof.** Rule: **≥3
  independent neoantigens OR ≥1 essential clonal driver.**

**The integrated logic (the "ideal layered system"):** a cell only fully escapes if it *independently* (a)
loses all K targets **or** loses MHC, **and** (b) evades NK, **and** (c) escapes the agnostic wave. Each is a
rare, independent failure → the probabilities **multiply** → escape driven toward zero. No single layer wins;
the *combination* leaves no route uncovered. This is exactly the multi-layer defence the field is converging
on — quantified, per layer, by our rungs.

---

## 4. The actionable design that falls out

> **Per patient (high-mutation cancer): target ≥3 clean clonal neoantigens — prefer essential drivers
> (KRAS/TP53/IDH1 class) — delivered as a personalised vaccine / TCR-T, PLUS an NK-engaging arm for the
> MHC-loss escapees, PLUS a resistance-agnostic bystander killer (ferroptosis/quorum). Each layer covers
> another's blind spot.**

- **Recognition:** clean clonal neoantigens (RUNG-11/16). Screen-priority handles: CTNNB1-S37F, EGFR-L858R,
  TP53-R248Q (RUNG-17); essential drivers IDH1-R132H / KRAS-G12D / TP53 (RUNG-12/20, presentation Boltz-confirmed).
- **Binding:** safety↔immunogenicity align → the safe handles are the recognisable ones.
- **Killing:** T-cell granzyme → caspase-3 = the cell's *own* apoptosis (Shriya's self-destruct, preserved).
- **Anti-escape:** multi-target + essential (Route 1) · NK + wave (Route 2).
- **Cancer is not one disease:** this works where mutation burden is high (melanoma, lung, MSI-CRC, bladder);
  low-TMB tumours (PDAC, most breast) lack the clean handles — for those, the autonomous / metabolic windows
  (TODO) or Shriya's MHC-independent self-destruct are the backup.

---

## 5. The irreducible wet-lab line (stated, never papered over)

Everything above is in-silico. What a computer *cannot* close, and a lab must:
1. **Does a real TCR exist** for each predicted handle? (RUNG-17/20 give propensity + presentation, not a receptor.)
2. **Can the trigger be delivered** into the body, and at what reach? (RUNG-13 says one seed suffices in
   principle; the delivery fraction is a wet question — TODO wave-as-injection.)
3. **Proteome-wide mimicry** — no MAGE-A3-style healthy look-alike. (RUNG-20 checked mut-vs-self only.)
4. **Real NK efficiency & exhaustion** in the immunosuppressive microenvironment (RUNG-21 models it as a parameter).

---

## 6. What this is, honestly

**Not** a cure, **not** biology. It is a **rigorous, falsifiable, public map** of where Shriya's recognition-
triggered self-destruct can work, where it can't, and what a complete therapy must layer to beat evolution —
with a quantitative bound at every stage and the wet-lab residual named at each. The headline finding:

> **The chain works in-silico at every stage; the bare recognition-gated wave is bounded by evolution; and a
> layered system (multi-target + essential drivers + NK cross-kill + agnostic wave) closes both escape routes
> — each layer covering another's blind spot.**

We found the door, checked it isn't locked, mapped every bolt, and specified the key. We have not walked
through it — that's the lab's step.

---

*Rungs: see `README.md` (hypothesis catalog + per-rung results). Next experiments: `TODO.md`. Source concept &
strategy: private (`memory/`, gitignored).*
