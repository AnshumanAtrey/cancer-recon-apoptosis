# RUNG 4 / Step-5 — logic-gate recognition designer (methodology + honest result)

**One line:** the selectivity layer that solves the wall Step-3 and RUNG-3 keep hitting — *no single
tumour-exclusive antigen exists*. It searches **AND / AND-NOT antigen combinations** and scores each by the
single most dangerous question, at **true single-cell resolution**: *is there any normal cell — especially
in heart / brain / kidney — that wears both badges at once?* A gate that any vital cell co-expresses is
thrown out.

This rung was designed + adversarially critiqued **before** coding. The critiques caught real flaws; the
fixes are in the build.

---

## What's validated this turn (real, locally run — all 8 integrity checks pass)

The **real antigen discovery needs the CELLxGENE single-cell atlases**, which (like Step-2's data) are GBs
and run on **Colab**, not committed. So this turn validates the **method** against *biological ground
truth* — a synthetic single-cell panel whose per-cell expression encodes externally-known facts. The
engine (`scripts/18`) must get the known answers right (`scripts/20`, RUN-TRUST):

| control | expected | result |
|---|---|---|
| **HER2 alone** | unsafe (on cardiomyocytes — Step-3 / the 2010 HER2 CAR-T death) | ✓ NON-SELECTIVE (vital) |
| **Trop2 alone** | unsafe (broad on normal epithelium) | ✓ NON-SELECTIVE (regen ceiling) |
| **same-cell vital pair** (co-positivity true-positive) | unsafe | ✓ NON-SELECTIVE (vital) |
| **Trop2 AND tumour-restricted partner** | safe | ✓ SELECTIVE |
| **HER2 AND-NOT HLA-A\*02-LOH** (Tmod model) | safe | ✓ SELECTIVE |
| **bulk-trap pair** (A, B on *different* liver cells) | single-cell safe; bulk would condemn | ✓ SELECTIVE; pseudobulk falsely sees **49%** vs single-cell **1%** |
| **NOT on an undetectable blocker** | flagged unfalsifiable | ✓ flagged |

**Sensitivity 100%, specificity 100%, RUN-TRUST PASS.** The engine correctly implements the safety logic —
including re-deriving our own Step-3 HER2→heart result from a totally different data construction. **This is
method validation, NOT a discovered gate.** No clean combination is *claimed* until the Colab atlas run.

---

## The three adversarial fixes (why you can trust it)

1. **The bulk illusion (the #1 way this is secretly fake).** Two antigens both "in the liver" is meaningless
   — if A is on hepatocytes and B on endothelium the gate is safe; if one hepatocyte has both, it's lethal.
   The engine computes **per-cell co-positivity** and reports the bulk-vs-single-cell gap explicitly. The
   benchmark proves it: a pair on *different* liver cells reads **49% "co-expressed"** to a pseudobulk method
   but **1%** to single-cell — bulk data would discard a safe gate (or, worse, hide a lethal one).
2. **Dropout & mRNA≠protein.** A single-cell zero is *undetected*, never *proven absent* — so the engine
   only credits a NOT/absence where the gene is robustly detectable elsewhere (else it's flagged
   *dropout-unfalsifiable*). And mRNA isn't surface protein (single-cell r~0.1–0.4): every gate is tagged
   **transcript-only** until CITE-seq confirms co-positivity; HPA IHC is **veto-only** (it can fail a gate,
   never bless one).
3. **Multiple testing + escape fragility.** The partner pool is kept small (curated, not a 2.5M-pair blind
   sweep). The valid multiple-testing control here is **held-out-donor replication** — that is **deferred to
   the next pass** (it needs a donor column), so a real run's selective gates are an explicit **DISCOVERY
   shortlist** pending replication, not confirmed hits (we do *not* claim a scrambled-label null in the real
   run — for per-cell co-positivity it would falsely void genuinely-clean gates). Crucially, an AND gate is
   **more** fragile to tumour evolution than a single antigen — a subclone escapes by losing *either*
   badge. So **escape-durability is reported as a separate axis, never multiplied into selectivity** — they
   are in direct tension. The benchmark shows the AND-gate's coverage half-life is **~half** the single's
   (18 vs 35 divisions), and for our **contact** death-wave an antigen-negative escaper is **unreachable**
   (RUNG-3b) — strictly worse than a CAR. We report that cost beside the safety, not hidden inside it.

Plus the tiered safety rule (your insight): **heart/brain/kidney co-expression FORBIDS a gate; liver/gut/
marrow is tolerated up to a finite recovery ceiling.** And the **no-multiply HARD RULE**: recognition is a
*fourth* axis, never fused with RUNG-1 (timing) / RUNG-2 (clustering, refuted) / RUNG-3 (tissue dynamics).

### Audit hardening (after an adversarial code review)

A code-grounded audit of the data layer caught a **fail-OPEN safety bug** and other gaps that *predated*
the speed refactor (the refactor itself was verified result-equivalent). Fixed this pass:

- **FAIL-CLOSED on vital coverage.** Previously, if a vital organ wasn't captured, leak defaulted to 0 and
  the gate read SAFE — "we never looked at the heart" was indistinguishable from "the heart is clean" (the
  exact HER2-CAR-T death mode). Now any required vital type (heart/brain/kidney/pancreas/adrenal/muscle) that
  isn't adequately captured forces the gate to **UNCERTAIN**, never SELECTIVE. A unit control (remove
  cardiomyocytes → a clean gate must become UNCERTAIN) is in `scripts/20` and passes.
- **Asymmetric cap.** Vital-parenchyma cells are now kept in **full** (only abundant non-vital types are
  subsampled), so a rare lethal double-positive cardiomyocyte can't be statistically erased — and the
  cell-type mapping runs **before** the cap (a fixed ordering bug).
- **Upper-bound leaks.** Vital safety is gated on the **Jeffreys one-sided upper confidence bound** of the
  per-group co-positive rate, not the point estimate — a 0/40 group reads ~7% upper, not 0%.
- **Restored all 9 vital/normal tissues** (pancreas, adrenal, skeletal muscle were silently dropped).
- **Honest controls.** Multiple-testing control (held-out-donor replication) is **deferred** — selective
  gates are a discovery shortlist; we no longer claim a control that doesn't run.

---

## Honest differentiation — we are NOT first at combinatorial search

Computational combinatorial-target search already exists: **Dannenfelser 2020** (bulk TCGA+GTEx, gene-level,
purity-confounded), **Kwon & Kang 2023** (single-cell AND/OR/NOT, surface-validated — the strongest
precedent), **MadHitter/Hooper** (ILP set-cover, optimizes OR coverage), **LogiCAR 2025** (GA over
single-cell surfaceome). **What's genuinely ours is integration, not the algorithm:** (1) it inherits the
project's *cell-type-resolved vital-parenchyma safety audit* (Step-3) + the tiered regenerating rule —
stricter than "low in bulk normal"; (2) it feeds a **recognition→apoptosis contact pipeline, not a CAR**, so
we simulate the antigen-loss/escape dynamics the AND-gate *creates* and report durability beside selectivity;
(3) it ships open, runlog-stamped, pre-registered, with **"no clean gate exists" as a first-class outcome**.

---

## How it's built

- `scripts/18_logicgate_search.py` — the engine (per-cell binarize, AND/AND-NOT firing, worst-case normal
  leak with vital broken out, NOT dropout-falsifiability, tiered safety, bulk-trap report, escape-durability,
  no-multiply assert). Pure numpy — runs locally and on Colab.
- `scripts/20_logicgate_calibration.py` — RUN-TRUST validation on biological ground truth (this turn's result).
- `scripts/17_logicgate_data.py` — the **Colab** data layer: streams normal per-tissue single-cell (all 9
  vital tissues) + tumour from CELLxGENE Census, emits the **vital-coverage census** (under-sampled vital
  types flagged **UNAUDITED**), scores the candidate gates on real cells.
- `notebooks/rung4_logicgate_colab.ipynb` — Colab driver.

Reproduce the validation: `python scripts/20_logicgate_calibration.py` (CPU, seconds).

---

## Your wet-lab directive, per assay

The tool **shrinks millions of antigen guesses to the one or two worth testing** — it cannot certify safety.

| wet-lab question | in-silico contribution | irreducible residual | cheapest path |
|---|---|---|---|
| do A & B sit on the same *protein-level* cell? | transcript co-positivity screen | mRNA≠protein (r~0.1–0.4) | normal-tissue **CITE-seq** (both in panel) — the only co-positivity-confirming data |
| is the gate safe on a real vital cell? | single-cell leak prediction + UNAUDITED flags | a rare cell *state* no atlas captured | **flow cytometry** on primary heart/kidney cells |
| does the engineered AND-circuit actually fire? | nothing (assumes the circuit works) | signal integration + DR5 clustering → caspase-8 | **co-culture** of the logic-gated construct (cloud lab / university core) |

---

## What RUNG 4 does NOT claim

- **Not a discovered gate** (this turn validates the *method*; discovery is the Colab run).
- **Not proof of safety** — transcript-level, mRNA≠protein, and "no normal cell co-expresses both" is only as
  true as the atlas is complete/deep (hence the UNAUDITED flags + snRNA-seq requirement for heart/brain).
- **Not a working circuit** — co-localisation ≠ a synNotch/Tmod circuit that integrates two signals and
  clusters DR5 to fire caspase-8 (wet-lab).
- The recognition score is **never multiplied** with RUNG-1/2/3 or with escape-durability.

**Next:** run `scripts/17` on Colab for the real discovery; add `scripts/19` (CITE-seq protein cross-check)
and `scripts/21` (full clonal antigen-loss durability sim). The most likely, equally-valid outcome is **"no
clean combinatorial gate exists for this tumour"** — published honestly, not forced into a fake winner.
