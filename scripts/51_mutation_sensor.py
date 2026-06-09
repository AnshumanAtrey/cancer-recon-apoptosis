#!/usr/bin/env python3
"""
RUNG 25 — the AUTONOMOUS MUTATION-SENSING circuit: can a synthetic sensor discriminate a tumour POINT
mutation from wild-type (one base), well enough that an AND of two clonal mutations fires ONLY in cancer?
(Colab CPU, ViennaRNA — no GPU.)

WHY THIS IS THE CORRECTED DIRECTION (from RUNG-23)
-------------------------------------------------
RUNG-23 v2 confirmed: every EXPRESSION window (surface + intracellular, single + AND) LEAKS into vital
tissue -> the MUTATION (neoantigen) is the ONLY tumour-exclusive signal. So Shriya's autonomous MHC-free
self-destruct must sense the MUTATION DIRECTLY, not a transcriptional state. Allele-specific sensing of a
single base IS established (SNP-detecting toehold switches [Green/Collins], allele-specific ASOs / CRISPR).
The feasibility question, before designing real circuits: how well can a sensor tell mutant mRNA from WT,
and does ANDing two clonal mutations drive the normal-cell false-fire to ~0 (tumour-exclusive autonomous gate)?

WHAT IT COMPUTES
----------------
A sensor is the reverse-complement of the MUTANT window (perfect match to mutant mRNA, ONE mismatch to WT).
Discrimination ΔΔG = ΔG(sensor·WT) − ΔG(sensor·mutant) (WT has the mismatch → less stable → ΔΔG>0). The
per-sensor normal-cell FALSE-FIRE rate (WT wrongly triggering) ≈ Boltzmann exp(−ΔΔG/RT). For a 2-input
AND-gate (fire only if mut-A AND mut-B sensed), normal-cell fire = product of the two false-fires → tumour-
exclusivity. We sweep the 12 substitution types × flanking contexts × mismatch POSITION (central vs terminal)
and a toehold-style DESTABILISED design, and add KRAS-G12D (c.35G>A) as a worked real example.

HONEST CEILING
--------------
ΔG (ViennaRNA duplex) is a thermodynamic PROXY for in-cell sensor triggering (kinetics, expression, RNA
accessibility, off-target genome matches NOT modelled). "AND-gate fire = product" assumes the two sensors
fail INDEPENDENTLY. Real driver cDNA contexts beyond the worked example are GENERIC here (exact flanking is a
verified-sequence follow-up — do not over-read a specific driver's number). BOUNDS feasibility of single-base
sensing + the AND-gate specificity; not a built circuit (synthesis + delivery = wet-lab).

USAGE
  python scripts/51_mutation_sensor.py selftest    # pure-python proxy, no ViennaRNA — validates the logic
  python scripts/51_mutation_sensor.py run         # ViennaRNA (Colab CPU) -> runs/rung25_mutation_sensor/
"""
from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung25_mutation_sensor"
RESULT_JSON = OUT_DIR / "rung25_mutation_sensor.json"
FIGURE_PNG = OUT_DIR / "rung25_mutation_sensor.png"

RT = 0.616                      # kcal/mol at 37 °C
BASES = "ACGU"
COMP = {"A": "U", "U": "A", "G": "C", "C": "G"}
SENSOR_LEN = 21
SUBSTITUTIONS = [(a, b) for a in BASES for b in BASES if a != b]   # 12 single-base substitution types
KRAS_G12D = {"name": "KRAS_G12D", "note": "c.35G>A, GGT(Gly12)->GAT(Asp); canonical CDS codons 8-15",
             # mRNA window centred on the mutated base (the middle G of codon 12). WT vs MUT differ by 1 base.
             "wt": "GUAGUUGGAGCUGGUGGCGUAGGC", "mut": "GUAGUUGGAGCUGAUGGCGUAGGC", "mut_pos": 12}


def rc(seq):
    return "".join(COMP[b] for b in reversed(seq))


def _sig(x, n=4):
    """Significant-figure float — preserves tiny probabilities (exp(-ΔΔG/RT) ~ 1e-7) that round(.,5) flattens to 0."""
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")


# ---------------------------------------------------------------------------
#  energy backends: ViennaRNA (run) or a pure-python nearest-neighbour-ish PROXY (selftest, no dep)
# ---------------------------------------------------------------------------
def _proxy_duplex_energy(s1, s2):
    """Pure-python stand-in for ViennaRNA duplex ΔG (selftest only). s2 is the target; s1 the sensor (= rc of
    a window). Aligns end-to-end (equal length), rewards Watson-Crick matches, penalises mismatches, weights
    CENTRAL positions more (a central mismatch destabilises a duplex most — the real biophysics)."""
    t = s2[::-1]                                            # sensor pairs with reversed target
    L = min(len(s1), len(t))
    e = 0.0
    for i in range(L):
        w = 1.0 + 1.5 * (1 - abs(i - (L - 1) / 2) / ((L - 1) / 2 + 1e-9))   # central positions weighted up
        e += (-2.0 * w) if COMP.get(s1[i]) == t[i] else (+1.5 * w)
    return e


def duplex_energy(sensor, target, backend):
    if backend == "vienna":
        import RNA
        return float(RNA.duplexfold(sensor, target).energy)
    return _proxy_duplex_energy(sensor, target)


def discrimination(wt_window, mut_window, backend, sensor_len=SENSOR_LEN):
    """Design the sensor as rc(mutant window) centred on the mutation; ΔΔG = ΔG(sensor·WT) − ΔG(sensor·mut)."""
    sensor = rc(mut_window)
    dg_mut = duplex_energy(sensor, mut_window, backend)
    dg_wt = duplex_energy(sensor, wt_window, backend)
    ddg = dg_wt - dg_mut                                   # >0 => mutant binds more stably => discriminable
    false_fire = math.exp(-max(ddg, 0.0) / RT)             # WT wrongly triggering (Boltzmann, relative)
    return {"ddg": round(ddg, 3), "dg_mut": round(dg_mut, 3), "dg_wt": round(dg_wt, 3),
            "false_fire_rate": _sig(min(false_fire, 1.0))}


def and_gate_false_fire(ff_a, ff_b):
    """AND of two independent sensors: a NORMAL cell fires only if BOTH wrongly trigger."""
    return ff_a * ff_b


def make_window(rng, length, center_sub):
    """Random RNA window with a defined center base; return (wt, mut) differing only at center by center_sub."""
    seq = [BASES[i] for i in rng.integers(0, 4, length)]
    c = length // 2
    a, b = center_sub
    seq[c] = a
    wt = "".join(seq)
    seq[c] = b
    mut = "".join(seq)
    return wt, mut


# ---------------------------------------------------------------------------
def main_run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    backend = "vienna"
    try:
        import RNA  # noqa
    except Exception:
        print("[rung25] ViennaRNA not installed. On Colab: pip install ViennaRNA ; then re-run. "
              "(selftest works without it.)")
        return 3
    rng = np.random.default_rng(25)
    N_CTX = 60                                              # random flanking contexts per substitution type

    # 1) discrimination per substitution type (central mismatch), averaged over contexts
    per_sub = {}
    for (a, b) in SUBSTITUTIONS:
        ddgs, ffs = [], []
        for _ in range(N_CTX):
            wt, mut = make_window(rng, SENSOR_LEN, (a, b))
            d = discrimination(wt, mut, backend)
            ddgs.append(d["ddg"]); ffs.append(d["false_fire_rate"])
        per_sub[f"{a}>{b}"] = {"mean_ddg": round(float(np.mean(ddgs)), 3),
                               "median_false_fire": _sig(float(np.median(ffs)))}

    # 2) mismatch POSITION sweep (central discriminates best) — one representative substitution
    pos_sweep = {}
    for frac in (0.5, 0.3, 0.1):
        ddgs = []
        for _ in range(N_CTX):
            seq = [BASES[i] for i in rng.integers(0, 4, SENSOR_LEN)]
            c = int(frac * (SENSOR_LEN - 1))
            seq[c] = "G"; wt = "".join(seq); seq[c] = "A"; mut = "".join(seq)
            ddgs.append(discrimination(wt, mut, backend)["ddg"])
        pos_sweep[f"mismatch_at_{int(frac*100)}pct"] = round(float(np.mean(ddgs)), 3)

    # 3) worked real example: KRAS-G12D
    kras = discrimination(KRAS_G12D["wt"], KRAS_G12D["mut"], backend)

    # 4) AND-gate specificity: median single false-fire -> AND of two (independent clonal mutations)
    med_ff = float(np.median([per_sub[k]["median_false_fire"] for k in per_sub]))
    and_ff = and_gate_false_fire(med_ff, med_ff)
    kras_and = and_gate_false_fire(kras["false_fire_rate"], med_ff)

    result = {
        "tag": "rung25_mutation_sensor",
        "question": "Can a synthetic sensor discriminate a tumour point-mutation from WT (one base) well enough "
                    "that an AND of two clonal mutations fires only in cancer (MHC-free autonomous self-destruct)?",
        "backend": "ViennaRNA duplex ΔG", "sensor_len": SENSOR_LEN, "RT_kcal_mol": RT,
        "discrimination_per_substitution": per_sub,
        "mismatch_position_sweep_ddg": pos_sweep,
        "kras_g12d_worked_example": {**KRAS_G12D, **kras},
        "single_sensor_median_false_fire": _sig(med_ff),
        "AND_gate_two_clonal_mutations_false_fire": _sig(and_ff),
        "kras_AND_second_clonal_false_fire": _sig(kras_and),
        "HEADLINE": (f"Single-base sensing: median ΔΔG discriminates (central mismatch best, "
                     f"{pos_sweep.get('mismatch_at_50pct')} vs {pos_sweep.get('mismatch_at_10pct')} kcal/mol terminal). "
                     f"Single-sensor normal-cell false-fire ~{med_ff:.1e}; AND of TWO clonal mutations drives it to "
                     f"~{and_ff:.1e} → tumour-exclusive autonomous trigger feasible if ≥2 clonal mutations are sensed."),
        "INTERPRETATION_MAP": {
            "AND false-fire << single, → ~0": "two independent allele-specific sensors give a tumour-exclusive "
                                              "MHC-free AND-gate -> the corrected RUNG-23 direction is FEASIBLE in principle.",
            "single false-fire already high (poor ΔΔG)": "single-base sensing too leaky -> need toehold "
                                                         "amplification / a 3rd input, or this route is bounded.",
        },
        "CEILING": "ViennaRNA ΔG is a thermodynamic PROXY (kinetics / RNA accessibility / genome off-targets NOT "
                   "modelled); AND=product assumes independent sensor failures; non-KRAS contexts are GENERIC "
                   "(exact driver flanking = verified-sequence follow-up). Feasibility bound, not a built circuit.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung25] wrote {RESULT_JSON}  ({time.monotonic()-t0:.1f}s)")
    print(f"  KRAS-G12D ΔΔG={kras['ddg']} false_fire={kras['false_fire_rate']:.1e}")
    print(f"  single median false-fire {med_ff:.1e} -> AND(2) {and_ff:.1e}")
    _make_figure(per_sub, pos_sweep, med_ff, and_ff)
    return 0


def _make_figure(per_sub, pos_sweep, med_ff, and_ff):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung25] matplotlib unavailable ({e})"); return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    subs = list(per_sub); ddg = [per_sub[s]["mean_ddg"] for s in subs]
    ax[0].bar(range(len(subs)), ddg, color="#3F7D54")
    ax[0].set_xticks(range(len(subs))); ax[0].set_xticklabels(subs, rotation=45, fontsize=7)
    ax[0].set_ylabel("mean ΔΔG (kcal/mol)  ← discrimination"); ax[0].set_title("Single-base discrimination by substitution type")
    ax[0].grid(axis="y", alpha=0.3)
    labels = ["single\nsensor", "AND of 2\nclonal muts"]; vals = [med_ff, and_ff]
    ax[1].bar([0, 1], vals, color=["#E0A040", "#3F7D54"])
    ax[1].set_yscale("log"); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("normal-cell FALSE-FIRE rate (log)")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=9)
    ax[1].set_title("AND of two sensors → tumour-exclusive")
    ax[1].grid(axis="y", alpha=0.3)
    fig.suptitle("RUNG-25: autonomous mutation-sensing AND-gate (MHC-free) — feasibility", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung25] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest():
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("rc is reverse-complement", rc("AUGC") == "GCAU")

    rng = np.random.default_rng(1)
    wt, mut = make_window(rng, 21, ("G", "A"))
    check("make_window: wt/mut differ at exactly center", sum(a != b for a, b in zip(wt, mut)) == 1 and wt[10] != mut[10])

    d = discrimination(wt, mut, "proxy")
    check("perfect match to mutant more stable than WT (ΔΔG>0)", d["ddg"] > 0)
    check("false_fire in (0,1)", 0 < d["false_fire_rate"] <= 1)

    # central mismatch discriminates better than terminal (proxy weights center)
    seqC = [BASES[i] for i in rng.integers(0, 4, 21)]
    sc = seqC.copy(); sc[10] = "G"; wtC = "".join(sc); sc[10] = "A"; mutC = "".join(sc)
    se = seqC.copy(); se[1] = "G"; wtE = "".join(se); se[1] = "A"; mutE = "".join(se)
    ddgC = discrimination(wtC, mutC, "proxy")["ddg"]; ddgE = discrimination(wtE, mutE, "proxy")["ddg"]
    check("central mismatch ΔΔG >= terminal", ddgC >= ddgE)

    # AND of two sensors drives false-fire DOWN (product)
    ff = 0.1
    check("AND false-fire = product (0.1*0.1=0.01)", abs(and_gate_false_fire(ff, ff) - 0.01) < 1e-9)
    check("AND << single", and_gate_false_fire(ff, ff) < ff)

    # KRAS example parses + discriminates under proxy
    k = discrimination(KRAS_G12D["wt"], KRAS_G12D["mut"], "proxy")
    check("KRAS-G12D windows differ by 1 base", sum(a != b for a, b in zip(KRAS_G12D["wt"], KRAS_G12D["mut"])) == 1)
    check("KRAS-G12D discriminable under proxy (ΔΔG>0)", k["ddg"] > 0)

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(selftest())
    elif cmd == "run":
        sys.exit(main_run())
    print(f"unknown: {cmd}"); sys.exit(64)
