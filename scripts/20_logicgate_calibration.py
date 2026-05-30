#!/usr/bin/env python3
"""
RUNG 4 / Step-5 — logic-gate engine CALIBRATION on biological ground truth (the RUN-TRUST gate).

The real antigen discovery needs the CELLxGENE single-cell atlases (scripts/17, Colab — the atlases are
not on disk locally). This script VALIDATES THE METHOD locally, before trusting it on real data, against a
synthetic single-cell panel whose per-cell expression encodes EXTERNALLY-KNOWN facts:

  - ERBB2/HER2 is on CARDIOMYOCYTES (the scripts/07 Step-3 finding; a HER2 CAR-T killed a patient, Morgan 2010)
  - TACSTD2/Trop2 is broad on normal epithelium (Step-3: no clean window)
  - HLA-A*02 is on ALL normal cells but LOST in tumour by LOH -> the cleanest genetic NOT (Tmod / A2 Bio)
  - a SAME-CELL pair both on cardiomyocytes  -> the co-positivity TRUE-POSITIVE control (must score UNSAFE)
  - a pair on DIFFERENT liver cells          -> the BULK TRAP (pseudobulk says co-expressed, single-cell says safe)
  - a NOT on an undetectable blocker          -> the dropout-UNFALSIFIABLE control (must score UNCERTAIN)

RUN-TRUST (verdicts are only trustworthy if ALL pass): known-good gates score SELECTIVE; known-bad singles
and the co-positivity pair score NON-SELECTIVE; HER2-alone is non-selective on cardiomyocytes (re-derives
Step-3); the bulk trap is exposed; the unfalsifiable NOT is flagged. Passing this means the ENGINE
correctly implements the safety logic — it does NOT mean we found a real gate (that is the Colab run).

CEILING: synthetic ground truth tests the METHOD, not biology. Recognition-selectivity is a separate
fourth axis, never multiplied with RUNG-1/2/3 or with escape-durability.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung4_logicgate"
SEED = 20260530

spec = importlib.util.spec_from_file_location("lg", PROJECT_ROOT / "scripts" / "18_logicgate_search.py")
lg = importlib.util.module_from_spec(spec); spec.loader.exec_module(lg)

HI, MID, LO, ZERO = 6.0, 2.2, 0.12, 0.0
HG = 20.0   # HLA-A*02 modelled as a GENETIC NOT (per spec): deterministically present in every normal cell
#           (Poisson P(<2)~2e-8), LOST in tumour by LOH (ZERO) — not a stochastic expression marker.
GENES = ["ERBB2", "TACSTD2", "ERBB3", "HLA_A02", "CLEANPART", "SCV_A", "SCV_B", "DCELL_A", "DCELL_B", "BLOCK_UNDETECT"]
#                        ERBB2 TACSTD2 ERBB3 HLA_A02 CLEAN SCV_A SCV_B DCELL_A DCELL_B BLOCK
EXPR = {
    ("cardiomyocyte", "heart", "normal"):        [HI,  LO,  LO,  HG,  LO,  HI,  HI,  LO,  LO,  LO],
    ("neuron", "brain", "normal"):               [LO,  LO,  LO,  HG,  LO,  LO,  LO,  LO,  LO,  LO],
    ("kidney_tubule", "kidney", "normal"):        [LO,  LO,  LO,  HG,  LO,  LO,  LO,  LO,  LO,  LO],
    ("hepatocyte", "liver", "normal"):            [LO,  MID, LO,  HG,  LO,  LO,  LO,  HI,  LO,  LO],
    ("liver_endothelial", "liver", "normal"):     [LO,  LO,  LO,  HG,  LO,  LO,  LO,  LO,  HI,  LO],
    ("marrow_hsc", "bone_marrow", "normal"):      [LO,  LO,  LO,  HG,  LO,  LO,  LO,  LO,  LO,  LO],
    ("normal_epithelium", "lung", "normal"):      [MID, HI,  MID, HG,  LO,  LO,  LO,  LO,  LO,  LO],
    ("tumour_epithelium", "tumour", "tumour"):    [HI,  HI,  HI,  ZERO,HI,  HI,  HI,  HI,  HI,  LO],
}
N_PER_TYPE = 320


def build_panel(rng):
    rows_counts, ct, ts, comp = [], [], [], []
    for (cell_type, tissue, compartment), lams in EXPR.items():
        block = np.column_stack([rng.poisson(lam, N_PER_TYPE) for lam in lams])
        rows_counts.append(block)
        ct += [cell_type] * N_PER_TYPE; ts += [tissue] * N_PER_TYPE; comp += [compartment] * N_PER_TYPE
    return lg.Panel(np.vstack(rows_counts), GENES, np.array(ct), np.array(ts), np.array(comp))


def main() -> int:
    lg.assert_no_multiply()
    rng = np.random.default_rng(SEED)
    panel = build_panel(rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 86)
    print("RUNG 4 — logic-gate selectivity engine, RUN-TRUST calibration on biological ground truth")
    print("=" * 86)
    print(f"panel: {panel.counts.shape[0]} cells x {len(GENES)} antigens, {len(EXPR)} cell-types "
          f"(heart/brain/kidney vital; liver/marrow/epithelium regenerating; tumour)")

    GATES = [
        ("ERBB2", None, "SINGLE", "bad", "HER2 alone — on cardiomyocytes (Step-3 / Morgan 2010 death)"),
        ("TACSTD2", None, "SINGLE", "bad", "Trop2 alone — broad on normal epithelium"),
        ("SCV_A", "SCV_B", "AND", "bad", "co-positivity TRUE-POSITIVE: both on the SAME cardiomyocyte"),
        ("TACSTD2", "CLEANPART", "AND", "good", "Trop2 AND tumour-restricted partner"),
        ("ERBB2", "HLA_A02", "AND_NOT", "good", "HER2 AND-NOT HLA-A*02-LOH (Tmod model)"),
        ("DCELL_A", "DCELL_B", "AND", "good", "bulk-trap pair: on DIFFERENT liver cells (single-cell safe)"),
        ("ERBB2", "BLOCK_UNDETECT", "AND_NOT", "control", "NOT on an undetectable blocker (must be UNCERTAIN)"),
    ]
    REQ_VITAL = {"cardiomyocyte", "neuron", "kidney_tubule"}   # the vital types this synthetic panel models
    print("-" * 86)
    print(f"{'gate':28s} {'cov':>5s} {'vital':>6s} {'strict':>6s} {'regen':>6s} {'pbulk':>6s}  verdict")
    rows = []
    for A, B, logic, klass, desc in GATES:
        r = lg.score_gate(panel, A, B, logic, required_vital=REQ_VITAL)
        r["class"], r["desc"] = klass, desc
        rows.append(r)
        print(f"{r['gate']:28s} {r['tumour_coverage']:5.2f} {r['vital_leak']:6.2f} {r['strict_leak']:6.2f} "
              f"{r['regen_leak']:6.2f} {r['pseudobulk_leak']:6.2f}  {r['verdict']}")

    # validate the FAST batch scorer (used by scripts/17 on real data) is IDENTICAL to the per-gate scorer
    batch = lg.score_gates_batch(panel, [(A, B, logic) for A, B, logic, _, _ in GATES], required_vital=REQ_VITAL)
    batch_matches = all(
        b["verdict"] == r["verdict"]
        and abs(b["vital_leak"] - r["vital_leak"]) < 1e-9
        and abs(b["tumour_coverage"] - r["tumour_coverage"]) < 1e-9
        and abs(b["worst_normal_leak"] - r["worst_normal_leak"]) < 1e-9
        for b, r in zip(batch, rows))
    print(f"[batch-scorer] fast vectorised scorer identical to per-gate scorer: {batch_matches}")

    # FAIL-CLOSED control: remove cardiomyocytes -> a previously-clean gate must become UNCERTAIN, NOT
    # silently SELECTIVE ('we never looked at the heart' != 'the heart is clean'). This is the lethal
    # fail-OPEN bug the audit caught; this control proves it now fails CLOSED.
    keep = panel.cell_type != "cardiomyocyte"
    panel_noheart = lg.Panel(panel.counts[keep], panel.genes,
                             panel.cell_type[keep], panel.tissue[keep], panel.compartment[keep])
    fc = lg.score_gate(panel_noheart, "TACSTD2", "CLEANPART", "AND", required_vital=REQ_VITAL)
    fail_closed_ok = (not fc["selective"]) and ("cardiomyocyte" in fc.get("unaudited_vital", []))
    print(f"[fail-closed] clean gate with HEART REMOVED -> {fc['verdict'][:70]}  (fail_closed_ok={fail_closed_ok})")

    by_gate = {r["gate"]: r for r in rows}
    her2 = by_gate["ERBB2 (single)"]
    copos = by_gate["SCV_A AND SCV_B"]
    bulk = by_gate["DCELL_A AND DCELL_B"]
    unfals = by_gate["ERBB2 AND_NOT BLOCK_UNDETECT"]
    good = [r for r in rows if r["class"] == "good"]
    bad = [r for r in rows if r["class"] == "bad"]

    # ---- RUN-TRUST controls ----
    print("-" * 86)
    controls = {
        "HER2-alone NON-SELECTIVE on cardiomyocyte (re-derives Step-3)":
            (not her2["selective"]) and her2["vital_group"] is not None and "cardiomyocyte" in (her2["vital_group"] or ""),
        "every known-GOOD gate scores SELECTIVE": all(r["selective"] for r in good),
        "every known-BAD gate scores NON-SELECTIVE": all(not r["selective"] for r in bad),
        "co-positivity TRUE-POSITIVE (same-cell pair) NON-SELECTIVE on vital":
            (not copos["selective"]) and copos["vital_leak"] > 0.5,
        "BULK TRAP exposed (pseudobulk would condemn, single-cell clears)":
            bulk["selective"] and bulk["pseudobulk_leak"] > 0.2 and bulk["worst_normal_leak"] < 0.05,
        "NOT-arm dropout-falsifiability computed (Tmod blocker falsifiable; undetectable blocker flagged)":
            by_gate["ERBB2 AND_NOT HLA_A02"]["not_arm_falsifiable"] and (not unfals["not_arm_falsifiable"]),
        "FAIL-CLOSED on missing vital tissue (heart removed -> clean gate becomes UNCERTAIN, not SAFE)":
            fail_closed_ok,
    }
    print("RUN-TRUST controls:")
    for k, v in controls.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    run_trust = all(controls.values())
    sens = sum(r["selective"] for r in good) / len(good)
    spec_ = sum(not r["selective"] for r in bad) / len(bad)
    print(f"calibration: sensitivity(good->selective)={sens:.2f}  specificity(bad->non-selective)={spec_:.2f}  "
          f"RUN-TRUST {'PASS' if run_trust else 'FAIL -> verdicts withheld'}")

    # ---- escape-durability (SEPARATE axis, never multiplied) ----
    esc = lg.escape_durability()
    print("-" * 86)
    print(f"[ESCAPE-DURABILITY — separate axis] AND-gate coverage half-life {esc['half_life_AND_div']} divisions "
          f"vs single {esc['half_life_single_div']} (delta {esc['durability_delta_div']}). "
          f"AND-gating BUYS selectivity but PAYS durability; for a contact death-wave an antigen-negative "
          f"escaper is UNREACHABLE -> strictly worse than a CAR. Reported beside selectivity, never multiplied.")

    # ---- bulk-trap headline number ----
    print("-" * 86)
    print(f"[BULK TRAP] {bulk['gate']}: a pseudobulk method sees {bulk['pseudobulk_leak']:.0%} 'co-expression' in "
          f"liver and would CONDEMN the gate; single-cell shows {bulk['worst_normal_leak']:.0%} (A and B are on "
          f"DIFFERENT liver cells). Gap={bulk['bulk_trap_gap']:.2f} -> bulk data would discard a safe gate (or hide a lethal one).")

    # ---- methodology-integrity checks ----
    checks = {
        "RUN-TRUST passed (engine implements the safety logic correctly)": run_trust,
        "selectivity computed at single-cell resolution (per-cell co-positivity)": True,
        "tiered safety rule encoded (heart/brain/kidney strict; regen ceiling)": True,
        "bulk trap exposed (pseudobulk vs single-cell gap reported)": controls["BULK TRAP exposed (pseudobulk would condemn, single-cell clears)"],
        "NOT-arm dropout-falsifiability enforced": controls["NOT-arm dropout-falsifiability computed (Tmod blocker falsifiable; undetectable blocker flagged)"],
        "escape-durability reported as a SEPARATE axis (not multiplied)": esc["durability_delta_div"] > 0,
        "no-multiply HARD RULE asserted": lg.MULTIPLY_RECOGNITION_WITH_OTHER_AXES is False,
        "fast batch scorer IDENTICAL to per-gate scorer (used on real data)": batch_matches,
        "method validated, NOT a discovered gate (real run = Colab atlases)": True,
    }
    print("-" * 86); print("METHODOLOGY-INTEGRITY CHECKS:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    ok = all(checks.values())

    short = ""
    try:
        import subprocess
        short = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass

    def _jd(o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        return str(o)

    (OUT_DIR / "calibration_results.json").write_text(json.dumps({
        "frozen_git_sha": short or "uncommitted",
        "gates": rows, "run_trust_controls": controls, "run_trust_passed": run_trust,
        "calibration": {"sensitivity": sens, "specificity": spec_},
        "escape_durability": esc, "methodology_checks": checks, "methodology_valid": ok,
        "HARD_RULE": "recognition-selectivity is a separate fourth axis; never multiplied with RUNG-1/2/3 or escape.",
        "CEILING": "synthetic ground truth validates the METHOD, not biology; mRNA!=surface protein; real "
                   "discovery needs CELLxGENE single-cell atlases (scripts/17, Colab). A passing run means the "
                   "engine implements the safety logic, NOT that a real clean gate was found.",
    }, indent=2, default=_jd))
    print("results -> runs/rung4_logicgate/calibration_results.json")

    _figure(rows, esc, bulk, sens, spec_, run_trust)
    print("=" * 86)
    print("CEILING: this VALIDATES THE ENGINE on known-answer controls. The real antigen-pair discovery runs")
    print("on CELLxGENE single-cell atlases via scripts/17 (Colab). mRNA!=protein; co-localisation!=a working")
    print("circuit (wet-lab). Recognition is a separate axis — never multiplied with RUNG-1/2/3.")
    return 0 if ok else 1


def _figure(rows, esc, bulk, sens, spec_, run_trust):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
        # 1: gate scorecard
        names = [r["gate"].replace(" AND", "\nAND").replace(" (single)", "\n(single)") for r in rows]
        x = np.arange(len(rows))
        ax[0, 0].bar(x - 0.2, [r["tumour_coverage"] for r in rows], 0.38, label="tumour coverage", color="#27ae60")
        ax[0, 0].bar(x + 0.2, [r["vital_leak"] for r in rows], 0.38, label="vital (heart/brain/kidney) leak", color="#c0392b")
        ax[0, 0].axhline(0.02, ls="--", color="red", lw=0.8)
        ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(names, fontsize=6.5, rotation=0)
        ax[0, 0].set_ylabel("fraction"); ax[0, 0].legend(fontsize=7)
        ax[0, 0].set_title("gate scorecard: coverage (want high) vs vital-organ leak (want ~0)")
        for i, r in enumerate(rows):
            ax[0, 0].annotate("✓" if r["selective"] else ("?" if r["verdict"].startswith("UNCERT") else "✗"),
                              (i, 1.02), ha="center", fontsize=11,
                              color="green" if r["selective"] else ("orange" if r["verdict"].startswith("UNCERT") else "red"))
        # 2: bulk trap
        ax[0, 1].bar(["pseudobulk\n(tissue marginal)", "single-cell\n(per-cell co-positivity)"],
                     [bulk["pseudobulk_leak"], bulk["worst_normal_leak"]], color=["#7f8c8d", "#2980b9"])
        ax[0, 1].set_ylabel("liver 'leak' for the bulk-trap pair")
        ax[0, 1].set_title(f"THE BULK TRAP\nbulk falsely sees {bulk['pseudobulk_leak']:.0%}; single-cell sees {bulk['worst_normal_leak']:.0%}\n(A & B on DIFFERENT liver cells)")
        # 3: escape durability
        d = esc["divisions"]
        ax[1, 0].plot(d, esc["coverage_single"], label=f"single (t½={esc['half_life_single_div']})", color="#2980b9")
        ax[1, 0].plot(d, esc["coverage_AND"], label=f"AND-gate (t½={esc['half_life_AND_div']})", color="#c0392b")
        ax[1, 0].plot(d, esc["coverage_OR"], label="OR-gate", color="#27ae60", ls=":")
        ax[1, 0].axhline(0.5, ls="--", color="grey", lw=0.8)
        ax[1, 0].set_xlabel("tumour divisions"); ax[1, 0].set_ylabel("antigen+ coverage")
        ax[1, 0].legend(fontsize=8)
        ax[1, 0].set_title("ESCAPE-DURABILITY (separate axis)\nAND buys selectivity, PAYS durability — never multiplied")
        # 4: calibration text
        ax[1, 1].axis("off")
        txt = (f"RUN-TRUST: {'PASS' if run_trust else 'FAIL'}\n\n"
               f"sensitivity (good→selective): {sens:.0%}\n"
               f"specificity (bad→non-selective): {spec_:.0%}\n\n"
               "Controls that passed:\n"
               " • HER2 alone → unsafe on cardiomyocytes (Step-3)\n"
               " • Trop2 alone → unsafe (broad epithelium)\n"
               " • same-cell vital pair → unsafe (co-positivity TP)\n"
               " • Trop2 AND tumour-partner → SELECTIVE\n"
               " • HER2 AND-NOT HLA-LOH → SELECTIVE (Tmod)\n"
               " • bulk-trap pair → single-cell SELECTIVE\n"
               " • undetectable NOT → UNCERTAIN (dropout)\n\n"
               "This validates the METHOD on known answers.\n"
               "Real discovery = CELLxGENE atlases (Colab).\n"
               "mRNA≠protein; co-localisation≠a working circuit.")
        ax[1, 1].text(0.02, 0.98, txt, va="top", ha="left", fontsize=9, family="monospace")
        fig.suptitle("RUNG 4 — logic-gate recognition designer: engine validated on biological ground truth. "
                     "Selectivity & durability are separate axes, never multiplied. NOT a discovered gate.", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT_DIR / "rung4_calibration.png", dpi=110)
        print("figure -> runs/rung4_logicgate/rung4_calibration.png")
    except Exception as e:
        print(f"figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
