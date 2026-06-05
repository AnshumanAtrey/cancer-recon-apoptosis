#!/usr/bin/env python3
"""
RUNG 6 / arm (b++) — PAN-CANCER HLA-LOH recognition-addressability atlas (laptop, no GPU, seconds).

scripts/24 + scripts/26 quantified the genetic NOT-gate for our three RUNG-5 cancers. This scales the SAME
tested gate logic across ALL 58 cancer types in the cohort to answer two programme-level questions:

  Q1. WHICH cancers are most addressable by a genetic (HLA-LOH) recognition gate? (the ceiling, per type)
  Q2. If you build ONE UNIVERSAL blocker panel (top-K most-useful HLA allotypes, chosen pan-cohort), how
      many patients does it address PER cancer — i.e. the reach of a single fixed reagent set?

WHY THIS ADVANCES RECOGNITION
-----------------------------
The Tmod architecture gets specificity from the NEGATIVE arm (blocker on a germline allele the tumour LOST),
so per-cancer addressability == fraction of patients with a usable lost-allele blocker. Mapping that across
58 cancers turns "the genetic gate buys ~5-28% in three cancers" into a PRIORITISED target list + the
economics of a universal panel. It is the honest 'scale it up' step.

Reuses scripts/26's gate logic verbatim (usable_blocker_alleles / greedy_panel_curve) so the pan-cancer
numbers are identical-by-construction to the 3-cancer result. Same data, same honest caveats:
patient-level NOT clonal (subclonal LOH => true reach LOWER; quantifying it needs controlled-access TRACERx
multi-region WES, NOT in these supplements); WGS-genotype cohort joined to RUNG-5's scRNA surface conclusion
(same cancer TYPES, different individuals); 4-digit allotype blocker unit (conservative).

USAGE
  python scripts/27_pancancer_loh_addressability.py            # real supplement -> JSON + figure
  python scripts/27_pancancer_loh_addressability.py selftest   # aggregation-logic checks
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPP = PROJECT_ROOT / "data" / "refs" / "mjimenez2023_MOESM6.xlsx"
SHEET = "GIE per sample"
OUT_DIR = PROJECT_ROOT / "runs" / "rung6_logicgate"
RESULT_JSON = OUT_DIR / "rung6_pancancer_addressability.json"
FIGURE_PNG = OUT_DIR / "rung6_pancancer.png"
MIN_N = 30                     # min patients for a cancer type to be reported (statistical meaning)
UNIVERSAL_K = (6, 12)          # universal-panel sizes to evaluate

# import scripts/26 by path (module name starts with a digit -> importlib)
_spec = importlib.util.spec_from_file_location("rung6_panel", PROJECT_ROOT / "scripts" / "26_blocker_panel_addressability.py")
_panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_panel)
usable_blocker_alleles = _panel.usable_blocker_alleles
greedy_panel_curve = _panel.greedy_panel_curve
patient_usable_sets = _panel.patient_usable_sets
jeffreys_lower, jeffreys_upper = _panel.jeffreys_lower, _panel.jeffreys_upper


def _ceiling_and_a02(usable):
    n = len(usable)
    ceil_k = sum(1 for s in usable if s)
    a02_k = sum(1 for s in usable if "A*02:01" in s)
    return n, ceil_k, a02_k


def reach_under_panel(usable, panel_alleles: set) -> float:
    n = len(usable)
    if not n:
        return 0.0
    return sum(1 for s in usable if s & panel_alleles) / n


def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SUPP, sheet_name=SHEET)
    print(f"[rung6/pan] loaded {len(df):,} samples, {df.cancer_type_code.nunique()} cancer types")

    # ---- pan-cohort universal panel (greedy over ALL patients) ----
    all_usable = patient_usable_sets(df)
    _, global_panel = greedy_panel_curve(all_usable, max_panel=max(UNIVERSAL_K))
    global_order = [a for a, _ in global_panel]
    universal = {k: set(global_order[:k]) for k in UNIVERSAL_K}
    print(f"[rung6/pan] universal greedy allele order (top {max(UNIVERSAL_K)}): {global_order}")

    # ---- per cancer type ----
    per_cancer = {}
    for code, sub in df.groupby("cancer_type_code"):
        usable = patient_usable_sets(sub)
        n, ceil_k, a02_k = _ceiling_and_a02(usable)
        if n < MIN_N:
            continue
        curve, panel = greedy_panel_curve(usable)
        frac = [c / n for c in curve]
        def size_for(t):
            for k, f in enumerate(frac):
                if f >= t:
                    return k
            return None
        per_cancer[code] = {
            "n": n,
            "single_a0201": round(a02_k / n, 4),
            "ceiling": round(ceil_k / n, 4),
            "ceiling_ci": [round(jeffreys_lower(ceil_k, n), 4), round(jeffreys_upper(ceil_k, n), 4)],
            "panel_for_10pct": size_for(0.10),
            "blockers_to_ceiling": len(panel),
            **{f"universal_top{k}_reach": round(reach_under_panel(usable, universal[k]), 4) for k in UNIVERSAL_K},
        }

    ranked = sorted(per_cancer.items(), key=lambda kv: kv[1]["ceiling"], reverse=True)

    # pan-cohort headline numbers for the universal panels
    universal_global = {f"top{k}": round(reach_under_panel(all_usable, universal[k]), 4) for k in UNIVERSAL_K}
    overall_ceiling = round(sum(1 for s in all_usable if s) / len(all_usable), 4)

    result = {
        "tag": "rung6_pancancer_loh_addressability",
        "question": "Across all 58 cancer types: which are most addressable by a genetic HLA-LOH recognition "
                    "gate, and what does ONE universal top-K blocker panel reach per cancer?",
        "data_source": "Martinez-Jimenez 2023 Nat Genet, MOESM6 'GIE per sample' (6,319 WGS tumours).",
        "min_n_per_type": MIN_N, "universal_panel_sizes": list(UNIVERSAL_K),
        "universal_panel_alleles": {str(k): sorted(universal[k]) for k in UNIVERSAL_K},
        "global_greedy_allele_order": global_order,
        "overall": {"n": len(all_usable), "ceiling_any_blocker": overall_ceiling,
                    "universal_reach": universal_global},
        "n_types_reported": len(per_cancer),
        "ranked_by_ceiling": [{"cancer": c, **v} for c, v in ranked],
        "most_addressable_top5": [c for c, _ in ranked[:5]],
        "least_addressable_bottom5": [c for c, _ in ranked[-5:]],
        "CEILING": "Patient-level NOT clonal (subclonal LOH => true reach LOWER; needs controlled-access "
                   "TRACERx multi-region WES, absent from public supplements). WGS cohort joined to RUNG-5's "
                   "scRNA surface conclusion by cancer TYPE, not same individuals. 4-digit blocker unit. "
                   "Assumes a broad activator exists per type (Tmod premise; the positive arm is not proven "
                   "free here). Some rare types have small n -> wide CIs (reported).",
        "INTERPRETATION": "Genetic-NOT-gate addressability is FREQUENCY-BOUNDED and varies widely by cancer "
                          "(set by each type's HLA-LOH rate). A single universal ~6-12 allele blocker panel "
                          "captures most of the reachable population in the high-LOH cancers. The ranked list "
                          "is a concrete indication-prioritisation + reagent-panel spec for a programme.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung6/pan] wrote {RESULT_JSON}")

    print(f"\n  overall ceiling (any usable blocker) = {overall_ceiling:.1%}; "
          f"universal {UNIVERSAL_K} reach = " + ", ".join(f"top{k}={universal_global[f'top{k}']:.1%}" for k in UNIVERSAL_K))
    print("\n  MOST addressable cancers (by HLA-LOH ceiling):")
    print("   cancer    n    A*02:01  ceiling        uTop6   uTop12")
    for c, v in ranked[:10]:
        print(f"   {c:8}{v['n']:4}  {v['single_a0201']:6.1%}  {v['ceiling']:5.1%} "
              f"[{v['ceiling_ci'][0]:.0%}-{v['ceiling_ci'][1]:.0%}]  {v[f'universal_top6_reach']:5.1%}  {v['universal_top12_reach']:5.1%}")
    print("  LEAST addressable (bottom 5):")
    for c, v in ranked[-5:]:
        print(f"   {c:8}{v['n']:4}  {v['single_a0201']:6.1%}  {v['ceiling']:5.1%}")

    _make_figure(ranked, universal_global)
    return 0


def _make_figure(ranked, universal_global) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung6/pan] matplotlib unavailable ({e}); skipped figure")
        return
    top = ranked[:20]
    names = [f"{c} (n={v['n']})" for c, v in top][::-1]
    ceil = [v["ceiling"] * 100 for _, v in top][::-1]
    a02 = [v["single_a0201"] * 100 for _, v in top][::-1]
    u6 = [v["universal_top6_reach"] * 100 for _, v in top][::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(y, ceil, color="#4C9F70", label="ceiling (any blocker)")
    ax.barh(y, u6, color="#2B6CB0", height=0.55, label="universal top-6 panel")
    ax.plot(a02, y, "x", color="#C1432B", ms=8, label="deployed single A*02 gate")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("% patients addressable by a genetic HLA-LOH NOT-gate (worst case, patient-level)")
    ax.set_title("RUNG-6: pan-cancer genetic recognition-addressability\n"
                 "(top-20 by HLA-LOH ceiling; single-allele gate vs universal 6-blocker panel vs ceiling)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung6/pan] wrote {FIGURE_PNG}")


def selftest() -> int:
    checks, ok = [], 0
    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # tiny cohort: 2 cancers, known usable sets
    rows = [
        # cancer X: one addressable via A*02:01, one not
        {"cancer_type_code": "X", "A1": "A*02:01", "A2": "A*01:01", "A1_CN": 0.0, "A2_CN": 2.0,
         "B1": "B*07:02", "B2": "B*08:01", "B1_CN": 2.0, "B2_CN": 2.0, "C1": "C*07:01", "C2": "C*07:01", "C1_CN": 2.0, "C2_CN": 2.0},
        {"cancer_type_code": "X", "A1": "A*03:01", "A2": "A*03:01", "A1_CN": 0.0, "A2_CN": 0.0,
         "B1": "B*07:02", "B2": "B*08:01", "B1_CN": 2.0, "B2_CN": 2.0, "C1": "C*07:01", "C2": "C*07:01", "C1_CN": 2.0, "C2_CN": 2.0},
        # cancer Y: one addressable via B*08:01
        {"cancer_type_code": "Y", "A1": "A*11:01", "A2": "A*24:02", "A1_CN": 2.0, "A2_CN": 2.0,
         "B1": "B*08:01", "B2": "B*44:02", "B1_CN": 0.0, "B2_CN": 2.0, "C1": "C*05:01", "C2": "C*05:01", "C1_CN": 2.0, "C2_CN": 2.0},
    ]
    df = pd.DataFrame(rows)

    ux = patient_usable_sets(df[df.cancer_type_code == "X"])
    uy = patient_usable_sets(df[df.cancer_type_code == "Y"])
    check("cancer X ceiling = 1/2", sum(1 for s in ux if s) == 1 and len(ux) == 2)
    check("cancer Y ceiling = 1/1", sum(1 for s in uy if s) == 1 and len(uy) == 1)
    check("reach under {A*02:01} on X == 0.5", reach_under_panel(ux, {"A*02:01"}) == 0.5)
    check("reach under {A*02:01} on Y == 0.0", reach_under_panel(uy, {"A*02:01"}) == 0.0)
    check("reach under {A*02:01,B*08:01} on Y == 1.0", reach_under_panel(uy, {"A*02:01", "B*08:01"}) == 1.0)
    check("reach is monotone in panel (superset >= subset)",
          reach_under_panel(ux, {"A*02:01", "A*03:01"}) >= reach_under_panel(ux, {"A*02:01"}))

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
