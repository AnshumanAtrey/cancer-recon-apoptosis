#!/usr/bin/env python3
"""
RUNG 18 — the MHC-I "display window" status across 6,319 real tumours (laptop, no GPU, seconds).

THE QUESTION (Shriya's challenge, made quantitative)
----------------------------------------------------
The whole immune route assumes the cancer cell keeps its MHC-I "display window" ON, so a T-cell can read the
neoantigen fragment. But cancer's single biggest escape trick is to DISABLE antigen presentation. RUNG-16/17
silently assumed the window is lit. This run MEASURES, per cancer type, how often it actually is — and grades
the failure, because not all "window off" is equal:

  * DARK_SYSTEMIC  (route DIES)      — B2M / TAP / NLRC5 / CALR / TAPBP disrupted. B2M is REQUIRED for ALL
                                       surface MHC-I, on EVERY allele. If it's gone, NO peptide is presented
                                       on ANY allele -> the entire neoantigen/T-cell route fails for that cell.
  * DIMMED_HLA     (route SURVIVES)  — HLA-LOH or HLA point-mutation: ONE (or some) allele lost. The window
                                       still presents on the remaining alleles -> fewer peptides shown, but the
                                       route is not dead. (This is the handle RUNG-6 USED; here it's the risk.)
  * IFN_BLIND      (can't re-light)  — JAK1/2 / STAT1 / IFNGR loss: the window can't be turned back UP by
                                       inflammation. Orthogonal flag (can co-occur).
  * INTACT (genetically)             — none of the above. The window is genetically present. (It may STILL be
                                       epigenetically silenced — see CEILING; that's the part genetics can't see.)

WHY THIS IS THE RIGHT NEXT TEST
-------------------------------
RUNG-6/8/9 measured the SAFETY side (normal-tissue HLA, HLA-LOH as a targeting handle, IFN rescue in vital
cells). NONE measured the EFFECTOR side: do the cancer cells we'd aim at still SHOW the window. Same dataset
(Martinez-Jimenez 2023), inverse question. The honest answer decides whether the immune route is genetically
viable, or whether Shriya's ORIGINAL autonomous self-destruct is needed as the backup for window-dark tumours.

DATA
----
data/refs/mjimenez2023_MOESM6.xlsx  sheet 'GIE per sample'  (6,319 WGS tumours, 58 types,
Hartwig + PCAWG; Martinez-Jimenez et al. 2023, Nat Genet, DOI 10.1038/s41588-023-01367-1).
Per-sample boolean calls for each escape pathway, derived from whole-genome sequencing.

HONEST CEILING (stated, never papered over)
-------------------------------------------
1. GENETIC ONLY. This sees structural events (deletion, LOH, mutation). It does NOT see TRANSCRIPTIONAL /
   EPIGENETIC silencing — a genetically-intact window can still be switched OFF at the mRNA/protein level
   (promoter methylation, PRC2/EZH2 repression, low NLRC5 expression). So DARK_SYSTEMIC here is a FLOOR on
   "fully dark", not the total. The expression-level complement (HLA-A/B/C + B2M transcription in MALIGNANT
   cells) needs single-cell tumour data -> Colab (RUNG-18b, described in INTERPRETATION, not yet run).
   NOTE the direction: genetic B2M loss is IRREVERSIBLE; epigenetic silencing is (often) REVERSIBLE by IFN /
   epigenetic drugs (the RUNG-9 territory). So genetic-dark is the HARD floor; epigenetic-dark is rescuable.
2. PATIENT-LEVEL, NOT CLONAL. A tumour is "DARK" if the event is called at all; but much escape is SUBCLONAL
   / late (Martinez-Jimenez, TRACERx), so a "DARK" tumour may have only a subset of dark cells, and an
   "INTACT" tumour can hide a dark subclone below WGS detection. Patient-level => an APPROXIMATION of the
   per-cell fraction, in both directions.
3. BULK WGS, not surface protein. The truth a T-cell sees is surface peptide-MHC; this is the genotype. Same
   modality caveat as RUNG-8/9.

USAGE
  python scripts/43_mhc_window_status.py selftest   # synthetic rows, hand-checked grading, no Excel
  python scripts/43_mhc_window_status.py run        # reads the local xlsx, writes runs/rung18_mhc_window/
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUPP = REPO / "data" / "refs" / "mjimenez2023_MOESM6.xlsx"
SHEET = "GIE per sample"
OUTDIR = REPO / "runs" / "rung18_mhc_window"
RESULT_JSON = OUTDIR / "rung18_mhc_window.json"
FIGURE_PNG = OUTDIR / "rung18_mhc_window.png"

# Our neoantigen-route cancers (where RUNG-16 said clean handles usually exist) + low-route contrasts.
ROUTE_CANCERS = ["SKCM", "NSCLC", "COREAD", "BLCA"]          # high-TMB, route-viable per RUNG-16
CONTRAST_CANCERS = ["PAAD", "BRCA", "PRAD"]                  # lower-TMB / different biology
MIN_N = 30                                                  # don't report per-cancer fractions below this n

# Columns we read (and assert exist).
COL_CODE = "cancer_type_code"
COL_NAME = "cancer_type"
BOOL_COLS = {
    "genetic_immune_escape": "any_gie",
    "systemic_app_pathway": "dark_systemic",     # B2M/TAP/NLRC5 -> WHOLE window off  (route dies)
    "targeted_escape": "targeted_hla",           # HLA LOH or HLA mut -> partial      (route survives)
    "ifn_gamma_pathway": "ifn_blind",            # JAK/STAT/IFNGR     -> can't re-light
    "loh_lilac": "hla_loh",                       # HLA-LOH specifically
    "mut_hla_lilac": "hla_mut",                   # HLA point mutation
    "epigenetic_regulators_pathway": "epigen_genetic",  # SETDB1 amp etc (a GENETIC route to silencing)
}


def _truthy(v) -> bool:
    """Robust against bool / 'True' / 'TRUE' / 1.0 / None / 'None'."""
    if v is True:
        return True
    if v is False or v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "1.0", "yes")


def grade(df: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean window-status flags. Severity ladder is applied in summarise(), not here."""
    out = df.copy()
    for raw, flag in BOOL_COLS.items():
        out[flag] = out[raw].map(_truthy) if raw in out.columns else False
    # mutually-exclusive severity class (worst wins): DARK_SYSTEMIC > DIMMED_HLA > INTACT
    cls = np.where(out["dark_systemic"], "DARK_SYSTEMIC",
          np.where(out["targeted_hla"], "DIMMED_HLA", "INTACT_GENETIC"))
    out["window_class"] = cls
    return out


def _wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson CI for a fraction (small-n honest, used for per-cancer bars)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(p, 4), round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def summarise(g: pd.DataFrame, code: str | None) -> dict:
    sub = g if code is None else g[g[COL_CODE] == code]
    n = int(len(sub))
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for flag in ("any_gie", "dark_systemic", "targeted_hla", "hla_loh", "hla_mut", "ifn_blind", "epigen_genetic"):
        k = int(sub[flag].sum())
        p, lo, hi = _wilson(k, n)
        out[flag] = {"k": k, "fraction": p, "ci_lower": lo, "ci_upper": hi}
    # the three derived headline classes
    out["window_class_fraction"] = {
        c: round(float((sub["window_class"] == c).mean()), 4)
        for c in ("DARK_SYSTEMIC", "DIMMED_HLA", "INTACT_GENETIC")
    }
    # route framing: the immune route DIES only when the window is fully dark (systemic). DIMMED still presents.
    out["route_dies_fraction"] = out["window_class_fraction"]["DARK_SYSTEMIC"]
    out["route_viable_fraction"] = round(1.0 - out["route_dies_fraction"], 4)
    return out


def main_run() -> int:
    if not SUPP.exists():
        print(f"[rung18] MISSING {SUPP}\n  download once (no auth):\n"
              f'  curl -L -o {SUPP} "https://static-content.springer.com/esm/'
              'art%3A10.1038%2Fs41588-023-01367-1/MediaObjects/41588_2023_1367_MOESM6_ESM.xlsx"')
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SUPP, sheet_name=SHEET)
    df = df[df["sample_id_2"].notna()].copy()
    missing = [c for c in (COL_CODE, COL_NAME, *BOOL_COLS) if c not in df.columns]
    if missing:
        print(f"[rung18] FATAL: columns absent from sheet: {missing}")
        return 3
    g = grade(df)
    print(f"[rung18] graded {len(g):,} tumours across {g[COL_CODE].nunique()} cancer types")

    overall = summarise(g, None)
    codes_present = [c for c in g[COL_CODE].value_counts().index if int((g[COL_CODE] == c).sum()) >= MIN_N]
    per_cancer = {c: summarise(g, c) for c in codes_present}
    code_to_name = dict(zip(g[COL_CODE], g[COL_NAME]))

    # genetic driver breakdown WITHIN the dark-systemic tumours (what actually killed the window)
    dark = g[g["dark_systemic"]]
    detail_counts = {}
    if "systemic_app_pathway_detail" in g.columns:
        from collections import Counter
        cc = Counter()
        for v in dark["systemic_app_pathway_detail"].dropna():
            for tok in str(v).replace(";", ",").split(","):
                tok = tok.strip()
                if tok and tok.lower() != "none":
                    cc[tok] += 1
        detail_counts = dict(cc.most_common(20))

    route = {c: per_cancer[c] for c in ROUTE_CANCERS if c in per_cancer}
    route_dies = {c: route[c]["route_dies_fraction"] for c in route}
    worst_route = max(route_dies, key=route_dies.get) if route_dies else None

    result = {
        "tag": "rung18_mhc_window_status",
        "hypothesis": "Does the cancer cell keep its MHC-I display window ON? Measure, per cancer type, how "
                      "often antigen presentation is genetically disabled, and GRADE it: full-dark "
                      "(systemic B2M/TAP, route dies) vs dimmed (HLA-LOH/mut, route survives) vs intact.",
        "data_source": "Martinez-Jimenez et al. 2023, Nat Genet, DOI 10.1038/s41588-023-01367-1, "
                       "Supplementary Data (MOESM6) sheet 'GIE per sample'; 6,319 WGS tumours, 58 types.",
        "n_samples_total": int(len(g)),
        "grading": {
            "DARK_SYSTEMIC": "systemic_app_pathway True (B2M/TAP/NLRC5/CALR/TAPBP). WHOLE window off -> immune route dies.",
            "DIMMED_HLA": "targeted_escape True (HLA-LOH or HLA mut), not systemic. Some alleles lost -> route survives reduced.",
            "INTACT_GENETIC": "neither -> window genetically present (may still be epigenetically silenced; see CEILING).",
            "IFN_BLIND": "ifn_gamma_pathway True -> window cannot be re-lit by inflammation (orthogonal flag).",
        },
        "OVERALL": overall,
        "ROUTE_CANCERS": route,
        "route_dies_fraction_by_cancer": route_dies,
        "worst_route_cancer_for_window_loss": worst_route,
        "per_cancer": per_cancer,
        "code_to_name": code_to_name,
        "dark_systemic_genetic_drivers": detail_counts,
        "HEADLINE": {
            "any_genetic_immune_escape": overall["any_gie"]["fraction"],
            "window_fully_dark_systemic_ROUTE_DIES": overall["dark_systemic"]["fraction"],
            "window_dimmed_HLA_only_route_survives": overall["window_class_fraction"]["DIMMED_HLA"],
            "window_intact_genetic": overall["window_class_fraction"]["INTACT_GENETIC"],
            "ifn_blind_cannot_relight": overall["ifn_blind"]["fraction"],
            "plain": "Across 6,319 real tumours: ~{:.0%} have SOME genetic immune escape, but the FATAL kind "
                     "(whole window dark, route dies) is only ~{:.0%}. Most escape (~{:.0%}) is HLA-LOH/mut — "
                     "the window DIMS but still presents on the remaining alleles, so the neoantigen route "
                     "survives there.".format(
                         overall["any_gie"]["fraction"],
                         overall["dark_systemic"]["fraction"],
                         overall["window_class_fraction"]["DIMMED_HLA"]),
        },
        "INTERPRETATION_MAP": {
            "dark_systemic_small (<0.10) in route cancers":
                "Genetic window mostly intact -> immune route is GENETICALLY viable in melanoma/NSCLC/CRC. "
                "The dominant REMAINING risk is EXPRESSION-level silencing (epigenetic), which genetics can't "
                "see -> next test is RUNG-18b single-cell HLA/B2M transcription in MALIGNANT cells (Colab).",
            "dark_systemic_large (>=0.10)":
                "Genetic escape alone kills the route in a large fraction -> Shriya's ORIGINAL autonomous "
                "self-destruct (no MHC needed) becomes essential as the backup for window-dark tumours.",
            "ifn_blind co-occurring with dark":
                "Tumours that are BOTH dark and IFN-blind cannot be rescued by inflammation -> these are the "
                "hard core where only an MHC-independent killer (NK-engager / autonomous trigger) can work.",
        },
        "CEILING": "GENETIC ONLY: epigenetic/transcriptional MHC-I silencing is NOT captured, so DARK_SYSTEMIC "
                   "is a FLOOR, not the total (genetically-intact windows can still be transcriptionally OFF; "
                   "that arm is reversible by IFN/epi-drugs and is RUNG-9 territory). PATIENT-LEVEL, not clonal "
                   "(much escape is subclonal/late -> a DARK tumour may have only some dark cells; an INTACT "
                   "one can hide a dark subclone). Bulk WGS genotype, not surface protein. NOT a wet result.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung18] wrote {RESULT_JSON}")

    # console summary
    print(f"\n  OVERALL (n={overall['n']:,}):  any-GIE {overall['any_gie']['fraction']:.1%}   "
          f"DARK(route dies) {overall['dark_systemic']['fraction']:.1%}   "
          f"DIMMED(survives) {overall['window_class_fraction']['DIMMED_HLA']:.1%}   "
          f"INTACT {overall['window_class_fraction']['INTACT_GENETIC']:.1%}   "
          f"IFN-blind {overall['ifn_blind']['fraction']:.1%}")
    print("\n  cancer   n    DARK(dies)  DIMMED(surv)  INTACT   IFN-blind")
    for c in ROUTE_CANCERS + [x for x in CONTRAST_CANCERS if x in per_cancer]:
        if c not in per_cancer:
            continue
        b = per_cancer[c]
        wc = b["window_class_fraction"]
        print(f"  {c:7}{b['n']:4}   {wc['DARK_SYSTEMIC']:.1%} [{b['dark_systemic']['ci_lower']:.1%}-{b['dark_systemic']['ci_upper']:.1%}]   "
              f"{wc['DIMMED_HLA']:.1%}        {wc['INTACT_GENETIC']:.1%}    {b['ifn_blind']['fraction']:.1%}")
    if detail_counts:
        print("\n  genetic drivers of the FULL-dark window:", detail_counts)
    _make_figure(overall, per_cancer)
    return 0


def _make_figure(overall: dict, per_cancer: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung18] matplotlib unavailable ({e}); skipped figure")
        return
    codes = [c for c in (ROUTE_CANCERS + CONTRAST_CANCERS) if c in per_cancer]
    labels = [f"{c}\n(n={per_cancer[c]['n']})" for c in codes]
    dark = [per_cancer[c]["window_class_fraction"]["DARK_SYSTEMIC"] * 100 for c in codes]
    dim = [per_cancer[c]["window_class_fraction"]["DIMMED_HLA"] * 100 for c in codes]
    intact = [per_cancer[c]["window_class_fraction"]["INTACT_GENETIC"] * 100 for c in codes]
    x = np.arange(len(codes))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # panel 1: stacked window status per cancer
    ax[0].bar(x, intact, 0.62, label="INTACT (window genetically on)", color="#3F7D54")
    ax[0].bar(x, dim, 0.62, bottom=intact, label="DIMMED (HLA-LOH/mut; still presents)", color="#E0A040")
    ax[0].bar(x, dark, 0.62, bottom=[intact[i] + dim[i] for i in range(len(codes))],
              label="DARK (systemic; route DIES)", color="#B23A2E")
    for i, c in enumerate(codes):
        ax[0].text(i, intact[i] + dim[i] + dark[i] + 1.2, f"{dark[i]:.1f}%", ha="center", fontsize=8, color="#B23A2E")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("% of tumours"); ax[0].set_ylim(0, 108)
    ax[0].set_title("MHC-I display-window status (genetic)\nred = fully dark = immune route dies")
    ax[0].legend(fontsize=7.5, loc="lower right"); ax[0].grid(axis="y", alpha=0.3)

    # panel 2: overall grade donut-ish bar
    grades = ["INTACT_GENETIC", "DIMMED_HLA", "DARK_SYSTEMIC"]
    vals = [overall["window_class_fraction"][k] * 100 for k in grades]
    cols = ["#3F7D54", "#E0A040", "#B23A2E"]
    ax[1].barh([0, 1, 2], vals, color=cols)
    for i, v in enumerate(vals):
        ax[1].text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=10)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(["INTACT\n(route ok)", "DIMMED HLA\n(route survives)", "DARK systemic\n(route DIES)"])
    ax[1].set_xlabel("% of all 6,319 tumours"); ax[1].set_xlim(0, max(vals) * 1.25)
    ax[1].set_title(f"Overall: full-dark (route dies) is only {vals[2]:.1f}%\nIFN-blind (can't re-light): "
                    f"{overall['ifn_blind']['fraction']:.1%}")
    ax[1].grid(axis="x", alpha=0.3)

    fig.suptitle("RUNG-18: is the cancer cell's MHC window on? — graded across 6,319 real tumours "
                 "(genetic; epigenetic silencing not captured)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung18] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    """Grade synthetic rows with hand-computed window classes (no Excel)."""
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond)))
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # cols: genetic_immune_escape, systemic_app_pathway, targeted_escape, ifn_gamma_pathway,
    #       loh_lilac, mut_hla_lilac, epigenetic_regulators_pathway  + expected window_class
    rows = [
        # gie ,  sys  , tgt  , ifn  , loh  , mut  , epi  , expect_class
        ("True", "True", "True", "False", "True", "False", "False", "DARK_SYSTEMIC"),   # systemic wins over HLA
        ("True", "False", "True", "False", "True", "False", "False", "DIMMED_HLA"),     # HLA-LOH only -> dimmed
        ("True", "False", "True", "True", "False", "True", "False", "DIMMED_HLA"),      # HLA-mut + IFN-blind -> dimmed class, ifn flag set
        ("False", "False", "False", "False", "False", "False", "False", "INTACT_GENETIC"),  # clean
        ("True", "True", "False", "False", "False", "False", "False", "DARK_SYSTEMIC"),  # systemic w/o HLA
        (True, False, False, True, False, False, False, "INTACT_GENETIC"),               # native bools + ifn-only (not a window class), intact
    ]
    cols = ["genetic_immune_escape", "systemic_app_pathway", "targeted_escape", "ifn_gamma_pathway",
            "loh_lilac", "mut_hla_lilac", "epigenetic_regulators_pathway", "_expect"]
    df = pd.DataFrame(rows, columns=cols)
    df[COL_CODE] = "TEST"
    df[COL_NAME] = "Test cancer"
    g = grade(df)

    for i, exp in enumerate(df["_expect"]):
        check(f"row{i} window_class == {exp}", g["window_class"].iloc[i] == exp)

    # flag truthiness
    check("native bool True parsed", bool(g["ifn_blind"].iloc[5]) is True)
    check("string 'False' -> False", bool(g["dark_systemic"].iloc[1]) is False)

    # severity ladder: systemic always beats targeted
    both = grade(pd.DataFrame([{"systemic_app_pathway": True, "targeted_escape": True}]))
    check("systemic overrides targeted in class", both["window_class"].iloc[0] == "DARK_SYSTEMIC")

    # summarise math on a known frame: 4 rows, 2 dark, 1 dimmed, 1 intact
    known = grade(pd.DataFrame({
        "systemic_app_pathway": [True, True, False, False],
        "targeted_escape": [False, False, True, False],
        "genetic_immune_escape": [True, True, True, False],
        "ifn_gamma_pathway": [False, False, False, False],
        "loh_lilac": [False, False, True, False],
        "mut_hla_lilac": [False, False, False, False],
        "epigenetic_regulators_pathway": [False, False, False, False],
        COL_CODE: ["X"] * 4, COL_NAME: ["x"] * 4,
    }))
    s = summarise(known, "X")
    check("summarise n==4", s["n"] == 4)
    check("summarise dark fraction==0.5", abs(s["window_class_fraction"]["DARK_SYSTEMIC"] - 0.5) < 1e-9)
    check("summarise dimmed fraction==0.25", abs(s["window_class_fraction"]["DIMMED_HLA"] - 0.25) < 1e-9)
    check("summarise route_dies==dark", abs(s["route_dies_fraction"] - 0.5) < 1e-9)
    check("summarise route_viable==0.5", abs(s["route_viable_fraction"] - 0.5) < 1e-9)

    # Wilson CI sanity: brackets the point estimate, within [0,1]
    p, lo, hi = _wilson(2, 4)
    check("wilson brackets p", lo <= p <= hi and 0 <= lo and hi <= 1)

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "selftest":
        return selftest()
    if cmd == "run":
        return main_run()
    print(f"unknown command: {cmd} (use selftest|run)")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
