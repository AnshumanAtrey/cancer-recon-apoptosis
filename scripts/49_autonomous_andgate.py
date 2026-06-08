#!/usr/bin/env python3
"""
RUNG 23 (v2, HARDENED) — the AUTONOMOUS intracellular AND-gate: a synthetic gene circuit sensing TWO
intracellular cancer signals (A AND B) -> self-destruct, NO MHC, NO immune system. (Census; CPU.)

THE IDEA (Shriya's ORIGINAL concept, the un-crowded route)
----------------------------------------------------------
The immune route dies on the MHC-DARK core (RUNG-18). A recognition gate that lives INSIDE the cell needs no
MHC. AND is the right operator (specificity), and it is buildable inside a cell (synthetic TFs / toehold
switches). Question on REAL data: is there a PAIR of intracellular transcriptional programs (A,B) such that
(A AND B) fires in TUMOUR cells but ~ZERO vital normal cells, where each single program LEAKS?

v2 FIXES (why v1's "0 clean gates" was not yet trustworthy — caught in audit)
-----------------------------------------------------------------------------
v1 had TWO methodological gaps that bias toward a FALSE negative:
  1. NO DEPTH GATE. scRNA dropout makes a program look "off" in shallow cells -> tumour COVERAGE deflated.
     FIX (the RUNG-18b discipline): score ONLY well-sequenced cells (>= HK_MIN housekeeping genes detected).
  2. SINGLE-WORST-DONOR leak. v1's 62% "leak" of PROLIF into POST-MITOTIC vital cells (cardiomyocytes/neurons
     — which cannot proliferate) is the exact single-donor artifact RUNG-8 already hit and fixed. FIX: report
     leak PER VITAL TYPE as a per-donor DISTRIBUTION (median / p90 / worst), and use the robust p90-of-donors,
     max over vital types, as the safety bar — so one outlier donor can't fake a leak, and we SEE which tissue
     (post-mitotic = artifact vs endothelium = plausibly real) drives any leak.

PROGRAMS: PROLIF, MYC, E2F, GLYCOLYSIS, HYPOXIA, WNT, STEMNESS, TELOMERASE (intracellular; detection-fraction).

HONEST CEILING
--------------
mRNA != protein/activity (program score is a PROXY for the state a circuit senses); depth-gate reduces but
doesn't erase dropout (read tumour-vs-vital CONTRAST); 'buildable' assumes a sensor per program + delivery
(wet-lab). BOUNDS whether a clean intracellular AND-pair exists; not a built circuit.

USAGE
  python scripts/49_autonomous_andgate.py selftest
  RUNG23_CACHE=/content/drive/MyDrive/cancer-recon python scripts/49_autonomous_andgate.py run   # Colab Census
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np

_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung23_autonomous"
RESULT_JSON = OUT_DIR / "rung23_autonomous.json"
FIGURE_PNG = OUT_DIR / "rung23_autonomous.png"

PROGRAMS = {
    "PROLIF": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CDK1", "BIRC5"],
    "MYC": ["MYC", "NPM1", "ODC1", "NCL", "SRM"],
    "E2F": ["E2F1", "MCM2", "MCM6", "CDC6", "CCNE1"],
    "GLYCOLYSIS": ["SLC2A1", "LDHA", "HK2", "PKM", "PGK1", "ENO1"],
    "HYPOXIA": ["VEGFA", "CA9", "SLC2A1", "NDRG1", "BNIP3"],
    "WNT": ["AXIN2", "LGR5", "ASCL2", "NKD1", "RNF43"],
    "STEMNESS": ["SOX2", "PROM1", "ALDH1A1", "POU5F1"],
    "TELOMERASE": ["TERT", "TERC", "DKC1"],
}
HK_GENES = ["ACTB", "GAPDH", "EEF1A1", "TMSB4X", "PTMA", "MALAT1"]   # INDEPENDENT depth gate (RUNG-18b lesson)
PROG_GENES = sorted({g for panel in PROGRAMS.values() for g in panel})
ALL_GENES = sorted(set(PROG_GENES) | set(HK_GENES))
HK_MIN = 4                      # a cell is "well-sequenced" iff it detects >= HK_MIN housekeeping genes
FIRE_FRAC = 0.5
DET = 1
PER_DONOR_CAP = 600
MIN_CELLS_PER_DONOR = 30
LEAK_SAFE = 0.01                # robust (p90-donor, max over vital types) leak bar
COVER_MIN = 0.30


def log(msg):
    print(f"[+{time.monotonic()-_T0:7.1f}s] [rung23] {msg}", flush=True)


# ---------------------------------------------------------------------------
#  CORE (selftestable) — depth-gated program fire, gate, per-type robust leak
# ---------------------------------------------------------------------------
def cell_state(counts: np.ndarray, gene_order: list[str]):
    """Returns (program_fire dict[bool[N]], well_seq bool[N]). A program fires if >=FIRE_FRAC of its panel
    detected; a cell is well-sequenced if >=HK_MIN housekeeping genes detected (the depth gate)."""
    idx = {g: j for j, g in enumerate(gene_order)}
    hk_cols = [idx[g] for g in HK_GENES if g in idx]
    well = (counts[:, hk_cols] >= DET).sum(axis=1) >= HK_MIN if hk_cols else np.ones(counts.shape[0], bool)
    pf = {}
    for prog, panel in PROGRAMS.items():
        cols = [idx[g] for g in panel if g in idx]
        if not cols:
            pf[prog] = np.zeros(counts.shape[0], bool); continue
        pf[prog] = (counts[:, cols] >= DET).sum(axis=1) >= np.ceil(FIRE_FRAC * len(cols))
    return pf, well


def gate_fire(pf, a, b, op):
    if op == "single":
        return pf[a]
    if op == "AND":
        return pf[a] & pf[b]
    if op == "OR":
        return pf[a] | pf[b]
    raise ValueError(op)


def _measuring(counts, dataset):
    any_det = (counts >= DET).any(axis=1)
    return {d for d in set(dataset.tolist()) if any_det[dataset == d].any()}


def per_type_leak(fire, label_vital, donor, use):
    """Per VITAL TYPE: per-donor fire fractions -> {worst, p90, median, n_donors}. Robust to single-donor artifact."""
    out = {}
    for t in sorted({x for x in label_vital[use] if x is not None}):
        sel = use & (label_vital == t)
        per_donor = []
        for d in np.unique(donor[sel]):
            m = sel & (donor == d)
            if m.sum() >= MIN_CELLS_PER_DONOR:
                per_donor.append(float(fire[m].mean()))
        if per_donor:
            pd = np.array(sorted(per_donor))
            out[t] = {"worst": round(float(pd.max()), 4), "p90": round(float(np.quantile(pd, 0.9)), 4),
                      "median": round(float(np.median(pd)), 4), "n_donors": int(len(pd))}
    return out


def evaluate_all(counts, gene_order, label_vital, donor, dataset, is_tumour):
    pf, well = cell_state(counts, gene_order)
    measuring = _measuring(counts, dataset)
    meas = np.array([d in measuring for d in dataset], bool)
    use = well & meas                                       # DEPTH-GATED + measuring
    tum = is_tumour & use
    progs = list(PROGRAMS)
    results = []

    def metrics(fire):
        cov = float(fire[tum].mean()) if tum.any() else 0.0
        types = per_type_leak(fire, label_vital, donor, use & (label_vital != None))
        # robust safety bar = max over vital types of the p90-donor leak (one outlier donor can't fake it)
        robust = max((v["p90"] for v in types.values()), default=0.0)
        worst = max((v["worst"] for v in types.values()), default=0.0)
        worst_type = max(types, key=lambda t: types[t]["worst"]) if types else None
        return round(cov, 4), round(robust, 4), round(worst, 4), worst_type, types

    for p in progs:
        cov, rob, wrst, wt, types = metrics(pf[p])
        results.append({"gate": p, "op": "single", "coverage": cov, "robust_leak_p90": rob,
                        "worst_donor_leak": wrst, "worst_type": wt,
                        "safe_effective": rob <= LEAK_SAFE and cov >= COVER_MIN})
    for a, b in combinations(progs, 2):
        for op in ("AND", "OR"):
            cov, rob, wrst, wt, types = metrics(gate_fire(pf, a, b, op))
            results.append({"gate": f"{a} {op} {b}", "op": op, "a": a, "b": b, "coverage": cov,
                            "robust_leak_p90": rob, "worst_donor_leak": wrst, "worst_type": wt,
                            "safe_effective": rob <= LEAK_SAFE and cov >= COVER_MIN})
    results.sort(key=lambda r: (-int(r["safe_effective"]), -r["coverage"], r["robust_leak_p90"]))
    n_well = int(well.sum())
    return results, {"n_datasets_measuring": len(measuring), "n_cells": int(counts.shape[0]),
                     "n_well_sequenced": n_well, "frac_well_sequenced": round(n_well / max(counts.shape[0], 1), 4),
                     "n_vital_well": int((use & (label_vital != None)).sum()), "n_tumour_well": int(tum.sum())}


# ---------------------------------------------------------------------------
#  Census fetch (reuses RUNG-8/29 vital pull + RUNG-4 malignant selection)
# ---------------------------------------------------------------------------
class Heartbeat:
    def __init__(self, interval=20):
        self.interval, self.label, self._stop = interval, "starting", False

    def set(self, label):
        self.label = label; log(label)

    def _run(self):
        while not self._stop:
            for _ in range(self.interval * 2):
                if self._stop:
                    return
                time.sleep(0.5)
            if not self._stop:
                print(f"[+{time.monotonic()-_T0:7.1f}s] [heartbeat] {self.label}", flush=True)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start(); return self

    def stop(self):
        self._stop = True


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _codes(col):
    import pyarrow as pa
    col = col.combine_chunks()
    if not pa.types.is_dictionary(col.type):
        col = col.dictionary_encode()
    return col.indices.to_numpy(zero_copy_only=False), [str(x) for x in col.dictionary.to_pylist()]


def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HB = Heartbeat().start()
    import cellxgene_census
    d4 = _load("d4", "17_logicgate_data.py")
    HB.set(f"opening Census {d4.CENSUS_VERSION} ...")
    census = cellxgene_census.open_soma(census_version=d4.CENSUS_VERSION)
    exp = census["census_data"]["homo_sapiens"]

    def fetch(vf, cap_label):
        tbl = exp.obs.read(value_filter=vf, column_names=["soma_joinid", "cell_type", "donor_id", "dataset_id"]).concat()
        if tbl.num_rows == 0:
            return None
        jid = tbl.column("soma_joinid").to_numpy()
        ct_codes, ct_vals = _codes(tbl.column("cell_type"))
        ds_codes, ds_vals = _codes(tbl.column("dataset_id"))
        dn_codes, dn_vals = _codes(tbl.column("donor_id"))
        gid = ds_codes.astype(np.int64) * (int(dn_codes.max()) + 1) + dn_codes
        keep = []
        for g in np.unique(gid):
            keep.extend(np.where(gid == g)[0][:PER_DONOR_CAP].tolist())
        keep = np.sort(np.array(keep, np.int64)); sel = jid[keep]
        HB.set(f"{cap_label}: materialising {len(ALL_GENES)} genes for {len(sel):,} cells ...")
        ad = cellxgene_census.get_anndata(census, organism="Homo sapiens", obs_coords=sel.tolist(),
                                          var_value_filter=f"feature_name in {ALL_GENES}",
                                          obs_column_names=["cell_type", "donor_id", "dataset_id"])
        vn = list(ad.var["feature_name"]) if "feature_name" in ad.var else list(ad.var_names)
        X = np.asarray(ad.X.todense() if hasattr(ad.X, "todense") else ad.X)
        counts = np.zeros((X.shape[0], len(ALL_GENES)), np.int32)
        for j, g in enumerate(ALL_GENES):
            if g in vn:
                counts[:, j] = X[:, vn.index(g)].astype(np.int32)
        ct = ad.obs["cell_type"].astype(str).to_numpy()
        donor = np.array([f"{a}::{b}" for a, b in zip(ad.obs["dataset_id"].astype(str), ad.obs["donor_id"].astype(str))], object)
        dataset = np.array([str(a) for a in ad.obs["dataset_id"].astype(str)], object)
        return counts, ct, donor, dataset

    all_c, all_lab, all_don, all_ds, all_tum = [], [], [], [], []
    for tissue in d4.NORMAL_TISSUES:
        r = fetch(f"is_primary_data == True and disease == 'normal' and tissue_general == '{tissue}'", f"normal {tissue}")
        if r is None:
            continue
        counts, ct, donor, dataset = r
        lab = np.array([next((v for k, v in d4.VITAL_AUDIT.items() if k in c.lower()), None) for c in ct], object)
        all_c.append(counts); all_lab.append(lab); all_don.append(donor); all_ds.append(dataset); all_tum.append(np.zeros(len(ct), bool))
    r = fetch("is_primary_data == True and cell_type in ['malignant cell','neoplastic cell']", "malignant")
    if r is not None:
        counts, ct, donor, dataset = r
        all_c.append(counts); all_lab.append(np.array([None] * len(ct), object)); all_don.append(donor); all_ds.append(dataset); all_tum.append(np.ones(len(ct), bool))
    HB.stop()

    counts = np.vstack(all_c); label = np.concatenate(all_lab); donor = np.concatenate(all_don)
    dataset = np.concatenate(all_ds); is_tum = np.concatenate(all_tum)
    results, meta = evaluate_all(counts, ALL_GENES, label, donor, dataset, is_tum)
    log(f"meta: {meta}")
    clean = [r for r in results if r["safe_effective"]]
    best_and = next((r for r in results if r["op"] == "AND" and r["safe_effective"]), None)
    # transparency: the leak audit — is any single program's leak driven by a POST-MITOTIC type (artifact)?
    POST_MITOTIC = {"cardiomyocyte", "neuron", "cardiac_conduction", "skeletal_myocyte", "pancreatic_islet"}
    single_audit = {r["gate"]: {"coverage": r["coverage"], "robust_leak_p90": r["robust_leak_p90"],
                                "worst_donor_leak": r["worst_donor_leak"], "worst_type": r["worst_type"],
                                "worst_type_is_post_mitotic": r["worst_type"] in POST_MITOTIC}
                    for r in results if r["op"] == "single"}

    out = {
        "tag": "rung23_autonomous_andgate_v2",
        "version": "v2 — depth-gated (HK panel) + robust per-vital-type p90-donor leak (fixes v1 dropout-deflated "
                   "coverage + single-worst-donor artifact)",
        "question": "Is there an intracellular AND-pair firing in tumour cells but ~zero vital normal cells "
                    "(robust p90-donor leak, depth-gated) -> an MHC-independent autonomous self-destruct gate?",
        "programs": PROGRAMS, "hk_depth_gate": HK_GENES, "hk_min": HK_MIN,
        "fire_frac": FIRE_FRAC, "leak_safe_p90": LEAK_SAFE, "cover_min": COVER_MIN,
        "meta": meta,
        "all_gates_ranked": results[:40],
        "safe_effective_gates": clean,
        "best_AND_gate": best_and,
        "single_program_leak_audit": single_audit,
        "HEADLINE": (f"v2 (depth-gated, robust leak): {len(clean)} safe&effective gates "
                     f"(p90-donor vital leak ≤{LEAK_SAFE}, coverage ≥{COVER_MIN}). Best AND: "
                     f"{best_and['gate'] if best_and else 'NONE'}. Compare to v1 (no depth-gate, single-worst-donor)."),
        "CEILING": "mRNA = PROXY for the state a circuit senses; depth-gate reduces not erases dropout (read "
                   "tumour-vs-vital CONTRAST); sensor-per-program + delivery = wet-lab. BOUNDS whether a clean "
                   "intracellular AND-pair exists; not a built circuit.",
    }
    RESULT_JSON.write_text(json.dumps(out, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"well-sequenced: {meta['frac_well_sequenced']:.1%}; vital_well {meta['n_vital_well']:,}; tumour_well {meta['n_tumour_well']:,}")
    log(f"safe&effective: {[r['gate'] for r in clean][:8] or 'NONE'}")
    log("single-program leak audit (is the leak a post-mitotic artifact?):")
    for g, a in single_audit.items():
        log(f"  {g:12} cov={a['coverage']:.3f} p90leak={a['robust_leak_p90']:.3f} worstDonor={a['worst_donor_leak']:.3f} "
            f"type={a['worst_type']} post_mitotic={a['worst_type_is_post_mitotic']}")
    _make_figure(results)
    return 0


def _make_figure(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable ({e})"); return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r in results:
        c = {"single": "#888", "AND": "#3F7D54", "OR": "#E0A040"}[r["op"]]
        ax.scatter(r["robust_leak_p90"] * 100, r["coverage"] * 100, color=c, s=28, alpha=0.8)
    ax.axvline(LEAK_SAFE * 100, ls="--", color="#B23A2E", label=f"safe p90 leak ≤{LEAK_SAFE*100:.0f}%")
    ax.axhline(COVER_MIN * 100, ls=":", color="grey", label=f"effective cov ≥{COVER_MIN*100:.0f}%")
    for op, c in [("single", "#888"), ("AND", "#3F7D54"), ("OR", "#E0A040")]:
        ax.scatter([], [], color=c, label=op)
    ax.set_xlabel("robust vital leak — p90-donor, max over types (%)  ← safer"); ax.set_ylabel("tumour coverage (%, depth-gated)")
    ax.set_title("RUNG-23 v2: autonomous intracellular gates (depth-gated + robust leak)\ntop-left green = clean buildable AND-gate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xscale("symlog", linthresh=1)
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=130)
    log(f"wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    go = ALL_GENES
    idx = {g: j for j, g in enumerate(go)}

    def cell(progs, deep=True):
        c = np.zeros(len(go), np.int32)
        for p in progs:
            for g in PROGRAMS[p]:
                c[idx[g]] = 5
        if deep:
            for g in HK_GENES:
                c[idx[g]] = 7                              # well-sequenced
        return c

    # depth gate: a shallow cell (no HK) is excluded even if a program "fires"
    cm = np.array([cell(["PROLIF"], deep=True), cell(["PROLIF"], deep=False)])
    pf, well = cell_state(cm, go)
    check("PROLIF fires in both (panel present)", pf["PROLIF"][0] and pf["PROLIF"][1])
    check("depth gate: deep cell well-seq, shallow NOT", well[0] and not well[1])

    # gate logic
    pf2, _ = cell_state(np.array([cell(["PROLIF", "GLYCOLYSIS"])]), go)
    check("AND requires both", gate_fire(pf2, "PROLIF", "GLYCOLYSIS", "AND")[0])
    check("OR requires either", gate_fire({"A": np.array([True]), "B": np.array([False])}, "A", "B", "OR")[0])

    # ROBUST leak kills the single-donor artifact: 1 outlier donor fires, 9 don't -> worst=1.0 but p90 small
    rows_c, rows_lab, rows_don, rows_ds, rows_tum = [], [], [], [], []
    for di in range(10):                                   # 10 vital donors, only donor 0 "fires" PROLIF
        for _ in range(40):
            rows_c.append(cell(["PROLIF"]) if di == 0 else cell([]))
            rows_lab.append("cardiomyocyte"); rows_don.append(f"dsV::V{di}"); rows_ds.append("dsV"); rows_tum.append(False)
    for _ in range(40):
        rows_c.append(cell(["PROLIF", "GLYCOLYSIS"])); rows_lab.append(None); rows_don.append("dsT::T1"); rows_ds.append("dsT"); rows_tum.append(True)
    C = np.array(rows_c); L = np.array(rows_lab, object); D = np.array(rows_don, object); DS = np.array(rows_ds, object); TUM = np.array(rows_tum, bool)
    res, meta = evaluate_all(C, go, L, D, DS, TUM)
    g = {r["gate"]: r for r in res}
    check("single-donor artifact: worst_donor_leak high (1.0)", abs(g["PROLIF"]["worst_donor_leak"] - 1.0) < 1e-9)
    check("ROBUST p90 leak resists the 1/10 outlier (<=0.1)", g["PROLIF"]["robust_leak_p90"] <= 0.1)
    check("worst_type reported (cardiomyocyte = post-mitotic flag)", g["PROLIF"]["worst_type"] == "cardiomyocyte")

    # make-or-break still holds: AND(PROLIF,GLYCOLYSIS) covers tumour, vital never fires GLYCOLYSIS -> clean
    check("AND covers tumour (cov==1)", abs(g["PROLIF AND GLYCOLYSIS"]["coverage"] - 1.0) < 1e-9)
    check("AND clean (p90 leak 0)", abs(g["PROLIF AND GLYCOLYSIS"]["robust_leak_p90"] - 0.0) < 1e-9)
    check("AND safe&effective, single PROLIF not (robust)", g["PROLIF AND GLYCOLYSIS"]["safe_effective"] and not g["PROLIF"]["safe_effective"])

    # depth-gate excludes shallow tumour cells from coverage
    Csh = np.vstack([C, np.array([cell(["PROLIF", "GLYCOLYSIS"], deep=False)] * 20)])
    Lsh = np.concatenate([L, np.array([None] * 20, object)]); Dsh = np.concatenate([D, np.array(["dsT::T2"] * 20, object)])
    DSsh = np.concatenate([DS, np.array(["dsT"] * 20, object)]); TUMsh = np.concatenate([TUM, np.ones(20, bool)])
    _, metash = evaluate_all(Csh, go, Lsh, Dsh, DSsh, TUMsh)
    check("shallow tumour cells excluded by depth gate", metash["n_tumour_well"] == meta["n_tumour_well"])

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(selftest())
    elif cmd == "run":
        sys.exit(main_run())
    print(f"unknown: {cmd}"); sys.exit(64)
