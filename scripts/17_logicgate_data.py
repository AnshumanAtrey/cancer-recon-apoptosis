#!/usr/bin/env python3
"""
RUNG 4 / Step-5 — logic-gate DATA LAYER (real single-cell discovery; Colab/GPU-free, runs on CELLxGENE).

The selectivity ENGINE (scripts/18) and its RUN-TRUST validation (scripts/20) run anywhere. THIS script
is the only part that needs the real single-cell atlases, so it is built to run on Colab where
`cellxgene_census` + network are available (the atlases are GBs and are NOT committed — gitignored, like
Step-2's data/cellxgene/).

What it does, one tissue in RAM at a time (never bulk/pseudobulk):
  1. Stream NORMAL per-tissue single-cell slices from CZ CELLxGENE Census for ALL Step-3 vital tissues
     (heart, brain, kidney, liver, lung, pancreas, adrenal gland, bone marrow, skeletal muscle), subset to
     the antigen-pool genes only (keeps each slice small).
  2. Emit a VITAL-COVERAGE CENSUS. SAFETY MECHANICS (audit-hardened): vital-parenchyma cells are kept in
     FULL (asymmetric cap — only abundant non-vital types are capped) so a rare lethal double-positive is
     not statistically erased; leaks are Jeffreys UPPER bounds (a false zero from dropout cannot pass); and
     it FAILS CLOSED — if any non-regen vital type (heart/brain/kidney/pancreas/adrenal/muscle) was NOT
     adequately captured, the gate is UNCERTAIN, never silently SELECTIVE ('never looked at the heart' !=
     'the heart is clean'). Multiple-testing control = held-out-donor replication is DEFERRED to the next
     pass; selective gates are a DISCOVERY shortlist until then.
  3. Pull the TUMOUR single-cell (reuse scripts/03's lung/breast/colon malignant pulls).
  4. Assemble a per-cell Panel and score the candidate AND / AND-NOT gates with scripts/18, emitting
     gate_selectivity.csv (both tumour COVERAGE and worst-case NORMAL LEAK, vital broken out).

HONEST: this is a transcript-level HYPOTHESIS screen. mRNA != surface protein (single-cell r~0.1-0.4);
a NOT/absence can be dropout; co-localisation != a functional circuit that fires caspase-8 (wet-lab).
HLA-LOH is a per-patient GENETIC NOT (NGS-stratified), not an atlas expression call. Recognition-
selectivity is a SEPARATE axis — never multiplied with RUNG-1/2/3 (asserted via scripts/18).

USAGE (Colab):  python scripts/17_logicgate_data.py
REQS        :  pip install cellxgene-census scanpy   (CPU; streams TileDB-SOMA)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_T0 = time.monotonic()


def log(msg):
    """Timestamped (elapsed-seconds) progress line so the run is never a blind box."""
    print(f"[+{time.monotonic() - _T0:7.1f}s] [rung4] {msg}", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung4_logicgate"
DATA_DIR = PROJECT_ROOT / "data" / "logicgate"
CENSUS_VERSION = "2024-07-01"            # pinned to match scripts/03
MIN_VITAL_CELLS = 200                    # below this a vital type is UNAUDITED
K = 2                                    # per-cell POSITIVE threshold (UMI >= K)
MAX_PER_TYPE = 1500                      # STRATIFIED cap: <=N cells per cell_type per tissue. Bounds the
#                                          transfer (the reason an uncapped pull took >20 min) AND preserves
#                                          rare vital types (per-cell-type co-positivity needs ~hundreds, not millions).
SEED = 20260530

lg_spec = importlib.util.spec_from_file_location("lg", PROJECT_ROOT / "scripts" / "18_logicgate_search.py")
lg = importlib.util.module_from_spec(lg_spec); lg_spec.loader.exec_module(lg)

NORMAL_TISSUES = ["heart", "brain", "kidney", "liver", "lung", "bone marrow",
                  "pancreas", "adrenal gland", "skeletal muscle"]   # all 9: non-regen vital + regen + tumour-matched
TUMOUR_DISEASES = ["lung adenocarcinoma", "breast carcinoma", "colorectal cancer"]
# Step-2 activator pool (all FAILED scripts/07 single-antigen safety -> MUST be gated).
ACTIVATORS = ["ERBB2", "ERBB3", "EPHB4", "TACSTD2", "MUC1", "SDC1", "CD74", "ITGB4"]
# curated surface partners for the AND co-input (kept small to control multiple testing).
PARTNERS = ["ERBB3", "EPHB4", "EPCAM", "CDH1", "MET", "PROM1", "CD24", "FOLR1", "MSLN", "CEACAM5",
            "NECTIN4", "CLDN6", "CLDN18", "ROR1", "MUC16"]
ALL_GENES = sorted(set(ACTIVATORS + PARTNERS))
# Canonical non-regenerating vital parenchyma to audit (cell_type substrings -> canonical label).
# Cardiac entries FIRST so 'cardiac muscle cell' maps to cardiomyocyte before any 'muscle' rule.
# NOTE: if a real Census label doesn't match here, that vital type stays UNAUDITED and the gate
# FAILS CLOSED (UNCERTAIN) — imperfect mapping is now conservative-safe, never lethally false-safe.
VITAL_AUDIT = {"cardiac muscle": "cardiomyocyte", "cardiomyocyte": "cardiomyocyte",
               "neuron": "neuron", "glial": "neuron",
               "kidney epithel": "kidney_tubule", "renal": "kidney_tubule", "podocyte": "kidney_podocyte",
               "pancreatic": "pancreatic_islet", "islet": "pancreatic_islet", "beta cell": "pancreatic_islet",
               "adrenal": "adrenal_cortical",
               "skeletal muscle": "skeletal_myocyte", "skeletal": "skeletal_myocyte"}


def _q(items):
    """SOMA value-filter list literal, e.g. ['heart','brain']."""
    return "[" + ", ".join(f"'{x}'" for x in items) + "]"


def _stream_pull(census, value_filter, label, tissue_index=0):
    """STREAM one slice via get_anndata(obs_value_filter=...) — a CONTIGUOUS, predicate-pushed read — then
    map cell_types to canonical labels and ASYMMETRICALLY subsample: keep ALL vital-parenchyma cells (so a
    rare lethal double-positive cardiomyocyte cannot be statistically erased), cap only abundant non-vital
    types at MAX_PER_TYPE. Mapping happens BEFORE the cap so vital cells are recognised first. Independent
    RNG per tissue. Returns (counts, canonical_cell_types, tissues) or None."""
    import cellxgene_census
    log(f"{label}: streaming get_anndata (contiguous predicate read) ...")
    ad = cellxgene_census.get_anndata(
        census, organism="Homo sapiens", obs_value_filter=value_filter,
        var_value_filter=f"feature_name in {ALL_GENES}",
        column_names={"obs": ["cell_type", "tissue_general"]})
    if ad.n_obs == 0:
        log(f"{label}: 0 cells matched — check disease/tissue labels"); return None
    mapped = np.array(_map_celltype(ad.obs["cell_type"].astype(str).to_numpy()))   # MAP BEFORE CAP
    log(f"{label}: {ad.n_obs:,} cells read; asymmetric cap (vital kept in FULL, non-vital <= {MAX_PER_TYPE}) ...")
    rng = np.random.default_rng([SEED, tissue_index])   # independent draws per tissue
    keep = []
    for lab in np.unique(mapped):
        idx = np.where(mapped == lab)[0]
        cap = None if lab in lg.VITAL_NONREGEN else MAX_PER_TYPE   # keep ALL vital-parenchyma cells
        keep.append(idx if (cap is None or len(idx) <= cap) else rng.choice(idx, cap, replace=False))
    keep = np.sort(np.concatenate(keep))
    ad = ad[keep]; mapped = mapped[keep]
    vital_kept = sorted(set(mapped.tolist()) & lg.VITAL_NONREGEN)
    log(f"{label}: kept {ad.n_obs:,} cells; vital types kept in FULL: {vital_kept or 'NONE in this tissue'}")
    return _dense_over(ad, ALL_GENES), list(mapped), list(ad.obs["tissue_general"].astype(str))


def fetch_panel():
    """Per-cell Panel over ALL_GENES, streamed PER TISSUE (contiguous reads), fully logged."""
    import cellxgene_census
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts_blocks, ct, ts, comp = [], [], [], []
    log(f"opening CELLxGENE Census version {CENSUS_VERSION} ...")
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    log("census open ✓")
    try:
        for ti, tissue in enumerate(NORMAL_TISSUES):   # one streaming read per tissue (bounds RAM)
            res = _stream_pull(census,
                               f"is_primary_data == True and disease == 'normal' and tissue_general == '{tissue}'",
                               f"NORMAL {tissue}", tissue_index=ti)
            if res is not None:
                c, cts, tss = res   # cts already mapped to canonical labels inside _stream_pull
                counts_blocks.append(c); ct += list(cts); ts += tss; comp += ["normal"] * len(cts)
        res = _stream_pull(census,
                           f"is_primary_data == True and disease in {_q(TUMOUR_DISEASES)}",
                           "TUMOUR (all diseases)", tissue_index=99)
        if res is not None:
            c, cts, tss = res
            counts_blocks.append(c); ct += ["tumour_epithelium"] * len(cts)
            ts += ["tumour"] * len(cts); comp += ["tumour"] * len(cts)
    finally:
        census.close(); log("census closed")
    if not counts_blocks:
        raise RuntimeError("no cells fetched — check Census version / tissue & disease labels / network")
    panel = lg.Panel(np.vstack(counts_blocks), ALL_GENES, np.array(ct), np.array(ts), np.array(comp))
    n_norm = int((panel.compartment == "normal").sum()); n_tum = int((panel.compartment == "tumour").sum())
    log(f"panel assembled: {panel.counts.shape[0]:,} cells ({n_norm:,} normal, {n_tum:,} tumour) x {len(ALL_GENES)} antigens")
    for tis in sorted(set(panel.tissue)):
        log(f"  tissue {tis:14s}: {int((panel.tissue == tis).sum()):,} cells")
    return panel


def _dense_over(ad, genes):
    """Return (n_cells, len(genes)) integer counts in ALL_GENES order (0 for genes absent in this slice)."""
    import scipy.sparse as sp
    name_to_col = {n: i for i, n in enumerate(ad.var["feature_name"].astype(str))}
    out = np.zeros((ad.n_obs, len(genes)), dtype=np.int32)
    X = ad.X.tocsc() if sp.issparse(ad.X) else ad.X
    for j, g in enumerate(genes):
        if g in name_to_col:
            col = X[:, name_to_col[g]]
            out[:, j] = (col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()).astype(np.int32)
    return out


def _map_celltype(raw):
    """Collapse Census cell_type strings to the canonical vital labels scripts/18 protects (else keep raw)."""
    mapped = []
    for c in raw:
        cl = c.lower()
        hit = next((v for key, v in VITAL_AUDIT.items() if key in cl), c)
        mapped.append(hit)
    return mapped


def vital_coverage_census(panel):
    """Per (vital type) cell counts; flag UNAUDITED below MIN_VITAL_CELLS (droplet under-sampling)."""
    rows = []
    for vt in sorted(set(lg.VITAL_NONREGEN)):
        n = int((panel.cell_type == vt).sum())
        rows.append({"vital_type": vt, "n_cells": n,
                     "status": "AUDITED" if n >= MIN_VITAL_CELLS else "UNAUDITED (under-sampled — use snRNA-seq)"})
    return rows


def candidate_gates():
    gates = []
    for a in ACTIVATORS:
        for b in PARTNERS:
            if b != a:
                gates.append((a, b, "AND"))
        gates.append((a, "HLA_A02_LOH", "AND_NOT"))   # Tmod genetic NOT (flagged genotype, not atlas expression)
    return gates


def main() -> int:
    lg.assert_no_multiply()
    if importlib.util.find_spec("cellxgene_census") is None:
        print("[rung4-data] cellxgene_census not installed — this script runs on COLAB.")
        print("[rung4-data] locally, the METHOD is validated by scripts/20 (synthetic ground truth).")
        print("[rung4-data] on Colab:  pip install cellxgene-census scanpy  then re-run.")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== RUNG 4 real discovery — CELLxGENE single-cell logic-gate search ===")
    panel = fetch_panel()

    cov = vital_coverage_census(panel)
    log("VITAL-COVERAGE CENSUS (heart/brain/kidney must be AUDITED to certify vital-safe):")
    for r in cov:
        log(f"  {r['vital_type']:16s} n={r['n_cells']:6d}  {r['status']}")
    unaudited = [r["vital_type"] for r in cov if r["status"].startswith("UNAUDITED")]

    # HLA-LOH is a per-patient GENOTYPE gate (NGS LOH), not an atlas-expression NOT -> not searched here.
    specs = [(a, b, logic) for a, b, logic in candidate_gates()
             if logic == "AND" and a in ALL_GENES and b in ALL_GENES]
    log(f"scoring {len(specs)} candidate AND gates over {panel.counts.shape[0]:,} cells (vectorised) ...")

    def _prog(i, n, r):
        if i % 10 == 0 or i == n or r["selective"]:
            tag = "SELECTIVE" if r["selective"] else "no"
            log(f"  scored {i:3d}/{n}  {r['gate']:26s} cov={r['tumour_coverage']:.2f} "
                f"leak={r['worst_normal_leak']:.2f} vital={r['vital_leak']:.2f} -> {tag}")

    # FAIL-CLOSED: require every non-regen vital type to be adequately captured; a gate where heart/brain/
    # kidney/pancreas/adrenal/muscle was NOT captured is UNCERTAIN, never silently SELECTIVE. Leaks are
    # Jeffreys UPPER bounds, vital cells kept in full (asymmetric cap) — so a false zero cannot pass.
    rows = lg.score_gates_batch(panel, specs, k=K, required_vital=lg.VITAL_NONREGEN, progress=_prog)
    for r in rows:
        r["protein_copositivity_status"] = "NO_SINGLECELL_PROTEIN_DATA"  # transcript-only until CITE-seq
        r["transcript_only"] = True
        r["multiple_testing_control"] = "held-out-donor replication DEFERRED (next pass) — treat selective as DISCOVERY shortlist"
    rows.sort(key=lambda r: (not r["selective"], r["worst_normal_leak"], -r["tumour_coverage"]))

    import csv
    with open(DATA_DIR / "gate_selectivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader(); w.writerows(rows)
    with open(DATA_DIR / "vital_coverage_census.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vital_type", "n_cells", "status"]); w.writeheader(); w.writerows(cov)

    log(f"wrote data/logicgate/gate_selectivity.csv ({len(rows)} gates) + vital_coverage_census.csv")
    selective = [r for r in rows if r["selective"]]
    log(f"scored {len(rows)} AND gates; {len(selective)} predicted SELECTIVE (transcript-only).")
    for r in selective[:10]:
        log(f"  SELECTIVE: {r['gate']:26s} cov={r['tumour_coverage']:.2f} leak={r['worst_normal_leak']:.2f}")
    if not selective:
        log("NO clean gate found in this pool — a FIRST-CLASS, valid outcome (no forced winner).")
    (OUT_DIR / "rung4_results.json").write_text(json.dumps({
        "census_version": CENSUS_VERSION, "n_cells": int(panel.counts.shape[0]),
        "vital_coverage": cov, "unaudited_vital_types": unaudited,
        "n_gates": len(rows), "n_selective": len(selective),
        "top_gates": rows[:15], "no_clean_gate": len(selective) == 0,
        "CEILING": "transcript-level hypothesis; mRNA!=surface protein (CITE-seq needed to confirm "
                   "co-positivity); HLA-LOH is an NGS genotype gate not modelled here; recognition is a "
                   "separate axis never multiplied with RUNG-1/2/3; the durability cost is in scripts/21.",
    }, indent=2, default=str))
    print("[rung4-data] -> data/logicgate/gate_selectivity.csv + runs/rung4_logicgate/rung4_results.json")
    _figure(rows, cov)
    print("[rung4-data] CEILING: transcript-only HYPOTHESIS; CITE-seq/flow/co-culture confirm; agonism = wet-lab.")
    return 0


def _figure(rows, cov):
    """Real-discovery figure: gate coverage-vs-leak frontier + vital-coverage census."""
    if not rows:
        return
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        for r in rows:
            ax[0].scatter(r["tumour_coverage"], r["worst_normal_leak"],
                          c="#27ae60" if r["selective"] else "#c0392b", s=45, alpha=0.7)
        ax[0].axhline(0.02, ls="--", color="green", lw=0.8); ax[0].axvline(0.30, ls="--", color="green", lw=0.8)
        ax[0].set_xlabel("tumour coverage (want high)"); ax[0].set_ylabel("worst normal-cell leak (want ~0)")
        ax[0].set_title("AND-gate frontier (transcript-only)\ngreen=selective; target=lower-right box")
        sel = [r for r in rows if r["selective"]][:8]
        for r in sel:
            ax[0].annotate(r["gate"][:18], (r["tumour_coverage"], r["worst_normal_leak"]), fontsize=6)
        ax[1].barh([c["vital_type"] for c in cov], [c["n_cells"] for c in cov],
                   color=["#2980b9" if c["status"] == "AUDITED" else "#e67e22" for c in cov])
        ax[1].axvline(MIN_VITAL_CELLS, ls="--", color="red", lw=0.8)
        ax[1].set_xlabel("cells captured"); ax[1].set_title("vital-coverage census\n(orange = UNAUDITED, < min)")
        fig.suptitle("RUNG 4 real discovery — transcript-only hypothesis (mRNA!=protein; CITE-seq confirms). "
                     "Recognition is a separate axis; 'no clean gate' is a valid result.", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_DIR / "rung4_discovery.png", dpi=110)
        print("[rung4-data] figure -> runs/rung4_logicgate/rung4_discovery.png")
    except Exception as e:
        print(f"[rung4-data] figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
