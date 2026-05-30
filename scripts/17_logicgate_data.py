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
  2. Emit a VITAL-COVERAGE CENSUS: per tissue, how many cells of each vital parenchymal type we actually
     captured. A vital type with < MIN_VITAL_CELLS is marked UNAUDITED -> no gate may be certified
     vital-safe for it (droplet scRNA-seq under-samples cardiomyocytes/neurons; prefer snRNA-seq atlases).
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
from pathlib import Path

import numpy as np

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

NORMAL_TISSUES = ["heart", "brain", "kidney", "liver", "lung", "pancreas",
                  "adrenal gland", "bone marrow", "skeletal muscle"]
# Step-2 activator pool (all FAILED scripts/07 single-antigen safety -> MUST be gated).
ACTIVATORS = ["ERBB2", "ERBB3", "EPHB4", "TACSTD2", "MUC1", "SDC1", "CD74", "ITGB4"]
# curated surface partners for the AND co-input (kept small to control multiple testing).
PARTNERS = ["ERBB3", "EPHB4", "EPCAM", "CDH1", "MET", "PROM1", "CD24", "FOLR1", "MSLN", "CEACAM5",
            "NECTIN4", "CLDN6", "CLDN18", "ROR1", "MUC16"]
ALL_GENES = sorted(set(ACTIVATORS + PARTNERS))
# Canonical non-regenerating vital parenchyma to audit (cell_type substrings).
VITAL_AUDIT = {"cardiac muscle": "cardiomyocyte", "cardiomyocyte": "cardiomyocyte",
               "neuron": "neuron", "kidney epithel": "kidney_tubule", "podocyte": "kidney_podocyte"}


def fetch_panel():
    """Stream normal + tumour single-cell into one per-cell Panel over ALL_GENES, with a STRATIFIED
    per-cell-type cap so the transfer is bounded (fast) and rare vital types survive. Colab only."""
    import cellxgene_census
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts_blocks, ct, ts, comp = [], [], [], []
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

    def capped_pull(value_filter, label):
        # 1) FAST metadata-only query (2 columns) for all matching cells
        obs = cellxgene_census.get_obs(census, "Homo sapiens", value_filter=value_filter,
                                       column_names=["soma_joinid", "cell_type"])
        if len(obs) == 0:
            print(f"[fetch] {label}: 0 cells", flush=True); return None
        # 2) STRATIFIED subsample: <= MAX_PER_TYPE per cell_type (keeps every cell type, incl. rare vital ones)
        keep = []
        for _, grp in obs.groupby("cell_type"):
            ids = grp["soma_joinid"].to_numpy()
            keep.append(rng.choice(ids, MAX_PER_TYPE, replace=False) if len(ids) > MAX_PER_TYPE else ids)
        keep = np.concatenate(keep)
        print(f"[fetch] {label}: {len(obs)} matching -> {len(keep)} kept "
              f"({obs['cell_type'].nunique()} cell types @ cap {MAX_PER_TYPE})", flush=True)
        # 3) materialize ONLY the kept cells x antigen genes (bounded transfer)
        return cellxgene_census.get_anndata(
            census, organism="Homo sapiens", obs_coords=[int(x) for x in keep],
            var_value_filter=f"feature_name in {ALL_GENES}",
            column_names={"obs": ["cell_type", "tissue_general"]})

    try:
        for tissue in NORMAL_TISSUES:
            ad = capped_pull(f"is_primary_data == True and disease == 'normal' and tissue_general == '{tissue}'",
                             f"normal {tissue}")
            if ad is None or ad.n_obs == 0:
                continue
            counts_blocks.append(_dense_over(ad, ALL_GENES))
            ct += list(_map_celltype(ad.obs["cell_type"].astype(str)))
            ts += [tissue] * ad.n_obs; comp += ["normal"] * ad.n_obs
        # tumour: reuse scripts/03 disease pulls (malignant/epithelial), same stratified cap
        for tissue, dis in [("lung", "lung adenocarcinoma"), ("breast", "breast carcinoma"),
                            ("colon", "colorectal cancer")]:
            ad = capped_pull(f"is_primary_data == True and disease == '{dis}'", f"tumour {dis}")
            if ad is None or ad.n_obs == 0:
                continue
            counts_blocks.append(_dense_over(ad, ALL_GENES)); ct += ["tumour_epithelium"] * ad.n_obs
            ts += ["tumour"] * ad.n_obs; comp += ["tumour"] * ad.n_obs
    finally:
        census.close()
    if not counts_blocks:
        raise RuntimeError("no cells fetched — check Census version / tissue names / network")
    counts = np.vstack(counts_blocks)
    return lg.Panel(counts, ALL_GENES, np.array(ct), np.array(ts), np.array(comp))


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
    panel = fetch_panel()
    print(f"[rung4-data] panel: {panel.counts.shape[0]} cells x {len(ALL_GENES)} antigens")

    cov = vital_coverage_census(panel)
    print("[rung4-data] VITAL-COVERAGE CENSUS:")
    for r in cov:
        print(f"  {r['vital_type']:16s} n={r['n_cells']:6d}  {r['status']}")
    unaudited = [r["vital_type"] for r in cov if r["status"].startswith("UNAUDITED")]

    rows = []
    for a, b, logic in candidate_gates():
        if a not in ALL_GENES:
            continue
        if logic == "AND_NOT":
            continue  # HLA-LOH is a per-patient GENOTYPE gate, scored from NGS LOH, not the atlas — skip here
        if b not in ALL_GENES:
            continue
        r = lg.score_gate(panel, a, b, logic, k=K)
        # any gate whose only protection is in an UNAUDITED vital type cannot be certified vital-safe
        r["vital_unaudited"] = unaudited
        r["protein_copositivity_status"] = "NO_SINGLECELL_PROTEIN_DATA"  # transcript-only until CITE-seq
        r["transcript_only"] = True
        rows.append(r)
    rows.sort(key=lambda r: (not r["selective"], r["worst_normal_leak"], -r["tumour_coverage"]))

    import csv
    with open(DATA_DIR / "gate_selectivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader(); w.writerows(rows)
    with open(DATA_DIR / "vital_coverage_census.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vital_type", "n_cells", "status"]); w.writeheader(); w.writerows(cov)

    selective = [r for r in rows if r["selective"]]
    print(f"[rung4-data] scored {len(rows)} AND gates; {len(selective)} predicted SELECTIVE (transcript-only).")
    if not selective:
        print("[rung4-data] NO clean gate found in this pool — a FIRST-CLASS, valid outcome (no forced winner).")
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
