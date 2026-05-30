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
    """Stream normal + tumour single-cell into one per-cell Panel over ALL_GENES. Colab only."""
    import cellxgene_census
    import scanpy as sc  # noqa: F401  (AnnData handling)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts_blocks, ct, ts, comp = [], [], [], []
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)
    try:
        for tissue in NORMAL_TISSUES:
            print(f"[fetch] normal {tissue} ...", flush=True)
            ad = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", var_value_filter=f"feature_name in {ALL_GENES}",
                obs_value_filter=(f"is_primary_data == True and disease == 'normal' and "
                                  f"tissue_general == '{tissue}'"),
                column_names={"obs": ["cell_type", "tissue_general"]})
            if ad.n_obs == 0:
                continue
            X = _dense_over(ad, ALL_GENES)
            counts_blocks.append(X)
            ct += list(_map_celltype(ad.obs["cell_type"].astype(str)))
            ts += [tissue] * ad.n_obs; comp += ["normal"] * ad.n_obs
        # tumour: reuse scripts/03 disease!=normal malignant/epithelial pulls
        for tissue, dis in [("lung", "lung adenocarcinoma"), ("breast", "breast carcinoma"),
                            ("colon", "colorectal cancer")]:
            print(f"[fetch] tumour {dis} ...", flush=True)
            ad = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", var_value_filter=f"feature_name in {ALL_GENES}",
                obs_value_filter=(f"is_primary_data == True and disease == '{dis}'"),
                column_names={"obs": ["cell_type", "tissue_general"]})
            if ad.n_obs == 0:
                continue
            X = _dense_over(ad, ALL_GENES)
            counts_blocks.append(X); ct += ["tumour_epithelium"] * ad.n_obs
            ts += ["tumour"] * ad.n_obs; comp += ["tumour"] * ad.n_obs
    finally:
        census.close()
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
    print("[rung4-data] CEILING: transcript-only HYPOTHESIS; CITE-seq/flow/co-culture confirm; agonism = wet-lab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
