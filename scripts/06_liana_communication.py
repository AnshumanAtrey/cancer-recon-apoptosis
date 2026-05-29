#!/usr/bin/env python3
"""
Step 2c — LIANA+ cell-cell communication annotation (supports the bystander narrative).

For each tissue, infer ligand-receptor communication between cell types with LIANA+
(consensus of CellPhoneDB/CellChat/NATMI/Connectome/log2FC/SingleCellSignalR) on the
tumour and normal AnnData, then annotate the Step-2b SURFACE shortlist with:
  - is the badge a RECEIVER in tumour communication? who SENDS to it?
  - is the signalling cancer→cancer (bystander — cancer cell talks to cancer cell)?
  - is it tumour-specific (present in tumour but not normal communication)?
  - the TRAIL(TNFSF10)→DR5(TNFRSF10B) death-trigger axis: is cancer a receiver?

This is the communication layer of Shriya's "recognise neighbour → trigger self-destruct"
idea. It is SUPPORTING evidence for the hybrid thesis (THESIS.md), not on the critical
path — the anchor specificity (Step 2b + Step 3) is what drives target choice.

Cancer-cell population per tissue uses the same adaptive rule as scripts/04
(malignant/neoplastic where labelled, else epithelial lineage). Mirrored here to keep
this script standalone (see scripts/04 for the canonical copy).

OUTPUT: data/cellxgene/communication_<tissue>_<cond>.csv (receptor-filtered L-R tables)
        data/cellxgene/targets_communication_annotated.csv (surface shortlist + comm flags)

USAGE:  python scripts/06_liana_communication.py
REQS :  scanpy, liana (CPU; no GPU). ~few min per condition.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cellxgene"
SURFACE_SHORTLIST = DATA_DIR / "targets_surface_shortlist.csv"
FULL_SHORTLIST = DATA_DIR / "targets_shortlist.csv"

TISSUES = ["lung", "breast", "colon"]
MIN_CELLS = 10          # LIANA: drop cell types with fewer cells
EXPR_PROP = 0.10        # LIANA: min fraction expressing for L and R
TOP_PER_TISSUE = 200    # keep this many strongest receptor-involving interactions per condition

# death-trigger axis
TRAIL = "TNFSF10"
DR5 = "TNFRSF10B"

# --- adaptive cancer-cell selection (mirrors scripts/04) ---
EXPLICIT_CANCER_LABELS = ["malignant cell", "neoplastic cell"]
EPITHELIAL_KEYWORDS = [
    "epithel", "pneumocyte", "enterocyte", "colonocyte", "luminal", "basal cell",
    "goblet", "club cell", "secretory", "ductal", "acinar", "keratinocyte",
    "tuft", "paneth", "enteroendocrine", "ionocyte", "hillock", "serous",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("step2c")


# ---------- pure helpers (unit-tested) ----------
def is_epithelial(cell_type: str) -> bool:
    c = cell_type.lower()
    return any(k in c for k in EPITHELIAL_KEYWORDS)


def select_cancer_celltypes(celltypes) -> list[str]:
    present = set(celltypes)
    explicit = [c for c in EXPLICIT_CANCER_LABELS if c in present]
    if explicit:
        return explicit
    return sorted({c for c in present if is_epithelial(c)})


def complex_subunits(complex_str: str) -> set[str]:
    """A LIANA complex like 'ITGA6_ITGB4' → {'ITGA6','ITGB4'}; single gene → {gene}."""
    return {s for s in str(complex_str).split("_") if s}


def receptor_hits(receptor_complex: str, targets: set[str]) -> set[str]:
    """Which target receptors appear among this interaction's receptor subunits."""
    return complex_subunits(receptor_complex) & targets


# ---------- LIANA run ----------
MAX_CELLS_LIANA = 15000        # safety cap; LIANA densifies during scaling → bound RAM


def _resource_genes() -> set[str]:
    """Genes (ligand + receptor subunits) in LIANA's consensus resource."""
    import liana
    res = liana.resource.select_resource("consensus")
    genes: set[str] = set()
    for col in ("ligand", "receptor"):
        for v in res[col].astype(str):
            genes.update(v.split("_"))
    genes.discard("")
    return genes


def run_liana(adata, label: str):
    import liana as li
    import scanpy as sc
    import numpy as np
    # Normalise on the FULL gene set (CP10k uses total counts), THEN restrict to the
    # L-R resource genes BEFORE rank_aggregate. LIANA only scores resource genes, so
    # this is lossless — but it shrinks the matrix ~15x and avoids the densification OOM
    # ([exit -9]) that 60k genes caused.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # cap cells (deterministic) as a second RAM guard
    if adata.n_obs > MAX_CELLS_LIANA:
        rng = np.random.default_rng(20260529)
        idx = np.sort(rng.choice(adata.n_obs, size=MAX_CELLS_LIANA, replace=False))
        adata = adata[idx].copy()
        log.info("[%s] subsampled to %d cells (RAM guard)", label, adata.n_obs)
    rgenes = _resource_genes()
    keep = [g for g in adata.var_names if g in rgenes]
    log.info("[%s] restricting %d→%d genes (L-R resource) before LIANA", label, adata.n_vars, len(keep))
    adata = adata[:, keep].copy()
    log.info("[%s] running LIANA rank_aggregate (groupby=cell_type, min_cells=%d, expr_prop=%.2f)…",
             label, MIN_CELLS, EXPR_PROP)
    li.mt.rank_aggregate(adata, groupby="cell_type", resource_name="consensus",
                         expr_prop=EXPR_PROP, min_cells=MIN_CELLS, use_raw=False, verbose=False)
    res = adata.uns["liana_res"].copy()
    res.columns = [c.replace(".", "_") for c in res.columns]   # normalise '.'/'_'
    log.info("[%s] LIANA interactions: %d (cell-type pairs × L-R)", label, len(res))
    return res


def annotate(res, targets: set[str], cancer_cts: list[str]):
    """Filter to interactions whose receptor involves a target; tag cancer receiver/sender."""
    keep = res[res["receptor_complex"].apply(lambda r: len(receptor_hits(r, targets)) > 0)].copy()
    keep["target_receptors"] = keep["receptor_complex"].apply(lambda r: ",".join(sorted(receptor_hits(r, targets))))
    cc = set(cancer_cts)
    keep["receiver_is_cancer"] = keep["target"].isin(cc)
    keep["sender_is_cancer"] = keep["source"].isin(cc)
    keep["bystander_cancer_to_cancer"] = keep["receiver_is_cancer"] & keep["sender_is_cancer"]
    sort_col = "magnitude_rank" if "magnitude_rank" in keep.columns else keep.columns[-1]
    return keep.sort_values(sort_col).head(TOP_PER_TISSUE).reset_index(drop=True)


def main() -> int:
    log.info("cancer-recon-apoptosis — Step 2c — LIANA communication annotation")
    try:
        import scanpy as sc
        import pandas as pd
        import liana  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s (pip install scanpy liana)", e); return 2

    # targets = surface shortlist if present, else full shortlist; always include DR5.
    src = SURFACE_SHORTLIST if SURFACE_SHORTLIST.exists() else FULL_SHORTLIST
    if not src.exists():
        log.error("no shortlist (%s) — run scripts/04 (+05) first", src); return 3
    shortlist = pd.read_csv(src)
    targets = set(shortlist["receptor"].astype(str)) | {DR5}
    log.info("targets from %s: %d receptors (+DR5)", src.name, len(targets))

    comm_rows = []     # per-receptor communication annotation
    for tissue in TISSUES:
        log.info("=" * 64)
        tum_p, nor_p = DATA_DIR / f"{tissue}__tumour.h5ad", DATA_DIR / f"{tissue}__normal.h5ad"
        if not tum_p.exists() or not nor_p.exists():
            log.warning("[%s] missing h5ad — skip", tissue); continue
        try:
            tum = sc.read_h5ad(tum_p); nor = sc.read_h5ad(nor_p)
            cancer_cts = select_cancer_celltypes(tum.obs["cell_type"].unique())
            log.info("[%s] cancer cell types: %s", tissue, cancer_cts[:6])

            res_t = run_liana(tum, f"{tissue}/tumour")
            ann_t = annotate(res_t, targets, cancer_cts)
            ann_t.to_csv(DATA_DIR / f"communication_{tissue}_tumour.csv", index=False)

            res_n = run_liana(nor, f"{tissue}/normal")
            ann_n = annotate(res_n, targets, select_cancer_celltypes(nor.obs["cell_type"].unique()))
            ann_n.to_csv(DATA_DIR / f"communication_{tissue}_normal.csv", index=False)

            # tumour L-R pairs (ligand→receptor) and whether they also occur in normal
            tum_pairs = set(zip(ann_t["ligand_complex"], ann_t["receptor_complex"]))
            nor_pairs = set(zip(ann_n["ligand_complex"], ann_n["receptor_complex"]))
            tumour_specific = tum_pairs - nor_pairs

            # per-receptor rollup (tumour)
            for rec in sorted(targets):
                sub = ann_t[ann_t["target_receptors"].str.contains(rec, regex=False, na=False)]
                if not len(sub):
                    continue
                comm_rows.append({
                    "tissue": tissue, "receptor": rec,
                    "n_tumour_interactions": len(sub),
                    "receiver_is_cancer": bool(sub["receiver_is_cancer"].any()),
                    "bystander_cc": bool(sub["bystander_cancer_to_cancer"].any()),
                    "top_ligands": ",".join(sub["ligand_complex"].head(5).astype(str).unique()),
                    "n_tumour_specific": int(sum(1 for lg, rc in zip(sub["ligand_complex"], sub["receptor_complex"])
                                                 if (lg, rc) in tumour_specific)),
                })

            # death-trigger axis report
            trail_dr5 = ann_t[(ann_t["receptor_complex"].apply(lambda r: DR5 in complex_subunits(r)))
                              & (ann_t["ligand_complex"].apply(lambda l: TRAIL in complex_subunits(l)))]
            if len(trail_dr5):
                rc = trail_dr5.iloc[0]
                log.info("[%s] TRAIL→DR5 axis present: %s→%s (receiver_is_cancer=%s)",
                         tissue, rc["source"], rc["target"], rc["receiver_is_cancer"])
            else:
                log.info("[%s] TRAIL→DR5 axis not in top interactions", tissue)
        except Exception as e:
            log.error("[%s] LIANA failed: %s: %s", tissue, type(e).__name__, e)
            continue

    if not comm_rows:
        log.error("no communication results produced"); return 1

    comm = pd.DataFrame(comm_rows)
    # roll up across tissues + merge onto the surface shortlist
    agg = comm.groupby("receptor").agg(
        n_tissues_comm=("tissue", "nunique"),
        any_cancer_receiver=("receiver_is_cancer", "any"),
        any_bystander=("bystander_cc", "any"),
        total_tumour_interactions=("n_tumour_interactions", "sum"),
    ).reset_index()
    merged = shortlist.merge(agg, on="receptor", how="left")
    out = DATA_DIR / "targets_communication_annotated.csv"
    merged.to_csv(out, index=False)
    comm.to_csv(DATA_DIR / "communication_per_tissue_receptor.csv", index=False)

    log.info("=" * 64)
    log.info("COMMUNICATION-ACTIVE SURFACE TARGETS (badge is a tumour receiver; ★=cancer→cancer bystander)")
    log.info("%-12s %6s %10s %8s %9s", "receptor", "nTiss", "cancerRcv", "bystdr", "nInteract")
    show = agg[agg["any_cancer_receiver"] == True].sort_values(
        ["any_bystander", "total_tumour_interactions"], ascending=[False, False])
    for _, r in show.head(25).iterrows():
        log.info("%-12s %6d %10s %8s %9d", r["receptor"], r["n_tissues_comm"],
                 "yes", "★" if r["any_bystander"] else "-", int(r["total_tumour_interactions"]))
    log.info("annotated shortlist saved → %s", out.relative_to(PROJECT_ROOT))
    log.info("✓ done — Step 2 complete. Next: Step 3 anchor specificity audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
