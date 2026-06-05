#!/usr/bin/env python3
"""
RUNG 10b — does a good surface AND-NOT blocker even EXIST? (decisive, CPU-only, no GPU).

arm (a) hit a wall: the cached panel only has TUMOUR-EXPRESSED genes, which make poor NOT-markers (they're in
the tumour too -> subtracting them kills coverage). The BEST-possible surface NOT-marker is TUMOUR-ABSENT but
VITAL-NORMAL-HIGH: ~negC is then TRUE in all tumour (keeps coverage) and FALSE in vital normal (spares it).
This run stops speculating and ASKS THE ATLAS directly: among the ~4300 surface genes that are absent in
tumour, are ANY high in vital normal tissue? Per vital cell type.

DECISIVE OUTCOME
  - If NO tumour-absent surface gene is vital-high for a vital type -> NO surface NOT-marker can spare that
    tissue -> a surface-only AND-NOT recognition gate is IMPOSSIBLE for it (the NOT-slot MUST be genetic).
  - If some EXIST -> we've found the candidate surface blockers -> the GPU sweep (next) tests whether any
    actually yields a SELECTIVE gate (a groundbreaking surface-only recognition gate) or still fails.

DATA: tumour coverage of all 5007 surface genes is FREE (cached RUNG-5 tumour panel, full surfaceome). We only
fetch the vital-normal expression of the TUMOUR-ABSENT genes (~4300), vital cells only, donor-capped, per-tissue
Drive tiles (resumable). Reuses RUNG-9's tested vital-cell fetch (scripts/30.pull_vital_genes).

HONEST CEILING: mRNA != surface protein; dropout deflates 'vital-high' (so this is CONSERVATIVE for finding a
blocker -> if we find none, the negative is strong; if we find some, they still need protein confirmation).
Worst-donor spirit: a blocker must be reliably high (we report per-vital-type detection + the worst-donor min).

USAGE
  python scripts/32_surface_blocker_discovery.py selftest                                  # synthetic
  RUNG10B_CACHE=/content/drive/MyDrive/cancer-recon/rung10b_tiles \
  LOGICGATE_CACHE=/content/drive/MyDrive/cancer-recon/rung5_cache.npz \
      python scripts/32_surface_blocker_discovery.py run                                    # Colab CPU
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung10b_blocker"
RESULT_JSON = OUT_DIR / "rung10b_blocker.json"
FIGURE_PNG = OUT_DIR / "rung10b_blocker.png"
CACHE = Path(os.environ["RUNG10B_CACHE"]) if os.environ.get("RUNG10B_CACHE") else None
TUM_FLOOR = float(os.environ.get("R10_TUM_FLOOR", "0.02"))    # gene is 'tumour-absent' if tumour coverage < this
VITAL_HIGH = float(os.environ.get("R10_VITAL_HIGH", "0.5"))   # 'vital-high' if detected (>=K) in > this frac of a vital type
K = 2


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------------------
def find_candidates(counts, label, tumour_absent_genes):
    """Pure (testable): per vital cell type, the tumour-absent genes that are vital-high (candidate blockers).
    counts:(n, G) over `tumour_absent_genes`; label:(n,) vital-type. Returns (report, vital_types)."""
    G = len(tumour_absent_genes)
    vital_types = sorted(set(x for x in label if x))
    per_type = {}
    high_mat = np.zeros((len(vital_types), G), bool)
    detect_mat = np.zeros((len(vital_types), G))
    for i, t in enumerate(vital_types):
        m = label == t
        det = (counts[m] >= K).mean(axis=0) if m.sum() else np.zeros(G)
        detect_mat[i] = det
        hi = det > VITAL_HIGH
        high_mat[i] = hi
        idx = np.where(hi)[0]
        top = [tumour_absent_genes[j] for j in idx[np.argsort(-det[idx])][:25]]
        per_type[t] = {"n_candidate_blockers": int(hi.sum()), "n_cells": int(m.sum()), "top_blockers": top}
    universal = [tumour_absent_genes[j] for j in np.where(high_mat.all(axis=0))[0]] if len(vital_types) else []
    n_no_blocker = sum(1 for t in vital_types if per_type[t]["n_candidate_blockers"] == 0)
    report = {
        "n_tumour_absent_genes_screened": G,
        "per_vital_type": per_type,
        "universal_blockers_high_in_ALL_vital": universal,
        "n_vital_types_with_NO_surface_blocker": n_no_blocker,
        "vital_types_with_no_blocker": [t for t in vital_types if per_type[t]["n_candidate_blockers"] == 0],
    }
    return report, vital_types


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r30 = _load("r30", "30_hla_ifn_inducibility.py")
    d5 = _load("d5", "25_logicgate_data_rung5.py")
    log, HB = r30.log, r30.HB
    HB.start()

    genes_full, src = d5.get_surfaceome()
    if not (d5.TUMOUR_CACHE and d5.TUMOUR_CACHE.exists()):
        log("ERROR: RUNG-5 tumour cache not found — set LOGICGATE_CACHE to your rung5_cache.npz path "
            "(same account as RUNG-5). It holds tumour coverage of all 5007 surface genes.")
        return 2
    tumour = d5._loadp(d5.TUMOUR_CACHE)
    tgenes = list(tumour.genes)
    tpos = tumour.counts >= K
    tum_cov = {g: float(tpos[:, j].mean()) for j, g in enumerate(tgenes)}
    tumour_absent = [g for g in genes_full if tum_cov.get(g, 1.0) < TUM_FLOOR]
    log(f"surfaceome {len(genes_full)}; tumour-absent (<{TUM_FLOOR}) = {len(tumour_absent)} -> fetching their "
        f"vital-normal expression (the only candidate AND-NOT blockers)")

    # fetch vital-normal expression of the tumour-absent genes only (reuse RUNG-9's tested vital fetch)
    r30.GENES = list(tumour_absent)
    r30.IDX = {g: i for i, g in enumerate(tumour_absent)}
    import cellxgene_census
    tissues = d5.d4.NORMAL_TISSUES
    tile_dir = CACHE if CACHE else (OUT_DIR / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"cache/tiles -> {tile_dir} (resumable; {len(tumour_absent)} genes x vital cells)")
    census = None
    A, L = [], []
    for ti, tissue in enumerate(tissues):
        tile = tile_dir / f"rung10b_{tissue.replace(' ', '_')}.npz"
        if tile.exists():
            d = np.load(tile, allow_pickle=True)
            log(f"[{ti+1}/{len(tissues)}] {tissue}: RESUMED from tile ({d['counts'].shape[0]:,} cells)")
            A.append(d["counts"]); L.append(d["label"]); continue
        if census is None:
            HB.set(f"opening CELLxGENE Census {d5.d4.CENSUS_VERSION} ...")
            census = cellxgene_census.open_soma(census_version=d5.d4.CENSUS_VERSION)
        res = r30.pull_vital_genes(census, d5.d4, tissue, ti, len(tissues))
        if res is None:
            continue
        np.savez_compressed(tile, counts=res["counts"], label=res["label"])
        log(f"[{ti+1}/{len(tissues)}] {tissue}: tile checkpointed -> {tile.name} (safe to disconnect)")
        A.append(res["counts"]); L.append(res["label"])

    HB.set("all tissues fetched — screening for tumour-absent + vital-high surface blockers ...")
    counts = np.vstack(A) if A else np.zeros((0, len(tumour_absent)), np.int32)
    label = np.concatenate(L) if L else np.array([], object)
    report, vtypes = find_candidates(counts, label, tumour_absent)

    no_blocker = report["n_vital_types_with_NO_surface_blocker"]
    decisive = ("DECISIVE NEGATIVE: vital types with NO possible surface blocker -> a surface-only AND-NOT gate "
                f"CANNOT spare them; the NOT-slot MUST be genetic for: {report['vital_types_with_no_blocker']}") \
        if no_blocker else \
        ("CANDIDATE BLOCKERS FOUND for every vital type -> a surface-only gate is not ruled out here; next: GPU "
         "sweep positives x these candidate blockers (RUNG-10c) to see if any gate is actually SELECTIVE.")

    result = {
        "tag": "rung10b_surface_blocker_discovery",
        "question": "Does a surface gene exist that is TUMOUR-ABSENT but VITAL-HIGH (the only possible good "
                    "AND-NOT blocker)? Asked of the atlas directly, per vital cell type.",
        "census_version": d5.d4.CENSUS_VERSION, "surfaceome_source": src,
        "tum_floor": TUM_FLOOR, "vital_high": VITAL_HIGH, "n_cells_total": int(counts.shape[0]),
        "DECISIVE": decisive, **report,
        "CEILING": "mRNA != surface protein; dropout DEFLATES vital-high so this is CONSERVATIVE for finding a "
                   "blocker (a 'no blocker' result is therefore strong; a 'blocker found' still needs protein "
                   "confirmation). Tumour coverage from cached RUNG-5 tumour panel (full 5007 surfaceome).",
        "INTERPRETATION": "Replaces speculation with the atlas's own answer. If even ONE vital tissue has no "
                          "tumour-absent vital-high surface gene, no surface AND-NOT gate can protect it -> the "
                          "genetic NOT-slot (HLA-LOH) is not a choice but a necessity. If candidates exist, they "
                          "are the exact surface blockers to put through the GPU gate sweep next.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"DECISIVE: {decisive}")
    for t in vtypes:
        v = report["per_vital_type"][t]
        log(f"  {t:22} n={v['n_cells']:>7,}  candidate surface blockers={v['n_candidate_blockers']:>4}  "
            f"e.g. {v['top_blockers'][:4]}")
    log(f"  universal (high in ALL vital): {len(report['universal_blockers_high_in_ALL_vital'])} -> "
        f"{report['universal_blockers_high_in_ALL_vital'][:8]}")
    HB.stop()
    _make_figure(report, vtypes)
    return 0


def _make_figure(report, vtypes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung10b] matplotlib unavailable ({e})"); return
    if not vtypes:
        return
    names = vtypes[::-1]
    vals = [report["per_vital_type"][t]["n_candidate_blockers"] for t in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names) + 1.5)))
    ax.barh(y, vals, color=["#C1432B" if v == 0 else "#4C9F70" for v in vals])
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("# tumour-absent surface genes that are vital-high (candidate AND-NOT blockers)")
    ax.set_title("RUNG-10b: do surface AND-NOT blockers exist per vital tissue?\n"
                 "red = ZERO blockers -> surface-only gate impossible there (NOT-slot must be genetic)", fontsize=10)
    ax.axvline(0.5, ls="--", color="grey")
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung10b] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # tumour-absent gene list (already filtered to tumour-absent); test the vital-high screen
    genes = ["BLOCK_CARD", "BLOCK_ALL", "ABSENT_EVERYWHERE"]
    rng = np.random.default_rng(11)
    rows, lab = [], []
    def add(t, n, vec):
        for _ in range(n):
            rows.append([int(rng.integers(3, 20)) if v else 0 for v in vec]); lab.append(t)
    # BLOCK_CARD high only in cardiomyocyte; BLOCK_ALL high in both vital types; ABSENT high nowhere
    add("cardiomyocyte", 100, [1, 1, 0])
    add("neuron", 100, [0, 1, 0])
    counts = np.array(rows, np.int32); label = np.array(lab, object)
    rep, vt = find_candidates(counts, label, genes)

    check("cardiomyocyte blockers include BLOCK_CARD and BLOCK_ALL",
          set(rep["per_vital_type"]["cardiomyocyte"]["top_blockers"]) == {"BLOCK_CARD", "BLOCK_ALL"})
    check("neuron blockers = only BLOCK_ALL (BLOCK_CARD absent there)",
          rep["per_vital_type"]["neuron"]["top_blockers"] == ["BLOCK_ALL"])
    check("ABSENT_EVERYWHERE is never a blocker",
          all("ABSENT_EVERYWHERE" not in rep["per_vital_type"][t]["top_blockers"] for t in vt))
    check("universal blocker = BLOCK_ALL only", rep["universal_blockers_high_in_ALL_vital"] == ["BLOCK_ALL"])
    check("no vital type with zero blockers in this synthetic", rep["n_vital_types_with_NO_surface_blocker"] == 0)

    # a vital type with NO blocker -> flagged
    rows2, lab2 = [], []
    for _ in range(100):
        rows2.append([0, 0, 0]); lab2.append("kidney_tubule")        # nothing detected -> no blocker
    rep2, _ = find_candidates(np.array(rows2, np.int32), np.array(lab2, object), genes)
    check("vital type with all-zero -> flagged as NO blocker", rep2["n_vital_types_with_NO_surface_blocker"] == 1)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
