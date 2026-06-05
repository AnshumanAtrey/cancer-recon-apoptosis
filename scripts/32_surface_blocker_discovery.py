#!/usr/bin/env python3
"""
RUNG 10b — does a usable surface AND-NOT blocker even EXIST? (decisive, CPU-only, no GPU).

arm (a) hit a wall: the cached panel only has TUMOUR-EXPRESSED genes, which make poor NOT-markers (they're in
the tumour too -> subtracting them kills coverage). The BEST-possible surface NOT-marker is TUMOUR-ABSENT but
VITAL-HIGH ON ITS WORST DONOR: ~negC is then TRUE in all tumour (keeps coverage) and reliably FALSE in vital
normal (spares it). This run stops speculating and ASKS THE ATLAS directly: among the ~4300 surface genes
absent in tumour, is ANY reliably high (worst-donor) in vital normal tissue? Per vital cell type.

DECISIVE OUTCOME (honestly scoped)
  - If NO tumour-absent surface gene is worst-donor-vital-high for a vital type -> NO single surface NOT-marker
    can spare it -> a SINGLE-BLOCKER 3-input AND-NOT surface gate is IMPOSSIBLE for it; the NOT-slot must be
    genetic. (This does NOT rule out panel-NOT / OR-gates / >=4-input surface logic — out of scope, stated.)
  - If candidates EXIST -> they are NECESSARY but NOT SUFFICIENT (a real gate must ALSO cover the non-vital-
    normal 'strict' leak and yield a SELECTIVE gate). They are the exact blockers for the GPU sweep next.

ENGINEERING (the three fixes the design review caught):
  * OOM-SAFE: never materialise the 700k x 4325 dense block (6 GB, peaks ~10-12 GB). We STREAM per-tissue tiles
    and accumulate per-(vital-type, donor) per-gene detection counts, discarding each tile. Peak << 1 panel.
  * WORST-DONOR: a blocker must be high on its WEAKEST powered donor (min over donors, >= MIN_DONOR_CELLS),
    matching the project's never-pooled safety ethos — not a pooled mean that one bad donor could fake.
  * HONEST SCOPE: the claim is about single-blocker 3-input AND-NOT, not 'surface recognition'.

CEILING: mRNA != surface protein; dropout DEFLATES vital-high -> CONSERVATIVE for finding a blocker (a 'no
blocker' result is strong; a 'blocker found' still needs the strict-leak check + protein confirmation).

USAGE
  python scripts/32_surface_blocker_discovery.py selftest
  RUNG10B_CACHE=/content/drive/MyDrive/cancer-recon/rung10b_tiles \
  LOGICGATE_CACHE=/content/drive/MyDrive/cancer-recon/rung5_cache.npz \
      python scripts/32_surface_blocker_discovery.py run
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung10b_blocker"
RESULT_JSON = OUT_DIR / "rung10b_blocker.json"
FIGURE_PNG = OUT_DIR / "rung10b_blocker.png"
CACHE = Path(os.environ["RUNG10B_CACHE"]) if os.environ.get("RUNG10B_CACHE") else None
TUM_FLOOR = float(os.environ.get("R10_TUM_FLOOR", "0.02"))     # 'tumour-absent' if tumour coverage < this
VITAL_HIGH = float(os.environ.get("R10_VITAL_HIGH", "0.5"))    # 'vital-high' if WORST-donor detection > this
MIN_DONOR_CELLS = 30                                           # a donor needs >= this many cells to be powered
K = 2


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------------------
def accumulate_tile(counts, label, donor, acc_cells, acc_det):
    """Stream one tile into per-(vital-type, donor) accumulators (OOM-safe: tile is discarded after)."""
    pos = counts >= K                                          # (n, G) bool
    for t in np.unique(label):
        if t is None:
            continue
        mt = label == t
        for d in np.unique(donor[mt]):
            m = mt & (donor == d)
            key = (str(t), str(d))
            acc_cells[key] += int(m.sum())
            det = pos[m].sum(axis=0).astype(np.int64)         # per-gene detected count for this (type,donor)
            acc_det[key] = acc_det.get(key) + det if key in acc_det else det


def find_candidates(acc_cells, acc_det, genes):
    """Worst-donor screen (pure/testable): per vital type, tumour-absent genes whose WEAKEST powered donor
    still detects them in > VITAL_HIGH of cells -> reliable blockers. Returns (report, vital_types)."""
    G = len(genes)
    by_type = defaultdict(list)                                # vital type -> [(donor, n, det_array)]
    for (t, d), n in acc_cells.items():
        if n >= MIN_DONOR_CELLS:
            by_type[t].append((d, n, acc_det[(t, d)]))
    vital_types = sorted(by_type)
    per_type = {}
    high_mat = []
    for t in vital_types:
        donors = by_type[t]
        frac = np.array([det / n for (_, n, det) in donors])  # (n_donors, G) per-donor detection fraction
        worst = frac.min(axis=0)                              # worst-donor detection per gene
        hi = worst > VITAL_HIGH
        high_mat.append(hi)
        idx = np.where(hi)[0]
        top = [genes[j] for j in idx[np.argsort(-worst[idx])][:25]]
        per_type[t] = {"n_powered_donors": len(donors), "n_cells": int(sum(n for _, n, _ in donors)),
                       "n_candidate_blockers": int(hi.sum()), "top_blockers": top}
    universal = ([genes[j] for j in np.where(np.array(high_mat).all(axis=0))[0]]
                 if high_mat else [])
    no_blocker = [t for t in vital_types if per_type[t]["n_candidate_blockers"] == 0]
    report = {"n_tumour_absent_genes_screened": G, "per_vital_type": per_type,
              "universal_blockers_high_in_ALL_vital": universal,
              "n_vital_types_with_NO_surface_blocker": len(no_blocker),
              "vital_types_with_no_blocker": no_blocker}
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
        log("ERROR: RUNG-5 tumour cache not found — set LOGICGATE_CACHE to your rung5_cache.npz (same account "
            "as RUNG-5). It holds tumour coverage of all 5007 surface genes.")
        return 2
    tumour = d5._loadp(d5.TUMOUR_CACHE)
    tpos = tumour.counts >= K
    tum_cov = {g: float(tpos[:, j].mean()) for j, g in enumerate(tumour.genes)}
    tumour_absent = [g for g in genes_full if tum_cov.get(g, 1.0) < TUM_FLOOR]
    log(f"surfaceome {len(genes_full)}; tumour-absent (<{TUM_FLOOR}) = {len(tumour_absent)} -> streaming their "
        f"vital-normal expression (the only candidate single-blocker NOT-markers)")

    r30.GENES = list(tumour_absent)
    r30.IDX = {g: i for i, g in enumerate(tumour_absent)}
    import cellxgene_census
    tissues = d5.d4.NORMAL_TISSUES
    tile_dir = CACHE if CACHE else (OUT_DIR / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"cache/tiles -> {tile_dir} (resumable; {len(tumour_absent)} genes x vital cells, STREAMED)")

    acc_cells = defaultdict(int)
    acc_det = {}
    census = None
    for ti, tissue in enumerate(tissues):
        tile = tile_dir / f"rung10b_{tissue.replace(' ', '_')}.npz"
        if tile.exists():
            d = np.load(tile, allow_pickle=True)
            log(f"[{ti+1}/{len(tissues)}] {tissue}: RESUMED tile ({d['counts'].shape[0]:,} cells) -> accumulate")
            accumulate_tile(d["counts"], d["label"], d["donor"], acc_cells, acc_det)
            del d
            continue
        if census is None:
            HB.set(f"opening CELLxGENE Census {d5.d4.CENSUS_VERSION} ...")
            census = cellxgene_census.open_soma(census_version=d5.d4.CENSUS_VERSION)
        res = r30.pull_vital_genes(census, d5.d4, tissue, ti, len(tissues))
        if res is None:
            continue
        np.savez_compressed(tile, counts=res["counts"], label=res["label"], donor=res["donor"])
        log(f"[{ti+1}/{len(tissues)}] {tissue}: tile checkpointed -> {tile.name} (safe to disconnect)")
        accumulate_tile(res["counts"], res["label"], res["donor"], acc_cells, acc_det)
        del res

    HB.set("all tissues streamed — worst-donor screen for tumour-absent + vital-high blockers ...")
    report, vtypes = find_candidates(acc_cells, acc_det, tumour_absent)

    nb = report["n_vital_types_with_NO_surface_blocker"]
    decisive = (f"DECISIVE NEGATIVE: NO tumour-absent surface gene is worst-donor vital-high for "
                f"{report['vital_types_with_no_blocker']} -> a SINGLE-BLOCKER 3-input AND-NOT surface gate "
                f"CANNOT spare them -> the NOT-slot must be genetic for those tissues. (Out of scope: panel-NOT "
                f"/ OR-gates / >=4-input surface logic.)") if nb else \
        ("Candidate single-blockers exist for every vital type — NECESSARY but NOT SUFFICIENT (they must also "
         "cover the non-vital-normal 'strict' leak and yield a SELECTIVE gate). Next: GPU sweep positives x "
         "these blockers (RUNG-10c).")

    result = {
        "tag": "rung10b_surface_blocker_discovery",
        "question": "Does a surface gene exist that is TUMOUR-ABSENT but reliably (worst-donor) VITAL-HIGH — the "
                    "only possible good single AND-NOT blocker? Asked of the atlas, per vital cell type.",
        "census_version": d5.d4.CENSUS_VERSION, "surfaceome_source": src,
        "tum_floor": TUM_FLOOR, "vital_high_worstdonor": VITAL_HIGH, "min_donor_cells": MIN_DONOR_CELLS,
        "DECISIVE": decisive, **report,
        "CEILING": "mRNA != surface protein; dropout DEFLATES worst-donor vital-high -> CONSERVATIVE for finding "
                   "a blocker (a 'no blocker' result is strong; 'blocker found' still needs the strict non-vital-"
                   "normal leak check + protein confirmation). Tumour coverage from cached RUNG-5 tumour panel "
                   "(full 5007 surfaceome). SCOPE: single-blocker 3-input AND-NOT only.",
        "INTERPRETATION": "Replaces speculation with the atlas's own answer. If even ONE vital tissue has no "
                          "tumour-absent worst-donor-vital-high surface gene, no single-blocker surface AND-NOT "
                          "gate can protect it -> the genetic NOT-slot is a necessity, not a choice (for this "
                          "gate class). If candidates exist, they are the exact blockers for the GPU gate sweep.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"DECISIVE: {decisive}")
    for t in vtypes:
        v = report["per_vital_type"][t]
        log(f"  {t:22} n={v['n_cells']:>7,} donors={v['n_powered_donors']:>3}  blockers={v['n_candidate_blockers']:>4}"
            f"  e.g. {v['top_blockers'][:4]}")
    log(f"  universal (worst-donor-high in ALL vital): {len(report['universal_blockers_high_in_ALL_vital'])} -> "
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
    ax.set_xlabel("# tumour-absent surface genes that are WORST-DONOR vital-high (candidate single blockers)")
    ax.set_title("RUNG-10b: do surface AND-NOT blockers exist per vital tissue?\n"
                 "red = ZERO -> single-blocker surface gate impossible there (NOT-slot must be genetic)", fontsize=10)
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

    genes = ["BLOCK_ALL", "BLOCK_CARD", "WEAK_ONE_DONOR", "ABSENT"]
    # build per-(type,donor) accumulators directly (counts of detected cells per gene)
    acc_cells, acc_det = {}, {}
    def grp(t, d, n, det):  # det = list of detected-cell counts per gene (<= n)
        acc_cells[(t, d)] = n; acc_det[(t, d)] = np.array(det, np.int64)
    # cardiomyocyte: 2 donors. BLOCK_ALL & BLOCK_CARD high in both; WEAK_ONE_DONOR high in D1, ZERO in D2
    grp("cardiomyocyte", "D1", 100, [90, 90, 90, 0])
    grp("cardiomyocyte", "D2", 100, [88, 85, 0, 0])     # WEAK_ONE_DONOR detected in 0/100 here -> worst-donor 0
    # neuron: 1 donor, BLOCK_ALL high, BLOCK_CARD absent
    grp("neuron", "D3", 100, [80, 0, 0, 0])
    # kidney_tubule: only an under-powered donor (10 cells) -> excluded -> NO powered donor -> flagged no-blocker
    grp("kidney_tubule", "D4", 10, [10, 10, 10, 10])

    rep, vt = find_candidates(acc_cells, acc_det, genes)
    check("cardiomyocyte blockers = BLOCK_ALL,BLOCK_CARD (WEAK_ONE_DONOR fails worst-donor)",
          set(rep["per_vital_type"]["cardiomyocyte"]["top_blockers"]) == {"BLOCK_ALL", "BLOCK_CARD"})
    check("WEAK_ONE_DONOR rejected (high in D1 but 0 in D2 -> worst-donor low)",
          "WEAK_ONE_DONOR" not in rep["per_vital_type"]["cardiomyocyte"]["top_blockers"])
    check("neuron blockers = only BLOCK_ALL", rep["per_vital_type"]["neuron"]["top_blockers"] == ["BLOCK_ALL"])
    check("ABSENT never a blocker", all("ABSENT" not in rep["per_vital_type"][t]["top_blockers"] for t in vt))
    check("universal (all powered vital) = BLOCK_ALL", rep["universal_blockers_high_in_ALL_vital"] == ["BLOCK_ALL"])
    check("under-powered kidney_tubule donor excluded -> not in vital_types", "kidney_tubule" not in vt)

    # all-zero vital type -> flagged as NO blocker
    rep2, _ = find_candidates({("neuron", "Dx"): 100}, {("neuron", "Dx"): np.zeros(4, np.int64)}, genes)
    check("vital type with all-zero -> NO blocker flagged", rep2["n_vital_types_with_NO_surface_blocker"] == 1)

    # accumulate_tile streams correctly (matches a direct per-group count)
    counts = np.array([[5, 0, 0, 0]] * 40 + [[0, 0, 0, 0]] * 60, np.int32)  # BLOCK_ALL detected in 40/100
    lab = np.array(["neuron"] * 100, object); dn = np.array(["D9"] * 100, object)
    ac, ad = defaultdict(int), {}
    accumulate_tile(counts, lab, dn, ac, ad)
    check("accumulate_tile counts detected cells per gene", ad[("neuron", "D9")][0] == 40 and ac[("neuron", "D9")] == 100)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
