#!/usr/bin/env python3
"""
RUNG 12P / Part A — the death-wave CONTAINMENT screen (propagation-arm, CPU-only, atlas).

THE IDEA UNDER TEST (Anshuman's): instead of every therapeutic agent independently recognising every tumour
cell (the per-cell recognition wall R5-R11 mapped), engineer ONE cancer cell to recognise it is cancer, kill
itself, and PROPAGATE the apoptosis signal to its neighbours -> a death wave that spreads cell-to-cell
through the tumour. Recognition then only has to succeed at a few SEED cells; the wave does the rest. This
DECOUPLES killing from per-cell recognition and could bypass the addressability ceiling entirely -- IF the
wave is contained to the tumour.

THE MAKE-OR-BREAK QUESTION (asked of the atlas, not guessed):
A PASSIVE death wave travels through GAP JUNCTIONS (connexins) -- the established bystander-effect channel
(HSV-TK/ganciclovir). It is contained only if a connexin exists that is expressed in TUMOUR (wave propagates)
but ABSENT in vital normal tissue (wave can't enter heart/liver/brain). So:

  Is there ANY connexin that is worst-donor VITAL-LOW across ALL vital cell types
  (a "vital-silent" coupling channel a wave could use without leaking)?

DECISIVE OUTCOME (honestly scoped):
  - If NO connexin is vital-silent everywhere -> every coupling channel leaks into some vital tissue ->
    a PASSIVE gap-junctional death wave CANNOT be contained -> propagation MUST be RECOGNITION-GATED per hop
    (a synNotch-style AND-gate at each step), not passive. (Expected: heart=Cx43-rich, liver=Cx32/Cx26-rich.)
  - If a vital-silent connexin EXISTS and is tumour-expressed -> a candidate passive containable channel
    (surprise +) -> route to Part B percolation sim to see if a wave on it clears the tumour.

THE RUNG-8 TRAP, HANDLED: CELLxGENE returns 0 for BOTH "measured & zero" AND "gene not measured in this
dataset". For RUNG-10b we hunted HIGH, so dropout DEFLATED -> conservative. HERE we hunt LOW (vital-silent),
where dropout/unmeasured would FAKE a containable channel (anti-conservative, the exact RUNG-8 v1 artifact).
FIX: a HOUSEKEEPING DEPTH CONTROL -- a connexin counts as 'off' in a cell ONLY among DEEP cells (>=1 HK gene
detected). A shallow cell (no HK) can't vote 'silent'. Worst-donor over deep cells.

CEILING: mRNA != gap-junction protein/coupling (a connexin transcript != a functional channel); HK-deep
filter mitigates but doesn't erase dropout; tumour connexin from cached RUNG-5 surfaceome panel where present.

USAGE
  python scripts/34_connexin_containment.py selftest
  RUNG12P_CACHE=/content/drive/MyDrive/cancer-recon/rung12p_tiles \
  LOGICGATE_CACHE=/content/drive/MyDrive/cancer-recon/rung5_cache.npz \
      python scripts/34_connexin_containment.py run
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
OUT_DIR = PROJECT_ROOT / "runs" / "rung12p_connexin"
RESULT_JSON = OUT_DIR / "rung12p_connexin.json"
FIGURE_PNG = OUT_DIR / "rung12p_connexin.png"
CACHE = Path(os.environ["RUNG12P_CACHE"]) if os.environ.get("RUNG12P_CACHE") else None

K = 2                                                          # UMI count >= K -> gene "detected" in a cell
MIN_DEEP = int(os.environ.get("R12_MIN_DEEP", "30"))          # a donor needs >= this many DEEP cells to be powered
LEAK_FLOOR = float(os.environ.get("R12_LEAK", "0.10"))       # connexin "leaks into" a vital type if pooled-deep > this
SILENT_FLOOR = float(os.environ.get("R12_SILENT", "0.05"))   # "vital-silent" if pooled-deep < this in ALL vital types
TUM_HIGH = float(os.environ.get("R12_TUM_HIGH", "0.10"))     # connexin "tumour-expressed" if tumour coverage > this

# connexins (GJ*) + pannexins (PANX*) -- the gap-junction / channel genes a passive death wave could travel.
CONNEXINS = ["GJA1", "GJA3", "GJA4", "GJA5", "GJA8", "GJA9", "GJA10",
             "GJB1", "GJB2", "GJB3", "GJB4", "GJB5", "GJB6", "GJB7",
             "GJC1", "GJC2", "GJC3", "GJD2", "GJD3", "GJD4",
             "PANX1", "PANX2", "PANX3"]
# housekeeping DEPTH CONTROL: a cell is "deep" (can vote 'connexin off') only if >=1 of these is detected.
HK_PANEL = ["ACTB", "GAPDH", "RPL13A", "MALAT1"]
COMMON = {"GJA1": "Cx43", "GJB1": "Cx32", "GJB2": "Cx26", "GJC1": "Cx45", "GJA5": "Cx40",
          "GJB6": "Cx30", "GJD2": "Cx36", "GJC2": "Cx47", "GJA4": "Cx37", "GJA3": "Cx46"}


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------------------
#  Depth-controlled accumulator (pure -> selftest exercises it directly).
#  Per (vital-type, donor): n_cells, n_deep (>=1 HK detected), and per-connexin detected-count AMONG DEEP cells.
# ---------------------------------------------------------------------------
def accumulate_tile(conn_counts, hk_counts, label, donor, acc):
    """conn_counts (n, Gc), hk_counts (n, Ghk). acc: key -> {n, n_deep, det_deep(Gc int64)}."""
    Gc = conn_counts.shape[1]
    deep = (hk_counts >= K).any(axis=1)                        # cell deep iff any HK gene detected
    cpos = conn_counts >= K
    for t in np.unique(label):
        if t is None:
            continue
        mt = label == t
        for d in np.unique(donor[mt]):
            m = mt & (donor == d)
            key = (str(t), str(d))
            rec = acc.get(key)
            if rec is None:
                rec = {"n": 0, "n_deep": 0, "det_deep": np.zeros(Gc, np.int64)}
                acc[key] = rec
            rec["n"] += int(m.sum())
            dm = m & deep
            rec["n_deep"] += int(dm.sum())
            if dm.any():
                rec["det_deep"] += cpos[dm].sum(axis=0).astype(np.int64)


def find_leak_channels(acc, conn_genes):
    """Per connexin x vital type: pooled & worst-donor detection AMONG DEEP cells. A connexin 'leaks into' a
    vital type if pooled-deep > LEAK_FLOOR; is 'vital-silent' if pooled-deep < SILENT_FLOOR in ALL types.
    Returns (report, vital_types)."""
    Gc = len(conn_genes)
    by_type = defaultdict(list)
    for (t, d), rec in acc.items():
        if rec["n_deep"] >= MIN_DEEP:
            by_type[t].append(rec)
    vital_types = sorted(by_type)
    per_conn, leak_map, silent_everywhere = {}, {}, []
    for gi, g in enumerate(conn_genes):
        type_fr, leaks = {}, []
        all_silent = bool(vital_types)
        for t in vital_types:
            donors = by_type[t]
            tot_deep = sum(r["n_deep"] for r in donors)
            tot_det = sum(int(r["det_deep"][gi]) for r in donors)
            pooled = (tot_det / tot_deep) if tot_deep else 0.0
            fracs = [int(r["det_deep"][gi]) / r["n_deep"] for r in donors]
            worst, top = min(fracs), max(fracs)               # worst (lowest) and top (highest) powered donor
            type_fr[t] = {"pooled_deep": round(pooled, 4), "worst_donor": round(worst, 4),
                          "top_donor": round(top, 4), "n_deep": int(tot_deep), "n_donors": len(donors)}
            # SAFETY uses the WORST-CASE (highest-expressing) donor: a channel expressed in even one patient's
            # vital tissue is a leak risk for that patient (never-pooled ethos). Conservative for containment.
            if top > LEAK_FLOOR:
                leaks.append(t)
            if top >= SILENT_FLOOR:
                all_silent = False
        per_conn[g] = {"common": COMMON.get(g, g), "by_vital_type": type_fr, "leaks_into": leaks}
        leak_map[g] = leaks
        if all_silent:
            silent_everywhere.append(g)
    report = {"n_connexins_screened": Gc, "n_vital_types": len(vital_types), "vital_types": vital_types,
              "per_connexin": per_conn, "leak_map": leak_map,
              "vital_silent_connexins_ALL_types": silent_everywhere,
              "n_vital_silent_connexins": len(silent_everywhere)}
    return report, vital_types


# ---------------------------------------------------------------------------
#  OOM-safe Census fetch (mirrors RUNG-10b: scout vital joinids, cell-chunked materialise, accumulate, discard).
# ---------------------------------------------------------------------------
def _fetch_chunked(census, d4, tissue, ti, ntis, genes, ci_conn, ci_hk, r30, r32, log, HB):
    import cellxgene_census
    HB.set(f"[{ti+1}/{ntis}] {tissue}: scouting vital cells ...")
    joinids = r32._scout_vital_joinids(census, d4, tissue, r30)
    acc = {}
    if len(joinids) == 0:
        log(f"[{ti+1}/{ntis}] {tissue}: no vital cells"); return acc, 0
    Gc, Ghk = len(ci_conn), len(ci_hk)
    CH = int(os.environ.get("R12_CHUNK", "40000"))
    nch = (len(joinids) + CH - 1) // CH
    for c in range(nch):
        HB.set(f"[{ti+1}/{ntis}] {tissue}: chunk {c+1}/{nch} ({len(joinids):,} vital cells x {len(genes)} genes)")
        chunk = joinids[c * CH:(c + 1) * CH]
        ad = cellxgene_census.get_anndata(
            census, organism="Homo sapiens", obs_coords=chunk.tolist(),
            var_value_filter=f"feature_name in {genes}",
            column_names={"obs": ["cell_type", "donor_id", "dataset_id"]})
        vnames = list(ad.var["feature_name"]) if "feature_name" in ad.var else list(ad.var_names)
        X = ad.X
        Xd = np.asarray(X.todense() if hasattr(X, "todense") else X)
        conn = np.zeros((Xd.shape[0], Gc), np.int16)
        hk = np.zeros((Xd.shape[0], Ghk), np.int16)
        for j, g in enumerate(vnames):
            if g in ci_conn:
                conn[:, ci_conn[g]] = Xd[:, j].astype(np.int16)
            elif g in ci_hk:
                hk[:, ci_hk[g]] = Xd[:, j].astype(np.int16)
        del Xd
        labels = np.array([r30._vital_label(c2, d4.VITAL_AUDIT) for c2 in ad.obs["cell_type"].astype(str)], dtype=object)
        donors = np.array([f"{ds}::{dn}" for ds, dn in
                           zip(ad.obs["dataset_id"].astype(str), ad.obs["donor_id"].astype(str))], dtype=object)
        accumulate_tile(conn, hk, labels, donors, acc)
        del ad, conn, hk
    return acc, len(joinids)


def _save_acc(tile, acc, conn_genes):
    keys = list(acc.keys())
    np.savez_compressed(
        tile, keys=np.array([f"{t}\t{d}" for (t, d) in keys], dtype=object),
        n=np.array([acc[k]["n"] for k in keys], np.int64),
        n_deep=np.array([acc[k]["n_deep"] for k in keys], np.int64),
        det=np.array([acc[k]["det_deep"] for k in keys], np.int64) if keys else np.zeros((0, len(conn_genes)), np.int64))


def _load_acc_tile(tile):
    d = np.load(tile, allow_pickle=True)
    acc = {}
    for ks, n, nd, det in zip(d["keys"], d["n"], d["n_deep"], d["det"]):
        t, dn = str(ks).split("\t")
        acc[(t, dn)] = {"n": int(n), "n_deep": int(nd), "det_deep": np.asarray(det, np.int64)}
    return acc


def _merge_acc(into, src):
    for k, r in src.items():
        if k in into:
            into[k]["n"] += r["n"]; into[k]["n_deep"] += r["n_deep"]; into[k]["det_deep"] = into[k]["det_deep"] + r["det_deep"]
        else:
            into[k] = {"n": r["n"], "n_deep": r["n_deep"], "det_deep": np.asarray(r["det_deep"]).copy()}


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r30 = _load("r30", "30_hla_ifn_inducibility.py")
    r32 = _load("r32", "32_surface_blocker_discovery.py")
    d5 = _load("d5", "25_logicgate_data_rung5.py")
    log, HB = r30.log, r30.HB
    HB.start()

    genes = CONNEXINS + HK_PANEL
    ci_conn = {g: i for i, g in enumerate(CONNEXINS)}
    ci_hk = {g: i for i, g in enumerate(HK_PANEL)}

    # tumour-side connexin coverage from the cached RUNG-5 surfaceome tumour panel (where connexins are present)
    tum_cov = {}
    if d5.TUMOUR_CACHE and d5.TUMOUR_CACHE.exists():
        tumour = d5._loadp(d5.TUMOUR_CACHE)
        tpos = tumour.counts >= K
        tcov = {g: float(tpos[:, j].mean()) for j, g in enumerate(tumour.genes)}
        tum_cov = {g: tcov.get(g) for g in CONNEXINS if g in tcov}
        log(f"tumour connexin coverage from RUNG-5 cache: {sum(v is not None for v in tum_cov.values())}/"
            f"{len(CONNEXINS)} connexins present in the surfaceome panel")
    else:
        log("WARN: no RUNG-5 tumour cache (LOGICGATE_CACHE) — tumour-expression side skipped; vital screen still decisive")

    import cellxgene_census
    tissues = d5.d4.NORMAL_TISSUES
    tile_dir = CACHE if CACHE else (OUT_DIR / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"connexin containment screen: {len(CONNEXINS)} connexins + {len(HK_PANEL)} HK depth-control genes; "
        f"tiles -> {tile_dir} (resumable, HK-deep filter, worst-donor)")

    acc = {}
    census = None
    for ti, tissue in enumerate(tissues):
        tile = tile_dir / f"rung12p_acc_{tissue.replace(' ', '_')}.npz"
        if tile.exists():
            tc = _load_acc_tile(tile)
            log(f"[{ti+1}/{len(tissues)}] {tissue}: RESUMED acc-tile ({len(tc)} (type,donor) groups)")
            _merge_acc(acc, tc); continue
        if census is None:
            HB.set(f"opening CELLxGENE Census {d5.d4.CENSUS_VERSION} ...")
            census = cellxgene_census.open_soma(census_version=d5.d4.CENSUS_VERSION)
        tc, n = _fetch_chunked(census, d5.d4, tissue, ti, len(tissues), genes, ci_conn, ci_hk, r30, r32, log, HB)
        if n == 0:
            continue
        _save_acc(tile, tc, CONNEXINS)
        log(f"[{ti+1}/{len(tissues)}] {tissue}: acc-tile checkpointed -> {tile.name} ({n:,} cells, safe to disconnect)")
        _merge_acc(acc, tc)

    HB.set("all tissues streamed — worst-donor leak/containment screen ...")
    report, vtypes = find_leak_channels(acc, CONNEXINS)

    # cross with tumour: which connexins are tumour-expressed, and (for those) which vital tissues they leak into
    tumour_expressed = sorted([g for g, v in tum_cov.items() if v is not None and v > TUM_HIGH],
                              key=lambda g: -tum_cov[g])
    usable_passive = [g for g in report["vital_silent_connexins_ALL_types"] if tum_cov.get(g, 0) and tum_cov[g] > TUM_HIGH]
    n_silent = report["n_vital_silent_connexins"]

    if n_silent == 0:
        decisive = (f"DECISIVE NEGATIVE: NO connexin is vital-silent across all {report['n_vital_types']} vital "
                    f"cell types — every coupling channel leaks into some vital tissue (e.g. tumour-expressed "
                    f"{[f'{g}/{COMMON.get(g,g)}->{report['leak_map'][g]}' for g in tumour_expressed[:3]]}). "
                    f"A PASSIVE gap-junctional death wave CANNOT be contained -> propagation MUST be "
                    f"RECOGNITION-GATED per hop (synNotch-style AND-gate), not passive. (Scope: connexin/pannexin "
                    f"channels, mRNA-level; doesn't rule out non-junctional gated relays — that's Part B.)")
    elif usable_passive:
        decisive = (f"SURPRISE +: vital-silent AND tumour-expressed connexin(s) {usable_passive} — a candidate "
                    f"PASSIVE containable channel. Route to Part B percolation sim to test if a wave on it clears "
                    f"the tumour. Verify against protein/coupling (mRNA != functional channel).")
    else:
        decisive = (f"PARTIAL: {n_silent} vital-silent connexin(s) {report['vital_silent_connexins_ALL_types']} "
                    f"exist but none are tumour-expressed (>{TUM_HIGH}) — no USABLE passive channel; gated relay "
                    f"still required. (Tumour side from cached surfaceome; connexins absent there are unknown.)")

    result = {
        "tag": "rung12p_connexin_containment",
        "idea": "Anshuman's propagation arm: seed-and-spread apoptosis decouples killing from per-cell "
                "recognition. This Part-A run asks whether a PASSIVE (gap-junctional) death wave can be "
                "contained to the tumour, or whether the relay must be recognition-gated per hop.",
        "question": "Is there a connexin worst-donor VITAL-LOW across ALL vital cell types (a containable "
                    "channel) that is also TUMOUR-expressed (usable)?",
        "census_version": d5.d4.CENSUS_VERSION,
        "K": K, "min_deep_cells": MIN_DEEP, "leak_floor": LEAK_FLOOR, "silent_floor": SILENT_FLOOR,
        "tum_high": TUM_HIGH, "hk_depth_control": HK_PANEL,
        "tumour_connexin_coverage": {g: (round(v, 4) if v is not None else None) for g, v in tum_cov.items()},
        "tumour_expressed_connexins": tumour_expressed,
        "usable_passive_channels": usable_passive,
        "DECISIVE": decisive, **report,
        "CEILING": "mRNA != functional gap-junction coupling (connexin transcript != open channel); HK-deep "
                   "filter mitigates but doesn't erase dropout; LOW calls are the anti-conservative direction so "
                   "the depth control is load-bearing; tumour side from cached RUNG-5 surfaceome (connexins not "
                   "in that panel are 'unknown', not 'absent'). SCOPE: passive connexin/pannexin channels.",
        "INTERPRETATION": "If no connexin is vital-silent everywhere, the bystander/gap-junction route can't be "
                          "tumour-contained -> Anshuman's wave must re-check tumour identity at each hop (gated "
                          "relay). That gated relay is exactly Part B: does per-hop gating with a LEAKIER "
                          "recognition than R5 required still clear the tumour without a normal-tissue death wave "
                          "(errors don't cascade)?",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"DECISIVE: {decisive}")
    log(f"  tumour-expressed connexins (>{TUM_HIGH}): {[(g, COMMON.get(g, g), round(tum_cov[g],3)) for g in tumour_expressed]}")
    for g in CONNEXINS:
        pc = report["per_connexin"][g]
        if pc["leaks_into"] or (tum_cov.get(g) or 0) > TUM_HIGH:
            log(f"  {g:5}/{COMMON.get(g,g):5} tumour={('%.2f'%tum_cov[g]) if tum_cov.get(g) is not None else '  ? '}"
                f"  leaks_into={pc['leaks_into']}")
    HB.stop()
    _make_figure(report, vtypes, tum_cov)
    return 0


def _make_figure(report, vtypes, tum_cov):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung12p] matplotlib unavailable ({e})"); return
    if not vtypes:
        return
    genes = [g for g in CONNEXINS if report["per_connexin"][g]["leaks_into"] or (tum_cov.get(g) or 0) > TUM_HIGH]
    if not genes:
        genes = CONNEXINS
    M = np.array([[report["per_connexin"][g]["by_vital_type"][t]["pooled_deep"] for t in vtypes] for g in genes])
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(vtypes) + 3), max(3, 0.42 * len(genes) + 2)))
    im = ax.imshow(M, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(vtypes))); ax.set_xticklabels(vtypes, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([f"{g}/{COMMON.get(g,g)}" + (f"  tum={tum_cov[g]:.2f}" if tum_cov.get(g) is not None else "")
                        for g in genes], fontsize=7)
    for i in range(len(genes)):
        for j in range(len(vtypes)):
            if M[i, j] > 0.10:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="connexin detection (pooled, among HK-deep cells)")
    ns = report["n_vital_silent_connexins"]
    ax.set_title(f"RUNG-12P/A: can a passive death wave be contained?\n"
                 f"connexin x vital tissue (red = expressed = LEAK channel). vital-silent-everywhere: {ns}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung12p] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    genes = ["CX_HEART", "CX_SILENT", "CX_ONEDONOR", "CX_LIVERONLY"]

    # --- accumulate_tile with HK depth control: shallow cells (no HK) must NOT count toward n_deep or det
    n = 100
    conn = np.zeros((n, 4), np.int32); hk = np.zeros((n, 1), np.int32)
    hk[:60, 0] = 5                                  # first 60 cells are DEEP; last 40 shallow
    conn[:60, 0] = 5                                # CX_HEART detected in all 60 deep cells
    conn[60:, 1] = 5                                # CX_SILENT detected ONLY in shallow cells -> must be ignored
    acc = {}
    accumulate_tile(conn, hk, np.array(["cardiomyocyte"] * n, object), np.array(["D1"] * n, object), acc)
    rec = acc[("cardiomyocyte", "D1")]
    check("HK-deep: n_deep counts only deep cells", rec["n_deep"] == 60 and rec["n"] == 100)
    check("HK-deep: CX_HEART detected in 60/60 deep", rec["det_deep"][0] == 60)
    check("HK-deep: CX_SILENT (only in shallow cells) NOT counted", rec["det_deep"][1] == 0)

    # --- find_leak_channels: build a controlled accumulator
    def rec_(n_deep, det):  # det = per-gene detected-among-deep counts
        return {"n": n_deep, "n_deep": n_deep, "det_deep": np.array(det, np.int64)}
    acc2 = {
        # cardiomyocyte: 2 donors. CX_HEART high in both; CX_ONEDONOR high in D1, ~0 in D2 (top-donor still high)
        ("cardiomyocyte", "D1"): rec_(100, [95, 0, 90, 0]),
        ("cardiomyocyte", "D2"): rec_(100, [92, 0, 1, 0]),
        # hepatocyte: CX_LIVERONLY high; others ~0
        ("hepatocyte", "D3"): rec_(100, [2, 0, 0, 88]),
        # neuron: all silent (CX_SILENT 0 everywhere)
        ("neuron", "D4"): rec_(100, [3, 0, 1, 0]),
        # under-powered donor (n_deep < MIN_DEEP) -> excluded entirely
        ("kidney_tubule", "D5"): rec_(10, [10, 10, 10, 10]),
    }
    rep, vt = find_leak_channels(acc2, genes)
    check("under-powered donor excluded from vital_types", "kidney_tubule" not in vt)
    check("CX_HEART leaks into cardiomyocyte", "cardiomyocyte" in rep["leak_map"]["CX_HEART"])
    check("CX_LIVERONLY leaks into hepatocyte only", rep["leak_map"]["CX_LIVERONLY"] == ["hepatocyte"])
    check("CX_ONEDONOR leaks (top-donor rule: one patient's heart expressing it IS a risk)",
          "cardiomyocyte" in rep["leak_map"]["CX_ONEDONOR"])
    check("CX_ONEDONOR is NOT vital-silent (top-donor high)",
          "CX_ONEDONOR" not in rep["vital_silent_connexins_ALL_types"])
    check("CX_SILENT is vital-silent across ALL types", "CX_SILENT" in rep["vital_silent_connexins_ALL_types"])
    check("CX_HEART is NOT vital-silent", "CX_HEART" not in rep["vital_silent_connexins_ALL_types"])
    # worst-donor (min) < top-donor (max) when one donor is low
    cm = rep["per_connexin"]["CX_ONEDONOR"]["by_vital_type"]["cardiomyocyte"]
    check("worst_donor < top_donor for split-donor connexin", cm["worst_donor"] < cm["top_donor"])

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
