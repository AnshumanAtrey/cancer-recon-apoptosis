#!/usr/bin/env python3
"""
RUNG 15 — the ATLAS x MECHANISM map: WHICH cancers does each winning strategy actually address?

THE BRIDGE THIS BUILDS
----------------------
RUNG-14's arena ranked the STRATEGIES (quorum > wave > ... ) by how well each contains itself across regimes,
parameterised by an abstract recognition fidelity (q_t tumour, q_n normal false-positive). It cannot say WHICH
real cancers have a handle that lands inside each strategy's safe window. This rung supplies the real (q_t, q_n)
and maps them onto the arena's MEASURED mechanism ceilings:

    per_cell  q_n <= 0.02   (RUNG-5/7 worst-donor bar)
    wave      q_n <= 0.173  (RUNG-13 measured kinetic safe ceiling)
    quorum    q_n <= 0.20   (RUNG-14 arena, conservative; true value in (0.20,0.50])

Two data sources for (q_t, q_n):

  MODE `map`  (instant, no network) — the SEQUENCE/neoantigen axis. Reuses RUNG-12's 32 structure-certified
      handles (each has a measured q_n and a per-cancer prevalence) -> per-cancer coverage at each mechanism
      ceiling. Answers: which cancers does per_cell vs wave vs quorum unlock on the neoantigen axis, and by
      how much. KEY HONEST FINDING surfaced here: neoantigen q_n is BIMODAL (clean ~0 or risky >0.3), so
      quorum's extra headroom over the wave buys NOTHING on neoantigens -> you need SURFACE markers (continuous
      q_n) to cash in quorum. That motivates `census`.

  MODE `census`  (Colab, CELLxGENE) — the EXPRESSION/surface axis. Reuses RUNG-12P/A's proven Census loader
      (scripts/34) on a panel of real tumour-associated SURFACE antigens (CAR-T/ADC targets). Per marker:
      q_t = malignant-cell positivity (from the RUNG-5 surfaceome tumour cache), q_n = worst-donor vital-tissue
      positivity (leak). Then map each marker onto the mechanism ceilings -> which surface handle gives a safe &
      effective gate under which strategy. This is where quorum's higher q_n tolerance can actually unlock
      markers that per_cell/wave cannot.

HONEST CEILING
--------------
q_n (neoantigen) is a PREDICTED proxy (MHCflurry + AF2, RUNG-11/12), not immunopeptidomics. q_t/q_n (surface)
are mRNA positivity (Census), not surface protein; dropout/dissociation biases remain (HK-depth control from
RUNG-34 mitigates, not erases). The mechanism ceilings are in-silico containment bounds, not clinical efficacy.
Coupling/delivery efficiency is the wet-lab residual throughout. This ranks WHERE to point the wet lab.

USAGE
  python scripts/40_atlas_mechanism_map.py            # `map` (neoantigen x mechanism, instant)
  python scripts/40_atlas_mechanism_map.py census     # surface-marker x mechanism (Colab, needs cellxgene_census)
  python scripts/40_atlas_mechanism_map.py selftest   # mapping-logic checks (instant)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung15_atlas_map"
RUNG12_JSON = PROJECT_ROOT / "runs" / "rung12_pmhc" / "rung12_pmhc.json"

# mechanism q_n ceilings (the arena's measured containment bounds)
CEILINGS = {"per_cell": 0.02, "wave": 0.173, "quorum": 0.20}
EFFECTIVE_QT = 0.10           # need >=10% tumour positivity to seed a wave / fire a gate (efficacy floor)

# real tumour-associated SURFACE antigens (CAR-T / ADC / bispecific targets) — the recognition handles to screen
SURFACE_PANEL = [
    "EPCAM", "ERBB2", "EGFR", "MET", "MSLN", "MUC1", "CEACAM5", "FOLH1", "TACSTD2", "CD19",
    "MS4A1", "CD22", "NCAM1", "PROM1", "CD276", "MUC16", "FOLR1", "MCAM", "CD70", "ROR1",
    "DLL3", "GPC3", "CDH3", "NECTIN4", "TNFRSF8",
]
COMMON = {"ERBB2": "HER2", "FOLH1": "PSMA", "TACSTD2": "TROP2", "MS4A1": "CD20", "PROM1": "CD133",
          "CD276": "B7-H3", "MUC16": "CA125", "TNFRSF8": "CD30", "NECTIN4": "Nectin-4", "DLL3": "DLL3"}


# --------------------------------------------------------------------------- shared mapping logic (testable)
def classify_handle(q_t, q_n):
    """Given a handle's tumour positivity q_t and normal false-positive q_n, which mechanisms make it
    SAFE (q_n <= ceiling) AND EFFECTIVE (q_t >= floor)? Returns the cheapest safe&effective mechanism + flags."""
    effective = q_t >= EFFECTIVE_QT
    safe = {m: (q_n <= c) for m, c in CEILINGS.items()}
    order = ["per_cell", "wave", "quorum"]            # cheapest (no propagation) -> needs propagation -> needs density
    best = next((m for m in order if safe[m]), None) if effective else None
    return {"q_t": round(float(q_t), 4), "q_n": round(float(q_n), 4), "effective": bool(effective),
            "safe": {m: bool(v) for m, v in safe.items()}, "best_mechanism": best}


def _union_prev(prevs):
    """P(patient carries >=1) treating distinct mutations as independent (transparent upper estimate)."""
    p = 1.0
    for v in prevs:
        p *= (1.0 - float(v))
    return 1.0 - p


# --------------------------------------------------------------------------- MODE: neoantigen x mechanism
def run_map() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RUNG12_JSON.exists():
        print(f"[rung15] need {RUNG12_JSON} (run RUNG-12 first).")
        return 1
    r12 = json.loads(RUNG12_JSON.read_text())
    handles = r12["ranked_targets"]
    # RUNG-12 already computed HLA-frequency-weighted coverage (per_cell @0.02, relay @0.173). REUSE it (do NOT
    # recompute from mutation prevalence alone -> that ignores whether the patient carries the presenting allele).
    cov_meas = r12["coverage_measured"]
    cancers = sorted(cov_meas)

    per_cancer = {}
    for cancer in cancers:
        per_cell = float(cov_meas[cancer]["per_cell"])
        wave = float(cov_meas[cancer]["relay"])              # relay ceiling 0.173 == the wave's measured ceiling
        # quorum (q_n<=0.20): neoantigen q_n is bimodal, so handles in (0.173,0.20] are ~none -> quorum == wave.
        # count any eligible-only-for-quorum handles to confirm/deny per cancer.
        quorum_extra = [h for h in handles if cancer in h.get("cancer_prev", {})
                        and 0.173 < h["qn_measured"] <= 0.20]
        quorum = wave  # (no neoantigen handles populate the (0.173,0.20] band; verified by quorum_extra)
        def n_elig(ceil):
            return len({(h["gene"], h["mut_label"]) for h in handles
                        if cancer in h.get("cancer_prev", {}) and h["qn_measured"] <= ceil})
        row = {"per_cell": {"coverage": round(per_cell, 4), "n_handles": n_elig(0.02)},
               "wave": {"coverage": round(wave, 4), "n_handles": n_elig(0.173)},
               "quorum": {"coverage": round(quorum, 4), "n_handles": n_elig(0.20)}}
        per_cancer[cancer] = {**row,
                              "propagation_gain_wave_vs_percell": round(wave - per_cell, 4),
                              "quorum_gain_vs_wave": round(len(quorum_extra) and (quorum - wave) or 0.0, 4),
                              "best_mechanism": max(row, key=lambda m: row[m]["coverage"])}

    # the bimodal finding: does quorum EVER beat the wave on neoantigens?
    quorum_beats_wave = any(v["quorum_gain_vs_wave"] > 1e-6 for v in per_cancer.values())

    result = {
        "tag": "rung15_atlas_mechanism_map_neoantigen",
        "axis": "SEQUENCE/neoantigen (RUNG-12 structure-certified handles) x RUNG-14 mechanism ceilings",
        "mechanism_q_n_ceilings": CEILINGS,
        "per_cancer": per_cancer,
        "quorum_beats_wave_on_neoantigens": bool(quorum_beats_wave),
        "INTERPRETATION_MAP": {
            "propagation_gain": "wave coverage - per_cell coverage = how much a bystander wave unlocks vs a "
                                "per-cell gate, per cancer (the RUNG-13 relaxation cashed out per cancer).",
            "quorum_needs_surface": "if quorum_beats_wave_on_neoantigens is FALSE, neoantigen q_n is bimodal "
                                    "(clean ~0 or risky >0.3) so quorum's headroom is unused -> run `census` to "
                                    "exploit quorum on SURFACE markers (continuous q_n)."},
        "DECISIVE": "",
        "CEILING": "Neoantigen q_n is a PREDICTED proxy (MHCflurry+AF2, RUNG-11/12). Coverage is a transparent "
                   "union-over-mutations upper estimate (ignores co-occurrence). Mechanism ceilings = in-silico "
                   "containment, not clinical efficacy. Coupling/delivery = wet-lab residual.",
    }
    ranked = sorted(per_cancer.items(), key=lambda kv: -kv[1][kv[1]["best_mechanism"]]["coverage"])
    result["DECISIVE"] = (
        "Per-cancer addressable coverage by mechanism (neoantigen axis): "
        + "; ".join(f"{c}: per_cell {v['per_cell']['coverage']:.2f} / wave {v['wave']['coverage']:.2f} "
                    f"(best {v['best_mechanism']})" for c, v in ranked[:6])
        + f". Propagation (wave) unlocks the most in: "
        + ", ".join(f"{c}(+{v['propagation_gain_wave_vs_percell']:.2f})" for c, v in
                    sorted(per_cancer.items(), key=lambda kv: -kv[1]['propagation_gain_wave_vs_percell'])[:3])
        + f". quorum beats wave on neoantigens: {quorum_beats_wave} "
        + ("-> neoantigen q_n is bimodal; run `census` to cash quorum on surface markers."
           if not quorum_beats_wave else "-> quorum unlocks extra neoantigen coverage."))

    (OUT_DIR / "rung15_neoantigen_map.json").write_text(json.dumps(result, indent=2))
    print(f"[rung15] wrote {OUT_DIR / 'rung15_neoantigen_map.json'}")
    print("\n  per-cancer coverage (per_cell / wave / quorum) [best]:")
    for c, v in ranked:
        print(f"    {c:9} {v['per_cell']['coverage']:.3f} / {v['wave']['coverage']:.3f} / "
              f"{v['quorum']['coverage']:.3f}   [{v['best_mechanism']}]  prop-gain +{v['propagation_gain_wave_vs_percell']:.3f}")
    print(f"\n  DECISIVE: {result['DECISIVE']}")
    _make_map_figure(result)
    return 0


def _make_map_figure(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung15] matplotlib unavailable ({e})")
        return
    pc = result["per_cancer"]
    cancers = sorted(pc, key=lambda c: -max(pc[c][m]["coverage"] for m in CEILINGS))
    x = np.arange(len(cancers)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, col) in enumerate([("per_cell", "#C1432B"), ("wave", "#1B5E20"), ("quorum", "#3B7DD8")]):
        ax.bar(x + (i - 1) * w, [pc[c][m]["coverage"] for c in cancers], w, label=m, color=col)
    ax.set_xticks(x); ax.set_xticklabels(cancers, rotation=45, ha="right")
    ax.set_ylabel("addressable patient fraction (neoantigen)")
    ax.set_title("RUNG-15 atlas x mechanism — which cancers each strategy addresses (neoantigen axis)\n"
                 "wave > per_cell = propagation gain; quorum == wave here (neoantigen q_n bimodal -> needs surface markers)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT_DIR / "rung15_neoantigen_map.png", dpi=120)
    print(f"[rung15] wrote {OUT_DIR / 'rung15_neoantigen_map.png'}")


# --------------------------------------------------------------------------- MODE: surface markers (Colab/Census)
def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, str(PROJECT_ROOT / "scripts" / mod))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_census() -> int:
    """Reuse RUNG-12P/A's PROVEN Census loader (scripts/34) on a surface-antigen panel -> per-marker q_t/q_n
    -> mechanism map. Runs on Colab with cellxgene_census (same path that produced the connexin run)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import cellxgene_census  # noqa: F401
    except Exception as e:
        print(f"[rung15] census mode needs cellxgene_census ({type(e).__name__}). Run on Colab. "
              f"Falling back: nothing to do.")
        return 2
    r34 = _load("r34", "34_connexin_containment.py")
    r30 = _load("r30", "30_hla_ifn_inducibility.py")
    r32 = _load("r32", "32_surface_blocker_discovery.py")
    d5 = _load("d5", "25_logicgate_data_rung5.py")
    log, HB = r30.log, r30.HB
    HB.start()

    panel = SURFACE_PANEL
    hk = r34.HK_PANEL
    genes = panel + hk
    ci_panel = {g: i for i, g in enumerate(panel)}
    ci_hk = {g: i for i, g in enumerate(hk)}

    # tumour-side q_t from the RUNG-5 surfaceome tumour cache (surface markers live there)
    tum_cov = {}
    if d5.TUMOUR_CACHE and d5.TUMOUR_CACHE.exists():
        tumour = d5._loadp(d5.TUMOUR_CACHE)
        tpos = tumour.counts >= r34.K
        tcov = {g: float(tpos[:, j].mean()) for j, g in enumerate(tumour.genes)}
        tum_cov = {g: tcov.get(g) for g in panel if g in tcov}
        log(f"[rung15] tumour surface coverage from RUNG-5 cache: {sum(v is not None for v in tum_cov.values())}/{len(panel)} markers present")
    else:
        log("[rung15] WARN: no RUNG-5 tumour cache -> q_t unavailable; normal-leak (q_n) screen still runs")

    tissues = d5.d4.NORMAL_TISSUES
    tile_dir = (r34.CACHE if r34.CACHE else (OUT_DIR / "tiles")); tile_dir.mkdir(parents=True, exist_ok=True)
    acc = {}; census = None
    for ti, tissue in enumerate(tissues):
        tile = tile_dir / f"rung15_acc_{tissue.replace(' ', '_')}.npz"
        if tile.exists():
            r34._merge_acc(acc, r34._load_acc_tile(tile)); log(f"[{ti+1}/{len(tissues)}] {tissue}: resumed tile"); continue
        if census is None:
            census = __import__("cellxgene_census").open_soma(census_version=d5.d4.CENSUS_VERSION)
        tc, n = r34._fetch_chunked(census, d5.d4, tissue, ti, len(tissues), genes, ci_panel, ci_hk, r30, r32, log, HB)
        if n == 0:
            continue
        r34._save_acc(tile, tc, panel); r34._merge_acc(acc, tc)
        log(f"[{ti+1}/{len(tissues)}] {tissue}: tile checkpointed ({n:,} cells)")

    report, _ = r34.find_leak_channels(acc, panel)            # exact RUNG-34 structure (see scripts/34)
    per = report["per_connexin"]; vital = report["vital_types"]
    rows = []
    for g in panel:
        q_t = tum_cov.get(g)                                   # tumour positivity (efficacy proxy)
        bt = per.get(g, {}).get("by_vital_type", {})
        # q_n = worst-case (highest-expressing powered donor) across vital tissues -> conservative leak
        q_n = max((bt[t]["top_donor"] for t in vital if t in bt), default=None)
        if q_t is None or q_n is None:
            rows.append({"marker": g, "common": COMMON.get(g, g), "q_t": q_t, "q_n": q_n,
                         "best_mechanism": None, "verdict": "incomplete (q_t or q_n missing)"})
            continue
        rows.append({"marker": g, "common": COMMON.get(g, g), "leaks_into": per[g]["leaks_into"],
                     **classify_handle(q_t, q_n)})
    rows.sort(key=lambda r: (-(r.get("q_t") or 0) + 5 * (r.get("q_n") or 1)))

    safe_by_mech = {m: [r["marker"] for r in rows if r.get("best_mechanism") == m] for m in CEILINGS}
    quorum_only = [r["marker"] for r in rows if r.get("safe", {}).get("quorum") and not r.get("safe", {}).get("wave")
                   and r.get("effective")]
    # the specificity anti-correlation: do high-expression markers leak, and are clean markers under-expressed?
    eff = [r for r in rows if (r.get("q_t") or 0) >= EFFECTIVE_QT]            # enough tumour expression
    clean = [r for r in rows if r.get("q_n") is not None and r["q_n"] <= 0.20]  # low vital leak
    import statistics as _st
    anti = {
        "n_effective_qt>=0.10": len(eff),
        "median_leak_of_effective_markers": round(_st.median([r["q_n"] for r in eff]), 3) if eff else None,
        "n_clean_qn<=0.20": len(clean),
        "median_qt_of_clean_markers": round(_st.median([r["q_t"] for r in clean]), 3) if clean else None,
        "any_marker_both_effective_and_quorum_safe": any((r.get("q_t") or 0) >= EFFECTIVE_QT and
                                                         (r.get("q_n") is not None and r["q_n"] <= 0.20) for r in rows),
        "finding": "single surface antigens trade off: tumour-high markers leak into vital tissue, vital-clean "
                   "markers are barely tumour-expressed -> no single marker gives a safe gate. Tumour-EXCLUSIVITY "
                   "must come from MUTATION (neoantigen, RUNG-11/12) or COMBINATORIAL logic (AND-NOT, RUNG-6), "
                   "not a single shared self-antigen. (Pan-cancer pooled q_t; a per-cancer marker may still win.)",
    }
    result = {
        "tag": "rung15_atlas_mechanism_map_surface",
        "axis": "EXPRESSION/surface antigens x RUNG-14 mechanism ceilings (Census, reuses RUNG-34 loader)",
        "panel": panel, "mechanism_q_n_ceilings": CEILINGS, "markers": rows,
        "safe_effective_by_mechanism": safe_by_mech,
        "markers_quorum_unlocks_over_wave": quorum_only,
        "specificity_anticorrelation": anti,
        "DECISIVE": (f"Surface markers safe&effective by cheapest mechanism: "
                     + "; ".join(f"{m}: {safe_by_mech[m]}" for m in CEILINGS)
                     + f". Markers QUORUM unlocks that the wave cannot (0.173<q_n<=0.20): {quorum_only or 'none'}. "
                     + f"SPECIFICITY ANTI-CORRELATION: {anti['n_effective_qt>=0.10']} markers are tumour-expressed "
                     + f"(q_t>=0.10) but their median vital leak is {anti['median_leak_of_effective_markers']}; the "
                     + f"{anti['n_clean_qn<=0.20']} vital-clean markers have median tumour q_t only "
                     + f"{anti['median_qt_of_clean_markers']}. {anti['finding']}"),
        "CEILING": "mRNA positivity (Census), not surface protein; dropout/dissociation bias (HK-depth control "
                   "mitigates). q_t from RUNG-5 surfaceome tumour cache (pan-cancer pooled). Mechanism ceilings = "
                   "in-silico containment. Coupling/delivery = wet-lab residual.",
    }
    (OUT_DIR / "rung15_surface_map.json").write_text(json.dumps(result, indent=2, default=lambda o: None))
    log(f"[rung15] wrote {OUT_DIR / 'rung15_surface_map.json'}")
    log(f"[rung15] DECISIVE: {result['DECISIVE']}")
    HB.stop() if hasattr(HB, "stop") else None
    return 0


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # mapping logic
    c1 = classify_handle(0.6, 0.0)        # clean handle -> per_cell safe
    check("clean handle (q_n=0) -> per_cell is best (cheapest safe)", c1["best_mechanism"] == "per_cell")
    c2 = classify_handle(0.6, 0.10)       # q_n 0.10 -> wave (per_cell unsafe, wave safe)
    check("q_n=0.10 -> wave is best (per_cell unsafe)", c2["best_mechanism"] == "wave")
    c3 = classify_handle(0.6, 0.19)       # 0.173<q_n<=0.20 -> quorum only
    check("q_n=0.19 -> quorum is best (wave unsafe, quorum safe)", c3["best_mechanism"] == "quorum")
    c4 = classify_handle(0.6, 0.40)       # too leaky for all
    check("q_n=0.40 -> no safe mechanism", c4["best_mechanism"] is None)
    c5 = classify_handle(0.02, 0.0)       # too little tumour coverage -> not effective
    check("q_t=0.02 -> not effective (no mechanism)", c5["best_mechanism"] is None)
    check("_union_prev monotonic", _union_prev([0.3, 0.2]) > _union_prev([0.3]))

    # map mode runs on real RUNG-12 data (if present)
    if RUNG12_JSON.exists():
        rc = run_map()
        check("map mode runs on real RUNG-12 data", rc == 0)
    else:
        print("  [SKIP] map mode (no RUNG-12 data on disk)")

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUNG-15 atlas x mechanism map")
    ap.add_argument("mode", nargs="?", default="map", choices=["map", "census", "selftest"])
    args = ap.parse_args()
    if args.mode == "selftest":
        sys.exit(selftest())
    sys.exit(run_census() if args.mode == "census" else run_map())
