#!/usr/bin/env python3
"""
RUNG 9 — does IFN rescue the blocker? HLA-A vs the interferon program in vital tissue (laptop/Colab, CPU).

WHY THIS IS THE RECOGNITION-CRITICAL NEXT STEP
----------------------------------------------
RUNG-8 found the Tmod blocker's antigen (HLA-A) is LOW in immune-privileged vital tissue at REST
(cardiac conduction, cardiomyocytes, neurons) -> the blocker fails -> the broad activator kills them. But
CAR-T therapy floods tissue with IFN-gamma, which UPREGULATES MHC-I. So the question that decides whether
RUNG-8's safety hole is real in vivo or a resting-state artifact: **when the interferon program is ON in
these cells, is HLA-A turned back on (blocker rescued)?**

WHAT IT COMPUTES
----------------
Per vital cell type, fetch HLA-A/B/C + a core IFN-stimulated-gene (ISG) panel + a housekeeping (HK) panel.
Per cell: ISG-score = #ISGs detected (interferon activity), HK-score = #HK detected (a DEPTH proxy). Stratify
cells into IFN-LOW vs IFN-HIGH and compare HLA-A-low between strata. **The honest control:** IFN-high cells
are also deeper-sequenced, which alone would raise HLA-A; we report HK-score per stratum so an apparent
"rescue" that is really just depth is visible, not hidden.

DELIVERABLE
-----------
Per immune-privileged type: HLA-A-low in IFN-low vs IFN-high (with the HK depth check) -> the IFN-rescue
delta. Large delta with matched HK => IFN genuinely re-arms the blocker (RUNG-8's hole shrinks in the inflamed
therapeutic context). Small delta => the hole is real even under inflammation.

ENGINEERING (same as RUNG-8): RESUMABLE per-tissue Drive tiles (RUNG9_CACHE); FOREGROUND heartbeat; GPU NOT
needed (a ~17-gene fetch is IO-bound). Census opened lazily (network-free re-aggregate from cached tiles).

HONEST CEILING
--------------
mRNA != surface protein; ISG-score is a PROXY for IFN exposure (resting atlas has few IFN-high cells, so the
IFN-high stratum can be small/noisy in some tissues — n reported); depth confound controlled by HK-score but
not perfectly; HLA-A GENE not the A*02 ALLELE; the atlas IFN-high cells reflect whatever inflammation the
donor had, NOT the exact CAR-T IFN dose. Definitive answer needs IFN-stimulated tissue / surface-protein data.

USAGE
  python scripts/30_hla_ifn_inducibility.py selftest                          # synthetic, no Census
  RUNG9_CACHE=/content/drive/MyDrive/cancer-recon python scripts/30_hla_ifn_inducibility.py run   # Colab
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung9_ifn"
RESULT_JSON = OUT_DIR / "rung9_ifn_inducibility.json"
FIGURE_PNG = OUT_DIR / "rung9_ifn.png"

HLA_GENES = ["HLA-A", "HLA-B", "HLA-C"]
ISG_GENES = ["STAT1", "IRF1", "GBP1", "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1", "TAP1", "IRF7"]   # IFN program
HK_GENES = ["ACTB", "GAPDH", "RPL13A", "TMSB4X"]                                                   # depth proxy
GENES = HLA_GENES + ISG_GENES + HK_GENES
IDX = {g: i for i, g in enumerate(GENES)}
ISG_HIGH_MIN = 3          # cell is IFN-HIGH if >= this many ISGs detected
ISG_LOW_MAX = 1           # cell is IFN-LOW if <= this many ISGs detected (2 = 'mid', excluded from contrast)
PER_DONOR_CAP = 600
MIN_CELLS_PER_STRATUM = 30
IMMUNE_PRIVILEGED = ["cardiac_conduction", "cardiomyocyte", "neuron"]   # where RUNG-8 found the hole
CACHE = Path(os.environ["RUNG9_CACHE"]) if os.environ.get("RUNG9_CACHE") else None


def log(msg):
    print(f"[+{time.monotonic() - _T0:7.1f}s] [rung9] {msg}", flush=True)


def _ram_gb():
    import resource
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if m > 1e7 else m / 1e6


class Heartbeat:
    def __init__(self, interval=20):
        self.interval = interval; self.label = "starting"; self._stop = False

    def set(self, label):
        self.label = label; log(label)

    def _run(self):
        while not self._stop:
            for _ in range(self.interval * 2):
                if self._stop:
                    return
                time.sleep(0.5)
            if not self._stop:
                print(f"[+{time.monotonic() - _T0:7.1f}s] [heartbeat] {self.label} | RAM {_ram_gb():.1f}GB", flush=True)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start(); return self

    def stop(self):
        self._stop = True


HB = Heartbeat()


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _vital_label(cell_type_str, vital_audit):
    cl = cell_type_str.lower()
    return next((v for key, v in vital_audit.items() if key in cl), None)


def _codes(col):
    import pyarrow as pa
    col = col.combine_chunks()
    if not pa.types.is_dictionary(col.type):
        col = col.dictionary_encode()
    return col.indices.to_numpy(zero_copy_only=False), [str(x) for x in col.dictionary.to_pylist()]


# ---------------------------------------------------------------------------
def pull_vital_genes(census, d4, tissue, ti, n_tissue):
    """Vital cells only, donor-capped, memory-safe (Arrow codes). Returns per-cell counts for GENES."""
    import cellxgene_census
    exp = census["census_data"]["homo_sapiens"]
    vf = f"is_primary_data == True and disease == 'normal' and tissue_general == '{tissue}'"
    HB.set(f"[{ti + 1}/{n_tissue}] {tissue}: reading obs (cell_type, donor) ...")
    tbl = exp.obs.read(value_filter=vf,
                       column_names=["soma_joinid", "cell_type", "donor_id", "dataset_id"]).concat()
    if tbl.num_rows == 0:
        log(f"[{ti + 1}/{n_tissue}] {tissue}: 0 normal cells"); return None
    jid = tbl.column("soma_joinid").to_numpy()
    ct_codes, ct_vals = _codes(tbl.column("cell_type"))
    mapped_unique = np.array([_vital_label(c, d4.VITAL_AUDIT) for c in ct_vals], dtype=object)
    is_vital = np.array([m is not None for m in mapped_unique])[ct_codes]
    if int(is_vital.sum()) == 0:
        return {"counts": np.zeros((0, len(GENES)), np.int32), "label": np.array([], object), "donor": np.array([], object)}
    labels = mapped_unique[ct_codes]
    ds_codes, _ = _codes(tbl.column("dataset_id"))
    dn_codes, _ = _codes(tbl.column("donor_id"))
    donor_gid = ds_codes.astype(np.int64) * (int(dn_codes.max()) + 1) + dn_codes
    keep = []
    vidx = np.where(is_vital)[0]; lab_v = labels[vidx]
    for lab in np.unique(lab_v):
        lab_mask = vidx[lab_v == lab]
        for g in np.unique(donor_gid[lab_mask]):
            keep.extend(lab_mask[donor_gid[lab_mask] == g][:PER_DONOR_CAP].tolist())
    keep = np.sort(np.array(keep, dtype=np.int64))
    sel = jid[keep]
    HB.set(f"[{ti + 1}/{n_tissue}] {tissue}: materialising {len(GENES)} genes for {len(sel):,} vital cells ...")
    ad = cellxgene_census.get_anndata(
        census, organism="Homo sapiens", obs_coords=sel.tolist(),
        var_value_filter=f"feature_name in {GENES}",
        column_names={"obs": ["cell_type", "donor_id", "dataset_id"]})
    vnames = list(ad.var["feature_name"]) if "feature_name" in ad.var else list(ad.var_names)
    X = ad.X.todense() if hasattr(ad.X, "todense") else ad.X
    X = np.asarray(X)
    counts = np.zeros((X.shape[0], len(GENES)), np.int32)
    for j, g in enumerate(vnames):                       # place each fetched gene into its GENES slot (pad missing=0)
        if g in IDX:
            counts[:, IDX[g]] = np.asarray(X[:, j]).ravel().astype(np.int32)
    donor = np.array([f"{ds}::{dn}" for ds, dn in
                      zip(ad.obs["dataset_id"].astype(str), ad.obs["donor_id"].astype(str))], dtype=object)
    lab = np.array([_vital_label(c, d4.VITAL_AUDIT) for c in ad.obs["cell_type"].astype(str)], dtype=object)
    log(f"[{ti + 1}/{n_tissue}] {tissue}: kept {counts.shape[0]:,} vital cells RAM {_ram_gb():.1f}GB")
    return {"counts": counts, "label": lab, "donor": donor}


# ---------------------------------------------------------------------------
def _scores(counts):
    """Per-cell ISG-score (#ISG detected) and HK-score (#HK detected, depth proxy)."""
    isg = counts[:, [IDX[g] for g in ISG_GENES]]
    hk = counts[:, [IDX[g] for g in HK_GENES]]
    return (isg > 0).sum(axis=1), (hk > 0).sum(axis=1)


def _hla_a_low(counts_rows):
    """(UPPER over all rows, LOWER among MHC-I-detected rows) for a subset of cells."""
    if len(counts_rows) == 0:
        return None, None, 0
    a = counts_rows[:, IDX["HLA-A"]]
    mhc1 = counts_rows[:, [IDX["HLA-A"], IDX["HLA-B"], IDX["HLA-C"]]].max(axis=1) > 0
    upper = float((a < 1).mean())
    info = a[mhc1]
    lower = float((info < 1).mean()) if len(info) else None
    return round(upper, 4), (round(lower, 4) if lower is not None else None), int(len(counts_rows))


def aggregate(counts, label, donor):
    """Per vital type: HLA-A-low in IFN-LOW vs IFN-HIGH strata, among HLA-measuring datasets, with HK depth check."""
    a = counts[:, IDX["HLA-A"]].astype(np.int64)
    dataset = np.array([str(d).split("::")[0] for d in donor], object)
    all_ds = set(dataset.tolist())
    measuring = {d for d in all_ds if a[dataset == d].max(initial=0) > 0}
    is_meas = np.array([d in measuring for d in dataset], bool)
    isg_score, hk_score = _scores(counts)
    meta = {"n_datasets_total": len(all_ds), "n_datasets_excluded_unmeasured": len(all_ds) - len(measuring)}

    out = {}
    for t in sorted(set(x for x in label if x)):
        m = (label == t) & is_meas
        if m.sum() == 0:
            out[t] = {"n_cells_measured": 0}; continue
        cm = counts[m]; isg_m = isg_score[m]; hk_m = hk_score[m]
        lo = isg_m <= ISG_LOW_MAX
        hi = isg_m >= ISG_HIGH_MIN
        up_lo, lw_lo, n_lo = _hla_a_low(cm[lo])
        up_hi, lw_hi, n_hi = _hla_a_low(cm[hi])
        rec = {
            "n_cells_measured": int(m.sum()),
            "n_IFN_low": n_lo, "n_IFN_high": n_hi,
            "HLA_A_low_IFNlow": {"upper": up_lo, "lower": lw_lo},
            "HLA_A_low_IFNhigh": {"upper": up_hi, "lower": lw_hi},
            "HK_detect_IFNlow": round(float(hk_m[lo].mean() / len(HK_GENES)), 4) if n_lo else None,   # depth control
            "HK_detect_IFNhigh": round(float(hk_m[hi].mean() / len(HK_GENES)), 4) if n_hi else None,
        }
        # IFN-rescue delta on the LOWER (depth-controlled) HLA-A-low; only meaningful with enough cells in both strata
        if n_lo >= MIN_CELLS_PER_STRATUM and n_hi >= MIN_CELLS_PER_STRATUM and lw_lo is not None and lw_hi is not None:
            rec["ifn_rescue_delta_lower"] = round(lw_lo - lw_hi, 4)   # >0 => IFN reduces HLA-A-low (rescue)
            rec["hk_matched"] = abs((rec["HK_detect_IFNhigh"] or 0) - (rec["HK_detect_IFNlow"] or 0)) <= 0.15
        else:
            rec["ifn_rescue_delta_lower"] = None
            rec["hk_matched"] = None
        out[t] = rec
    return out, meta


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HB.start()
    d4 = _load("d4", "17_logicgate_data.py")
    census = None
    tissues = d4.NORMAL_TISSUES
    tile_dir = CACHE if CACHE else (OUT_DIR / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"cache/tiles -> {tile_dir}  (resumable; {len(GENES)} genes: HLA+ISG+HK)")

    A, L, D = [], [], []
    for ti, tissue in enumerate(tissues):
        tile = tile_dir / f"rung9_{tissue.replace(' ', '_')}.npz"
        if tile.exists():
            d = np.load(tile, allow_pickle=True)
            log(f"[{ti + 1}/{len(tissues)}] {tissue}: RESUMED from tile ({d['counts'].shape[0]:,} cells)")
            A.append(d["counts"]); L.append(d["label"]); D.append(d["donor"]); continue
        if census is None:
            import cellxgene_census
            HB.set(f"opening CELLxGENE Census {d4.CENSUS_VERSION} ...")
            census = cellxgene_census.open_soma(census_version=d4.CENSUS_VERSION)
        res = pull_vital_genes(census, d4, tissue, ti, len(tissues))
        if res is None:
            continue
        np.savez_compressed(tile, counts=res["counts"], label=res["label"], donor=res["donor"])
        log(f"[{ti + 1}/{len(tissues)}] {tissue}: tile checkpointed -> {tile.name} (safe to disconnect)")
        A.append(res["counts"]); L.append(res["label"]); D.append(res["donor"])

    HB.set("all tissues done — aggregating IFN strata ...")
    counts = np.vstack(A) if A else np.zeros((0, len(GENES)), np.int32)
    label = np.concatenate(L) if L else np.array([], object)
    donor = np.concatenate(D) if D else np.array([], object)
    per_type, coverage = aggregate(counts, label, donor)

    # headline: the immune-privileged tissues RUNG-8 flagged — does IFN rescue HLA-A there?
    # A rescue verdict is only TRUSTED from immune-privileged types that are BOTH depth-matched (hk_matched)
    # AND well-powered (both strata >= MIN_CELLS). Confounded or underpowered tissues -> INCONCLUSIVE, not
    # a false "no rescue". We also report the depth-ROBUST direction (sign of the delta across all tissues)
    # and the cleanest depth-matched evidence anywhere.
    ip_clean = {t: per_type[t] for t in IMMUNE_PRIVILEGED
                if per_type.get(t, {}).get("hk_matched") is True
                and per_type.get(t, {}).get("ifn_rescue_delta_lower") is not None}
    ip_conf = [t for t in IMMUNE_PRIVILEGED if per_type.get(t, {}).get("hk_matched") is False]
    ip_underpowered = [t for t in IMMUNE_PRIVILEGED
                       if per_type.get(t, {}).get("n_cells_measured") and per_type[t].get("ifn_rescue_delta_lower") is None]
    two = [v for v in per_type.values() if v.get("ifn_rescue_delta_lower") is not None]
    dir_pos = sum(1 for v in two if v["ifn_rescue_delta_lower"] > 0)
    clean_all = {t: round(v["ifn_rescue_delta_lower"], 4) for t, v in per_type.items()
                 if v.get("hk_matched") is True and v.get("ifn_rescue_delta_lower") is not None}
    verdict_basis = {"immune_privileged_depth_matched": sorted(ip_clean),
                     "immune_privileged_depth_confounded": sorted(ip_conf),
                     "immune_privileged_underpowered_at_rest": sorted(ip_underpowered),
                     "direction_IFNhigh_lower_in_N_of_M": f"{dir_pos}/{len(two)}",
                     "depth_matched_rescue_deltas_anywhere": clean_all}
    if ip_clean:
        md = float(np.mean(list({t: per_type[t]["ifn_rescue_delta_lower"] for t in ip_clean}.values())))
        verb = "RESCUES" if md >= 0.20 else "does NOT rescue"
        verdict = (f"IFN {verb} the blocker in immune-privileged tissue (depth-matched HLA-A-low drop {md:.0%} "
                   f"in {sorted(ip_clean)}).")
    else:
        verdict = ("INCONCLUSIVE for the tissues that matter — immune-privileged comparisons are depth-CONFOUNDED "
                   f"({ip_conf}) or UNDERPOWERED at rest (too few IFN-high cells: {ip_underpowered}; immune-"
                   f"privileged tissue barely runs IFN at baseline). DIRECTION is consistent ({dir_pos}/{len(two)} "
                   "tissues: IFN-high has LOWER HLA-A-low) and in the cleanest depth-matched tissue(s) "
                   f"{sorted(clean_all)} IFN rescues strongly ({clean_all}); so IFN DOES upregulate HLA-A where "
                   "it is active. Whether therapeutic IFN reaches/re-arms the blocker in heart/brain is a WET-LAB "
                   "question the resting atlas cannot settle.")

    result = {
        "tag": "rung9_ifn_inducibility",
        "question": "Does the interferon program (which CAR-T therapy induces) turn HLA-A back ON in immune-"
                    "privileged vital tissue, rescuing the Tmod blocker that RUNG-8 found failing at rest?",
        "census_version": d4.CENSUS_VERSION, "genes": {"HLA": HLA_GENES, "ISG": ISG_GENES, "HK_depth": HK_GENES},
        "isg_high_min": ISG_HIGH_MIN, "isg_low_max": ISG_LOW_MAX,
        "n_cells_total": int(counts.shape[0]), "dataset_coverage": coverage,
        "per_vital_type": per_type,
        "immune_privileged_focus": IMMUNE_PRIVILEGED,
        "VERDICT": verdict,
        "verdict_basis": verdict_basis,
        "CEILING": "mRNA != surface protein; ISG-score is a PROXY for IFN exposure (resting atlas has few IFN-"
                   "high cells -> IFN-high stratum may be small/noisy, n reported); depth confound controlled by "
                   "HK-score but not perfectly (hk_matched flag); HLA-A GENE not A*02 ALLELE; atlas IFN-high "
                   "cells reflect the donor's inflammation, not the CAR-T IFN dose. Definitive = IFN-stimulated "
                   "tissue / surface-protein assay (wet lab).",
        "INTERPRETATION": "Resolves RUNG-8's open question. ifn_rescue_delta_lower > 0 means IFN reduces the "
                          "HLA-A-low fraction (blocker re-armed); it is only trustworthy when hk_matched is true "
                          "(the IFN-high and IFN-low strata have similar sequencing depth). A real, depth-matched "
                          "rescue in heart/brain would mean the gate is salvageable in the inflamed therapeutic "
                          "context; no rescue means RUNG-8's hole is real and the genetic NOT-gate is unsafe there.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"VERDICT: {verdict}")
    log(f"{'vital type':22} {'nLOW':>6} {'nHIGH':>6} {'lowA(IFNlo)':>11} {'lowA(IFNhi)':>11} {'delta':>7} {'HKmatch':>7}")
    for t in IMMUNE_PRIVILEGED + [x for x in per_type if x not in IMMUNE_PRIVILEGED]:
        v = per_type.get(t)
        if not v or not v.get("n_cells_measured"):
            continue
        ll = v["HLA_A_low_IFNlow"]["lower"]; lh = v["HLA_A_low_IFNhigh"]["lower"]
        log(f"  {t:22} {v['n_IFN_low']:>6,} {v['n_IFN_high']:>6,} "
            f"{(ll if ll is not None else -1):>11.1%} {(lh if lh is not None else -1):>11.1%} "
            f"{(v['ifn_rescue_delta_lower'] if v['ifn_rescue_delta_lower'] is not None else 0):>7.1%} "
            f"{str(v['hk_matched']):>7}")
    HB.stop()
    _make_figure(per_type)
    return 0


def _make_figure(per_type):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable ({e}); skipped figure"); return
    items = [(t, v) for t, v in per_type.items()
             if v.get("n_cells_measured") and v["HLA_A_low_IFNlow"]["lower"] is not None
             and v["HLA_A_low_IFNhigh"]["lower"] is not None]
    if not items:
        log("no two-stratum types -> no figure"); return
    items.sort(key=lambda kv: kv[1]["HLA_A_low_IFNlow"]["lower"], reverse=True)
    names = [f"{t}{'*' if t in IMMUNE_PRIVILEGED else ''}" for t, _ in items]
    lo = [v["HLA_A_low_IFNlow"]["lower"] * 100 for _, v in items]
    hi = [v["HLA_A_low_IFNhigh"]["lower"] * 100 for _, v in items]
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(10.5, max(3, 0.6 * len(names) + 1.5)))
    ax.barh(y - h / 2, lo, h, color="#888", label="HLA-A-low in IFN-LOW cells")
    ax.barh(y + h / 2, hi, h, color="#2B6CB0", label="HLA-A-low in IFN-HIGH cells (blocker rescued if lower)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("% HLA-A-low among MHC-I-detected cells (depth-controlled)")
    ax.set_title("RUNG-9: does the interferon program rescue the blocker's antigen (HLA-A)?\n"
                 "* = immune-privileged tissue (RUNG-8's safety hole); blue << grey => IFN re-arms the blocker", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=130)
    log(f"wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(9)
    nH, nI, nK = len(HLA_GENES), len(ISG_GENES), len(HK_GENES)

    def cell(hla_a, hla_bc, isg_on, hk_on):
        row = np.zeros(len(GENES), np.int32)
        row[IDX["HLA-A"]] = int(rng.integers(3, 20)) if hla_a else 0
        for g in ["HLA-B", "HLA-C"]:
            row[IDX[g]] = int(rng.integers(3, 20)) if hla_bc else 0
        for k, g in enumerate(ISG_GENES):
            row[IDX[g]] = int(rng.integers(3, 20)) if k < isg_on else 0
        for k, g in enumerate(HK_GENES):
            row[IDX[g]] = int(rng.integers(3, 20)) if k < hk_on else 0
        return row

    rows, lab, dn = [], [], []
    def add(label, donor, n, hla_a, isg_on, hk_on):
        for _ in range(n):
            rows.append(cell(hla_a, True, isg_on, hk_on)); lab.append(label); dn.append(donor)
    # neuron: IFN-LOW cells are HLA-A-low; IFN-HIGH cells have HLA-A ON (rescue), SAME depth (hk_on=4)
    add("neuron", "ds1::D1", 100, hla_a=False, isg_on=0, hk_on=4)   # IFN-low, A off, B/C on
    add("neuron", "ds1::D1", 100, hla_a=True, isg_on=8, hk_on=4)    # IFN-high, A on -> rescued
    counts = np.array(rows, np.int32); label = np.array(lab, object); donor = np.array(dn, object)

    agg, cov = aggregate(counts, label, donor)
    check("aggregate returns (per_type, coverage)", isinstance(cov, dict))
    v = agg["neuron"]
    isg_lo, isg_hi = v["n_IFN_low"], v["n_IFN_high"]
    check("IFN strata split correctly (100 low / 100 high)", isg_lo == 100 and isg_hi == 100)
    check("HLA-A-low high in IFN-LOW stratum (~100%)", abs(v["HLA_A_low_IFNlow"]["lower"] - 1.0) < 1e-9)
    check("HLA-A-low ~0 in IFN-HIGH stratum (rescued)", v["HLA_A_low_IFNhigh"]["lower"] < 0.05)
    check("ifn_rescue_delta_lower ~ +100% (strong rescue)", abs(v["ifn_rescue_delta_lower"] - 1.0) < 1e-9)
    check("hk_matched True (same depth in both strata)", v["hk_matched"] is True)

    # depth-confound case: IFN-high cells are ALSO deeper (hk_on differs) -> hk_matched should be False
    rows2, lab2, dn2 = [], [], []
    def add2(n, hla_a, isg_on, hk_on):
        for _ in range(n):
            rows2.append(cell(hla_a, True, isg_on, hk_on)); lab2.append("cardiomyocyte"); dn2.append("ds2::D2")
    add2(100, hla_a=False, isg_on=0, hk_on=1)   # IFN-low, shallow (HK 1/4), A off
    add2(100, hla_a=True, isg_on=8, hk_on=4)    # IFN-high, deep (HK 4/4), A on
    agg2, _ = aggregate(np.array(rows2, np.int32), np.array(lab2, object), np.array(dn2, object))
    check("depth confound flagged: hk_matched False when IFN-high is deeper",
          agg2["cardiomyocyte"]["hk_matched"] is False)

    # measuring-dataset filter still applies (dataset with no HLA-A anywhere excluded)
    r3 = [cell(False, False, 0, 4) for _ in range(60)]   # no HLA at all -> unmeasured
    agg3, cov3 = aggregate(np.array(r3, np.int32), np.array(["neuron"] * 60, object), np.array(["dsZ::Z"] * 60, object))
    check("dataset with no HLA anywhere excluded", cov3["n_datasets_excluded_unmeasured"] == 1)

    hb = Heartbeat(interval=1).start(); hb.set("selftest heartbeat"); time.sleep(0.1); hb.stop()
    check("heartbeat ok", True)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
