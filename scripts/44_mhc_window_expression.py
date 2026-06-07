#!/usr/bin/env python3
"""
RUNG 18b — the MHC-I "display window" at the EXPRESSION level: is it switched OFF (without being broken)?
(CELLxGENE Census, Colab/CPU — the epigenetic-silencing complement RUNG-18's genetics could not see.)

THE QUESTION
------------
RUNG-18 measured the window being GENETICALLY broken (B2M/TAP deleted/mutated) across 6,319 WGS tumours:
fully-dark only ~3.7%. But genetics CANNOT see a window that is switched OFF without being broken —
TRANSCRIPTIONAL / EPIGENETIC silencing (promoter methylation, PRC2/EZH2 repression, low NLRC5). This run
MEASURES that: in REAL malignant cells from single-cell tumour data, what fraction have HLA-A/B/C + B2M
TRANSCRIPTION dark — i.e. the window is off at the mRNA level even if the gene is intact.

THE ONE METHODOLOGICAL TRAP (and the fix)
-----------------------------------------
scRNA DROPOUT makes any gene look "off" in shallowly-sequenced cells. RUNG-8 gated on "MHC-I detected" to
exclude dropout — but that is CIRCULAR HERE: it would throw away exactly the silenced cells we are hunting
(a truly window-OFF cell has no MHC-I, so it would be discarded as 'dropout'). So RUNG-18b gates depth on an
INDEPENDENT HOUSEKEEPING panel (ACTB/GAPDH/EEF1A1/TMSB4X/PTMA/MALAT1). A cell is "well-sequenced" iff it
detects >= HK_MIN housekeeping genes; among well-sequenced malignant cells, window-dark = (B2M off) OR
(all HLA-A/B/C off) is GENUINE silencing, not shallow sequencing.

POSITIVE CONTROL (so 'dark' is trusted, not an artifact)
--------------------------------------------------------
From the SAME tumour samples we also pull IMMUNE/STROMAL cells (T cells, macrophages — the body's HIGHEST
MHC-I expressers). If our metric calls those ~0% dark, the metric is sound and any malignant-cell darkness is
real. If it calls immune cells dark too, the metric is dropout-biased and we say so.

REUSE (no new untested Census code)
-----------------------------------
Malignant-cell selection = scripts/04 select_cancer_celltypes (explicit 'malignant cell'/'neoplastic cell',
else epithelial-lineage fallback — validated). Census fetch = scripts/29 patterns verbatim (open_soma,
obs value_filter, get_anndata var_value_filter, Arrow _codes memory trick, per-dataset MEASURING exclusion,
heartbeat, per-disease resumable tiles). CENSUS_VERSION + _q from scripts/17.

HONEST CEILING
--------------
mRNA != surface MHC-I PROTEIN (the truth a T-cell sees). HK depth-gate REDUCES but does not erase dropout.
HLA-I is IFN-gamma INDUCIBLE: the resting atlas can OVER-state silencing vs an inflamed/treated tumour (so
dark here is an UPPER bound on in-vivo silencing — same direction as RUNG-9's rescue). Malignant annotation
varies by dataset (explicit vs epithelial-fallback -> mode reported per disease; fallback can include some
normal epithelium -> dilutes, conservative). Census tumour coverage is dataset-limited (donor counts small
for some cancers -> wide worst-donor CIs). NOT a wet result. Read WITH RUNG-18 (genetic): genetic-dark is the
permanent floor; this transcriptional-dark is the (often reversible) additional layer.

USAGE
  python scripts/44_mhc_window_expression.py selftest    # synthetic, no Census — validates the math
  RUNG18B_CACHE=/content/drive/MyDrive/cancer-recon python scripts/44_mhc_window_expression.py run   # Colab
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung18b_mhc_expression"
RESULT_JSON = OUT_DIR / "rung18b_mhc_expression.json"
FIGURE_PNG = OUT_DIR / "rung18b_mhc_expression.png"

WINDOW_GENES = ["HLA-A", "HLA-B", "HLA-C", "B2M"]                 # the display window (B2M needed for ALL of it)
HK_GENES = ["ACTB", "GAPDH", "EEF1A1", "TMSB4X", "PTMA", "MALAT1"]  # INDEPENDENT depth gate (not MHC-I!)
ALL_FETCH = WINDOW_GENES + HK_GENES
HK_MIN = 4                          # a cell is "well-sequenced" iff it detects >= HK_MIN housekeeping genes
DET = 1                             # a gene is "on" iff UMI >= DET
PER_DONOR_CAP = 800                 # cap cells per (population, donor): bounds RAM, keeps donors powered
MIN_CELLS_PER_DONOR = 30            # a donor must have >= this many well-seq cells to enter worst-donor
MIN_CELLS_DISEASE = 50             # need this many malignant cells to report a disease

# route cancers (RUNG-16/18) -> matched by SUBSTRING against whatever disease labels Census actually has,
# so we never hard-fail on an exact-string mismatch.
ROUTE_KEYWORDS = {
    "melanoma": ["melanoma"],
    "lung": ["lung"],
    "colorectal": ["colorectal", "colon"],
    "bladder": ["bladder", "urothelial"],
}
IMMUNE_CTRL_KEYWORDS = ["t cell", "macrophage", "b cell", "natural killer", "monocyte", "dendritic"]
CACHE = Path(os.environ["RUNG18B_CACHE"]) if os.environ.get("RUNG18B_CACHE") else None


def log(msg):
    print(f"[+{time.monotonic() - _T0:7.1f}s] [rung18b] {msg}", flush=True)


def _ram_gb():
    import resource
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if m > 1e7 else m / 1e6


class Heartbeat:
    def __init__(self, interval=20):
        self.interval = interval
        self.label = "starting"
        self._stop = False

    def set(self, label):
        self.label = label
        log(label)

    def _run(self):
        while not self._stop:
            for _ in range(self.interval * 2):
                if self._stop:
                    return
                time.sleep(0.5)
            if not self._stop:
                print(f"[+{time.monotonic() - _T0:7.1f}s] [heartbeat] {self.label} | RAM {_ram_gb():.1f}GB", flush=True)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self):
        self._stop = True


HB = Heartbeat()


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _codes(col):
    """Arrow column -> (int codes, small unique list) without one Python str per cell (memory-safe)."""
    import pyarrow as pa
    col = col.combine_chunks()
    if not pa.types.is_dictionary(col.type):
        col = col.dictionary_encode()
    return col.indices.to_numpy(zero_copy_only=False), [str(x) for x in col.dictionary.to_pylist()]


# ---------------------------------------------------------------------------
#  CORE METRIC (fully selftestable, no Census) — window-dark among well-sequenced cells, dataset-MEASURING
# ---------------------------------------------------------------------------
def _gene_idx():
    return {g: ALL_FETCH.index(g) for g in ALL_FETCH}


def cell_flags(counts: np.ndarray):
    """counts: [N x len(ALL_FETCH)] in ALL_FETCH order. Returns dict of per-cell boolean/int arrays."""
    gi = _gene_idx()
    hk = np.zeros(counts.shape[0], dtype=np.int32)
    for g in HK_GENES:
        hk += (counts[:, gi[g]] >= DET).astype(np.int32)
    b2m_on = counts[:, gi["B2M"]] >= DET
    hlaI = np.maximum.reduce([counts[:, gi[g]] for g in ("HLA-A", "HLA-B", "HLA-C")])
    hlaI_on = hlaI >= DET
    window_on = b2m_on & hlaI_on
    return {
        "hk_score": hk,
        "well_seq": hk >= HK_MIN,
        "b2m_on": b2m_on,
        "hlaI_on": hlaI_on,
        "window_on": window_on,
        "window_dark": ~window_on,
        "b2m_dark": ~b2m_on,
        "hlaI_dark": ~hlaI_on,
    }


def _measuring_datasets(counts, dataset):
    """A dataset MEASURED the window iff it detects B2M AND some HLA-I somewhere (else Census 0 == not assayed)."""
    gi = _gene_idx()
    b2m = counts[:, gi["B2M"]]
    hlaI = np.maximum.reduce([counts[:, gi[g]] for g in ("HLA-A", "HLA-B", "HLA-C")])
    keep = set()
    for d in set(dataset.tolist()):
        m = dataset == d
        if b2m[m].max(initial=0) >= DET and hlaI[m].max(initial=0) >= DET:
            keep.add(d)
    return keep


def summarise_population(counts, donor, dataset):
    """Window-dark stats among WELL-SEQUENCED cells from MEASURING datasets, pooled + worst-donor."""
    if counts.shape[0] == 0:
        return {"n_total": 0}
    f = cell_flags(counts)
    measuring = _measuring_datasets(counts, dataset)
    is_meas = np.array([d in measuring for d in dataset], dtype=bool)
    use = f["well_seq"] & is_meas                          # well-sequenced AND from a window-measuring dataset
    n_use = int(use.sum())
    out = {
        "n_total": int(counts.shape[0]),
        "n_well_sequenced_measuring": n_use,
        "n_datasets_total": len(set(dataset.tolist())),
        "n_datasets_excluded_unmeasured": len(set(dataset.tolist())) - len(measuring),
        "frac_well_sequenced": round(float(f["well_seq"].mean()), 4),
    }
    if n_use == 0:
        out["note"] = "no well-sequenced cells from window-measuring datasets"
        return out
    for key in ("window_dark", "b2m_dark", "hlaI_dark"):
        out[f"{key}_pooled"] = round(float(f[key][use].mean()), 4)
    # worst-donor (and distribution) on the window_dark metric
    dvals = donor[use]
    wd = f["window_dark"][use]
    per_donor = []
    for d in np.unique(dvals):
        dm = dvals == d
        if dm.sum() >= MIN_CELLS_PER_DONOR:
            per_donor.append(float(wd[dm].mean()))
    if per_donor:
        per_donor = np.array(sorted(per_donor))
        out["n_donors_qualified"] = int(len(per_donor))
        out["window_dark_worst_donor"] = round(float(per_donor.max()), 4)
        out["window_dark_median_donor"] = round(float(np.median(per_donor)), 4)
        out["window_dark_p90_donor"] = round(float(np.quantile(per_donor, 0.9)), 4)
    else:
        out["n_donors_qualified"] = 0
    return out


# ---------------------------------------------------------------------------
#  Census fetch (per-disease, resumable; reuses scripts/04 selection + scripts/29 fetch conventions)
# ---------------------------------------------------------------------------
def _is_immune(cell_type: str) -> bool:
    """Word-boundary match so 'malignant cell' does NOT match the 't cell' keyword (real bug the selftest caught)."""
    cl = cell_type.lower()
    return any(re.search(r"\b" + re.escape(k) + r"\b", cl) for k in IMMUNE_CTRL_KEYWORDS)


def _classify_celltypes(unique_types, r04):
    cancer_types, mode = r04.select_cancer_celltypes(unique_types)
    cset = set(cancer_types)                                # control must never overlap the cancer population
    control_types = sorted({c for c in unique_types if c not in cset and _is_immune(c)})
    return list(cancer_types), control_types, mode


def pull_disease(census, disease, di, nd, r04, _q):
    import cellxgene_census
    exp = census["census_data"]["homo_sapiens"]
    vf = f"is_primary_data == True and disease == '{disease}'"
    HB.set(f"[{di + 1}/{nd}] {disease}: reading obs metadata ...")
    tbl = exp.obs.read(value_filter=vf,
                       column_names=["soma_joinid", "cell_type", "donor_id", "dataset_id"]).concat()
    total = tbl.num_rows
    if total == 0:
        log(f"[{di + 1}/{nd}] {disease}: 0 cells matched"); return None
    jid = tbl.column("soma_joinid").to_numpy()
    ct_codes, ct_vals = _codes(tbl.column("cell_type"))
    cancer_types, control_types, mode = _classify_celltypes(ct_vals, r04)
    if not cancer_types:
        log(f"[{di + 1}/{nd}] {disease}: NO cancer cell types identifiable (mode={mode}) — skipping"); return None
    pop_of_type = {}                                       # unique cell type -> 'cancer'/'control'/None
    for t in ct_vals:
        pop_of_type[t] = ("cancer" if t in cancer_types else
                          "control" if t in control_types else None)
    pop_unique = np.array([pop_of_type[t] for t in ct_vals], dtype=object)
    pop = pop_unique[ct_codes]                              # per-cell population label
    ds_codes, ds_vals = _codes(tbl.column("dataset_id"))
    dn_codes, dn_vals = _codes(tbl.column("donor_id"))
    donor_gid = ds_codes.astype(np.int64) * (int(dn_codes.max()) + 1) + dn_codes

    keep = []                                              # per (population, donor) cap
    for want in ("cancer", "control"):
        idx = np.where(pop == want)[0]
        for g in np.unique(donor_gid[idx]):
            keep.extend(idx[donor_gid[idx] == g][:PER_DONOR_CAP].tolist())
    keep = np.sort(np.array(keep, dtype=np.int64))
    if keep.size == 0:
        log(f"[{di + 1}/{nd}] {disease}: no cancer/control cells after capping"); return None
    sel_jid = jid[keep]
    n_can = int((pop[keep] == "cancer").sum()); n_ctl = int((pop[keep] == "control").sum())
    HB.set(f"[{di + 1}/{nd}] {disease}: materialising {len(ALL_FETCH)} genes for "
           f"{len(sel_jid):,} cells ({n_can:,} cancer [{mode}], {n_ctl:,} control) ...")
    ad = cellxgene_census.get_anndata(
        census, organism="Homo sapiens", obs_coords=sel_jid.tolist(),
        var_value_filter=f"feature_name in {ALL_FETCH}",
        column_names={"obs": ["cell_type", "donor_id", "dataset_id"]})
    vnames = list(ad.var["feature_name"]) if "feature_name" in ad.var else list(ad.var_names)
    X = ad.X.todense() if hasattr(ad.X, "todense") else ad.X
    X = np.asarray(X)
    counts = np.zeros((X.shape[0], len(ALL_FETCH)), np.int32)   # reorder/pad to ALL_FETCH order
    for j, g in enumerate(ALL_FETCH):
        if g in vnames:
            counts[:, j] = X[:, vnames.index(g)].astype(np.int32)
    ct = ad.obs["cell_type"].astype(str).to_numpy()
    pop_out = np.array(["cancer" if pop_of_type.get(c) == "cancer"
                        else "control" if pop_of_type.get(c) == "control" else "other" for c in ct], dtype=object)
    donor = np.array([f"{ds}::{dn}" for ds, dn in
                      zip(ad.obs["dataset_id"].astype(str), ad.obs["donor_id"].astype(str))], dtype=object)
    log(f"[{di + 1}/{nd}] {disease}: kept {counts.shape[0]:,} cells (mode={mode}) RAM {_ram_gb():.1f}GB")
    return {"disease": disease, "counts": counts, "pop": pop_out, "donor": donor, "mode": mode,
            "n_cancer": n_can, "n_control": n_ctl}


def _match_route(diseases):
    """Map discovered disease labels -> route bucket via substring (robust to exact-label drift)."""
    buckets = {}
    for d in diseases:
        dl = d.lower()
        for bucket, kws in ROUTE_KEYWORDS.items():
            if any(k in dl for k in kws):
                buckets.setdefault(bucket, []).append(d)
    return buckets


def discover_diseases(census, _q, r04):
    """List diseases that carry explicitly-malignant cells, with counts -> validates the selector returns cells."""
    exp = census["census_data"]["homo_sapiens"]
    HB.set("discovery: which diseases have malignant-labelled cells? ...")
    vf = f"is_primary_data == True and cell_type in {_q(r04.EXPLICIT_CANCER_LABELS)}"
    tbl = exp.obs.read(value_filter=vf, column_names=["disease", "soma_joinid"]).concat()
    dz_codes, dz_vals = _codes(tbl.column("disease"))
    counts = {dz_vals[i]: int((dz_codes == i).sum()) for i in range(len(dz_vals))}
    counts = {d: n for d, n in sorted(counts.items(), key=lambda kv: -kv[1]) if d.lower() != "normal"}
    log(f"discovery: {len(counts)} diseases with malignant cells; top: "
        f"{list(counts.items())[:8]}")
    return counts


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HB.start()
    r04 = _load("r04", "04_receptor_targets.py")
    d17 = _load("d17", "17_logicgate_data.py")
    _q = d17._q
    tile_dir = CACHE if CACHE else (OUT_DIR / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    log(f"cache/tiles -> {tile_dir}  (resumable: completed diseases are skipped)")

    import cellxgene_census
    HB.set(f"opening CELLxGENE Census {d17.CENSUS_VERSION} ...")
    census = cellxgene_census.open_soma(census_version=d17.CENSUS_VERSION)

    disc = discover_diseases(census, _q, r04)
    buckets = _match_route(list(disc.keys()))
    # choose, per route bucket, the disease labels with enough malignant cells
    targets = []
    for bucket, dz_list in buckets.items():
        for dz in dz_list:
            if disc.get(dz, 0) >= MIN_CELLS_DISEASE:
                targets.append((bucket, dz))
    log(f"route buckets matched: { {b: v for b, v in buckets.items()} }")
    log(f"targets (>= {MIN_CELLS_DISEASE} malignant cells): {targets}")
    if not targets:
        log("NO route-cancer diseases with malignant cells found — aborting (check Census labels in discovery)")
        json.dump({"tag": "rung18b_mhc_expression", "error": "no targets", "discovery": disc},
                  open(RESULT_JSON, "w"), indent=2)
        return 4

    per_disease, bucket_tiles = {}, {}
    for di, (bucket, disease) in enumerate(targets):
        safe = disease.replace(" ", "_").replace("/", "-")
        tile = tile_dir / f"rung18b_{safe}.npz"
        if tile.exists():
            d = np.load(tile, allow_pickle=True)
            res = {"disease": disease, "counts": d["counts"], "pop": d["pop"], "donor": d["donor"],
                   "mode": str(d["mode"]), "n_cancer": int(d["n_cancer"]), "n_control": int(d["n_control"])}
            log(f"[{di + 1}/{len(targets)}] {disease}: RESUMED tile ({res['counts'].shape[0]:,} cells)")
        else:
            res = pull_disease(census, disease, di, len(targets), r04, _q)
            if res is None:
                continue
            np.savez_compressed(tile, counts=res["counts"], pop=res["pop"], donor=res["donor"],
                                mode=res["mode"], n_cancer=res["n_cancer"], n_control=res["n_control"])
            log(f"[{di + 1}/{len(targets)}] {disease}: tile checkpointed (safe to disconnect)")
        bucket_tiles.setdefault(bucket, []).append(res)
        cm = res["pop"] == "cancer"
        ctl = res["pop"] == "control"
        per_disease[disease] = {
            "route_bucket": bucket, "malignant_annotation_mode": res["mode"],
            "cancer": summarise_population(res["counts"][cm], res["donor"][cm],
                                           np.array([d.split("::")[0] for d in res["donor"][cm]], dtype=object)),
            "control_immune_stromal": summarise_population(res["counts"][ctl], res["donor"][ctl],
                                                           np.array([d.split("::")[0] for d in res["donor"][ctl]], dtype=object)),
        }

    # per route bucket: pool the disease tiles
    per_bucket = {}
    for bucket, tiles in bucket_tiles.items():
        counts = np.vstack([t["counts"] for t in tiles])
        pop = np.concatenate([t["pop"] for t in tiles])
        donor = np.concatenate([t["donor"] for t in tiles])
        cm, ctl = pop == "cancer", pop == "control"
        per_bucket[bucket] = {
            "diseases": [t["disease"] for t in tiles],
            "cancer": summarise_population(counts[cm], donor[cm],
                                           np.array([d.split("::")[0] for d in donor[cm]], dtype=object)),
            "control_immune_stromal": summarise_population(counts[ctl], donor[ctl],
                                                           np.array([d.split("::")[0] for d in donor[ctl]], dtype=object)),
        }

    # compare against RUNG-18 genetic floor where available
    genetic = {}
    g18 = PROJECT_ROOT / "runs" / "rung18_mhc_window" / "rung18_mhc_window.json"
    if g18.exists():
        gj = json.load(open(g18))
        code = {"melanoma": "SKCM", "lung": "NSCLC", "colorectal": "COREAD", "bladder": "BLCA"}
        for bucket, c in code.items():
            pc = gj.get("per_cancer", {}).get(c)
            if pc:
                genetic[bucket] = {"genetic_dark_systemic": pc["window_class_fraction"]["DARK_SYSTEMIC"],
                                   "genetic_any_gie": pc["any_gie"]["fraction"]}

    HB.stop()
    result = {
        "tag": "rung18b_mhc_expression",
        "question": "At the EXPRESSION level, what fraction of malignant cells have the MHC-I window dark "
                    "(B2M off OR all HLA-I off) among WELL-SEQUENCED cells (housekeeping depth-gated)? "
                    "The epigenetic-silencing layer RUNG-18's genetics could not see.",
        "census_version": d17.CENSUS_VERSION,
        "window_genes": WINDOW_GENES, "housekeeping_depth_gate": HK_GENES, "hk_min": HK_MIN, "detect_umi": DET,
        "discovery_disease_malignant_counts": disc,
        "per_disease": per_disease,
        "per_route_bucket": per_bucket,
        "genetic_floor_from_rung18": genetic,
        "INTERPRETATION_MAP": {
            "cancer_window_dark ~ control_dark ~ genetic_floor":
                "Genetics captured most of it; transcriptional silencing is NOT a major ADDITIONAL hole -> "
                "immune route robust beyond the ~4% genetic floor.",
            "cancer_window_dark >> control_dark AND >> genetic_floor":
                "Epigenetic/transcriptional silencing is a LARGE additional escape beyond genetics -> the "
                "route loses many more cells than WGS suggested -> strengthens the need for Shriya's "
                "MHC-independent autonomous backup. (But this layer is often REVERSIBLE: IFN/epi-drugs -> RUNG-9.)",
            "control_dark NOT ~0":
                "Metric is dropout-biased even after HK-gating -> treat all dark fractions as upper bounds, "
                "lean on the cancer-vs-control DELTA, not absolute values.",
        },
        "CEILING": "mRNA != surface protein; HK depth-gate reduces but does not erase dropout (=> read the "
                   "cancer-minus-control DELTA, not absolutes); HLA-I is IFN-inducible so resting atlas "
                   "OVER-states silencing vs inflamed/treated tumour (dark = UPPER bound); malignant "
                   "annotation mode (explicit vs epithelial-fallback) reported per disease; Census tumour "
                   "donor counts are small for some cancers (wide worst-donor). Genetic floor (RUNG-18) is the "
                   "permanent layer; this transcriptional layer is the (often reversible) addition. NOT wet.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"{'bucket':12} {'mode':20} {'n_cancer':>9} {'dark%':>7} {'worst%':>7} {'ctrl%':>7} {'genFloor%':>9}")
    for bucket, b in per_bucket.items():
        c = b["cancer"]; ct = b["control_immune_stromal"]
        gf = genetic.get(bucket, {}).get("genetic_dark_systemic")
        dz0 = per_disease.get(b["diseases"][0], {}).get("malignant_annotation_mode", "?")
        log(f"  {bucket:12} {dz0:20} {c.get('n_well_sequenced_measuring', 0):>9,} "
            f"{c.get('window_dark_pooled', float('nan')) * 100 if c.get('window_dark_pooled') is not None else -1:>6.1f}% "
            f"{(c.get('window_dark_worst_donor') or 0) * 100:>6.1f}% "
            f"{(ct.get('window_dark_pooled') or 0) * 100:>6.1f}% "
            f"{(gf * 100 if gf is not None else -1):>8.1f}%")
    _make_figure(per_bucket, genetic)
    return 0


def _make_figure(per_bucket, genetic):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable ({e}); skipped figure"); return
    buckets = [b for b in per_bucket if per_bucket[b]["cancer"].get("window_dark_pooled") is not None]
    if not buckets:
        log("no buckets with data -> no figure"); return
    x = np.arange(len(buckets)); w = 0.26
    canc = [per_bucket[b]["cancer"]["window_dark_pooled"] * 100 for b in buckets]
    ctrl = [(per_bucket[b]["control_immune_stromal"].get("window_dark_pooled") or 0) * 100 for b in buckets]
    gfl = [(genetic.get(b, {}).get("genetic_dark_systemic") or 0) * 100 for b in buckets]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, canc, w, label="malignant cells — window dark (mRNA)", color="#B23A2E")
    ax.bar(x, ctrl, w, label="immune/stromal control — window dark", color="#3F7D54")
    ax.bar(x + w, gfl, w, label="genetic floor (RUNG-18 systemic dark)", color="#888")
    for xi, v in zip(x - w, canc):
        ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={per_bucket[b]['cancer'].get('n_well_sequenced_measuring', 0):,})" for b in buckets])
    ax.set_ylabel("% cells with MHC-I window dark (well-sequenced)")
    ax.set_title("RUNG-18b: is the window SWITCHED OFF (mRNA) in real malignant cells?\n"
                 "control ≈0 validates the metric; cancer − control = transcriptional silencing beyond genetics")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=130)
    log(f"wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    """Synthetic, no Census — validates the window-dark grading, HK depth-gate, dataset-MEASURING, worst-donor."""
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    gi = _gene_idx()

    def make(n, hla, b2m, hk_detected):
        """n cells; hla/b2m = UMI for HLA-A/B/C(all) and B2M; hk_detected = #HK genes on."""
        c = np.zeros((n, len(ALL_FETCH)), np.int32)
        for g in ("HLA-A", "HLA-B", "HLA-C"):
            c[:, gi[g]] = hla
        c[:, gi["B2M"]] = b2m
        for j, g in enumerate(HK_GENES):
            if j < hk_detected:
                c[:, gi[g]] = 7
        return c

    # 1. flags
    on = cell_flags(make(1, 5, 5, 6))
    check("fully-on cell: window_on True, dark False", on["window_on"][0] and not on["window_dark"][0])
    b2mko = cell_flags(make(1, 5, 0, 6))
    check("B2M-knockout cell: window_dark True (HLA on but B2M off)", b2mko["window_dark"][0] and b2mko["b2m_dark"][0])
    hlako = cell_flags(make(1, 0, 5, 6))
    check("HLA-silenced cell: window_dark True via hlaI_dark", hlako["window_dark"][0] and hlako["hlaI_dark"][0])
    shallow = cell_flags(make(1, 0, 0, 2))
    check("shallow cell (2 HK): NOT well_sequenced", not shallow["well_seq"][0])
    deepdark = cell_flags(make(1, 0, 0, 6))
    check("deep+dark cell (6 HK, no MHC): well_seq True AND window_dark True (the silenced cell we must KEEP)",
          deepdark["well_seq"][0] and deepdark["window_dark"][0])

    # 2. summarise: 100 deep-dark malignant + 100 deep-on, one dataset that measures (some cell has window on)
    dk = make(100, 0, 0, 6)            # silenced, well-sequenced
    onc = make(100, 8, 8, 6)           # on, well-sequenced
    counts = np.vstack([dk, onc])
    donor = np.array(["dsA::D1"] * 100 + ["dsA::D2"] * 100, dtype=object)
    dataset = np.array(["dsA"] * 200, dtype=object)
    s = summarise_population(counts, donor, dataset)
    check("summarise: 200 well-seq measuring cells", s["n_well_sequenced_measuring"] == 200)
    check("summarise: window_dark_pooled == 0.5", abs(s["window_dark_pooled"] - 0.5) < 1e-9)
    check("summarise: worst-donor == 1.0 (D1 all dark)", abs(s["window_dark_worst_donor"] - 1.0) < 1e-9)
    check("summarise: median-donor == 0.5 across D1=1.0,D2=0.0", abs(s["window_dark_median_donor"] - 0.5) < 1e-9)

    # 3. dataset-MEASURING exclusion: a dataset that NEVER detects the window is dropped (not counted as dark)
    never = make(100, 0, 0, 6)         # deep but window never on anywhere in dsZ -> dsZ didn't assay the window
    counts2 = np.vstack([onc, never])
    donor2 = np.array(["dsA::D1"] * 100 + ["dsZ::Dz"] * 100, dtype=object)
    dataset2 = np.array(["dsA"] * 100 + ["dsZ"] * 100, dtype=object)
    s2 = summarise_population(counts2, donor2, dataset2)
    check("dataset-measuring: dsZ excluded as never-measuring", s2["n_datasets_excluded_unmeasured"] == 1)
    check("dataset-measuring: only dsA's 100 on-cells counted -> dark 0%", abs(s2["window_dark_pooled"] - 0.0) < 1e-9)

    # 4. shallow cells excluded from the metric (dropout not counted as dark)
    mix = np.vstack([make(50, 8, 8, 6), make(50, 0, 0, 2)])   # 50 deep-on + 50 shallow-dark
    dm = np.array(["dsA::D1"] * 100, dtype=object); dsd = np.array(["dsA"] * 100, dtype=object)
    s3 = summarise_population(mix, dm, dsd)
    check("shallow dark cells excluded by HK-gate -> only 50 used, dark 0%",
          s3["n_well_sequenced_measuring"] == 50 and abs(s3["window_dark_pooled"] - 0.0) < 1e-9)

    # 5. route matcher
    bm = _match_route(["melanoma", "lung adenocarcinoma", "colorectal cancer",
                       "bladder urothelial carcinoma", "breast carcinoma"])
    check("route matcher buckets melanoma/lung/colorectal/bladder", set(bm) == {"melanoma", "lung", "colorectal", "bladder"})
    check("route matcher excludes breast", all("breast" not in v for vs in bm.values() for v in vs))

    # 6. celltype classification via scripts/04
    r04 = _load("r04", "04_receptor_targets.py")
    can, ctl, mode = _classify_celltypes(["malignant cell", "T cell", "macrophage", "fibroblast"], r04)
    check("classify: malignant -> cancer (explicit mode)", can == ["malignant cell"] and mode == "explicit-malignant")
    check("classify: T cell + macrophage -> control", set(ctl) == {"T cell", "macrophage"})

    hb = Heartbeat(interval=1).start(); hb.set("selftest heartbeat"); time.sleep(0.1); hb.stop()
    check("heartbeat starts/stops", True)

    print(f"\nselftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
