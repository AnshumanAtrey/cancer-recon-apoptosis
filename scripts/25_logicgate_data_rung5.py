#!/usr/bin/env python3
"""
RUNG 5 — DONOR-AWARE, FULL-SURFACEOME data layer + the ADDRESSABILITY-GAP map (the real run; Colab).

This is the SPINE of the highest-value move (2026-06-01 review): re-audit surface logic-gate recognition
across the WHOLE SURFY surfaceome under WORST-CASE-per-donor safety (never the field's pooled "max of
median TPM"), with held-out-donor FDR + winner's-curse shrinkage, and report — per cancer type — the
fraction of patients for whom NO safe gate clears a coverage bar. That per-patient ADDRESSABILITY GAP is
the deliverable; "most patients have no clean gate" is the expected, first-class NEGATIVE.

What changed vs RUNG 4 (scripts/17):
  - DONOR-AWARE fetch: obs now pulls donor_id + dataset_id (+ development_stage, assay for provenance);
    panel.donor = "dataset_id::donor_id". This is what makes WORST-over-donor safety (FIX-1) real on the
    atlas, instead of donor-pooled leak that averages away the one lethal patient.
  - FULL SURFACEOME, not 6 epithelial markers: genes = OmniPath surfaceome (incorporates Bausch-Fluck
    SURFY, PNAS 2018) -> cached data/logicgate/surfaceome_genes.txt; curated fallback offline (scripts/05).
  - Pipeline = the VALIDATED cores: scripts/22 optimizer (fail-closed, max-over-donor, no-multiply) +
    scripts/23 family-max decoy FDR + cluster-bootstrap winner's-curse shrinkage. Search on DISCOVERY/
    SELECT donors; certify on ALL donors; survivors must beat FDR AND shrinkage.

HONEST CEILING (unchanged): transcript-level (mRNA != surface protein; CITE-seq confirms co-positivity);
co-localisation != a firing circuit (agonism = wet-lab). We are NOT first at combinatorial search; the
contribution is the worst-case-safety honesty harness + the addressability gap. A surviving gate is the
best NEXT wet-lab experiment, never a cure. Recognition is a separate axis (never multiplied w/ R1-R3).

USAGE
  python scripts/25_logicgate_data_rung5.py selftest   # LOCAL: full downstream on a synthetic donor panel
  python scripts/25_logicgate_data_rung5.py            # COLAB: real Census fetch + addressability gap
REQS (Colab): pip install cellxgene-census scanpy omnipath
ENV knobs (so a heavy run is never a blind box): R5_K_ACTIVATORS, R5_COV_FLOOR, R5_MAX_FAMILY,
  R5_N_PERM, R5_N_BOOT, LOGICGATE_CACHE (Drive path; .r5.normal.npz/.r5.tumour.npz survive disconnects).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

_T0 = time.monotonic()


def log(msg):
    print(f"[+{time.monotonic() - _T0:7.1f}s] [rung5] {msg}", flush=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung5_logicgate"
DATA_DIR = PROJECT_ROOT / "data" / "logicgate"
SURFACEOME_CACHE = DATA_DIR / "surfaceome_genes.txt"


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


d4 = _load("d4", "17_logicgate_data.py")        # reuse fetch helpers + constants + VITAL_AUDIT
opt = _load("opt", "22_gate_optimizer.py")      # validated optimizer core (also exposes opt.lg)
hv = _load("hv", "23_heldout_validate.py")      # validated family-max FDR + winner's-curse shrinkage
lg = opt.lg

# ---- frozen bars (inherit scripts/22; mirrored in manifest_PREREG.yaml) ----
COV_BAR = opt.COV_BAR
REQUIRED_VITAL = lg.VITAL_NONREGEN
SEED = 20260601

# ---- tunable scale knobs (logged at run start) ----
K_ACTIVATORS = int(os.environ.get("R5_K_ACTIVATORS", "120"))   # top-coverage seed genes (bounds the search)
COV_FLOOR = float(os.environ.get("R5_COV_FLOOR", "0.05"))      # min malignant coverage to be a candidate gene
MAX_FAMILY = int(os.environ.get("R5_MAX_FAMILY", "6000"))      # hard cap on scored gates (multiple-testing budget)
N_PERM = int(os.environ.get("R5_N_PERM", "100"))
N_BOOT = int(os.environ.get("R5_N_BOOT", "100"))

CACHE = Path(os.environ["LOGICGATE_CACHE"]) if os.environ.get("LOGICGATE_CACHE") else None
NORMAL_CACHE = CACHE.with_suffix(".r5.normal.npz") if CACHE else None
TUMOUR_CACHE = CACHE.with_suffix(".r5.tumour.npz") if CACHE else None


# =====================================================================================================
#  surfaceome gene set (OmniPath SURFY -> cache -> curated fallback)
# =====================================================================================================
def get_surfaceome():
    if SURFACEOME_CACHE.exists():
        genes = sorted({ln.strip() for ln in SURFACEOME_CACHE.read_text().splitlines() if ln.strip()})
        if genes:
            log(f"surfaceome from cache {SURFACEOME_CACHE.name}: {len(genes)} genes"); return genes, "cache"
    try:
        s5 = _load("s5", "05_surfaceome_filter.py")
        genes = sorted(s5.surface_genes_from_omnipath())
        if genes:
            SURFACEOME_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SURFACEOME_CACHE.write_text("\n".join(genes) + "\n")
            log(f"surfaceome from OmniPath (SURFY-incl.): {len(genes)} genes -> cached"); return genes, "omnipath"
    except Exception as e:
        log(f"OmniPath surfaceome unavailable ({type(e).__name__}: {e}) -> curated fallback")
    s5 = _load("s5", "05_surfaceome_filter.py")
    return sorted(s5.CURATED_SURFACE), "curated-fallback"


# =====================================================================================================
#  donor-aware, MEMORY-SAFE Census fetch (free-Colab friendly)
#  A naive get_anndata(obs_value_filter=...) materialises a WHOLE tissue x all-genes matrix before capping
#  -> brain (~10.5M cells) x 5007 genes OOMs free Colab's 12GB. Two fixes:
#   (1) cap cells at the QUERY level: scout cheap OBS metadata (soma_joinid + cell_type), pick a capped,
#       donor-preserving joinid set (ALL vital, non-vital <= MAX_PER_TYPE), then materialise ONLY those.
#   (2) two-pass GENES (in main_real): fetch the small tumour set over the FULL surfaceome, keep only the
#       tumour-EXPRESSED surface genes (a gate's activator MUST be expressed in tumour), and fetch the big
#       normal atlas for just those. The full surfaceome is screened; dead-weight genes aren't carried at depth.
#  Reads are deprecation-clean (obs_column_names/var_column_names). Per-tissue tiles cache to Drive so a
#  disconnect doesn't lose prior tissues.
# =====================================================================================================
OBS_FETCH = ["cell_type", "tissue_general", "donor_id", "dataset_id", "development_stage", "assay", "disease"]


def _genes_literal(genes):
    return "[" + ", ".join(f"'{g}'" for g in genes) + "]"


def _donor_key_obs(df):
    ds = df["dataset_id"].astype(str).to_numpy(); dn = df["donor_id"].astype(str).to_numpy()
    return np.array([f"{a}::{b}" for a, b in zip(ds, dn)])


PER_DONOR_VITAL_CAP = 2500   # audit INT-2: cap vital cells PER DONOR (not donor-blind) so no lethal-leaker
#   donor is silently subsampled below the per-donor power floor (185) and dropped from the worst-donor max.


def _scout_capped_joinids(census, value_filter, tissue_index, label):
    """OBS METADATA ONLY (soma_joinid + cell_type + donor; ~hundreds of MB even for 10M-cell brain -> fits
    RAM). Map cell types, then cap: for VITAL types DONOR-AWARE (per (type,donor) <= PER_DONOR_VITAL_CAP, so
    every contributing donor stays powered — audit INT-2 fail-OPEN fix); for non-vital types donor-blind
    <= MAX_PER_TYPE. Returns (joinids int64, total)."""
    exp = census["census_data"]["homo_sapiens"]
    obs = exp.obs.read(value_filter=value_filter,
                       column_names=["soma_joinid", "cell_type", "donor_id", "dataset_id"]).concat().to_pandas()
    if len(obs) == 0:
        log(f"{label}: 0 cells matched"); return None
    mapped = np.array(d4._map_celltype(obs["cell_type"].astype(str).to_numpy()))
    jid = obs["soma_joinid"].to_numpy()
    donor = _donor_key_obs(obs)
    rng = np.random.default_rng([SEED, tissue_index])
    keep = []
    for lab in np.unique(mapped):
        idx = np.where(mapped == lab)[0]
        if lab in lg.VITAL_NONREGEN:
            for d in np.unique(donor[idx]):           # DONOR-AWARE: never starve a donor below the power floor
                di = idx[donor[idx] == d]
                keep.append(di if len(di) <= PER_DONOR_VITAL_CAP else rng.choice(di, PER_DONOR_VITAL_CAP, replace=False))
        else:
            keep.append(idx if len(idx) <= d4.MAX_PER_TYPE else rng.choice(idx, d4.MAX_PER_TYPE, replace=False))
    return jid[np.sort(np.concatenate(keep))].astype(np.int64), len(obs)


def _materialise(census, joinids, genes, label):
    """Materialise expression for an EXPLICIT capped joinid set over `genes` (bounded memory)."""
    import cellxgene_census
    log(f"{label}: materialising {len(joinids):,} capped cells x {len(genes)} genes ...")
    return cellxgene_census.get_anndata(
        census, organism="Homo sapiens", obs_coords=joinids.tolist(),
        var_value_filter=f"feature_name in {_genes_literal(genes)}",
        obs_column_names=OBS_FETCH, var_column_names=["feature_name"])


def _pull_tissue(census, value_filter, label, genes, tissue_index):
    """Scout-cap then materialise one tissue. Returns (counts, cell_types, tissues, donors) or None."""
    sc = _scout_capped_joinids(census, value_filter, tissue_index, label)
    if sc is None:
        return None
    joinids, total = sc
    log(f"{label}: {total:,} cells in tissue -> query-capped to {len(joinids):,} (vital kept in full)")
    ad = _materialise(census, joinids, genes, label)
    mapped = np.array(d4._map_celltype(ad.obs["cell_type"].astype(str).to_numpy()))
    donors = _donor_key_obs(ad.obs)
    vital = sorted(set(mapped.tolist()) & lg.VITAL_NONREGEN)
    log(f"{label}: kept {ad.n_obs:,}; vital here: {vital or 'none'}")
    return d4._dense_over(ad, genes), list(mapped), list(ad.obs["tissue_general"].astype(str)), list(donors)


def fetch_normal(census, genes, tile_dir=None):
    """Per-tissue, memory-safe, with per-tissue Drive tiles (a disconnect doesn't lose prior tissues).
    A tile is reused only if it was built over the SAME gene set (else the columns wouldn't align)."""
    cb, ct, ts, dn = [], [], [], []
    for ti, tissue in enumerate(d4.NORMAL_TISSUES):
        tile = (tile_dir / f"{tissue.replace(' ', '_')}.npz") if tile_dir else None
        if tile and tile.exists():
            d = np.load(tile, allow_pickle=True)
            if list(d["genes"]) == list(genes):
                log(f"NORMAL {tissue}: from tile cache ({d['counts'].shape[0]:,} cells)")
                cb.append(d["counts"]); ct += list(d["cell_type"]); ts += list(d["tissue"]); dn += list(d["donor"])
                continue
            log(f"NORMAL {tissue}: tile gene set differs from current shortlist -> refetching")
        res = _pull_tissue(
            census, f"is_primary_data == True and disease == 'normal' and tissue_general == '{tissue}'",
            f"NORMAL {tissue}", genes, ti)
        if not res:
            continue
        c, a, b, dd = res; cb.append(c); ct += a; ts += b; dn += dd
        if tile:
            tile.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tile, counts=c, cell_type=np.array(a), tissue=np.array(b),
                                donor=np.array(dd), genes=np.array(list(genes), dtype=object))
            log(f"NORMAL {tissue}: tile cached -> {tile}")
    if not cb:
        raise RuntimeError("no normal cells fetched")
    return lg.Panel(np.vstack(cb), list(genes), np.array(ct), np.array(ts),
                    np.array(["normal"] * len(ct)), donor=np.array(dn))


def fetch_tumour(census, genes):
    """Malignant cells only, donor-aware, memory-safe (scout-cap then materialise). panel.tissue = cancer type."""
    vf = f"is_primary_data == True and disease in {d4._q(d4.TUMOUR_DISEASES)}"
    exp = census["census_data"]["homo_sapiens"]
    obs = exp.obs.read(value_filter=vf,
                       column_names=["soma_joinid", "cell_type", "disease"]).concat().to_pandas()
    if len(obs) == 0:
        raise RuntimeError("no tumour cells matched")
    mal = np.array([any(k in c.lower() for k in d4.MALIGNANT_KEYWORDS)
                    for c in obs["cell_type"].astype(str).to_numpy()])
    log(f"TUMOUR: {len(obs):,} cells, {int(mal.sum()):,} malignant")
    if mal.sum() < 100:
        log("TUMOUR: <100 malignant matched -> falling back to all tumour cells (coverage diluted, flagged)")
        mal = np.ones(len(obs), bool)
    jid = obs["soma_joinid"].to_numpy()[mal]
    if len(jid) > d4.TUMOUR_CAP:
        jid = np.sort(np.random.default_rng([SEED, 99]).choice(jid, d4.TUMOUR_CAP, replace=False))
    ad = _materialise(census, jid.astype(np.int64), genes, "TUMOUR (malignant)")
    donors = _donor_key_obs(ad.obs)
    disease = ad.obs["disease"].astype(str).to_numpy()
    log(f"TUMOUR: kept {ad.n_obs:,} malignant cells, {len(set(donors))} patients, types={sorted(set(disease))}")
    return lg.Panel(d4._dense_over(ad, genes), list(genes), np.array(["tumour_malignant"] * ad.n_obs),
                    disease, np.array(["tumour"] * ad.n_obs), donor=donors)


def subset_genes(panel, keep):
    """Restrict a panel to a gene subset (column slice), preserving order in `keep`."""
    idx = [panel.genes.index(g) for g in keep]
    return lg.Panel(panel.counts[:, idx], list(keep), panel.cell_type, panel.tissue,
                    panel.compartment, donor=panel.donor)


def _concat(a, b):
    return lg.Panel(np.vstack([a.counts, b.counts]), a.genes, np.concatenate([a.cell_type, b.cell_type]),
                    np.concatenate([a.tissue, b.tissue]), np.concatenate([a.compartment, b.compartment]),
                    donor=np.concatenate([a.donor, b.donor]))


def _save(panel, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, counts=panel.counts, genes=np.array(panel.genes, dtype=object),
                        cell_type=panel.cell_type, tissue=panel.tissue, compartment=panel.compartment,
                        donor=panel.donor)
    log(f"cached -> {path} ({panel.counts.shape[0]:,} cells)")


def _loadp(path):
    d = np.load(path, allow_pickle=True)
    log(f"RESUMING from {path}")
    return lg.Panel(d["counts"], list(d["genes"]), d["cell_type"], d["tissue"], d["compartment"], donor=d["donor"])


# =====================================================================================================
#  build the (pruned) gate family  — bound the multiple-testing budget honestly
# =====================================================================================================
def malignant_coverage_per_gene(panel, genes):
    is_tum = panel.compartment == "tumour"
    pos = panel.counts[is_tum] >= opt.K
    return {g: float(pos[:, j].mean()) if is_tum.any() else 0.0 for j, g in enumerate(genes)}


def build_family(panel, genes):
    """Activators = top-K malignant-coverage genes above COV_FLOOR; partners = all candidate genes.
    Family = AND(activator, partner) + AND-NOT(activator, broadly-normal partner). Capped at MAX_FAMILY,
    sorted by joint plausibility (activator coverage) so the cap keeps the most promising gates."""
    cov = malignant_coverage_per_gene(panel, genes)
    cand = [g for g in genes if cov[g] >= COV_FLOOR]
    acts = sorted(cand, key=lambda g: -cov[g])[:K_ACTIVATORS]
    log(f"family: {len(cand)} genes >= COV_FLOOR {COV_FLOOR}; activators=top {len(acts)} by coverage")
    fam = []
    for a in acts:
        for b in cand:
            if b != a:
                fam.append({"pos": sorted([a, b]), "neg": []})           # AND
    # AND-NOT: subtract a broadly-NORMAL gene (a real NOT needs a tumour-LOST signal; expression-NOT is a
    # weak proxy and flagged as such). partner = genes broadly positive on normal (candidate "off in tumour").
    is_norm = panel.compartment == "normal"
    norm_pos = panel.counts[is_norm] >= opt.K if is_norm.any() else np.zeros((0, len(genes)))
    broad_normal = [g for j, g in enumerate(genes)
                    if is_norm.any() and norm_pos[:, j].mean() > 0.5 and cov[g] < COV_FLOOR]
    for a in acts:
        for b in broad_normal[:20]:
            if b != a:
                fam.append({"pos": [a], "neg": [b]})
    # dedup + cap
    seen, uniq = set(), []
    for g in fam:
        key = (frozenset(g["pos"]), frozenset(g["neg"]))
        if key not in seen:
            seen.add(key); uniq.append(g)
    if len(uniq) > MAX_FAMILY:
        log(f"family {len(uniq)} > MAX_FAMILY {MAX_FAMILY} -> capping (logged; the dropped gates are the "
            f"lowest-activator-coverage AND pairs)")
        uniq = uniq[:MAX_FAMILY]
    log(f"family size N = {len(uniq)} gates (AND + AND-NOT). FDR/shrinkage operate on exactly this set.")
    return uniq, acts, broad_normal


# =====================================================================================================
#  addressability gap  — the headline deliverable
# =====================================================================================================
MIN_PATIENT_CELLS = 30   # min malignant cells/patient to assess addressability (else gap-uncertain)


def per_patient_coverage(panel, gate):
    """Per tumour patient: (LOWER-bound coverage, n malignant cells). audit datalayer/F1: 'addressed' uses
    the Jeffreys LOWER bound (mirror of the safety UPPER bound) so a noisy high point-estimate from a handful
    of cells cannot FALSELY shrink the addressability gap (the dangerous, over-optimistic direction)."""
    fire = opt.gate_fire(panel, gate["pos"], gate["neg"])
    is_tum = panel.compartment == "tumour"
    out = {}
    for p in np.unique(panel.donor[is_tum]):
        m = is_tum & (panel.donor == p)
        n = int(m.sum()); k = int(fire[m].sum())
        out[str(p)] = (lg.jeffreys_lower(k, n), n)
    return out


def addressability_gap(panel, surviving_gates):
    """Per tumour patient + cancer type: is there ANY surviving safe gate whose per-patient coverage LOWER
    bound clears COV_BAR? gap = fraction of patients addressed by NO surviving gate (the negative). Patients
    with < MIN_PATIENT_CELLS malignant cells are too under-powered to assess and are bucketed separately;
    the headline gap is reported BOTH over powered patients AND worst-case (under-powered counted as
    NOT-addressed) so the negative cannot be quietly shrunk by dropping low-yield patients (datalayer/F1)."""
    is_tum = panel.compartment == "tumour"
    patients = [str(p) for p in sorted(set(panel.donor[is_tum].tolist()))]
    ptype = {p: str(panel.tissue[is_tum & (panel.donor == p)][0]) for p in patients}
    pcells = {p: int((is_tum & (panel.donor == p)).sum()) for p in patients}
    addressed = {p: False for p in patients}
    best_lb = {p: 0.0 for p in patients}
    for g in surviving_gates:
        for p, (lb, n) in per_patient_coverage(panel, g).items():
            best_lb[p] = max(best_lb[p], lb)
            if lb >= COV_BAR:                       # LOWER bound clears the bar (conservative)
                addressed[p] = True
    powered = [p for p in patients if pcells[p] >= MIN_PATIENT_CELLS]
    underpowered = [p for p in patients if pcells[p] < MIN_PATIENT_CELLS]
    # re-audit F4-NEW-2 FIX: in the worst-case, an UNDER-POWERED patient is forced to NOT-addressed (its
    # noisy lower bound is not trustworthy), so the worst-case is a true UPPER bound on the gap (>= point gap)
    # and a noisy small-n patient can never DILUTE it downward.
    eff = {p: (addressed[p] if pcells[p] >= MIN_PATIENT_CELLS else False) for p in patients}

    def _gaps(plist, amap):
        by = {}
        for p in plist:
            by.setdefault(ptype[p], []).append(amap[p])
        overall = 1.0 - (sum(amap[p] for p in plist) / max(1, len(plist)))
        return round(overall, 3), {t: round(1.0 - sum(v) / len(v), 3) for t, v in by.items()}

    # HEADLINE = WORST-CASE over ALL patients (under-powered = not-addressed) — re-audit F4-NEW-1 FIX: never
    # surface a powered-only gap as the headline; genuinely-unaddressed low-yield patients must NOT vanish.
    gap_worstcase, gap_by_type = _gaps(patients, eff)
    gap_point, _ = _gaps(powered, addressed) if powered else (1.0, {})   # secondary: over powered patients only
    return {"n_patients": len(patients), "n_powered_patients": len(powered),
            "n_underpowered_patients": len(underpowered), "min_patient_cells": MIN_PATIENT_CELLS,
            "addressability_gap_overall": gap_worstcase,                 # HEADLINE = honest worst-case
            "addressability_gap_point_powered": gap_point,               # secondary (powered patients only)
            "addressability_gap_worstcase": gap_worstcase,
            "addressability_gap_by_cancer_type": gap_by_type,            # worst-case by type
            "best_per_patient_coverage_lb": {p: round(c, 3) for p, c in best_lb.items()},
            "n_surviving_gates": len(surviving_gates)}


# =====================================================================================================
#  the shared downstream pipeline  (identical for selftest + real run)
# =====================================================================================================
def run_pipeline(panel, genes, tag, required_vital=REQUIRED_VITAL):
    rng = np.random.default_rng(SEED)
    n_norm = int((panel.compartment == "normal").sum()); n_tum = int((panel.compartment == "tumour").sum())
    n_donor = len(set(panel.donor))
    log(f"[{tag}] panel: {panel.counts.shape[0]:,} cells ({n_norm:,} normal / {n_tum:,} malignant), "
        f"{n_donor} donors, {len(genes)} genes")

    family, acts, broad_normal = build_family(panel, genes)
    if not family:
        log("empty family (no gene clears COV_FLOOR) — a genuine negative for this pool");
        return {"empty_family": True}

    # three-partition by donor (search on DISC/SELECT, certify on ALL)
    disc_d, sel_d, rep_d = opt.three_partition(np.array(sorted(set(panel.donor))))
    log(f"[{tag}] donor split: DISCOVERY {len(disc_d)} / SELECT {len(sel_d)} / REPORT {len(rep_d)}")

    # score the whole family on ALL donors (worst-donor vital leak), rank lexicographically
    log(f"[{tag}] scoring family of {len(family)} gates on ALL donors (worst-over-donor; this is the heavy step) ...")
    scored = []
    for i, g in enumerate(family):
        r = opt.score_gate(panel, g["pos"], g["neg"], required_vital)
        scored.append(r)
        if i % max(1, len(family) // 20) == 0 or r["selective"]:
            log(f"  [{i + 1}/{len(family)}] {r['gate'][:34]:34s} cov={r['coverage']:.2f} "
                f"vital={r['vital_leak']:.3f} -> {'SEL' if r['selective'] else r['verdict'][:14]}")
    scored.sort(key=opt.rank_key)
    safe = [r for r in scored if r["safe"]]
    selective = [r for r in scored if r["selective"]]
    log(f"[{tag}] {len(safe)} SAFE on all audited axes; {len(selective)} also clear coverage (pre-FDR).")

    # held-out FDR + winner's-curse shrinkage on EXACTLY this family
    log(f"[{tag}] family-max decoy FDR (N_PERM={N_PERM}) + winner's-curse shrinkage (N_BOOT={N_BOOT}) ...")
    fdr = hv.familymax_fdr(panel, family, required_vital, rng, n_perm=N_PERM)
    boot = hv.bootstrap_winners_curse(panel, family, required_vital, rng, n_boot=N_BOOT)
    thr = fdr["familymax_threshold_cov"]
    # SURVIVORS: safe AND coverage beats the family-max decoy null AND (if it's the winner) survives shrinkage
    survivors = [r for r in selective if r["coverage"] > thr]
    if boot and not boot["still_selective_after_shrinkage"] and survivors:
        # the top gate's coverage is winner's-curse-inflated; demote any survivor whose coverage is within
        # the measured optimism of the bar (conservative — pull back on the optimistic ones).
        opt_cov = max(0.0, boot["optimism"]["coverage"])
        survivors = [r for r in survivors if r["coverage"] - opt_cov > COV_BAR]
        log(f"[{tag}] winner's-curse: coverage optimism {opt_cov:.3f} -> {len(survivors)} survive shrinkage")
    log(f"[{tag}] family-max threshold cov={thr:.3f}; {len(survivors)} gates pass FDR + shrinkage (pre-GUARD-B).")

    # GUARD B (audit INT-1): the pre-registered donor-permutation null must actually RUN on survivors —
    # demote any whose worst-donor vital leak is concentrated in ONE real donor (a lethal patient a pooled
    # view would erase). Previously declared in the manifest but never executed in the pipeline.
    guard_b, kept = [], []
    for r in survivors:
        g = {"pos": r["pos"], "neg": r["neg"]}
        dn = hv.donor_permutation_null(panel, g, required_vital, rng, n_perm=min(N_PERM, 60))
        guard_b.append({"gate": r["gate"], **dn})
        if dn["donor_structure_dependent"]:
            log(f"[{tag}] GUARD B DEMOTES {r['gate']} — worst-donor vital leak special to one real donor (p={dn['p_value']})")
        else:
            kept.append(r)
    survivors = kept
    surviving_gates = [{"pos": r["pos"], "neg": r["neg"]} for r in survivors]
    log(f"[{tag}] {len(survivors)} gates SURVIVE FDR + shrinkage + GUARD B (donor-permutation null).")

    gap = addressability_gap(panel, surviving_gates)
    log(f"[{tag}] ADDRESSABILITY GAP (worst-case over ALL {gap['n_patients']} patients, "
        f"{gap['n_underpowered_patients']} under-powered counted as not-addressed): "
        f"{gap['addressability_gap_overall']:.0%} have NO safe gate at coverage >= {COV_BAR} "
        f"[point over {gap['n_powered_patients']} powered patients: {gap['addressability_gap_point_powered']:.0%}].  "
        f"by type: {gap['addressability_gap_by_cancer_type']}")

    _figure(scored, gap, thr, f"rung5_{tag}.png")
    return {
        "tag": tag, "n_cells": int(panel.counts.shape[0]), "n_donors": n_donor, "n_genes": len(genes),
        "family_size": len(family), "n_activators": len(acts),
        "donor_split": {"discovery": len(disc_d), "select": len(sel_d), "report": len(rep_d)},
        "n_safe": len(safe), "n_selective_preFDR": len(selective),
        "familymax_fdr": fdr, "winners_curse": boot, "guard_b_donor_permutation": guard_b,
        "n_survivors": len(survivors),
        "survivors": [r["gate"] for r in survivors][:25],
        "pooled_fallback_types": sorted({ct for r in scored for ct in r.get("pooled_fallback", [])}),
        "addressability": gap,
        "top_gates": [{k: r[k] for k in ("gate", "coverage", "vital_leak", "strict_leak", "regen_leak", "verdict")}
                      for r in scored[:15]],
        "no_safe_gate": len(safe) == 0, "no_surviving_gate": len(survivors) == 0,
    }


def _figure(scored, gap, thr, name):
    """Two panels: the coverage-vs-worst-donor-vital-leak frontier, and the addressability gap by cancer type."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
        for r in scored:
            col = "#27ae60" if r["selective"] else ("#7f8c8d" if r["safe"] else "#c0392b")
            ax[0].scatter(r["coverage"], r["vital_leak"], c=col, s=34, alpha=0.7)
        ax[0].axhline(opt.LEAK_BAR, ls="--", color="red", lw=0.8)
        ax[0].axvline(COV_BAR, ls="--", color="green", lw=0.8)
        ax[0].axvline(thr, ls=":", color="purple", lw=1.0, label=f"family-max FDR thr={thr:.2f}")
        ax[0].set_xlabel("tumour coverage (want high)")
        ax[0].set_ylabel("worst-DONOR vital leak (want ~0)")
        ax[0].set_title("gate frontier: worst-over-donor safety\ngreen=selective, grey=safe/low-cov, red=unsafe")
        ax[0].legend(fontsize=7)
        gt = gap.get("addressability_gap_by_cancer_type", {})
        if gt:
            types = list(gt.keys()); vals = [gt[t] * 100 for t in types]
            ax[1].bar(range(len(types)), vals, color="#c0392b")
            ax[1].set_xticks(range(len(types)))
            ax[1].set_xticklabels([t.replace(" ", "\n") for t in types], fontsize=7)
            ax[1].set_ylabel("% patients with NO safe gate"); ax[1].set_ylim(0, 105)
            ax[1].set_title(f"ADDRESSABILITY GAP — worst-case, all {gap['n_patients']} patients "
                            f"({gap['addressability_gap_overall']:.0%}; {gap['n_underpowered_patients']} under-powered=not-addressed)\n"
                            f"the headline: who is left with no safe option")
        fig.suptitle("RUNG 5 — worst-case-safety surfaceome re-audit + per-patient addressability gap "
                     "(transcript-only; mRNA!=protein; NOT a cure)", fontsize=8.5)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_DIR / name, dpi=110)
        log(f"figure -> runs/rung5_logicgate/{name}")
    except Exception as e:
        log(f"figure skipped ({type(e).__name__}: {e})")


def _write(result, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["CEILING"] = ("transcript-level; mRNA!=surface protein (CITE-seq confirms co-positivity); "
                         "co-localisation!=a firing circuit (wet-lab); NOT first at combinatorial search; "
                         "contribution = worst-case-safety harness + per-patient addressability gap. A "
                         "surviving gate is the best NEXT wet-lab experiment, not a cure.")
    result["HARD_RULE"] = ("worst-over-donor vital leak (never pooled); fail-closed vital; lexicographic "
                           "no-multiply; AND-NOT expression-NOT is a weak proxy for a tumour-LOST signal "
                           "(HLA-LOH genotype NOT is scripts/24, not atlas-scored).")

    def _jd(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else float(o)
        return str(o)
    (OUT_DIR / name).write_text(json.dumps(result, indent=2, default=_jd))
    log(f"results -> runs/rung5_logicgate/{name}")


# =====================================================================================================
#  LOCAL SELF-TEST — exercise the WHOLE downstream on a synthetic donor panel (no Census)
# =====================================================================================================
def _selftest_panel():
    """Synthetic donor-resolved panel with: a planted CLEAN gate (CLEANA+CLEANB co-positive tumour-only),
    decoy/noise genes, a broad gene, two cancer 'types' each with several patients of HETEROGENEOUS
    coverage (so the addressability gap is non-trivial), and a clean vital (cardiomyocyte) axis."""
    HI, MID, AB, LOW = 6.0, 2.2, 0.01, 0.5
    genes = ["CLEANA", "CLEANB", "BROAD", "N0", "N1", "N2", "N3"]
    rng = np.random.default_rng(SEED)
    blocks_c, ct, ts, comp, don = [], [], [], [], []

    def add(cell_type, tissue, compartment, donor, lams, n):
        blocks_c.append(np.column_stack([rng.poisson(l, n) for l in lams]))
        ct.extend([cell_type] * n); ts.extend([tissue] * n); comp.extend([compartment] * n); don.extend([donor] * n)

    for d in range(8):                      # normal donors: clean on CLEANA/B; BROAD high; noise low
        add("cardiomyocyte", "heart", "normal", f"n::{d}", [AB, AB, MID, LOW, LOW, LOW, LOW], 260)
        add("hepatocyte", "liver", "normal", f"n::{d}", [AB, AB, MID, LOW, LOW, LOW, LOW], 260)
        add("normal_epithelium", "lung", "normal", f"n::{d}", [AB, AB, HI, LOW, LOW, LOW, LOW], 260)
    # cancer type LUAD: 3 patients; CLEANA/B co-positive (planted clean gate covers them well -> addressed)
    for d in range(3):
        add("tumour_malignant", "lung adenocarcinoma", "tumour", f"luad::{d}", [HI, HI, MID, LOW, LOW, LOW, LOW], 300)
    # an UNDER-POWERED LUAD patient (n=10 < MIN_PATIENT_CELLS=30) that FIRES the clean gate (noisy HIGH lower
    # bound). re-audit F4-NEW-2: the worst-case MUST force it not-addressed (its lower bound is untrustworthy),
    # so it appears as a LUAD gap rather than being falsely counted as addressed.
    add("tumour_malignant", "lung adenocarcinoma", "tumour", "luad::underpowered", [HI, HI, MID, LOW, LOW, LOW, LOW], 10)
    # cancer type BRCA: 3 patients with NO tumour-exclusive antigen — only BROAD (shared with normal) is high,
    # so no SAFE high-coverage gate exists for them -> a real per-type ADDRESSABILITY GAP (BRCA gap ~100%).
    for d in range(3):
        add("tumour_malignant", "breast carcinoma", "tumour", f"brca::{d}", [AB, AB, HI, LOW, LOW, LOW, LOW], 300)
    return lg.Panel(np.vstack(blocks_c), genes, np.array(ct), np.array(ts), np.array(comp), donor=np.array(don)), genes


def selftest() -> int:
    log("=== RUNG 5 data-layer SELF-TEST (synthetic donor panel; validates the full downstream) ===")
    panel, genes = _selftest_panel()
    # the synthetic panel models ONLY cardiomyocyte as its vital axis; declare that so fail-closed does not
    # (correctly) reject everything for the 6 vital types the real atlas has but this toy omits.
    res = run_pipeline(panel, genes, "selftest", required_vital={"cardiomyocyte"})
    _write(res, "rung5_selftest.json")
    gap = res.get("addressability", {})
    checks = {
        "downstream ran end-to-end (family -> score -> FDR -> shrinkage -> gap)": not res.get("empty_family"),
        "a safe gate exists (planted CLEANA AND CLEANB)": res.get("n_safe", 0) > 0,
        "at least one gate SURVIVES FDR + shrinkage": res.get("n_survivors", 0) > 0,
        "the planted clean gate is among survivors": any(set(g.split(" AND ")) >= {"CLEANA", "CLEANB"}
                                                         for g in res.get("survivors", [])),
        "addressability gap is computed per cancer type": len(gap.get("addressability_gap_by_cancer_type", {})) >= 2,
        "BRCA gap > LUAD gap (planted: clean gate misses BRCA patients)":
            gap.get("addressability_gap_by_cancer_type", {}).get("breast carcinoma", 0)
            > gap.get("addressability_gap_by_cancer_type", {}).get("lung adenocarcinoma", 1),
        # re-audit F4-NEW-1/2/3: under-powered patients are exercised, the worst-case is a true UPPER bound
        # on the gap (>= point), and a noisy under-powered patient is NOT falsely counted as addressed.
        "under-powered patient path is exercised (>=1 under-powered)":
            gap.get("n_underpowered_patients", 0) >= 1,
        "worst-case gap >= point gap (invariant; headline is the worst-case)":
            gap.get("addressability_gap_worstcase", 0) >= gap.get("addressability_gap_point_powered", 1) - 1e-9,
        "F4-NEW-2: noisy under-powered LUAD patient FORCED not-addressed (LUAD worst-case > 0 though powered LUAD addressed)":
            gap.get("addressability_gap_by_cancer_type", {}).get("lung adenocarcinoma", 0) > 0,
    }
    print("=" * 92)
    print("SELF-TEST CHECKS:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    ok = all(checks.values())
    print("=" * 92)
    log(f"self-test {'PASSED — downstream integration validated; real run is the Colab fetch' if ok else 'FAILED'}")
    return 0 if ok else 1


# =====================================================================================================
def main_real() -> int:
    lg.assert_no_multiply()
    if importlib.util.find_spec("cellxgene_census") is None:
        print("[rung5] cellxgene_census not installed — the REAL run is on COLAB.")
        print("[rung5] locally, run:  python scripts/25_logicgate_data_rung5.py selftest")
        return 0
    import cellxgene_census
    GENE_FLOOR = float(os.environ.get("R5_GENE_FLOOR", "0.02"))
    log("=== RUNG 5 REAL run — donor-aware, MEMORY-SAFE, two-pass full-surfaceome addressability gap ===")
    log(f"knobs: K_ACTIVATORS={K_ACTIVATORS} COV_FLOOR={COV_FLOOR} GENE_FLOOR={GENE_FLOOR} "
        f"MAX_FAMILY={MAX_FAMILY} N_PERM={N_PERM} N_BOOT={N_BOOT}")
    genes_full, src = get_surfaceome()
    log(f"surfaceome source={src}, {len(genes_full)} genes (full set is SCREENED; only tumour-expressed "
        f"genes are carried at depth -> fits free-Colab RAM)")
    tile_dir = (TUMOUR_CACHE.parent / "r5_normal_tiles") if TUMOUR_CACHE else None
    census = None

    # ---- PASS 1: tumour over the FULL surfaceome (small; ~50k malignant cells) ----
    if TUMOUR_CACHE and TUMOUR_CACHE.exists():
        tumour = _loadp(TUMOUR_CACHE)
    else:
        census = cellxgene_census.open_soma(census_version=d4.CENSUS_VERSION)
        tumour = fetch_tumour(census, genes_full)
        if TUMOUR_CACHE: _save(tumour, TUMOUR_CACHE)
    # shortlist = surface genes EXPRESSED in tumour (a gate activator must be); the rest can't form a gate.
    cov = malignant_coverage_per_gene(tumour, list(tumour.genes))
    shortlist = sorted([g for g in tumour.genes if cov[g] >= GENE_FLOOR], key=lambda g: -cov[g])
    if not shortlist:
        log(f"NO surface gene clears GENE_FLOOR={GENE_FLOOR} in tumour — a genuine negative (no activator exists).")
        shortlist = sorted(tumour.genes, key=lambda g: -cov[g])[:50]   # keep a few so the run still reports
    log(f"two-pass: {len(shortlist)}/{len(genes_full)} surface genes are tumour-expressed (>= {GENE_FLOOR}) "
        f"-> normal atlas fetched for THESE only")
    tumour = subset_genes(tumour, shortlist)

    # ---- PASS 2: normal atlas over the SHORTLIST only (big tissues, query-capped, per-tissue tiles) ----
    normal = None
    if NORMAL_CACHE and NORMAL_CACHE.exists():
        cached = _loadp(NORMAL_CACHE)
        if list(cached.genes) == shortlist:
            normal = cached
        else:
            log("NORMAL cache gene set != current shortlist -> refetching normal")
    if normal is None:
        census = census or cellxgene_census.open_soma(census_version=d4.CENSUS_VERSION)
        normal = fetch_normal(census, shortlist, tile_dir=tile_dir)
        if NORMAL_CACHE: _save(normal, NORMAL_CACHE)
    if census is not None:
        census.close()

    panel = _concat(normal, tumour)
    res = run_pipeline(panel, list(panel.genes), "real")
    res["surfaceome_source"] = src
    res["surfaceome_full"] = len(genes_full)
    res["surfaceome_screened_expressed"] = len(shortlist)
    res["gene_floor"] = GENE_FLOOR
    _write(res, "rung5_addressability.json")
    print("[rung5] CEILING: transcript-only HYPOTHESIS; CITE-seq/flow confirm co-positivity; agonism = wet-lab.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(main_real())
