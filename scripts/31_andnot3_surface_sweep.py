#!/usr/bin/env python3
"""
RUNG 10 / arm (a) — the GPU 3-input SURFACE AND-NOT sweep (the confirmatory run RUNG-6 promised).

THE QUESTION
------------
RUNG-5 found 0/1000 single + 2-input-AND(/AND-NOT) SURFACE gates worst-donor-safe. RUNG-6 argued the NOT-slot
must therefore be a GENETIC-loss signal (HLA-LOH), not a surface marker. arm (a) tests the one case RUNG-5 did
not exhaustively cover: does ANY 3-input SURFACE AND-NOT gate (posA AND posB AND-NOT negC) close the gap? A
negative confirms — at the bigger search scale where the GPU earns its place — that a surface NOT-slot cannot
selectively protect vital tissue, so the NOT-slot MUST be genetic. (Prior: low, given RUNG-8 showed surface
markers are broadly normal-expressed.)

HOW (maximal reuse of audited code — no new scorer, no new GPU code)
-------------------------------------------------------------------
* PANEL: the EXACT RUNG-5 atlas panel, via scripts/25's cached loaders (LOGICGATE_CACHE). Point it at the same
  Drive path you used for RUNG-5 -> the .r5.normal/.r5.tumour caches load INSTANTLY (no re-fetch). If absent,
  it runs the same two-pass fetch as RUNG-5 (per-tissue Drive tiles -> resumable).
* GATES: 3-input AND-NOT family {pos:[A,B], neg:[C]} — A,B = top tumour-expressed surface genes, C = broadly-
  normal surface markers. Pruned to fit a T4 session (top_pos x top_pos/2 x n_neg).
* SCORER: scripts/22 opt.score_gates_vec — the SAME audited worst-donor / Jeffreys-UB / fail-closed-vital /
  AND-NOT scorer RUNG-5 used, with its existing CuPy GPU path (R5_GPU=1 on a T4). Identical semantics => a
  valid apples-to-apples extension of RUNG-5.

ENGINEERING: RESUMABLE per-batch checkpoints to Drive (a disconnect resumes mid-sweep); FOREGROUND heartbeat;
GPU genuinely earns its place here (5e5+ gates x 1.26M cells is compute-bound, unlike the fetch-bound rungs).

HONEST CEILING: transcript-level (mRNA != surface protein); a surviving gate is a HYPOTHESIS needing the full
FDR/permutation/bootstrap rigor (scripts/23) + wet-lab agonism, NOT a cure. Expected result is a negative that
strengthens the genetic-NOT-gate thesis.

USAGE
  python scripts/31_andnot3_surface_sweep.py selftest                                   # synthetic, no atlas
  LOGICGATE_CACHE=/content/drive/MyDrive/cancer-recon/rung5_cache.npz R5_GPU=1 \
      python scripts/31_andnot3_surface_sweep.py run                                      # Colab T4
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung10_andnot3"
RESULT_JSON = OUT_DIR / "rung10_andnot3.json"
FIGURE_PNG = OUT_DIR / "rung10_andnot3.png"

TOP_POS = int(os.environ.get("R10_TOP_POS", "120"))     # # positive (tumour-expressed) genes -> C(TOP_POS,2) pairs
N_NEG = int(os.environ.get("R10_N_NEG", "40"))          # # broadly-normal NOT-slot markers
BATCH = int(os.environ.get("R10_BATCH", "20000"))       # gates per scoring batch (checkpoint granularity)
NORM_BROAD = float(os.environ.get("R10_NORM_BROAD", "0.5"))   # a NOT marker must be positive in > this frac of normal


def log(msg):
    print(f"[+{time.monotonic() - _T0:7.1f}s] [rung10] {msg}", flush=True)


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


# ---------------------------------------------------------------------------
def load_or_build_panel(d5):
    """The EXACT RUNG-5 panel via scripts/25's cached two-pass (reuses LOGICGATE_CACHE; full-fetch fallback)."""
    import cellxgene_census
    genes_full, src = d5.get_surfaceome()
    census = None
    if d5.TUMOUR_CACHE and d5.TUMOUR_CACHE.exists():
        tumour = d5._loadp(d5.TUMOUR_CACHE)
    else:
        HB.set("PASS 1: fetching tumour over full surfaceome (no cache) ...")
        census = cellxgene_census.open_soma(census_version=d5.d4.CENSUS_VERSION)
        tumour = d5.fetch_tumour(census, genes_full)
        if d5.TUMOUR_CACHE:
            d5._save(tumour, d5.TUMOUR_CACHE)
    cov = d5.malignant_coverage_per_gene(tumour, list(tumour.genes))
    shortlist = sorted([g for g in tumour.genes if cov[g] >= d5.GENE_FLOOR], key=lambda g: -cov[g])
    if not shortlist:
        shortlist = sorted(tumour.genes, key=lambda g: -cov[g])[:50]
    tumour = d5.subset_genes(tumour, shortlist)
    log(f"two-pass: {len(shortlist)}/{len(genes_full)} surface genes tumour-expressed (>= {d5.GENE_FLOOR})")

    normal = None
    if d5.NORMAL_CACHE and d5.NORMAL_CACHE.exists():
        cached = d5._loadp(d5.NORMAL_CACHE)
        if list(cached.genes) == shortlist:
            normal = cached
        else:
            log("NORMAL cache gene set != shortlist -> refetching normal")
    if normal is None:
        HB.set("PASS 2: fetching normal atlas over shortlist (no/changed cache) ...")
        census = census or cellxgene_census.open_soma(census_version=d5.d4.CENSUS_VERSION)
        tile_dir = (d5.TUMOUR_CACHE.parent / "r5_normal_tiles") if d5.TUMOUR_CACHE else None
        normal = d5.fetch_normal(census, shortlist, tile_dir=tile_dir)
        if d5.NORMAL_CACHE:
            d5._save(normal, d5.NORMAL_CACHE)
    if census is not None:
        census.close()
    import gc
    panel = d5._concat(normal, tumour)
    del normal, tumour; gc.collect()
    if panel.counts.dtype != np.int16:
        panel.counts = panel.counts.astype(np.int16)
    gc.collect()
    log(f"panel ready: {panel.counts.shape[0]:,} cells x {panel.counts.shape[1]} genes "
        f"(dtype={panel.counts.dtype}, ram {_ram_gb():.1f}GB)")
    return panel, src, len(genes_full), len(shortlist)


def build_andnot3_family(panel, opt, top_pos=TOP_POS, n_neg=N_NEG):
    """3-input AND-NOT gates {pos:[A,B], neg:[C]}: A,B top tumour-expressed; C broadly-normal NOT markers.
    Deterministic order (sorted) so batch checkpoints align across resumed runs."""
    genes = list(panel.genes)
    is_tum = panel.compartment == "tumour"
    is_norm = panel.compartment == "normal"
    tum_cov = {g: float((panel.counts[is_tum, j] >= opt.K).mean()) if is_tum.any() else 0.0
               for j, g in enumerate(genes)}
    norm_pos = (panel.counts[is_norm] >= opt.K) if is_norm.any() else np.zeros((0, len(genes)))
    pos_genes = sorted([g for g in genes], key=lambda g: -tum_cov[g])[:top_pos]
    neg_genes = sorted([g for j, g in enumerate(genes)
                        if is_norm.any() and norm_pos[:, j].mean() > NORM_BROAD and tum_cov[g] < 0.05],
                       key=lambda g: -(norm_pos[:, genes.index(g)].mean()))[:n_neg]
    log(f"family: {len(pos_genes)} positives (top tumour-cov) x C(.,2) x {len(neg_genes)} broadly-normal NOT markers")
    fam = []
    for a, b in itertools.combinations(pos_genes, 2):          # deterministic (sorted input)
        for c in neg_genes:
            if c != a and c != b:
                fam.append({"pos": [a, b], "neg": [c]})
    log(f"3-input AND-NOT family size N = {len(fam):,} gates")
    return fam, pos_genes, neg_genes


# ---------------------------------------------------------------------------
def _ckpt_path():
    base = os.environ.get("LOGICGATE_CACHE")
    d = Path(base).parent if base else OUT_DIR
    return d / "rung10_sweep_ckpt.json"


def sweep(panel, family, opt, required_vital):
    """Batch-score the family via the AUDITED opt.score_gates_vec (GPU via R5_GPU). Resumable: per-batch
    checkpoint of (n_scored, n_safe, safe_gates, best). Heartbeat shows batch progress."""
    ck = _ckpt_path()
    ck.parent.mkdir(parents=True, exist_ok=True)
    state = {"n_scored": 0, "n_safe": 0, "safe_gates": [], "best": [], "verdict_counts": {}}
    if ck.exists():
        try:
            state = json.loads(ck.read_text())
            log(f"RESUMING sweep from checkpoint: {state['n_scored']:,}/{len(family):,} scored, "
                f"{state['n_safe']} safe so far")
        except Exception as e:
            log(f"checkpoint unreadable ({e}); starting fresh")
    start = state["n_scored"]
    nb = (len(family) - start + BATCH - 1) // BATCH
    for bi in range(nb):
        lo = start + bi * BATCH
        hi = min(lo + BATCH, len(family))
        HB.set(f"scoring gates {lo:,}-{hi:,} of {len(family):,} (batch {bi + 1}/{nb}) ...")
        res = opt.score_gates_vec(panel, family[lo:hi], required_vital)
        for r in res:
            state["verdict_counts"][_verdict_key(r["verdict"])] = \
                state["verdict_counts"].get(_verdict_key(r["verdict"]), 0) + 1
            if r["safe"]:
                state["n_safe"] += 1
                if len(state["safe_gates"]) < 200:
                    state["safe_gates"].append({k: r[k] for k in ("gate", "coverage", "vital_leak", "vital_worst", "verdict")})
            # keep the 10 lowest-vital-leak gates as "closest to safe" (the honest near-miss frontier)
            state["best"].append({"gate": r["gate"], "vital_leak": r["vital_leak"], "strict_leak": r["strict_leak"],
                                  "coverage": r["coverage"], "verdict": r["verdict"]})
        state["best"] = sorted(state["best"], key=lambda x: (x["vital_leak"], x["strict_leak"]))[:10]
        state["n_scored"] = hi
        ck.write_text(json.dumps(state))
        log(f"  batch done: {hi:,}/{len(family):,} scored, {state['n_safe']} safe, "
            f"best vital_leak so far={state['best'][0]['vital_leak'] if state['best'] else 'n/a'} (ckpt saved)")
    return state


def _verdict_key(v):
    return v.split(" (")[0].strip()       # collapse "NON-SELECTIVE (vital leak ...)" -> "NON-SELECTIVE"


# ---------------------------------------------------------------------------
def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HB.start()
    d5 = _load("d5", "25_logicgate_data_rung5.py")
    opt = d5.opt
    panel, src, n_full, n_short = load_or_build_panel(d5)
    family, pos_genes, neg_genes = build_andnot3_family(panel, opt)
    if not family:
        log("EMPTY family (no positives or no broadly-normal NOT markers) — a genuine structural negative.")
    state = sweep(panel, family, opt, d5.lg.VITAL_NONREGEN)

    closes_gap = None
    if state["n_safe"] > 0:
        # a SURPRISE positive: report it, but it is NOT believed until the full RUNG-5 rigor (scripts/23 FDR +
        # donor-permutation null + bootstrap) and wet-lab agonism. Flag loudly.
        log(f"*** {state['n_safe']} 3-input SURFACE AND-NOT gates passed the worst-donor safety bar — SURPRISE. "
            f"These are HYPOTHESES; route through scripts/23 (FDR/perm/bootstrap) before believing.")
        closes_gap = "candidate(s) found — UNVERIFIED (needs FDR/permutation/bootstrap + wet-lab agonism)"
    else:
        log("0 of N 3-input surface AND-NOT gates are worst-donor-safe — CONFIRMS the NOT-slot must be genetic "
            "(surface NOT markers are broadly normal-expressed; they cannot selectively spare vital tissue).")
        closes_gap = "NO — 0 safe; addressability gap stays 100% with surface-only 3-input AND-NOT gates"

    result = {
        "tag": "rung10_andnot3_surface_sweep",
        "question": "Does any 3-input SURFACE AND-NOT gate (posA AND posB AND-NOT negC) close the addressability "
                    "gap that single/2-input surface gates could not (RUNG-5: 0/1000)?",
        "scorer": "scripts/22 opt.score_gates_vec (audited worst-donor / Jeffreys-UB / fail-closed-vital / AND-NOT; GPU via R5_GPU)",
        "surfaceome_source": src, "surfaceome_full": n_full, "surfaceome_tumour_expressed": n_short,
        "n_positives": len(pos_genes), "n_neg_markers": len(neg_genes),
        "n_gates_scored": state["n_scored"], "family_size": len(family),
        "n_safe": state["n_safe"], "closes_gap": closes_gap,
        "verdict_distribution": state["verdict_counts"],
        "safe_gates_sample": state["safe_gates"][:50],
        "closest_to_safe_near_miss_frontier": state["best"],
        "positives_used": pos_genes, "neg_markers_used": neg_genes,
        "CEILING": "Transcript-level (mRNA != surface protein). A surviving gate is a HYPOTHESIS needing the "
                   "full FDR/permutation/bootstrap rigor (scripts/23) + wet-lab agonism, NOT a cure. Same "
                   "worst-donor semantics as RUNG-5 (apples-to-apples). Pruned search (top positives x broadly-"
                   "normal NOT markers) — not the full C(682,2)x682 (~1.6e8, ~44 days CPU); the pruning keeps "
                   "the most plausible gates and is stated, not silent.",
        "INTERPRETATION": "Expected (and found if n_safe=0): adding a 3rd SURFACE slot does not rescue safety — "
                          "the bottleneck is the TYPE of signal (broadly-expressed surface), not gate arity. "
                          "This closes RUNG-6 arm (a) and confirms the NOT-slot must be a GENETIC-loss signal "
                          "(HLA-LOH), exactly the direction RUNG-6/7/8 pursued.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    log(f"wrote {RESULT_JSON}")
    log(f"RESULT: {closes_gap}")
    log(f"verdict distribution: {state['verdict_counts']}")
    if state["best"]:
        b = state["best"][0]
        log(f"closest-to-safe gate: {b['gate']}  vital_leak={b['vital_leak']}  verdict={b['verdict']}")
    HB.stop()
    _make_figure(state, len(family))
    return 0


def _make_figure(state, n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable ({e}); skipped figure"); return
    vc = state.get("verdict_counts", {})
    if not vc:
        log("no verdicts -> no figure"); return
    items = sorted(vc.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].barh(range(len(labels)), vals, color=["#C1432B" if "SELECTIVE" not in k or "NON" in k else "#4C9F70" for k in labels])
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel(f"# of {n:,} 3-input surface AND-NOT gates"); ax[0].set_xscale("log")
    ax[0].set_title(f"RUNG-10 arm(a): verdicts ({state['n_safe']} safe of {n:,})")
    best = state.get("best", [])[:10][::-1]
    if best:
        names = [b["gate"][:34] for b in best]; leaks = [b["vital_leak"] * 100 for b in best]
        ax[1].barh(range(len(names)), leaks, color="#888")
        ax[1].axvline(2.0, ls="--", color="#C1432B", label="safety bar (2% vital leak)")
        ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7)
        ax[1].set_xlabel("worst-donor vital leak %"); ax[1].set_title("closest-to-safe gates (near-miss frontier)")
        ax[1].legend(fontsize=8)
    fig.suptitle("RUNG-10 arm(a): no 3-input SURFACE AND-NOT gate is worst-donor-safe -> NOT-slot must be genetic"
                 if state["n_safe"] == 0 else "RUNG-10 arm(a): surprise candidate(s) — UNVERIFIED", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIGURE_PNG, dpi=130)
    log(f"wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    """Synthetic panel — validate family build + batching/checkpoint + reuse of the audited scorer."""
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    opt = _load("opt", "22_gate_optimizer.py")
    lg = _load("lg", "18_logicgate_search.py")
    rng = np.random.default_rng(10)
    genes = ["ACT1", "ACT2", "ACT3", "NEGBROAD", "NEG2"]
    cells, ct, comp, dn = [], [], [], []

    def add(cell_type, compartment, donor, n, vec):
        for _ in range(n):
            cells.append([int(rng.poisson(v)) for v in vec]); ct.append(cell_type); comp.append(compartment); dn.append(donor)
    # tumour: ACT1/ACT2/ACT3 ON, NEGBROAD OFF
    for d in range(3):
        add("tumour_malignant", "tumour", f"T{d}", 200, [8, 8, 8, 0, 0])
    # vital normal (cardiomyocyte): ACT1 ON (leaks), but NEGBROAD ON in normal -> AND-NOT NEGBROAD can spare it
    for d in range(4):
        add("cardiomyocyte", "normal", f"N{d}", 200, [8, 0, 0, 8, 8])   # ACT1 leaks, NEGBROAD high -> blocked by NOT
    # another vital (neuron): ACT2 leaks; NEGBROAD ON here too -> NEGBROAD broadly-normal (>50% of normal cells)
    for d in range(4):
        add("neuron", "normal", f"M{d}", 200, [0, 8, 0, 8, 8])
    panel = lg.Panel(np.array(cells, np.int16), genes, np.array(ct), np.array(["x"] * len(ct)),
                     np.array(comp), donor=np.array(dn))

    fam, pos, neg = build_andnot3_family(panel, opt, top_pos=3, n_neg=2)
    check("family built with 3-input AND-NOT gates", len(fam) > 0 and all(len(g["pos"]) == 2 and len(g["neg"]) == 1 for g in fam))
    check("positives are tumour-expressed genes", set(pos) <= {"ACT1", "ACT2", "ACT3"})
    check("NEGBROAD selected as a broadly-normal NOT marker", "NEGBROAD" in neg)

    # score via the AUDITED scorer; structure + a sanity check
    res = opt.score_gates_vec(panel, fam, lg.VITAL_NONREGEN)
    check("scorer returns one verdict per gate", len(res) == len(fam))
    check("every result has safe/verdict/vital_leak", all({"safe", "verdict", "vital_leak"} <= set(r) for r in res))
    # ACT1 AND ACT2 AND-NOT NEGBROAD: cardiomyocyte (ACT1 on, ACT2 off) doesn't fire; neuron (ACT2 on, ACT1 off) doesn't fire -> low leak
    g = next((r for r in res if set(r["pos"]) == {"ACT1", "ACT2"} and r["neg"] == ["NEGBROAD"]), None)
    check("AND of two non-co-expressed activators has low vital leak", g is not None and g["vital_leak"] <= opt.LEAK_BAR + 1e-9)

    # checkpoint round-trip via sweep() with a temp cache dir
    os.environ["LOGICGATE_CACHE"] = str(OUT_DIR / "selftest_cache.npz")
    ck = _ckpt_path()
    if ck.exists():
        ck.unlink()
    st = sweep(panel, fam, opt, lg.VITAL_NONREGEN)
    check("sweep scores the whole family", st["n_scored"] == len(fam))
    check("checkpoint file written", ck.exists())
    st2 = sweep(panel, fam, opt, lg.VITAL_NONREGEN)        # resume: should be a no-op (already complete)
    check("resume is idempotent (no re-scoring past completion)", st2["n_scored"] == len(fam))
    ck.unlink()

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    args = ap.parse_args()
    sys.exit(selftest() if args.mode == "selftest" else main_run())
