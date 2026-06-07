#!/usr/bin/env python3
"""
RUNG 20 (Boltz 2b) — structural BINDING specificity of the top clean handles, by an INDEPENDENT SOTA model.
(Boltz-2, GPU/cloud. The orthogonal cross-check on the binding axis — Shriya stage 2: recognise -> BIND.)

WHAT THIS CLOSES
----------------
RUNG-11 (MHCflurry) and RUNG-12 (ESMFold) both said our top clean neoantigen handles present the MUTANT
peptide on MHC while the WILD-TYPE does NOT (so a TCR raised on the mutant won't fire on self = the safety
basis). But both share blind spots: MHCflurry is sequence-only; ESMFold is single-sequence and dockets the
peptide imperfectly (its OWN stated ceiling). This rung re-asks the question with **Boltz-2**, a SOTA
co-folding model (AlphaFold3-class), as an INDEPENDENT third opinion:

  for each top clean handle, fold  MHC-groove + MUTANT peptide  AND  MHC-groove + WILD-TYPE peptide,
  and ask: does the MUTANT present confidently (interface pLDDT / ipTM high) while the WT presents WORSE?

THREE-WAY AGREEMENT = the honest deliverable
--------------------------------------------
Per handle we compare Boltz's mut-vs-WT discrimination against RUNG-11 (MHCflurry mut_rank vs wt_rank) and
RUNG-12 (ESMFold D). Where all THREE agree the handle is CERTIFIED across independent methods; where Boltz
DISAGREES it is FLAGGED as model-dependent (lower confidence -> needs wet validation). We do NOT fit anything;
the thresholds are a-priori (interface pLDDT >= 0.70, the RUNG step-1 / script-02 convention).

THE MAGE-A3 CAVEAT (why this is specificity, not just presentation)
-------------------------------------------------------------------
Presentation is necessary, not sufficient: MAGE-A3's TCR presented fine and still cross-reacted with a CARDIAC
titin peptide and killed patients. Boltz here checks MUTANT-vs-WT discrimination (the self-tolerance axis), NOT
a proteome-wide mimicry scan (a separate, heavier test). So a PASS = "mutant presents, self-WT doesn't" — it
REDUCES the residual, it does not prove no dangerous mimic exists. Stated, not hidden.

REUSE (no new untested infra)
-----------------------------
Peptide pairs (pep_mut/pep_wt/p_in_pep) + the MHC `groove` come straight from runs/rung12_pmhc/
rung12_manifest.json (already built). Boltz YAML/CLI mirrors scripts/01_boltz_smoketest.py; interface metrics
mirror scripts/02_interface_metrics.py. ESMFold D for the cross-check comes from rung12_pmhc.json.

HONEST CEILING
--------------
Boltz confidence (ipTM / interface pLDDT) is a STRUCTURAL plausibility, not a measured affinity (Boltz-2's
affinity head is small-molecule only — peptides are categorically rejected, same as RUNG-1's note). pMHC class-I
peptide docking is hard for any model; a single diffusion sample is used for cost (raise --diffusion_samples for
production). MUTANT-vs-WT discrimination only (self-tolerance), NOT proteome-wide mimicry. NOT a wet result —
this is a third in-silico opinion that RAISES or LOWERS confidence in the RUNG-11/12 calls.

USAGE
  python scripts/46_boltz_pmhc_specificity.py selftest    # no GPU — validates selection, YAML, metric, cross-check
  python scripts/46_boltz_pmhc_specificity.py prepare     # no GPU — write the Boltz inputs + target list, inspect
  python scripts/46_boltz_pmhc_specificity.py run         # GPU — fold mut+WT pMHC for each handle, score, cross-check
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "runs" / "rung12_pmhc" / "rung12_manifest.json"
RUNG12 = PROJECT_ROOT / "runs" / "rung12_pmhc" / "rung12_pmhc.json"
RUNG17 = PROJECT_ROOT / "runs" / "rung17_immunogenicity" / "rung17_immunogenicity.json"
OUT_DIR = PROJECT_ROOT / "runs" / "rung20_boltz_specificity"
INPUT_DIR = OUT_DIR / "inputs"
PRED_DIR = OUT_DIR / "preds"
RESULT_JSON = OUT_DIR / "rung20_boltz_specificity.json"
FIGURE_PNG = OUT_DIR / "rung20_boltz_specificity.png"

N_HANDLES = 6                 # top clean handles to fold (×2 for mut+WT = 12 Boltz jobs)
IPLDDT_MIN = 0.70             # a-priori "credible interface" (script 02 convention), [0,1]
DIFFUSION_SAMPLES = 1         # raise for production; 1 keeps the Colab GPU run ~40-60 min


# ---------------------------------------------------------------------------
#  handle selection — clean, prioritised by RUNG-17 immunogenicity then cancer prevalence
# ---------------------------------------------------------------------------
def load_handles():
    manifest = json.load(open(MANIFEST))["handles"]
    by_id = {h["id"]: h for h in manifest}
    rung17 = json.load(open(RUNG17)).get("top_clean_handles", []) if RUNG17.exists() else []
    # composite_z per (gene, allele) from RUNG-17 (immunogenicity priority)
    z_of = {}
    for h in rung17:
        key = (h.get("gene"), h.get("allele"), h.get("peptide"))
        z_of[key] = h.get("composite_z", 0.0)
    return manifest, by_id, z_of


def select_handles(manifest, z_of, n=N_HANDLES, max_per_driver=2):
    """Clean handles, ranked by RUNG-17 immunogenicity composite (if known) then cancer prevalence;
    at most `max_per_driver` alleles per (gene,mut) so the set spans drivers, not one driver's alleles."""
    clean = [h for h in manifest if h.get("tier") == "clean" and h.get("pep_mut") and h.get("pep_wt")]

    def score(h):
        z = max((z_of.get((h["gene"], h["allele"], h["pep_mut"]), None),
                 z_of.get((h["gene"], h["allele"], h.get("pep_mut", "")[:9]), None),
                 -99), key=lambda v: (v is not None, v))
        z = z if isinstance(z, (int, float)) else -99
        return (z, h.get("max_prev", 0.0))

    ranked = sorted(clean, key=score, reverse=True)
    per_driver, picked = {}, []
    for h in ranked:
        key = (h["gene"], h["mut_label"])
        if per_driver.get(key, 0) >= max_per_driver:
            continue
        per_driver[key] = per_driver.get(key, 0) + 1
        picked.append(h)
        if len(picked) >= n:
            break
    return picked


# ---------------------------------------------------------------------------
#  Boltz IO (mirrors scripts/01) — pMHC = groove (chain A) + peptide (chain B)
# ---------------------------------------------------------------------------
def write_pmhc_yaml(yaml_path: Path, groove: str, peptide: str, name: str):
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"# Boltz-2 pMHC input for {name} (MHC groove = A, peptide = B; analysed via interface metrics)\n"
        f"sequences:\n"
        f"  - protein:\n      id: A\n      sequence: {groove}\n"
        f"  - protein:\n      id: B\n      sequence: {peptide}\n"
    )


def have_boltz() -> bool:
    return shutil.which("boltz") is not None


def run_boltz(yaml_path: Path, out_dir: Path, diffusion_samples: int = DIFFUSION_SAMPLES) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["boltz", "predict", str(yaml_path), "--use_msa_server",
           "--diffusion_samples", str(diffusion_samples), "--out_dir", str(out_dir)]
    print(f"[rung20] invoking: {' '.join(cmd)}", flush=True)
    with open(out_dir / "boltz.log", "w") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); logf.write(line)
        return proc.wait()


def read_confidence(out_dir: Path) -> dict | None:
    js = sorted(out_dir.rglob("confidence_*.json"))
    if not js:
        return None
    d = json.load(open(js[0]))
    return {"complex_iplddt": d.get("complex_iplddt"), "iptm": d.get("iptm"),
            "complex_plddt": d.get("complex_plddt"), "confidence_score": d.get("confidence_score"),
            "complex_ipde": d.get("complex_ipde")}


# ---------------------------------------------------------------------------
#  metric — structural mut-vs-WT discrimination + three-way agreement
# ---------------------------------------------------------------------------
def boltz_discrimination(mut_conf: dict, wt_conf: dict, iplddt_min: float = IPLDDT_MIN) -> dict:
    """MHC-level structural discrimination: mut presents (interface pLDDT high) while WT presents WORSE.
    D_boltz in [0,1] = clipped presentation gap; presents_mut = mut clears the a-priori credible-interface bar."""
    mi = mut_conf.get("complex_iplddt")
    wi = wt_conf.get("complex_iplddt")
    if mi is None or wi is None:
        return {"D_boltz": None, "presents_mut": None, "note": "missing interface pLDDT"}
    presents_mut = bool(mi >= iplddt_min)
    gap = float(mi - wi)                      # >0 => mutant presents more confidently than WT (discriminable)
    D_boltz = float(np.clip(gap / 0.30, 0.0, 1.0))   # 0.30 iplddt gap => fully discriminable (a-priori scale)
    return {"D_boltz": round(D_boltz, 3), "presents_mut": presents_mut,
            "mut_iplddt": round(float(mi), 3), "wt_iplddt": round(float(wi), 3),
            "iplddt_gap": round(gap, 3),
            "mut_iptm": round(float(mut_conf.get("iptm") or 0), 3),
            "wt_iptm": round(float(wt_conf.get("iptm") or 0), 3)}


def three_way(D_boltz, presents_mut, D_esmfold, mhcflurry_M):
    """Agreement across Boltz / ESMFold / MHCflurry. Each 'discriminates' if its signal >= 0.5."""
    votes = {"boltz": (D_boltz is not None and D_boltz >= 0.5),
             "esmfold": (D_esmfold is not None and D_esmfold >= 0.5),
             "mhcflurry": (mhcflurry_M is not None and mhcflurry_M >= 0.5)}
    n_yes = sum(votes.values())
    if presents_mut is False:
        verdict = "FAIL_no_presentation"          # mutant doesn't even present -> not a usable handle
    elif n_yes == 3:
        verdict = "CERTIFIED_3of3"
    elif n_yes == 2:
        verdict = "SUPPORTED_2of3"
    elif n_yes == 1:
        verdict = "FLAGGED_1of3"
    else:
        verdict = "REJECTED_0of3"
    return {"votes": votes, "n_methods_discriminating": n_yes, "verdict": verdict}


def mhcflurry_M(wt_rank, mut_rank):
    """MHC-level discrimination proxy from RUNG-11 ranks: WT poorly presented while mut presents well."""
    if wt_rank is None or mut_rank is None:
        return None
    # WT 'not presented' if rank>2 (weak), mut 'presented' if rank<2 (strong); scale smoothly
    return float(np.clip((min(wt_rank, 20.0) - 2.0) / 8.0, 0.0, 1.0)) if mut_rank <= 2.0 else 0.0


# ---------------------------------------------------------------------------
def prepare() -> list[dict]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, by_id, z_of = load_handles()
    picked = select_handles(manifest, z_of)
    r12 = {t["id"]: t for t in json.load(open(RUNG12))["ranked_targets"]} if RUNG12.exists() else {}
    targets = []
    for h in picked:
        for kind, pep in (("mut", h["pep_mut"]), ("wt", h["pep_wt"])):
            name = f"{h['id']}_{kind}"
            write_pmhc_yaml(INPUT_DIR / f"{name}.yaml", h["groove"], pep, name)
        targets.append({"id": h["id"], "gene": h["gene"], "mut_label": h["mut_label"], "allele": h["allele"],
                        "pep_mut": h["pep_mut"], "pep_wt": h["pep_wt"], "p_in_pep": h.get("p_in_pep"),
                        "wt_rank": h.get("wt_rank"), "mut_rank": h.get("mut_rank"),
                        "max_prev": h.get("max_prev"),
                        "esmfold_D": r12.get(h["id"], {}).get("D"),
                        "esmfold_per_cell_safe": r12.get(h["id"], {}).get("per_cell_safe")})
    (OUT_DIR / "targets.json").write_text(json.dumps(targets, indent=2))
    print(f"[rung20] prepared {len(targets)} handles ({2*len(targets)} Boltz jobs) -> {INPUT_DIR}")
    for t in targets:
        print(f"  {t['id']:24} {t['gene']}-{t['mut_label']:7} {t['allele']:12} "
              f"mut={t['pep_mut']} wt={t['pep_wt']} ESMfold_D={t['esmfold_D']}")
    return targets


def main_run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not have_boltz():
        print("[rung20] boltz not installed. On Colab GPU: pip install boltz ; then re-run. "
              "(prepare/selftest work without it.)")
        return 3
    targets = prepare()
    results = []
    for ti, t in enumerate(targets):
        rec = dict(t)
        confs = {}
        for kind in ("mut", "wt"):
            name = f"{t['id']}_{kind}"
            pred = PRED_DIR / name
            if read_confidence(pred) is None:
                print(f"[rung20] [{ti+1}/{len(targets)}] folding {name} ...", flush=True)
                run_boltz(INPUT_DIR / f"{name}.yaml", pred)
            confs[kind] = read_confidence(pred) or {}
        disc = boltz_discrimination(confs["mut"], confs["wt"])
        Mf = mhcflurry_M(t.get("wt_rank"), t.get("mut_rank"))
        agree = three_way(disc.get("D_boltz"), disc.get("presents_mut"), t.get("esmfold_D"), Mf)
        rec.update({"boltz": disc, "mhcflurry_M": None if Mf is None else round(Mf, 3), "agreement": agree})
        results.append(rec)
        print(f"  {t['id']:24} D_boltz={disc.get('D_boltz')} presents_mut={disc.get('presents_mut')} "
              f"-> {agree['verdict']}")

    n_cert = sum(r["agreement"]["verdict"].startswith("CERTIFIED") for r in results)
    n_supp = sum(r["agreement"]["verdict"].startswith("SUPPORTED") for r in results)
    n_flag = sum(r["agreement"]["verdict"].startswith(("FLAGGED", "REJECTED", "FAIL")) for r in results)
    result = {
        "tag": "rung20_boltz_specificity",
        "axis": "BINDING (Shriya stage 2) — structural mut-vs-WT presentation discrimination, independent model",
        "model": "Boltz-2 (co-folding; interface pLDDT / ipTM), diffusion_samples=%d" % DIFFUSION_SAMPLES,
        "iplddt_min": IPLDDT_MIN,
        "n_handles": len(results),
        "results": results,
        "summary": {"certified_3of3": n_cert, "supported_2of3": n_supp, "flagged_or_failed": n_flag},
        "HEADLINE": f"Boltz-2 (independent SOTA) cross-checked {len(results)} top clean handles: "
                    f"{n_cert} CERTIFIED (3/3 methods agree mut presents & self-WT doesn't), {n_supp} SUPPORTED "
                    f"(2/3), {n_flag} flagged/failed. Agreement RAISES confidence in the RUNG-11/12 calls; "
                    f"disagreement marks model-dependent handles for wet validation first.",
        "INTERPRETATION_MAP": {
            "CERTIFIED_3of3": "Boltz + ESMFold + MHCflurry all agree mut presents & WT doesn't -> highest-"
                              "confidence in-silico binding-axis pass; top of the TCR-discovery shortlist.",
            "SUPPORTED_2of3": "two independent methods agree -> credible; the dissenting method's blind spot "
                              "is noted (MHCflurry=sequence-only, ESMFold=single-seq docking, Boltz=1 sample).",
            "FLAGGED/REJECTED/FAIL": "Boltz disagrees or mut doesn't present structurally -> do NOT advance "
                                     "on prediction alone; needs wet pMHC stability / TCR data.",
        },
        "CEILING": "Boltz confidence = structural plausibility, NOT measured affinity (affinity head is small-"
                   "molecule only). pMHC-I peptide docking is hard for any model; 1 diffusion sample for cost. "
                   "MUT-vs-WT (self-tolerance) only, NOT proteome-wide mimicry (the MAGE-A3 failure mode is a "
                   "separate test). A third in-silico opinion that raises/lowers confidence; NOT a wet result.",
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\n[rung20] wrote {RESULT_JSON}")
    print(f"[rung20] {n_cert} certified / {n_supp} supported / {n_flag} flagged-or-failed of {len(results)}")
    _make_figure(results)
    return 0


def _make_figure(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung20] matplotlib unavailable ({e}); skipped figure"); return
    rows = [r for r in results if r.get("boltz", {}).get("D_boltz") is not None]
    if not rows:
        print("[rung20] no folded handles -> no figure"); return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    names = [f"{r['gene']}-{r['mut_label']}\n{r['allele']}" for r in rows]
    y = np.arange(len(rows))
    mut = [r["boltz"]["mut_iplddt"] for r in rows]
    wt = [r["boltz"]["wt_iplddt"] for r in rows]
    ax[0].barh(y - 0.2, mut, 0.4, label="MUTANT pMHC", color="#B23A2E")
    ax[0].barh(y + 0.2, wt, 0.4, label="wild-type pMHC", color="#3F7D54")
    ax[0].axvline(IPLDDT_MIN, ls="--", color="grey", label=f"credible interface {IPLDDT_MIN}")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=7.5); ax[0].invert_yaxis()
    ax[0].set_xlabel("Boltz interface pLDDT"); ax[0].legend(fontsize=8); ax[0].grid(axis="x", alpha=0.3)
    ax[0].set_title("Boltz-2: does MUTANT present while self-WT doesn't?")
    # cross-check: Boltz D vs ESMFold D
    bd = [r["boltz"]["D_boltz"] for r in rows]
    ed = [(r.get("esmfold_D") if r.get("esmfold_D") is not None else np.nan) for r in rows]
    cols = {"CERTIFIED": "#3F7D54", "SUPPORTED": "#E0A040"}
    for r, x_, y_ in zip(rows, ed, bd):
        v = r["agreement"]["verdict"]
        c = next((cc for k, cc in cols.items() if v.startswith(k)), "#B23A2E")
        ax[1].scatter(x_, y_, color=c, s=60)
        ax[1].annotate(r["gene"], (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax[1].set_xlabel("ESMFold D (RUNG-12)"); ax[1].set_ylabel("Boltz D (RUNG-20)")
    ax[1].set_title("Independent agreement (green=3/3, amber=2/3, red=flagged)")
    ax[1].grid(alpha=0.3); ax[1].set_xlim(-0.05, 1.05); ax[1].set_ylim(-0.05, 1.05)
    fig.suptitle("RUNG-20 (Boltz 2b): structural binding-axis cross-check of the top clean handles", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PNG, dpi=130)
    print(f"[rung20] wrote {FIGURE_PNG}")


# ---------------------------------------------------------------------------
def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # selection + manifest available
    if MANIFEST.exists():
        manifest, by_id, z_of = load_handles()
        picked = select_handles(manifest, z_of, n=6)
        check("select returns clean handles only", all(h.get("tier") == "clean" for h in picked))
        check("select returns up to N with mut+wt peptides", 1 <= len(picked) <= 6 and all(h["pep_mut"] and h["pep_wt"] for h in picked))
        # mut and WT differ at exactly p_in_pep
        h0 = picked[0]; p = h0.get("p_in_pep")
        if p:
            diffs = [i for i, (a, b) in enumerate(zip(h0["pep_mut"], h0["pep_wt"]), 1) if a != b]
            check("mut vs WT differ at exactly the mutated position", diffs == [p])
        # YAML writer produces 2 chains with the right sequences
        tmp = OUT_DIR / "_selftest.yaml"
        write_pmhc_yaml(tmp, "GROOVESEQ", "PEPTIDE", "t")
        txt = tmp.read_text(); tmp.unlink()
        check("YAML has groove (A) + peptide (B)", "id: A" in txt and "GROOVESEQ" in txt and "id: B" in txt and "PEPTIDE" in txt)
    else:
        check("manifest present (RUNG-12 must have run)", False)

    # boltz_discrimination metric
    d_disc = boltz_discrimination({"complex_iplddt": 0.85, "iptm": 0.8}, {"complex_iplddt": 0.45, "iptm": 0.4})
    check("mut presents, WT doesn't => D_boltz high + presents_mut True",
          d_disc["D_boltz"] >= 0.9 and d_disc["presents_mut"] is True)
    d_same = boltz_discrimination({"complex_iplddt": 0.85, "iptm": 0.8}, {"complex_iplddt": 0.84, "iptm": 0.8})
    check("mut and WT both present => D_boltz low (not discriminable)", d_same["D_boltz"] < 0.1)
    d_nopres = boltz_discrimination({"complex_iplddt": 0.40, "iptm": 0.3}, {"complex_iplddt": 0.38, "iptm": 0.3})
    check("mut below credible interface => presents_mut False", d_nopres["presents_mut"] is False)
    check("missing iplddt => None", boltz_discrimination({}, {})["D_boltz"] is None)

    # mhcflurry_M proxy
    check("clean (mut strong, WT weak) => M high", mhcflurry_M(10.0, 0.1) >= 0.9)
    check("WT also strong => M ~ 0", mhcflurry_M(0.5, 0.1) < 0.2)
    check("mut not presented (rank>2) => M 0", mhcflurry_M(20.0, 5.0) == 0.0)

    # three-way agreement
    a3 = three_way(0.9, True, 0.9, 0.9)
    check("3/3 discriminate + presents => CERTIFIED_3of3", a3["verdict"] == "CERTIFIED_3of3")
    a2 = three_way(0.9, True, 0.9, 0.1)
    check("2/3 => SUPPORTED_2of3", a2["verdict"] == "SUPPORTED_2of3")
    afail = three_way(0.9, False, 0.9, 0.9)
    check("mut doesn't present => FAIL regardless of votes", afail["verdict"] == "FAIL_no_presentation")
    a1 = three_way(0.9, True, 0.1, 0.1)
    check("1/3 => FLAGGED_1of3", a1["verdict"] == "FLAGGED_1of3")

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(selftest())
    elif cmd == "prepare":
        prepare(); sys.exit(0)
    elif cmd == "run":
        sys.exit(main_run())
    print(f"unknown command: {cmd} (use selftest|prepare|run)"); sys.exit(64)
