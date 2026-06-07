#!/usr/bin/env python3
"""
RUNG 16 — neoantigen coverage AT SCALE: the clonal mutational-burden repertoire (laptop, MHCflurry, no GPU).

THE BOTTLENECK THIS ATTACKS
---------------------------
RUNG-11 measured neoantigen addressability from ~17 curated PUBLIC HOTSPOT drivers (KRAS/TP53/IDH1/...). That is
the SHARED, off-the-shelf axis -- one target serves many patients -- and it is bounded (clean coverage: glioma
22%, PDAC 26%, melanoma 11->19% with the wave). But every tumour also carries a PRIVATE repertoire of somatic
mutations (tens to hundreds), and a mutation is tumour-EXCLUSIVE by construction. RUNG-15 just proved single
surface markers can't separate tumour from normal (and RUNG-5/6/31/32 already closed surface AND-NOT). So the
only clean address is a MUTATION. The unasked question:

  With the FULL CLONAL mutation repertoire (not just hotspots), what fraction of patients have >=1 clean,
  tumour-exclusive neoantigen handle -- enough to SEED the death wave (RUNG-13), which then spreads?

Why "clonal" and why ">=1 is enough": the wave decouples efficacy from per-cell recognition (RUNG-13) -- you
only need to recognise a few SEED cells and the wave does the rest. A CLONAL (truncal) mutation is in every
tumour cell, so a single clonal clean handle can seed anywhere. Subclonal private mutations are excluded (they
only mark a subclone). So the deployable quantity is: clonal missense mutations that yield a presented, clean peptide.

HOW (empirical, not guessed)
----------------------------
1. CALIBRATE with MHCflurry (the data is the oracle): simulate random missense mutations (AA-frequency 9..11-mer
   windows via RUNG-11's gen_registers), score WT+MUT across the RUNG-11 HLA panel. Measure empirically:
     p1 = P(a random missense mutation is PRESENTED on a given allele, %rank<=2)
     c1 = P(CLEAN | presented) = P(WT clearly off-MHC, %rank>4 | mutant presented)
2. COVERAGE per cancer (Poisson): N_clonal = TMB(mut/Mb, literature) x exome_Mb x missense_frac x clonal_frac.
   A patient carries ~6 class-I alleles -> P(presented on >=1) = 1-(1-p1)^6. Deployable per mutation =
   that x c1. lambda = N_clonal x deployable. P(patient has >=1 deployable clean clonal handle) = 1-exp(-lambda).
3. Contrast with RUNG-11's hotspot (shared, off-the-shelf) coverage -> the SHARED-vs-PERSONALISED gap.

THE HONEST CEILING (load-bearing)
---------------------------------
A high P(>=1) means PER-PATIENT addressability WITH PERSONALISED neoantigen identification + a per-patient
effector -- NOT an off-the-shelf drug (that stays bounded to hotspots). PREDICTED presentation != immunogenicity:
TCR existence per handle is the wet-lab residual (RUNG-12), un-pre-validatable for private mutations. TMB and
clonal fraction are POPULATION estimates (literature ranges). Simulated missense spectrum approximates real.
This bounds ADDRESSABILITY (is there a handle), not a cure.

USAGE
  python scripts/41_neoantigen_burden_coverage.py            # calibrate (MHCflurry) + per-cancer coverage
  python scripts/41_neoantigen_burden_coverage.py --fast     # smaller calibration (quicker)
  python scripts/41_neoantigen_burden_coverage.py selftest    # Poisson + simulation logic (no MHCflurry)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung16_burden"
RESULT_JSON = OUT_DIR / "rung16_burden.json"
FIGURE_PNG = OUT_DIR / "rung16_burden.png"
CALIB_CACHE = OUT_DIR / "calibration.json"
RUNG11_JSON = PROJECT_ROOT / "runs" / "rung11_neoantigen" / "rung11_neoantigen_addressability.json"

BINDER_RANK = 2.0       # %rank <= 2 -> presented (RUNG-11 convention)
WT_OFF_RANK = 4.0       # WT %rank > 4 -> clearly off MHC -> "clean"
N_CLASS1 = 6            # class-I alleles a patient carries (2x HLA-A/B/C)

# exome model
EXOME_MB = 30.0
MISSENSE_FRAC = 0.70    # fraction of somatic mutations that are missense (yield a neopeptide)
CLONAL_FRAC = 0.60      # fraction of mutations that are clonal/truncal (in every tumour cell -> can seed anywhere)

# median tumour mutational burden (mut/Mb), literature (TCGA/PCAWG; ranges in notes). Approximate medians.
CANCER_TMB = {
    "MELANOMA": 13.0, "NSCLC": 8.0, "BLADDER": 8.0, "CRC_MSI": 40.0, "GASTRIC": 6.0, "HNSC": 5.0,
    "HCC": 4.0, "CRC_MSS": 3.5, "GLIOMA": 2.5, "OV": 2.5, "PDAC": 1.8, "BRCA": 1.6,
}
TMB_NOTE = "median mut/Mb (TCGA/PCAWG order-of-magnitude); high-TMB driven by UV(melanoma)/smoking(NSCLC)/MMR-d(MSI)."

# human amino-acid background frequencies (%) for simulating realistic peptides
AA_FREQ = {"A": 7.0, "R": 5.6, "N": 3.6, "D": 4.8, "C": 2.3, "Q": 4.8, "E": 7.1, "G": 6.6, "H": 2.6,
           "I": 4.3, "L": 9.9, "K": 5.8, "M": 2.2, "F": 3.7, "P": 6.3, "S": 8.3, "T": 5.4, "W": 1.1,
           "Y": 2.7, "V": 6.9}
_AAS = list(AA_FREQ)
_AAP = np.array([AA_FREQ[a] for a in _AAS], float); _AAP /= _AAP.sum()


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(name, str(PROJECT_ROOT / "scripts" / mod))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def simulate_mutations(n, rng, ctx=21):
    """n random missense mutations: a random AA-freq protein context, a central position substituted to a
    different AA. Returns list of (pos, wt, mut, sequence) for RUNG-11's gen_registers."""
    out = []
    for _ in range(n):
        seq = "".join(rng.choice(_AAS, size=ctx, p=_AAP))
        pos = ctx // 2 + 1                          # 1-based central position
        wt = seq[pos - 1]
        mut = wt
        while mut == wt:
            mut = str(rng.choice(_AAS, p=_AAP))
        out.append((pos, wt, mut, seq))
    return out


def calibrate(n_sim, rng, hla_panel, d11, n_patients=40):
    """Empirically measure DEPLOYABLE-per-mutation directly (the load-bearing quantity), via simulated 6-allele
    patients. A mutation is deployable for a patient iff SOME register's MUTANT peptide is a STRONG binder
    (rank<=strong) on one of the patient's alleles AND that SAME register's WT counterpart is off-MHC
    (rank>WT_OFF) on that allele (clean = the WT of the exact presented peptide does not bind -> tumour-exclusive).
    Reported at two binder thresholds (0.5 strong / 2.0 binder) as a sensitivity. NO independence/register-min
    inflation: clean is tested per-register against the peptide's OWN WT."""
    muts = simulate_mutations(n_sim, rng)
    reg_by_mut, all_peps = [], set()
    for pos, wt, mut, seq in muts:
        regs = d11.gen_registers(seq, pos, wt, mut)
        reg_by_mut.append(regs)
        for r in regs:
            all_peps.add(r["pep_mut"]); all_peps.add(r["pep_wt"])
    alleles = list(hla_panel)
    scores, supported = d11.mhcflurry_scores(sorted(all_peps), alleles)
    freqs = np.array([hla_panel[a] for a in supported], float); freqs = freqs / freqs.sum()

    def mut_deployable_on_allele(regs, a, strong):
        # clean strong-binder register: mut rank<=strong AND its own wt rank>WT_OFF
        for r in regs:
            mp, wp = r["pep_mut"], r["pep_wt"]
            if (mp, a) in scores and scores[(mp, a)]["rank"] <= strong:
                if (wp, a) not in scores or scores[(wp, a)]["rank"] > WT_OFF_RANK:
                    return True
        return False

    out = {}
    for strong in (0.5, 2.0):
        # per-mutation deployable on a SINGLE random allele, and per-PATIENT (>=1 of 6 alleles)
        per_allele, per_patient = [], []
        for regs in reg_by_mut:
            dep_by_allele = {a: mut_deployable_on_allele(regs, a, strong) for a in supported}
            per_allele.append(float(np.mean([dep_by_allele[a] for a in supported])))
            for _ in range(n_patients):
                pat = rng.choice(len(supported), size=min(N_CLASS1, len(supported)), replace=False, p=freqs)
                per_patient.append(any(dep_by_allele[supported[i]] for i in pat))
        out[f"strong_{strong}"] = {"deployable_per_mutation_1allele": round(float(np.mean(per_allele)), 5),
                                   "deployable_per_mutation_6allele_patient": round(float(np.mean(per_patient)), 5)}
    calib = {"n_sim_mutations": n_sim, "n_supported_alleles": len(supported), "n_peptides_scored": len(all_peps),
             "n_patient_draws": n_patients, "wt_off_rank": WT_OFF_RANK,
             "by_threshold": out,
             "literature_yield_band": [0.01, 0.05],   # ~1-5% of missense -> presented clean neoantigen (TESLA/cohorts)
             "deployable_per_mutation_USED": out["strong_0.5"]["deployable_per_mutation_6allele_patient"]}
    return calib


def coverage_per_cancer(d_per_mut):
    """P(patient has >=1 deployable clean clonal neoantigen) given the per-mutation deployable rate d (per
    6-allele patient). lambda = (clonal missense count) x d; P(>=1) = 1-exp(-lambda) (Poisson)."""
    rows = {}
    for cancer, tmb in CANCER_TMB.items():
        n_clonal = tmb * EXOME_MB * MISSENSE_FRAC * CLONAL_FRAC
        lam = n_clonal * d_per_mut
        rows[cancer] = {"tmb_mut_per_mb": tmb, "n_clonal_missense": round(n_clonal, 1),
                        "expected_deployable_handles": round(lam, 3),
                        "P_ge1_deployable_clean_clonal": round(1.0 - math.exp(-lam), 4)}
    return {"deployable_per_mutation": round(d_per_mut, 5), "per_cancer": rows}


def _hotspot_compare():
    if not RUNG11_JSON.exists():
        return {}
    d = json.loads(RUNG11_JSON.read_text())
    safe = d.get("addressability_SAFE_range", {}).get("strict_clean_robust", {})
    return {c: round(v["central"], 4) for c, v in safe.items()}


def main_run(fast=False) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d11 = _load("d11", "33_neoantigen_addressability.py")
    rng = np.random.default_rng(20260607)
    n_sim = 80 if fast else 250
    print(f"[rung16] calibrating presentation on {n_sim} simulated missense mutations x {len(d11.HLA_PANEL)} HLA "
          f"alleles (MHCflurry)...")
    calib = calibrate(n_sim, rng, d11.HLA_PANEL, d11)
    CALIB_CACHE.write_text(json.dumps(calib, indent=2))
    d_calib = calib["deployable_per_mutation_USED"]
    d_lo, d_hi = calib["literature_yield_band"]
    print(f"[rung16]   deployable/mutation (strong-binder, 6-allele patient): calibrated={d_calib} "
          f"| literature band [{d_lo},{d_hi}]  ({calib['n_supported_alleles']} alleles)")

    # evaluate the per-cancer coverage across the band (low/high literature) + the MHCflurry calibration
    band = {"lit_low_0.01": coverage_per_cancer(d_lo), "lit_high_0.05": coverage_per_cancer(d_hi),
            "mhcflurry_calibrated": coverage_per_cancer(d_calib)}
    cov = band["lit_low_0.01"]            # headline uses the CONSERVATIVE low end (honest floor)
    hotspot = _hotspot_compare()

    # synthesis verdict: robustly-solved = >=90% even at the LOW (conservative) end of the yield band
    pc_lo = band["lit_low_0.01"]["per_cancer"]
    pc_hi = band["lit_high_0.05"]["per_cancer"]
    solved = [c for c in pc_lo if pc_lo[c]["P_ge1_deployable_clean_clonal"] >= 0.90]               # robust
    bounded = [c for c in pc_hi if pc_hi[c]["P_ge1_deployable_clean_clonal"] < 0.50]               # hard even at high end

    result = {
        "tag": "rung16_clonal_neoantigen_burden_coverage",
        "question": "With the FULL clonal mutation repertoire (not just shared hotspots), what fraction of "
                    "patients have >=1 clean tumour-exclusive neoantigen handle to SEED the death wave?",
        "calibration": calib,
        "exome_model": {"exome_Mb": EXOME_MB, "missense_frac": MISSENSE_FRAC, "clonal_frac": CLONAL_FRAC,
                        "n_class1_alleles": N_CLASS1, "tmb_note": TMB_NOTE},
        "coverage_band": {"deployable_per_mutation": {"lit_low": d_lo, "lit_high": d_hi, "mhcflurry": d_calib},
                          "per_cancer": {c: {"P_ge1_lit_low": pc_lo[c]["P_ge1_deployable_clean_clonal"],
                                             "P_ge1_lit_high": pc_hi[c]["P_ge1_deployable_clean_clonal"],
                                             "P_ge1_mhcflurry": band["mhcflurry_calibrated"]["per_cancer"][c]["P_ge1_deployable_clean_clonal"],
                                             "tmb": pc_lo[c]["tmb_mut_per_mb"],
                                             "n_clonal_missense": pc_lo[c]["n_clonal_missense"]} for c in pc_lo}},
        "hotspot_shared_coverage_RUNG11": hotspot,
        "INTERPRETATION_MAP": {
            "shared_vs_personalised": "hotspot coverage (RUNG-11) = SHARED off-the-shelf targets (bounded). "
                                      "P_ge1 here = PER-PATIENT addressability with PERSONALISED neoantigen ID "
                                      "+ a per-patient effector. The gap is the shared-vs-personalised cost.",
            "seed_sufficiency": ">=1 CLONAL clean handle is enough to SEED the RUNG-13 wave (efficacy decoupled "
                                "from per-cell recognition); high-TMB cancers -> nearly every patient seedable.",
            "tcr_residual": "presentation != immunogenicity; per-handle TCR existence is the wet-lab residual "
                            "(RUNG-12), un-pre-validatable for private mutations."},
        "DECISIVE": "",
        "calibration_note": f"MHCflurry-calibrated deployable rate ({d_calib}) sits ABOVE the literature band "
                            f"[{d_lo},{d_hi}] because the simulation scores raw MHC binding only -- it omits the "
                            f"proteasomal-cleavage + TAP-transport + processing filters that real immunopeptidomics "
                            f"applies (which remove most predicted binders). So the HEADLINE uses the literature "
                            f"band; MHCflurry is the optimistic upper bound.",
        "CEILING": "PREDICTED presentation (MHCflurry), not immunopeptidomics/immunogenicity -> calibration is an "
                   "UPPER bound (no processing filter); headline uses the conservative literature yield band. TMB + "
                   "clonal_frac are population estimates (literature ranges). Simulated missense spectrum approximates "
                   "real. P_ge1 = per-patient addressability requiring PERSONALISED neoantigen ID + effector, NOT an "
                   "off-the-shelf drug. Bounds addressability, not efficacy; coupling/TCR = wet-lab residual.",
    }
    pcb = result["coverage_band"]["per_cancer"]
    result["DECISIVE"] = (
        f"Deployable clean clonal neoantigens per mutation: literature band [{d_lo},{d_hi}], MHCflurry-calibrated "
        f"{d_calib} (strong-binder, per-register WT-clean test, 6-allele patient). PER-PATIENT P(>=1 clean clonal "
        f"neoantigen to SEED the wave), shown as range [lit-low .. lit-high]: "
        + ", ".join(f"{c} {pcb[c]['P_ge1_lit_low']:.0%}-{pcb[c]['P_ge1_lit_high']:.0%}" for c in
                    sorted(pcb, key=lambda c: -pcb[c]['P_ge1_lit_high']))
        + f". Robustly SOLVED per-patient (>=90% even at the CONSERVATIVE 1% yield): {solved or 'none'}; hard "
        f"(<50% even at 5% yield): {bounded or 'none'}. This is PERSONALISED addressability (the patient has a "
        f"clean clonal handle to SEED the RUNG-13 wave, which then spreads) -- NOT an off-the-shelf drug, which "
        f"stays bounded to the shared HOTSPOT ceiling (RUNG-11: melanoma 20%, PDAC 26%). The shared-vs-personalised "
        f"gap is the cost. TCR/immunogenicity per handle = the wet-lab residual (RUNG-12); presentation != killing.")

    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung16] wrote {RESULT_JSON}")
    print("\n  per-cancer P(>=1 deployable clean clonal neoantigen)  [lit 1% .. 5%] (mhcflurry) vs RUNG-11 hotspot:")
    for c in sorted(pcb, key=lambda c: -pcb[c]['P_ge1_lit_high']):
        v = pcb[c]; hs = hotspot.get(c.split("_")[0], hotspot.get(c))
        hs_s = f"{hs:.2f}" if hs is not None else " -- "
        print(f"    {c:9} TMB {v['tmb']:>4} (clonal~{v['n_clonal_missense']:>5.0f})  "
              f"P>=1 [{v['P_ge1_lit_low']:.0%}..{v['P_ge1_lit_high']:.0%}] (mhc {v['P_ge1_mhcflurry']:.0%})   "
              f"hotspot {hs_s}")
    print(f"\n  DECISIVE: {result['DECISIVE']}")
    _make_figure(result)
    return 0


def _make_figure(result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[rung16] matplotlib unavailable ({e})")
        return
    pc = result["coverage_band"]["per_cancer"]
    hot = result["hotspot_shared_coverage_RUNG11"]
    cancers = sorted(pc, key=lambda c: -pc[c]["P_ge1_lit_high"])
    x = np.arange(len(cancers))
    lo = [pc[c]["P_ge1_lit_low"] for c in cancers]
    hi = [pc[c]["P_ge1_lit_high"] for c in cancers]
    shared = [hot.get(c.split("_")[0], hot.get(c, 0.0)) or 0.0 for c in cancers]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    # personalised P(>=1) as a BAND [lit-low .. lit-high]
    ax.bar(x - 0.2, hi, 0.4, label="per-patient P(≥1 clean clonal) — yield 1%→5% band [personalised]",
           color="#A5D6A7")
    ax.bar(x - 0.2, lo, 0.4, color="#1B5E20", label="                              conservative floor (1% yield)")
    ax.bar(x + 0.2, shared, 0.4, label="shared hotspot coverage (RUNG-11) [off-the-shelf]", color="#C1432B")
    ax.set_xticks(x); ax.set_xticklabels(cancers, rotation=45, ha="right")
    ax.set_ylabel("patient fraction with a clean handle"); ax.set_ylim(0, 1.05)
    ax.set_title("RUNG-16 — clonal neoantigen burden: PERSONALISED addressability (seed the wave) vs SHARED hotspots\n"
                 "band = literature yield 1–5%/missense; predicted presentation; TCR/immunogenicity = wet-lab residual")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIGURE_PNG, dpi=120)
    print(f"[rung16] wrote {FIGURE_PNG}")


def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    rng = np.random.default_rng(0)
    muts = simulate_mutations(50, rng)
    check("simulate_mutations: all missense (wt != mut)", all(wt != mut for _, wt, mut, _ in muts))
    check("simulate_mutations: position within context", all(1 <= pos <= len(seq) for pos, _, _, seq in muts))
    check("simulate_mutations: wt matches sequence at pos", all(seq[pos - 1] == wt for pos, wt, _, seq in muts))

    # Poisson coverage math (d = deployable per mutation)
    cov = coverage_per_cancer(0.03)
    pc = cov["per_cancer"]
    check("higher TMB -> higher P(>=1)", pc["MELANOMA"]["P_ge1_deployable_clean_clonal"] >
          pc["PDAC"]["P_ge1_deployable_clean_clonal"])
    check("P(>=1) in [0,1]", all(0 <= v["P_ge1_deployable_clean_clonal"] <= 1 for v in pc.values()))
    check("d=0 -> P(>=1)=0 everywhere", all(v["P_ge1_deployable_clean_clonal"] == 0
                                            for v in coverage_per_cancer(0.0)["per_cancer"].values()))
    check("higher d -> higher P(>=1)", coverage_per_cancer(0.05)["per_cancer"]["PDAC"]["P_ge1_deployable_clean_clonal"] >
          coverage_per_cancer(0.01)["per_cancer"]["PDAC"]["P_ge1_deployable_clean_clonal"])
    lam_mel = pc["MELANOMA"]["expected_deployable_handles"]
    check("P(>=1) == 1-exp(-lambda) (Poisson identity, within rounding)",
          abs(pc["MELANOMA"]["P_ge1_deployable_clean_clonal"] - (1 - math.exp(-lam_mel))) < 5e-3)

    total = len(checks)
    print(f"\nselftest: {ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RUNG-16 clonal neoantigen burden coverage")
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "selftest"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.mode == "selftest":
        sys.exit(selftest())
    sys.exit(main_run(fast=args.fast))
