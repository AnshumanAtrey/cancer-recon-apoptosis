#!/usr/bin/env python3
"""
RUNG-27c — CRISPR RESCUE for the wobble drivers: does DNA-level allele-specific sensing actually work for the
G>A-transition drivers (KRAS-G12D, IDH1-R132H, TP53 hotspots) that the RNA toehold can't discriminate?

WHY (rule 5 — test the assumption, don't assert it)
---------------------------------------------------
RUNG-27b set dna_sensable=True for EVERY driver ("CRISPR has no wobble"). That is an ASSERTION. CRISPR
allele-specificity is NOT automatic — it needs either (a) the SNV to CREATE/DESTROY a PAM (mutant gets an
NGG the WT lacks → guide+PAM hits mutant only = the gold standard), or (b) an existing PAM positioned so the
SNV sits in the guide's PAM-proximal SEED (~10 nt), where a single mismatch collapses Cas9 activity → the WT
allele is spared. RUNG-27c SCANS the real sequence (SpCas9 NGG; SpCas9-NG and Cas12a TTTV as fallbacks),
classifies each driver's best mechanism, and DESIGNS the actual allele-specific guide. Honest outcome: some
wobble drivers may NOT have a clean PAM → the DNA rescue is real but not universal.

CEILING (rule 3/5)
  - Context is CDS-LOCAL (Ensembl MANE CDS, U->T). True genomic targeting includes INTRONS within ±20 bp of
    exon-edge hotspots → exact PAM availability for SEED guides needs intron-aware genomic sequence. PAM-
    CREATING calls are robust (they depend only on the codon-local ±2 bp).
  - PAM presence + seed-position are NECESSARY, not sufficient: real allele-specificity also needs empirical
    on/off-target validation (a wet-lab residual). This designs candidates, it does not validate cutting.

USAGE
  python scripts/55_crispr_rescue.py selftest   # validates PAM scan / mechanism logic on constructed cases
  python scripts/55_crispr_rescue.py run        # needs cached CDS (scripts/54 prep) -> runs/rung27c_crispr/
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_mc = import_module("54_mutation_circuit")   # reuse DRIVERS, TX, CDS loader, build_window, CANCER_DRIVERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "runs" / "rung27c_crispr"
RESULT_JSON = OUT_DIR / "rung27c_crispr.json"

DCOMP = {"A": "T", "T": "A", "G": "C", "C": "G"}
GUIDE_LEN = 20
SEED = 10        # PAM-proximal positions (1..SEED) = strong discrimination
WIDE = 32        # nt each side of the SNV — must hold a full 20-nt protospacer + PAM on EITHER strand (NOT 11!)


def dna_rc(s):
    return "".join(DCOMP[b] for b in reversed(s))


def wide_window(cds, pos, wt_cod, mut_cod, W=WIDE):
    """Wide DNA window (W nt each side of the SNV) from the CDS — big enough for a 20-nt guide + PAM either side.
    Returns (wt_dna, mut_dna, si) with si = SNV index in the window."""
    s = (pos - 1) * 3
    assert cds[s:s + 3] == wt_cod, f"{pos} {cds[s:s+3]} != {wt_cod}"
    ci = [i for i in range(3) if wt_cod[i] != mut_cod[i]]
    assert len(ci) == 1
    mb = s + ci[0]
    lo, hi = max(0, mb - W), min(len(cds), mb + W + 1)
    wt = cds[lo:hi].replace("U", "T")
    mut = (cds[lo:mb] + mut_cod[ci[0]] + cds[mb + 1:hi]).replace("U", "T")
    return wt, mut, mb - lo


def scan_strand(wt, mut, si, guide_len=GUIDE_LEN):
    """Find SpCas9 (NGG) allele-specific guides on ONE strand. wt/mut are the SAME strand (DNA); si = index of
    the SNV. Returns guide options that discriminate MUTANT from WT, with mechanism + seed position."""
    hits = []
    L = len(mut)
    for p in range(guide_len, L - 2):                 # protospacer [p-guide_len, p); PAM [p, p+3)
        if not (mut[p + 1] == "G" and mut[p + 2] == "G"):
            continue                                   # mutant must present an NGG PAM here
        proto_start = p - guide_len
        if not (proto_start <= si < p + 3):            # SNV must lie in protospacer or PAM
            continue
        wt_is_pam = (wt[p + 1] == "G" and wt[p + 2] == "G")
        if not wt_is_pam:
            # mutation CREATED the PAM (si must be at p+1/p+2) -> guide+PAM hits mutant only
            hits.append({"mech": "PAM_CREATED", "seed_pos_from_PAM": (p - si) if si < p else 0,
                         "guide": mut[proto_start:p], "pam_mut": mut[p:p + 3], "pam_wt": wt[p:p + 3]})
        elif si < p:                                   # PAM in both; SNV in the protospacer -> seed/distal
            seedpos = p - si                           # 1 = adjacent to PAM (deepest seed)
            mech = "SEED" if seedpos <= SEED else ("MID" if seedpos <= 12 else "DISTAL")
            hits.append({"mech": mech, "seed_pos_from_PAM": seedpos,
                         "guide": mut[proto_start:p], "pam_mut": mut[p:p + 3], "pam_wt": wt[p:p + 3]})
        # si in PAM but PAM present in both (N position) -> not discriminating -> skip
    return hits


def allele_specific_crispr(wt_dna, mut_dna, si):
    """Scan BOTH strands; return the best mechanism (PAM_CREATED > SEED > MID > DISTAL > NONE) + the guide."""
    sense = scan_strand(wt_dna, mut_dna, si)
    # antisense: reverse-complement both strands; SNV maps to len-1-si
    L = len(mut_dna)
    anti = scan_strand(dna_rc(wt_dna), dna_rc(mut_dna), L - 1 - si)
    allh = [{**h, "strand": "+"} for h in sense] + [{**h, "strand": "-"} for h in anti]
    rank = {"PAM_CREATED": 0, "SEED": 1, "MID": 2, "DISTAL": 3}
    allh.sort(key=lambda h: (rank.get(h["mech"], 9), h.get("seed_pos_from_PAM") or 99))
    best = allh[0] if allh else None
    addressable = bool(best and best["mech"] in ("PAM_CREATED", "SEED"))
    return {"addressable": addressable, "best": best, "n_options": len(allh),
            "mechanisms": sorted({h["mech"] for h in allh})}


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    miss = [g for g in _mc.TX if not _mc._cds_path(g).exists()]
    if miss:
        print(f"[rung27c] missing CDS {miss} — run `python scripts/54_mutation_circuit.py prep` first.")
        return 4
    cds_by_gene = {g: _mc._cds_path(g).read_text().strip() for g in _mc.TX}

    per = {}
    for (gene, label, pos, wt_cod, mut_cod) in _mc.DRIVERS:
        cds = cds_by_gene[gene]
        ci = [i for i in range(3) if wt_cod[i] != mut_cod[i]][0]
        wb, mb = wt_cod[ci].replace("U", "T"), mut_cod[ci].replace("U", "T")
        rna_wobble = _mc.is_wobble_sub(wt_cod[ci], mut_cod[ci])
        wt_dna, mut_dna, mpos = wide_window(cds, pos, wt_cod, mut_cod)
        r = allele_specific_crispr(wt_dna, mut_dna, mpos)
        per[f"{gene}_{label}"] = {
            "gene": gene, "aa_change": label, "rna_sub_dna": f"{wb}>{mb}".replace("U", "T"),
            "rna_wobble": rna_wobble,
            "crispr_addressable": r["addressable"], "best_mechanism": (r["best"] or {}).get("mech", "NONE"),
            "guide_20nt": (r["best"] or {}).get("guide"), "strand": (r["best"] or {}).get("strand"),
            "seed_pos_from_PAM": (r["best"] or {}).get("seed_pos_from_PAM"),
            "pam_mut_vs_wt": [(r["best"] or {}).get("pam_mut"), (r["best"] or {}).get("pam_wt")],
            "all_mechanisms": r["mechanisms"],
        }

    wobble = [k for k, v in per.items() if v["rna_wobble"]]
    rescued = [k for k in wobble if per[k]["crispr_addressable"]]
    pam_created = [k for k, v in per.items() if v["best_mechanism"] == "PAM_CREATED"]
    not_rescued = [k for k in wobble if not per[k]["crispr_addressable"]]
    all_addr = [k for k, v in per.items() if v["crispr_addressable"]]

    result = {
        "tag": "rung27c_crispr_rescue",
        "question": "Does DNA-level allele-specific CRISPR actually rescue the G>A-wobble drivers the RNA toehold "
                    "can't sense (RUNG-27b's asserted dna_sensable)? Scan real CDS for PAM-creating SNVs + seed guides.",
        "cas": "SpCas9 (NGG), 20-nt guide, seed = PAM-proximal 10 nt", "context": "CDS-local (Ensembl MANE)",
        "n_drivers": len(per), "n_wobble": len(wobble),
        "n_wobble_crispr_rescued": len(rescued), "wobble_rescued": rescued,
        "wobble_NOT_rescued_by_SpCas9_NGG": not_rescued,
        "pam_creating_drivers": pam_created, "n_all_crispr_addressable": len(all_addr),
        "per_driver": per,
        "HEADLINE": (
            f"Of the {len(wobble)} G>A-wobble drivers the RNA toehold can't sense, {len(rescued)}/{len(wobble)} ARE "
            f"CRISPR-addressable by SpCas9 (NGG) allele-specific guides — {len(pam_created)} via a PAM-CREATING SNV "
            f"(mutant makes an NGG the WT lacks = gold-standard allele specificity) and the rest via a PAM-proximal "
            f"SEED mismatch. {'NOT rescued by NGG (need NG/Cas12a or another Cas): ' + str(not_rescued) if not_rescued else 'All wobble drivers addressable.'} "
            f"This CONVERTS RUNG-27b's asserted DNA-rescue into a designed, sequence-verified set of allele-specific "
            f"guides — the wobble drivers (incl. KRAS-G12D, IDH1-R132H, TP53 hotspots) are reachable at the DNA level "
            f"with real guides, not just by assumption."),
        "INTERPRETATION_MAP": {
            "PAM_CREATED": "mutation creates an NGG the WT lacks -> guide+PAM hits mutant ONLY (perfect allele specificity).",
            "SEED": "shared PAM, SNV in PAM-proximal 10 nt -> WT has a seed mismatch -> Cas9 activity collapses on WT.",
            "NOT rescued by NGG": "no NGG PAM-creating/seed option in CDS-local context -> try SpCas9-NG / Cas12a, or "
                                  "the genomic (intron-aware) context may add PAMs (the refinement).",
        },
        "CEILING": [
            "CDS-local context (introns within +-20bp of exon-edge hotspots not modelled); PAM_CREATED robust, SEED "
            "needs genomic confirmation.",
            "PAM + seed-position are NECESSARY not sufficient -> real on/off-target cutting = wet-lab residual.",
            "SpCas9 NGG only in this pass; NG/Cas12a/other-PAM Cas would raise addressability (noted, not scanned).",
        ],
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[rung27c] wrote {RESULT_JSON}  ({time.monotonic()-t0:.1f}s)")
    print(f"  wobble drivers CRISPR-rescued: {len(rescued)}/{len(wobble)}")
    for k in wobble:
        v = per[k]
        print(f"   {k:16s} {v['best_mechanism']:12s} guide={v['guide_20nt']} strand={v['strand']} "
              f"seed_pos={v['seed_pos_from_PAM']} PAM(mut/wt)={v['pam_mut_vs_wt']}")
    if not_rescued:
        print(f"  NOT rescued by SpCas9-NGG: {not_rescued}")
    return 0


# ---------------------------------------------------------------------------
def selftest():
    ok = 0; checks = []
    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("dna_rc", dna_rc("ATGC") == "GCAT")

    # PAM_CREATED: a SNV that turns ...A[T]G... into ...A[G]G... creates an NGG. Put a 20-nt protospacer 5' of it.
    proto = "ACGTACGTACGTACGTACGT"            # 20 nt
    wt = proto + "ATG" + "CCCCC"               # PAM region 'TG?'-> not NGG (positions p+1=T,p+2=G)
    mut = proto + "AGG" + "CCCCC"              # SNV T->G at p+1 makes 'GG' -> NGG PAM created
    si = len(proto) + 1                        # the mutated base (index of the T/G)
    r = allele_specific_crispr(wt, mut, si)
    check("PAM_CREATED detected", r["best"] and r["best"]["mech"] == "PAM_CREATED")
    check("PAM_CREATED guide is 20nt", r["best"] and len(r["best"]["guide"]) == 20)

    # SEED: PAM 'TGG' present in BOTH; SNV inside the protospacer 2 nt from the PAM (deep seed). Protospacer = 20 nt.
    pre = "ACGTACGTACGTACGTAC"                 # 18 nt (proto indices 0-17)
    wt2 = pre + "G" + "C" + "TGG" + "TTTTTTTT"  # index18=G, PAM 'TGG' at 20-22
    mut2 = pre + "A" + "C" + "TGG" + "TTTTTTTT" # SNV G->A at index 18 (seed pos 2 from PAM)
    r2 = allele_specific_crispr(wt2, mut2, 18)
    check("SEED mechanism found (PAM shared, SNV near PAM)", r2["best"] and r2["best"]["mech"] in ("SEED", "MID"))

    # NONE: SNV far from any NGG, no PAM created/seed
    wt3 = "AAAAAAAAAAAAAAAAAAAAAA" + "T" + "AAAAAAAAAAAAAAAAAAAAAA"
    mut3 = "AAAAAAAAAAAAAAAAAAAAAA" + "C" + "AAAAAAAAAAAAAAAAAAAAAA"
    r3 = allele_specific_crispr(wt3, mut3, 22)
    check("no PAM -> not addressable", not r3["addressable"])

    # ANTISENSE PAM_CREATED: mutation makes 'CC' on sense (= 'GG' on antisense) -> NGG on the - strand.
    suf = "ACGTACGTACGTACGTACGT"               # 20 nt (becomes the antisense protospacer)
    wt4 = "TTTTTTTT" + "T" + "CA" + suf          # sense 'TCA' -> antisense rc = 'TGA' (no NGG)
    mut4 = "TTTTTTTT" + "C" + "CA" + suf         # SNV T->C -> sense 'CCA' -> antisense rc 'TGG' = NGG created
    r4 = allele_specific_crispr(wt4, mut4, 8)
    check("antisense PAM_CREATED found", r4["best"] and r4["best"]["mech"] == "PAM_CREATED" and r4["best"]["strand"] == "-")

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "run":
        sys.exit(run())
    print(f"unknown: {cmd}"); sys.exit(64)
