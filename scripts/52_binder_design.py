#!/usr/bin/env python3
"""
RUNG 26 — DESIGN THE KEY: de novo MUTANT-SPECIFIC binder design for a clean neoantigen-pMHC.
(4-hour GPU campaign: RFdiffusion → ProteinMPNN → AlphaFold2 validation. The molecular-invention frontier.)

WHY — we mapped the lock, now design the key
---------------------------------------------
The whole recognition arc converged: the MUTATION (neoantigen) on MHC is the only tumour-exclusive signal
(RUNG-5/11/15/23), it presents structurally (RUNG-12/20), and safety↔immunogenicity align (RUNG-17). What we
DON'T have is the actual recognition MOLECULE. This is the GPU swing that designs it: a de novo mini-binder
that grips the MUTANT peptide-MHC and NOT the wild-type — the structural discrimination RUNG-20 said we need
(its confidence metric couldn't do it; a designed binder that physically contacts the mutated residue can).

THE TARGET
----------
IDH1-R132H / HLA-A*01:01 (clean glioma neoantigen; RUNG-12 per-cell-safe, RUNG-20 presentation-confirmed).
mut peptide IIG[H]HAYGDQY vs WT IIG[R]HAYGDQY — the mutated residue is peptide position 4 (R132->H). A binder
designed with a HOTSPOT on that residue must read the mutation -> mutant-specific by construction.

PIPELINE (GPU, in the notebook; this script = the selftestable orchestration + scoring core)
--------------------------------------------------------------------------------------------
  1. fold the MUT pMHC target (ColabFold/AF2) -> target PDB (chain A=MHC groove, chain B=peptide)
  2. RFdiffusion: generate binder backbones against the pMHC, HOTSPOT = mutated peptide residue
  3. ProteinMPNN: design sequences for each backbone
  4. AF2 (ColabFold, initial-guess): fold binder+MUT-pMHC -> pae_interaction + binder pLDDT (is it a binder?)
  5. for the binders, fold binder+WT-pMHC -> discrimination = pae_wt − pae_mut (mutant-specific?)
  6. rank: specific binders first (this script)

DISCIPLINE (rule 5 — never burn 4 GPU-h on a broken install)
------------------------------------------------------------
The notebook SMOKE-TESTS one full design end-to-end before the batch, the batch is TIME-BOXED and RESUMABLE
(every design's metrics saved), so a crash/timeout never wastes the run — partial results still rank.

HONEST CEILING
--------------
Designs are IN-SILICO predictions (AF2 pae_interaction is the field-standard de novo binder filter, ~good but
not truth); a real binder needs wet-lab expression + SPR/affinity + the pMHC-on-cell context. pMHC binder
design is HARD (small, flat epitope); most designs fail — that's expected, we keep the few that pass. Free
T4 completes fewer designs than an A100 (resumable -> still useful). NOT a validated molecule.

USAGE
  python scripts/52_binder_design.py selftest       # mock metrics — validates spec, scoring, ranking, resume
  python scripts/52_binder_design.py spec <id>      # print the design spec (target, hotspot) for a target
  # the GPU pipeline runs in notebooks/binder_design_colab.ipynb and calls `rank` on the saved metrics:
  python scripts/52_binder_design.py rank <designs_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "runs" / "rung12_pmhc" / "rung12_manifest.json"
OUT_DIR = PROJECT_ROOT / "runs" / "rung26_binder_design"

# binder/discrimination filters (a-priori, field-standard for AF2 de novo binder validation)
PAE_BIND = 10.0          # pae_interaction <= this on MUT pMHC => credible binder (Bennett et al. 2023)
PLDDT_MIN = 80.0         # binder-chain pLDDT >= this => well-folded design
DISC_MIN = 3.0           # (pae_WT − pae_MUT) >= this => mutant-SPECIFIC (binds mut, not WT)

DEFAULT_TARGETS = ["IDH1_R132H_A0101", "BRAF_V600E_A0101"]   # clean, presentation-confirmed


def load_targets(ids=None):
    handles = {h["id"]: h for h in json.load(open(MANIFEST))["handles"]}
    ids = ids or DEFAULT_TARGETS
    return [handles[i] for i in ids if i in handles]


def design_spec(target: dict, binder_len=(60, 100)) -> dict:
    """RFdiffusion spec: pMHC target (chain A=MHC groove, B=peptide) + HOTSPOT on the mutated peptide residue
    (and its immediate neighbours) so every binder is forced to contact the mutation."""
    p = target["p_in_pep"]                                   # 1-based mutated position in the peptide
    pep = target["pep_mut"]
    hot = sorted({p_ for p_ in (p - 1, p, p + 1) if 1 <= p_ <= len(pep)})
    return {
        "id": target["id"], "gene": target["gene"], "mut_label": target["mut_label"], "allele": target["allele"],
        "pep_mut": target["pep_mut"], "pep_wt": target["pep_wt"], "mhc_groove": target.get("groove", ""),
        "mut_residue_in_peptide": p, "mut_aa": target["mut_aa"], "wt_aa": target["wt_aa"],
        "hotspot_peptide_residues": [f"B{r}" for r in hot],  # chain B (peptide) hotspot for RFdiffusion
        "binder_len_range": list(binder_len),
        "note": "design binders to chain A(MHC)+B(peptide); hotspot forces contact with the mutated residue.",
    }


def score_design(mut_metrics: dict, wt_metrics: dict | None) -> dict:
    """From AF2 validation metrics: is it a binder (to MUT pMHC)? is it mutant-SPECIFIC (vs WT)?"""
    mut_pae = mut_metrics.get("pae_interaction")
    binder_plddt = mut_metrics.get("binder_plddt")
    is_binder = (mut_pae is not None and mut_pae <= PAE_BIND and
                 binder_plddt is not None and binder_plddt >= PLDDT_MIN)
    discrimination = None
    is_specific = False
    if wt_metrics is not None and wt_metrics.get("pae_interaction") is not None and mut_pae is not None:
        discrimination = round(float(wt_metrics["pae_interaction"]) - float(mut_pae), 3)
        is_specific = bool(is_binder and discrimination >= DISC_MIN)
    return {"is_binder": bool(is_binder), "is_mutant_specific": is_specific,
            "discrimination_pae_wt_minus_mut": discrimination,
            "mut_pae_interaction": mut_pae, "binder_plddt": binder_plddt,
            "wt_pae_interaction": (wt_metrics or {}).get("pae_interaction")}


def rank_designs(designs_dir: Path) -> list[dict]:
    """Load every saved design metrics.json under designs_dir, score, and rank (specific binders first)."""
    out = []
    for mj in sorted(Path(designs_dir).rglob("metrics.json")):
        try:
            d = json.load(open(mj))
        except Exception:
            continue
        s = score_design(d.get("mut", {}), d.get("wt"))
        out.append({"design": d.get("design_id", mj.parent.name), "target": d.get("target"),
                    "sequence": d.get("sequence"), **s})
    out.sort(key=lambda r: (r["is_mutant_specific"], r["is_binder"],
                            (r["discrimination_pae_wt_minus_mut"] or -1e9),
                            -(r["mut_pae_interaction"] if r["mut_pae_interaction"] is not None else 1e9)),
             reverse=True)
    return out


def already_done(designs_dir: Path, design_id: str) -> bool:
    """Resumability: a design with saved metrics is skipped on re-run."""
    return (Path(designs_dir) / design_id / "metrics.json").exists()


# ---------------------------------------------------------------------------
def _cmd_spec(argv):
    ids = [argv[2]] if len(argv) > 2 else None
    for t in load_targets(ids):
        print(json.dumps(design_spec(t), indent=2))


def _cmd_rank(argv):
    d = Path(argv[2]) if len(argv) > 2 else OUT_DIR
    ranked = rank_designs(d)
    n_spec = sum(r["is_mutant_specific"] for r in ranked)
    n_bind = sum(r["is_binder"] for r in ranked)
    result = {"tag": "rung26_binder_design", "n_designs": len(ranked),
              "n_binders": n_bind, "n_mutant_specific": n_spec,
              "filters": {"pae_bind": PAE_BIND, "binder_plddt_min": PLDDT_MIN, "discrimination_min": DISC_MIN},
              "top_designs": ranked[:25],
              "HEADLINE": (f"{len(ranked)} designs evaluated → {n_bind} credible binders to the MUT pMHC → "
                           f"{n_spec} MUTANT-SPECIFIC (bind mut, not WT). Top = the candidate recognition "
                           f"molecule for the clean neoantigen (in-silico; wet-lab validation = the residual)."),
              "CEILING": "AF2 pae_interaction is the field-standard de novo binder filter (good, not truth); "
                         "pMHC epitopes are small/flat (most designs fail — expected); wet-lab expression + "
                         "SPR + on-cell pMHC context required. NOT a validated molecule."}
    (Path(d)).mkdir(parents=True, exist_ok=True)
    (Path(d) / "rung26_binder_design.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["HEADLINE"], indent=2))
    for r in ranked[:10]:
        print(f"  {r['design']:24} binder={r['is_binder']} specific={r['is_mutant_specific']} "
              f"disc={r['discrimination_pae_wt_minus_mut']} mut_pae={r['mut_pae_interaction']} plddt={r['binder_plddt']}")
    return result


def selftest() -> int:
    checks, ok = [], 0

    def check(name, cond):
        nonlocal ok
        checks.append((name, bool(cond))); ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    if not MANIFEST.exists():
        check("rung12_manifest present", False); print("\n  selftest: 0 — manifest missing"); return 1
    targets = load_targets()
    check("targets load (clean, presentation-confirmed)", len(targets) >= 1)
    spec = design_spec(targets[0])
    # hotspot must include the mutated peptide residue
    p = targets[0]["p_in_pep"]
    check("hotspot includes the mutated residue", f"B{p}" in spec["hotspot_peptide_residues"])
    check("spec carries mut+WT peptides", spec["pep_mut"] and spec["pep_wt"] and spec["pep_mut"] != spec["pep_wt"])

    # scoring: binder + specific
    s1 = score_design({"pae_interaction": 7.0, "binder_plddt": 88}, {"pae_interaction": 18.0})
    check("good binder, WT weak => mutant-specific", s1["is_binder"] and s1["is_mutant_specific"] and s1["discrimination_pae_wt_minus_mut"] == 11.0)
    # binder but NOT specific (binds WT too)
    s2 = score_design({"pae_interaction": 7.0, "binder_plddt": 88}, {"pae_interaction": 7.5})
    check("binds both => binder but NOT specific", s2["is_binder"] and not s2["is_mutant_specific"])
    # not a binder (high pae)
    s3 = score_design({"pae_interaction": 20.0, "binder_plddt": 88}, {"pae_interaction": 30.0})
    check("high mut pae => not a binder", not s3["is_binder"] and not s3["is_mutant_specific"])
    # not a binder (low plddt)
    s4 = score_design({"pae_interaction": 7.0, "binder_plddt": 60}, {"pae_interaction": 18.0})
    check("low binder pLDDT => not a binder", not s4["is_binder"])
    # no WT yet (mut-only fold): binder flagged, specificity unknown
    s5 = score_design({"pae_interaction": 7.0, "binder_plddt": 88}, None)
    check("WT not folded yet => binder True, specific False, disc None", s5["is_binder"] and not s5["is_mutant_specific"] and s5["discrimination_pae_wt_minus_mut"] is None)

    # ranking + resume (write 3 mock designs to a temp dir)
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    def write(did, mut_pae, plddt, wt_pae):
        dd = tmp / did; dd.mkdir(parents=True)
        json.dump({"design_id": did, "target": "T", "sequence": "AAAA",
                   "mut": {"pae_interaction": mut_pae, "binder_plddt": plddt}, "wt": {"pae_interaction": wt_pae}},
                  open(dd / "metrics.json", "w"))
    write("d_specific", 6.0, 90, 20.0)     # specific
    write("d_binder", 6.0, 90, 7.0)        # binder not specific
    write("d_fail", 25.0, 90, 30.0)        # not a binder
    ranked = rank_designs(tmp)
    check("rank: specific design ranks first", ranked[0]["design"] == "d_specific" and ranked[0]["is_mutant_specific"])
    check("rank: failure ranks last", ranked[-1]["design"] == "d_fail")
    check("resume: done design detected", already_done(tmp, "d_specific") and not already_done(tmp, "d_missing"))

    print(f"\n  selftest: {ok}/{len(checks)} passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(selftest())
    if cmd == "spec":
        _cmd_spec(sys.argv); sys.exit(0)
    if cmd == "rank":
        _cmd_rank(sys.argv); sys.exit(0)
    print(f"unknown: {cmd} (selftest|spec|rank)"); sys.exit(64)
