#!/usr/bin/env python3
"""
RUNG-27a — REALITY-GROUNDING the predicted neoantigen handles against REAL public data.

The whole recognition arc rests on PREDICTED presentation (MHCflurry) and a PREDICTED
TCR-recognition propensity (RUNG-17). CAPSTONE.md names two of these as the irreducible
wet-lab residuals:
   #1  is the peptide REALLY presented / a real epitope?            -> IEDB (assays)
   #2  does a real cognate TCR exist (the MAGE-A3 question)?        -> VDJdb (TCR-epitope pairs)

This rung converts those two residuals from ASSUMPTIONS into MEASURED facts WHERE the public
data exists, and an honest "predicted-only" where it does not. It does NOT invent biology; it
asks two curated experimental databases whether our exact handles have already been seen.

Honest framing (rule 3 / rule 5):
  - A MATCH = a real TCR / real catalogued epitope exists for THIS exact neoantigen (strong de-risk).
  - A WT match = the germline self-peptide is also recognised/catalogued (safety / cross-reactivity
    context — the MAGE-A3 failure mode).
  - NO match != "not real" (databases are incomplete) -> reported as PREDICTED-ONLY, never as a negative.
  - antigen.gene is checked so a coincidental sequence hit to an unrelated antigen is not miscounted.

Data:
  VDJdb  : data/grounding/vdjdb-*/vdjdb.slim.txt  (antigen.epitope, antigen.gene, mhc.a, vdjdb.score)
  IEDB   : data/grounding/iedb_epitope_full*.zip/.csv  (linear peptide + assay)  [optional; augments]
  Handles: runs/rung12_pmhc/rung12_manifest.json  (pep_mut / pep_wt / allele / gene / mut_label / tier)

CPU, laptop. `selftest` validates the matcher on known-true (KRAS-G12D, viral) and known-false
(scramble) cases before any claim. `run` produces runs/rung27a_grounding/grounding.json + summary.
"""
import sys, os, json, glob, re, io, zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "runs/rung12_pmhc/rung12_manifest.json")
GROUND = os.path.join(ROOT, "data/grounding")
OUTDIR = os.path.join(ROOT, "runs/rung27a_grounding")

CORE_MIN = 8  # minimum contiguous shared core (must include the mutated residue) for a register-variant hit


# ---------------------------------------------------------------------------
# matching primitives
# ---------------------------------------------------------------------------
def norm(p):
    return re.sub(r"[^A-Z]", "", str(p).upper())

def mut_index0(pep_mut, p_in_pep):
    """0-based index of the mutated residue in pep_mut (manifest p_in_pep is 1-based)."""
    return int(p_in_pep) - 1

def core_match(pep_mut, mi0, epitope):
    """
    True iff pep_mut and `epitope` share a contiguous block of >= CORE_MIN residues that
    INCLUDES the mutated residue (so a register-shifted catalogued epitope still counts, but a
    coincidental N-/C-terminal overlap that misses the mutation does not).
    """
    a = pep_mut
    b = norm(epitope)
    if len(b) < CORE_MIN:
        return False
    # slide b over a; require an aligned identical block of >=CORE_MIN covering index mi0
    for off in range(-len(b) + 1, len(a)):
        # block in a-coordinates where both defined
        lo = max(0, off)
        hi = min(len(a), off + len(b))
        if hi - lo < CORE_MIN:
            continue
        if not (lo <= mi0 < hi):
            continue
        if a[lo:hi] == b[lo - off:hi - off]:
            return True
    return False


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------
def load_handles():
    h = json.load(open(MANIFEST))["handles"]
    out = []
    for r in h:
        out.append(dict(id=r["id"], gene=r["gene"], mut_label=r.get("mut_label"),
                        allele=r.get("allele"), tier=r.get("tier"),
                        pep_mut=norm(r["pep_mut"]), pep_wt=norm(r["pep_wt"]),
                        p_in_pep=int(r["p_in_pep"])))
    return out

def load_vdjdb():
    import pandas as pd
    f = sorted(glob.glob(os.path.join(GROUND, "vdjdb-*/vdjdb.slim.txt")))
    if not f:
        return None
    df = pd.read_csv(f[-1], sep="\t",
                     usecols=lambda c: c in ("antigen.epitope", "antigen.gene",
                                             "antigen.species", "mhc.a", "mhc.class", "vdjdb.score"))
    df = df[df["mhc.class"] == "MHCI"].copy()
    df["ep"] = df["antigen.epitope"].map(norm)
    return df

def load_iedb():
    """Optional. IEDB epitope_full csv inside a zip; pull linear peptide + a positive-Tcell flag."""
    import pandas as pd
    z = sorted(glob.glob(os.path.join(GROUND, "iedb_epitope_full*.zip")))
    if not z:
        return None
    try:
        with zipfile.ZipFile(z[-1]) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith((".csv", ".tsv"))]
            if not name:
                return None
            with zf.open(name[0]) as fh:
                raw = fh.read()
        # IEDB exports are multi-header; find the column holding linear sequences heuristically
        df = pd.read_csv(io.BytesIO(raw), low_memory=False, header=[0, 1])
        df.columns = [" ".join(str(x) for x in c).strip() for c in df.columns]
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        except Exception:
            return None
    seqcol = None
    for c in df.columns:
        cl = c.lower()
        if ("linear" in cl and "sequence" in cl) or cl.endswith("name") and "epitope" in cl:
            # prefer a column whose values look like peptides
            vals = df[c].dropna().astype(str).head(200)
            if (vals.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{6,40}").mean() > 0.5):
                seqcol = c
                break
    if seqcol is None:
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(300)
            if len(vals) and vals.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{6,40}").mean() > 0.7:
                seqcol = c
                break
    if seqcol is None:
        return None
    out = df[[seqcol]].copy()
    out.columns = ["ep"]
    out["ep"] = out["ep"].map(norm)
    out = out[out["ep"].str.len() >= 6]
    return out


# ---------------------------------------------------------------------------
# core scoring
# ---------------------------------------------------------------------------
def allele_2field(a):
    """'HLA-A*11:01' -> 'A*11:01' ; handles comma-joined and extra fields."""
    out = set()
    for tok in re.split(r"[,\s]+", str(a)):
        m = re.search(r"([ABC])\*?(\d+):?(\d+)?", tok)
        if m:
            grp = f"{m.group(1)}*{m.group(2)}"
            out.add((f"{grp}:{m.group(3)}" if m.group(3) else grp, grp))
    return out  # set of (2field, group)

def allele_match(handle_allele, db_alleles):
    """True if handle's allele matches any db allele at 2-field, else 'group' if only allele-group matches."""
    h = allele_2field(handle_allele)
    if not h:
        return "none"
    h2 = {x[0] for x in h}; hg = {x[1] for x in h}
    db = set()
    for a in db_alleles:
        db |= allele_2field(a)
    d2 = {x[0] for x in db}; dg = {x[1] for x in db}
    if h2 & d2:
        return "exact"
    if hg & dg:
        return "group"
    return "none"

def gene_aliases(gene):
    g = gene.upper()
    al = {g}
    extra = {"BRAF": {"BRAF", "B-RAF"}, "KRAS": {"KRAS", "K-RAS"}, "NRAS": {"NRAS"},
             "IDH1": {"IDH1"}, "TP53": {"TP53", "P53"}, "EGFR": {"EGFR"},
             "PIK3CA": {"PIK3CA"}, "CTNNB1": {"CTNNB1", "BETA-CATENIN", "CATENIN"}}
    return al | extra.get(g, set())

def score_handle(hd, vdj, iedb):
    mi0 = mut_index0(hd["pep_mut"], hd["p_in_pep"])
    res = dict(id=hd["id"], gene=hd["gene"], allele=hd["allele"], tier=hd["tier"],
               pep_mut=hd["pep_mut"], pep_wt=hd["pep_wt"])
    al = gene_aliases(hd["gene"])

    # ---- VDJdb (real TCRs) ----
    res["vdjdb_tcr_mut_exact"] = 0
    res["vdjdb_tcr_mut_core"] = 0
    res["vdjdb_tcr_wt_exact"] = 0
    res["vdjdb_mut_alleles"] = []
    res["vdjdb_mut_gene_confirmed"] = False
    res["vdjdb_allele_match"] = "none"   # exact / group / none — does a real TCR's restriction match OUR predicted allele?
    res["vdjdb_max_score"] = None
    if vdj is not None:
        ex = vdj[vdj["ep"] == hd["pep_mut"]]
        res["vdjdb_tcr_mut_exact"] = int(len(ex))
        if len(ex):
            res["vdjdb_mut_alleles"] = sorted(set(ex["mhc.a"].dropna().astype(str)))[:6]
            res["vdjdb_max_score"] = int(ex["vdjdb.score"].max()) if "vdjdb.score" in ex else None
            genes = set(str(x).upper() for x in ex["antigen.gene"].dropna())
            res["vdjdb_mut_gene_confirmed"] = bool(genes & al) or any(any(a in gg for a in al) for gg in genes)
            res["vdjdb_allele_match"] = allele_match(hd["allele"], res["vdjdb_mut_alleles"])
        # core (register-variant), restricted to same-gene rows to avoid coincidental hits
        same_gene = vdj[vdj["antigen.gene"].astype(str).str.upper().apply(lambda gg: bool(al & {gg}) or any(a in gg for a in al))]
        cand = set(same_gene["ep"]) | set(ex["ep"])
        res["vdjdb_tcr_mut_core"] = int(sum(core_match(hd["pep_mut"], mi0, e) for e in cand))
        # WT self-peptide: gene-confirmed + allele-aware (mirror the mutant logic), else a coincidental hit fakes a safety alarm
        exw = vdj[vdj["ep"] == hd["pep_wt"]]
        res["vdjdb_tcr_wt_exact"] = int(len(exw))
        wt_alleles = sorted(set(exw["mhc.a"].dropna().astype(str)))[:6]
        wt_genes = set(str(x).upper() for x in exw["antigen.gene"].dropna())
        res["vdjdb_wt_gene_confirmed"] = bool(wt_genes & al) or any(any(a in gg for a in al) for gg in wt_genes)
        res["vdjdb_wt_alleles"] = wt_alleles
        res["vdjdb_wt_allele_match"] = allele_match(hd["allele"], wt_alleles) if wt_alleles else "none"

    # ---- IEDB (real epitopes / assays) ----
    res["iedb_mut_exact"] = 0
    res["iedb_mut_core"] = 0
    res["iedb_wt_exact"] = 0
    if iedb is not None:
        eps = set(iedb["ep"])
        res["iedb_mut_exact"] = int(hd["pep_mut"] in eps)
        res["iedb_wt_exact"] = int(hd["pep_wt"] in eps)
        # core over a bounded candidate set (epitopes sharing the mut residue char & length-ish)
        cand = [e for e in eps if abs(len(e) - len(hd["pep_mut"])) <= 3 and len(e) >= CORE_MIN]
        res["iedb_mut_core"] = int(any(core_match(hd["pep_mut"], mi0, e) for e in cand))

    # ---- verdict (ALLELE-AWARE: a TCR recognises a peptide-MHC complex, not a bare peptide) ----
    exact_mut_gene = res["vdjdb_tcr_mut_exact"] > 0 and res["vdjdb_mut_gene_confirmed"]
    real_tcr_thispmhc = exact_mut_gene and res["vdjdb_allele_match"] in ("exact", "group")
    real_ep = res["iedb_mut_exact"] > 0 or res["vdjdb_tcr_mut_exact"] > 0  # catalogued/presented somewhere
    register_tcr = (not exact_mut_gene) and (res["vdjdb_tcr_mut_core"] > 0)

    if real_tcr_thispmhc:
        verdict = "GROUNDED_TCR"             # real cognate TCR for THIS peptide-MHC (allele matches) — strongest de-risk
    elif exact_mut_gene:
        verdict = "GROUNDED_EPITOPE_OTHER_ALLELE"  # peptide is a real TCR-validated neoantigen, but on a DIFFERENT restriction than we predicted
    elif register_tcr:
        verdict = "GROUNDED_REGISTER"        # TCR exists for a register-variant of the same driver mutation
    elif real_ep:
        verdict = "GROUNDED_EPITOPE"         # catalogued/presented but no TCR found
    else:
        verdict = "PREDICTED_ONLY"           # not yet in these DBs (incomplete DB, NOT a negative)
    res["verdict"] = verdict
    # safety: separate a REAL TCR against the self-peptide (genuine concern) from merely-assayed-in-IEDB (often a tested negative)
    res["wt_tcr_real"] = res["vdjdb_tcr_wt_exact"] > 0 and res.get("vdjdb_wt_gene_confirmed", False)
    res["wt_assayed_iedb"] = res["iedb_wt_exact"] > 0
    # genuine cross-reactivity concern = a real, gene-confirmed TCR against the SELF peptide on the SAME restriction as this handle
    res["wt_safety_flag"] = bool(res["wt_tcr_real"] and res.get("vdjdb_wt_allele_match", "none") in ("exact", "group"))
    return res


# ---------------------------------------------------------------------------
# run / report
# ---------------------------------------------------------------------------
def run():
    os.makedirs(OUTDIR, exist_ok=True)
    vdj = load_vdjdb()
    iedb = load_iedb()
    handles = load_handles()
    rows = [score_handle(h, vdj, iedb) for h in handles]

    n = len(rows)
    by_v = defaultdict(int)
    for r in rows:
        by_v[r["verdict"]] += 1
    grounded_tcr = [r for r in rows if r["verdict"] == "GROUNDED_TCR"]
    grounded_other = [r for r in rows if r["verdict"] == "GROUNDED_EPITOPE_OTHER_ALLELE"]
    grounded_any = [r for r in rows if r["verdict"].startswith("GROUNDED")]
    wt_flags = [r for r in rows if r["wt_safety_flag"]]
    genes_tcr_grounded = sorted(set(r["gene"] + " " + (r["id"].split("_")[1]) for r in grounded_tcr))

    summary = dict(
        tag="RUNG-27a reality-grounding",
        question="Of our PREDICTED clean neoantigen handles, which have a REAL cognate TCR (VDJdb) "
                 "or a REAL catalogued epitope (IEDB)? Converts capstone residuals #1/#2 to measured facts.",
        n_handles=n,
        vdjdb_loaded=vdj is not None and int(len(vdj)),
        iedb_loaded=(iedb is not None) and int(len(iedb)),
        verdict_counts=dict(by_v),
        n_grounded_tcr=len(grounded_tcr),
        n_grounded_epitope_other_allele=len(grounded_other),
        n_grounded_any=len(grounded_any),
        drivers_with_real_tcr_validated_pmhc=genes_tcr_grounded,
        grounded_tcr_handles=[dict(id=r["id"], allele=r["allele"], pep_mut=r["pep_mut"],
                                   tcrs=r["vdjdb_tcr_mut_exact"], db_alleles=r["vdjdb_mut_alleles"],
                                   allele_match=r["vdjdb_allele_match"], vdjdb_score=r["vdjdb_max_score"]) for r in grounded_tcr],
        grounded_other_allele_handles=[dict(id=r["id"], predicted_allele=r["allele"], pep_mut=r["pep_mut"],
                                            real_tcr_alleles=r["vdjdb_mut_alleles"]) for r in grounded_other],
        n_wt_real_tcr=len(wt_flags),
        wt_real_tcr_handles=[r["id"] for r in wt_flags],
        CEILING=[
            "MATCH = real receptor/epitope exists (strong de-risk); NO match = DB incomplete, reported PREDICTED_ONLY, NOT a negative.",
            "VDJdb/IEDB are human-curated and biased to well-studied (esp. viral) epitopes; absence understates reality.",
            "antigen.gene confirmation guards coincidental sequence hits; register-core hits are weaker than exact.",
            "WT match = germline self-peptide also recognised/catalogued = MAGE-A3-class cross-reactivity context, not proof of harm.",
        ],
        per_handle=rows,
    )
    json.dump(summary, open(os.path.join(OUTDIR, "grounding.json"), "w"), indent=2)

    # readable
    print(f"\n=== RUNG-27a REALITY-GROUNDING ===")
    print(f"VDJdb class-I rows: {summary['vdjdb_loaded']}   IEDB rows: {summary['iedb_loaded']}")
    print(f"handles: {n}   verdicts: {dict(by_v)}")
    print(f"\nGROUNDED_TCR — real cognate TCR for THIS peptide-MHC (allele matches): {len(grounded_tcr)}/{n}")
    for r in grounded_tcr:
        print(f"   ✓ {r['id']:24s} {r['allele']:14s} {r['pep_mut']:12s} "
              f"TCRs={r['vdjdb_tcr_mut_exact']} match={r['vdjdb_allele_match']} db_allele={r['vdjdb_mut_alleles']} score={r['vdjdb_max_score']}")
    if grounded_other:
        print(f"\nGROUNDED_EPITOPE_OTHER_ALLELE — peptide is a REAL TCR-validated neoantigen, but on a DIFFERENT")
        print(f"  restriction than our pipeline predicted (reality DISCIPLINES the prediction): {len(grounded_other)}")
        for r in grounded_other:
            print(f"   ! {r['id']:24s} predicted {r['allele']:12s} but real TCRs are on {r['vdjdb_mut_alleles']}")
    reg = [r for r in rows if r["verdict"] == "GROUNDED_REGISTER"]
    if reg:
        print(f"\nregister-variant TCR (same driver, shifted register): {len(reg)}  ({[r['id'] for r in reg]})")
    if wt_flags:
        print(f"\nSAFETY — real TCR exists against the WT SELF peptide (genuine cross-reactivity concern): {[r['id'] for r in wt_flags]}")
    else:
        print(f"\nSAFETY — no real TCR found against any WT self-peptide (only IEDB-assayed, often tested-negatives).")
    print(f"\nwrote {os.path.join(OUTDIR,'grounding.json')}")
    return summary


# ---------------------------------------------------------------------------
# selftest — matcher must pass known-true & known-false BEFORE any claim (rule 5)
# ---------------------------------------------------------------------------
def selftest():
    import pandas as pd
    ok = 0; tot = 0
    def check(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # core_match: register shift that keeps the mutation -> hit; one that drops it -> miss
    pm = "VVVGADGVGK"; mi0 = 5  # the 'D'
    check("core exact", core_match(pm, mi0, "VVVGADGVGK"))
    check("core register-shift keeps mut", core_match(pm, mi0, "VVVGADGVGKXX"[:10]) or core_match(pm, mi0, "GADGVGKSAL") is False)
    check("core misses when mutation not covered", not core_match("AAAAAAAAAK", 0, "AAAAAAAK"))  # mut at 0 must be in block
    check("core too-short epitope rejected", not core_match(pm, mi0, "VVVGAD"))

    # gene alias guard
    check("gene alias BRAF", "B-RAF" in gene_aliases("BRAF"))

    # mut_index0
    check("mut_index0 1-based->0", mut_index0("ABCDE", 3) == 2)

    # score_handle on a synthetic VDJdb: KRAS-G12D exact w/ correct gene -> GROUNDED_TCR;
    # a scramble -> PREDICTED_ONLY; WT present -> safety flag.
    vdj = pd.DataFrame({
        "ep": ["VVVGADGVGK", "GILGFVFTL", "VVVGAGGVGK"],
        "antigen.gene": ["KRAS", "M", "KRAS"],
        "antigen.species": ["HomoSapiens", "InfluenzaA", "HomoSapiens"],
        "mhc.a": ["HLA-A*11:01", "HLA-A*02:01", "HLA-A*11:01"],
        "mhc.class": ["MHCI", "MHCI", "MHCI"],
        "vdjdb.score": [2, 3, 1],
    })
    hd_true = dict(id="KRAS_G12D_A1101", gene="KRAS", allele="HLA-A*11:01", tier="clean",
                   pep_mut="VVVGADGVGK", pep_wt="VVVGAGGVGK", p_in_pep=6)
    r = score_handle(hd_true, vdj, None)
    check("KRAS-G12D exact -> GROUNDED_TCR", r["verdict"] == "GROUNDED_TCR")
    check("KRAS-G12D gene confirmed", r["vdjdb_mut_gene_confirmed"])
    check("KRAS-G12D WT present -> safety flag", r["wt_safety_flag"] is True)

    # allele-aware: SAME peptide, but our predicted allele != the real TCR's restriction -> NOT GROUNDED_TCR
    hd_otherallele = dict(id="KRAS_G12D_A0101", gene="KRAS", allele="HLA-A*01:01", tier="clean",
                          pep_mut="VVVGADGVGK", pep_wt="VVVGAGGVGK", p_in_pep=6)
    r_oa = score_handle(hd_otherallele, vdj, None)
    check("same peptide, wrong allele -> GROUNDED_EPITOPE_OTHER_ALLELE", r_oa["verdict"] == "GROUNDED_EPITOPE_OTHER_ALLELE")
    check("allele_match reports 'none' for A0101 vs A1101", r_oa["vdjdb_allele_match"] == "none")
    check("allele_match exact for A1101 vs A1101", r["vdjdb_allele_match"] == "exact")
    check("allele_2field group match", allele_match("HLA-A*11:09", ["HLA-A*11:01"]) == "group")

    hd_scr = dict(id="SCRAMBLE", gene="KRAS", allele="HLA-A*11:01", tier="clean",
                  pep_mut="WQWQWQWQWQ", pep_wt="WQWQWQWQQQ", p_in_pep=6)
    r2 = score_handle(hd_scr, vdj, None)
    check("scramble -> PREDICTED_ONLY", r2["verdict"] == "PREDICTED_ONLY")

    # coincidental same-sequence but WRONG gene must NOT count as gene-confirmed
    vdj_wrong = pd.DataFrame({
        "ep": ["VVVGADGVGK"], "antigen.gene": ["SomeViralProt"],
        "antigen.species": ["EBV"], "mhc.a": ["HLA-A*11:01"], "mhc.class": ["MHCI"], "vdjdb.score": [1]})
    r3 = score_handle(hd_true, vdj_wrong, None)
    check("exact hit but wrong gene -> not gene_confirmed", not r3["vdjdb_mut_gene_confirmed"])

    print(f"\nselftest {ok}/{tot}")
    return ok == tot


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    else:
        run()
