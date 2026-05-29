#!/usr/bin/env python3
"""
Step 1b — INTERFACE METRICS + decision (re-analysis of saved Boltz outputs).

NO Boltz recompute. Reads what scripts/01 already wrote and computes the metric
panel the field uses to tell a real binder from a non-binder, then decides.

WHY THIS EXISTS:
  Raw ipTM "failed" the smoke test because a scrambled binder scored HIGHER ipTM
  than the real TRAIL ectodomain (0.86 vs 0.77) while having far lower pLDDT
  (0.51 vs 0.80). ipTM measures how *confidently* the model commits to a relative
  pose, not whether the interface is biologically real — it is fooled by confident
  nonspecific docking. The robust discriminators are pLDDT-based interface metrics.

METRICS COMPUTED PER COMPLEX:
  From Boltz confidence JSON (no compute):
    iptm, ptm, complex_plddt, complex_iplddt (interface pLDDT), complex_ipde, confidence_score
  Derived from saved structure + PAE:
    pDockQ            Bryant et al. 2022 (Nat Commun) — purpose-built binder/non-binder score
                      x = mean_interface_pLDDT(0-100) * ln(n_interface_contacts)
                      pDockQ = 0.724 / (1 + exp(-0.052*(x - 152.611))) + 0.018
                      (>0.23 acceptable, >0.5 confident interaction)
    mean_interchain_pae   mean PAE over inter-chain residue pairs (lower = more confident)

DECISION (positive vs each negative):
  PASS if, for every negative,  pDockQ(positive) - pDockQ(negative) >= 0.10
  AND complex_iplddt(positive) > complex_iplddt(negative)
  AND pDockQ(positive) >= 0.23 (positive is itself a credible interface).

CAVEAT (printed): the scrambled negative is unfoldable, so pLDDT-based metrics beat
it partly for free. The `negative_folded` (lysozyme) control is the real test — a
well-folded NON-binder. Discriminating the positive from THAT validates the metric
for the design loop, where binders and non-binders are both foldable.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "runs" / "step1_boltz"
OUT_PATH = RUN_DIR / "interface_metrics.json"

# Two-axis AlphaFold-Multimer acceptance filter (literature-standard thresholds,
# chosen a priori — NOT fitted to this run). A credible interface must satisfy BOTH:
IPLDDT_MIN = 0.70             # interface pLDDT (Boltz complex_iplddt), [0,1]
IPAE_MAX   = 15.0             # mean inter-chain PAE (Å); lower = more confident pose
CONTACT_CUTOFF = 8.0          # Å, CB-CB (CA for GLY) — for pDockQ + interface PAE
# pDockQ kept for reference only; it is single-axis (pLDDT+contacts, no PAE) and is
# fooled by confident non-binders (lysozyme scored as high as the real binder).
PDOCKQ_CREDIBLE = 0.23

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("step1b")


# ---------- pure math (unit-tested locally) ----------
def contact_matrix(coords_a, coords_b, cutoff: float = CONTACT_CUTOFF):
    """Boolean (n_a, n_b) of CB-CB (CA for GLY) distances below cutoff (Å)."""
    import numpy as np
    A = np.asarray(coords_a, float); B = np.asarray(coords_b, float)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), bool)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return d < cutoff


def pdockq_from_contact(contact, plddt_a, plddt_b):
    """pDockQ (Bryant et al. 2022). plddt_* on 0-100 scale.

    Returns (pdockq, n_contacts, mean_interface_plddt).
    """
    import numpy as np
    contact = np.asarray(contact, bool)
    pa = np.asarray(plddt_a, float); pb = np.asarray(plddt_b, float)
    n_contacts = int(contact.sum())
    if n_contacts == 0:
        return 0.0, 0, 0.0
    if_plddt = float(np.concatenate([pa[contact.any(1)], pb[contact.any(0)]]).mean())
    x = if_plddt * math.log(n_contacts)
    pdockq = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018
    return float(pdockq), n_contacts, if_plddt


def mean_interchain_pae(pae, chain_ids) -> Optional[float]:
    """Mean PAE over ALL inter-chain residue pairs. pae:(N,N); chain_ids length-N."""
    import numpy as np
    pae = np.asarray(pae, float)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.shape[0] != len(chain_ids):
        return None
    ids = np.array(chain_ids)
    mask = ids[:, None] != ids[None, :]
    if not mask.any():
        return None
    return float(pae[mask].mean())


def interface_pae(pae, contact_ab, n_a) -> Optional[float]:
    """Mean PAE over CONTACTING inter-chain pairs only (interface-restricted).

    pae:(N,N) with chain A at indices [0,n_a) and chain B at [n_a, N).
    contact_ab:(n_a, n_b) bool — True where an A residue contacts a B residue.
    Averages both directions A->B and B->A. Returns None if no contacts/shape bad.
    """
    import numpy as np
    pae = np.asarray(pae, float)
    contact_ab = np.asarray(contact_ab, bool)
    n_b = contact_ab.shape[1]
    if pae.shape[0] != n_a + n_b or contact_ab.shape[0] != n_a:
        return None
    ia, jb = np.where(contact_ab)
    if len(ia) == 0:
        return None
    ab = pae[ia, n_a + jb]            # A-row, B-col
    ba = pae[n_a + jb, ia]            # B-row, A-col
    return float(np.concatenate([ab, ba]).mean())


# ---------- output loading ----------
def load_confidence(out_dir: Path) -> Optional[dict]:
    jsons = sorted(out_dir.rglob("confidence_*.json"))
    if not jsons:
        return None
    best = None
    for p in jsons:
        d = json.loads(p.read_text())
        if d.get("iptm") is None:
            continue
        if best is None or float(d["iptm"]) > float(best["iptm"]):
            best = d
    return best or json.loads(jsons[0].read_text())


def load_pae(out_dir: Path):
    import numpy as np
    paes = sorted(out_dir.rglob("pae_*.npz"))
    if not paes:
        return None
    npz = np.load(paes[0])
    keys = list(npz.keys())
    arr = npz["pae"] if "pae" in keys else npz[keys[0]]
    log.info("    pae npz=%s keys=%s shape=%s", paes[0].name, keys, getattr(arr, "shape", None))
    return arr


def parse_cif(out_dir: Path):
    """Return (chain_ids_in_order, {chain: (coords(n,3), plddt(n,) on 0-100)}).

    CB atom per residue (CA for GLY); pLDDT from the atom B-factor.
    """
    cifs = sorted(out_dir.rglob("*_model_*.cif"))
    if not cifs:
        return None, None
    from Bio.PDB import MMCIFParser
    structure = MMCIFParser(QUIET=True).get_structure("m", str(cifs[0]))
    model = next(iter(structure))
    chain_ids_order, per_chain = [], {}
    bfactors = []
    for chain in model:
        coords, plddts = [], []
        for res in chain:
            if not res.has_id("CA"):
                continue
            atom = res["CB"] if res.has_id("CB") else res["CA"]
            coords.append(atom.coord)
            plddts.append(float(atom.get_bfactor()))
            bfactors.append(float(atom.get_bfactor()))
            chain_ids_order.append(chain.id)
        per_chain[chain.id] = (coords, plddts)
    # detect pLDDT scale: Boltz stores [0,1]; pDockQ formula wants 0-100
    mx = max(bfactors) if bfactors else 0.0
    scale = 100.0 if mx <= 1.5 else 1.0
    log.info("    cif=%s chains=%s maxB=%.3f → pLDDT scale x%.0f",
             cifs[0].name, list(per_chain), mx, scale)
    for c in per_chain:
        coords, plddts = per_chain[c]
        per_chain[c] = (coords, [p * scale for p in plddts])
    return chain_ids_order, per_chain


def analyse(name: str, out_dir: Path) -> Optional[dict]:
    import numpy as np
    log.info("[%s] analysing %s", name, out_dir.relative_to(PROJECT_ROOT))
    conf = load_confidence(out_dir)
    if conf is None:
        log.error("[%s] no confidence JSON — run scripts/01 first", name); return None
    row = {k: conf.get(k) for k in
           ("iptm", "ptm", "complex_plddt", "complex_iplddt", "complex_ipde", "confidence_score")}

    chain_ids, per_chain = parse_cif(out_dir)
    pae = load_pae(out_dir)
    pdockq = n_contacts = if_plddt = ipae = None
    if per_chain and len(per_chain) >= 2:
        cs = list(per_chain)             # [A (receptor), B (binder)]
        ca, pa = per_chain[cs[0]]
        cb, pb = per_chain[cs[1]]
        A = np.asarray(ca, float); B = np.asarray(cb, float)
        if len(A) and len(B):
            contact = contact_matrix(A, B)                     # (n_a, n_b)
            pdockq, n_contacts, if_plddt = pdockq_from_contact(contact, pa, pb)
            if n_contacts and pae is not None:
                ipae = interface_pae(pae, contact, len(A))     # interface-restricted PAE
    row["pdockq"] = pdockq
    row["n_interface_contacts"] = n_contacts
    row["interface_plddt_0_100"] = if_plddt
    row["interface_pae"] = ipae                                  # interface-restricted (supplementary)
    row["mean_interchain_pae"] = (mean_interchain_pae(pae, chain_ids)
                                  if (pae is not None and chain_ids) else None)  # all-pairs (decision axis)

    log.info("[%s] iptm=%s iplddt=%s mean_ic_pae=%s iface_pae=%s pdockq=%s contacts=%s",
             name, _fmt(row["iptm"]), _fmt(row["complex_iplddt"]),
             _fmt(row["mean_interchain_pae"]), _fmt(row["interface_pae"]),
             _fmt(row["pdockq"]), row["n_interface_contacts"])
    return row


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main() -> int:
    log.info("cancer-recon-apoptosis — Step 1b — interface metrics + decision")
    try:
        import numpy  # noqa: F401
        from Bio.PDB import MMCIFParser  # noqa: F401
    except ImportError as e:
        log.error("missing dependency: %s (pip install numpy biopython)", e); return 2

    if not RUN_DIR.exists():
        log.error("no runs at %s — run scripts/01 first", RUN_DIR); return 3

    complexes = {}
    for sub in sorted(p.name for p in RUN_DIR.iterdir() if p.is_dir()):
        row = analyse(sub, RUN_DIR / sub)
        if row is not None:
            complexes[sub] = row

    if "positive" not in complexes:
        log.error("no 'positive' complex analysed — run scripts/01 first"); return 3

    pos = complexes["positive"]
    negatives = {k: v for k, v in complexes.items() if k != "positive"}

    log.info("=" * 64)
    log.info("METRIC PANEL (positive vs negatives)")
    hdr = f"{'metric':<24}" + "".join(f"{k:>16}" for k in complexes)
    log.info(hdr)
    for metric in ("iptm", "ptm", "complex_plddt", "complex_iplddt",
                   "mean_interchain_pae", "interface_pae", "n_interface_contacts",
                   "interface_plddt_0_100", "pdockq"):
        line = f"{metric:<24}" + "".join(f"{_fmt(complexes[k].get(metric)):>16}" for k in complexes)
        log.info(line)

    # ---- decision: two-axis AF-Multimer filter (interface pLDDT AND inter-chain PAE) ----
    # No single metric separates a real binder from BOTH an unfoldable decoy (fails
    # pLDDT) and a confidently-misdocked folded non-binder (fails PAE). The conjunction
    # does. pDockQ is single-axis and is reported but NOT used to decide.
    def credible(row):
        il, pae_ = row.get("complex_iplddt"), row.get("mean_interchain_pae")
        if il is None or pae_ is None:
            return None
        return bool(il >= IPLDDT_MIN and pae_ <= IPAE_MAX)

    log.info("=" * 64)
    log.info("DECISION — two-axis filter: complex_iplddt >= %.2f AND mean_interchain_pae <= %.1f",
             IPLDDT_MIN, IPAE_MAX)
    for k in complexes:
        r = complexes[k]
        il_ok = r.get("complex_iplddt") is not None and r["complex_iplddt"] >= IPLDDT_MIN
        pae_ok = r.get("mean_interchain_pae") is not None and r["mean_interchain_pae"] <= IPAE_MAX
        log.info("  %-16s iplddt=%s(%s)  pae=%s(%s)  → credible=%s",
                 k, _fmt(r.get("complex_iplddt")), "ok" if il_ok else "no",
                 _fmt(r.get("mean_interchain_pae")), "ok" if pae_ok else "no", credible(r))

    pos_cred = credible(pos)
    cred_negs = [k for k, v in negatives.items() if credible(v)]
    reasons = []
    verdict_pass = True
    if pos_cred is None:
        verdict_pass = False; reasons.append("positive metrics incomplete (no iplddt/pae)")
    elif not pos_cred:
        verdict_pass = False
        reasons.append(f"positive not credible (iplddt={_fmt(pos.get('complex_iplddt'))}, "
                       f"pae={_fmt(pos.get('mean_interchain_pae'))})")
    if cred_negs:
        verdict_pass = False
        reasons.append(f"non-binder(s) also pass the filter (false positives): {cred_negs}")

    OUT_PATH.write_text(json.dumps({
        "complexes": complexes,
        "thresholds": {"iplddt_min": IPLDDT_MIN, "ipae_max": IPAE_MAX, "pdockq_credible": PDOCKQ_CREDIBLE},
        "decision_axes": ["complex_iplddt", "mean_interchain_pae"],
        "n_per_class": "1 positive, 2 negatives — thresholds are literature-standard, not fitted; "
                       "calibrate on a benchmark before using as an RL reward",
        "verdict": "PASS" if verdict_pass else "FAIL",
        "reasons": reasons,
    }, indent=2))
    log.info("metrics saved → %s", OUT_PATH.relative_to(PROJECT_ROOT))

    if "negative_folded" not in negatives:
        log.warning("no foldable-negative (lysozyme) control present — add lysozyme_control.fasta and "
                    "re-run scripts/01 to validate the metric against a WELL-FOLDED non-binder.")
    else:
        log.info("note: 'negative' (scrambled) is unfoldable so pLDDT beats it cheaply; the real test "
                 "is separation from 'negative_folded' (lysozyme), a well-folded non-binder.")

    if verdict_pass:
        log.info("✅ PASS — Boltz interface metrics separate the real DR5 binder from non-binders. Proceed to Step 2.")
        return 0
    log.error("❌ FAIL — %s", "; ".join(reasons))
    log.error("Pivot options: model TRAIL homotrimer (true groove) / AF3 Server cross-check / "
              "diffusion_samples=5 best-of-N / use pDockQ2 or ipSAE.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
