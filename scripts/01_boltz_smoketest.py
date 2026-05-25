#!/usr/bin/env python3
"""
Step 1 — Boltz-2 oracle smoke test (CLOUD-GRADE).

Runs Boltz-2 predictions on two protein-peptide complexes:
  (A) DR5 ECD + TRAIL DR5-binding loop  ← positive control, should bind
  (B) DR5 ECD + scrambled peptide       ← negative control, should not bind

Decision rule:
  ΔG(A) − ΔG(B) ≤ -2 kcal/mol  →  oracle separates signal from noise  →  proceed to Step 2.
  Otherwise → see ASSESSMENT.md Day-1 kill criteria; pivot oracle stack.

Requirements:
  - GPU (A10G / A100 / H100). Not runnable on Mac M2 8GB.
  - `pip install boltz` (or follow https://github.com/jwohlwend/boltz)
  - ~10GB disk for model weights (auto-downloaded on first run)

Usage:
  python scripts/01_boltz_smoketest.py
  # Outputs land under runs/step1_boltz/{positive,negative}/

Reference:
  Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction" (bioRxiv 2025).
  https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = PROJECT_ROOT / "data" / "sequences"
RUN_DIR = PROJECT_ROOT / "runs" / "step1_boltz"

POSITIVE = RUN_DIR / "positive"
NEGATIVE = RUN_DIR / "negative"

# Decision threshold — see ASSESSMENT.md
KILL_CRITERION_KCAL = 2.0  # need ≥ 2 kcal/mol separation


def read_fasta_sequence(path: Path) -> str:
    """Read a single FASTA file and return the concatenated sequence (uppercase, no whitespace)."""
    lines = path.read_text().strip().splitlines()
    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
    return "".join(seq_lines).upper().replace(" ", "")


def write_boltz_yaml(yaml_path: Path, receptor_seq: str, ligand_seq: str, name: str) -> None:
    """Write a Boltz-2 input YAML for a protein-peptide complex with affinity prediction.

    Schema follows boltz CLI input convention (see https://github.com/jwohlwend/boltz).
    """
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = f"""# Boltz-2 input for {name}
sequences:
  - protein:
      id: A
      sequence: {receptor_seq}
  - protein:
      id: B
      sequence: {ligand_seq}
properties:
  - affinity:
      binder: B
"""
    yaml_path.write_text(yaml)


def have_boltz() -> bool:
    return shutil.which("boltz") is not None


def run_boltz(input_yaml: Path, out_dir: Path, diffusion_samples: int = 5) -> int:
    """Invoke boltz CLI. Returns the subprocess exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "boltz",
        "predict",
        str(input_yaml),
        "--use_msa_server",
        "--diffusion_samples",
        str(diffusion_samples),
        "--out_dir",
        str(out_dir),
    ]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False)
    return proc.returncode


@dataclass
class AffinityResult:
    name: str
    affinity_pred_value: Optional[float]   # in Boltz-2 units (log Ki ~ pKd; lower = tighter binder)
    affinity_probability_binary: Optional[float]
    delta_g_kcal_mol: Optional[float]      # converted if affinity_pred_value is pKd
    raw_path: Path


def parse_boltz_affinity(out_dir: Path, name: str) -> AffinityResult:
    """Walk the boltz output dir and parse the affinity JSON if present."""
    candidates = list(out_dir.rglob("affinity*.json"))
    if not candidates:
        return AffinityResult(name=name, affinity_pred_value=None,
                              affinity_probability_binary=None,
                              delta_g_kcal_mol=None,
                              raw_path=out_dir)
    raw_path = candidates[0]
    data = json.loads(raw_path.read_text())
    pred = data.get("affinity_pred_value")
    prob = data.get("affinity_probability_binary")
    dg = None
    if pred is not None:
        # Boltz-2 affinity_pred_value is on a pKd-like log scale.
        # ΔG (kcal/mol) ≈ -RT * ln(10) * pKd ≈ -1.364 * pKd at 298K.
        dg = -1.364 * float(pred)
    return AffinityResult(name=name,
                          affinity_pred_value=pred,
                          affinity_probability_binary=prob,
                          delta_g_kcal_mol=dg,
                          raw_path=raw_path)


def main() -> int:
    print("=" * 72)
    print("cancer-recon-apoptosis — Step 1 — Boltz-2 oracle smoke test (CLOUD)")
    print("=" * 72)

    if not have_boltz():
        print(
            "\n[ERROR] `boltz` CLI not found in PATH.\n"
            "        Install with: pip install boltz\n"
            "        Or follow:    https://github.com/jwohlwend/boltz\n"
            "        Requires GPU (A10G / A100). Not runnable on Mac M2 8GB.\n"
            "        For local plumbing test, use: scripts/01_local_smoketest.py\n"
        )
        return 2

    # Load sequences
    dr5_ecd = read_fasta_sequence(SEQ_DIR / "dr5_ecd_human.fasta")
    trail_loop = read_fasta_sequence(SEQ_DIR / "trail_dr5_binding_loop.fasta")
    scrambled = read_fasta_sequence(SEQ_DIR / "scrambled_control.fasta")
    print(f"  receptor DR5 ECD len: {len(dr5_ecd)}")
    print(f"  positive (TRAIL loop) len: {len(trail_loop)}")
    print(f"  negative (scrambled)  len: {len(scrambled)}")

    # Generate Boltz YAML inputs
    POSITIVE.mkdir(parents=True, exist_ok=True)
    NEGATIVE.mkdir(parents=True, exist_ok=True)
    pos_yaml = POSITIVE / "input.yaml"
    neg_yaml = NEGATIVE / "input.yaml"
    write_boltz_yaml(pos_yaml, dr5_ecd, trail_loop, "DR5+TRAIL_loop")
    write_boltz_yaml(neg_yaml, dr5_ecd, scrambled, "DR5+scrambled")
    print(f"\n  wrote {pos_yaml.relative_to(PROJECT_ROOT)}")
    print(f"  wrote {neg_yaml.relative_to(PROJECT_ROOT)}")

    # Run boltz for each
    print("\n[step] running boltz on positive control (DR5 + TRAIL loop) ...")
    rc_pos = run_boltz(pos_yaml, POSITIVE)
    print(f"  exit={rc_pos}")
    print("\n[step] running boltz on negative control (DR5 + scrambled) ...")
    rc_neg = run_boltz(neg_yaml, NEGATIVE)
    print(f"  exit={rc_neg}")

    # Parse affinities
    pos_res = parse_boltz_affinity(POSITIVE, "positive_DR5+TRAIL_loop")
    neg_res = parse_boltz_affinity(NEGATIVE, "negative_DR5+scrambled")

    # Decision
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    for r in (pos_res, neg_res):
        print(f"  {r.name}")
        print(f"    affinity_pred_value         = {r.affinity_pred_value}")
        print(f"    affinity_probability_binary = {r.affinity_probability_binary}")
        print(f"    ΔG (kcal/mol, derived)      = {r.delta_g_kcal_mol}")
        print()

    if pos_res.delta_g_kcal_mol is None or neg_res.delta_g_kcal_mol is None:
        print("[ERROR] affinity JSON not found in run outputs.")
        print(f"  positive raw: {pos_res.raw_path}")
        print(f"  negative raw: {neg_res.raw_path}")
        return 3

    gap = neg_res.delta_g_kcal_mol - pos_res.delta_g_kcal_mol  # positive should bind tighter (more negative ΔG)
    print(f"  gap = ΔG(negative) − ΔG(positive) = {gap:+.2f} kcal/mol")
    print(f"  decision threshold (≥ +{KILL_CRITERION_KCAL:.1f} kcal/mol means positive binds tighter)")

    if gap >= KILL_CRITERION_KCAL:
        print("\n  ✅ PASS — oracle separates signal from noise. Proceed to Step 2.")
        return 0
    else:
        print("\n  ❌ FAIL — oracle does NOT separate by the required margin.")
        print("     See ASSESSMENT.md Day-1 kill criteria. Pivot options:")
        print("       (i) switch to AlphaFold 3 Server")
        print("       (ii) layer ABFE (OpenFE + Boltz-2 ensemble) on top")
        print("       (iii) add OpenMM MD refinement post-Boltz-2")
        return 1


if __name__ == "__main__":
    sys.exit(main())
