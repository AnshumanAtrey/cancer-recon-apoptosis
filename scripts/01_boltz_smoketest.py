#!/usr/bin/env python3
"""
Step 1 — Boltz-2 oracle smoke test (CLOUD-GRADE).

Runs Boltz-2 predictions on two protein-peptide complexes:
  (A) DR5 ECD + TRAIL DR5-binding loop  ← positive control, should bind
  (B) DR5 ECD + scrambled peptide       ← negative control, should not bind

Decision rule:
  ΔG(B) − ΔG(A) ≥ +2 kcal/mol  →  oracle separates signal from noise  →  proceed to Step 2.
  Otherwise → see ASSESSMENT.md Day-1 kill criteria; pivot oracle stack.

Resumability:
  Each complex caches its affinity_pred_value in `runs/step1_boltz/state.json`.
  If you rerun the script after a partial completion (e.g. positive done,
  negative crashed), the positive is SKIPPED and only the negative is rerun.
  To force a full rerun, delete `runs/step1_boltz/state.json` or the
  per-complex subdir (`runs/step1_boltz/positive/` etc).

Requirements:
  - GPU (A10G / A100 / H100). Not runnable on Mac M2 8GB.
  - `pip install boltz` (or follow https://github.com/jwohlwend/boltz)
  - ~10GB disk for model weights (auto-downloaded on first run)

Usage:
  python scripts/01_boltz_smoketest.py
  # Outputs land under runs/step1_boltz/{positive,negative}/

Reference:
  Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity
  Prediction" (bioRxiv 2025).
  https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = PROJECT_ROOT / "data" / "sequences"
RUN_DIR = PROJECT_ROOT / "runs" / "step1_boltz"
STATE_PATH = RUN_DIR / "state.json"

POSITIVE = RUN_DIR / "positive"
NEGATIVE = RUN_DIR / "negative"

# Decision threshold — see ASSESSMENT.md
KILL_CRITERION_KCAL = 2.0

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,                    # ← Colab swallows stderr; send logs to stdout
)
log = logging.getLogger("step1")


# ---------- helpers ----------
def read_fasta_sequence(path: Path) -> str:
    lines = path.read_text().strip().splitlines()
    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
    return "".join(seq_lines).upper().replace(" ", "")


def write_boltz_yaml(yaml_path: Path, receptor_seq: str, ligand_seq: str, name: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"# Boltz-2 input for {name}\n"
        f"sequences:\n"
        f"  - protein:\n      id: A\n      sequence: {receptor_seq}\n"
        f"  - protein:\n      id: B\n      sequence: {ligand_seq}\n"
        f"properties:\n  - affinity:\n      binder: B\n"
    )


def have_boltz() -> bool:
    return shutil.which("boltz") is not None


def find_affinity_json(out_dir: Path) -> Optional[Path]:
    """Return path to an affinity JSON inside the per-complex out_dir, or None."""
    matches = list(out_dir.rglob("affinity*.json"))
    return matches[0] if matches else None


def run_boltz(input_yaml: Path, out_dir: Path, diffusion_samples: int = 1) -> int:
    """Run boltz CLI, mirror output to stdout AND a per-complex boltz.log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "boltz.log"
    cmd = [
        "boltz", "predict", str(input_yaml),
        "--use_msa_server",
        "--diffusion_samples", str(diffusion_samples),
        "--out_dir", str(out_dir),
    ]
    log.info("invoking: %s", " ".join(cmd))
    log.info("boltz log mirrored to %s", log_path.relative_to(PROJECT_ROOT))
    t0 = time.time()
    # Stream output: read line-by-line from boltz, write to BOTH stdout (for Colab) and log file
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
    log.info("boltz exited rc=%d in %.1fs", rc, time.time() - t0)
    return rc


def tail_log(log_path: Path, n: int = 30) -> None:
    """Print the last n lines of a boltz log for post-mortem diagnostics."""
    if not log_path.exists():
        log.warning("no boltz.log at %s", log_path)
        return
    lines = log_path.read_text().splitlines()
    log.error("--- last %d lines of %s ---", min(n, len(lines)), log_path.name)
    for ln in lines[-n:]:
        print(f"    {ln}", flush=True)
    log.error("--- end of %s ---", log_path.name)


@dataclass
class AffinityResult:
    name: str
    affinity_pred_value: Optional[float]
    affinity_probability_binary: Optional[float]
    delta_g_kcal_mol: Optional[float]
    raw_path: Optional[str]
    status: str   # "DONE" | "CACHED" | "MISSING" | "FAILED"


def parse_affinity(out_dir: Path, name: str, cached: bool) -> AffinityResult:
    p = find_affinity_json(out_dir)
    if p is None:
        return AffinityResult(name, None, None, None, None, "MISSING")
    data = json.loads(p.read_text())
    pred = data.get("affinity_pred_value")
    prob = data.get("affinity_probability_binary")
    # Boltz-2 affinity_pred_value is a pKd-like log scale; ΔG ≈ -1.364 * pKd at 298 K
    dg = -1.364 * float(pred) if pred is not None else None
    status = "CACHED" if cached else "DONE"
    return AffinityResult(name, pred, prob, dg, str(p), status)


def process_complex(name: str, receptor: str, ligand: str, out_dir: Path) -> AffinityResult:
    """Idempotent: returns cached result if already complete, else runs boltz."""
    log.info("=" * 60)
    log.info("[%s] start (out_dir=%s)", name, out_dir.relative_to(PROJECT_ROOT))
    cached = find_affinity_json(out_dir)
    if cached is not None:
        log.info("[%s] SKIP — affinity JSON already present at %s",
                 name, cached.relative_to(out_dir))
        return parse_affinity(out_dir, name, cached=True)

    yaml_path = out_dir / "input.yaml"
    write_boltz_yaml(yaml_path, receptor, ligand, name)
    log.info("[%s] wrote YAML → %s", name, yaml_path.relative_to(PROJECT_ROOT))

    rc = run_boltz(yaml_path, out_dir)
    if rc != 0:
        log.error("[%s] boltz failed with rc=%d", name, rc)
        tail_log(out_dir / "boltz.log")
        log.error("[%s] out_dir contents:", name)
        for p in sorted(out_dir.rglob("*")):
            print(f"    {p.relative_to(out_dir)}", flush=True)
        return AffinityResult(name, None, None, None, None, "FAILED")

    res = parse_affinity(out_dir, name, cached=False)
    if res.status == "MISSING":
        log.error("[%s] boltz rc=0 but no affinity JSON found — likely OOM mid-inference", name)
        tail_log(out_dir / "boltz.log")
        log.error("[%s] out_dir contents:", name)
        for p in sorted(out_dir.rglob("*")):
            print(f"    {p.relative_to(out_dir)}", flush=True)
    return res


def save_state(positive: AffinityResult, negative: AffinityResult, gap: Optional[float]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "positive": asdict(positive),
        "negative": asdict(negative),
        "gap_kcal_mol": gap,
        "decision_threshold_kcal_mol": KILL_CRITERION_KCAL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    log.info("state saved → %s", STATE_PATH.relative_to(PROJECT_ROOT))


# ---------- main ----------
def main() -> int:
    log.info("cancer-recon-apoptosis — Step 1 — Boltz-2 oracle smoke test (CLOUD)")

    if not have_boltz():
        log.error("`boltz` CLI not found in PATH.")
        log.error("Install with: pip install boltz")
        log.error("For local plumbing test instead: python scripts/01_local_smoketest.py")
        return 2

    # Load reference sequences
    try:
        dr5_ecd    = read_fasta_sequence(SEQ_DIR / "dr5_ecd_human.fasta")
        trail_loop = read_fasta_sequence(SEQ_DIR / "trail_dr5_binding_loop.fasta")
        scrambled  = read_fasta_sequence(SEQ_DIR / "scrambled_control.fasta")
    except FileNotFoundError as e:
        log.error("FASTA missing: %s", e)
        return 4
    log.info("DR5_ECD len=%d  TRAIL_loop len=%d  Scrambled len=%d",
             len(dr5_ecd), len(trail_loop), len(scrambled))

    # Process each complex (idempotent — skips if already done)
    pos = process_complex("POSITIVE", dr5_ecd, trail_loop, POSITIVE)
    neg = process_complex("NEGATIVE", dr5_ecd, scrambled,  NEGATIVE)

    # Report
    log.info("=" * 60)
    log.info("RESULTS")
    for r in (pos, neg):
        log.info("[%s] status=%s pKd=%s ΔG=%s",
                 r.name, r.status, r.affinity_pred_value, r.delta_g_kcal_mol)

    if pos.delta_g_kcal_mol is None or neg.delta_g_kcal_mol is None:
        log.error("One or both complexes have no affinity output — cannot decide.")
        save_state(pos, neg, None)
        return 3

    gap = neg.delta_g_kcal_mol - pos.delta_g_kcal_mol
    log.info("gap = ΔG(negative) − ΔG(positive) = %+.2f kcal/mol", gap)
    log.info("decision threshold ≥ +%.1f kcal/mol", KILL_CRITERION_KCAL)
    save_state(pos, neg, gap)

    if gap >= KILL_CRITERION_KCAL:
        log.info("✅ PASS — oracle separates signal from noise. Proceed to Step 2.")
        return 0
    log.error("❌ FAIL — see ASSESSMENT.md Day-1 kill criteria. Pivot options:")
    log.error("  (i) AlphaFold 3 Server cross-check")
    log.error("  (ii) ABFE pipeline (OpenFE + Boltz-2 ensemble)")
    log.error("  (iii) OpenMM MD refinement after Boltz-2")
    return 1


if __name__ == "__main__":
    sys.exit(main())
