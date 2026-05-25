#!/usr/bin/env python3
"""
Step 1 (LOCAL PROXY) — ESM-2 plumbing smoke test for M2 8GB.

This is NOT a real binding affinity oracle. It is a *pipeline plumbing test*:
verifies that we can load a small protein language model, compute embeddings for
the three reference sequences, and produce a simple signal-vs-noise contrast.

The actual Step-1 decision must come from `01_boltz_smoketest.py` on a cloud GPU.

What this does:
  1. Load ESM-2-t6-8M (8M params, ~30MB, runs on CPU comfortably).
  2. Compute mean-pooled per-residue embeddings for DR5 ECD, TRAIL loop,
     scrambled peptide, and DR4 ECD.
  3. Compute cosine similarity of each peptide to (a) DR5 ECD and (b) DR4 ECD.
  4. Report whether the TRAIL loop has more affinity-like character to DR5 than
     the scrambled control does, as a SANITY check of the workflow.

This proxy is sequence-only and tells us nothing about 3D binding. Treat any
positive signal here as "the wires are connected," not "the science works."

Requirements:
  pip install torch transformers
  (M2 will use CPU; that's fine for ESM-2-t6-8M.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = PROJECT_ROOT / "data" / "sequences"
RUN_DIR = PROJECT_ROOT / "runs" / "step1_local"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Tiny ESM-2 (8M params). For better signal switch to esm2_t12_35M_UR50D or
# esm2_t30_150M_UR50D — but those want more RAM.
MODEL_ID = "facebook/esm2_t6_8M_UR50D"


def read_fasta(path: Path) -> str:
    lines = path.read_text().strip().splitlines()
    return "".join(ln.strip() for ln in lines if not ln.startswith(">")).upper().replace(" ", "")


def embed(model, tokenizer, seq: str, device: str) -> torch.Tensor:
    """Return mean-pooled embedding over non-special-token residues."""
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    # last_hidden_state: [1, L, D]
    hidden = out.last_hidden_state[0]            # [L, D]
    # Drop [CLS] and [EOS] tokens (positions 0 and -1)
    res_hidden = hidden[1:-1]                    # [L-2, D]
    return res_hidden.mean(dim=0)                # [D]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> int:
    print("=" * 72)
    print("cancer-recon-apoptosis — Step 1 (LOCAL PROXY) — ESM-2 plumbing test (M2/CPU)")
    print("=" * 72)
    print("\nThis is a plumbing test, NOT a real binding oracle.")
    print("Real Step-1 decision must come from scripts/01_boltz_smoketest.py on a cloud GPU.\n")

    # Lazy imports so we can show a friendly error if deps are missing
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"[ERROR] missing dependency: {e}")
        print("        Install with: pip install -r requirements.txt")
        return 2

    # Load sequences
    sequences: Dict[str, str] = {
        "DR5_ECD":            read_fasta(SEQ_DIR / "dr5_ecd_human.fasta"),
        "TRAIL_loop_pos":     read_fasta(SEQ_DIR / "trail_dr5_binding_loop.fasta"),
        "Scrambled_neg":      read_fasta(SEQ_DIR / "scrambled_control.fasta"),
        "DR4_ECD_homolog":    read_fasta(SEQ_DIR / "dr4_ecd_human.fasta"),
    }
    for name, seq in sequences.items():
        print(f"  {name:<20s} len={len(seq):4d}  preview={seq[:20]}...")

    # Device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"\n  device: {device}")

    # Load model
    print(f"  loading model: {MODEL_ID} (first run downloads ~30MB) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    print("  model ready.\n")

    # Embed
    embeddings: Dict[str, torch.Tensor] = {}
    for name, seq in sequences.items():
        print(f"  embedding {name} ...")
        embeddings[name] = embed(model, tokenizer, seq, device)
    print("  all embeddings computed.\n")

    # Cosine similarities — peptide vs receptor
    dr5 = embeddings["DR5_ECD"]
    dr4 = embeddings["DR4_ECD_homolog"]
    pos = embeddings["TRAIL_loop_pos"]
    neg = embeddings["Scrambled_neg"]

    sim_pos_dr5 = cosine(pos, dr5)
    sim_neg_dr5 = cosine(neg, dr5)
    sim_pos_dr4 = cosine(pos, dr4)
    sim_neg_dr4 = cosine(neg, dr4)

    print("=" * 72)
    print("RESULTS (cosine similarity of mean-pooled embeddings)")
    print("=" * 72)
    print(f"  cosine(TRAIL_loop , DR5) = {sim_pos_dr5:+.4f}     ← positive vs target")
    print(f"  cosine(Scrambled  , DR5) = {sim_neg_dr5:+.4f}     ← negative vs target")
    print(f"    margin (pos − neg)     = {sim_pos_dr5 - sim_neg_dr5:+.4f}")
    print()
    print(f"  cosine(TRAIL_loop , DR4) = {sim_pos_dr4:+.4f}     ← specificity diag")
    print(f"  cosine(Scrambled  , DR4) = {sim_neg_dr4:+.4f}")
    print()

    # Persist
    out = RUN_DIR / "results.json"
    import json
    out.write_text(json.dumps({
        "model_id": MODEL_ID,
        "device": device,
        "sequences": {k: {"len": len(v), "preview": v[:30]} for k, v in sequences.items()},
        "similarities": {
            "TRAIL_loop_vs_DR5":   sim_pos_dr5,
            "Scrambled_vs_DR5":    sim_neg_dr5,
            "TRAIL_loop_vs_DR4":   sim_pos_dr4,
            "Scrambled_vs_DR4":    sim_neg_dr4,
            "margin_pos_minus_neg_on_DR5": sim_pos_dr5 - sim_neg_dr5,
        },
        "note": "ESM-2 sequence-embedding cosine is NOT a binding-affinity oracle. "
                "Used here only to verify the local pipeline runs end-to-end on M2.",
    }, indent=2))
    print(f"  saved → {out.relative_to(PROJECT_ROOT)}\n")

    # Light interpretation — this proxy has no real decision threshold, so we
    # just check the pipeline produced finite numbers.
    if not all(map(lambda x: -1.0 < x < 1.0,
                   [sim_pos_dr5, sim_neg_dr5, sim_pos_dr4, sim_neg_dr4])):
        print("  ❌ PIPELINE FAULT — unexpected similarity values; investigate.")
        return 1

    print("  ✅ PLUMBING OK — model loaded, embeddings computed, file written.")
    print("     Next: run scripts/01_boltz_smoketest.py on a cloud GPU for the")
    print("     real Step-1 decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
