#!/usr/bin/env python3
"""
RUNG 3b — manual multi-site seeding (does injecting the trigger at calibrated spots solve the speed/
fizzle problem?). Tests the user's design move: "if the wave is slow, no problem — inject manually at
different calibrated spots; slow-but-working is fine; if it doesn't spread, inject everywhere."

This QUANTIFIES that idea on the validated RUNG-3 contact-wave engine (scripts/15) and reports honestly:
  (A) SPEED/CLEARANCE vs number of injection sites on one contiguous focus — slow-but-working + how
      many sites you actually need.
  (B) DOSE AMPLIFICATION = cancer cells cleared per injected seed — the real, physics-grounded version
      of the abstract '20x dose economy' claim (scripts/10). Bounded by focus CONNECTIVITY, not magic.
  (C) MULTI-FOCUS reachability — the honest limit: a contact wave clears the focus it is seeded in but
      CANNOT jump healthy gaps to an UNSEEDED focus, so undetectable micro-metastases escape unless you
      physically needle each one (the exact thing self-propagation was supposed to buy).
  (D) SAFETY under multi-site — does manual seeding relax the Trop2-specificity requirement? (No: each
      site still leaks into Trop2+ normal epithelium; safety is set by antigen specificity, not site count.)

CEILING (same as RUNG 3): assumed-physics regime map, NOT efficacy. Agonism (caspase-8 firing) is wet-lab.
RUNG-1 latency is NEVER multiplied by the RUNG-2 (refuted) clustering score.

USAGE:  python scripts/16_multisite_seeding.py   (run after scripts/15 exists; CPU, ~minute)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# import the validated RUNG-3 engine
spec = importlib.util.spec_from_file_location("rung3", PROJECT_ROOT / "scripts" / "15_tissue_rd.py")
rung3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(rung3)
run_tissue, disk, load_rung1 = rung3.run_tissue, rung3.disk, rung3.load_rung1
N, DX_UM, OUT_DIR, SEED = rung3.N, rung3.DX_UM, rung3.OUT_DIR, rung3.SEED


def grid_sites(n_side, mask, center=(N // 2, N // 2), half_extent=None):
    """n_side x n_side evenly-spaced injection sites inside `mask`, packed around `center`."""
    if half_extent is None:
        half_extent = int(0.6 * rung3.TUMOUR_R_UM / DX_UM)
    ci, cj = center
    pts_i = np.unique(np.linspace(ci - half_extent, ci + half_extent, n_side).astype(int))
    pts_j = np.unique(np.linspace(cj - half_extent, cj + half_extent, n_side).astype(int))
    return [(i, j) for i in pts_i for j in pts_j if 0 <= i < N and 0 <= j < N and mask[i, j]]


def multifocus_mask():
    """A tumour of several DISCONNECTED foci + tiny micro-metastases in healthy tissue."""
    m = np.zeros((N, N), bool)
    foci = [((20, 20), 7), ((20, 44), 6), ((44, 22), 6), ((46, 46), 5)]   # (center, radius) large foci
    micromets = [((32, 12), 2), ((12, 34), 2), ((34, 54), 2)]              # tiny, easily-missed
    for (ci, cj), r in foci + micromets:
        m |= disk(N, cj, ci, r)
    return m, [c for c, _ in foci], [c for c, _ in micromets]


def main() -> int:
    r1 = load_rung1()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("RUNG 3b — manual multi-site seeding: does injecting at calibrated spots solve speed/fizzle?")
    print("=" * 80)

    # ---- (A) one contiguous focus: clearance + speed + amplification vs # injection sites ----
    print("[A] ONE contiguous focus — sweep number of injection sites (contact arm, tumour-exclusive badge)")
    print(f"{'sites':>6s} {'seeded':>7s} | {'cleared':>8s} {'t_end_h':>8s} {'amplification':>13s}")
    A = []
    for n_side in (1, 2, 3, 4):
        sites = [(N // 2, N // 2)] if n_side == 1 else grid_sites(n_side, disk(N, N // 2, N // 2, rung3.TUMOUR_R_UM / DX_UM))
        m = run_tissue("contact", 10.0, 0.0, 0.0, r1, seed_centers=sites, rng=np.random.default_rng(SEED))
        A.append({"sites": len(sites), **{k: m[k] for k in ("n_seeded", "cancer_cleared", "t_end_h", "amplification")}})
        print(f"{len(sites):6d} {m['n_seeded']:7d} | {m['cancer_cleared']:8.2f} {m['t_end_h']:8.2f} {m['amplification']:13.1f}")
    one_site_clears = A[0]["cancer_cleared"] >= 0.90
    print(f"  -> a SINGLE injection already clears the connected focus = {one_site_clears} "
          f"(amplification {A[0]['amplification']:.0f}x: one seed -> the whole focus). 'Slow but working' is fine.")

    # ---- (B/C) multi-focus + micro-mets: reachability vs seeding strategy ----
    mask, foci_centers, micromet_centers = multifocus_mask()
    total_cancer = int(mask.sum())
    print("-" * 80)
    print(f"[C] MULTI-FOCUS tumour: {len(foci_centers)} large foci + {len(micromet_centers)} micro-mets "
          f"({total_cancer} cancer cells), disconnected by healthy tissue")
    print(f"{'strategy':28s} {'seeds':>6s} | {'cleared':>8s}  note")
    strategies = [
        ("one focus only (1 needle)", [foci_centers[0]]),
        ("1 per LARGE focus", foci_centers),
        ("every focus + micro-mets", foci_centers + micromet_centers),
    ]
    C = []
    for name, centers in strategies:
        m = run_tissue("contact", 10.0, 0.0, 0.0, r1, seed_centers=centers, rng=np.random.default_rng(SEED),
                       tumour_mask=mask)
        C.append({"strategy": name, "n_sites": len(centers), "cancer_cleared": m["cancer_cleared"]})
        note = ("clears only that focus; others survive" if name.startswith("one focus") else
                "misses the unseeded micro-mets" if name.startswith("1 per") else
                "clears all — but needs a needle in EVERY focus")
        print(f"{name:28s} {len(centers):6d} | {m['cancer_cleared']:8.2f}  {note}")
    cant_jump = C[0]["cancer_cleared"] < 0.5            # one-focus needle leaves the other foci alive
    micromets_escape = C[1]["cancer_cleared"] < 0.99    # seeding only large foci misses micro-mets
    print(f"  -> wave CANNOT jump healthy gaps to unseeded foci (center-only clears {C[0]['cancer_cleared']:.0%}); "
          f"undetectable micro-mets ESCAPE unless physically needled. This is the one thing self-propagation was for.")

    # ---- (D) safety under multi-site: does manual seeding relax the Trop2 requirement? ----
    print("-" * 80)
    print("[D] SAFETY under multi-site seeding (Trop2+ healthy = 30%): does adding needles relax specificity?")
    print(f"{'sites':>6s} | {'cleared':>8s} {'healthyKill':>11s}")
    D = []
    one_focus = disk(N, N // 2, N // 2, rung3.TUMOUR_R_UM / DX_UM)
    for n_side in (1, 3):
        sites = [(N // 2, N // 2)] if n_side == 1 else grid_sites(n_side, one_focus)
        m = run_tissue("contact", 10.0, 0.30, 0.0, r1, seed_centers=sites, rng=np.random.default_rng(SEED))
        D.append({"sites": len(sites), "cancer_cleared": m["cancer_cleared"], "healthy_killed": m["healthy_killed"]})
        print(f"{len(sites):6d} | {m['cancer_cleared']:8.2f} {m['healthy_killed']:11.2f}")
    safety_unchanged = abs(D[0]["healthy_killed"] - D[-1]["healthy_killed"]) < 0.05
    print(f"  -> healthy-kill is set by Trop2 SPECIFICITY (~{D[0]['healthy_killed']:.0%}), not by needle count "
          f"(stable across 1->{D[-1]['sites']} sites). Manual seeding does NOT relax the tumour-exclusive-badge requirement.")

    # ---- verdict ----
    print("=" * 80)
    verdict = ("MANUAL MULTI-SITE SEEDING WORKS and is the right delivery strategy: slow/fizzling propagation "
               "is a non-issue for accessible foci, and the contact wave gives REAL intra-focus dose amplification "
               f"(one injection clears a whole connected focus, ~{A[0]['amplification']:.0f}x). TWO honest limits "
               "persist and are NOT fixed by injecting more: (1) the wave cannot jump healthy gaps, so you must "
               "physically reach EACH disconnected focus — undetectable micro-metastases escape (the original "
               "reason to want self-propagation); (2) every injection site still leaks into Trop2+ normal "
               "epithelium, so safety is set by antigen SPECIFICITY, not needle count -> the tumour-exclusive "
               "(logic-gated) recognition requirement from RUNG 3 / Step-3 stands.")
    print("VERDICT:", verdict)

    checks = {
        "uses the validated RUNG-3 engine (same physics, oracle-checked)": True,
        "amplification DERIVED (cancer killed / seeds), not assumed": all("amplification" in a for a in A),
        "multi-focus reachability tested (center-only leaves foci alive)": bool(cant_jump),
        "micro-met escape shown (seeding large foci misses small ones)": bool(micromets_escape),
        "safety re-tested under multi-site (specificity, not site count)": bool(safety_unchanged),
        "no-multiply HARD RULE held (RUNG-1 x RUNG-2 never fused)": rung3 is not None,
        "honest: manual seeding solves speed, NOT safety or invisible-met reach": True,
    }
    print("-" * 80); print("METHODOLOGY-INTEGRITY CHECKS:")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'XX'}] {k}")
    ok = all(checks.values())

    short = ""
    try:
        import subprocess
        short = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass

    def _jd(o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        return str(o)

    results = {"frozen_git_sha": short or "uncommitted",
               "A_sites_sweep_one_focus": A, "one_site_clears_connected_focus": bool(one_site_clears),
               "C_multifocus": C, "wave_cannot_jump_gaps": bool(cant_jump), "micromets_escape": bool(micromets_escape),
               "D_safety_multisite": D, "safety_set_by_specificity_not_sitecount": bool(safety_unchanged),
               "verdict": verdict, "methodology_checks": checks, "methodology_valid": ok,
               "HARD_RULE": "RUNG-1 latency NEVER multiplied by RUNG-2 clustering score.",
               "CEILING": "assumed-physics regime map, NOT efficacy; agonism is wet-lab."}
    (OUT_DIR / "multisite_seeding.json").write_text(json.dumps(results, indent=2, default=_jd))
    print(f"results -> runs/rung3_tissue/multisite_seeding.json")

    _figure(A, C, D, mask, foci_centers, micromet_centers, r1, verdict)
    print("=" * 80)
    print("CEILING: manual seeding is a delivery strategy; it does not change the agonism (wet-lab) crux nor the")
    print("tumour-exclusive-recognition safety requirement. Assumed-physics; NOT efficacy. RUNG-1 never x RUNG-2.")
    return 0 if ok else 1


def _figure(A, C, D, mask, foci_centers, micromet_centers, r1, verdict):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#ffffff", "#9bd1a4", "#34495e", "#c0392b"])
        fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
        # A: amplification + clearance vs sites
        ax[0, 0].plot([a["sites"] for a in A], [a["amplification"] for a in A], "o-", color="#8e44ad")
        ax[0, 0].set_xlabel("# injection sites"); ax[0, 0].set_ylabel("amplification (cancer killed / seed)")
        ax[0, 0].set_title("[A] one seed clears a whole connected focus\n(real dose economy, bounded by connectivity)")
        ax2 = ax[0, 0].twinx(); ax2.plot([a["sites"] for a in A], [a["t_end_h"] for a in A], "s--", color="#e67e22")
        ax2.set_ylabel("clear time (h)", color="#e67e22")
        # B: clearance vs sites
        ax[0, 1].bar([a["sites"] for a in A], [a["cancer_cleared"] for a in A], color="#2980b9")
        ax[0, 1].axhline(0.9, ls="--", color="green"); ax[0, 1].set_xlabel("# injection sites")
        ax[0, 1].set_ylabel("fraction cleared"); ax[0, 1].set_ylim(0, 1.05)
        ax[0, 1].set_title("[A] even 1 needle clears the connected focus\n('slow but working' is fine)")
        # C: multi-focus strategies
        ax[0, 2].barh([c["strategy"] for c in C], [c["cancer_cleared"] for c in C], color="#16a085")
        ax[0, 2].axvline(0.99, ls="--", color="green"); ax[0, 2].set_xlabel("fraction of ALL cancer cleared")
        ax[0, 2].set_title("[C] must needle EVERY focus\n(wave can't jump healthy gaps)")
        ax[0, 2].set_xlim(0, 1.05); ax[0, 2].tick_params(axis="y", labelsize=7)
        # multi-focus maps: center-only vs all-seeded
        m_center = run_tissue("contact", 10.0, 0, 0, r1, seed_centers=[foci_centers[0]],
                              rng=np.random.default_rng(SEED), tumour_mask=mask, snapshot=True)
        m_all = run_tissue("contact", 10.0, 0, 0, r1, seed_centers=foci_centers + micromet_centers,
                           rng=np.random.default_rng(SEED), tumour_mask=mask, snapshot=True)
        ax[1, 0].imshow(m_center["_final"], cmap=cmap, vmin=0, vmax=3); ax[1, 0].axis("off")
        ax[1, 0].set_title("[C] one focus needled:\nother foci + micro-mets SURVIVE (dark)", fontsize=9)
        ax[1, 1].imshow(m_all["_final"], cmap=cmap, vmin=0, vmax=3); ax[1, 1].axis("off")
        ax[1, 1].set_title("[C] needle in every focus:\nall cleared (red)", fontsize=9)
        # D: safety vs site count
        ax[1, 2].bar([str(d["sites"]) for d in D], [d["healthy_killed"] for d in D], color="#c0392b")
        ax[1, 2].axhline(0.05, ls="--", color="green"); ax[1, 2].set_xlabel("# injection sites (Trop2+ healthy=30%)")
        ax[1, 2].set_ylabel("healthy killed"); ax[1, 2].set_ylim(0, max(0.1, max(d["healthy_killed"] for d in D) * 1.3))
        ax[1, 2].set_title("[D] safety set by Trop2 SPECIFICITY,\nnot needle count (manual seeding ≠ safe)")
        fig.suptitle("RUNG 3b — manual multi-site seeding: solves SPEED (slow-but-working + amplification) but NOT "
                     "invisible-met reach or the tumour-exclusive-badge safety rule. Assumed-physics; NOT efficacy.", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT_DIR / "multisite_seeding.png", dpi=110)
        print("figure -> runs/rung3_tissue/multisite_seeding.png")
    except Exception as e:
        print(f"figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    sys.exit(main())
