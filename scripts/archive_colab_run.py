#!/usr/bin/env python3
"""
Archive a Colab run bundle into a clean, immutable per-run folder — ONE command per run.

Cell 6 of the RUNG-5 notebook downloads a SINGLE zip named:
    rung5_run_<UTC-timestamp>_<git-sha>.zip
This script takes that bundle (newest in ~/Downloads by default) and:
  1. extracts it bit-for-bit into  runs/rung5_logicgate/colab_runs/<UTC-timestamp>_<git-sha>/
     (folder name == the runlog == the exact code commit; nothing ever overwrites another run)
  2. mirrors the canonical "current result" (rung5_addressability.json + rung5_real.png) one level up
  3. git-adds everything; with --commit it also commits (message references the run's sha) and pushes.

USAGE
  python scripts/archive_colab_run.py                 # newest ~/Downloads/rung5_run_*.zip, stage only
  python scripts/archive_colab_run.py <path-to-zip>   # a specific bundle
  python scripts/archive_colab_run.py --commit         # also commit + push
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path(os.path.expanduser("~/Downloads"))
# rung-agnostic: any  <rungN>_run_<UTC-timestamp>_<git-sha>.zip  is archived to that rung's runs/ dir.
BUNDLE_RE = re.compile(r"(rung\d+)_run_(\d{8}T\d{6}Z)_([0-9a-f]+)\.zip$")
# map a bundle prefix -> the rung's results directory under runs/ (fallback: runs/<prefix>)
RUNG_DIRS = {"rung5": "rung5_logicgate", "rung8": "rung8_hla", "rung9": "rung9_ifn", "rung10": "rung10_andnot3"}


def _rung_dir(prefix: str) -> Path:
    return PROJECT_ROOT / "runs" / RUNG_DIRS.get(prefix, prefix)


def _newest_bundle() -> Path | None:
    zips = sorted(DOWNLOADS.glob("rung*_run_*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--commit"]
    do_commit = "--commit" in sys.argv[1:]
    bundle = Path(args[0]) if args else _newest_bundle()
    if not bundle or not bundle.exists():
        print(f"[archive] no bundle found. Expected ~/Downloads/<rungN>_run_<ts>_<sha>.zip "
              f"(the notebook's last cell makes it). Pass a path explicitly if it's elsewhere.")
        return 1
    m = BUNDLE_RE.search(bundle.name)
    if not m:
        print(f"[archive] '{bundle.name}' is not a <rungN>_run_<UTC-timestamp>_<git-sha>.zip bundle.")
        return 2
    prefix, ts, sha = m.group(1), m.group(2), m.group(3)
    rung_dir = _rung_dir(prefix)
    canonical = rung_dir                                      # where the "latest" result/figure are mirrored
    folder = rung_dir / "colab_runs" / f"{ts}_{sha}"
    folder.mkdir(parents=True, exist_ok=True)
    rel = canonical.relative_to(PROJECT_ROOT)

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        zf.extractall(folder)
    print(f"[archive] {bundle.name}  ->  {rel}/colab_runs/{ts}_{sha}/  ({len(names)} files)")
    for n in sorted(names):
        print(f"          + {n}")

    # mirror EVERY .json / .png up as the canonical "current result" (rung-agnostic)
    mirrored = []
    for src in sorted(folder.glob("*.json")) + sorted(folder.glob("*.png")):
        shutil.copy2(src, canonical / src.name)
        mirrored.append(src.name)
    if mirrored:
        print(f"[archive] mirrored as current result -> {rel}/: {', '.join(mirrored)}")

    subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "-A",
                    str(folder.relative_to(PROJECT_ROOT)),
                    *[f"{rel}/{n}" for n in mirrored]], check=False)
    if do_commit:
        msg = (f"{prefix.upper()}: archive Colab run {ts}_{sha} (bit-for-bit, {len(names)} files) + mirror current result")
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "commit", "-q", "-m", msg], check=False)
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "push", "origin", "HEAD"], check=False)
        print(f"[archive] committed + pushed: {msg}")
    else:
        print("[archive] staged. Review with `git status`, then commit, or re-run with --commit to auto-commit+push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
