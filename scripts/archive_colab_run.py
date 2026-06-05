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
ARCHIVE = PROJECT_ROOT / "runs" / "rung5_logicgate" / "colab_runs"
CANONICAL = PROJECT_ROOT / "runs" / "rung5_logicgate"        # where the "latest" result/figure are mirrored
MIRROR = ("rung5_addressability.json", "rung5_real.png")     # files copied up as the current result
BUNDLE_RE = re.compile(r"rung5_run_(\d{8}T\d{6}Z)_([0-9a-f]+)\.zip$")


def _newest_bundle() -> Path | None:
    zips = sorted(DOWNLOADS.glob("rung5_run_*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--commit"]
    do_commit = "--commit" in sys.argv[1:]
    bundle = Path(args[0]) if args else _newest_bundle()
    if not bundle or not bundle.exists():
        print(f"[archive] no bundle found. Expected ~/Downloads/rung5_run_<ts>_<sha>.zip "
              f"(Cell 6 of the notebook makes it). Pass a path explicitly if it's elsewhere.")
        return 1
    m = BUNDLE_RE.search(bundle.name)
    if not m:
        print(f"[archive] '{bundle.name}' is not a rung5_run_<UTC-timestamp>_<git-sha>.zip bundle.")
        return 2
    ts, sha = m.group(1), m.group(2)
    folder = ARCHIVE / f"{ts}_{sha}"
    folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        zf.extractall(folder)
    print(f"[archive] {bundle.name}  ->  runs/rung5_logicgate/colab_runs/{ts}_{sha}/  ({len(names)} files)")
    for n in sorted(names):
        print(f"          + {n}")

    mirrored = []
    for name in MIRROR:                                       # update the canonical "current result"
        src = folder / name
        if src.exists():
            shutil.copy2(src, CANONICAL / name)
            mirrored.append(name)
    if mirrored:
        print(f"[archive] mirrored as current result -> runs/rung5_logicgate/: {', '.join(mirrored)}")

    subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "-A",
                    str(folder.relative_to(PROJECT_ROOT)),
                    *[f"runs/rung5_logicgate/{n}" for n in mirrored]], check=False)
    if do_commit:
        msg = (f"RUNG 5: archive Colab run {ts}_{sha} (bit-for-bit, {len(names)} files) + mirror current result")
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "commit", "-q", "-m", msg], check=False)
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "push", "origin", "HEAD"], check=False)
        print(f"[archive] committed + pushed: {msg}")
    else:
        print("[archive] staged. Review with `git status`, then commit, or re-run with --commit to auto-commit+push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
