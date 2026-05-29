#!/usr/bin/env python3
"""
runlog — capture script output to a timestamped, commit-stamped log file.

Purpose: make every Colab run reproducible and shareable. Instead of copy-pasting
cell output by hand, each run is teed to BOTH the cell display and a file:

    runs/logs/<step>_<UTC-timestamp>_<gitSHA>.log

The git commit SHA is in the filename AND the header, so the output is intrinsically
linked to the exact code version that produced it — regardless of which commit later
stores the log. Download it from Colab and share it; commit it to runs/logs/.

Usage (in a Colab notebook, after the repo is cloned):

    import sys
    from scripts.runlog import new_log, run_logged, finalize
    RUNLOG = new_log("step2")                              # one log per session
    run_logged([sys.executable, "-u", "scripts/03_census_fetch.py", "explore"], RUNLOG)
    run_logged([sys.executable, "-u", "scripts/03_census_fetch.py", "fetch"],   RUNLOG)
    finalize(RUNLOG)                                       # prints path + downloads
"""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path


def _git(repo_dir: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _utc_stamp() -> str:
    # timezone-aware UTC (utcnow() is deprecated in 3.12+)
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_log(step: str, repo_dir: str | Path = ".", logs_dir: str | Path | None = None) -> Path:
    """Create a fresh run-log file with a header. Returns its Path."""
    repo_dir = Path(repo_dir).resolve()
    short = _git(repo_dir, "rev-parse", "--short", "HEAD") or "nogit"
    full = _git(repo_dir, "rev-parse", "HEAD") or "nogit"
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = bool(_git(repo_dir, "status", "--porcelain"))
    ts = _utc_stamp()

    logs = Path(logs_dir) if logs_dir else repo_dir / "runs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{step}_{ts}_{short}.log"

    header = [
        "# cancer-recon-apoptosis — run log",
        f"# step          : {step}",
        f"# utc           : {ts}",
        f"# commit_short  : {short}{'  (+uncommitted changes)' if dirty else ''}",
        f"# commit_full   : {full}",
        f"# branch        : {branch}",
        f"# python        : {platform.python_version()}",
        f"# platform      : {platform.platform()}",
        "#" + "=" * 64,
        "",
    ]
    path.write_text("\n".join(header))
    print(f"[runlog] writing → {path}  (commit {short}{'+dirty' if dirty else ''})")
    return path


def run_logged(cmd: list[str], log_path: str | Path, label: str | None = None) -> int:
    """Run `cmd`, tee each line to stdout AND append to `log_path`. Returns exit code."""
    log_path = Path(log_path)
    with open(log_path, "a") as f:
        banner = f"\n$ {label or ' '.join(cmd)}\n"
        sys.stdout.write(banner); sys.stdout.flush(); f.write(banner); f.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except FileNotFoundError as e:
            msg = f"[runlog] command not found: {e}\n"
            sys.stdout.write(msg); f.write(msg)
            return 127
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        rc = proc.wait()
        tail = f"[exit {rc}]\n"
        sys.stdout.write(tail); sys.stdout.flush(); f.write(tail); f.flush()
    return rc


def finalize(log_path: str | Path, download: bool = True) -> Path:
    """Print the log path and (in Colab) trigger a browser download."""
    log_path = Path(log_path)
    size = log_path.stat().st_size if log_path.exists() else 0
    print(f"\n[runlog] complete → {log_path}  ({size} bytes)")
    print(f"[runlog] share this file; commit it to runs/logs/ to link output ↔ commit")
    if download:
        try:
            from google.colab import files  # type: ignore
            files.download(str(log_path))
        except ImportError:
            print("[runlog] (not in Colab — download skipped)")
        except Exception as e:
            print(f"[runlog] (download skipped: {type(e).__name__}: {e})")
    return log_path


if __name__ == "__main__":
    # tiny self-test: log an echo and finalize without download
    p = new_log("selftest")
    run_logged([sys.executable, "-c", "print('hello from runlog')"], p)
    finalize(p, download=False)
