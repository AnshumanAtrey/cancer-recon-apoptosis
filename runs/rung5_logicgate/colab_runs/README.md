# RUNG 5 — Colab run archive (bit-for-bit, one folder per run)

Every real Colab run of `scripts/25_logicgate_data_rung5.py` is archived here, **complete and bit-for-bit**,
in its own folder. No file ever overwrites another — each run is immutable history.

## Folder naming

```
<UTC-timestamp>_<git-sha>/
```
matching the run's runlog name (`rung5_addressability_<UTC-timestamp>_<git-sha>.log`), so the folder, the log,
and the exact code commit that produced the run are all linked. `git show <sha>` reproduces the code state.

## What's inside each folder

Whatever that run produced + downloaded (Cell 6 pulls all of it), with Chrome's ` (N)` dedup-suffix stripped:
- `rung5_addressability_<ts>_<sha>.log` — the full commit-stamped run log (every step + RAM trajectory)
- `rung5_addressability.json` — the real result (per-patient addressability gap, top gates, leaks)
- `rung5_real.png` — the result figure (gate frontier + gap-by-cancer-type)
- `surfaceome_genes.txt` — the exact gene set screened that run
- `rung5_selftest.{png,json}`, `rung5_heldout_validation.png`, `heldout_validation.json` — the Cell-3
  validation outputs reproduced on that runtime

A run that crashed before finishing (e.g. an OOM) contains only the files it got to produce — typically just
the log (which is exactly what documents the crash).

## The runs so far

| folder (UTC_sha) | what happened |
|---|---|
| `20260601T152943Z_fd23b06` | first real run — OOM (exit -9) at the brain materialise |
| `20260601T213723Z_9e28c3d` | streaming scout — OOM at the brain scout |
| `20260602T060537Z_c12b437` | Arrow-code scout — **full fetch OK** (brain 10.5M→407k), OOM at scoring |
| `20260604T212700Z_5c2bfed` | fast scorer — fetch + FDR ran, OOM in the bootstrap |
| `20260605T112400Z_6ec9d3b` | **the completing run** — full result: 0/1000 gates worst-case-safe, gap = 100% |

The latest run's `rung5_addressability.json` + `rung5_real.png` are also mirrored one level up
(`runs/rung5_logicgate/`) as the canonical "current result"; this folder is the immutable full archive.
