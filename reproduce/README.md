# Reproducing the manuscript

`run_all.py` is the single entry point. It builds a working copy of the code with the credentialed data linked in, recomputes manuscript numbers, runs the existing figure and table scripts, and compares each registered number with its manuscript value in `claims.csv`.

```bash
export NESI_DATA_ROOT=/path/to/data        # holds github_moved/, s3/ and drive/
export NESI_BUILD_ROOT=/path/to/build      # optional; working copy and output/
export NESI_TORCH_PYTHON=/path/to/python   # optional; a Python with torch
python reproduce/run_all.py                # everything available
python reproduce/run_all.py --list         # every figure and table, with status
```

Results are written to `$NESI_BUILD_ROOT/output/`: `REPORT.md`, `claims_checked.csv`, `numbers.json`, `tables/` and `logs/`.

## Data

`github_moved/` mirrors the paths in [`docs/DATA_FILES.md`](../docs/DATA_FILES.md). `s3/` holds files from `s3://bdsp-opendata-credentialed/yama/`. `drive/` holds `i0001_RASS_Meds_NESI.csv`, which is not yet in the S3 release.

## Current coverage

This is a work in progress. Some manuscript analyses have no code in this repository yet; `--list` shows them as `missing`, and the report shows them on every run. `claims.csv` registers only part of the numbers in the manuscript. A `DIFF` in the report means a recomputed value differs from the manuscript and needs review.
