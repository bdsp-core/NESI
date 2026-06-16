"""
Batch-render EEG + multitaper-spectrogram PNGs for every unique
MorgothOutputFilename in the PaCMAP coords (CAMS, ICANS, RASS).

  • Streams each .mat from s3://bdsp-opendata-credentialed/yama/...
  • Renders one PNG via poc_render.render_segment_png
  • Deletes the temp .mat to keep disk usage bounded
  • Resumable: skips files whose output PNG already exists
  • Parallel via ProcessPoolExecutor (BLAS pinned to 1 thread per worker)

Output: NESI/InteractiveMap/eeg_pngs/{Dataset}/{stem}.png
"""
from __future__ import annotations
import os

# Pin BLAS to 1 thread BEFORE numpy import so ProcessPoolExecutor workers
# don't oversubscribe CPU cores.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "eeg_pngs"
COORDS_CSV = (SCRIPT_DIR.parent / "MorgothFeatureEmbedding"
              / "NESI_pacmap_coords.csv")

BUCKET = "bdsp-opendata-credentialed"
AWS_PROFILE = "opendata"

DATASET_S3_PREFIX = {
    "CAMS":  "yama/cohort_models/CAMS/CAMS_10minEEGSegments/",
    "ICANS": "yama/cohort_models/ICANS/ICANS_10minEEGSegments/",
    "RASS":  "yama/cohort_models/RASS/RASS_EEG10minSegments/",
}


def _stem(fname_csv: str) -> str:
    return fname_csv[:-4] if fname_csv.endswith(".csv") else fname_csv


def s3_key_for(dataset: str, fname_csv: str) -> str:
    return DATASET_S3_PREFIX[dataset] + _stem(fname_csv) + ".mat"


def out_png_for(dataset: str, fname_csv: str) -> Path:
    return OUT_DIR / dataset / (_stem(fname_csv) + ".png")


# One boto3 S3 client per worker process (re-used across tasks),
# configured to download each .mat in 4 parallel parts.
_s3 = None
_s3_cfg = None


def _get_s3():
    global _s3, _s3_cfg
    if _s3 is None:
        import boto3
        from boto3.s3.transfer import TransferConfig
        _s3 = boto3.Session(profile_name=AWS_PROFILE).client("s3")
        _s3_cfg = TransferConfig(
            max_concurrency=4,
            multipart_threshold=4 * 1024 * 1024,
            multipart_chunksize=4 * 1024 * 1024,
            use_threads=True,
        )
    return _s3, _s3_cfg


def render_one(args):
    """Worker entry point: download, render, cleanup. Returns
    (status, dataset, filename, error_msg or None)."""
    dataset, fname_csv, nesi = args
    out_path = out_png_for(dataset, fname_csv)
    if out_path.exists():
        return ("skip", dataset, fname_csv, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Import poc_render lazily so the parent process doesn't load matplotlib
    # before forking workers.
    sys.path.insert(0, str(SCRIPT_DIR))
    from poc_render import render_segment_png  # noqa: E402

    key = s3_key_for(dataset, fname_csv)
    stem = _stem(fname_csv)
    title = f"{dataset} (NESI={nesi:+.2f}) — {stem}"

    tmp = Path(tempfile.mkstemp(suffix=".mat",
                                  prefix=f"yama_{dataset}_")[1])
    try:
        s3, cfg = _get_s3()
        s3.download_file(BUCKET, key, str(tmp), Config=cfg)
        render_segment_png(tmp, out_path, title=title)
        return ("ok", dataset, fname_csv, None)
    except Exception as e:
        # Don't leave a partial PNG on disk; clean up so retry works.
        try: out_path.unlink(missing_ok=True)
        except Exception: pass
        return ("err", dataset, fname_csv, f"{type(e).__name__}: {e}")
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel worker processes (each uses 4 download "
                         "threads, so total S3 streams = workers * 4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="render at most N segments (for smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be done; don't download or render")
    args = ap.parse_args()

    coords = pd.read_csv(COORDS_CSV)
    todo_df = (coords[coords.Dataset.isin(DATASET_S3_PREFIX)]
                .drop_duplicates("MorgothOutputFilename")
                .sort_values(["Dataset", "MorgothOutputFilename"]))
    todo = list(todo_df[["Dataset", "MorgothOutputFilename", "NESI"]]
                .itertuples(index=False, name=None))
    pending = [t for t in todo if not out_png_for(t[0], t[1]).exists()]
    print(f"Total unique segments: {len(todo)}   "
          f"pending: {len(pending)}   "
          f"already done: {len(todo) - len(pending)}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"--limit {args.limit}: processing {len(pending)} segments")

    if args.dry_run:
        for t in pending[:20]:
            print("  would render:", t[0], _stem(t[1]))
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    if not pending:
        print("Nothing to do.")
        return

    t0 = time.time()
    done = errors = 0
    err_lines = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, t): t for t in pending}
        for fut in as_completed(futures):
            status, ds, fn, err = fut.result()
            done += 1
            if status == "err":
                errors += 1
                err_lines.append(f"{ds}/{fn}: {err}")
            if done % 25 == 0 or done == len(pending):
                rate = done / max(time.time() - t0, 1e-9)
                eta_s = (len(pending) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(pending)}  "
                      f"{rate:.2f} seg/s  "
                      f"ETA {eta_s/60:.1f} min  "
                      f"errors={errors}")
    print(f"\nDone in {(time.time() - t0)/60:.1f} min   "
          f"errors={errors}")
    if err_lines:
        log_path = SCRIPT_DIR / "batch_render_errors.log"
        log_path.write_text("\n".join(err_lines))
        print(f"Error details -> {log_path}")


if __name__ == "__main__":
    main()
