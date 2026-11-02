#!/usr/bin/env python3
"""Master reproduction script for the NESI manuscript.

    python reproduce/run_all.py                 # run everything available
    python reproduce/run_all.py --list          # show every item and its status
    python reproduce/run_all.py --only numbers  # numbers | torch | scripts

It does four things, in order:
  1. Builds a working copy of the code with the credentialed data linked in at
     the paths the existing scripts expect (see docs/DATA_FILES.md).
  2. Recomputes manuscript numbers (analyses.py, torch_checks.py).
  3. Runs the existing figure and table scripts.
  4. Compares every number with claims.csv and writes output/REPORT.md.

Items in MANIFEST with no script are reported as MISSING, so the report always
shows what still cannot be reproduced.

Environment:
  NESI_DATA_ROOT    folder holding github_moved/, s3/ and drive/
  NESI_BUILD_ROOT   where to build the working copy and write outputs
  NESI_TORCH_PYTHON Python with torch (default: this interpreter)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
if "NESI_DATA_ROOT" not in os.environ:
    sys.exit("Set NESI_DATA_ROOT to the folder holding github_moved/, s3/ and drive/ (see reproduce/README.md)")
ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
TORCH_PY = os.environ.get("NESI_TORCH_PYTHON", sys.executable)
OUT = BUILD / "output"

# (item, how it is produced, owner of anything missing)
MISSING_ARKA = "Arka"
MISSING_CE = "Christine"
SCRIPT, NUMBERS, DRAWN, MISSING = "script", "numbers", "drawn", "missing"
MANIFEST = [
    ("Fig 1", DRAWN, "figures/main/Figure1.drawio.png (needs CAM-S and estimation-head relabel)"),
    ("Fig 2", SCRIPT, "fig2"),
    ("Fig 3", SCRIPT, "fig3"),
    ("Fig 4", SCRIPT, "fig4"),
    ("Fig 5 (six adjusted models)", MISSING, MISSING_ARKA),
    ("Fig 6 (revised panels)", SCRIPT, "fig6"),
    ("Fig 7 (external validation)", MISSING, MISSING_ARKA + "/Erika"),
    ("Fig 8 (agent groups)", MISSING, MISSING_CE),
    ("Table 1", SCRIPT, "table1_rass,table1_gcs,table1_cams,table1_icans"),
    ("Table 2", NUMBERS, "RASS column partly; medication-exposure pipeline needs raw MAR"),
    ("Table 3", NUMBERS, "propofol"),
    ("Table 4", MISSING, MISSING_ARKA + "/" + MISSING_CE),
    ("Table 5", MISSING, MISSING_CE),
    ("Supp Fig 1 (consort)", MISSING, MISSING_ARKA),
    ("Supp Figs 2-3", MISSING, MISSING_ARKA),
    ("Supp Fig 4-5 (schematics)", DRAWN, "not in repo"),
    ("Supp Fig 6", SCRIPT, "sfig6"),
    ("Supp Figs 7-9", SCRIPT, "sfig7_9"),
    ("Supp Figs 10-17", MISSING, MISSING_ARKA),
    ("Supp Figs 18-23 (trajectories)", MISSING, MISSING_ARKA),
    ("Supp Figs 24-27", MISSING, MISSING_ARKA),
    ("Supp Figs 28-30 (mortality)", MISSING, MISSING_ARKA),
    ("Supp Figs 31-32 (qEEG)", MISSING, MISSING_ARKA),
    ("Supp Fig 33 (ablation)", NUMBERS, "torch; figure itself not regenerated"),
    ("Supp Figs 34-37", MISSING, MISSING_ARKA),
    ("Supp Fig 38 (external timing)", MISSING, MISSING_ARKA + "/Erika"),
    ("Supp Fig 39 (agent offsets)", MISSING, MISSING_CE),
    ("Supp Table 1", NUMBERS, "supp_table1"),
    ("Supp Table 2", NUMBERS, "supp_tables_2_3; ECE binning unspecified"),
    ("Supp Table 3", NUMBERS, "supp_tables_2_3"),
    ("Supp Table 4-5 (definitions)", DRAWN, "no computation"),
    ("Supp Tables 6-7 (24 mappings)", MISSING, MISSING_ARKA),
    ("Supp Tables 8-10 (agent offsets)", MISSING, MISSING_CE),
    ("Supp Table 11", NUMBERS, "supp_table11"),
    ("Supp Table 12", NUMBERS, "institutions_age_duration; CAM-S rows missing"),
]

# Existing scripts, run unchanged from the working copy
SCRIPTS = {
    "fig2": "figures/main/Codes/FAST_NESIvsBeSpoke_plot.py",
    "fig3": "NESI/ScaleVsNESI/fit_nesi_curves.py",
    "fig4": "figures/main/Codes/FAST_NESI_group_morgothactivation.py",
    "fig5_old": "figures/main/Codes/Test_results_plot_GCSvsNESI.py",
    "fig6": "figures/main/Codes/Figure6_pacmap_embedding.py",
    "table1_rass": "cohort_models/RASS/Cohort/Table1_RASS.py",
    "table1_gcs": "cohort_models/GCS/Cohort/Table1_GCS.py",
    "table1_cams": "cohort_models/CAMS/Cohort/Table1_CAMS.py",
    "table1_icans": "cohort_models/ICANS/Cohort/Table1_ICANS.py",
    "sfig6": "figures/supplementary_score_prediction/ResultsFig_4OrdinalModels/ScorePredResultsAll.py",
    "sfig7_9": "figures/supplementary_score_prediction/PacMAP_4OrdinalModels/AllPacMAPSConsolidated.py",
}


# Paths the scripts expect -> file name under NESI_DATA_ROOT/s3
S3_LINKS = {
    "cohort_models/GCS/Cohort/GCS_BIDMC_HarvardEEG_metadata.csv": "GCS_BIDMC_HarvardEEG_metadata.csv",
    "cohort_models/CAMS/Cohort/CAMS_cohort_metadata.csv": "CAMS_cohort_metadata.csv",
    "NESI/Bespoke_models/Results/NESIscores_full_dataset_results.csv": "NESIscores_full_dataset_results.csv",
    "cohort_models/ICANS/Cohort/ICANS_cohort_metadata.csv": "ICANS_cohort_metadata.csv",
    "DiagnosisMetadtafiles/File1.xlsx": "File1.xlsx",
    "DiagnosisMetadtafiles/File2.xlsx": "File2.xlsx",
    "DiagnosisMetadtafiles/File3.xlsx": "File3.xlsx",
}


# Scripts expect a moved file under a second path as well
GH_LINKS = {"NESI/MorgothFeatureEmbedding/NESI_pacmap_coords.csv": "NESI/PacMAP/NESI_pacmap_coords.csv"}


def build_workspace():
    """Copy code without .git, then link data back at its original paths."""
    work = BUILD / "NESI"
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "output"))
    linked = 0
    for base in (ROOT / "github_moved",):
        for src in base.rglob("*"):
            if src.is_file() and src.name != "MANIFEST.csv":
                dst = work / src.relative_to(base)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src)
                    linked += 1
    # Files the scripts expect locally but which were only ever on S3
    for rel, name in S3_LINKS.items():
        src, dst = ROOT / "s3" / name, work / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            linked += 1
    for rel, src in GH_LINKS.items():
        src, dst = ROOT / "github_moved" / src, work / rel
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
            linked += 1
    print(f"workspace: {work} ({linked} data links)")
    return work


def run_numbers(boot):
    import analyses

    ctx = analyses.Context(REPO, ROOT, OUT, boot)
    log = {}
    for fn in analyses.ALL:
        t = time.time()
        try:
            fn(ctx)
            log[fn.__name__] = f"ok {time.time() - t:.0f}s"
        except Exception as e:
            log[fn.__name__] = f"FAILED {type(e).__name__}: {str(e)[:160]}"
        print(f"  {fn.__name__}: {log[fn.__name__]}", flush=True)
    return {k: float(v) for k, v in ctx.num.items()}, log


def run_torch():
    tmp = OUT / "torch_numbers.json"
    env = dict(os.environ, OMP_NUM_THREADS="4")
    r = subprocess.run([TORCH_PY, str(HERE / "torch_checks.py"), str(REPO), str(ROOT / "s3"), str(tmp)], capture_output=True, text=True, env=env)
    if r.returncode:
        return {}, f"FAILED {r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else r.returncode}"
    return json.load(open(tmp)), "ok"


def run_scripts(work, timeout=900):
    res, logs = {}, OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    for key, rel in SCRIPTS.items():
        path = work / rel
        scan = lambda: {p: p.stat().st_mtime for ext in ("*.png", "*.pdf") for p in work.rglob(ext)}
        before = scan()
        t = time.time()
        try:
            r = subprocess.run([sys.executable, path.name], cwd=path.parent, capture_output=True, text=True, env=env, timeout=timeout)
            (logs / f"{key}.log").write_text(r.stdout[-200000:] + "\n--STDERR--\n" + r.stderr[-20000:])
            new = [str(p.relative_to(work)) for p, m in scan().items() if before.get(p) != m]
            last = r.stderr.strip().splitlines()[-1][:170] if r.stderr.strip() else ""
            res[key] = dict(ok=r.returncode == 0, seconds=round(time.time() - t), figures=new, error="" if r.returncode == 0 else last)
        except subprocess.TimeoutExpired:
            res[key] = dict(ok=False, seconds=timeout, figures=[], error="timed out")
        print(f"  {key}: {'ok' if res[key]['ok'] else 'FAILED'} {res[key]['seconds']}s {res[key]['error']}", flush=True)
    return res


def check_claims(num):
    rows = []
    with open(HERE / "claims.csv", newline="") as f:
        for c in csv.DictReader(f):
            got = num.get(c["key"])
            exp, tol = float(c["expected"]), float(c["tolerance"])
            if got is None:
                status, shown = "NOT_COMPUTED", ""
            else:
                status, shown = ("PASS" if abs(abs(got) - abs(exp)) <= tol else "DIFF"), f"{got:.4g}"
            rows.append({**c, "recomputed": shown, "status": status})
    with open(OUT / "claims_checked.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def report(rows, log, torch_log, scripts):
    from collections import Counter

    c = Counter(r["status"] for r in rows)
    L = ["# NESI reproduction report", "", f"Run {time.strftime('%Y-%m-%d %H:%M')}. Data root: `{ROOT}`.", ""]
    L += [f"**Registered numbers:** {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()), ""]
    L += ["## Numbers that differ from the manuscript", "", "| Where | Claim | Paper | Recomputed |", "|---|---|---:|---:|"]
    L += [f"| {r['location']} | {r['description']} | {r['expected']} | {r['recomputed']} |" for r in rows if r["status"] == "DIFF"]
    L += ["", "## Analysis steps", ""] + [f"- `{k}`: {v}" for k, v in log.items()] + [f"- `torch_checks`: {torch_log}", ""]
    L += ["## Existing scripts", "", "| Script | Result | Figures written | Error |", "|---|---|---|---|"]
    L += [f"| `{k}` | {'ok' if v['ok'] else 'FAILED'} | {len(v['figures'])} | {v['error']} |" for k, v in scripts.items()]
    L += ["", "## Every figure and table", "", "| Item | Status | Detail |", "|---|---|---|"]
    for item, kind, detail in MANIFEST:
        st = kind
        if kind == SCRIPT:
            runs = [scripts.get(k, {}) for k in detail.split(",")]
            st = "FAILED" if not all(r.get("ok") for r in runs) else ("ok" if item.startswith("Table") or all(r["figures"] for r in runs) else "ran, wrote no figure")
        L.append(f"| {item} | {st} | {detail} |")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nReport: {OUT / 'REPORT.md'}\nNumbers: {dict(c)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["numbers", "torch", "scripts"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for item, kind, detail in MANIFEST:
            print(f"{kind:8} {item:38} {detail}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    num, log, torch_log, scripts = {}, {}, "not run", {}
    if a.only in (None, "scripts"):
        scripts = run_scripts(build_workspace())
    if a.only in (None, "numbers"):
        num, log = run_numbers(a.boot)
    if a.only in (None, "torch"):
        extra, torch_log = run_torch()
        print("  torch_checks:", torch_log)
        num.update(extra)
    json.dump(num, open(OUT / "numbers.json", "w"), indent=1)
    report(check_claims(num), log, torch_log, scripts)


if __name__ == "__main__":
    main()
