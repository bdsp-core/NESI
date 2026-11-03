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
  4. Runs the revision analyses and figures from saved intermediates.
  5. Compares every number with claims.csv and writes output/REPORT.md.

Nothing here retrains a model or reads raw EEG unless an intermediate is missing.
See reproduce/README.md for the route from source data to each intermediate.

MANIFEST lists every figure and table; items whose inputs were not saved are
reported as not reproduced.

Environment:
  NESI_DATA_ROOT    folder holding github_moved/, s3/ and drive/
  NESI_BUILD_ROOT   where to build the working copy and write outputs
  NESI_TORCH_PYTHON Python with torch (default: this interpreter)
"""
import argparse
import csv
import hashlib
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

# Every figure and table in the manuscript (current numbering), and how it is produced.
# script: an original repository script run by this file (status comes from that run)
# reproduce: a script in reproduce/ run from saved intermediates
# drawn: a schematic with no data; the final image is in figures/final/
# not reproduced: the inputs were not saved; reported as published
SCRIPT, REPRO, DRAWN, MISSING = "script", "reproduce", "drawn", "not reproduced"
MANIFEST = [
    ("Fig 1", DRAWN, "pipeline schematic; labels updated by figures_relabel.py"),
    ("Fig 2", SCRIPT, "fig2"),
    ("Fig 3", SCRIPT, "fig3"),
    ("Fig 4", SCRIPT, "fig4"),
    ("Fig 5", REPRO, "mortality.py (rebuilt; the original results files had NESI and GCS swapped)"),
    ("Fig 6", REPRO, "figure6_composite.py (saved PaCMAP coordinates and Qwen3-14B indication labels)"),
    ("Fig 7", REPRO, "external_validation.py"),
    ("Fig 8", REPRO, "sedation.py (Christine Eckhardt's sedation code)"),
    ("Table 1", SCRIPT, "table1_rass,table1_gcs,table1_cams,table1_icans"),
    ("Table 2", REPRO, "analyses.py, partly: the full table needs the raw medication administration records"),
    ("Table 3", REPRO, "sedation.py (propofol variance decomposition)"),
    ("Table 4", REPRO, "analyses.py, partly: the full table needs the raw medication administration records"),
    ("Table 5", REPRO, "sedation.py"),
    ("Supp Fig 1", DRAWN, "cohort flow diagram"),
    ("Supp Fig 2", REPRO, "supp_figures_fast.py"),
    ("Supp Fig 3", REPRO, "run_cohort_figures.py"),
    ("Supp Figs 4-5", DRAWN, "schematics"),
    ("Supp Fig 6", SCRIPT, "sfig6"),
    ("Supp Figs 7-9", SCRIPT, "sfig7_9,sfig7"),
    ("Supp Fig 10", REPRO, "supp_fig10_diagnosis.py"),
    ("Supp Figs 11-13", REPRO, "run_cohort_figures.py"),
    ("Supp Figs 14-17", REPRO, "cross_score.py --exact"),
    ("Supp Figs 18-23", MISSING, "example trajectories; the per-recording NESI series were not saved (figures_relabel.py and figures_elapsed_time.py only relabel the published images)"),
    ("Supp Figs 24-25", REPRO, "supp_figures_fast.py"),
    ("Supp Fig 26", REPRO, "ordinal_agreement.py"),
    ("Supp Fig 27", REPRO, "indication_stratified.py"),
    ("Supp Fig 28", REPRO, "mortality.py"),
    ("Supp Fig 29", REPRO, "qeeg_supp_fig31.py"),
    ("Supp Fig 30", MISSING, "qEEG-trained heads; the per-channel qEEG features for the test set were not saved (qeeg_baseline_check.py compares single features)"),
    ("Supp Fig 31", REPRO, "ablation_figure.py"),
    ("Supp Fig 32", REPRO, "icans_segment_ablation.py (numbers; panels are the published ones)"),
    ("Supp Fig 33", REPRO, "loco_summary.py"),
    ("Supp Fig 34", REPRO, "supp_figures_fast.py"),
    ("Supp Fig 35", REPRO, "morgoth_overlap.py"),
    ("Supp Fig 36", REPRO, "external_validation.py"),
    ("Supp Fig 37", REPRO, "sedation.py"),
    ("Supp Table 1", REPRO, "analyses.py (supp_table1)"),
    ("Supp Tables 2-3", REPRO, "analyses.py (supp_tables_2_3)"),
    ("Supp Tables 4-5", DRAWN, "definitions, no computation"),
    ("Supp Tables 6-7", REPRO, "mapping_summary.py (24 retrained mappings)"),
    ("Supp Tables 8-11", REPRO, "sedation.py"),
    ("Supp Table 12", REPRO, "by_diagnosis.py"),
    ("Supp Tables 13-17", REPRO, "sedation.py"),
    ("Supp Table 18", MISSING, "external cohort characteristics, from Northwestern's records"),
    ("Supp Table 19", REPRO, "retrain_summary.py (downsample and upweight retrains)"),
]

# Existing scripts, run unchanged from the working copy
SCRIPTS = {
    "fig2": "figures/main/Codes/FAST_NESIvsBeSpoke_plot.py",
    "fig3": "NESI/ScaleVsNESI/fit_nesi_curves.py",
    "fig4": "figures/main/Codes/FAST_NESI_group_morgothactivation.py",
    "fig6": "figures/main/Codes/Figure6_pacmap_embedding.py",
    "table1_rass": "cohort_models/RASS/Cohort/Table1_RASS.py",
    "table1_gcs": "cohort_models/GCS/Cohort/Table1_GCS.py",
    "table1_cams": "cohort_models/CAMS/Cohort/Table1_CAMS.py",
    "table1_icans": "cohort_models/ICANS/Cohort/Table1_ICANS.py",
    "sfig6": "figures/supplementary_score_prediction/ResultsFig_4OrdinalModels/ScorePredResultsAll.py",
    "sfig7_9": "figures/supplementary_score_prediction/PacMAP_4OrdinalModels/AllPacMAPSConsolidated.py",
    "sfig7": "figures/supplementary_score_prediction/PacMAPS_colored_byMorgoth_NESI_heads/FAST_MorgothNESI_FeatureColored_PacMAPs.py",
}


TORCH_SCRIPTS = {"sfig6", "sfig7_9", "sfig7"}

# Paths the scripts expect -> file name under NESI_DATA_ROOT/s3
S3_LINKS = {
    "cohort_models/GCS/Cohort/GCS_BIDMC_HarvardEEG_metadata.csv": "GCS_BIDMC_HarvardEEG_metadata.csv",
    "cohort_models/CAMS/Cohort/CAMS_cohort_metadata.csv": "CAMS_cohort_metadata.csv",
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
    # Some scripts locate the project root by a pre-restructure RASS/ folder
    (work / "RASS").symlink_to(work / "cohort_models" / "RASS")
    # and the supplementary PaCMAP scripts by its pre-restructure name
    (work / "SupplementaryScorePredictionResults").symlink_to(work / "figures" / "supplementary_score_prediction")
    # Imported but unused by the figure scripts; not in every environment
    (BUILD / "stubs").mkdir(exist_ok=True)
    (BUILD / "stubs" / "torchsummary.py").write_text("def summary(*a, **k):\n    pass\n")
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
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore", PYTHONPATH=str(BUILD / "stubs"), OMP_NUM_THREADS="4")
    for key, rel in SCRIPTS.items():
        path = work / rel
        scan = lambda: {p: p.stat().st_mtime for root in (work, OUT / "figures") for ext in ("*.png", "*.pdf") for p in root.rglob(ext)}
        before = scan()
        t = time.time()
        try:
            r = subprocess.run([TORCH_PY if key in TORCH_SCRIPTS else sys.executable, str(HERE / "_run_script.py"), str(path), str(OUT / "figures" / key)], cwd=path.parent, capture_output=True, text=True, env=env, timeout=timeout)
            (logs / f"{key}.log").write_text(r.stdout[-200000:] + "\n--STDERR--\n" + r.stderr[-20000:])
            new = [str(p) for p, m in scan().items() if before.get(p) != m]
            last = r.stderr.strip().splitlines()[-1][:170] if r.stderr.strip() else ""
            res[key] = dict(ok=r.returncode == 0, seconds=round(time.time() - t), figures=new, error="" if r.returncode == 0 else last)
        except subprocess.TimeoutExpired:
            res[key] = dict(ok=False, seconds=timeout, figures=[], error="timed out")
        print(f"  {key}: {'ok' if res[key]['ok'] else 'FAILED'} {res[key]['seconds']}s {res[key]['error']}", flush=True)
    return res


# Revision analyses and figures. Each reads saved intermediates under NESI_DATA_ROOT/intermediates
# and runs in seconds to minutes; slow steps rebuild an intermediate only if it is missing.
REVISION = [
    ("by_diagnosis", "by_diagnosis.py", "plain"),
    ("rass_positive", "rass_positive.py", "torch"),
    ("artifact_tolerance", "artifact_tolerance.py", "torch"),
    ("retrain_summary", "retrain_summary.py", "torch"),
    ("sedation", "sedation.py", "plain"),
    ("external_validation", "external_validation.py", "torch"),
    ("ordinal_agreement", "ordinal_agreement.py", "torch"),
    ("ablation_figure", "ablation_figure.py", "torch"),
    ("mortality", "mortality.py analyze", "torch"),
    ("fivefold_wilcoxon", "fivefold_wilcoxon.py", "torch"),
    ("icans_subgroups", "icans_subgroups.py", "torch"),
    ("morgoth_overlap", "morgoth_overlap.py", "torch"),
    ("eeg_type", "eeg_type.py", "torch"),
    ("supp_figures_fast", "supp_figures_fast.py", "torch"),
    ("run_cohort_figures", "run_cohort_figures.py", "torch"),
    ("figures_final", "figures_final.py", "plain"),
    ("figures_relabel", "figures_relabel.py", "torch"),
    ("figures_elapsed_time", "figures_elapsed_time.py", "torch"),
    ("graphical_abstract", "graphical_abstract.py", "torch"),
    ("cross_score", "cross_score.py --exact", "torch"),
    ("qeeg_supp_fig31", "qeeg_supp_fig31.py", "plain"),
    ("loco_summary", "loco_summary.py", "plain"),
    ("mapping_summary", "mapping_summary.py", "plain"),
    ("icans_segment_ablation", "icans_segment_ablation.py", "plain"),
    ("indication_stratified", "indication_stratified.py", "plain"),
    ("supp_fig10_diagnosis", "supp_fig10_diagnosis.py", "plain"),
    ("figure6_composite", "figure6_composite.py", "plain"),
]


def run_revision():
    res, logs = {}, OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    for key, script, kind in REVISION:
        script, *args = script.split()
        if not (HERE / script).exists():
            res[key] = "script not in this checkout"
            continue
        t = time.time()
        r = subprocess.run([TORCH_PY if kind == "torch" else sys.executable, str(HERE / script), *args], capture_output=True, text=True, env=env)
        (logs / f"{key}.log").write_text(r.stdout[-200000:] + "\n--STDERR--\n" + r.stderr[-20000:])
        last = r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else ""
        res[key] = f"ok {time.time() - t:.0f}s" if r.returncode == 0 else f"FAILED {last}"
        print(f"  {key}: {res[key]}", flush=True)
    return res


def freeze_intermediates(write):
    """Record or check a checksum for every saved intermediate."""
    man = ROOT / "intermediates" / "MANIFEST.csv"
    files = [p for p in sorted((ROOT / "intermediates").rglob("*")) if p.is_file() and p != man]
    files += sorted((ROOT.parent / "retrain").glob("*/test_scores.csv"))
    now = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if write:
        with open(man, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "sha256"])
            w.writerows(now.items())
        return f"manifest written for {len(now)} files"
    if not man.exists():
        return "no manifest; run with --freeze"
    old = {r["path"]: r["sha256"] for r in csv.DictReader(open(man))}
    changed = [p for p in old if now.get(p) != old[p]]
    return f"{len(old) - len(changed)} of {len(old)} intermediates unchanged" + (f"; changed: {changed[:5]}" if changed else "")


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
    ap.add_argument("--only", choices=["numbers", "torch", "scripts", "revision"])
    ap.add_argument("--freeze", action="store_true", help="record checksums of the saved intermediates")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for item, kind, detail in MANIFEST:
            print(f"{kind:15} {item:16} {detail}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    num, log, torch_log, scripts, revision = {}, {}, "not run", {}, {}
    if a.only in (None, "scripts"):
        scripts = run_scripts(build_workspace())
    if a.only in (None, "revision"):
        revision = run_revision()
    if a.only in (None, "numbers"):
        num, log = run_numbers(a.boot)
    if a.only in (None, "torch"):
        extra, torch_log = run_torch()
        print("  torch_checks:", torch_log)
        num.update(extra)
    json.dump(num, open(OUT / "numbers.json", "w"), indent=1)
    log.update(revision)
    log["intermediates"] = freeze_intermediates(a.freeze)
    report(check_claims(num), log, torch_log, scripts)


if __name__ == "__main__":
    main()
