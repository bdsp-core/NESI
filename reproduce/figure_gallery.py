#!/usr/bin/env python3
"""Side-by-side review page: every manuscript figure as published and as regenerated.

    python reproduce/figure_gallery.py PUBLISHED_MAIN.docx PUBLISHED_SUPP.docx CURRENT_MAIN_DIR CURRENT_SUPP_DIR

The two .docx files are the versions before revision; the two directories are the
unpacked current documents. Writes $NESI_BUILD_ROOT/output/figure_review/index.html.
"""
import html
import os
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(os.environ["NESI_DATA_ROOT"])
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build"))
FIG = BUILD / "output" / "figures"
WORK = BUILD / "NESI"
OUT = BUILD / "output" / "figure_review"
W_ = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R_ = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

# figure: (regenerated files, status, how it is made)
MAIN = {
    "1": (["final/Fig1.png"], "Relabeled", "Drawn schematic, no data. CAM-S and estimation-head labels and the direction of NESI added by figures_relabel.py."),
    "2": (["final/Fig2.png"], "Regenerated", "figures_final.py. Boxes and n match; CIs now patient-level, matching Results."),
    "3": (["final/Fig3.png", "_diff_Fig3.png"], "Regenerated", "figures_final.py (fit_nesi_curves.py). Second image: pixel difference from the published figure."),
    "4": (["final/Fig4.png", "_diff_Fig4.png"], "Regenerated", "figures_final.py. Second image: pixel difference from the published figure."),
    "5": (["mortality_auroc_available_covariates.png"], "Rebuilt, changed", "mortality.py. The published Fig 5 came from mislabeled NESI and GCS results; rebuilt with the covariates available."),
    "6": (["@figures/main/Figure6.png"], "Not reproducible yet", "The repo's Figure6_pacmap_embedding.py makes an earlier two-panel layout. The published six-panel figure comes from NESI_composite_pacmap.py, which needs Arka's indication files."),
    "7": (["final/Fig7.png"], "Regenerated, matches", "external_validation.py from Erika's files. Values and n match the published figure."),
    "8": (["sedation/figure_sedation_combined_stratum.png"], "Regenerated, CIs pending", "sedation.py (Christine's code). Panel a matches; CIs here are from a test run until the 1,000-resample AWS run finishes."),
}
SUPP = {
    "1": ([], "Not regenerated", "CONSORT diagram, drawn by hand."),
    "2": (["SuppFig2_class_distributions.png"], "Regenerated", "supp_figures_fast.py."),
    "3": (["cohort/RASS_Swimmers_plot_1.png", "cohort/GCSSwimmersPlot_1.png"], "Regenerated; GCS axis differs", "run_cohort_figures.py. RASS panel matches. The repo's GCS script draws a log time axis where the published panel is linear; same data. Caption n corrected to 6,191 and 7,112."),
    "4": ([], "Not regenerated", "Schematic."),
    "5": ([], "Not regenerated", "Schematic."),
    "6": (["sfig6/ScorePredResultsAll_1.png"], "Regenerated, matches", "ScorePredResultsAll.py, run unchanged by run_all.py. Matches by eye."),
    "7": (["sfig7/FAST_MorgothNESI_FeatureColored_PacMAPs_1.png"], "Regenerated, matches", "FAST_MorgothNESI_FeatureColored_PacMAPs.py. Matches by eye."),
    "8": (["sfig7_9/AllPacMAPSConsolidated_1.png"], "Regenerated, matches", "AllPacMAPSConsolidated.py. Matches by eye."),
    "9": (["final/SuppFig9.png"], "Relabeled", "CAM-S label only."),
    "10": ([], "Not reproducible yet", "Needs the diagnosis CSV from Arka's machine."),
    "11": (["cohort/RASS_results_MAE_1levelACC_Corr_1.png"], "Regenerated, matches", "RASS_results_MAE_1levelACC_Corr.py. Matches by eye."),
    "12": (["cohort/CAMS_Result_Compare_3class_3.png", "final/SuppFig12.png"], "Regenerated, matches", "CAMS_Result_Compare_3class.py. Matches by eye; the second image is the published panel relabeled CAM-S."),
    "13": (["cohort/ICANS_Results_Compare_3.png"], "Regenerated, matches", "ICANS_Results_Compare.py. Matches by eye."),
    "14": (["final/SuppFig14_exact.png"], "Regenerated, corrected", "cross_score.py --exact. Matches the published figure box for box when run the original way; this version chains the ordinal model's conditional outputs correctly."),
    "15": (["final/SuppFig15_exact.png"], "Regenerated, corrected", "cross_score.py --exact. Matches the published figure box for box when run the original way; this version chains the ordinal model's conditional outputs correctly."),
    "16": (["final/SuppFig16_exact.png"], "Regenerated, corrected", "cross_score.py --exact. Matches the published figure box for box when run the original way; this version chains the ordinal model's conditional outputs correctly, and fixes the swapped Mild and Severe row labels."),
    "17": (["final/SuppFig17_exact.png"], "Regenerated, corrected", "cross_score.py --exact. Matches the published figure when run the original way; this version chains the ordinal model's conditional outputs correctly."),
    **{str(k): ([f"final/SuppFig{k}.png"], "Relabeled", "Example trajectory. Calendar dates replaced by elapsed hours; data not regenerated (needs MORGOTH on full recordings).") for k in range(18, 24)},
    "24": (["SuppFig24_calibration.png"], "Regenerated", "supp_figures_fast.py. Thresholds and AUROCs match exactly."),
    "25": (["SuppFig25_roc.png"], "Regenerated", "supp_figures_fast.py. Thresholds and AUROCs match exactly."),
    "26": (["supp_fig26.png"], "Regenerated, changed", "ordinal_agreement.py. Midpoint cutoff rule now stated; three values changed slightly."),
    "27": (["final/SuppFig27.png"], "Relabeled", "Indication labels came from an LLM; counts can't be regenerated without Arka's files."),
    "28": (["mortality_decision_curve.png"], "New", "mortality.py. Replaces the likelihood-ratio figure, which came from the mislabeled analysis."),
    "29": ([], "Removed", "Calibration of the mislabeled mortality models."),
    "30": ([], "Removed", "NRI and IDI of the mislabeled mortality models."),
    "31": (["final/SuppFig31.png"], "Relabeled", "qEEG comparison; prints segment-level CIs. Not regenerated."),
    "32": ([], "Partly checked", "qeeg_baseline_check.py: for RASS and GCS, NESI (0.75, 0.70) far exceeds every qEEG feature (|rho| at most 0.13); CAM-S and ICANS not yet read."),
    "33": (["supp_fig33.png"], "Regenerated, changed", "ablation_figure.py. Baseline heads retrained on NESI's own split (the published figure used another split); panel a counts corrected."),
    "34": ([], "Not reproducible yet", "Needs MORGOTH on full recordings."),
    "35": ([], "Retraining", "Leave-one-cohort-out retrains are running on AWS."),
    "36": (["SuppFig36_threshold_auroc.png"], "Regenerated", "supp_figures_fast.py. Matches."),
    "37": (["final/SuppFig37.png"], "Regenerated, changed", "morgoth_overlap.py. Excludes all 193 overlapping patients; patient-level CIs."),
    "38": (["final/SuppFig38.png"], "Regenerated, matches", "external_validation.py. The published figure is kept; values agree within 0.01."),
    "39": (["sedation/offset_by_level_panels_stratum.png"], "Regenerated, CIs pending", "sedation.py (Christine's code). Test-run CIs until the AWS run finishes."),
}
MAIN_MEDIA = {"1": "image6", "2": "image3", "3": "image2", "4": "image1", "5": "image4", "6": "image5", "7": "added901", "8": "added902"}


def supp_media(unpacked, min_side=300):
    """Map each supplementary figure number to the image in or just before its caption paragraph."""
    from lxml import etree

    rels = (unpacked / "word/_rels/document.xml.rels").read_text()
    target = {a["Id"]: a["Target"].split("/")[-1] for a in (dict(re.findall(r'(\w+)="([^"]*)"', y)) for y in re.findall(r"<Relationship\b[^>]*>", rels))}
    ps = list(etree.parse(str(unpacked / "word/document.xml")).getroot().iter(W_ + "p"))
    out = {}
    for i, p in enumerate(ps):
        text = "".join(t.text or "" for t in p.iter(W_ + "t")).strip()
        m = re.match(r"\[?\]?Supplementary Fig\. (\d+)", text)
        if not m:
            continue
        used = set(out.values())
        for q in [p] + [ps[i - j] for j in range(1, 5)]:
            ids = [target[e.get(R_)].rsplit(".", 1)[0] for e in q.iter() if e.get(R_)]
            ids = [x for x in ids if x not in used and big(unpacked, x, min_side)]
            if ids:
                out.setdefault(m[1], ids[0])
                break
    return out


def big(unpacked, stem, min_side):
    """Skip inline equation images, which Word stores as small pictures."""
    hits = list((unpacked / "word/media").glob(stem + ".*"))
    if not hits:
        return True
    try:
        return min(Image.open(hits[0]).size) >= min_side
    except Exception:
        return False


def thumb(src, name):
    im = Image.open(src).convert("RGB")
    im.thumbnail((1800, 1800))
    im.save(OUT / "img" / name, quality=85)
    return f"img/{name}"


def published(docx, stem):
    with zipfile.ZipFile(docx) as z:
        hits = [n for n in z.namelist() if n.startswith("word/media/" + stem + ".")]
        if not hits:
            return None
        p = OUT / "tmp" / Path(hits[0]).name
        p.write_bytes(z.read(hits[0]))
        return p


def main(pub_main, pub_supp, cur_main, cur_supp):
    import shutil

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "tmp").mkdir(exist_ok=True)
    # Map from the submitted supplement, so removed figures still show
    sub = OUT / "tmp" / "supp"
    with zipfile.ZipFile(pub_supp) as z:
        z.extractall(sub, [n for n in z.namelist() if n in ("word/document.xml", "word/_rels/document.xml.rels") or n.startswith("word/media/")])
    sm = supp_media(sub)
    sm.setdefault("20", "image38")  # its caption paragraph is separated from the image by a table
    rows = []
    for label, table, pub, cur, media in [("Fig.", MAIN, pub_main, cur_main, MAIN_MEDIA), ("Supplementary Fig.", SUPP, pub_supp, cur_supp, sm)]:
        for k, (files, status, how) in table.items():
            stem = media.get(k)
            src = published(pub, stem) if stem else None
            if src is None and stem:
                c = list(Path(cur, "word/media").glob(stem + ".*"))
                src = c[0] if c else None
            tag = f"{'main' if label == 'Fig.' else 'supp'}{k}"
            left = thumb(src, f"{tag}_published.jpg") if src else None
            right = []
            for i, f in enumerate(files):
                p = WORK / f[1:] if f.startswith("@") else FIG / f
                if p.exists():
                    right.append(thumb(p, f"{tag}_regen{i}.jpg"))
            rows.append((f"{label} {k}", status, how, left, right))
    css = "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}h2{margin:32px 0 4px}.s{font-weight:600}.g{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}img{width:100%;border:1px solid #ccc}.c{font-size:13px;color:#555;margin:4px 0 8px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:3px 10px;text-align:left;font-size:13px}"
    h = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>NESI figure review</title><style>{css}</style></head><body>",
         "<h1>NESI figure review</h1><p>Left: as submitted. Right: regenerated by the code in <code>reproduce/</code>. Rebuild this page with <code>reproduce/figure_gallery.py</code>.</p>",
         "<table><tr><th>Figure</th><th>Status</th></tr>"]
    h += [f"<tr><td><a href='#{html.escape(n)}'>{html.escape(n)}</a></td><td>{html.escape(s)}</td></tr>" for n, s, *_ in rows]
    h.append("</table>")
    for n, s, how, left, right in rows:
        h.append(f"<h2 id='{html.escape(n)}'>{html.escape(n)}</h2><div class='c'><span class='s'>{html.escape(s)}.</span> {html.escape(how)}</div><div class='g'>")
        h.append(f"<div><div class='c'>As submitted</div>{f'<img src={left!r}>' if left else '<p>(none)</p>'}</div>")
        h.append("<div><div class='c'>Regenerated</div>" + ("".join(f"<img src={r!r}>" for r in right) or "<p>(none)</p>") + "</div></div>")
    h.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(h))
    import shutil

    shutil.rmtree(OUT / "tmp")
    print(OUT / "index.html", len(rows), "figures")


if __name__ == "__main__":
    main(*sys.argv[1:5])
