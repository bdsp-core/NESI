#!/usr/bin/env python3
"""Supp Fig 33: NESI heads on mean/median MORGOTH features, compared with the full model.

The repo's ablation checkpoints were trained on their own split (Ablation_MeanMedian_
Features_NESIHead.py, train_test_split seed 42), not the split of the published NESI
model. So they cannot be compared with NESI on either test set. --retrain fits the
same heads on NESI's own training and validation sets; the figure uses those heads.

    python reproduce/ablation_figure.py --features            # about 1 min; needs train_f32.npy.mem
    python reproduce/ablation_figure.py --retrain Median Mean # about 15 min per head and seed
    python reproduce/ablation_figure.py                       # seconds; numbers and figure from intermediates
"""
import argparse
import ast
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
INTER = ROOT / "intermediates"
HEADS = INTER / "ablation_heads"
ABL = REPO / "NESI/model/AblationStudy"
SCORES = ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv"
SCHEMATIC = Path(os.environ.get("NESI_SFIG33_SOURCE", INTER / "supp_fig33_original.png"))
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
LABEL = {"RASS": "RASS", "GCS": "GCS", "CAMS": "CAM-S", "ICANS": "ICANS"}
SEEDS = [2026, 2027, 2028]


def make_features():
    INTER.mkdir(exist_ok=True)
    idx, mean, median = [], [], []
    for n in ["train", "val", "test"]:
        r = pd.read_csv(ROOT / "s3" / f"NESI_{n}_results.csv", low_memory=False)
        if n == "train":
            x = np.memmap(ROOT.parent / "train_f32.npy.mem", dtype="float32", mode="r", shape=(len(r), 591, 17))
        else:
            x = np.stack(pickle.load(open(ROOT / "s3" / f"NESITripletIP_Morgoth_{n}_data.pkl", "rb"))).astype("float32")
        me, md = np.zeros((len(r), 17), "float32"), np.zeros((len(r), 17), "float32")
        for i in range(0, len(r), 2000):
            b = np.array(x[i : i + 2000])
            me[i : i + 2000], md[i : i + 2000] = b.mean(1), np.median(b, 1)
        idx.append(r[["Dataset", "Filename", "WhichSet", "TrueScore", "PredictedNESI", "TransformedTrueScore"]])
        mean.append(me), median.append(md)
    pd.concat(idx, ignore_index=True).to_csv(INTER / "ablation_summary_index.csv", index=False)
    np.savez_compressed(INTER / "ablation_summary_features.npz", mean=np.concatenate(mean), median=np.concatenate(median))


def load_index():
    m = pd.read_csv(INTER / "ablation_summary_index.csv", low_memory=False)
    s = pd.read_csv(SCORES, low_memory=False)[["Dataset", "MorgothOutputFilename", "BDSPPatientID", "RawScore", "GroupedScore", "Split"]]
    return m.merge(s, left_on=["Dataset", "Filename"], right_on=["Dataset", "MorgothOutputFilename"], how="left", validate="one_to_one")


def definitions():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset
    from tqdm import tqdm

    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "Dataset": Dataset, "random": random, "tqdm": tqdm}
    src = (ABL / "Training/Ablation_MeanMedian_Features_NESIHead.py").read_text()
    names = {"create_pairs", "PairDataset", "EEGScoringModel", "pairwise_loss", "train_model"}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module([node], []), "ablation", "exec"), ns)
    return ns, torch


def retrain(heads, seeds):
    from torch.utils.data import DataLoader

    ns, torch = definitions()
    HEADS.mkdir(parents=True, exist_ok=True)
    m = load_index()
    z = np.load(INTER / "ablation_summary_features.npz")
    for head in heads:
        x = z[head.lower()]
        for seed in seeds:
            out = HEADS / f"{head}_seed{seed}"
            if (out.with_suffix(".csv")).exists():
                continue
            random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
            pairs = {}
            for k, w in [("train", "Train"), ("val", "Validation")]:
                i = np.where(m.WhichSet == w)[0]
                pairs[k] = ns["create_pairs"](embeddings=x[i], y_raw_transformed=m.TransformedTrueScore.values[i], y_grouped=m.GroupedScore.values[i], dataset_names=m.Dataset.values[i], mode="within_plus_global")
            model = ns["EEGScoringModel"]()
            ns["train_model"](
                model=model,
                train_loader=DataLoader(ns["PairDataset"](pairs["train"]), batch_size=128, shuffle=True, drop_last=True),
                val_loader=DataLoader(ns["PairDataset"](pairs["val"]), batch_size=128, shuffle=False),
                optimizer=torch.optim.Adam(model.parameters(), lr=0.005),
                device="cpu", epochs=100, save_path=str(out.with_suffix(".pth")), patience=20,
            )
            model.load_state_dict(torch.load(out.with_suffix(".pth"), map_location="cpu", weights_only=False)["model_state_dict"])
            model.eval()
            te = m[m.Split == "Test"].copy()
            with torch.no_grad():
                te["HeadScore"] = model(torch.tensor(x[te.index.values])).numpy().ravel()
            te[["Dataset", "MorgothOutputFilename", "BDSPPatientID", "RawScore", "TrueScore", "PredictedNESI", "HeadScore"]].to_csv(out.with_suffix(".csv"), index=False)
            print(head, seed, "done", flush=True)


def released_checkpoints():
    """Where the published panel values came from: released checkpoints on the script's own split."""
    from sklearn.model_selection import train_test_split

    ns, torch = definitions()
    t = ROOT / "github_moved/NESI/model/Training"
    pid = lambda f: f.split("_")[0].split("-")[1][5:]
    c = pd.read_csv(t / "CAMSTraining_Final_Metadata.csv").assign(Dataset="CAMS")
    i = pd.read_csv(t / "ICANSTraining_Final_Metadata.csv").rename(columns={"StudyID": "BDSPPatientID"}).assign(Dataset="ICANS")
    g = pd.read_csv(t / "GCSTraining_Final_Metadata.csv").assign(Dataset="GCS")
    g["MorgothOutputFilename"] = g.Filename.astype(str) + ".csv"
    g["BDSPPatientID"] = g.MorgothOutputFilename.apply(pid)
    r = pd.read_csv(t / "RASSTraining_Final_Metadata.csv").assign(Dataset="RASS")
    r["MorgothOutputFilename"] = r.Filename.str.rstrip("'") + ".csv"
    r = r[~r.RASS_value.isin([1, 2, 3, 4])]
    r["BDSPPatientID"] = r.MorgothOutputFilename.apply(pid)
    a = pd.concat([d[["BDSPPatientID", "Dataset", "MorgothOutputFilename"]] for d in [r, g, c, i]], ignore_index=True)
    s = a.drop_duplicates()[["BDSPPatientID", "Dataset"]].drop_duplicates()
    tv, te = train_test_split(s, test_size=0.2, stratify=s.Dataset, random_state=42)
    tr, va = train_test_split(tv, test_size=0.125, stratify=tv.Dataset, random_state=42)
    sets = [("Train", set(tr.BDSPPatientID)), ("Val", set(va.BDSPPatientID)), ("Test", set(te.BDSPPatientID))]
    a["AblationSplit"] = a.BDSPPatientID.map(lambda p: next(n for n, v in sets if p in v))
    m = load_index().merge(a[["Dataset", "MorgothOutputFilename", "AblationSplit"]], on=["Dataset", "MorgothOutputFilename"])
    z = np.load(INTER / "ablation_summary_features.npz")
    rows = []
    for head in ["Median", "Mean"]:
        ck = torch.load(ABL / f"ModelCheckpoints/{head}_best_model.pth", map_location="cpu", weights_only=False)
        model = ns["EEGScoringModel"]()
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        with torch.no_grad():
            m["h"] = model(torch.tensor(z[head.lower()])).numpy().ravel()
        for lab, sel in [("script's own test split", m.AblationSplit == "Test"), ("manuscript test set", m.Split == "Test")]:
            for ds, d in m[sel].groupby("Dataset"):
                trained = (d.AblationSplit == "Train").mean()
                rows.append(dict(Head=head, TestSet=lab, Scale=ds, Segments=len(d), rho=abs(stats.spearmanr(d.TrueScore, d.h)[0]), share_in_head_training=trained))
    pd.DataFrame(rows).round(3).to_csv(BUILD / "tables/ablation_released_checkpoints.csv", index=False)


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


def summarize(boot=1000):
    rows, frames = [], {}
    for f in sorted(HEADS.glob("*_seed*.csv")):
        head, seed = f.stem.split("_seed")
        d = pd.read_csv(f, low_memory=False)
        frames[(head, int(seed))] = d
        for ds in SCALES:
            g = d[d.Dataset == ds]
            ids = {k: np.asarray(v) for k, v in g.groupby("BDSPPatientID").indices.items()}
            keys = list(ids)
            t, h, n = g.TrueScore.values, g.HeadScore.values, g.PredictedNESI.values
            rng = np.random.default_rng(2026)
            b, dl = [], []
            for _ in range(boot):
                i = np.concatenate([ids[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
                if len(np.unique(t[i])) > 1:
                    a = rho(t[i], h[i])
                    b.append(a), dl.append(a - rho(t[i], n[i]))
            rows.append(dict(Head=head, Seed=int(seed), Scale=ds, Segments=len(g), Patients=len(keys), rho=rho(t, h), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), rho_NESI=rho(t, n), delta=rho(t, h) - rho(t, n), delta_lo=np.percentile(dl, 2.5), delta_hi=np.percentile(dl, 97.5)))
    r = pd.DataFrame(rows).round(3)
    r.to_csv(BUILD / "tables/ablation_summary.csv", index=False)
    return r, frames


def figure(r, frames, seed=SEEDS[0]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image, ImageDraw, ImageFont

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.weight": "bold"})
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2), dpi=200)
    for row, (head, title) in enumerate([("Median", "median"), ("Mean", "mean")]):
        d = frames[(head, seed)]
        for ax, ds in zip(axes[row], SCALES):
            s = d[d.Dataset == ds].sort_values("RawScore")
            sns.boxplot(data=s, x="RawScore", y="HeadScore", hue="RawScore", legend=False, ax=ax, palette="pastel", showfliers=True, boxprops={"edgecolor": "black", "linewidth": 1.1}, whiskerprops={"color": "black", "linewidth": 1.1}, capprops={"color": "black", "linewidth": 1.1}, medianprops={"color": "black", "linewidth": 1.2}, flierprops={"markerfacecolor": "grey", "markeredgecolor": "black", "markersize": 4})
            v = r[(r.Head == head) & (r.Seed == seed) & (r.Scale == ds)].iloc[0]
            ax.set_title(f"{LABEL[ds]} Dataset\nSpearman ρ = {v.rho:.3f} [{v.lo:.3f}, {v.hi:.3f}]", fontweight="bold", fontsize=9, pad=8)
            ax.set_xlabel("Raw Score", fontweight="bold")
            ax.set_ylabel("Estimated NESI" if ds == "RASS" else "", fontweight="bold")
            for sp in ax.spines.values():
                sp.set_edgecolor("black"), sp.set_visible(True), sp.set_linewidth(1.2)
        fig.text(0.5, 0.985 if row == 0 else 0.495, f"NESI Distribution using {title}-feature-based NESI head", ha="center", va="top", fontweight="bold", fontsize=14)
        fig.text(0.992, 0.75 if row == 0 else 0.26, "(b)" if row == 0 else "(c)", ha="right", fontsize=13, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 0.975, 0.965), h_pad=4)
    panels = BUILD / "figures/supp_fig33_panels_bc.png"
    fig.savefig(panels, bbox_inches="tight")
    plt.close(fig)
    if not SCHEMATIC.exists():
        print("Schematic not found, wrote panels b and c only:", SCHEMATIC)
        return
    old = Image.open(SCHEMATIC).convert("RGB")
    a = old.crop((0, 0, old.width, 335))
    # The manuscript now calls this the estimation head
    dr = ImageDraw.Draw(a)
    dr.rectangle((1153, 78, 1327, 201), fill="white")
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 29)
    for k, line in enumerate(["NESI", "Estimation", "head"]):
        dr.text((1240, 106 + 32 * k), line, fill="black", font=font, anchor="mm")
    p = Image.open(panels).convert("RGB")
    w = max(p.width, 2400)
    a = a.resize((w, round(a.height * w / a.width)), Image.LANCZOS)
    p = p.resize((w, round(p.height * w / p.width)), Image.LANCZOS)
    c = Image.new("RGB", (w, a.height + p.height + 20), "white")
    c.paste(a, (0, 0)), c.paste(p, (0, a.height + 20))
    # Panel a printed cohort sizes that match no final cohort; show the analyzed patients
    d = ImageDraw.Draw(c)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 44)
    for (x0, y0, x1, y1), n in [((222, 100, 448, 148), 6191), ((488, 88, 712, 136), 7112), ((226, 420, 428, 470), 523), ((508, 420, 710, 470), 237)]:
        d.rectangle((x0, y0, x1, y1), fill="white")
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), f"#Pts: {n}", fill="#222222", font=font, anchor="mm")
    c.save(BUILD / "figures/supp_fig33.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--retrain", nargs="+", choices=["Median", "Mean"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--released", action="store_true", help="Also score the released checkpoints")
    args = ap.parse_args()
    (BUILD / "tables").mkdir(parents=True, exist_ok=True), (BUILD / "figures").mkdir(parents=True, exist_ok=True)
    if args.features:
        make_features()
    elif args.retrain:
        retrain(args.retrain, args.seeds)
    else:
        if args.released:
            released_checkpoints()
        table, fr = summarize()
        print(table.to_string(index=False))
        figure(table, fr)
