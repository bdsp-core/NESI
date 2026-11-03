#!/usr/bin/env python3
"""Sensitivity analysis: retrain NESI without test-set patients in the training data.

Keeps the model's original train/validation/test assignment and the full test
set, and withholds from training and validation every segment belonging to a
patient who is in the test set. Patients are matched across cohorts using one
ID, which the original split did not do. Procedure and hyperparameters come
from Train_TripletEncoder.py and Train_NESI.py, except the learning rate: every
retrain reported in the revision used 0.0005, the value those scripts held at the
time, while the published model was trained with 0.005. Use --lr 0.005 to match it.

    python reproduce/retrain_split_fix.py                 # withhold overlapping patients
    python reproduce/retrain_split_fix.py --keep-overlap  # same data as the original, new seed
    python reproduce/retrain_split_fix.py --mode downsample  # R3.6: RASS and GCS training patients reduced to 10%
    python reproduce/retrain_split_fix.py --mode upweight    # R3.6: anchors weighted toward CAM-S and ICANS
    python reproduce/retrain_split_fix.py --leave-out GCS   # leave-one-cohort-out (Supplementary Fig. 35)
    python reproduce/retrain_split_fix.py --mapping 7        # SPECTRA mapping M7 of Supplementary Table 5
    python reproduce/retrain_split_fix.py --smoke         # quick end-to-end test
"""
import argparse
import ast
import copy
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
if "NESI_DATA_ROOT" not in os.environ:
    sys.exit("Set NESI_DATA_ROOT (see reproduce/README.md)")
ROOT = Path(os.environ["NESI_DATA_ROOT"])
TRAIN_DIR = REPO / "NESI/model/Training"
CACHE = ROOT.parent / "train_f32.npy.mem"


def load_defs(path, names, ns, patches=()):
    src = Path(path).read_text()
    for old, new in patches:
        if old not in src:
            sys.exit(f"Expected text not found in {path}: {old}")
        src = src.replace(old, new)
    for node in ast.parse(src).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns


class Features:
    """Array-backed features that accept the tensor indices TripletDataset passes."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.arr[int(i)], dtype=np.float32))


def load_sets():
    """Return features and labels for each set, in the order the result files use."""
    s3 = ROOT / "s3"
    scores = pd.read_csv(ROOT / "github_moved/NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv", low_memory=False)
    out = {}
    for name in ["train", "val", "test"]:
        r = pd.read_csv(s3 / f"NESI_{name}_results.csv", low_memory=False).merge(
            scores[["Dataset", "MorgothOutputFilename", "BDSPPatientID", "RawScore", "GroupedScore", "Split"]],
            left_on=["Dataset", "Filename"],
            right_on=["Dataset", "MorgothOutputFilename"],
            how="left",
            validate="one_to_one",
        )
        if name == "train":
            x = np.memmap(CACHE, dtype="float32", mode="r", shape=(len(r), 591, 17))
        else:
            x = np.stack(pickle.load(open(s3 / f"NESITripletIP_Morgoth_{name}_data.pkl", "rb"))).astype("float32")
        out[name] = (x, r)
    return out


def withheld_mask(r, withhold):
    keys = pd.MultiIndex.from_frame(withhold[["Dataset", "MorgothOutputFilename"]])
    return pd.MultiIndex.from_frame(r[["Dataset", "MorgothOutputFilename"]]).isin(keys)


def embed(model, x, idx, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(model(torch.from_numpy(np.array(x[idx[i : i + bs]], dtype=np.float32)).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-overlap", action="store_true")
    ap.add_argument("--mode", choices=["standard", "downsample", "upweight"], default="standard")
    ap.add_argument("--leave-out", choices=["RASS", "GCS", "CAMS", "ICANS"])
    ap.add_argument("--mapping", type=int, choices=range(1, 25), metavar="1-24")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lr", type=float, default=0.0005, help="0.0005 reproduces the reported retrains; the published model used 0.005")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    name = ("smoke_" if a.smoke else "") + ("keep_overlap" if a.keep_overlap else "withhold_overlap") + f"_seed{a.seed}"
    if a.mode != "standard":
        name = f"{a.mode}_{name}"
    if a.leave_out:
        name = f"loco_{a.leave_out}_{name}"
    if a.mapping:
        name = f"mapping_M{a.mapping}_{name}"
    out = ROOT.parent / "retrain" / name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "Dataset": Dataset, "os": os, "copy": copy, "random": random, "tqdm": tqdm}
    enc_patches, head_patches = (), ()
    if a.mode == "upweight":
        # Anchors favor the small cohorts: 40% each for CAM-S and ICANS, 10% each for RASS and GCS
        ns["ANCHOR_P"] = {"CAMS": 0.4, "GCS": 0.1, "ICANS": 0.4, "RASS": 0.1}
        enc_patches = [("np.random.choice(self.unique_datasets)", "np.random.choice(self.unique_datasets, p=[ANCHOR_P[str(d)] for d in self.unique_datasets])")]
        head_patches = [
            ('"RASS": 0.2', '"RASS": 0.1'), ('"GCS": 0.2', '"GCS": 0.1'), ('"CAMS": 0.3', '"CAMS": 0.4'), ('"ICANS": 0.3', '"ICANS": 0.4'),
            ("random.sample(indices, min(anchor_count, len(indices)))", "random.choices(indices, k=anchor_count)"),
        ]
    load_defs(TRAIN_DIR / "Train_TripletEncoder.py", {"TripletDataset", "ResidualBlock1D", "MORGOTH_ResNet1D_onlyGAP_CORAL", "Training_triplet_function"}, ns, enc_patches)
    load_defs(TRAIN_DIR / "Train_NESI.py", {"create_pairs", "PairDataset", "EEGScoringModel", "pairwise_loss", "train_model"}, ns, head_patches)

    sets = load_sets()
    if a.mapping:
        sys.path.insert(0, str(REPO / "reproduce"))
        from spectra_mappings import regroup

        for k in sets:
            sets[k][1]["GroupedScore"] = regroup(sets[k][1], a.mapping)
    withhold = pd.read_csv(ROOT.parent / "overlap_rows_to_withhold.csv")
    keep = {}
    for k in ["train", "val"]:
        r = sets[k][1]
        mask = np.zeros(len(r), bool) if a.keep_overlap else withheld_mask(r, withhold)
        if a.mode == "downsample":
            # Keep a fixed 10% of RASS and GCS patients in each set; CAM-S and ICANS stay whole
            ids = r[r.Dataset.isin(["RASS", "GCS"])].BDSPPatientID.astype(str).drop_duplicates().sort_values().values
            kept = set(np.random.default_rng(a.seed).choice(ids, max(1, len(ids) // 10), replace=False))
            mask = mask | (r.Dataset.isin(["RASS", "GCS"]).values & ~r.BDSPPatientID.astype(str).isin(kept).values)
        if a.leave_out:
            mask = mask | (r.Dataset.values == a.leave_out)
        keep[k] = np.where(~mask)[0]
        print(f"{k}: {len(r)} rows, withheld {mask.sum()}, using {len(keep[k])}", flush=True)
    if a.smoke:
        g = np.random.default_rng(0)
        keep = {k: np.sort(g.choice(v, min(3000, len(v)), replace=False)) for k, v in keep.items()}

    def triplets(k):
        x, r = sets[k]
        rr = r.iloc[keep[k]]
        arr = x[keep[k]] if k != "train" else np.array(x[keep[k]], dtype=np.float32)
        return ns["TripletDataset"](Features(arr), torch.tensor(rr.GroupedScore.values, dtype=torch.long), torch.tensor(rr.TrueScore.values, dtype=torch.float32), rr.Dataset.values)

    # Stage 1: contrastive encoder
    t = time.time()
    enc = ns["MORGOTH_ResNet1D_onlyGAP_CORAL"](num_features=17)
    tr_dl = DataLoader(triplets("train"), batch_size=128, shuffle=True)
    va_dl = DataLoader(triplets("val"), batch_size=128, shuffle=False)
    ns["Training_triplet_function"](
        model=enc, lr=a.lr, n_epochs=1 if a.smoke else 200, train_dl=tr_dl, valid_dl=va_dl,
        save_loc=str(out), model_prefix="ResNetGAP", margin=0.2, device=device,
    )
    print(f"encoder done in {(time.time() - t) / 3600:.2f} h", flush=True)
    enc.load_state_dict(torch.load(out / "ResNetGAP_BestModel.pth", map_location=device))
    enc.to(device).eval()

    # Stage 2: ranking head on frozen embeddings
    t = time.time()
    emb = {k: embed(enc, sets[k][0], keep[k], device) for k in ["train", "val"]}
    pairs = {}
    for k in ["train", "val"]:
        rr = sets[k][1].iloc[keep[k]]
        pairs[k] = ns["create_pairs"](
            embeddings=emb[k], y_raw_transformed=rr.TransformedTrueScore.values, y_grouped=rr.GroupedScore.values,
            dataset_names=rr.Dataset.values, mode="within_plus_global",
        )
    head = ns["EEGScoringModel"]()
    ns["train_model"](
        model=head,
        train_loader=DataLoader(ns["PairDataset"](pairs["train"]), batch_size=128, shuffle=True, drop_last=True),
        val_loader=DataLoader(ns["PairDataset"](pairs["val"]), batch_size=128, shuffle=False),
        optimizer=torch.optim.Adam(head.parameters(), lr=a.lr),
        device=device, epochs=2 if a.smoke else 100, save_path=str(out / "NESI_best_model.pth"), patience=20,
    )
    print(f"head done in {(time.time() - t) / 60:.1f} min", flush=True)
    head.load_state_dict(torch.load(out / "NESI_best_model.pth", map_location=device, weights_only=False)["model_state_dict"])
    head.to(device).eval()

    # Evaluate on the manuscript's test set
    x, r = sets["test"]
    test_idx = np.where(r.Split.values == "Test")[0]
    with torch.no_grad():
        scores = head(torch.from_numpy(embed(enc, x, test_idx, device)).to(device)).cpu().numpy()
    t_df = r.iloc[test_idx].copy()
    t_df["NewNESI"] = scores
    t_df[["Dataset", "MorgothOutputFilename", "BDSPPatientID", "TrueScore", "PredictedNESI", "NewNESI"]].to_csv(out / "test_scores.csv", index=False)
    res = {"seed": a.seed, "withheld": not a.keep_overlap}
    for ds, g in t_df.groupby("Dataset"):
        res[ds] = {
            "n": len(g),
            "rho_new": abs(spearmanr(g.TrueScore, g.NewNESI)[0]),
            "rho_published": abs(spearmanr(g.TrueScore, g.PredictedNESI)[0]),
            "corr_new_vs_published": float(np.corrcoef(g.NewNESI, g.PredictedNESI)[0, 1]),
        }
    if a.leave_out:
        # The excluded scale's whole cohort, across all three sets (Supplementary Fig. 35 panels A-D)
        parts = []
        for k in ["train", "val", "test"]:
            xk, rk = sets[k]
            i = np.where(rk.Dataset.values == a.leave_out)[0]
            if a.smoke:
                i = i[:50]
            with torch.no_grad():
                sc = head(torch.from_numpy(embed(enc, xk, i, device)).to(device)).cpu().numpy()
            parts.append(rk.iloc[i][["Dataset", "MorgothOutputFilename", "BDSPPatientID", "TrueScore", "PredictedNESI"]].assign(NewNESI=sc, Set=k))
        pd.concat(parts).to_csv(out / "heldout_scores.csv", index=False)
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
