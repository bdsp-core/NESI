"""Run the published weights on the published features. Needs torch; run by run_all.py."""
import ast
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def classes(path, names):
    src = Path(path).read_text()
    ns = {"torch": torch, "nn": nn, "F": F, "np": np, "Dataset": Dataset}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns


def predict(model, x):
    with torch.no_grad():
        return np.concatenate([np.ravel(model(x[i : i + 256]).numpy()) for i in range(0, len(x), 256)])


def main(repo, s3, out):
    repo, s3, num = Path(repo), Path(s3), {}
    ns = classes(repo / "NESI/model/Testing/NESI_Test_Universal_Fulldataset.py", {"ResidualBlock1D", "MORGOTH_ResNet1D_onlyGAP", "EEGScoringModel"})
    ckpt = repo / "NESI/model/ModelCheckpoints"
    enc = ns["MORGOTH_ResNet1D_onlyGAP"](num_features=17)
    enc.load_state_dict(torch.load(ckpt / "ResNetGAP_BestModel.pth", map_location="cpu", weights_only=False))
    head = ns["EEGScoringModel"]()
    head.load_state_dict(torch.load(ckpt / "NESI_best_model.pth", map_location="cpu", weights_only=False)["model_state_dict"])
    enc.eval(), head.eval()

    full = nn.Sequential(enc, head)
    test_x = None
    for name in ["test", "val"]:
        x = torch.tensor(np.stack(pickle.load(open(s3 / f"NESITripletIP_Morgoth_{name}_data.pkl", "rb"))), dtype=torch.float32)
        r = pd.read_csv(s3 / f"NESI_{name}_results.csv")
        p = predict(full, x)
        num[f"INFER.{name}.r"] = float(np.corrcoef(p, r.PredictedNESI)[0, 1])
        num[f"INFER.{name}.max_abs_diff"] = float(np.abs(p - r.PredictedNESI).max())
        num[f"INFER.{name}.n"] = len(p)
        if name == "test":
            test_x, test_r = x.numpy(), r

    # Ablation heads: mean/median of the MORGOTH matrix instead of the contrastive encoder
    a = repo / "NESI/model/AblationStudy"
    ns2 = classes(a / "Testing/Testing_Ablation_MeanMedian_NESI.py", {"EEGScoringModel"})
    for nm, fx in [("Mean", lambda v: v.mean(1)), ("Median", lambda v: np.median(v, 1))]:
        ck = torch.load(a / f"ModelCheckpoints/{nm}_best_model.pth", map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        first = sd["net.0.weight"].shape
        m = ns2["EEGScoringModel"](input_dim=first[1], hidden_dim=first[0])
        m.load_state_dict(sd)
        m.eval()
        p = predict(m, torch.tensor(fx(test_x), dtype=torch.float32))
        for ds, idx in test_r.groupby("Dataset").groups.items():
            num[f"ABL.{nm}.{ds}"] = float(abs(stats.spearmanr(test_r.TrueScore.values[idx], p[idx])[0]))
    json.dump(num, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
