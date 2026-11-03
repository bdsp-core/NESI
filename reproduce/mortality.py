#!/usr/bin/env python3
"""In-hospital mortality analyses.

    python reproduce/mortality.py features   # slow, once: NESI for every mortality segment (from S3)
    python reproduce/mortality.py analyze    # seconds to minutes, from saved intermediates

`features` follows Train_Death_Prediction_with_NESI_vsGCS.py to define the sequential
cohort, the hourly GCS (last observation carried forward) and the segment order, and
writes intermediates/mortality/mortality_segments.csv. `analyze` refits the sequential
logistic regressions, keeps per-patient out-of-fold predictions, and writes tables
and figures. The manuscript's SOFA and mechanical ventilation covariates are not in
any released file, so the clinical model here uses what is available (see COVARIATES).
"""
import argparse
import ast
import concurrent.futures as cf
import io
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ["NESI_DATA_ROOT"])
REPO = Path(__file__).resolve().parent.parent
BUILD = Path(os.environ.get("NESI_BUILD_ROOT", ROOT.parent / "build")) / "output"
TABLES, FIGS = BUILD / "tables", BUILD / "figures"
INTER = ROOT / "intermediates" / "mortality"
BUCKET, PREFIX = "bdsp-opendata-credentialed", "yama/mortality_analysis/MorgothActivations"
FOLDERS = ["SLEEP", "NM", "BS", "FOCGEN", "SLOWING", "IIIC"]  # column order used in training
HOURS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SEEDS = [10, 12, 42, 5, 30]
COVARIATES = ["age", "male", "vasopressor_6h", "sedative_6h"]
SETS = {
    "Clinical*": ("cov",),
    "GCS": ("GCS",),
    "NESI": ("NESI",),
    "Clinical* + GCS": ("cov", "GCS"),
    "Clinical* + NESI": ("cov", "NESI"),
    "Clinical* + GCS + NESI": ("cov", "GCS", "NESI"),
}


def clean_id(x):
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


def sequential_gcs(row):
    """Hourly grid from EEG start, last observation carried forward, stopping at the last GCS."""
    start = row.EEGBeginDTS
    pairs = sorted(zip(pd.to_datetime(row.times), row.vals), key=lambda p: p[0])
    last = pairs[-1][0]
    delta = (last - start).total_seconds()
    if delta <= 0:
        return []
    seq, cur, j = [], pairs[0][1], 0
    for h in range(1, int(np.ceil(delta / 3600)) + 1):
        t = start + pd.Timedelta(hours=h)
        while j < len(pairs) and pairs[j][0] <= t:
            cur = pairs[j][1]
            j += 1
        seq.append(cur)
        if t >= last:
            break
    return seq


def load_gcs():
    g = ROOT / "github_moved/cohort_models/GCS/Cohort"
    m = pd.concat([pd.read_csv(g / f"GCS_{s}_HarvardEEG_metadata.csv", low_memory=False) for s in ["BWH", "MGH"]])
    return m.dropna(subset=["R CPN GLASGOW COMA SCALE SCORE"]).rename(columns={"R CPN GLASGOW COMA SCALE SCORE": "GCS"})


def build_cohort():
    """One row per patient: session, outcome and hourly GCS."""
    single = pd.read_csv(ROOT / "github_moved/mortality_analysis/Cohort/YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv", low_memory=False)
    bidmc = pd.read_csv(ROOT / "s3/GCS_BIDMC_HarvardEEG_metadata.csv", low_memory=False)
    m = load_gcs()
    d = single[single.BDSPPatientID.isin(set(m.BDSPPatientID) | set(bidmc.BDSPPatientID))].copy()
    key = ["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS"]
    agg = m.merge(d[key], on=key).groupby(key).agg(times=("GCSRecordedDTS", list), vals=("GCS", list)).reset_index()
    d = d.merge(agg, on=key)
    d["EEGBeginDTS"] = pd.to_datetime(d.EEGBeginDTS)
    d["hours_eeg"] = np.floor(d.DurationInSeconds / 3600).astype(int)
    d["seq"] = [sequential_gcs(r) for r in d.itertuples()]
    d["seq"] = [s[1:] if h < len(s) else s for s, h in zip(d.seq, d.hours_eeg)]
    d["n_seq"] = d.seq.map(len)
    d["pid"] = d.BDSPPatientID.map(clean_id)
    return d


def wanted_files(d):
    names = [x.strip() for x in open(INTER / "bs_files.txt") if x.strip()]
    f = pd.DataFrame({"name": [n[:-4] for n in names if n.endswith(".csv")]})
    f["pid"] = f.name.str.split("_").str[0].str.split("-").str[1].str[5:]
    f["seg"] = f.name.str.extract(r"_seg(\d+)$").astype(int)
    f = f[f.pid.isin(set(d.pid))].sort_values(["pid", "seg"])
    n = d.set_index("pid").n_seq
    f["n_files"] = f.groupby("pid").name.transform("count")
    f["n_seq"] = f.pid.map(n)
    f["rank"] = f.groupby("pid").cumcount()
    return f[(f["rank"] < f.n_seq) & (f.n_files >= f.n_seq)]


def split_ids(d):
    """Patients the original split ran over, in its order (sorted string ID)."""
    f = wanted_files(d)
    return sorted(f.pid.unique())


def model_classes():
    import torch
    from torch import nn
    from torch.nn import functional as F

    src = (REPO / "NESI/DeathPrediction_NESIvsGCS/model_SeqLR/Train_Death_Prediction_with_NESI_vsGCS.py").read_text()
    ns = {"torch": torch, "nn": nn, "F": F, "np": np}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name in {"ResidualBlock1D", "MORGOTH_ResNet1D_onlyGAP", "EEGScoringModel"}:
            exec(compile(ast.Module([node], []), "death", "exec"), ns)
    return ns


def features(threads):
    import boto3
    import torch

    INTER.mkdir(parents=True, exist_ok=True)
    out = INTER / "mortality_segments.csv"
    d = build_cohort()
    f = wanted_files(d)
    print(f"cohort {d.pid.nunique()} patients, {f.pid.nunique()} with enough segments, {len(f)} segments", flush=True)
    f = f[f.n_seq < 24]  # models use patients under 24 hours only
    done = set(pd.read_csv(out, usecols=["name"]).name) if out.exists() else set()
    todo = f[~f.name.isin(done)]
    print(f"to do {len(todo)}", flush=True)
    ns = model_classes()
    ck = REPO / "NESI/model/ModelCheckpoints"
    enc = ns["MORGOTH_ResNet1D_onlyGAP"](num_features=17)
    enc.load_state_dict(torch.load(ck / "ResNetGAP_BestModel.pth", map_location="cpu", weights_only=True))
    head = ns["EEGScoringModel"]()
    head.load_state_dict(torch.load(ck / "NESI_best_model.pth", map_location="cpu", weights_only=True)["model_state_dict"])
    enc.eval(), head.eval()
    torch.set_num_threads(4)
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "opendata")).client("s3", config=boto3.session.Config(max_pool_connections=threads + 4))

    def one(name):
        parts = []
        for fd in FOLDERS:
            b = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{fd}/{name}.csv")["Body"].read()
            t = pd.read_csv(io.BytesIO(b))
            if "pred_class" in t.columns:
                t = t.drop(columns="pred_class")
            parts.append(t.values.astype("float32"))
        return name, np.concatenate(parts, axis=1)

    def flush(batch, first):
        by = {}
        for n, x in batch:
            by.setdefault(x.shape, []).append((n, x))
        rows = []
        for group in by.values():
            xb = torch.tensor(np.stack([x for _, x in group]))
            with torch.no_grad():
                s = head(enc(xb)).squeeze(-1).numpy()
            rows += [(n, float(v)) for (n, _), v in zip(group, s)]
        pd.DataFrame(rows, columns=["name", "NESI"]).to_csv(out, mode="a", header=first, index=False)

    first, batch, miss = not out.exists(), [], 0
    with cf.ThreadPoolExecutor(threads) as ex:
        names = iter(todo.name.tolist())
        pending = set()
        k = 0
        while True:
            while len(pending) < threads * 4:
                n = next(names, None)
                if n is None:
                    break
                pending.add(ex.submit(one, n))
            if not pending:
                break
            finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fu in finished:
                k += 1
                try:
                    batch.append(fu.result())
                except Exception as e:
                    miss += 1
                    if miss < 20:
                        print("miss", type(e).__name__, str(e)[:100], flush=True)
                if len(batch) >= 256:
                    flush(batch, first)
                    first, batch = False, []
                if k % 5000 == 0:
                    print(k, "miss", miss, flush=True)
    if batch:
        flush(batch, first)
    print("done, missed", miss, flush=True)


# ---------------------------------------------------------------- analyses


def covariates():
    """Age and sex from the cohort file; 6-hour infusion flags from the medication pipeline."""
    single = pd.read_csv(ROOT / "github_moved/mortality_analysis/Cohort/YAMA_FINAL_DEATH_RANDOM_SINGLE_SESSIONS_COHORT.csv", low_memory=False)
    c = pd.DataFrame({"pid": single.BDSPPatientID.map(clean_id), "age": single.AgeAtVisit, "male": (single.SexDSC == "Male").astype(float), "EEGBeginDTS": single.EEGBeginDTS, "EEGExamEndDTS": single.EEGExamEndDTS})
    src = next(d for d in [ROOT / "sedation_repo", ROOT.parent / "christine/sedation_repo"] if d.exists()) / "medication-exposure/data/i0001_per_eeg_master_exposure_windows_opiates.csv"
    use = ["BDSPPatientID", "EEGBeginDTS", "EEGExamEndDTS", "_cohort", "sedative_inf_6h", "any_vasopressor_inf_6h"]
    e = pd.read_csv(src, usecols=use, low_memory=False)
    e = e[e._cohort == "GCS"].assign(pid=lambda x: x.BDSPPatientID.map(clean_id))
    g = e.groupby(["pid", "EEGBeginDTS", "EEGExamEndDTS"]).agg(sedative_6h=("sedative_inf_6h", "max"), vasopressor_6h=("any_vasopressor_inf_6h", "max")).reset_index()
    c = c.merge(g, on=["pid", "EEGBeginDTS", "EEGExamEndDTS"], how="left")
    return c.drop(columns=["EEGBeginDTS", "EEGExamEndDTS"]).drop_duplicates("pid")


def delong(y, a, b):
    """Paired DeLong: difference in AUROC (a - b), its SE, and a two-sided p."""
    y = np.asarray(y).astype(bool)

    def comps(s):
        x, z = s[y], s[~y]
        v10 = np.array([((xi > z).sum() + 0.5 * (xi == z).sum()) / len(z) for xi in x])
        v01 = np.array([((x > zi).sum() + 0.5 * (x == zi).sum()) / len(x) for zi in z])
        return v10, v01, v10.mean()

    a10, a01, aa = comps(a)
    b10, b01, ab = comps(b)
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    var = s10[0, 0] / len(a10) + s01[0, 0] / len(a01) + s10[1, 1] / len(b10) + s01[1, 1] / len(b01) - 2 * (s10[0, 1] / len(b10) + s01[0, 1] / len(b01))
    d, se = aa - ab, np.sqrt(max(var, 1e-12))
    return d, se, 2 * stats.norm.sf(abs(d) / se)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    a = np.maximum.accumulate(np.minimum(1, p[o] * (len(p) - np.arange(len(p)))))
    r = np.empty(len(p))
    r[o] = a
    return r


def fit_predict(train_x, train_y, test_x, scale=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if scale:
        sc = StandardScaler().fit(train_x)
        train_x, test_x = sc.transform(train_x), sc.transform(test_x)
    m = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", C=0.5, l1_ratio=0.7, max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_x, train_y)
    return m.predict_proba(test_x)[:, 1]


def labels_table():
    t = pd.concat([pd.read_csv(INTER / f"Subject_inhospDied_label_batch{k}_MGB_BIDMC.csv") for k in (1, 2)])
    t["pid"] = t.BDSPPatientID.map(clean_id)
    multi = t.groupby("pid").SiteID.nunique().gt(1)
    t = t[~t.pid.isin(multi[multi].index) | (t.SiteID == "S0001")].drop_duplicates("pid")
    return t.set_index("pid").DiedInHospital.astype(str).str.lower().eq("yes").astype(int)


def assemble(order):
    """Per-patient sequences. 'numeric' is time order; 'lexical' repeats the released code,
    which sorts segments by filename so seg10 precedes seg2."""
    d = build_cohort()
    s = pd.read_csv(INTER / "mortality_segments.csv").drop_duplicates("name")
    s["pid"] = s.name.str.split("_").str[0].str.split("-").str[1].str[5:]
    s["seg"] = s.name.str.extract(r"_seg(\d+)$").astype(int)
    y = labels_table()
    seqs = d.set_index("pid").seq
    rows = []
    for pid, g in s.groupby("pid"):
        q = seqs.get(pid)
        if q is None or pid not in y.index or len(q) >= 24 or len(g) < len(q):
            continue
        g = g.sort_values("seg").iloc[: len(q)]
        gcs, nesi = np.array(q, float), g.NESI.values
        if order == "lexical":
            o = np.argsort(g.name.values)
            gcs, nesi = gcs[o], nesi[o]
        rows.append(dict(pid=pid, n=len(q), died=int(y[pid]), GCS=gcs, NESI=nesi))
    p = pd.DataFrame(rows).merge(covariates(), on="pid", how="left")
    ids = split_ids(d)
    ids = [i for i in ids if i in y.index]
    return p, ids, y.reindex(ids).values


def design(df, h, members):
    parts = []
    for mb in members:
        if mb == "cov":
            parts.append(df[COVARIATES].astype(float).values)
        else:
            parts.append(np.vstack([v[:h] for v in df[mb]]))
    return np.hstack(parts)


def run_models(p, ids, died, sets, scale):
    """The released code's repeated stratified 85/15 hold-out (5 seeds); test predictions kept."""
    from sklearn.model_selection import train_test_split

    rows = []
    for fold, seed in enumerate(SEEDS):
        tr_ids, te_ids = train_test_split(np.array(ids), test_size=0.15, random_state=seed, stratify=died)
        tr, te = p[p.pid.isin(set(tr_ids))], p[p.pid.isin(set(te_ids))]
        for h in HOURS:
            a, b = tr[tr.n >= h], te[te.n >= h]
            for name, members in sets.items():
                ok_tr, ok_te = a, b
                if "cov" in members:
                    ok_tr, ok_te = a.dropna(subset=COVARIATES), b.dropna(subset=COVARIATES)
                pr = fit_predict(design(ok_tr, h, members), ok_tr.died.values, design(ok_te, h, members), scale)
                rows += [dict(fold=fold, hour=h, model=name, pid=i, died=yy, prob=q) for i, yy, q in zip(ok_te.pid, ok_te.died, pr)]
        print("fold", fold, flush=True)
    return pd.DataFrame(rows)


def net_benefit(y, p, th):
    n = len(y)
    tp = ((p >= th) & (y == 1)).sum()
    fp = ((p >= th) & (y == 0)).sum()
    return tp / n - fp / n * th / (1 - th)


def recalibrate(y, p):
    """Maps class-weighted probabilities to observed risk with a logistic fit on the logit."""
    from sklearn.linear_model import LogisticRegression

    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).predict_proba(lp)[:, 1]


def analyze():
    from sklearn.metrics import roc_auc_score

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    # 1. Same order and fit as the released code, to compare with its result pickles
    cache = INTER / "mortality_predictions_released_order.csv"
    p, ids, died = assemble("lexical")
    print(f"patients {len(p)}, deaths {p.died.sum()}, split over {len(ids)}", flush=True)
    if cache.exists():
        rel = pd.read_csv(cache, dtype={"pid": str})
    else:
        rel = run_models(p, ids, died, {k: SETS[k] for k in ["GCS", "NESI"]}, scale=False)
        rel.to_csv(cache, index=False)
    r = rel.groupby(["fold", "hour", "model"]).apply(lambda g: roc_auc_score(g.died, g.prob), include_groups=False).rename("auroc").reset_index()
    rows = []
    for k in ["GCS", "NESI"]:
        pk = pickle.load(open(REPO / f"NESI/DeathPrediction_NESIvsGCS/Results/results_{k}_Death.pkl", "rb"))
        for fold, res in enumerate(pk):
            for h in HOURS:
                rows.append(dict(fold=fold, hour=h, model=k, released=res[h]["auroc"], released_n=res[h]["n_test"]))
    cmp_ = pd.DataFrame(rows).merge(r, on=["fold", "hour", "model"])
    cmp_.round(4).to_csv(TABLES / "mortality_vs_released.csv", index=False)
    print(cmp_.groupby(["hour", "model"])[["released", "auroc"]].mean().round(3).unstack().to_string())
    print("max abs difference", (cmp_.released - cmp_.auroc).abs().max().round(4), "correlation", cmp_[["released", "auroc"]].corr().iloc[0, 1].round(4), flush=True)

    # 2. Corrected time order, with the clinical variables we hold
    cache = INTER / "mortality_predictions.csv"
    p, ids, died = assemble("numeric")
    print(f"patients {len(p)}, deaths {p.died.sum()}; with covariates {p.dropna(subset=COVARIATES).shape[0]}", flush=True)
    p.drop(columns=["GCS", "NESI"]).to_csv(INTER / "mortality_patients.csv", index=False)
    if cache.exists():
        pr = pd.read_csv(cache, dtype={"pid": str})
    else:
        pr = run_models(p, ids, died, SETS, scale=True)
        pr.to_csv(cache, index=False)

    auc = pr.groupby(["fold", "hour", "model"]).apply(lambda g: pd.Series(dict(auroc=roc_auc_score(g.died, g.prob), n=len(g), deaths=g.died.sum())), include_groups=False).reset_index()
    auc.to_csv(TABLES / "mortality_auroc_by_fold.csv", index=False)
    tab = auc.groupby(["hour", "model"]).auroc.mean().unstack().round(3)
    tab.to_csv(TABLES / "mortality_auroc_mean.csv")
    print(tab.to_string())

    # Paired DeLong within each repeat, averaged across repeats, Holm across durations
    pairs = [("Clinical* + NESI", "Clinical*"), ("Clinical* + GCS", "Clinical*"), ("NESI", "GCS"), ("Clinical* + NESI", "Clinical* + GCS"), ("Clinical* + GCS + NESI", "Clinical* + GCS"), ("Clinical* + GCS + NESI", "Clinical* + NESI")]
    rows = []
    for a, b in pairs:
        for h in HOURS:
            ds, ses, ps = [], [], []
            for fold in range(len(SEEDS)):
                g = pr[(pr.fold == fold) & (pr.hour == h)]
                w = g[g.model == a].merge(g[g.model == b], on=["pid", "died"], suffixes=("_a", "_b"))
                d, se, q = delong(w.died.values, w.prob_a.values, w.prob_b.values)
                ds.append(d), ses.append(se), ps.append(q)
            d, se = np.mean(ds), np.mean(ses)
            rows.append(dict(a=a, b=b, hour=h, delta=d, lo=d - 1.96 * se, hi=d + 1.96 * se, p=2 * stats.norm.sf(abs(d) / se)))
    dl = pd.DataFrame(rows)
    dl["p_holm"] = dl.groupby(["a", "b"]).p.transform(holm)
    dl.round(5).to_csv(TABLES / "mortality_delong.csv", index=False)
    print(dl[dl.hour.isin([4, 20])].round(4).to_string(index=False))

    # Decision curves: pooled test predictions over repeats, recalibrated within repeat and duration
    ths = np.arange(0.05, 0.61, 0.01)
    rows = []
    for h in [8, 12, 16, 20]:
        for name in SETS:
            g = pr[(pr.hour == h) & (pr.model == name)]
            for fold, f in g.groupby("fold"):
                q = recalibrate(f.died.values, f.prob.values)
                for th in ths:
                    rows.append(dict(hour=h, model=name, fold=fold, th=th, nb=net_benefit(f.died.values, q, th)))
        g = pr[(pr.hour == h) & (pr.model == "NESI")]
        for fold, f in g.groupby("fold"):
            prev = f.died.mean()
            for th in ths:
                rows.append(dict(hour=h, model="Treat all", fold=fold, th=th, nb=prev - (1 - prev) * th / (1 - th)))
    dc = pd.DataFrame(rows).groupby(["hour", "model", "th"]).nb.agg(["mean", "min", "max"]).reset_index()
    dc.round(5).to_csv(TABLES / "mortality_decision_curve.csv", index=False)
    plot_dca(dc)
    plot_auroc(auc)
    summary(p, auc, dl, dc)


def plot_auroc(auc):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for name in SETS:
        g = auc[auc.model == name].groupby("hour").auroc.agg(["mean", "min", "max"])
        line, = ax.plot(g.index, g["mean"], marker="o", ms=3, lw=1.6, label=name)
        ax.fill_between(g.index, g["min"], g["max"], color=line.get_color(), alpha=0.06, lw=0)
    ax.set(xlabel="Monitoring duration (hours)", ylabel="AUROC", xticks=HOURS)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIGS / "mortality_auroc_available_covariates.png", dpi=300)
    plt.close(fig)


def plot_dca(dc):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharey=True, sharex=True)
    show = ["Clinical*", "GCS", "NESI", "Clinical* + GCS", "Clinical* + NESI", "Clinical* + GCS + NESI"]
    for ax, h, lab in zip(axes.flat, [8, 12, 16, 20], "ABCD"):
        d = dc[dc.hour == h]
        for name in show:
            g = d[d.model == name]
            ax.plot(g.th, g["mean"], label=name, lw=1.4)
        g = d[d.model == "Treat all"]
        ax.plot(g.th, g["mean"], color="0.5", ls="--", lw=1, label="Treat all")
        ax.axhline(0, color="k", lw=0.8, label="Treat none")
        ax.set_title(f"{lab}. {h} hours", loc="left", fontsize=10)
        ax.set_ylim(-0.03, 0.22)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Threshold probability of in-hospital death")
    for ax in axes[:, 0]:
        ax.set_ylabel("Net benefit")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "mortality_decision_curve.png", dpi=300)
    plt.close(fig)


def summary(p, auc, dl, dc):
    n = {"patients": len(p), "deaths": int(p.died.sum()), "with_covariates": int(p.dropna(subset=COVARIATES).shape[0])}
    m = auc.groupby(["hour", "model"]).auroc.mean().unstack()
    for h in [4, 20]:
        for k in m.columns:
            n[f"auroc_{h}h_{k}"] = round(float(m.loc[h, k]), 3)
    for th in [0.2, 0.3, 0.4]:
        for name in ["Clinical* + GCS", "Clinical* + NESI"]:
            v = dc[(dc.hour == 12) & (dc.model == name) & (np.isclose(dc.th, th))]["mean"]
            n[f"nb_12h_{th}_{name}"] = round(float(v.iloc[0]), 4)
    pd.Series(n).to_csv(TABLES / "mortality_summary.csv", header=False)
    print(pd.Series(n).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["features", "analyze"])
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    if a.step == "features":
        features(a.threads)
    else:
        analyze()


if __name__ == "__main__":
    sys.exit(main())
