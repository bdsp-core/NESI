"""Recompute manuscript numbers from the released result files.

Each public function fills ``ctx.num`` with key -> value pairs and may write a
table to ``ctx.out``. ``run_all.py`` then compares those values with
``claims.csv``. Nothing here prints or writes patient-level rows.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, roc_auc_score, roc_curve

SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
SEVERE = {
    "RASS": lambda s: s <= -4,
    "GCS": lambda s: s <= 8,
    "CAMS": lambda s: s >= 6,
    "ICANS": lambda s: s >= 3,
}


class Context:
    def __init__(self, repo, root, out, boot, seed=0):
        self.repo, self.root, self.out = Path(repo), Path(root), Path(out)
        self.gh, self.s3, self.drive = self.root / "github_moved", self.root / "s3", self.root / "drive"
        self.boot, self.num = boot, {}
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def table(self, df, name):
        path = self.out / "tables" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def load(self, key, path, **kw):
        if key not in self._cache:
            self._cache[key] = pd.read_csv(path, low_memory=False, **kw)
        return self._cache[key]

    def scores(self):
        return self.load("scores", self.gh / "NESI/Bespoke_models/Results/Final_NESIscores_full_dataset_result.csv")

    def model_sets(self):
        if "sets" not in self._cache:
            r = pd.concat(
                pd.read_csv(self.s3 / f"NESI_{n}_results.csv", low_memory=False) for n in ["train", "val", "test"]
            ).rename(columns={"Filename": "MorgothOutputFilename"})
            self._cache["sets"] = self.scores().merge(
                r[["Dataset", "MorgothOutputFilename", "WhichSet", "PredictedNESI"]],
                on=["Dataset", "MorgothOutputFilename"],
            )
        return self._cache["sets"]

    def meds(self):
        if "meds" not in self._cache:
            m = pd.read_csv(self.drive / "i0001_RASS_Meds_NESI.csv", low_memory=False)
            m["n"] = pd.to_numeric(m.NESI, errors="coerce")
            m["r"] = pd.to_numeric(m["R PHS IP RASS"], errors="coerce")
            self._cache["meds"] = m[m.n.notna() & m.r.between(-5, 0)].reset_index(drop=True)
        return self._cache["meds"]

    def meta(self, cohort):
        files = {
            "RASS": ["RASS_BWH", "RASS_MGH"],
            "GCS": ["GCS_BWH", "GCS_MGH"],
        }[cohort]
        out = []
        for f in files:
            d = pd.read_csv(self.gh / f"cohort_models/{cohort}/Cohort/{f}_HarvardEEG_metadata.csv", low_memory=False)
            d["Site"] = f.split("_")[1]
            out.append(d)
        return pd.concat(out)


def rho(a, b):
    return abs(stats.spearmanr(a, b)[0])


def _patient_boot(ctx, t, fn, n=None):
    groups = {p: g.index.values for p, g in t.groupby("BDSPPatientID")}
    ids = list(groups)
    vals = []
    for _ in range(n or ctx.boot):
        ix = np.concatenate([groups[ids[j]] for j in ctx.rng.integers(0, len(ids), len(ids))])
        vals.append(fn(t.loc[ix]))
    return np.array(vals)


def _segment_boot(ctx, a, b):
    v = [rho(a[i], b[i]) for i in (ctx.rng.integers(0, len(a), len(a)) for _ in range(ctx.boot))]
    return np.mean(v), *np.percentile(v, [2.5, 97.5])


def cohorts_and_splits(ctx):
    f, n = ctx.scores(), ctx.num
    for ds in SCALES:
        t = f[f.Dataset == ds]
        n[f"{ds}.patients"] = t.BDSPPatientID.nunique()
        n[f"{ds}.obs"] = len(t)
        n[f"{ds}.test_obs"] = (t.Split == "Test").sum()
        n[f"{ds}.test_patients"] = t[t.Split == "Test"].BDSPPatientID.nunique()
    # ICANS study IDs mapped to BDSP IDs where known, so patients in several cohorts count once
    idx = pd.read_csv(ctx.s3 / "segment_index.csv", low_memory=False, dtype={"subject_id": str, "resolved_BDSPPatientID": str})
    ic = idx[idx.cohort == "ICANS"].dropna(subset=["resolved_BDSPPatientID"]).drop_duplicates("subject_id")
    ic_map = dict(zip(ic.subject_id, ic.resolved_BDSPPatientID.str.replace(r"\.0$", "", regex=True)))
    pid = f.BDSPPatientID.astype(str).where(f.Dataset != "ICANS", f.BDSPPatientID.astype(str).map(lambda x: ic_map.get(x, "ICANS_" + x)))
    n["all.unique_patients"] = pid.nunique()
    n["all.test_patients"] = f[f.Split == "Test"].BDSPPatientID.nunique()
    n["all.unique_segments"] = f.MorgothOutputFilename.nunique()
    n["all.segments_multi_cohort"] = (f.groupby("MorgothOutputFilename").Dataset.nunique() > 1).sum()
    pts = {k: set(v.BDSPPatientID) for k, v in f.groupby("Dataset")}
    n["overlap.RASS_GCS"] = len(pts["RASS"] & pts["GCS"])
    n["split.patients_multi_repo_labels"] = (f.groupby("BDSPPatientID").Split.nunique() > 1).sum()
    m = ctx.model_sets()
    # IDs harmonized as above. Within a cohort no patient crosses sets; across cohorts they do.
    m_pid = m.merge(pd.DataFrame({"Dataset": f.Dataset, "MorgothOutputFilename": f.MorgothOutputFilename, "pid": pid}), on=["Dataset", "MorgothOutputFilename"], how="left")
    n["split.patients_multi_model_sets"] = (m_pid.groupby("pid").WhichSet.nunique() > 1).sum()
    n["split.patients_multi_within_cohort"] = (m_pid.groupby(["Dataset", "pid"]).WhichSet.nunique() > 1).sum()
    te = m_pid[m_pid.MorgothOutputFilename.isin(f[f.Split == "Test"].MorgothOutputFilename) & (m_pid.WhichSet == "Test")]
    tv = m_pid[m_pid.WhichSet != "Test"]
    n["split.test_patients"] = te.pid.nunique()
    n["split.test_patients_with_train_val"] = len(set(te.pid) & set(tv.pid))
    n["split.train_val_rows_of_test_patients"] = tv.pid.isin(set(te.pid)).sum()
    for ds in ["CAMS", "ICANS"]:
        n[f"split.{ds}.test_patients_with_train_val"] = len(set(te[te.Dataset == ds].pid) & set(tv.pid))
    n["split.segments_multi_model_sets"] = (m.groupby("MorgothOutputFilename").WhichSet.nunique() > 1).sum()
    n["split.model_test_rows"] = (m.WhichSet == "Test").sum()


def correlations(ctx):
    f, n, rows = ctx.scores(), ctx.num, []
    for ds in SCALES:
        b = pd.read_csv(ctx.gh / f"NESI/Bespoke_models/Results/{ds}Bespokescore_results.csv")
        t = f[(f.Dataset == ds) & (f.Split == "Test")].merge(
            b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename"
        ).reset_index(drop=True)
        a, u, c = t.RawScore.values, t.NESI.values, t.BespokeBadnessScore.values
        su, sc = _segment_boot(ctx, a, u), _segment_boot(ctx, a, c)
        B = _patient_boot(ctx, t, lambda x: (rho(x.RawScore, x.NESI), rho(x.RawScore, x.BespokeBadnessScore)))
        d = B[:, 0] - B[:, 1]
        n[f"{ds}.rho"], n[f"{ds}.rho_mean_boot"] = rho(a, u), su[0]
        n[f"{ds}.rho_seg_lo"], n[f"{ds}.rho_seg_hi"] = su[1], su[2]
        n[f"{ds}.rho_pt_lo"], n[f"{ds}.rho_pt_hi"] = np.percentile(B[:, 0], [2.5, 97.5])
        n[f"{ds}.bespoke"] = rho(a, c)
        n[f"{ds}.bespoke_seg_lo"], n[f"{ds}.bespoke_seg_hi"] = sc[1], sc[2]
        n[f"{ds}.bespoke_mean_boot"] = sc[0]
        n[f"{ds}.bespoke_pt_lo"], n[f"{ds}.bespoke_pt_hi"] = np.percentile(B[:, 1], [2.5, 97.5])
        n[f"{ds}.delta"] = rho(a, u) - rho(a, c)
        n[f"{ds}.delta_lo"], n[f"{ds}.delta_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append({k.split(".", 1)[1]: round(v, 3) for k, v in n.items() if k.startswith(ds + ".")} | {"Scale": ds})
    ctx.table(pd.DataFrame(rows), "fig2_correlations")


def cams_sensitivity(ctx):
    """CAM-S test patients who also have GCS/RASS data in training or validation."""
    m, n = ctx.model_sets(), ctx.num
    b = pd.read_csv(ctx.gh / "NESI/Bespoke_models/Results/CAMSBespokescore_results.csv")
    c = m[(m.Dataset == "CAMS") & (m.Split == "Test")].merge(
        b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename"
    ).reset_index(drop=True)
    seen = set(m[m.WhichSet.isin(["Train", "Validation"]) & (m.Dataset != "CAMS")].BDSPPatientID)
    clean = c[~c.BDSPPatientID.isin(seen)].reset_index(drop=True)
    n["CAMS.leak_patients"] = len(c) - len(clean)
    n["CAMS.clean_rho"] = rho(clean.RawScore, clean.NESI)
    n["CAMS.clean_bespoke"] = rho(clean.RawScore, clean.BespokeBadnessScore)
    B = _patient_boot(ctx, clean, lambda x: (rho(x.RawScore, x.NESI), rho(x.RawScore, x.BespokeBadnessScore)), n=4000)
    n["CAMS.clean_rho_lo"], n["CAMS.clean_rho_hi"] = np.percentile(B[:, 0], [2.5, 97.5])
    n["CAMS.clean_delta"] = n["CAMS.clean_rho"] - n["CAMS.clean_bespoke"]
    n["CAMS.clean_delta_lo"], n["CAMS.clean_delta_hi"] = np.percentile(B[:, 0] - B[:, 1], [2.5, 97.5])


def supp_table1(ctx):
    f, rows = ctx.scores(), []
    for ds in SCALES:
        for lvl, v in f[(f.Dataset == ds) & (f.Split == "Test")].groupby("RawScore").NESI:
            q = (v.median(), v.quantile(0.25), v.quantile(0.75))
            rows.append(dict(Scale=ds, Score=int(lvl), Median=q[0], Q1=q[1], Q3=q[2], N=len(v)))
            ctx.num[f"T1.{ds}.{int(lvl)}.n"] = len(v)
            for name, val in zip(["med", "q1", "q3"], q):
                ctx.num[f"T1.{ds}.{int(lvl)}.{name}"] = val
    ctx.table(pd.DataFrame(rows).round(3), "supp_table1")


def supp_tables_2_3(ctx):
    f, rows = ctx.scores(), []
    for ds, sev in SEVERE.items():
        t = f[f.Dataset == ds]
        tr, te = t[t.Split != "Test"], t[t.Split == "Test"]
        y, yt = sev(tr.RawScore).astype(int), sev(te.RawScore).astype(int)
        p = LogisticRegression(C=1e6).fit(tr[["NESI"]], y).predict_proba(te[["NESI"]])[:, 1]
        fp, tp, th = roc_curve(y, tr.NESI)
        cut = th[np.argmax(tp - fp)]
        hit = te.NESI >= cut
        res = dict(
            cut=cut,
            sens=(hit & (yt == 1)).sum() / yt.sum(),
            spec=((~hit) & (yt == 0)).sum() / (yt == 0).sum(),
            ppv=(hit & (yt == 1)).sum() / hit.sum(),
            npv=((~hit) & (yt == 0)).sum() / (~hit).sum(),
            auc=roc_auc_score(yt, te.NESI),
            brier=brier_score_loss(yt, p),
        )
        for k, v in res.items():
            ctx.num[f"T3.{ds}.{k}"] = v
        rows.append(dict(Scale=ds, **res))
    ctx.table(pd.DataFrame(rows).round(3), "supp_tables_2_3")


def threshold_aurocs(ctx):
    f = ctx.scores()
    for ds in SCALES:
        te = f[(f.Dataset == ds) & (f.Split == "Test")]
        vals = []
        for k in np.sort(te.RawScore.unique())[:-1]:
            y = (te.RawScore <= k) if ds in ("RASS", "GCS") else (te.RawScore > k)
            vals.append(roc_auc_score(y, te.NESI))
        ctx.num[f"AUC.{ds}.min"], ctx.num[f"AUC.{ds}.max"] = min(vals), max(vals)


def ordinal_agreement(ctx):
    """Cutoffs are midpoints of the median NESI of adjacent classes in training and validation data."""
    f = ctx.scores()
    for ds in SCALES:
        t = f[f.Dataset == ds]
        te, tr = t[t.Split == "Test"], t[t.Split != "Test"]
        g = np.sort(tr.GroupedScore.unique())
        med = [tr[tr.GroupedScore == x].NESI.median() for x in g]
        cuts = sorted((med[i] + med[i + 1]) / 2 for i in range(len(g) - 1))
        truth = te.GroupedScore.rank(method="dense").astype(int) - 1
        if med[0] > med[-1]:
            truth = truth.max() - truth
        pred = np.digitize(te.NESI, cuts)
        ctx.num[f"ORD.{ds}.maoe"] = np.abs(truth - pred).mean()
        ctx.num[f"ORD.{ds}.qwk"] = cohen_kappa_score(truth, pred, weights="quadratic")


def institutions_age_duration(ctx):
    """Supplementary Table 15, institution and age rows. Site comes from each segment's BIDS prefix (S0001 MGH,
    S0002 BWH; the I0002 recordings are BIDMC), age group from the patient's median age over their test segments.
    Patient-bootstrap CIs use their own seed, so other bootstraps are unchanged."""
    f, rows, n = ctx.scores(), [], ctx.num
    rng = np.random.default_rng(2027)

    def boot(t):
        groups = [g.index.values for _, g in t.groupby("BDSPPatientID")]
        vals = [rho(t.loc[ix, "RawScore"], t.loc[ix, "NESI"]) for ix in
                (np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(ctx.boot))]
        return np.percentile(vals, [2.5, 97.5])

    def put(prefix, v, group, ds):
        n[f"{prefix}.rho"], n[f"{prefix}.patients"], n[f"{prefix}.segments"] = rho(v.RawScore, v.NESI), v.BDSPPatientID.nunique(), len(v)
        n[f"{prefix}.lo"], n[f"{prefix}.hi"] = boot(v.reset_index(drop=True))
        rows.append(dict(Scale=ds, Group=group, Patients=n[f"{prefix}.patients"], Segments=len(v), rho=n[f"{prefix}.rho"],
                         lo=n[f"{prefix}.lo"], hi=n[f"{prefix}.hi"]))

    for ds in ["RASS", "GCS"]:
        md = ctx.meta(ds)
        x = md.drop_duplicates(["BDSPPatientID", "SessionID"]).DurationInSeconds / 3600
        n[f"DUR.{ds}.median_h"] = x.median()
        n[f"DUR.{ds}.pct_le1h"] = (x <= 1).mean() * 100
    for ds in ["RASS", "GCS", "CAMS"]:
        t = f[(f.Dataset == ds) & (f.Split == "Test")].copy()
        code = t.MorgothOutputFilename.str.extract(r"sub-(S\d{4})")[0]
        t["Site"] = code.map({"S0001": "MGH", "S0002": "BWH"}).fillna("BIDMC")
        for s, v in t.groupby("Site"):
            put(f"SITE.{ds}.{s}", v, s, ds)
        if ds == "CAMS":
            continue
        md = ctx.meta(ds)
        m = md.drop_duplicates(["BDSPPatientID", "SessionID"])[["BDSPPatientID", "SessionID", "AgeAtVisit"]].copy()
        m["ses"] = pd.to_numeric(m.SessionID, errors="coerce")
        t["ses"] = t.MorgothOutputFilename.str.extract(r"ses-(\d+)")[0].astype(float)
        t = t.merge(m.drop(columns="SessionID"), on=["BDSPPatientID", "ses"], how="left")
        t["Age"] = t.BDSPPatientID.map(t.groupby("BDSPPatientID").AgeAtVisit.median())
        for nm, v in [("lt65", t[t.Age < 65]), ("ge65", t[t.Age >= 65])]:
            put(f"AGE.{ds}.{nm}", v, nm, ds)
    ctx.table(pd.DataFrame(rows).round(3), "supp_table12_site_age")
    # diagnosis rows of the same table, from by_diagnosis.py
    dx = ctx.out / "tables" / "performance_by_diagnosis.csv"
    for r in pd.read_csv(dx).itertuples():
        key = f"DXP.{r.Scale}.{r.Dx.replace('/', '_')}"
        n[f"{key}.patients"], n[f"{key}.segments"] = r.Patients, r.Segments
        for k in ["rho", "lo", "hi"]:
            if pd.notna(getattr(r, k)):
                n[f"{key}.{k}"] = getattr(r, k)


def table1(ctx):
    n = ctx.num
    rm = ctx.meta("RASS")
    num = pd.to_numeric(rm["R PHS IP RASS"], errors="coerce")
    for s in [1, 2, 3, 4]:
        n[f"RASSpos.{s}"] = (num == s).sum()
    n["RASS.raw_patients"] = rm.BDSPPatientID.nunique()
    p = rm.sort_values("EEGBeginDTS").groupby("BDSPPatientID").first()
    n["T1.RASS.male"], n["T1.RASS.female"] = (p.SexDSC == "Male").sum(), (p.SexDSC == "Female").sum()
    n["T1.RASS.age_med"] = p.AgeAtVisit.median()
    for ds, d in [("RASS", rm), ("GCS", ctx.meta("GCS"))]:
        a, e = pd.to_datetime(d.HospitalAdmitDTS, errors="coerce").dt.normalize(), pd.to_datetime(d.HospitalDischargeDTS, errors="coerce").dt.normalize()
        x = pd.to_datetime(d.DateOfDeath, errors="coerce").dt.normalize()
        g = d.assign(i=(x >= a) & (x <= e), h=x.notna()).groupby("BDSPPatientID").agg(i=("i", "max"), h=("h", "max"))
        if ds == "GCS":
            # BIDMC has placeholders in death_date: parse dates rather than test for non-null
            b = pd.read_csv(ctx.s3 / "GCS_BIDMC_HarvardEEG_metadata.csv").groupby("BDSPPatientID").first()
            bx = pd.to_datetime(b.death_date, errors="coerce").dt.normalize()
            ba, be = pd.to_datetime(b.t0_admit, errors="coerce").dt.normalize(), pd.to_datetime(b.t1_discharge, errors="coerce").dt.normalize()
            g = pd.concat([g, pd.DataFrame({"i": (bx >= ba) & (bx <= be), "h": bx.notna()})])
            n["GCS.raw_patients"] = len(g)
            n["T1.GCS.placeholder_deaths"] = b.death_date.notna().sum() - bx.notna().sum()
        n[f"T1.{ds}.inhosp"] = g.i.sum()
        n[f"T1.{ds}.other_death"] = (g.h & ~g.i).sum()
        n[f"T1.{ds}.no_death"] = (~g.h).sum()
    c = pd.read_csv(ctx.s3 / "CAMS_cohort_metadata.csv").drop_duplicates("BDSPPatientID")
    n["CAMS.raw_patients"] = len(c)
    n["T1.CAMS.age_med"] = pd.to_numeric(c["Age "], errors="coerce").median()
    n["T1.CAMS.male"], n["T1.CAMS.female"] = (c.Gender == "Male").sum(), (c.Gender == "Female").sum()
    n["T1.CAMS.any_death"] = c["DateOfDeath (shifted)"].notna().sum()


def mortality_folds(ctx):
    import pickle

    p = ctx.repo / "NESI/DeathPrediction_NESIvsGCS/Results"
    a = {k: pickle.load(open(p / f"results_{k}_Death.pkl", "rb")) for k in ["NESI", "GCS"]}
    for h in [4, 20]:
        u = np.array([x[h]["auroc"] for x in a["NESI"]])
        g = np.array([x[h]["auroc"] for x in a["GCS"]])
        ctx.num[f"MORT.NESI.{h}h"], ctx.num[f"MORT.GCS.{h}h"] = u.mean(), g.mean()
        if h == 20:
            ctx.num["MORT.p_paired_t_one_sided"] = stats.ttest_rel(u, g)[1] / 2


def retest(ctx):
    n, t = ctx.num, ctx.meds().copy()
    flag = lambda c: t[c].fillna(0).astype(float) > 0
    t["t"] = pd.to_datetime(t.RASSRecordedDTS, errors="coerce")
    t["state"] = flag("sedative_inf_now").astype(int) * 2 + flag("opioid_inf_now").astype(int)
    t = t.sort_values(["BDSPPatientID", "SessionID", "t"])
    g = t.groupby(["BDSPPatientID", "SessionID"])
    t["dt"], t["pr"], t["ps"], t["pn"] = g.t.diff().dt.total_seconds() / 3600, g.r.shift(), g.state.shift(), g.n.shift()
    p = t[(t.dt > 0) & (t.dt <= 2) & (t.r == t.pr) & (t.state == t.ps)]
    sem = np.sqrt(((p.n - p.pn) ** 2).mean() / 2)
    n["RETEST.pairs"], n["RETEST.patients"], n["RETEST.sem"] = len(p), p.BDSPPatientID.nunique(), sem
    n["RETEST.mdc"] = 1.96 * np.sqrt(2) * sem
    n["RETEST.icc"] = 1 - sem**2 / np.concatenate([p.n, p.pn]).var(ddof=1)
    med = ctx.meds().groupby("r").n.median()  # median NESI at each RASS level, all assessments
    adj = (med.loc[-4:-1].values[:-1] - med.loc[-4:-1].values[1:])
    n["RETEST.adj_min"], n["RETEST.adj_max"] = adj.min(), adj.max()
    n["RETEST.range_5_0"] = med.loc[-5] - med.loc[0]


def _sedation_groups(ctx):
    """Per-assessment sedative groups (Christine Eckhardt's labels, build_strata_labels.py: infusion running or
    stopped within 6 h; each sedative pooled with or without an opioid infusion), joined on patient and RASS time."""
    d = ctx.meds().copy()
    lab = pd.read_csv(ctx.root / "sedation_repo/medication-exposure/data/stratified_feasibility/labels/strata_labels_rass.csv")
    lab["t"] = pd.to_datetime(lab.assessment_time)
    d["t"] = pd.to_datetime(d.RASSRecordedDTS, errors="coerce")
    d = d.merge(lab.drop(columns=["assessment_time", "score"]), on=["BDSPPatientID", "t"], how="left", validate="many_to_one")
    b = lambda c: d[c].fillna(0).astype(float) > 0
    bz, asm = b("FastActing_Benzo_Exposure"), b("LongActingASM_Exposure")
    none24 = ~b("sedative_inf_24h") & ~b("opioid_inf_24h") & ~bz
    G = {
        "Reference": b("Reference (no sedative or opioid infusion)"),
        "Propofol": b("Propofol (+/- opioid)"),
        "Dexmedetomidine": b("Dexmedetomidine (+/- opioid)"),
        "Midazolam": b("Midazolam (+/- opioid)"),
        "MidazolamPropofol": b("Midazolam + propofol (+/- opioid)"),
        "Ketamine": b("Ketamine (+/- opioid)"),
        "OpioidOnly": b("Opioid only"),
        "All": pd.Series(True, index=d.index),
        "NoInfusion24h": none24,
        "NoInfusion24hNoASM": none24 & ~asm,
    }
    return d, G


def supp_table11(ctx):
    """NESI-RASS performance by sedative group (groups from _sedation_groups)."""
    n = ctx.num
    d, G = _sedation_groups(ctx)
    metrics = {"rho": lambda x: rho(x.r, x.n), "auc3": lambda x: roc_auc_score(x.r <= -3, x.n), "auc4": lambda x: roc_auc_score(x.r <= -4, x.n)}
    rows = []
    for k, v in G.items():
        x = d[v].reset_index(drop=True)
        res = dict(obs=len(x), patients=x.BDSPPatientID.nunique())
        for name, fn in metrics.items():
            res[name] = fn(x)
            res[name + "_lo"], res[name + "_hi"] = np.percentile(_patient_boot(ctx, x, fn), [2.5, 97.5])
        for name, val in res.items():
            n[f"T11.{k}.{name}"] = val
        rows.append(dict(Group=k, **res))
    ctx.table(pd.DataFrame(rows).round(3), "supp_table11")


def sedation_range(ctx):
    """How much of the lower NESI-RASS correlation within sedative groups comes from their narrower RASS range:
    the reference group is resampled, within each RASS level, to each group's RASS distribution (200 draws of
    5,000 observations; own seed, so other bootstraps are unchanged), and its correlation recomputed."""
    n = ctx.num
    d, G = _sedation_groups(ctx)
    for k, v in G.items():
        n[f"T11.{k}.rass_sd"] = d[v].r.std()
    rng = np.random.default_rng(2026)
    ref = d[G["Reference"]]
    by_level = {r: g.index.values for r, g in ref.groupby("r")}
    for k in ["Reference", "Propofol", "Dexmedetomidine", "Midazolam", "MidazolamPropofol", "Ketamine", "OpioidOnly"]:
        x = d[G[k]]
        n[f"T11.{k}.deep_pct"] = 100 * (x.r <= -4).mean()
        if k == "Reference":
            continue
        share = x.r.value_counts(normalize=True)
        vals = []
        for _ in range(200):
            ix = np.concatenate([rng.choice(by_level[r], int(round(w * 5000))) for r, w in share.items() if r in by_level])
            vals.append(rho(ref.loc[ix, "r"], ref.loc[ix, "n"]))
        n[f"T11.{k}.ref_matched_rho"] = np.mean(vals)


def propofol(ctx):
    import statsmodels.formula.api as smf

    n = ctx.num
    # Corrected Eleveld V2 age coefficient (Christine Eckhardt's revision file)
    w = pd.read_csv(ctx.root / "sedation_repo/eleveld-pk-analysis/data/wt_ht_per_hosp_FULL_with_eleveld_expanded.csv", low_memory=False)
    w["NE"], w["r"] = pd.to_numeric(w.PredictedNESI, errors="coerce"), pd.to_numeric(w["R PHS IP RASS"], errors="coerce")
    q = w[(w.propofol_only_recomputed == True) & (w.PropofolSimStatus == "simulated") & w.Ce_ug_per_mL.notna() & w.NE.notna() & w.r.between(-5, 0)].rename(columns={"Ce_ug_per_mL": "CE"})  # noqa: E712
    n["PROP.obs"], n["PROP.patients"] = len(q), q.BDSPPatientID.nunique()
    Q = q[q.groupby("BDSPPatientID").NE.transform("size") >= 5].copy()
    n["VAR.obs"], n["VAR.patients"] = len(Q), Q.BDSPPatientID.nunique()
    # Spearman rho on the same rows as the variance decomposition (rank_correlation.py, Nov 3)
    n["PROP.rho_nesi"], n["PROP.rho_rass"] = stats.spearmanr(Q.CE, Q.NE)[0], stats.spearmanr(Q.CE, Q.r)[0]
    Q["m"], Q["d"] = Q.groupby("BDSPPatientID").CE.transform("mean"), Q.CE - Q.groupby("BDSPPatientID").CE.transform("mean")
    for y, tag in [("NE", "nesi"), ("r", "rass")]:
        fit = smf.mixedlm(f"{y}~CE", Q, groups=Q.BDSPPatientID).fit(reml=True)
        vf, vr, ve = np.var(fit.fe_params.CE * Q.CE), float(fit.cov_re.iloc[0, 0]), fit.scale
        tot = vf + vr + ve
        n[f"VAR.{tag}.pct_ce"], n[f"VAR.{tag}.pct_patient"], n[f"VAR.{tag}.pct_resid"] = 100 * vf / tot, 100 * vr / tot, 100 * ve / tot
        n[f"VAR.{tag}.slope"] = fit.fe_params.CE
        wb = smf.mixedlm(f"{y}~d+m", Q, groups=Q.BDSPPatientID).fit(reml=True)
        n[f"WB.{tag}.within"], n[f"WB.{tag}.between"] = wb.fe_params.d, wb.fe_params.m
        per = Q.groupby("BDSPPatientID")[["CE", y]].apply(lambda g: stats.spearmanr(g.CE, g[y])[0] if g.CE.nunique() > 1 and g[y].nunique() > 1 else np.nan).dropna()
        n[f"WB.{tag}.median_rho"], n[f"WB.{tag}.n"] = per.median(), len(per)


def sedation_strata(ctx):
    """Point estimates from Christine Eckhardt's pipeline, read from its output when present."""
    f = ctx.root / "sedation_repo/medication-exposure/data/stratified_feasibility/model/results_main_stratum_full.csv"
    d = pd.read_csv(f)
    for r in d.itertuples():
        k = re.sub(r"[^A-Za-z]+", "", r.agent.split("(")[0])
        ctx.num[f"T5.{k}.obs"], ctx.num[f"T5.{k}.patients"], ctx.num[f"T5.{k}.offset"] = r.n_obs, r.n_patients, r.add_offset


def revision_tables(ctx):
    """Values from reproduce/by_diagnosis.py and reproduce/rass_positive.py, when run."""
    t = ctx.out / "tables"
    d = pd.read_csv(t / "performance_by_diagnosis.csv").dropna(subset=["rho"])
    for ds in ["RASS", "GCS"]:
        v = d[d.Scale == ds]
        ctx.num[f"DX.{ds}.min"], ctx.num[f"DX.{ds}.max"] = v.rho.min(), v.rho.max()
    r = pd.read_csv(t / "rass_positive_by_level.csv").set_index("RASS").Median
    for lvl in [0, 1, 2]:
        ctx.num[f"RASSPOS.median.{lvl}"] = r.loc[lvl]
    a = pd.read_csv(t / "ablation_summary.csv")
    a = a[a.Seed == 2026]
    for x in a.itertuples():
        ctx.num[f"ABL.{x.Head}.{x.Scale}.rho"], ctx.num[f"ABL.{x.Head}.{x.Scale}.delta"] = x.rho, x.delta
    ctx.num.update(json.load(open(ctx.root / "intermediates/external/numbers.json")))
    m = pd.read_csv(t / "mortality_summary.csv", header=None, index_col=0)[1]
    for k, v in m.items():
        ctx.num[f"MORT2.{k}"] = v
    d = pd.read_csv(t / "mortality_delong.csv")
    for x in d.itertuples():
        k = f"MORT2.delta.{x.a}|{x.b}|{x.hour}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"], ctx.num[k + ".p_holm"] = x.delta, x.lo, x.hi, x.p_holm
    # likelihood-ratio tests, NRI and calibration (mortality_extra.py)
    for x in pd.read_csv(t / "mortality_lr.csv").itertuples():
        ctx.num[f"MORT3.lr.{x.full}|{x.reduced}|{x.hour}.p"] = x.p
    for x in pd.read_csv(t / "mortality_nri.csv").itertuples():
        k = f"MORT3.nri.{x.new}|{x.old}|{x.hour}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.nri, x.lo, x.hi
    cal = pd.read_csv(t / "mortality_calibration.csv")
    ctx.num["MORT3.cal_slope_min"], ctx.num["MORT3.cal_slope_max"] = cal.slope.min(), cal.slope.max()
    ctx.num["MORT3.cal_int_corrected_min"], ctx.num["MORT3.cal_int_corrected_max"] = cal.intercept_corrected.min(), cal.intercept_corrected.max()
    # Supplementary Table 19: Eleveld implementation check (sedation.py)
    e = pd.read_csv(t / "eleveld_vs_pas_agreement.csv")
    for _, x in e.iterrows():
        for k in ["MDPE_%", "MDAPE_%", "P95_abs_%", "max_abs_%", "max_abs_ugml", "n"]:
            ctx.num[f"ELEV.{x['subset']}.{k}"] = float(x["ce_" + k])
    # Supplementary Fig. 30: qEEG-trained heads (qeeg_heads.py)
    for x in pd.read_csv(t / "qeeg_heads.csv").itertuples():
        k = f"QHEAD.{x.Dataset}.{x.Feature}"
        ctx.num[k], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.Spearman, x.CI_Lower, x.CI_Upper
    # Supplementary Table 20, the rows computed from the analyzed recordings
    r = pd.read_csv(ctx.root / "intermediates/external/recordings.csv")
    ctx.num["EXT20.patients"] = len(r)
    for c in ["gcs", "rass"]:
        ctx.num[f"EXT20.{c}.median"], ctx.num[f"EXT20.{c}.q1"], ctx.num[f"EXT20.{c}.q3"] = r[c].median(), r[c].quantile(0.25), r[c].quantile(0.75)
    for lab, v in {"positive": "Positive", "negative": "Negative", "uta": "UTA, RASS score -4 or -5"}.items():
        ctx.num[f"EXT20.cam.{lab}"] = int((r.cam == v).sum())
    m = r.duration_s / 60
    ctx.num.update({"EXT20.minutes.median": m.median(), "EXT20.minutes.q1": m.quantile(0.25), "EXT20.minutes.q3": m.quantile(0.75),
                    "EXT20.minutes.min": m.min(), "EXT20.minutes.max": m.max()})
    # and the rows for all enrolled patients, from Northwestern's date-free per-recording file (ages over 89 coded 90)
    d = pd.read_csv(ctx.root / "intermediates/external/demographics.csv")
    ctx.num["EXT20.enrolled"] = len(d)
    ctx.num["EXT20.age.median"], ctx.num["EXT20.age.q1"], ctx.num["EXT20.age.q3"] = d.Age.median(), d.Age.quantile(0.25), d.Age.quantile(0.75)
    for c in ["Sex", "Race", "Ethnicity", "AdmissionDiagnosis_categories", "EEG_indication"]:
        for v, k in d[c].value_counts().items():
            ctx.num[f"EXT20.{c}.{v}"] = int(k)
    ctx.num["EXT20.death.hospital"] = int(d.Death_during_hospitalization.sum())
    ctx.num["EXT20.death.1yr"] = int(d.Death_within_1yr_postdischarge.sum())
    # Methods: sampling rate of every Northwestern EDF (Northwestern's channel check) and the RASS range analyzed
    f = pd.read_csv(ctx.root / "intermediates/external/eeg_sampling.csv")
    ctx.num["EXT.edf.n"] = len(f)
    for hz, k in f.Sampling_frequency_Hz.value_counts().items():
        ctx.num[f"EXT.edf.fs.{hz}"] = int(k)
    ctx.num["EXT.rass.max"] = r.rass.max()
    a = pd.read_csv(t / "artifact_tolerance.csv")
    for x in a.itertuples():
        ctx.num[f"ART.{x.Scale}.{x.Artifact}.rho"], ctx.num[f"ART.{x.Scale}.{x.Artifact}.lo"], ctx.num[f"ART.{x.Scale}.{x.Artifact}.hi"] = x.rho, x.lo, x.hi
    r = pd.read_csv(t / "retrain_summary.csv")
    for x in r.itertuples():
        ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.rho"], ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.lo"], ctx.num[f"RETRAIN.{x.Run}.{x.Scale}.hi"] = x.rho_new, x.lo, x.hi
    q = pd.read_csv(t / "qeeg_supp_fig31.csv")
    for x in q.itertuples():
        ctx.num[f"QEEG.{x.Scale}.{x.Feature}.rho"], ctx.num[f"QEEG.{x.Scale}.{x.Feature}.lo"], ctx.num[f"QEEG.{x.Scale}.{x.Feature}.hi"] = x.rho, x.lo, x.hi
    lo = pd.read_csv(t / "loco_summary.csv")
    for x in lo.itertuples():
        for c in ["full_rho", "full_lo", "full_hi", "loco_rho", "loco_lo", "loco_hi", "uni_rho", "uni_lo", "uni_hi", "delta", "delta_lo", "delta_hi"]:
            ctx.num[f"LOCO.{x.Scale}.{c}"] = getattr(x, c)
    mp = pd.read_csv(t / "mapping_summary.csv")
    for x in mp.itertuples():
        k = f"MAP.{x.Mapping.replace(' ', '_')}.{x.Scale}"
        ctx.num[k + ".rho"], ctx.num[k + ".lo"], ctx.num[k + ".hi"], ctx.num[k + ".delta"], ctx.num[k + ".p_holm"] = x.rho, x.lo, x.hi, x.delta, x.p_holm
    ia = pd.read_csv(t / "icans_segment_ablation.csv")
    for x in ia.itertuples():
        ctx.num[f"ICSEG.{x.Set}.rho"], ctx.num[f"ICSEG.{x.Set}.lo"], ctx.num[f"ICSEG.{x.Set}.hi"] = x.rho, x.lo, x.hi
    sub = pd.read_csv(t / "icans_subgroups.csv")
    for i, x in enumerate(sub.itertuples()):
        ctx.num[f"ICSUB.{i}.rho"], ctx.num[f"ICSUB.{i}.lo"], ctx.num[f"ICSUB.{i}.hi"] = x.rho, x.lo, x.hi
    ind = t.parent / "indication"
    short = lambda s: s.split(" / ")[0].replace(" ", "_")
    for x in pd.read_csv(ind / "NESI_Indication_Spearman_Train_Test_Bootstrap1000.csv").itertuples():
        k = f"IND.{x.Split}.{short(x.Indication)}.{x.Scale}"
        ctx.num[k + ".rho"], ctx.num[k + ".lo"], ctx.num[k + ".hi"] = x.Rho, x.CI_Lower, x.CI_Upper
    for x in pd.read_csv(ind / "NESI_Indication_UniquePatientCounts_Train_Test_All.csv").itertuples():
        ctx.num[f"INDN.{short(x.Indication)}.{x.Scale}"] = x._5
    for x in pd.read_csv(t / "supp_fig10_mae_by_dx.csv").itertuples():
        ctx.num[f"SF10.{x.Dx.split('/')[0]}.median"] = x.median
    f6 = pd.read_csv(t / "figure6_counts.csv")
    for x in f6.itertuples():
        ctx.num["FIG6." + x.Panel.replace(" ", "_")] = x.Selected


def table1_scripts(ctx):
    """Every Table 1 cell from the original Table1_<cohort>.py scripts (their printed output, written to
    output/logs/table1_<cohort>.log by the scripts step). Percentages are recomputed over the cohort size."""
    import re

    labels = {
        "Male": "male", "Female": "female", "Unknown or not disclosed": "sex_unknown", "Unknown": "sex_unknown",
        "White": "white", "Black/African American": "black", "Black": "black", "Asian": "asian",
        "More than one race": "multi", "Other/Unknown": "race_other",
        "Censored (no death date)": "no_death", "Died in hospital": "inhosp",
        "No matching in-hospital death": "outhosp", "Died outside recorded hospital windows": "outhosp",
        "Neurological": "dx_neuro", "Cardiac/Circulatory": "dx_cardiac", "Systemic/Metabolic": "dx_systemic",
        "Respiratory": "dx_resp", "Unknown/Other": "dx_other",
    }
    for coh in ["RASS", "GCS", "CAMS", "ICANS"]:
        log = ctx.out / "logs" / f"table1_{coh.lower()}.log"
        txt = log.read_text().split("--STDERR--")[0]
        n = ctx.num
        N = int(re.findall(r"Total Unique subjects in \w+ cohort == (\d+)", txt)[-1])
        n[f"T1S.{coh}.N"] = N
        med, q1, q3 = map(float, re.search(r"median \[Q1, Q3\]\): ([\d.]+) \[([\d.]+), ([\d.]+)\]", txt).groups())
        n[f"T1S.{coh}.age_med"], n[f"T1S.{coh}.age_q1"], n[f"T1S.{coh}.age_q3"] = med, q1, q3
        section = ""
        for line in txt.splitlines():
            if "DIAGNOSIS-MALIGNANCY" in line:
                section = "malig"
            elif "DIAGNOSIS" in line:
                section = "dx"
            m = re.match(r"\s*([A-Za-z/ ()-]+?):?\s+(\d+)(?: \(\d|\s*$)", line)
            if not m or section == "malig":
                continue
            key = labels.get(m.group(1).strip())
            if key is None or (key.startswith("dx_") and section != "dx"):
                continue
            if f"T1S.{coh}.{key}.n" in n and key == "no_death":
                continue
            n[f"T1S.{coh}.{key}.n"] = int(m.group(2))
            n[f"T1S.{coh}.{key}.pct"] = 100 * int(m.group(2)) / N
        if f"T1S.{coh}.sex_unknown.n" not in n:  # printed only when nonzero
            u = N - n[f"T1S.{coh}.male.n"] - n[f"T1S.{coh}.female.n"]
            n[f"T1S.{coh}.sex_unknown.n"], n[f"T1S.{coh}.sex_unknown.pct"] = u, 100 * u / N
        if coh == "ICANS":  # every ICANS patient is in the haematological-malignancy group, shown as Unknown/Other
            n["T1S.ICANS.dx_other.n"], n["T1S.ICANS.dx_other.pct"] = N, 100.0
            for k in ["dx_neuro", "dx_cardiac", "dx_systemic", "dx_resp", "multi"]:
                n.setdefault(f"T1S.ICANS.{k}.n", 0)
                n.setdefault(f"T1S.ICANS.{k}.pct", 0.0)


def sedation_tables(ctx):
    """Every number in the sedation tables as rebuilt by sedation.py (Christine Eckhardt's build_tables.py writes them
    to nesi_tables.docx): key SEDT.<table>.<i>, the i-th number in reading order. Table X is the main Table 2;
    S1-S5b are Supplementary Tables 8-13; S6, S6b, S6c are Supplementary Tables 16-18."""
    import re, zipfile
    from lxml import etree

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    path = ctx.root / "sedation_repo/medication-exposure/data/stratified_feasibility/tables/nesi_tables.docx"
    root = etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    cap, seq = None, {}
    for el in root.find(W + "body"):
        t = "".join(x.text or "" for x in el.iter(W + "t"))
        if el.tag == W + "p" and t.strip():
            m = re.match(r"\s*Table ([A-Z0-9]+[a-z]?)\.", t)
            cap = m.group(1) if m else cap
        elif el.tag == W + "tbl" and cap:
            for tc in el.iter(W + "tc"):
                cell = "".join(x.text or "" for x in tc.iter(W + "t"))
                seq.setdefault(cap, []).extend(re.findall(r"\d[\d,]*(?:\.\d+)?", cell))
    for cap, toks in seq.items():
        for i, x in enumerate(toks):
            ctx.num[f"SEDT.{cap}.{i}"] = float(x.replace(",", ""))


def calibration_table(ctx):
    """Supplementary Table 2: ECE and Brier score of severe-state probabilities on the test set, with patient-bootstrap
    95% CIs. Port of RevisionRelatedMaterials/NESIScaleCalibration/NESI_calibration_analysis.py (Arka Roy): severe =
    grouped score 2; unpenalized logistic model on training and validation data; 6 equal-width bins; 1,000 resamples,
    seed 42, cohorts in the order RASS, GCS, CAM-S, ICANS."""
    from sklearn.linear_model import LogisticRegression

    f = ctx.scores().copy()
    f["Severe"] = (pd.to_numeric(f.GroupedScore, errors="coerce") == 2).astype(int)
    rng, nb = np.random.default_rng(42), 6

    def ece(y, p, b):
        c = np.bincount(b, minlength=nb); pos = np.bincount(b, weights=y, minlength=nb); pr = np.bincount(b, weights=p, minlength=nb)
        v = c > 0
        return np.sum(c[v] / len(y) * np.abs(pos[v] / c[v] - pr[v] / c[v]))

    for ds in ["RASS", "GCS", "CAMS", "ICANS"]:
        d = f[f.Dataset == ds]
        dev = d[d.Split.isin(["Train", "Val"])]
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-8).fit(dev[["NESI"]], dev.Severe)
        t = d[d.Split == "Test"].copy().reset_index(drop=True)
        t["p"] = m.predict_proba(t[["NESI"]])[:, 1]
        t = t.dropna(subset=["BDSPPatientID", "Severe", "p"]).reset_index(drop=True)
        y, p = t.Severe.to_numpy(int), t.p.to_numpy(float)
        b = np.digitize(p, np.linspace(0, 1, nb + 1)[1:-1])
        groups = [g.index.to_numpy() for _, g in t.astype({"BDSPPatientID": str}).groupby("BDSPPatientID", sort=False)]
        be, bb = [], []
        for _ in range(1000):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            be.append(ece(y[ix], p[ix], b[ix])); bb.append(np.mean((p[ix] - y[ix]) ** 2))
        n = ctx.num
        n[f"CAL.{ds}.obs"], n[f"CAL.{ds}.patients"] = len(t), len(groups)
        n[f"CAL.{ds}.ece"], n[f"CAL.{ds}.brier"] = ece(y, p, b), np.mean((p - y) ** 2)
        n[f"CAL.{ds}.ece_lo"], n[f"CAL.{ds}.ece_hi"] = np.percentile(be, [2.5, 97.5])
        n[f"CAL.{ds}.brier_lo"], n[f"CAL.{ds}.brier_hi"] = np.percentile(bb, [2.5, 97.5])


def threshold_cis(ctx):
    """Supplementary Table 3: patient-bootstrap 95% CIs (1,000 resamples, own seed) for sensitivity, specificity, PPV,
    NPV and AUROC at the training-set Youden cutoff of supp_tables_2_3."""
    f, rng = ctx.scores(), np.random.default_rng(2028)
    for ds, sev in SEVERE.items():
        t = f[f.Dataset == ds]
        tr, te = t[t.Split != "Test"], t[t.Split == "Test"].reset_index(drop=True)
        fp, tp, th = roc_curve(sev(tr.RawScore).astype(int), tr.NESI)
        cut = th[np.argmax(tp - fp)]
        yt, hit = sev(te.RawScore).astype(int).to_numpy(), (te.NESI >= cut).to_numpy()
        groups = [g.index.to_numpy() for _, g in te.groupby("BDSPPatientID")]
        out = {k: [] for k in ["sens", "spec", "ppv", "npv", "auc"]}
        for _ in range(1000):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            y, h = yt[ix], hit[ix]
            out["sens"].append((h & (y == 1)).sum() / max(y.sum(), 1))
            out["spec"].append((~h & (y == 0)).sum() / max((y == 0).sum(), 1))
            out["ppv"].append((h & (y == 1)).sum() / max(h.sum(), 1))
            out["npv"].append((~h & (y == 0)).sum() / max((~h).sum(), 1))
            out["auc"].append(roc_auc_score(y, te.NESI.to_numpy()[ix]) if 0 < y.sum() < len(y) else np.nan)
        for k, v in out.items():
            ctx.num[f"T3.{ds}.{k}_lo"], ctx.num[f"T3.{ds}.{k}_hi"] = np.nanpercentile(v, [2.5, 97.5])


MED_ROWS = {
    "asm": ["LongActingASM_Exposure"], "benzo_par": ["FastActing_Benzo_Exposure"], "benzo_ent": ["SlowActing_Benzo_Exposure"],
    "sed_inf": ["Propofol_Exposure", "Propofol_Infusion_Exposure", "Midazolam_Infusion_Exposure", "Ketamine_Infusion_Exposure",
                "Dexmedetomidine_Infusion_Exposure"],
    "propofol": ["Propofol_Exposure", "Propofol_Infusion_Exposure"], "midazolam": ["Midazolam_Infusion_Exposure"],
    "ketamine": ["Ketamine_Infusion_Exposure"], "dexmed": ["Dexmedetomidine_Infusion_Exposure"],
    "opi_inf": ["Fentanyl_Infusion_Exposure", "Morphine_Infusion_Exposure", "Hydromorphone_Infusion_Exposure"],
    "fentanyl": ["Fentanyl_Infusion_Exposure"], "morphine": ["Morphine_Infusion_Exposure"], "hydromorphone": ["Hydromorphone_Infusion_Exposure"],
    "opi_par": ["FastActing_Opiate_Exposure"], "opi_ent": ["SlowActing_Opiate_Exposure"], "antipsych": ["Antipsychotic_Exposure"],
}


def medication_tables(ctx):
    """Supplementary Tables 22 (patients) and 23 (EEG segments): exposure within 24 h before each EEG/score, from the
    per-EEG master exposure files of Christine Eckhardt's medication-exposure pipeline (build_final_exposure_tables.py):
    rows kept if they match a NESI value by patient and assessment time (all CAM-S rows have one); a missing exposure
    flag means no medication administration record. Keys MED22/MED23.<cohort>.<row>.n/.pct, plus .mar/.total."""
    norm = lambda s: s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    def keep(df, lookup, ts_df, ts_lk):
        lk = pd.read_csv(ctx.s3 / lookup, low_memory=False, usecols=["BDSPPatientID", ts_lk, "NESI"])
        lk = lk[lk.NESI.notna()]
        keys = set(zip(norm(lk.BDSPPatientID), pd.to_datetime(lk[ts_lk], errors="coerce").astype("int64")))
        k = zip(norm(df.BDSPPatientID), pd.to_datetime(df[ts_df], errors="coerce").astype("int64"))
        return df[[x in keys for x in k]]

    a = pd.read_csv(ctx.s3 / "i0001_per_eeg_master_exposure.csv", low_memory=False)
    b = pd.read_csv(ctx.s3 / "i0002_per_eeg_master_exposure.csv", low_memory=False)
    cohorts = {
        "I0001GCS": keep(a[a._cohort == "GCS"], "GCS_NESI_lookup.csv", "GCSRecordedDTS", "GCSRecordedDTS"),
        "I0002GCS": keep(b, "GCS_NESI_lookup.csv", "RecordedDTS", "GCSRecordedDTS"),
        "RASS": keep(a[a._cohort == "RASS"], "RASS_NESI_lookup.csv", "RASSRecordedDTS", "RASSRecordedDTS"),
        "CAMS": pd.read_csv(ctx.s3 / "i0001_cams_per_eeg_master_exposure.csv", low_memory=False),
    }
    n = ctx.num
    for c, d in cohorts.items():
        d = d.reset_index(drop=True)
        has = d.LongActingASM_Exposure.notna()
        dm = d[has]
        pts = dm.BDSPPatientID.nunique()
        n[f"MED22.{c}.mar"], n[f"MED22.{c}.total"] = pts, d.BDSPPatientID.nunique()
        n[f"MED23.{c}.mar"], n[f"MED23.{c}.total"] = len(dm), len(d)
        n[f"MED22.{c}.mar_pct"], n[f"MED23.{c}.mar_pct"] = 100 * pts / d.BDSPPatientID.nunique(), 100 * len(dm) / len(d)
        for r, cols in MED_ROWS.items():
            e = dm[cols].fillna(0).max(axis=1).astype(int)
            npt = int(dm.assign(e=e).groupby("BDSPPatientID").e.max().sum())
            n[f"MED22.{c}.{r}.n"], n[f"MED22.{c}.{r}.pct"] = npt, 100 * npt / pts
            n[f"MED23.{c}.{r}.n"], n[f"MED23.{c}.{r}.pct"] = int(e.sum()), 100 * e.sum() / len(dm)


def ext20_percentages(ctx):
    """Supplementary Table 20 percentages: demographics, deaths, diagnosis and indication over the 1,000 enrolled
    patients; CAM categories over the 990 analyzed."""
    n = ctx.num
    for k in [k for k in list(n) if k.startswith("EXT20.") and k.split(".")[1] in
              ("Sex", "Race", "Ethnicity", "death", "AdmissionDiagnosis_categories", "EEG_indication")]:
        n[k + ".pct"] = 100 * n[k] / n["EXT20.enrolled"]
    for k in ["EXT20.cam.positive", "EXT20.cam.negative", "EXT20.cam.uta"]:
        n[k + ".pct"] = 100 * n[k] / n["EXT20.patients"]


def output_tables(ctx):
    """Every numeric cell of the small tables the figure and revision scripts write to output/tables, as
    CSV.<file>.<row>.<column>, so numbers quoted in the text from those tables can be checked."""
    for f in sorted((ctx.out / "tables").glob("*.csv")):
        if f.stat().st_size > 1_000_000:
            continue
        d = pd.read_csv(f)
        for i, row in d.iterrows():
            for c, v in row.items():
                v = pd.to_numeric(v, errors="coerce")
                if pd.notna(v):
                    ctx.num[f"CSV.{f.stem}.{i}.{c}"] = float(v)


def delta_bootstrap(ctx):
    """Universal versus bespoke model on the same test set (Methods): paired cluster bootstrap over patients, 2,000
    resamples (own seed), 95% percentile CI for the difference in rho and a two-sided bootstrap p, Holm-adjusted across
    the four scales."""
    from statsmodels.stats.multitest import multipletests

    f, n, rng, ps = ctx.scores(), ctx.num, np.random.default_rng(2029), {}
    for ds in SCALES:
        b = pd.read_csv(ctx.gh / f"NESI/Bespoke_models/Results/{ds}Bespokescore_results.csv")
        t = f[(f.Dataset == ds) & (f.Split == "Test")].merge(
            b[["MorgothOutputFilename", "BespokeBadnessScore"]], on="MorgothOutputFilename").reset_index(drop=True)
        groups = [g.index.to_numpy() for _, g in t.groupby("BDSPPatientID")]
        a, u, c = t.RawScore.to_numpy(), t.NESI.to_numpy(), t.BespokeBadnessScore.to_numpy()
        d = []
        for _ in range(2000):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            d.append(rho(a[ix], u[ix]) - rho(a[ix], c[ix]))
        d = np.array(d)
        n[f"DELTA.{ds}.lo"], n[f"DELTA.{ds}.hi"] = np.percentile(d, [2.5, 97.5])
        ps[ds] = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        n[f"DELTA.{ds}.p"] = ps[ds]
    adj = multipletests(list(ps.values()), method="holm")[1]
    for ds, q in zip(ps, adj):
        n[f"DELTA.{ds}.p_holm"] = q
    n["DELTA.max_p_holm"] = max(adj)


def text_numbers(ctx):
    """Numbers quoted in the text that come from script outputs or simple ratios of other checked numbers."""
    import re
    n = ctx.num
    log = (ctx.out / "logs" / "rass_positive.log").read_text()
    m = re.search(r"held-out RASS>0 segments (\d+) patients (\d+)", log)
    n["RASSPOS.excluded_segments"], n["RASSPOS.excluded_patients"] = int(m.group(1)), int(m.group(2))
    n["MORT2.death_pct"] = 100 * n["MORT2.deaths"] / n["MORT2.patients"]
    for ds in SCALES:
        if f"{ds}.test_patients" in n and f"{ds}.patients" in n:
            n[f"split.{ds}.test_pct"] = 100 * n[f"{ds}.test_patients"] / n[f"{ds}.patients"]


ALL = [
    cohorts_and_splits,
    correlations,
    delta_bootstrap,
    cams_sensitivity,
    supp_table1,
    supp_tables_2_3,
    calibration_table,
    threshold_cis,
    threshold_aurocs,
    ordinal_agreement,
    institutions_age_duration,
    table1,
    table1_scripts,
    mortality_folds,
    retest,
    supp_table11,
    sedation_range,
    propofol,
    sedation_strata,
    sedation_tables,
    revision_tables,
    ext20_percentages,
    medication_tables,
    output_tables,
    text_numbers,
]
