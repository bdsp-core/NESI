
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path


current = Path(__file__).resolve()
NESI_ROOT = None
for parent in [current.parent, *current.parents]:
    if parent.name == "NESI":
        NESI_ROOT = parent
        break

if NESI_ROOT is None:
    raise RuntimeError("NESI root folder not found")


ROOT = Path(NESI_ROOT/"RevisionRelatedMaterials"/"EEGIndication_Analysis"/"NeuroIndication_Datasetwise")
FILES = {
    "RASS": ROOT / "RASS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "GCS": ROOT / "GCS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "CAMS": ROOT / "CAMS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "ICANS": ROOT / "ICANS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv"
}
OUT = ROOT / "NESI_Indication_Performance"
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Altered Mental Status / Encephalopathy",
    "Suspected Seizures / Spells",
    "Post-Cardiac Arrest",
    "Known Epilepsy / Status Epilepticus",
    "Stroke / Cerebrovascular Disease",
    "Other / Unknown"
]
SHORT = ["AMS / Enceph.", "Suspected Seizures", "Cardiac Arrest",
         "Epilepsy / SE", "Stroke / CVD", "Other / Unknown"]
SCALES = ["RASS", "GCS", "CAMS", "ICANS"]
COLORS = {"RASS": "#F28E2B", "GCS": "#4E79A7",
          "CAMS": "#59A14F", "ICANS": "#B07AA1"}

N_BOOT = 1000
SEED = 42
MIN_PATIENTS = 10

def load_scale(scale, path):
    df = pd.read_csv(path, low_memory=False)
    required = ["BDSPPatientID", "RawScore", "NESI", "Split", "IndicationNeuroreport_LLM"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{scale}: missing columns {missing}")

    if "Dataset" in df.columns:
        df = df[df["Dataset"].astype(str).str.strip().str.upper() == scale].copy()

    df["PatientID"] = df["BDSPPatientID"].astype("string").str.strip()
    if scale == "ICANS" and "StudyID" in df.columns:
        fallback = df["StudyID"].astype("string").str.strip()
        missing_id = df["PatientID"].isna() | df["PatientID"].eq("")
        df.loc[missing_id, "PatientID"] = "StudyID_" + fallback[missing_id]

    df["NESI"] = pd.to_numeric(df["NESI"], errors="coerce")
    df["Severity"] = pd.to_numeric(df["RawScore"], errors="coerce")
    if scale == "GCS":
        df["Severity"] = 15 - df["Severity"]
    elif scale == "RASS":
        df["Severity"] = -df["Severity"]

    df["Split"] = df["Split"].astype("string").str.strip().str.upper()
    df = df[df["Split"].isin(["TRAIN", "TEST"])].copy()
    df["SplitGroup"] = df["Split"].map({"TRAIN": "Training", "TEST": "Testing"})
    df = df.dropna(subset=["PatientID", "NESI", "Severity"]).copy()
    df = df[df["PatientID"].ne("")].copy()

    df["IndicationNeuroreport_LLM"] = (
        df["IndicationNeuroreport_LLM"].astype("string").str.strip()
        .replace(r"(?i)^(nan|none|null|na|n/a)?$", "Other / Unknown", regex=True)
        .fillna("Other / Unknown")
    )
    df["Indication"] = df["IndicationNeuroreport_LLM"].str.split(";")
    df = df.explode("Indication")
    df["Indication"] = df["Indication"].str.strip()
    df = df[df["Indication"].isin(CATEGORIES)].copy()

    id_cols = [c for c in ["PatientID", "MorgothOutputFilename", "Filename", "Indication", "Split"] if c in df.columns]
    df = df.drop_duplicates(subset=id_cols)
    return df

def patient_bootstrap(df, n_boot=N_BOOT, seed=SEED):
    n_patients = df["PatientID"].nunique()
    result = dict(Rho=np.nan, CI_Lower=np.nan, CI_Upper=np.nan,
                  N_Patients=n_patients, N_EEGs=len(df), N_Valid_Boot=0)
    if n_patients < MIN_PATIENTS or df["Severity"].nunique() < 2 or df["NESI"].nunique() < 2:
        return result

    x = df["NESI"].to_numpy(dtype=float)
    y = df["Severity"].to_numpy(dtype=float)
    rho = spearmanr(x, y).statistic
    if not np.isfinite(rho):
        return result

    patient_codes, patient_ids = pd.factorize(df["PatientID"], sort=False)
    groups = [np.flatnonzero(patient_codes == i) for i in range(len(patient_ids))]
    rng = np.random.default_rng(seed)
    boot = []

    for _ in range(n_boot):
        selected = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in selected])
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boot.append(r)

    result["Rho"] = rho
    result["N_Valid_Boot"] = len(boot)
    if len(boot) >= 0.9 * n_boot:
        result["CI_Lower"], result["CI_Upper"] = np.percentile(boot, [2.5, 97.5])
    return result

def calculate_indication_performance():
    all_data = {}
    for scale in SCALES:
        if not FILES[scale].exists():
            print(f"Skipping {scale}: {FILES[scale]} not found")
            continue
        all_data[scale] = load_scale(scale, FILES[scale])
        df = all_data[scale]
        print(f"{scale}: {len(df):,} indication memberships, {df['PatientID'].nunique():,} patients")
        print(df.groupby("SplitGroup")["PatientID"].nunique().to_string())

    results = []
    for split in ["Training", "Testing"]:
        for indication in CATEGORIES:
            for scale in SCALES:
                if scale not in all_data:
                    continue
                df = all_data[scale]
                subset = df[(df["SplitGroup"] == split) & (df["Indication"] == indication)]
                stats = patient_bootstrap(subset)
                results.append(dict(Split=split, Indication=indication, Scale=scale, **stats))
                print(f"{split:8s} | {scale:5s} | {indication[:22]:22s} | "
                      f"N={stats['N_Patients']:5d} | rho={stats['Rho']:.3f}")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "NESI_Indication_Spearman_Train_Test_Bootstrap1000.csv", index=False)
    return res

def plot_indication_performance(res, show_rho_ci=False, font_size=12,
                                annotation_font_size=8, figsize=(22, 8),
                                subplot_titles=("(a) Training", "(b) Testing")):
    with plt.rc_context({"font.size": font_size}):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        x = np.arange(len(CATEGORIES))
        width = 0.17

        for panel_idx, (ax, split) in enumerate(zip(axes, ["Training", "Testing"])):
            for j, scale in enumerate(SCALES):
                positions = x + (j - 1.5) * width
                sub = res[(res["Split"] == split) & (res["Scale"] == scale)]
                lookup = sub.set_index("Indication")

                for i, category in enumerate(CATEGORIES):
                    if category not in lookup.index:
                        continue
                    row = lookup.loc[category]
                    rho = row["Rho"]
                    if not np.isfinite(rho):
                        continue

                    ax.bar(positions[i], rho, width=width * 0.85,
                           color=COLORS[scale], edgecolor="black",
                           linewidth=1.5, zorder=3)

                    lower, upper = row["CI_Lower"], row["CI_Upper"]
                    if np.isfinite(lower) and np.isfinite(upper):
                        err = np.array([[max(0, rho-lower)], [max(0, upper-rho)]])
                        ax.errorbar(positions[i], rho, yerr=err, fmt="none",
                                    ecolor="black", capsize=2.5,
                                    elinewidth=0.9, zorder=5)

                    if show_rho_ci:
                        if np.isfinite(lower) and np.isfinite(upper):
                            label = f"{rho:.3f} [{lower:.3f}–{upper:.3f}]"
                            label_y = upper + 0.012
                        else:
                            label = f"{rho:.3f} [CI N/A]"
                            label_y = rho + 0.025
                        ax.text(positions[i], label_y, label, rotation=90,
                                ha="center", va="bottom",
                                fontsize=annotation_font_size,
                                color="black", clip_on=False, zorder=6)

            ax.set_xticks(x)
            ax.set_xticklabels(SHORT, rotation=35, ha="right", fontsize=font_size)
            ax.set_title(subplot_titles[panel_idx], fontsize=font_size + 3, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.tick_params(axis="y", labelsize=font_size)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel("Spearman correlation (ρ)", fontsize=font_size + 1)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[s],
                                 edgecolor="black", linewidth=0.5) for s in SCALES]
        fig.legend(handles, ["CAM-S" if s == "CAMS" else s for s in SCALES], loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 0.96), frameon=False, fontsize=font_size)
        fig.suptitle("NESI Performance Across EEG Monitoring Indications",
                     fontsize=font_size + 4, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.87 if show_rho_ci else 0.90])

        suffix = "WithRhoCI" if show_rho_ci else "WithoutRhoCI"
        png_path = OUT / f"NESI_Indication_Train_Test_{suffix}.png"
        pdf_path = OUT / f"NESI_Indication_Train_Test_{suffix}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"\nPNG saved: {png_path}")
        print(f"PDF saved: {pdf_path}")
        plt.show()

def summarize_indication_patients():
    summary = []

    for scale in SCALES:
        if not FILES[scale].exists():
            continue

        df = load_scale(scale, FILES[scale])

        for indication in CATEGORIES:
            d = df[df["Indication"] == indication]

            summary.append({
                "Indication": indication,
                "Scale": scale,
                "Training": d[d["SplitGroup"] == "Training"]["PatientID"].nunique(),
                "Testing": d[d["SplitGroup"] == "Testing"]["PatientID"].nunique(),
                "Train+Test": d["PatientID"].nunique()
            })

    df_counts = pd.DataFrame(summary)

    for split in ["Training", "Testing", "Train+Test"]:
        print(f"\n========== {split.upper()} : UNIQUE PATIENTS ==========")
        print(
            df_counts.pivot(
                index="Indication",
                columns="Scale",
                values=split
            ).reindex(index=CATEGORIES, columns=SCALES).to_string()
        )

    df_counts.to_csv(
        OUT / "NESI_Indication_UniquePatientCounts_Train_Test_All.csv",
        index=False
    )

    return df_counts


res = calculate_indication_performance()

plot_indication_performance(
    res,
    show_rho_ci=False,
    font_size=16,
    annotation_font_size=8,
    figsize=(22, 8),
    subplot_titles=("A. NESI Training Set", "B. NESI Testing Set")
)

print("\nPatient counts by indication and scale:")
print(res.pivot_table(
    index=["Split", "Indication"],
    columns="Scale",
    values="N_Patients",
    aggfunc="first"
).to_string())

df_counts = summarize_indication_patients()
