from pathlib import Path
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

current = Path(__file__).resolve()
CONTPRED_ROOT = None

for parent in current.parents:
    if parent.name == "ExternalValidation":
        CONTPRED_ROOT = parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("ExternalValidation folder not found.")



NESI_results_path = CONTPRED_ROOT / "Products" / "NESIPredictions"

input_path = NESI_results_path

# Get only CSV filenames from input_path
file_names = sorted([f.name for f in input_path.glob("*.csv")])
df_results = pd.DataFrame({"File name": file_names})
df_results["SubjectID"] = df_results["File name"].str.split("_").str[0]
df_results["seg_start_rel_s"] = df_results["File name"].str.extract(r"_ErikaSegment_([^_]+)_predictions")[0]


def get_median(csv_path, column):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        print(f"Column '{column}' not found in: {csv_path}")
        return None
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    return values.median() if len(values) > 0 else None

# ------------------------ Gathering Pedicted Class for RASS and GCS and CAMS-SF -----------------------------

df_results["NESI"] = [
    get_median(NESI_results_path / filename, "NESI")
    for filename in df_results["File name"]
]

# ------------------------ Gathering True Annotations for RASS and GCS -----------------------------
eeg_metadata_path = CONTPRED_ROOT / "KIMCHI_FINAL_METADATA.xlsx"
df_eegmetadata = pd.read_excel(eeg_metadata_path)

# Make sure matching columns have compatible types
df_results["SubjectID"] = df_results["SubjectID"].astype(str).str.strip()
df_eegmetadata["Filename"] = df_eegmetadata["Filename"].astype(str).str.strip()

df_results["seg_start_rel_s"] = pd.to_numeric(df_results["seg_start_rel_s"], errors="coerce")
df_eegmetadata["seg_start_rel_s"] = pd.to_numeric(df_eegmetadata["seg_start_rel_s"], errors="coerce")

# Keep only required metadata columns
df_true = df_eegmetadata[
    ["Filename", "seg_start_rel_s", "RASS_numeric_closest", "GCS_closest"]
].copy()

# Rename for matching/output
df_true = df_true.rename(columns={
    "Filename": "SubjectID",
    "RASS_numeric_closest": "TrueRASS",
    "GCS_closest": "TrueGCS"
})

# Match on BOTH SubjectID and segment start time
df_results = df_results.merge(
    df_true,
    on=["SubjectID", "seg_start_rel_s"],
    how="left"
)

def group_gcs(gcs):
    if pd.isna(gcs):
        return None
    if 3 <= gcs <= 8:
        return "Severe"
    elif 9 <= gcs <= 12:
        return "Moderate"
    elif 13 <= gcs <= 15:
        return "Mild"
    else:
        return None

df_results["TrueGCSGrouped"] = df_results["TrueGCS"].apply(group_gcs)

# Higher transformed score = greater clinical severity
df_results["TransformedTrueRASS"]=-pd.to_numeric(df_results["TrueRASS"],errors="coerce")
df_results["TransformedTrueGCS"]=15-pd.to_numeric(df_results["TrueGCS"],errors="coerce")
pd.set_option("display.max_rows", None)
print(df_results.to_string(index=False))







def spearman_bootstrap_ci(x,y,n_boot=1000,ci=95,seed=42):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    mask=np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    rho=spearmanr(x,y).statistic
    rng=np.random.default_rng(seed)
    boot=[]
    n=len(x)
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        xb,yb=x[idx],y[idx]
        if len(np.unique(xb))>1 and len(np.unique(yb))>1:
            r=spearmanr(xb,yb).statistic
            if np.isfinite(r):
                boot.append(r)
    alpha=(100-ci)/2
    lo,hi=np.percentile(boot,[alpha,100-alpha])
    return rho,lo,hi

def draw_boxplot(ax,df,x_col,corr_col,y_col,categories,color,title,xlabel,
                 ylabel=None,ylim=None,n_boot=1000,seed=42):
    d=df[[x_col,corr_col,y_col]].copy()
    d[x_col]=pd.to_numeric(d[x_col],errors="coerce")
    d[corr_col]=pd.to_numeric(d[corr_col],errors="coerce")
    d[y_col]=pd.to_numeric(d[y_col],errors="coerce")
    d=d.dropna()
    d=d[d[x_col].isin(categories)]

    data=[d.loc[d[x_col]==c,y_col].values for c in categories]
    counts=[len(v) for v in data]
    positions=np.arange(len(categories))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        boxprops=dict(facecolor=color,edgecolor="black",linewidth=1.2),
        medianprops=dict(color="black",linewidth=1.5),
        whiskerprops=dict(color="black",linewidth=1.1),
        capprops=dict(color="black",linewidth=1.1),
        flierprops=dict(
            marker="o",
            markersize=2.2,
            markerfacecolor="lightgray",
            markeredgecolor="none",
            alpha=0.65
        )
    )

    rho,lo,hi=spearman_bootstrap_ci(
        d[corr_col].values,
        d[y_col].values,
        n_boot=n_boot,
        seed=seed
    )

    ax.set_title(
        f"{title}\nρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]",
        fontsize=14,
        fontweight="bold",
        pad=5
    )

    ax.set_xlabel(xlabel,fontsize=13)

    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=13)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in categories],fontsize=11)
    ax.tick_params(axis="y",labelsize=11)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.yaxis.grid(True,linestyle="--",linewidth=0.7,alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ymin,ymax=ax.get_ylim()
    yrange=ymax-ymin
    ax.set_ylim(ymin-0.09*yrange,ymax)

    for pos,n in zip(positions,counts):
        ax.text(
            pos,
            ymin-0.025*yrange,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=9
        )

    return rho,lo,hi,counts

# Assessment-level data
df_rass=df_results.dropna(
    subset=["TrueRASS","TransformedTrueRASS","NESI"]
).copy()

df_gcs=df_results.dropna(
    subset=["TrueGCS","TransformedTrueGCS","NESI"]
).copy()

# Original clinical scores are retained on x-axis
rass_categories=[-5,-4,-3,-2,-1,0]
gcs_categories=list(range(3,16))

fig,axes=plt.subplots(1,2,figsize=(15,5.5),dpi=150)

# RASS
rass_stats=draw_boxplot(
    axes[0],
    df_rass,
    x_col="TrueRASS",
    corr_col="TransformedTrueRASS",
    y_col="NESI",
    categories=rass_categories,
    color="#7FAED0",
    title="RASS",
    xlabel="RASS score",
    ylabel="NESI",
    n_boot=1000,
    seed=42
)

# GCS — NO Y LABEL
gcs_stats=draw_boxplot(
    axes[1],
    df_gcs,
    x_col="TrueGCS",
    corr_col="TransformedTrueGCS",
    y_col="NESI",
    categories=gcs_categories,
    color="#F2B653",
    title="GCS",
    xlabel="GCS score",
    ylabel=None,
    n_boot=1000,
    seed=42
)

plt.tight_layout(w_pad=4)
fig_save_path = CONTPRED_ROOT / "Figure2_Recreate_using_NESI" / "Fig2_External_Validation.png"
plt.savefig(fig_save_path ,dpi=500,bbox_inches="tight")

plt.show()

print(
    f"RASS: ρ={rass_stats[0]:.3f}, "
    f"95% CI [{rass_stats[1]:.3f}, {rass_stats[2]:.3f}], "
    f"n={sum(rass_stats[3])}"
)

print(
    f"GCS: ρ={gcs_stats[0]:.3f}, "
    f"95% CI [{gcs_stats[1]:.3f}, {gcs_stats[2]:.3f}], "
    f"n={sum(gcs_stats[3])}"
)

