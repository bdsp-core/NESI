"""
CONTINUOUS NESI vs RASS/GCS ANALYSIS
====================================

PROCESS:
1. Each *_predictions.csv file corresponds to one Erika EEG segment (>10 min).
2. Previously, all continuous NESI predictions within that CSV were collapsed
   into ONE value by taking their median.
3. In this analysis, NO median aggregation is performed.
4. Every individual NESI prediction from each prediction CSV is retained.
5. SubjectID and seg_start_rel_s are extracted from the prediction filename.
6. The corresponding clinical RASS and GCS scores are obtained from
   KIMCHI_FINAL_METADATA.xlsx by matching:
       SubjectID + seg_start_rel_s
7. Because one clinical assessment corresponds to the entire Erika segment,
   the same RASS/GCS score is replicated for every NESI prediction belonging
   to that segment.

Example:
    One segment prediction file contains:
        NESI = [0.21, 0.25, 0.31, 0.34]

    Metadata for that segment:
        RASS = -4
        GCS  = 7

    Expanded dataframe:
        NESI    RASS    GCS
        0.21     -4      7
        0.25     -4      7
        0.31     -4      7
        0.34     -4      7

8. The boxplots therefore show the distribution of ALL continuous NESI
   predictions associated with each clinical score, rather than one median
   NESI value per EEG segment.

9. Spearman correlation is calculated between NESI and transformed clinical
   severity:
       RASS severity = -RASS
       GCS severity  = 15-GCS
   Therefore, a positive rho means that NESI increases with greater
   neurological severity.

10. Since multiple NESI predictions from the same EEG segment/patient are
    correlated and are NOT independent observations, the 95% CI for Spearman
    rho is calculated using PATIENT-LEVEL CLUSTERED BOOTSTRAP resampling.
    Patients are sampled with replacement, and ALL observations belonging to
    each sampled patient are retained together.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

current=Path(__file__).resolve()
CONTPRED_ROOT=None

for parent in current.parents:
    if parent.name=="ExternalValidation":
        CONTPRED_ROOT=parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("ExternalValidation folder not found.")

NESI_results_path=CONTPRED_ROOT/"Products"/"NESIPredictions"
eeg_metadata_path=CONTPRED_ROOT/"KIMCHI_FINAL_METADATA.xlsx"

# =============================================================================
# 1. READ ALL CONTINUOUS NESI PREDICTIONS
# =============================================================================

all_nesi_rows=[]

file_names=sorted(NESI_results_path.glob("*.csv"))

print(f"Total NESI prediction files: {len(file_names)}")

for i,csv_path in enumerate(file_names,1):
    filename=csv_path.name
    subject_id=filename.split("_")[0]

    try:
        seg_start=float(
            filename.split("_ErikaSegment_")[1].split("_predictions")[0]
        )
    except Exception:
        print(f"Could not extract segment start from: {filename}")
        continue

    temp=pd.read_csv(csv_path)

    if "NESI" not in temp.columns:
        print(f"NESI column not found: {filename}")
        continue

    nesi=pd.to_numeric(temp["NESI"],errors="coerce").dropna()

    if len(nesi)==0:
        continue

    temp_out=pd.DataFrame({
        "File name":filename,
        "SubjectID":str(subject_id).strip(),
        "seg_start_rel_s":seg_start,
        "NESI_Window_Index":np.arange(len(nesi)),
        "NESI":nesi.values
    })

    all_nesi_rows.append(temp_out)

    if i%100==0 or i==len(file_names):
        print(
            f"Processed {i}/{len(file_names)} files | "
            f"Current file NESI predictions: {len(nesi)}"
        )

if len(all_nesi_rows)==0:
    raise RuntimeError("No valid NESI predictions were found.")

df_results=pd.concat(all_nesi_rows,ignore_index=True)

print("\nExpanded continuous NESI dataframe:")
print(f"Total NESI observations: {len(df_results)}")
print(f"Unique subjects: {df_results['SubjectID'].nunique()}")
print(f"Unique EEG segments: {df_results[['SubjectID','seg_start_rel_s']].drop_duplicates().shape[0]}")

# =============================================================================
# 2. READ TRUE RASS/GCS METADATA
# =============================================================================

df_eegmetadata=pd.read_excel(eeg_metadata_path)

df_results["SubjectID"]=df_results["SubjectID"].astype(str).str.strip()
df_eegmetadata["Filename"]=df_eegmetadata["Filename"].astype(str).str.strip()

df_results["seg_start_rel_s"]=pd.to_numeric(
    df_results["seg_start_rel_s"],errors="coerce"
)

df_eegmetadata["seg_start_rel_s"]=pd.to_numeric(
    df_eegmetadata["seg_start_rel_s"],errors="coerce"
)

df_true=df_eegmetadata[
    [
        "Filename",
        "seg_start_rel_s",
        "RASS_numeric_closest",
        "GCS_closest"
    ]
].copy()

df_true=df_true.rename(columns={
    "Filename":"SubjectID",
    "RASS_numeric_closest":"TrueRASS",
    "GCS_closest":"TrueGCS"
})

# Prevent accidental many-to-many expansion if metadata contains duplicate keys
duplicate_metadata=df_true.duplicated(
    subset=["SubjectID","seg_start_rel_s"],
    keep=False
)

if duplicate_metadata.any():
    print(
        f"\nWARNING: {duplicate_metadata.sum()} metadata rows have duplicate "
        f"SubjectID + seg_start_rel_s keys."
    )

    df_true=df_true.drop_duplicates(
        subset=["SubjectID","seg_start_rel_s"],
        keep="first"
    )

# =============================================================================
# 3. MATCH EACH NESI PREDICTION TO THE SEGMENT'S TRUE RASS/GCS
# =============================================================================

df_results=df_results.merge(
    df_true,
    on=["SubjectID","seg_start_rel_s"],
    how="left",
    validate="many_to_one"
)

print("\nAfter metadata matching:")
print(f"Total NESI observations: {len(df_results)}")
print(f"NESI observations with RASS: {df_results['TrueRASS'].notna().sum()}")
print(f"NESI observations with GCS: {df_results['TrueGCS'].notna().sum()}")

# =============================================================================
# 4. CREATE CLINICAL SEVERITY VARIABLES
# =============================================================================

def group_gcs(gcs):
    if pd.isna(gcs):
        return None
    if 3<=gcs<=8:
        return "Severe"
    elif 9<=gcs<=12:
        return "Moderate"
    elif 13<=gcs<=15:
        return "Mild"
    return None

df_results["TrueGCSGrouped"]=df_results["TrueGCS"].apply(group_gcs)

df_results["TransformedTrueRASS"]=-pd.to_numeric(
    df_results["TrueRASS"],errors="coerce"
)

df_results["TransformedTrueGCS"]=15-pd.to_numeric(
    df_results["TrueGCS"],errors="coerce"
)

# =============================================================================
# 5. PATIENT-LEVEL CLUSTERED BOOTSTRAP FOR SPEARMAN 95% CI
# =============================================================================

def spearman_cluster_bootstrap_ci(
    df,
    patient_col,
    x_col,
    y_col,
    n_boot=1000,
    ci=95,
    seed=42
):
    d=df[[patient_col,x_col,y_col]].copy()

    d[x_col]=pd.to_numeric(d[x_col],errors="coerce")
    d[y_col]=pd.to_numeric(d[y_col],errors="coerce")
    d=d.dropna()

    rho=spearmanr(
        d[x_col].values,
        d[y_col].values
    ).statistic

    patients=d[patient_col].unique()
    rng=np.random.default_rng(seed)
    boot=[]

    patient_data={
        pid:d[d[patient_col]==pid][[x_col,y_col]].copy()
        for pid in patients
    }

    for _ in range(n_boot):
        sampled_patients=rng.choice(
            patients,
            size=len(patients),
            replace=True
        )

        x_boot=[]
        y_boot=[]

        for pid in sampled_patients:
            temp=patient_data[pid]
            x_boot.extend(temp[x_col].values)
            y_boot.extend(temp[y_col].values)

        x_boot=np.asarray(x_boot,dtype=float)
        y_boot=np.asarray(y_boot,dtype=float)

        if len(np.unique(x_boot))>1 and len(np.unique(y_boot))>1:
            r=spearmanr(x_boot,y_boot).statistic

            if np.isfinite(r):
                boot.append(r)

    alpha=(100-ci)/2
    lo,hi=np.percentile(boot,[alpha,100-alpha])

    return rho,lo,hi

# =============================================================================
# 6. BOXPLOT FUNCTION
# =============================================================================

def draw_boxplot(
    ax,
    df,
    patient_col,
    x_col,
    corr_col,
    y_col,
    categories,
    color,
    title,
    xlabel,
    ylabel=None,
    ylim=None,
    n_boot=1000,
    seed=42
):
    d=df[[patient_col,x_col,corr_col,y_col]].copy()

    d[x_col]=pd.to_numeric(d[x_col],errors="coerce")
    d[corr_col]=pd.to_numeric(d[corr_col],errors="coerce")
    d[y_col]=pd.to_numeric(d[y_col],errors="coerce")

    d=d.dropna()
    d=d[d[x_col].isin(categories)]

    data=[
        d.loc[d[x_col]==c,y_col].values
        for c in categories
    ]

    counts=[
        len(v)
        for v in data
    ]

    positions=np.arange(len(categories))

    ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        boxprops=dict(
            facecolor=color,
            edgecolor="black",
            linewidth=1.2
        ),
        medianprops=dict(
            color="black",
            linewidth=1.5
        ),
        whiskerprops=dict(
            color="black",
            linewidth=1.1
        ),
        capprops=dict(
            color="black",
            linewidth=1.1
        ),
        flierprops=dict(
            marker="o",
            markersize=2.2,
            markerfacecolor="lightgray",
            markeredgecolor="none",
            alpha=0.65
        )
    )

    rho,lo,hi=spearman_cluster_bootstrap_ci(
        d,
        patient_col=patient_col,
        x_col=corr_col,
        y_col=y_col,
        n_boot=n_boot,
        seed=seed
    )

    ax.set_title(
        f"{title}\nρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]",
        fontsize=14,
        fontweight="bold",
        pad=5
    )

    ax.set_xlabel(
        xlabel,
        fontsize=13
    )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=13
        )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        [str(c) for c in categories],
        fontsize=11
    )

    ax.tick_params(
        axis="y",
        labelsize=11
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.30
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ymin,ymax=ax.get_ylim()
    yrange=ymax-ymin

    ax.set_ylim(
        ymin-0.09*yrange,
        ymax
    )

    for pos,n in zip(positions,counts):
        ax.text(
            pos,
            ymin-0.025*yrange,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=9,
            rotation=40
        )

    return rho,lo,hi,counts

# =============================================================================
# 7. PREPARE RASS/GCS DATA
# =============================================================================

df_rass=df_results.dropna(
    subset=[
        "TrueRASS",
        "TransformedTrueRASS",
        "NESI"
    ]
).copy()

df_gcs=df_results.dropna(
    subset=[
        "TrueGCS",
        "TransformedTrueGCS",
        "NESI"
    ]
).copy()

rass_categories=[-5,-4,-3,-2,-1,0]
gcs_categories=list(range(3,16))

print("\nRASS analysis:")
print(f"NESI observations: {len(df_rass)}")
print(f"Unique patients: {df_rass['SubjectID'].nunique()}")
print(f"Unique segments: {df_rass[['SubjectID','seg_start_rel_s']].drop_duplicates().shape[0]}")

print("\nGCS analysis:")
print(f"NESI observations: {len(df_gcs)}")
print(f"Unique patients: {df_gcs['SubjectID'].nunique()}")
print(f"Unique segments: {df_gcs[['SubjectID','seg_start_rel_s']].drop_duplicates().shape[0]}")

# =============================================================================
# 8. PLOT ALL CONTINUOUS NESI PREDICTIONS
# =============================================================================

fig,axes=plt.subplots(
    1,2,
    figsize=(15,5.5),
    dpi=150
)

rass_stats=draw_boxplot(
    axes[0],
    df_rass,
    patient_col="SubjectID",
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

gcs_stats=draw_boxplot(
    axes[1],
    df_gcs,
    patient_col="SubjectID",
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

fig_save_path=(
    CONTPRED_ROOT/
    "Figure2_Recreate_using_NESI"/
    "Fig2_External_Validation_AllContinuousNESI.png"
)

plt.savefig(
    fig_save_path,
    dpi=500,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# 9. PRINT FINAL STATISTICS
# =============================================================================

print(
    f"\nRASS: ρ={rass_stats[0]:.3f}, "
    f"95% CI [{rass_stats[1]:.3f}, {rass_stats[2]:.3f}], "
    f"NESI observations={sum(rass_stats[3])}, "
    f"patients={df_rass['SubjectID'].nunique()}"
)

print(
    f"GCS: ρ={gcs_stats[0]:.3f}, "
    f"95% CI [{gcs_stats[1]:.3f}, {gcs_stats[2]:.3f}], "
    f"NESI observations={sum(gcs_stats[3])}, "
    f"patients={df_gcs['SubjectID'].nunique()}"
)

print(f"\nFigure saved to:\n{fig_save_path}")