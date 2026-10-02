from pathlib import Path
import pandas as pd

current = Path(__file__).resolve()
CONTPRED_ROOT = None

for parent in current.parents:
    if parent.name == "ExternalValidation":
        CONTPRED_ROOT = parent
        break

if CONTPRED_ROOT is None:
    raise RuntimeError("ExternalValidation folder not found.")


RASS_results_path = CONTPRED_ROOT / "Products" / "RASSPredictions"
GCS_results_path = CONTPRED_ROOT / "Products" / "GCSPredictions"
CAMS_results_path = CONTPRED_ROOT / "Products" / "CAMSPredictions"

input_path = RASS_results_path

# Get only CSV filenames from input_path
file_names = sorted([f.name for f in input_path.glob("*.csv")])
df_results = pd.DataFrame({"File name": file_names})
df_results["SubjectID"] = df_results["File name"].str.split("_").str[0]
df_results["seg_start_rel_s"] = df_results["File name"].str.extract(r"_ErikaSegment_([^_]+)_predictions")[0]


def get_mode(csv_path, column):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if column not in df.columns:
        print(f"Column '{column}' not found in: {csv_path}")
        return None

    mode = df[column].dropna().mode()
    return mode.iloc[0] if len(mode) > 0 else None

# ------------------------ Gathering Pedicted Class for RASS and GCS and CAMS-SF -----------------------------

df_results["PredictedRASS"] = [
    get_mode(RASS_results_path / filename, "RASSMappingClass")
    for filename in df_results["File name"]
]

df_results["PredictedGCS"] = [
    get_mode(GCS_results_path / filename, "GCSMappingClass")
    for filename in df_results["File name"]
]

df_results["PredictedCAMS"] = [
    get_mode(CAMS_results_path / filename, "CAMSMappingClass")
    for filename in df_results["File name"]
]

# ------------------------ Gathering TRue Annotations for RASS and GCS -----------------------------
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

pd.set_option("display.max_rows", None)
print(df_results.to_string(index=False))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, accuracy_score

# ---------------- RASS ----------------
rass_df = df_results[["TrueRASS", "PredictedRASS"]].dropna().copy()
rass_df["TrueRASS"] = rass_df["TrueRASS"].astype(int)
rass_df["PredictedRASS"] = rass_df["PredictedRASS"].astype(int)

rass_labels = [-5, -4, -3, -2, -1, 0]
cm_rass = confusion_matrix(rass_df["TrueRASS"], rass_df["PredictedRASS"], labels=rass_labels)

# RASS traditional/exact accuracy
rass_acc = accuracy_score(rass_df["TrueRASS"], rass_df["PredictedRASS"])
# RASS ±1-level accuracy
rass_one_level_acc = np.mean(
    np.abs(rass_df["TrueRASS"].values - rass_df["PredictedRASS"].values) <= 1
)
# RASS MAE
rass_mae = mean_absolute_error(rass_df["TrueRASS"], rass_df["PredictedRASS"])

# ---------------- GCS ----------------
gcs_df = df_results[["TrueGCSGrouped", "PredictedGCS"]].dropna().copy()

gcs_labels = ["Severe", "Moderate", "Mild"]
cm_gcs = confusion_matrix(
    gcs_df["TrueGCSGrouped"],
    gcs_df["PredictedGCS"],
    labels=gcs_labels
)
# GCS traditional/exact accuracy
gcs_acc = accuracy_score(gcs_df["TrueGCSGrouped"], gcs_df["PredictedGCS"])

print('*'*80)
print(f"RASS Prediction Accuracy: {rass_acc*100:.2f}%")
print(f"RASS ±1-Level Accuracy: {rass_one_level_acc*100:.2f}%")
print(f"RASS MAE: {rass_mae:.3f}")
print('*'*80)
print(f"GCS Prediction Accuracy: {gcs_acc*100:.2f}%")

# ---------------- Plot ----------------
plt.figure(figsize=(12, 5), dpi=100)

ax1 = plt.subplot(121)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_rass, display_labels=rass_labels)
disp1.plot(ax=ax1, cmap="Blues", colorbar=False)
ax1.set_title("RASS Prediction")
ax1.set_xlabel("Predicted RASS")
ax1.set_ylabel("True RASS")

ax2 = plt.subplot(122)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_gcs, display_labels=gcs_labels)
disp2.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("GCS Prediction")
ax2.set_xlabel("Predicted GCS")
ax2.set_ylabel("True GCS")

plt.tight_layout()
plt.show()

csv_save_path = CONTPRED_ROOT / "Final_Results_Prediction_RASS_GCS_CAMS_Cohort_Model.csv"
df_results.to_csv(csv_save_path, index = False)
