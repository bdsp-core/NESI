from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path("/home/ayush/Desktop/NESI_review_material/PacMAP")
IN_CSV = SCRIPT_DIR / "NESI_pacmap_coords.csv"
INDICATION_DIR = Path("/home/ayush/Desktop/NESI_review_material/EEGIndication_Analysis/NeuroIndication_Datasetwise")

INDICATION_FILES = {
    "RASS": "RASS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "GCS": "GCS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "CAMS": "CAMS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv",
    "ICANS": "ICANS_NESI_fullmetadata_with_NeuroReports_Qwen3_V2.csv"
}

def normalize_filename(series):
    return series.astype("string").str.strip().str.replace(r"(?i)\.csv$", "", regex=True)

def attach_indication_metadata(pacmap_csv=IN_CSV, indication_dir=INDICATION_DIR):
    df_pacmap = pd.read_csv(pacmap_csv, low_memory=False)
    original_count = len(df_pacmap)
    df_pacmap["_match_dataset"] = df_pacmap["Dataset"].astype("string").str.strip().str.upper()
    df_pacmap["_match_filename"] = normalize_filename(df_pacmap["MorgothOutputFilename"])

    indication_frames = []

    for dataset, filename in INDICATION_FILES.items():
        df_ind = pd.read_csv(Path(indication_dir) / filename, low_memory=False)

        if "MorgothOutputFilename" in df_ind.columns:
            filename_col = "MorgothOutputFilename"
        elif "Filename" in df_ind.columns:
            filename_col = "Filename"
        else:
            raise ValueError(f"{dataset}: No filename column found.")

        if "IndicationNeuroreport_LLM" not in df_ind.columns:
            raise ValueError(f"{dataset}: IndicationNeuroreport_LLM column not found.")

        df_ind = df_ind[[filename_col, "IndicationNeuroreport_LLM"]].copy()
        df_ind["_match_filename"] = normalize_filename(df_ind[filename_col])
        df_ind["_match_dataset"] = dataset
        df_ind = df_ind.dropna(subset=["_match_filename"])
        df_ind = df_ind[df_ind["_match_filename"] != ""]

        duplicates = df_ind.duplicated(["_match_dataset", "_match_filename"], keep=False)
        if duplicates.any():
            print(f"WARNING: {dataset} has {duplicates.sum():,} duplicate filename rows. Keeping first occurrence.")
            df_ind = df_ind.drop_duplicates(["_match_dataset", "_match_filename"], keep="first")

        indication_frames.append(df_ind[["_match_dataset", "_match_filename", "IndicationNeuroreport_LLM"]])
        print(f"{dataset}: {len(df_ind):,} unique filenames | Matching column: {filename_col}")

    df_indications = pd.concat(indication_frames, ignore_index=True)

    df_pacmap_enriched = df_pacmap.merge(
        df_indications,
        on=["_match_dataset", "_match_filename"],
        how="left",
        sort=False,
        validate="many_to_one"
    )

    if len(df_pacmap_enriched) != original_count:
        raise ValueError("Row count changed after merging!")

    df_pacmap_enriched = df_pacmap_enriched.drop(columns=["_match_dataset", "_match_filename"])

    print("\n========== MATCHING SUMMARY ==========")
    summary = df_pacmap_enriched.groupby("Dataset").agg(
        Total=("Dataset", "size"),
        Matched=("IndicationNeuroreport_LLM", "count")
    )
    summary["Unmatched"] = summary["Total"] - summary["Matched"]
    summary["Match_Percentage"] = (100 * summary["Matched"] / summary["Total"]).round(2)
    print(summary.to_string())

    print(f"\nOriginal PaCMAP rows: {original_count:,}")
    print(f"Enriched PaCMAP rows: {len(df_pacmap_enriched):,}")
    print(f"Total non-null indications: {df_pacmap_enriched['IndicationNeuroreport_LLM'].notna().sum():,}")
    print("\nAdded column: IndicationNeuroreport_LLM")

    return df_pacmap_enriched


df_pacmap_enriched = attach_indication_metadata()


from pathlib import Path
import numpy as np
import pandas as pd

DIAG_ROOT = Path("/home/ayush/Desktop/NESI_review_material")
diag_files = [
    DIAG_ROOT / "DiagnosisMetadtafiles" / "File1.xlsx",
    DIAG_ROOT / "DiagnosisMetadtafiles" / "File2.xlsx",
    DIAG_ROOT / "DiagnosisMetadtafiles" / "File3.xlsx"
]

def pct(n, denom):
    return round(100.0*n/denom, 2) if denom else 0.0

def norm_id(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "")
    if s.endswith(".0"): s = s[:-2]
    return s if s and s.lower() not in ["nan","none","null","<na>"] else np.nan

def extract_pid(filename):
    if pd.isna(filename): return np.nan
    filename = str(filename).strip().replace("\\","/").split("/")[-1]
    part = filename.split("_")[0]
    if part.startswith("sub-"):
        pid_full = part.split("-")[1]
        pid = pid_full[5:]
    else:
        pid = part
    return norm_id(pid)

def norm_cols(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+"," ",regex=True).str.lower()
    return df

def map_dx(cat):
    if pd.isna(cat): return np.nan
    c = str(cat).strip().lower()
    if any(k in c for k in [
        "intracerebral","ich","hemorrhage","ischemic","stroke","subdural","sdh",
        "traumatic","tbi","brain tumor","tumor",
        "seizure","sz","dement","neuroinfection","neuroinflamm","neuro"
    ]): return "Neurological"
    if any(k in c for k in [
        "cardiac arrest","arrest","endocarditis","cardiac","circulatory"
    ]): return "Cardiac/Circulatory"
    if any(k in c for k in [
        "sepsis","toxic","metabolic","tme","encephalopathy",
        "renal failure","renal","liver failure","hepatic","liver"
    ]): return "Systemic/Metabolic"
    if any(k in c for k in ["respiratory","pneumonia"]):
        return "Respiratory"
    if any(k in c for k in [
        "altered mental status","ams","psychiatric","other","unknown","unspecified","na","n/a"
    ]): return "Unknown/Other"
    return "Unknown/Other"

# Load diagnosis metadata
diag_dfs = []
for f in diag_files:
    d = norm_cols(pd.read_excel(f))
    id_cols = [c for c in d.columns if "patientid" in c]
    cat_cols = [c for c in d.columns if "diagnosis" in c and "category" in c]
    if not id_cols or not cat_cols:
        raise ValueError(f"Required columns missing in {f.name}: {list(d.columns)}")
    d = d.rename(columns={id_cols[0]:"bdsppatientid",cat_cols[0]:"diagnosis category"})
    d = d[["bdsppatientid","diagnosis category"]].copy()
    d["SourceFile"] = f.name
    diag_dfs.append(d)
    print(f"Loaded {f.name}: {len(d):,} rows")

diagnosis = pd.concat(diag_dfs,ignore_index=True)
diagnosis["BDSPPatientID"] = diagnosis["bdsppatientid"].apply(norm_id)
diagnosis["diagnosis category"] = diagnosis["diagnosis category"].astype("string").str.strip().str.lower()
diagnosis["diagnosis category"] = diagnosis["diagnosis category"].replace(
    {"":pd.NA,"nan":pd.NA,"none":pd.NA,"null":pd.NA}
)
diagnosis["DxGroup"] = diagnosis["diagnosis category"].apply(map_dx)

# Diagnosis data quality
missing_id = diagnosis["BDSPPatientID"].isna()
missing_dx = diagnosis["diagnosis category"].isna()
valid = diagnosis.loc[~missing_id & ~missing_dx].copy()

print("\n========== DIAGNOSIS DATA QUALITY ==========")
print(f"Total diagnosis records: {len(diagnosis):,}")
print(f"Missing patient ID only: {(missing_id & ~missing_dx).sum():,}")
print(f"Missing diagnosis only: {(~missing_id & missing_dx).sum():,}")
print(f"Missing both: {(missing_id & missing_dx).sum():,}")
print(f"Valid diagnosis records: {len(valid):,}")
print(f"Unique patients with valid diagnosis: {valid['BDSPPatientID'].nunique():,}")
print(f"Repeated patient records: {valid.duplicated('BDSPPatientID').sum():,}")

conflicts = valid.groupby("BDSPPatientID")["diagnosis category"].nunique()
conflicting_ids = conflicts[conflicts>1].index
print(f"Patients with conflicting diagnoses: {len(conflicting_ids):,}")

if len(conflicting_ids):
    print("\nExamples of conflicting diagnoses:")
    print(valid.loc[
        valid["BDSPPatientID"].isin(conflicting_ids),
        ["BDSPPatientID","diagnosis category","SourceFile"]
    ].drop_duplicates().head(15).to_string(index=False))
    raise ValueError("Conflicting patient diagnoses found. Resolve before merging.")

# Prepare patient-level diagnosis lookup
dx = valid[["BDSPPatientID","diagnosis category","DxGroup"]].copy()
dx = dx.drop_duplicates(subset="BDSPPatientID",keep="first")
dx = dx.rename(columns={"diagnosis category":"AdmissionPrimaryDiagnosis"})

# Extract BDSPPatientID from PaCMAP filenames
df = df_pacmap_enriched.copy()
original_count = len(df)

if "MorgothOutputFilename" not in df.columns:
    raise ValueError(f"MorgothOutputFilename not found. Available columns: {list(df.columns)}")

df["BDSPPatientID"] = df["MorgothOutputFilename"].apply(extract_pid)

# Verify extraction
test_filename = "sub-S0002122422458_ses-1_eeg_10min_seg5.csv"
test_pid = extract_pid(test_filename)
assert test_pid == "122422458", f"Incorrect extraction: {test_pid}"


# Merge original and broad diagnosis categories
df = df.drop(columns=["AdmissionPrimaryDiagnosis","DxGroup"],errors="ignore")
df["_row_order"] = np.arange(original_count)
df = df.merge(dx,on="BDSPPatientID",how="left",sort=False,validate="many_to_one")
df = df.sort_values("_row_order").reset_index(drop=True)
df = df.drop(columns=["_row_order"])

if len(df) != original_count:
    raise ValueError("PaCMAP row count changed after merging!")

df_pacmap_enriched = df
matched = df_pacmap_enriched["AdmissionPrimaryDiagnosis"].notna()
df_pacmap_enriched["DxGroup"] = df_pacmap_enriched["DxGroup"].fillna("Unknown/Other")
print("\n========== DIAGNOSIS MATCHING SUMMARY ==========")
print(f"Total EEG rows: {len(df_pacmap_enriched):,}")
print(f"Matched EEG rows: {matched.sum():,}")
print(f"Unmatched EEG rows: {(~matched).sum():,}")
print(f"Match percentage: {pct(matched.sum(),len(df_pacmap_enriched))}%")

print("\n========== PATIENT-LEVEL MATCHING ==========")
patient_match = df_pacmap_enriched.dropna(subset=["BDSPPatientID"]).groupby(
    "BDSPPatientID"
)["AdmissionPrimaryDiagnosis"].apply(lambda x: x.notna().any())

print(f"Unique PaCMAP patients: {len(patient_match):,}")
print(f"Matched patients: {patient_match.sum():,}")
print(f"Unmatched patients: {(~patient_match).sum():,}")
print(f"Patient match percentage: {pct(patient_match.sum(),len(patient_match))}%")

print("\n========== BROAD DIAGNOSIS DISTRIBUTION ==========")
print(df_pacmap_enriched["DxGroup"].value_counts(dropna=False).to_string())

# print("\n========== ORIGINAL DIAGNOSIS DISTRIBUTION ==========")
# print(df_pacmap_enriched["AdmissionPrimaryDiagnosis"].value_counts(dropna=False).to_string())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from IPython.display import display

IIIC_PALETTE = {
    'Seizure':'#E13238','LPD':'#F08C2A','GPD':'#F2D549',
    'LRDA':'#AABF45','GRDA':'#7AC8E3','Other':'#5C3A87','Burst':'#000000'
}
IIIC_LEGEND_ORDER = ['Burst','Seizure','LPD','GPD','LRDA','GRDA','Other']
DATASET_NAMES = {'RASS':'RASS','GCS':'GCS','CAMS':'CAM-S','ICANS':'ICANS'}
CAT_PROB_COL = {
    'Seizure':'IIIC_Seizure',
    'LPD':'IIIC_LPD',
    'GPD':'IIIC_GPD',
    'LRDA':'IIIC_LRDA',
    'GRDA':'IIIC_GRDA',
    'Other':'IIIC_Other',
    'Burst':'Burst_vs_NoBurst'
}

def category_cmap(cat):
    base = IIIC_PALETTE[cat]
    pale = tuple(0.8 + 0.2*np.array(to_rgb(base)))
    if cat == 'Other':
        pale = tuple(0.85 + 0.15*np.array(to_rgb(base)))
        return LinearSegmentedColormap.from_list(cat,[base,pale])
    return LinearSegmentedColormap.from_list(cat,[pale,base])

def make_labels_rgb(df,burst_threshold=0.5):
    cols = ['IIIC_Seizure','IIIC_LPD','IIIC_GPD','IIIC_LRDA','IIIC_GRDA','IIIC_Other']
    names = np.array(['Seizure','LPD','GPD','LRDA','GRDA','Other'])
    labels = names[np.argmax(df[cols].to_numpy(),axis=1)].copy()
    labels[df['Burst_vs_NoBurst'].to_numpy() >= burst_threshold] = 'Burst'
    rgb = np.zeros((len(df),3))
    for cat in IIIC_PALETTE:
        mask = labels == cat
        if not mask.any(): continue
        probs = df.loc[mask,CAT_PROB_COL[cat]].to_numpy()
        lo,hi = np.quantile(probs,[0.05,0.95]) if len(probs)>20 else (probs.min(),probs.max())
        if hi-lo<1e-6: hi=lo+1e-6
        rgb[mask] = category_cmap(cat)(np.clip((probs-lo)/(hi-lo),0,1))[:,:3]
    return labels,rgb

def has_indication(value,indication):
    if pd.isna(value): return False
    return bool(re.search(re.escape(indication),str(value),flags=re.IGNORECASE))

def plot_nesi_pacmap(
    df_data, fig_save_path,
    rass_scores=(-5,),gcs_scores=(3,),cams_scores=(7,),icans_scores=(4,),
    rass_nesi_min=1.5,gcs_nesi_min=1.5,cams_nesi_min=0.0,icans_nesi_min=0.0,
    rass_color="#9467BD",gcs_color="#D62728",cams_color="#FF7F0E",icans_color="#2CA02C",
    subplot4=None,subplot5=None,subplot6=None,
    rotation_degrees=90,flip_vertical=True,jitter_copies=4,jitter_sigma_frac=0.006,
    dot_size=4.0,burst_threshold=0.5,figsize=(20,13),dpi=400,
    title_fontsize=12,legend_fontsize=7,legend_title_fontsize=8,

):
    df = df_data.copy()

    default_panels = [
        {
            'indication':'Post-Cardiac Arrest',
            'pattern':'Burst',
            'pattern_threshold':0.7,
            'nesi_min':0.0,
            'other_color':'#D3D3D3',
            'other_alpha':0.30,
            'title':'D. Post-Cardiac Arrest'
        },
        {
            'indication':'Stroke / Cerebrovascular Disease',
            'pattern':'LPD',
            'pattern_threshold':0.7,
            'nesi_min':0.25,
            'other_color':'#D3D3D3',
            'other_alpha':0.30,
            'title':'E. Stroke / Cerebrovascular Disease'
        },
        {
            'indication':'Known Epilepsy / Status Epilepticus',
            'pattern':'Seizure',
            'pattern_threshold':0.7,
            'nesi_min':0.0,
            'other_color':'#D3D3D3',
            'other_alpha':0.30,
            'title':'F. Known Epilepsy / Status Epilepticus'
        }
    ]

    user_panels = [subplot4,subplot5,subplot6]
    panels = [
        {**default_panels[i],**(user_panels[i] or {})}
        for i in range(3)
    ]

    for i,panel in enumerate(panels,4):
        if panel['pattern'] not in CAT_PROB_COL:
            raise ValueError(
                f"Subplot {i}: Invalid pattern '{panel['pattern']}'. "
                f"Choose from {list(CAT_PROB_COL)}"
            )

    xy = df[['pacmap_x_nesi','pacmap_y_nesi']].to_numpy(dtype=float)
    theta = np.deg2rad(rotation_degrees)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    xy_rot = (xy-xy.mean(axis=0)) @ R.T
    if flip_vertical: xy_rot[:,1] *= -1

    labels,rgb = make_labels_rgb(df,burst_threshold)
    rng = np.random.default_rng(0)
    sx,sy = xy_rot.std(axis=0)*jitter_sigma_frac
    xy_aug = np.vstack([
        xy_rot]+[
        xy_rot+np.column_stack([
            rng.normal(0,sx,len(df)),
            rng.normal(0,sy,len(df))
        ]) for _ in range(jitter_copies)
    ])

    rgb_aug = np.tile(rgb,(jitter_copies+1,1))
    labels_aug = np.tile(labels,jitter_copies+1)
    nesi_values = pd.to_numeric(df['NESI'],errors='coerce').to_numpy()
    nesi_aug = np.tile(nesi_values,jitter_copies+1)
    counts = pd.Series(labels).value_counts()
    datasets = df['Dataset'].astype(str).str.strip().str.upper().to_numpy()
    scores = pd.to_numeric(df['TrueRawScores'],errors='coerce').to_numpy()

    selected_scores = {
        'RASS':list(rass_scores),
        'GCS':list(gcs_scores),
        'CAMS':list(cams_scores),
        'ICANS':list(icans_scores)
    }
    nesi_thresholds = {
        'RASS':rass_nesi_min,
        'GCS':gcs_nesi_min,
        'CAMS':cams_nesi_min,
        'ICANS':icans_nesi_min
    }
    extreme_colors = {
        'RASS':rass_color,
        'GCS':gcs_color,
        'CAMS':cams_color,
        'ICANS':icans_color
    }

    extreme_labels = np.full(len(df),'Other',dtype=object)
    for dataset,values in selected_scores.items():
        mask = (
            (datasets==dataset)&
            np.isin(scores,values)&
            (nesi_values>=nesi_thresholds[dataset])
        )
        extreme_labels[mask] = dataset
    extreme_aug = np.tile(extreme_labels,jitter_copies+1)

    fig = plt.figure(figsize=figsize,dpi=dpi)
    gs = fig.add_gridspec(
        2,3,wspace=0.06,hspace=0.08,
        left=0.03,right=0.97,top=0.96,bottom=0.03
    )
    axes = [
        fig.add_subplot(gs[i//3,i%3])
        for i in range(6)
    ]
    axA,axB,axC,axD,axE,axF = axes

    # A. NESI continuous
    order = np.argsort(np.abs(nesi_aug))
    axA.scatter(
        xy_aug[order,0],xy_aug[order,1],
        c=nesi_aug[order],cmap='RdBu_r',
        vmin=-3,vmax=3,s=dot_size,
        marker='o',edgecolors='none',alpha=0.55
    )
    order = np.argsort(np.abs(nesi_aug))
    sc = axA.scatter(
        xy_aug[order,0],xy_aug[order,1],
        c=nesi_aug[order],cmap='RdBu_r',
        vmin=-3,vmax=3,s=dot_size,
        marker='o',edgecolors='none',alpha=0.55
    )

    cbar = fig.colorbar(sc, ax=axA, orientation='horizontal',
                        fraction=0.035, pad=0.02, shrink=0.65)
    cbar.set_label('NESI', fontsize=legend_title_fontsize)
    cbar.ax.tick_params(labelsize=legend_fontsize)
    cbar.set_ticks([-3,-2,-1,0,1,2,3])

    # B. Dominant IIIC category
    for cat in ['Other','GRDA','LRDA','GPD','LPD','Seizure','Burst']:
        mask = labels_aug==cat
        axB.scatter(
            xy_aug[mask,0],xy_aug[mask,1],
            c=rgb_aug[mask],s=dot_size,
            marker='o',edgecolors='none',alpha=0.55
        )

    handles = [
        mpatches.Patch(
            color=IIIC_PALETTE[cat],
            label=f'{cat} (n={int(counts.get(cat,0))})'
        ) for cat in IIIC_LEGEND_ORDER
    ]
    axB.legend(
        handles=handles,loc='upper right',
        fontsize=legend_fontsize,frameon=False,
        title='IIIC category',title_fontsize=legend_title_fontsize
    )

    # C. Severe clinical states
    mask = extreme_aug=='Other'
    axC.scatter(
        xy_aug[mask,0],xy_aug[mask,1],
        c='#D3D3D3',s=dot_size,
        marker='o',edgecolors='none',alpha=0.30
    )

    for dataset in selected_scores:
        mask = extreme_aug==dataset
        axC.scatter(
            xy_aug[mask,0],xy_aug[mask,1],
            c=extreme_colors[dataset],s=dot_size,
            marker='o',edgecolors='none',alpha=0.75
        )

    handles = []
    for dataset,values in selected_scores.items():
        if not values: continue
        score_text = ', '.join(str(v) for v in values)
        label = f'{DATASET_NAMES[dataset]} {score_text}, (n={np.sum(extreme_labels==dataset)})'
        handles.append(mpatches.Patch(color=extreme_colors[dataset],label=label))

    handles.append(
        mpatches.Patch(
            color='#D3D3D3',
            label=f'Other (n={np.sum(extreme_labels=="Other")})'
        )
    )
    axC.legend(
        handles=handles,loc='upper right',
        fontsize=legend_fontsize,frameon=False,
        title='Clinical state',title_fontsize=legend_title_fontsize
    )

    # D, E, F. Configurable clinical indication + EEG pattern
    panel_results = []

    for ax,panel in zip([axD,axE,axF],panels):
        indication_mask = df['IndicationNeuroreport_LLM'].apply(
            lambda x: has_indication(x,panel['indication'])
        ).to_numpy()

        pattern_col = CAT_PROB_COL[panel['pattern']]
        pattern_values = pd.to_numeric(
            df[pattern_col],errors='coerce'
        ).to_numpy()

        selected = (
            indication_mask&
            (pattern_values>=panel['pattern_threshold'])&
            (nesi_values>=panel['nesi_min'])
        )
        selected_aug = np.tile(selected,jitter_copies+1)

        mask_other = ~selected_aug
        ax.scatter(
            xy_aug[mask_other,0],xy_aug[mask_other,1],
            c=panel['other_color'],s=dot_size,
            marker='o',edgecolors='none',
            alpha=panel['other_alpha'],zorder=1
        )

        xy_selected = xy_aug[selected_aug]
        nesi_selected = nesi_aug[selected_aug]
        order_panel = np.argsort(np.abs(nesi_selected))

        ax.scatter(
            xy_selected[order_panel,0],
            xy_selected[order_panel,1],
            c=nesi_selected[order_panel],
            cmap='RdBu_r',vmin=-3,vmax=3,
            s=dot_size,marker='o',
            edgecolors='none',alpha=0.75,zorder=2
        )

        n_selected = int(selected.sum())
        n_other = len(df)-n_selected

        handles = [
            mpatches.Patch(
                color='#D62728',
                label=f"{panel['indication']} (n={n_selected})"
            ),
            mpatches.Patch(
                color=panel['other_color'],
                label=f'Other (n={n_other})'
            )
        ]

        ax.legend(
            handles=handles,loc='upper right',
            fontsize=legend_fontsize,frameon=False,
            title=f"{panel['pattern']} Probability ≥ {panel['pattern_threshold']:g}",
            title_fontsize=legend_title_fontsize
        )

        panel_results.append({
            'panel':panel,
            'indication_count':int(indication_mask.sum()),
            'selected_count':n_selected,
            'other_count':n_other,
            'selected_nesi':nesi_values[selected]
        })

    titles = [
        'A. NESI (continuous)',
        'B. Dominant IIIC category',
        'C. Severe clinical states'
    ] + [p['title'] for p in panels]

    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=title_fontsize,fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal',adjustable='box')
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(fig_save_path, dpi=500, bbox_inches="tight")
    display(fig)
    plt.close(fig)

    print(
        f'Input: {len(df):,} windows | '
        f'Rotation: {rotation_degrees}° CCW | '
        f'Vertical flip: {flip_vertical}'
    )

    print('\nSevere clinical states:')
    for dataset,values in selected_scores.items():
        print(
            f'{DATASET_NAMES[dataset]} {values}, '
            f'NESI ≥ {nesi_thresholds[dataset]}: '
            f'{np.sum(extreme_labels==dataset):,} windows'
        )

    for letter,result in zip(['D','E','F'],panel_results):
        panel = result['panel']
        print(f'\nClinical indication selection (subplot {letter}):')
        print(f"Indication: {panel['indication']}")
        print(f"EEG pattern: {panel['pattern']}")
        print(f"Pattern probability threshold: ≥ {panel['pattern_threshold']}")
        print(f"NESI threshold: ≥ {panel['nesi_min']}")
        print(f"Total indication rows: {result['indication_count']:,}")
        print(f"Selected EEG windows: {result['selected_count']:,}")
        print(f"Other EEG windows: {result['other_count']:,}")

        if result['selected_count']:
            selected_nesi = pd.Series(result['selected_nesi'])
            print(
                f'NESI range: {selected_nesi.min():.3f} to '
                f'{selected_nesi.max():.3f}'
            )
            print(
                f'NESI median (IQR): {selected_nesi.median():.3f} '
                f'({selected_nesi.quantile(0.25):.3f}–'
                f'{selected_nesi.quantile(0.75):.3f})'
            )

    return None


plot_nesi_pacmap(
    df_data=df_pacmap_enriched,
    fig_save_path = '/home/ayush/Desktop/NESI_review_material/PacMAP/NESI_composite_PacMAP.png',

    # Subplot C: Severe clinical states
    rass_scores=[-5],
    gcs_scores=[3],
    cams_scores=[7,6],
    icans_scores=[4,3],
    rass_nesi_min=1.5,
    gcs_nesi_min=1.5,
    cams_nesi_min=0.0,
    icans_nesi_min=-0.05,
    rass_color="#E13238",
    gcs_color="#000000",
    cams_color="#1F77B4",
    icans_color="green",

    # Subplot D: Post-cardiac arrest + burst suppression
    subplot4={
        'indication':'Post-Cardiac Arrest',
        'pattern':'Burst',
        'pattern_threshold':0.7,
        'nesi_min':0.0,
        'other_color':'#D3D3D3',
        'other_alpha':0.30,
        'title':'D. Post-Cardiac Arrest with Burst'
    },

    # Subplot E: Post-cardiac arrest + GPD 
    subplot5={
        'indication':'Post-Cardiac Arrest',
        'pattern':'GPD',
        'pattern_threshold':0.7,
        'nesi_min':0.0,
        'other_color':'#D3D3D3',
        'other_alpha':0.30,
        'title':'F. Post-Cardiac Arrest with GPD'
    },

    # Subplot F: Stroke + LPD
    subplot6={
        'indication':'Stroke / Cerebrovascular Disease',
        'pattern':'LPD',
        'pattern_threshold':0.7,
        'nesi_min':0.25,
        'other_color':'#D3D3D3',
        'other_alpha':0.30,
        'title':'E. Stroke with LPD'
    },

    # Figure settings
    figsize=(20,13),
    title_fontsize=14,
    legend_fontsize=10,
    legend_title_fontsize=10
)