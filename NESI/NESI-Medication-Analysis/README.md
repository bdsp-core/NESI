# NESI-Medication paper analysis code

This repository contains the analysis code for the medication analysis. It is organized as two parallel pipelines:

1. **Eleveld PK analysis** — quantifies how propofol effect-site
   concentration (Ce) relates to two measures of sedation depth (RASS and
   NESI) in the propofol-only subset of the ICU cohort.

2. **Medication exposure summary** — extracts, classifies, and summarizes
   medication exposure across the four BDSP cohorts (I0001 GCS, I0001
   RASS, I0001 CAMS, I0002 GCS), producing Table 1 of the manuscript.

The two pipelines are independent and share no runtime state, but both
analyze patients drawn from the same source cohorts.

## Repository structure

    NESI-Medication_Analysis/
    ├── README.md                                 (this file)
    ├── eleveld-pk-analysis/                      (Pipeline 1)
    │   ├── path_configs.py
    │   ├── (analysis scripts at top level)
    │   ├── data/
    │   └── outputs/
    └── medication-exposure/                      (Pipeline 2)
        ├── med_config.py
        ├── (analysis scripts at top level)
        └── data/

## Requirements

Python 3.10+ recommended.

Common packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `duckdb`,
`matplotlib`, `python-docx`, `pyarrow`.

Install with:

    pip install pandas numpy scipy statsmodels duckdb matplotlib python-docx pyarrow

The `python-docx` package is imported as `docx` — do NOT install the
unrelated `docx` package (Python 2 only).

Additional requirements for the medication exposure pipeline:

- AWS CLI configured with credentials for the BDSP S3 bucket (only for
  Stage 0 S3 extraction; not required for downstream analyses on shipped data)
- ~80 GB free disk for I0001 extraction intermediate files
- 32 GB RAM recommended

---

# Pipeline 1: Eleveld PK analysis

Analyzes the relationship between Eleveld-predicted propofol effect-site
concentration (Ce) and two measures of sedation depth:

- **RASS** (Richmond Agitation-Sedation Scale) — clinical assessment
- **NESI** — EEG-derived encephalopathy/sedation index

Uses Spearman correlations with paired patient-clustered bootstrap, and
mixed-effects models with patient as a random intercept for variance
decomposition.

## What's in `eleveld-pk-analysis/`

    eleveld-pk-analysis/
    ├── path_configs.py                                (path resolution; no edits needed)
    ├── eleveld_propofol.py                            (Eleveld 2018 PK model implementation)
    ├── eleveld_run_cohort.py                          (runs PK simulation across the cohort)
    ├── eleveld_cohort_sanity_checks.py                (validates Cp/Ce values behave correctly)
    ├── plot_eleveld_one_patient.py                    (per-patient overview plots)
    ├── nesi_vs_rass_correlation_comparison.py         (headline correlation numbers)
    ├── variance_decomposition_eleveld_ce_vs_pump.py   (headline variance components)
    ├── data/                                          (de-identified inputs)
    └── outputs/                                       (populated by running scripts)

## Data

All files in `eleveld-pk-analysis/data/` are de-identified. Patient
identifiers (`BDSPPatientID`) and timestamps have been stripped of links
to real individuals.

- `wt_ht_per_hosp_FULL_with_eleveld.csv` — one row per RASS observation; includes demographics, propofol-only flag, NESI, RASS, Eleveld-predicted Cp/Ce columns
- `sedative_exposures/propofol_intervals.csv` — propofol infusion intervals
- `sedative_exposures/propofol_boluses.csv` — propofol bolus events
- `eleveld_timeseries.parquet` — per-patient Cp/Ce time series (~1 GB; required for sanity checks and per-patient plots)

## How to run

From `eleveld-pk-analysis/`:

    python path_configs.py    # verify paths

Then run the headline-results scripts:

    python nesi_vs_rass_correlation_comparison.py
    python variance_decomposition_eleveld_ce_vs_pump.py

The variance decomposition takes ~2 hours due to the paired bootstrap. The
correlation comparison runs in a few minutes. Sanity checks and per-patient
plots can be run separately and are fast.

## Key analytic choices

- **Effect-site (Ce) vs plasma (Cp)**: All analyses use Ce. Cp doesn't account for the blood-brain equilibration delay that produces hysteresis between concentration and clinical effect.
- **Intersection cohort for headline numbers**: The Spearman comparison and variance decomposition both restrict to rows where Ce, NESI, and RASS (in [−5, 0]) are all simultaneously available. This enables direct apples-to-apples comparison of the two outcomes.
- **Cohort restriction**: Rows flagged `propofol_only=True` (no concurrent sedatives or opioids within ±12h of the EEG) are used.
- **Patient clustering**: All inferential statistics use patient-clustered bootstrap (1000 iterations).
- **RASS restrictions**: Positive RASS values (agitation, +1 to +4) are excluded; agitation in this population is typically driven by delirium, withdrawal, or under-sedation rather than direct propofol response.

## The Eleveld model

> Eleveld DJ, Colin P, Absalom AR, Struys MMRF. Pharmacokinetic-pharmacodynamic
> model for propofol for broad application in anaesthesia and sedation.
> *British Journal of Anaesthesia*. 2018;120(5):942-959.
> https://doi.org/10.1016/j.bja.2018.01.018

The implementation in `eleveld_propofol.py` is hand-rolled and was validated against the SimTIVA reference values.

## Reproducibility scope

The shipped `wt_ht_per_hosp_FULL_with_eleveld.csv` is the canonical starting point. To regenerate it:

- `eleveld_run_cohort.py` reads the shipped wt_ht CSV (demographics/timestamp columns only), regenerates Cp/Ce, and writes a parallel `_REGENERATED` file. ~30 minutes.
- Cohort assembly and de-identification from raw EHR data is described in the manuscript methods.

---

# Pipeline 2: Medication exposure summary

For each patient and EEG, computes binary exposure flags over a 24-hour
pre-EEG window across:

- Maintenance antiseizure medications (16 drugs)
- Antipsychotics (8 drugs, any route)
- Parenteral benzodiazepines (IV/IM/IN bolus)
- Enteral benzodiazepines (PO/SL/PR/buccal)
- Per-drug sedative infusion exposure (propofol, midazolam, ketamine, dexmedetomidine)
- Per-drug opiate infusion exposure (fentanyl, morphine, hydromorphone)
- Parenteral opiate (IV/IM/IN bolus)
- Enteral opiate (PO/SL/transdermal/PR/buccal)

Final outputs are summary CSVs and a Word table
(`Table_1_medication_exposure.docx`).

## Before you run anything

This pipeline reads all paths from `med_config.py`. The file is shipped as 
a **template** with `TODO` placeholders throughout. Before running any 
script, open `med_config.py`, replace every `TODO_path/...` entry with the 
actual path on your system, and save. Scripts will fail with import or 
file-not-found errors until this is done.

The `_verify_paths()` helper at the bottom of the template will flag any 
remaining TODOs if uncommented at import — useful as a sanity check while 
you're filling things in.

## Reproducibility scope

This pipeline supports two workflows:

**Full reproduction from S3.** Requires BDSP S3 access. Run the Stage 0 
extraction scripts, then the downstream pipelines. All intermediate and 
final outputs are regenerated.

**Final-stage reproduction only.** Uses the shipped per-EEG master exposure 
CSVs (and the metadata + NESI lookup CSVs) to regenerate Table 1 directly, 
without re-running the upstream extraction. This is the path most users 
will take.

In either case, `med_config.py` defines where each input and output lives. 
Edit it to point at your local directory structure. If you're following 
the final-stage-only workflow with the shipped data, point the relevant 
paths at the corresponding files in `medication-exposure/data/`.

## What's in `medication-exposure/`

    medication-exposure/
    ├── med_config.py                                   (paths and constants)
    ├── drug_configs.py                                 (sedative drug patterns)
    ├── opiate_configs.py                               (opiate drug patterns)
    ├── benzo_bolus_configs.py
    ├── i0002_benzo_bolus_configs.py
    ├── i0002_asm_antipsych_configs.py
    ├── infusion_reconstruction.py                      (bolus + infusion logic)
    ├── bolus_route_classify.py                         (IV / non-IV bucketing)
    │
    ├── # I0001 GCS+RASS pipeline scripts
    ├── extract_meds_for_cohort_v2.py                   (S3 extraction; Stage 0)
    ├── combine_per_part_meds.py
    ├── build_sedative_exposure.py
    ├── build_opiate_exposure.py
    ├── build_bolus_exposure.py
    ├── extract_i0001_asm_antipsych_administrations.py
    ├── build_i0001_asm_antipsych_exposure.py
    ├── build_i0001_master_exposure.py
    │
    ├── # I0002 GCS pipeline scripts
    ├── extract_i0002_meds.py
    ├── extract_i0002_meds_v2_fast.py
    ├── build_i0002_pivoted_meds.py
    ├── patch_i0002_pivoted_meds.py
    ├── build_i0002_benzo_boluses.py
    ├── build_i0002_propofol_boluses.py
    ├── build_i0002_opiate_boluses.py
    ├── extract_i0002_asm_antipsych_administrations.py
    ├── build_i0002_asm_antipsych_exposure.py
    ├── build_i0002_master_exposure.py
    │
    ├── # I0001 CAMS pipeline scripts
    ├── extract_meds_for_cams_cohort.py
    ├── combine_cams_per_part_meds.py
    ├── build_cams_sedative_exposure.py
    ├── build_cams_opiate_exposure.py
    ├── build_cams_bolus_exposure.py
    ├── extract_cams_asm_antipsych_administrations.py
    ├── build_cams_asm_antipsych_exposure.py
    ├── build_cams_master_exposure.py
    │
    ├── # Pipeline runners
    ├── run_i0001_gcs_rass_pipeline_meds.py
    ├── run_cams_pipeline_meds.py
    ├── run_i0002_pipeline_meds.py
    ├── run_all_pipelines_meds.py
    │
    ├── # Final reporting
    ├── build_final_exposure_tables.py                  (summary CSVs)
    ├── build_table_1_docx.py                           (Word Table 1)
    │
    └── data/                                           (inputs)

## Required input files (shipped in `medication-exposure/data/`)

For **Workflow 1**, you need the metadata CSVs and the NESI lookup CSVs 
(shipped). Cohort medication parquets are generated by Stage 0 from S3.

For **Workflow 2**, you need the metadata CSVs, the NESI lookup CSVs, 
**and** the per-EEG master exposure CSVs — all shipped in 
`medication-exposure/data/`.


### Cohort metadata CSVs 

Each metadata CSV has one row per EEG segment.

- **I0001 GCS metadata**: `GCS_i0001a_HarvardEEG_metadata.csv` and `GCS_i0001b_HarvardEEG_metadata.csv` (cohort is split into two parts) — columns include `BDSPPatientID`, `GCSRecordedDTS`.
- **I0001 RASS metadata**: `RASS_i0001a_HarvardEEG_metadata.csv` and `RASS_i0001b_HarvardEEG_metadata.csv` (cohort is split into two parts) — columns include `BDSPPatientID`, `RASSRecordedDTS`.
- **I0001 CAMS metadata**: `CAMS_i0001_HarvardEEG_metadata.csv` — columns include `BDSPPatientID`, `Snippet_StartDTS`.
- **I0002 GCS metadata**: `GCS_i0002_HarvardEEG_metadata.csv` — columns include `BDSPPatientID`, `RecordedDTS`.

### NESI lookup CSVs (one per outcome family)

Used to filter the medication exposure tables to EEG segments that produced a NESI value (made it through the NESI model).

- **GCS NESI lookup**: `GCS_NESI_lookup.csv` — columns: `BDSPPatientID`, `GCSRecordedDTS`, `NESI`, `Why Didnt pass through my analysis`. Covers both I0001 GCS and I0002 GCS cohorts.
- **RASS NESI lookup**: `RASS_NESI_lookup.csv` — columns: `BDSPPatientID`, `RASSRecordedDTS`, `NESI`, `Why Didnt pass through my analysis`. Covers I0001 RASS cohort.

I0001 CAMS does not need a NESI lookup — all its EEG segments have NESI values by construction.

### S3 extraction outputs (Stage 0; produced from BDSP S3 access)

- `COHORT_ALL_MEDS_PARQUET_NEW` (I0001 GCS+RASS combined medications)
- `I0001_CAMS_COHORT_MEDS_PARQUET` (I0001 CAMS medications)
- `I0002_PIVOTED_MEDS_PATCHED_PARQUET` (I0002 GCS medications, pivoted and patched)

These are the medication administration records used by the downstream
pipelines. Stage 0 (the S3 extraction that produces them) requires BDSP S3
access and takes several hours per cohort.

## Configuration

All paths are centralized in `med_config.py`. Before running anything, edit
that file to point at your local directories. Required keys:

    # I0001 (GCS + RASS share the same cohort medications parquet)
    I0001_MEDS_S3_BASE                  # e.g. "s3://bucket/path"
    I0001_GCS_METADATA_CSVS             # list of CSV paths
    I0001_RASS_METADATA_CSVS            # list of CSV paths
    COHORT_ALL_MEDS_PARQUET_NEW         # output of S3 extraction
    BOLUS_EXPOSURE_DIR
    SEDATIVE_EXPOSURE_DIR
    OPIATE_EXPOSURE_DIR
    I0001_ASM_ANTIPSYCH_DIR

    # I0001 CAMS
    I0001_CAMS_METADATA_CSVS
    I0001_CAMS_EXTRACT_PER_PART_DIR
    I0001_CAMS_COHORT_MEDS_PARQUET
    I0001_CAMS_BOLUS_EXPOSURE_DIR
    I0001_CAMS_SEDATIVE_EXPOSURE_DIR
    I0001_CAMS_OPIATE_EXPOSURE_DIR
    I0001_CAMS_ASM_ANTIPSYCH_DIR

    # I0002 GCS
    I0002_GCS_METADATA_CSV
    I0002_COHORT_MEDS_DIR
    I0002_PIVOTED_MEDS_PARQUET
    I0002_PIVOTED_MEDS_PATCHED_PARQUET
    I0002_BOLUS_EXPOSURE_DIR
    I0002_ASM_ANTIPSYCH_DIR

    # NESI lookup CSVs (used by build_final_exposure_tables.py)
    GCS_NESI_LOOKUP_CSV
    RASS_NESI_LOOKUP_CSV

    # Shared
    DUCKDB_TEMP_DIR

## How to run

### Reproduce from S3 extraction (requires BDSP S3 access)

1. Edit `med_config_template.py` and save as `med_config.py`.
2. For each cohort, open the corresponding pipeline runner and uncomment the `SCRIPTS_STAGE_0` section. Execute each Stage 0 script. Expect several hours per cohort.
3. After Stage 0 completes for all cohorts:

       python run_all_pipelines_meds.py

   This runs Stage 1+ for all four cohorts and produces the final Table 1 outputs (~30 minutes).

### Reproduce from shipped cohort medication parquets (skip Stage 0)

If you have the cohort medication parquets and metadata CSVs in place:

    python run_all_pipelines_meds.py

### Run a single cohort

    python run_i0001_gcs_rass_pipeline_meds.py
    python run_cams_pipeline_meds.py
    python run_i0002_pipeline_meds.py

### Just regenerate the final tables

If the per-EEG master exposure files already exist:

    python build_final_exposure_tables.py
    python build_table_1_docx.py

The first script reads the per-cohort master files plus the NESI lookup
CSVs, applies the NESI filter, and produces the summary CSVs. The second
reads the summary CSVs and produces the Word table.

## Outputs

After `run_all_pipelines_meds.py`:

- **Per-cohort master exposure files** (one row per EEG, with binary exposure flags):
  - `I0001_ASM_ANTIPSYCH_DIR/i0001_per_eeg_master_exposure.csv`
  - `I0001_CAMS_ASM_ANTIPSYCH_DIR/i0001_cams_per_eeg_master_exposure.csv`
  - `I0002_ASM_ANTIPSYCH_DIR/i0002_per_eeg_master_exposure.csv`

- **Final reporting** (in `I0001_ASM_ANTIPSYCH_DIR`):
  - `exposure_summary_by_patient_final.csv`
  - `exposure_summary_by_eeg_final.csv`
  - `Table_1_medication_exposure.docx` — publication-ready Word table

- **NESI filter audit files** (one per cohort that gets NESI-filtered):
  - `I0001_GCS_dropped_no_NESI.csv`
  - `I0002_GCS_dropped_no_NESI.csv`
  - `I0001_RASS_dropped_no_NESI.csv`
  - List of EEG segments excluded from Table 1 because no matching non-null NESI value was found in the lookup CSV.

When `build_final_exposure_tables.py` runs, it also prints a cohort flow
summary to console:

    Cohort                Metadata      +NESI    +MedData   EEGs(MedData)

These numbers are intended to be pasted into the methods section.

## Methodology notes

- **24-hour pre-EEG window**: All exposure flags are computed over the 24 hours preceding the EEG/score timestamp.
- **Drug-name matching**: Uses `MedicationDSC` and `MedicationDisplayNM` for I0001 (Epic-style EHR). For I0002 (different EHR), falls back to `ProductDescription`. Substring matching is case-insensitive.
- **Route classification**: Boluses are bucketed as parenteral (IV/IM/IN) or enteral (PO/SL/PR/buccal/transdermal) based on `MedicationRouteDSC` plus product-form keywords for null-route rows.
- **Infusion reconstruction**: A state machine converts MAR action codes (started, stopped, rate-changed) into continuous infusion intervals.
- **Midazolam handling**: I0001 midazolam appears as a sedative infusion. For the benzodiazepine bolus category, only midazolam boluses given OUTSIDE active midazolam infusions are counted (avoids double-counting).
- **Propofol bolus**: Patients with propofol bolus exposure (without infusion) are lumped with infusion-exposed patients in the Propofol row of Table 1 (small numbers; clinically equivalent CNS exposure within the 24-hour window).
- **NESI filtering**: Final exposure tables are filtered to EEG segments that produced a NESI value (passed through the NESI model). For I0001 GCS, I0002 GCS, and I0001 RASS, this is enforced by matching each master-exposure row against the NESI lookup CSV by `BDSPPatientID + EEG timestamp`. I0001 CAMS is exempt because all its EEG segments have NESI values by construction.
- **I0002 infusion data**: Reliable infusion rate data are not available in the I0002 cohort's medication records. Infusion-based exposure categories (sedative infusion, opiate infusion, and per-drug subcategories) are reported as "—" rather than as counts in the I0002 GCS column of Table 1.
- **Table 1 denominators**: The header row of each Table 1 section shows `n_with_med / n_with_NESI (pct%)` — among patients with NESI values, the proportion with medication data. Subsequent category rows show `n_exposed (pct%)` where the denominator is `n_with_med`. This distinguishes "no exposure" from "no data available."

## Limitations

- **Missing MAR data**: A subset of patients in each cohort had no medication administration records. They receive NA (not 0) in exposure columns and are excluded from the denominators in Table 1.
- **ICANS cohort excluded**: The ICANS cohort had >50% missing MAR records and was excluded from medication exposure analysis. ICANS is described elsewhere in the paper.
- **I0002 infusion data**: As noted above, infusion-rate data are not extractable from I0002 medication records.

---

# Citation

[Author list and citation to be added upon publication.]

# Contact

Questions about the code or data structure: caeckhardt.
