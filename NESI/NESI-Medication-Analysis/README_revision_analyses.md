# NESI-Medication paper analysis code



This repository contains the analysis code for the medication analysis. It is organized as two parallel pipelines:



1\. **Eleveld PK analysis** — quantifies how propofol effect-site

   concentration (Ce) relates to two measures of sedation depth (RASS and

   NESI) in the propofol-only subset of the ICU cohort.



2\. **Medication exposure summary** — extracts, classifies, and summarizes

   medication exposure across the four BDSP cohorts (I0001 GCS, I0001

   RASS, I0001 CAMS, I0002 GCS), producing Table 1 of the manuscript.



The two pipelines are independent and share no runtime state, but both

analyze patients drawn from the same source cohorts.



3\. **Sedation-stratified analysis** (added in revision) — relates NESI to RASS

   within sedation strata (agent and opioid co-infusion), and builds the

   sedation labels used for the performance and mortality analyses.



## Revision analyses: run order



The revision added analyses to Pipelines 1 and 2 and a new Pipeline 3.

Each runner calls the same scripts with command-line flags, so no settings

are edited by hand; running a script with no flags gives the primary

analysis. Add `--n-boot 2` to any runner for a quick smoke test.



| Step | Run (from the folder shown) | Produces |

|---|---|---|

| 1 | `eleveld-pk-analysis/validate_eleveld_vs_pas.py` | Eleveld implementation verification vs PAS (supplement) |

| 2 | `eleveld-pk-analysis/eleveld_run_cohort_expanded.py` | Expanded propofol cohort (opioids allowed) with Ce |

| 3 | `medication-exposure/add_sedation_windows_opiates.py` | Sedative, opioid and vasopressor exposure flags per assessment |

| 4 | `eleveld-pk-analysis/run_variance_all.py` | Propofol exposure–response: primary, sensitivity, within/between, expanded cohort |

| 5 | `sedation-strata/run_stratified_all.py` | Strata, sedation labels, stratified models and sensitivities, all tables and figures |



Step 5 reads the reports written by step 4, so run them in this order.

Approximate run times at 1000 bootstrap resamples: step 4, several hours;

step 5, several hours (three model runs). Steps 1–3 take minutes.



## Repository structure



    NESI-Medication_Analysis/

    ├── README.md                                 (this file)

    ├── eleveld-pk-analysis/                      (Pipeline 1; paths from medication-exposure/med_config.py)

    │   ├── (analysis scripts at top level)

    │   ├── data/

    │   └── outputs/

    ├── medication-exposure/                      (Pipeline 2)

    │   ├── med_config.py

    │   ├── (analysis scripts at top level)

    │   └── data/

    └── sedation-strata/                          (Pipeline 3, revision)

        ├── (analysis scripts; paths from medication-exposure/med_config.py)

        └── data/                                 (full-cohort NESI file, from BDSP)



## Configuration



All three pipelines read paths from one file, `medication-exposure/med_config.py`

(copy `med_config_template.py`, fill in the TODOs, and save it as

`med_config.py`). Scripts in `eleveld-pk-analysis/` and `sedation-strata/`

find it automatically.



## Requirements



Python 3.10+ recommended.



Common packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `duckdb`,

`matplotlib`, `python-docx`, `pyarrow`, `joblib`.



Install with:



    pip install pandas numpy scipy statsmodels duckdb matplotlib python-docx pyarrow joblib



For the Eleveld verification (`validate_eleveld_vs_pas.py`) only:



    pip install control

    pip install python-anesthesia-simulator --no-deps



Figures use the DejaVu Sans font (bundled with matplotlib). To edit the SVG

figures in Illustrator, install DejaVuSans.ttf and DejaVuSans-Oblique.ttf

from matplotlib's `mpl-data/fonts/ttf` folder.



The `python-docx` package is imported as `docx` — do NOT install the

unrelated `docx` package (Python 2 only).



Additional requirements for the medication exposure pipeline:



\- AWS CLI configured with credentials for the BDSP S3 bucket (only for

  Stage 0 S3 extraction; not required for downstream analyses on shipped data)

\- \~80 GB free disk for I0001 extraction intermediate files

\- 32 GB RAM recommended



\---



# Pipeline 1: Eleveld PK analysis



Analyzes the relationship between Eleveld-predicted propofol effect-site

concentration (Ce) and two measures of sedation depth:



\- **RASS** (Richmond Agitation-Sedation Scale) — clinical assessment

\- **NESI** — EEG-derived encephalopathy/sedation index



Uses mixed-effects models with patient as a random intercept for variance

decomposition, with patient-clustered bootstrap confidence intervals.



## What's in `eleveld-pk-analysis/`



    eleveld-pk-analysis/

    ├── eleveld_propofol.py                            (Eleveld 2018 PK model implementation)

    ├── eleveld_run_cohort_expanded.py                 (runs PK simulation across the cohort; both cohort flags)

    ├── variance_decomposition_eleveld_ce_vs_pump.py   (variance components; see revision additions)

    ├── data/                                          (de-identified inputs)

    └── outputs/                                       (populated by running scripts)



## Data



All files in `eleveld-pk-analysis/data/` are de-identified. Patient

identifiers (`BDSPPatientID`) and timestamps have been stripped of links

to real individuals.



\- `wt_ht_per_hosp_FULL_with_eleveld_expanded.csv` — one row per RASS observation; includes demographics, cohort flags (`propofol_only_recomputed` for the propofol-only cohort, `propofol_no_other_sedative` for the expanded cohort with opioids allowed), NESI, RASS, Eleveld-predicted Cp/Ce columns

\- `sedative_exposures/propofol_intervals.csv` — propofol infusion intervals

\- `sedative_exposures/propofol_boluses.csv` — propofol bolus events. Extracted in May 2026 with an earlier version of the bolus extraction (before boluses were split by route into `_iv` / `_non_iv` files in Pipeline 2); shipped as a fixed input, so Ce values reproduce exactly.

\- `eleveld_timeseries_expanded.parquet` — per-patient Cp/Ce and infusion-rate time series (\~1 GB; required for the pump-rate analysis)



## How to run



All scripts read paths from `medication-exposure/med_config.py`; fill it in

first (see Pipeline 2, "Before you run anything"). Then, from

`eleveld-pk-analysis/`:



    python eleveld_run_cohort_expanded.py    # only to regenerate Cp/Ce

    python run_variance_all.py               # variance analyses, primary and sensitivity



See "Revision additions to Pipeline 1" below for what each step produces.



## Key analytic choices



\- **Effect-site (Ce) vs plasma (Cp)**: All analyses use Ce. Cp doesn't account for the blood-brain equilibration delay that produces hysteresis between concentration and clinical effect.

\- **Intersection cohort**: The variance decomposition restricts to rows where Ce, NESI, and RASS (in \[−5, 0]) are all simultaneously available, so NESI and RASS are analyzed on the same observations.

\- **Cohort restriction**: Rows flagged `propofol_only=True` (no concurrent sedatives or opioids within ±12h of the EEG) are used.

\- **Patient clustering**: All inferential statistics use patient-clustered bootstrap (1000 iterations).

\- **RASS restrictions**: Positive RASS values (agitation, +1 to +4) are excluded; agitation in this population is typically driven by delirium, withdrawal, or under-sedation rather than direct propofol response.



## The Eleveld model



> Eleveld DJ, Colin P, Absalom AR, Struys MMRF. Pharmacokinetic-pharmacodynamic

> model for propofol for broad application in anaesthesia and sedation.

> *British Journal of Anaesthesia*. 2018;120(5):942-959.

> https://doi.org/10.1016/j.bja.2018.01.018



The implementation in `eleveld_propofol.py` is hand-rolled and was validated against the SimTIVA reference values.



## Reproducibility scope



The shipped `wt_ht_per_hosp_FULL_with_eleveld_expanded.csv` is the canonical starting point. To regenerate it:



\- `eleveld_run_cohort_expanded.py` regenerates Cp/Ce across the cohort and writes the expanded CSV (both cohort flags) and the time-series parquet.

\- Cohort assembly and de-identification from raw EHR data is described in the manuscript methods.





## Revision additions to Pipeline 1



Like the rest of Pipeline 1, these scripts read paths from

`medication-exposure/med_config.py` (keys `OUTPUT_CSV`, `OUTPUT_PARQUET`,

`I0001_ASM_ANTIPSYCH_DIR`, `ELEVELD_VALIDATION_DIR`).



    eleveld-pk-analysis/

    ├── validate_eleveld_vs_pas.py                     (implementation verification vs PAS)

    ├── variance_decomposition_eleveld_ce_vs_pump.py   (updated: --exclude-intermittent, --within-between)

    ├── variance_decomposition_expanded_opioid.py      (expanded cohort, opioid terms)

    ├── propofol_benzo_count.py                        (counts of intermittent doses in the propofol cohort)

    └── run_variance_all.py                            (runs the four variance analyses in order)



**Eleveld verification.** `validate_eleveld_vs_pas.py` compares

`eleveld_propofol.py` with an independent Eleveld 2018 implementation

(python-anesthesia-simulator, PAS) across 96 scenarios: age (20, 50, 80, 90)

× sex × three body sizes × opioid flag × two dosing schedules (infusion

steps; bolus plus infusion with a second bolus). Cp and Ce are sampled every

6 s. Agreement is summarized as median performance error (MDPE, bias) and

median absolute performance error (MDAPE) with PAS as reference (Varvel

1992), plus the 95th percentile and maximum absolute error. Outputs in

`ELEVELD_VALIDATION_DIR`: `eleveld_vs_pas_report.md`,

`eleveld_vs_pas_agreement.csv`, `eleveld_vs_pas_summary.csv`,

`eleveld_vs_pas_timeseries.csv`, `eleveld_vs_pas_overlay.png`.



**Variance analyses.** `run_variance_all.py` runs, in order:



| Step | Command | Report (in `OUTPUT_PARQUET`) |

|---|---|---|

| 1 Primary (Ce and pump rate) | `variance_decomposition_eleveld_ce_vs_pump.py` | `variance_decomposition_eleveld_ce_vs_pump_v3.md` |

| 2 Excluding intermittent benzodiazepine or opioid doses | `... --exclude-intermittent` | `..._v3_no_intermittent.md` |

| 3 Within- vs between-patient Ce | `... --within-between` | `..._v3_within_between.md` |

| 4 Expanded cohort with opioids | `variance_decomposition_expanded_opioid.py` | `variance_decomposition_expanded_opioid_v2.md` |



    python run_variance_all.py               # all four, 1000 resamples each

    python run_variance_all.py --n-boot 2    # smoke test

    python run_variance_all.py --only 3      # one step



Key analytic choices (revision):



\- **Intermittent doses**: a dose is "on board" if a fast-acting route (IV,

  PCA, IM, intranasal) was given within 6 h or a slow-acting route (PO, SL,

  SQ, rectal, transdermal, buccal) within 12 h, looking backward only. Step 2

  excludes observations with an intermittent benzodiazepine dose (no

  midazolam infusion) or intermittent opioid dose (no opioid infusion) on

  board, before the ≥5 observations/patient filter.

\- **Within/between decomposition**: outcome \~ Ce_within + Ce_between +

  (1 | patient), where Ce_between is the patient's mean Ce and Ce_within the

  deviation from it. Same rows and bootstrap seed as step 1.

\- **Expanded cohort**: propofol with no other sedative infusion, opioid

  infusions allowed (`propofol_no_other_sedative`). Opioid covariate =

  `opioid_on_board`. Models: Ce only; Ce + opioid; Ce + opioid + Ce × opioid,

  with Ce centered at the cohort mean (in the interaction model the Ce slope

  is the slope with no opioid on board). Bootstrap resamples run in parallel.

\- **Variance shares** are reported descriptively for NESI and RASS; no test

  of the NESI − RASS difference is used in the paper.



\---



# Pipeline 2: Medication exposure summary



For each patient and EEG, computes binary exposure flags over a 24-hour

pre-EEG window across:



\- Maintenance antiseizure medications (16 drugs)

\- Antipsychotics (8 drugs, any route)

\- Parenteral benzodiazepines (IV/IM/IN bolus)

\- Enteral benzodiazepines (PO/SL/PR/buccal)

\- Per-drug sedative infusion exposure (propofol, midazolam, ketamine, dexmedetomidine)

\- Per-drug opiate infusion exposure (fentanyl, morphine, hydromorphone)

\- Parenteral opiate (IV/IM/IN bolus)

\- Enteral opiate (PO/SL/transdermal/PR/buccal)



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



\- **I0001 GCS metadata**: `GCS_i0001a_HarvardEEG_metadata.csv` and `GCS_i0001b_HarvardEEG_metadata.csv` (cohort is split into two parts) — columns include `BDSPPatientID`, `GCSRecordedDTS`.

\- **I0001 RASS metadata**: `RASS_i0001a_HarvardEEG_metadata.csv` and `RASS_i0001b_HarvardEEG_metadata.csv` (cohort is split into two parts) — columns include `BDSPPatientID`, `RASSRecordedDTS`.

\- **I0001 CAMS metadata**: `CAMS_i0001_HarvardEEG_metadata.csv` — columns include `BDSPPatientID`, `Snippet_StartDTS`.

\- **I0002 GCS metadata**: `GCS_i0002_HarvardEEG_metadata.csv` — columns include `BDSPPatientID`, `RecordedDTS`.



### NESI lookup CSVs (one per outcome family)



Used to filter the medication exposure tables to EEG segments that produced a NESI value (made it through the NESI model).



\- **GCS NESI lookup**: `GCS_NESI_lookup.csv` — columns: `BDSPPatientID`, `GCSRecordedDTS`, `NESI`, `Why Didnt pass through my analysis`. Covers both I0001 GCS and I0002 GCS cohorts.

\- **RASS NESI lookup**: `RASS_NESI_lookup.csv` — columns: `BDSPPatientID`, `RASSRecordedDTS`, `NESI`, `Why Didnt pass through my analysis`. Covers I0001 RASS cohort.



I0001 CAMS does not need a NESI lookup — all its EEG segments have NESI values by construction.



### S3 extraction outputs (Stage 0; produced from BDSP S3 access)



\- `COHORT_ALL_MEDS_PARQUET_NEW` (I0001 GCS+RASS combined medications)

\- `I0001_CAMS_COHORT_MEDS_PARQUET` (I0001 CAMS medications)

\- `I0002_PIVOTED_MEDS_PATCHED_PARQUET` (I0002 GCS medications, pivoted and patched)



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



    # Revision analyses (Pipelines 1 and 3, exposure windows)

    OUTPUT_CSV                          # Eleveld outputs: wt_ht_..._expanded.csv

    OUTPUT_PARQUET                      # eleveld_timeseries_expanded.parquet; variance reports

    ELEVELD_VALIDATION_DIR              # outputs of validate_eleveld_vs_pas.py

    VASOPRESSOR_EXPOSURE_DIR            # I0001 vasopressor intervals (add_sedation_windows_opiates.py)

    I0001_CAMS_VASOPRESSOR_EXPOSURE_DIR # I0001 CAMS vasopressor intervals



## How to run



### Reproduce from S3 extraction (requires BDSP S3 access)



1\. Edit `med_config_template.py` and save as `med_config.py`.

2\. For each cohort, open the corresponding pipeline runner and uncomment the `SCRIPTS_STAGE_0` section. Execute each Stage 0 script. Expect several hours per cohort.

3\. After Stage 0 completes for all cohorts:



       python run_all_pipelines_meds.py



   This runs Stage 1+ for all four cohorts and produces the final Table 1 outputs (\~30 minutes).



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



\- **Per-cohort master exposure files** (one row per EEG, with binary exposure flags):

  - `I0001_ASM_ANTIPSYCH_DIR/i0001_per_eeg_master_exposure.csv`

  - `I0001_CAMS_ASM_ANTIPSYCH_DIR/i0001_cams_per_eeg_master_exposure.csv`

  - `I0002_ASM_ANTIPSYCH_DIR/i0002_per_eeg_master_exposure.csv`



\- **Final reporting** (in `I0001_ASM_ANTIPSYCH_DIR`):

  - `exposure_summary_by_patient_final.csv`

  - `exposure_summary_by_eeg_final.csv`

  - `Table_1_medication_exposure.docx` — publication-ready Word table



\- **NESI filter audit files** (one per cohort that gets NESI-filtered):

  - `I0001_GCS_dropped_no_NESI.csv`

  - `I0002_GCS_dropped_no_NESI.csv`

  - `I0001_RASS_dropped_no_NESI.csv`

  - List of EEG segments excluded from Table 1 because no matching non-null NESI value was found in the lookup CSV.



When `build_final_exposure_tables.py` runs, it also prints a cohort flow

summary to console:



    Cohort                Metadata      +NESI    +MedData   EEGs(MedData)



These numbers are intended to be pasted into the methods section.



## Methodology notes



\- **24-hour pre-EEG window**: All exposure flags are computed over the 24 hours preceding the EEG/score timestamp.

\- **Drug-name matching**: Uses `MedicationDSC` and `MedicationDisplayNM` for I0001 (Epic-style EHR). For I0002 (different EHR), falls back to `ProductDescription`. Substring matching is case-insensitive.

\- **Route classification**: Boluses are bucketed as parenteral (IV/IM/IN) or enteral (PO/SL/PR/buccal/transdermal) based on `MedicationRouteDSC` plus product-form keywords for null-route rows.

\- **Infusion reconstruction**: A state machine converts MAR action codes (started, stopped, rate-changed) into continuous infusion intervals.

\- **Midazolam handling**: I0001 midazolam appears as a sedative infusion. For the benzodiazepine bolus category, only midazolam boluses given OUTSIDE active midazolam infusions are counted (avoids double-counting).

\- **Propofol bolus**: Patients with propofol bolus exposure (without infusion) are lumped with infusion-exposed patients in the Propofol row of Table 1 (small numbers; clinically equivalent CNS exposure within the 24-hour window).

\- **NESI filtering**: Final exposure tables are filtered to EEG segments that produced a NESI value (passed through the NESI model). For I0001 GCS, I0002 GCS, and I0001 RASS, this is enforced by matching each master-exposure row against the NESI lookup CSV by `BDSPPatientID + EEG timestamp`. I0001 CAMS is exempt because all its EEG segments have NESI values by construction.

\- **I0002 infusion data**: Reliable infusion rate data are not available in the I0002 cohort's medication records. Infusion-based exposure categories (sedative infusion, opiate infusion, and per-drug subcategories) are reported as "—" rather than as counts in the I0002 GCS column of Table 1.

\- **Table 1 denominators**: The header row of each Table 1 section shows `n_with_med / n_with_NESI (pct%)` — among patients with NESI values, the proportion with medication data. Subsequent category rows show `n_exposed (pct%)` where the denominator is `n_with_med`. This distinguishes "no exposure" from "no data available."





## Revision addition to Pipeline 2: exposure windows



`add_sedation_windows_opiates.py` adds time-windowed exposure flags to the

per-EEG master exposure files (I0001 GCS+RASS and I0001 CAMS; I0002 is

skipped because it has no reliable infusion data). Windows are anchored to

the assessment time t and look backward only:



\- `now`: infusion running at t (zero-rate intervals not counted); `6h`, `12h`, `24h`: any exposure in \[t − window, t]

\- Infusions: sedatives (propofol, midazolam, dexmedetomidine, ketamine), opioids (fentanyl, hydromorphone, morphine), vasopressors (phenylephrine; other vasopressors; any vasopressor)

\- Intermittent doses: opioid and benzodiazepine fast-acting (IV, PCA, IM, intranasal) and slow-acting (PO, SL, SQ, rectal, transdermal, buccal) doses, 6/12/24 h

\- `opioid_on_board`, `benzo_on_board`: infusion running at t, OR fast-acting dose in the prior 6 h, OR slow-acting dose in the prior 12 h



Writes `<master>_windows_opiates.csv` next to each master file, e.g.

`I0001_ASM_ANTIPSYCH_DIR/i0001_per_eeg_master_exposure_windows_opiates.csv`.

The 24 h columns are checked against the existing 24 h exposure columns and

the agreement is printed. Read-only on inputs; run after

`run_all_pipelines_meds.py`.



    python add_sedation_windows_opiates.py



## Limitations



\- **Missing MAR data**: A subset of patients in each cohort had no medication administration records. They receive NA (not 0) in exposure columns and are excluded from the denominators in Table 1.

\- **ICANS cohort excluded**: The ICANS cohort had >50% missing MAR records and was excluded from medication exposure analysis. ICANS is described elsewhere in the paper.

\- **I0002 infusion data**: As noted above, infusion-rate data are not extractable from I0002 medication records.





\---



# Pipeline 3: Sedation-stratified analysis (revision)



Relates NESI to RASS within sedation strata, to show how the NESI–RASS

relationship behaves under each sedative and opioid combination, and

writes the sedation labels used for the performance-by-sedation and

mortality analyses.



## What's in `sedation-strata/`



    sedation-strata/

    ├── build_strata.py              (NESI + exposure flags -> stratum for each RASS assessment)

    ├── build_strata_labels.py       (sedation labels for every RASS, GCS and CAM-S assessment)

    ├── agent_stratified_model.py    (mixed models, cluster bootstrap, reports and figures)

    ├── obs_distribution.py          (observations per patient; used to set the cap K)

    ├── build_tables.py              (main-text propofol table + Supplementary Tables S1–S6)

    ├── md_to_html.py                (HTML copies of the model reports)

    ├── run_stratified_all.py        (runs everything in order)

    └── data/                        (full-cohort NESI file; download from BDSP)



Paths come from `medication-exposure/med_config.py` (`OUTPUT_CSV`,

`OUTPUT_PARQUET`, `I0001_ASM_ANTIPSYCH_DIR`). Before the first run, set

`NESI_CSV` at the top of `build_strata.py` to the full-cohort NESI file.

Outputs go to `I0001_ASM_ANTIPSYCH_DIR/stratified_feasibility/`.



Inputs:



\- **Full-cohort NESI file** (`NESI_CSV`; place in `sedation-strata/data/`) — one row per RASS assessment per

  EEG session, with `BDSPPatientID`, `RASSRecordedDTS`, `NESI` and

  `SessionID`. Same NESI model output as `RASS_NESI_lookup.csv` (Pipeline 2),

  but with one row per session, so RASS times covered by two overlapping EEG

  files can be identified; `build_strata.py` drops those with discordant

  values. `RASS_NESI_lookup.csv` is used only to filter Table 1.

\- **Exposure flags** — `i0001_per_eeg_master_exposure_windows_opiates.csv`

  from `add_sedation_windows_opiates.py` (Pipeline 2).



## How to run



    python run_stratified_all.py               # everything, 1000 resamples per model run

    python run_stratified_all.py --n-boot 2    # smoke test

    python run_stratified_all.py --only 6 7    # rebuild tables and HTML only



| Step | Command | Output |

|---|---|---|

| 1 | `build_strata.py` | `strata_analysis_rows.csv` (one row per RASS assessment, with stratum) |

| 2 | `build_strata_labels.py` | `labels/strata_labels_{rass,gcs,cams}.csv`, label counts |

| 3 | `agent_stratified_model.py` | primary models: `model/` (results, reports, figures) |

| 4 | `agent_stratified_model.py --cap p95` | cap sensitivity: `model/cap_p95/` |

| 5 | `agent_stratified_model.py --strata strict` | strict-rule sensitivity: `model/*_stratum_strict_*` |

| 6 | `build_tables.py` | `tables/nesi_tables.docx` and `tables/csv/` |

| 7 | `md_to_html.py` | HTML copies of each `report_*.md` |



Other flags for `agent_stratified_model.py`: `--plots-only` redraws all

figures from saved results without fitting models.



## Strata definitions (set from counts only, before any NESI modeling)



\- **Exposure at the assessment time**: infusion running at t; intermittent

  doses on board by the 6 h / 12 h rule above. No washout.

\- **Benzodiazepines**: midazolam, lorazepam, diazepam and others; clonazepam

  and clobazam excluded as antiseizure medications. **Opioids**: fentanyl,

  hydromorphone, morphine.

\- **Primary rule**: opioids classified by infusion only; assessments with an

  intermittent benzodiazepine dose on board (no midazolam infusion) are

  excluded. **Strict rule**: intermittent benzodiazepine or opioid doses on

  board exclude.

\- **Reference**: no sedative or opioid infusion and no intermittent

  benzodiazepine dose.

\- **Strata** are mutually exclusive: each agent alone and with an opioid

  infusion, midazolam + propofol, midazolam + dexmedetomidine (each ± opioid),

  ketamine (± other sedatives), opioid only. Other combinations of concurrent

  sedative infusions (mainly propofol with dexmedetomidine) are excluded.

\- **Inclusion rule** (pre-specified): ≥100 observations from ≥20 patients

  within the RASS range analyzed. Per-RASS-level estimates: ≥30 observations

  from ≥10 patients.

\- **Overlapping EEG files**: RASS assessments covered by two EEG files with

  discordant NESI values are dropped.



## Models



\- Mixed models with a patient random intercept; RASS centered at −2.

  Additive model: offset of each stratum vs the reference at matched RASS.

  Interaction model: within-stratum NESI–RASS slope and its difference from

  the reference slope.

\- **Pooled groups** (main text): each agent pooled with and without an

  opioid infusion — propofol, dexmedetomidine, midazolam (each ± opioid),

  plus opioid only. The same five groups are used for the

  performance-by-sedation analysis.

\- **Vasopressor control**: offset and slope of vasopressor-exposed vs

  unexposed assessments within the reference group.

\- Offsets at each RASS level are fit separately within each level (no

  linearity assumption).

\- Ranges: RASS −5 to 0 (primary) and −2 to 0 (light sedation).

\- 1000-resample cluster bootstrap over patients; 95% percentile CIs; fixed

  seed (42).

\- **Cap sensitivity**: at most K observations per patient per stratum,

  K = 95th percentile of observations per patient in the reference group,

  sampled within RASS levels in proportion to the patient's RASS mix.



## Sedation labels for other analyses



`build_strata_labels.py` writes one file per assessment type (RASS, GCS,

CAM-S) with patient ID, assessment time, score and one 0/1 column per

group (the five performance groups first), plus "Included under strict

rule" and "Vasopressor". These are joined to NESI and outcome data on

patient ID + assessment time for the performance-by-sedation and

mortality analyses.



\---



# Citation



\[Author list and citation to be added upon publication.]



# Contact



Questions about the code or data structure: caeckhardt.

