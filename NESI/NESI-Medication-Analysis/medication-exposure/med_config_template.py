# med_config.py
"""
Centralized paths and constants for the NESI-YAMA medication exposure pipeline.

This is a TEMPLATE. Before running any pipeline scripts, fill in all paths
marked with TODO. THEN SAVE THIS AS med_config.py for it to work with the other scripts. 
The directory structure can be whatever works for your
environment — these variables are just centralized references.

Conventions:
  - Use raw strings (r"...") for Windows paths to avoid escape issues.
  - Path objects are preferred over strings for portability.
  - Per-part directories should be on a fast disk (NVMe/SSD) if possible.
  - DUCKDB_TEMP_DIR should be on a drive with at least 50GB free.
"""

from pathlib import Path


# ─── Shared / system ──────────────────────────────────────────────────────────

# DuckDB spill-to-disk directory. Used heavily for the S3 extraction step.
# Put on a fast drive with plenty of free space (50+ GB).
DUCKDB_TEMP_DIR        = Path(r"TODO_path_to_duckdb_temp_directory")
I0002_EXTRACT_TEMP_DIR = Path(r"TODO_path/i0002/extract_temp")

# Download cache for S3 part files during I0001 extraction. Needs ~50GB free.
EXTRACT_TEMP_DIR       = Path(r"TODO_path/i0001/extract_temp")


# ─── I0001 (shared by GCS + RASS sub-cohorts) ────────────────────────────────

# S3 source of I0001 medication data
I0001_MEDS_S3_BASE = "s3://TODO_bucket/TODO_path/to/i0001_medications"

# I0001 GCS EEG metadata: per-EEG rows with BDSPPatientID and GCSRecordedDTS
I0001_GCS_METADATA_CSVS = [
    Path(r"TODO_path/GCS_i0001a_HarvardEEG_metadata.csv"),
    Path(r"TODO_path/GCS_i0001b_HarvardEEG_metadata.csv"),
]

# I0001 RASS EEG metadata: per-EEG rows with BDSPPatientID and RASSRecordedDTS
I0001_RASS_METADATA_CSVS = [
    Path(r"TODO_path/RASS_i0001a_HarvardEEG_metadata.csv"),
    Path(r"TODO_path/RASS_i0001b_HarvardEEG_metadata.csv"),
]

# Per-part filtered medication files (intermediate output of Stage 0)
EXTRACT_PER_PART_DIR = Path(r"TODO_path/i0001/per_part")

# Output of I0001 S3 extraction (Stage 0): single parquet, cohort-filtered
# medications for GCS + RASS patients
COHORT_ALL_MEDS_PARQUET_NEW = Path(r"TODO_path/i0001_cohort_all_medications.parquet")

# Output directories for I0001 downstream processing
BOLUS_EXPOSURE_DIR      = Path(r"TODO_path/i0001/bolus_exposure")
SEDATIVE_EXPOSURE_DIR   = Path(r"TODO_path/i0001/sedative_exposure")
OPIATE_EXPOSURE_DIR     = Path(r"TODO_path/i0001/opiate_exposure")
I0001_ASM_ANTIPSYCH_DIR = Path(r"TODO_path/i0001/asm_antipsych_exposure")


# ─── I0001 CAMS sub-cohort ───────────────────────────────────────────────────

# CAMS EEG metadata: per-EEG rows with BDSPPatientID and Snippet_StartDTS
I0001_CAMS_METADATA_CSVS = [
    Path(r"TODO_path/CAMS_i0001_HarvardEEG_metadata.csv"),
]

# CAMS uses same I0001 S3 source as GCS/RASS — see I0001_MEDS_S3_BASE above

# Per-part filtered files (output of Stage 0 extraction)
I0001_CAMS_EXTRACT_PER_PART_DIR = Path(r"TODO_path/i0001_cams/per_part")

# Combined cohort medications parquet (output of Stage 0 combine step)
I0001_CAMS_COHORT_MEDS_PARQUET = Path(r"TODO_path/i0001_cams/cohort_all_medications.parquet")

# Output directories for CAMS downstream processing
I0001_CAMS_BOLUS_EXPOSURE_DIR    = Path(r"TODO_path/i0001_cams/bolus_exposure")
I0001_CAMS_SEDATIVE_EXPOSURE_DIR = Path(r"TODO_path/i0001_cams/sedative_exposure")
I0001_CAMS_OPIATE_EXPOSURE_DIR   = Path(r"TODO_path/i0001_cams/opiate_exposure")
I0001_CAMS_ASM_ANTIPSYCH_DIR     = Path(r"TODO_path/i0001_cams/asm_antipsych_exposure")


# ─── I0002 GCS ───────────────────────────────────────────────────────────────

# I0002 GCS EEG metadata: per-EEG rows with BDSPPatientID and RecordedDTS
I0002_GCS_METADATA_CSV = Path(r"TODO_path/GCS_i0002_HarvardEEG_metadata.csv")

# I0002 medication source data (mar_details, mar_nax) on AWS
# I0002 uses different S3 paths than I0001 — see the I0002 extract scripts
# for the specific S3 keys; cohort-filtered outputs live below
I0002_COHORT_MEDS_DIR = Path(r"TODO_path/I0002/I0002_AWS")

# Per-part filtered medication files (intermediate output of Stage 0)
I0002_EXTRACT_PER_PART_DIR = Path(r"TODO_path/I0002/per_part")

# Pivoted EAV (wide format) of I0002 medication data
I0002_PIVOTED_MEDS_PARQUET = Path(r"TODO_path/I0002/i0002_pivoted_meds.parquet")

# Pivoted parquet with null MARActionDSC values patched
I0002_PIVOTED_MEDS_PATCHED_PARQUET = Path(r"TODO_path/I0002/i0002_pivoted_meds_patched.parquet")

# I0002 ICD-10 parquet directory
I0002_ICD10_DIR = Path(r"TODO_path/I0002/I0002_AWS/I0002_icd10")

# Output directories for I0002 downstream processing
I0002_BOLUS_EXPOSURE_DIR = Path(r"TODO_path/I0002/bolus_exposure")
I0002_ASM_ANTIPSYCH_DIR  = Path(r"TODO_path/I0002/asm_antipsych_exposure")

I0002_TABLES_TO_EXTRACT = [
    "mar_nax_2025_parquet",
    "mar_details_nax_2025_parquet",
    "mar_ext_nax_2025_parquet",
    "mar_product_nax_2025_parquet",
    "mar_product_details_nax_2025_parquet",
    "medication_nax_2024_parquet",
    "poe_order_med_nax_2025_parquet",
    "poe_order_nax_2025_parquet",
    "poe_order_details_nax_2025_parquet",
    "inpt_paml_meds_nax_2025_parquet",
]


# ─── NESI lookup CSVs ────────────────────────────────────────────────────────
# Used to filter exposure tables to EEG segments that produced a NESI value
# (made it through the NESI model). GCS lookup covers both I0001 GCS and
# I0002 GCS cohorts. RASS lookup covers I0001 RASS cohort.

GCS_NESI_LOOKUP_CSV  = Path(r"TODO_path/GCS_NESI_lookup.csv")
RASS_NESI_LOOKUP_CSV = Path(r"TODO_path/RASS_NESI_lookup.csv")


# ─── General output ──────────────────────────────────────────────────────────

OUTPUT_DIR = Path(r"TODO_path/medications_output")


# ─── Validation ──────────────────────────────────────────────────────────────

def _verify_paths():
    """Optional helper: warn about any TODO paths that haven't been filled in."""
    import warnings
    todos = []
    for name, value in list(globals().items()):
        if name.startswith("_") or name.isupper() is False:
            continue
        if isinstance(value, Path) and "TODO" in str(value):
            todos.append(name)
        elif isinstance(value, list) and any(
            isinstance(v, Path) and "TODO" in str(v) for v in value
        ):
            todos.append(name)
        elif isinstance(value, str) and "TODO" in value:
            todos.append(name)
    if todos:
        warnings.warn(
            f"med_config.py has {len(todos)} TODO paths not yet filled in: "
            + ", ".join(todos)
        )

# Uncomment to enable validation warning at import:
# _verify_paths()