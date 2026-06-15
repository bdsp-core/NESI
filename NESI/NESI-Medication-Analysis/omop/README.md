# Medication data from the OMOP database

The medication data used in the NESI analyses was originally pulled from medication-administration
record (MAR) **parquet files on S3** and wrangled with DuckDB + per-drug name/route/concentration
parsing (see `../medication-exposure/`). The same data now lives in the **BDSP OMOP Aurora
database** (`bdsp_omop`, schema `omop_prod`), where it is much easier to query.

## Why it's easier

| | Original parquet pipeline | OMOP |
|---|---|---|
| Cohort filter | list parquet files, DuckDB `WHERE BDSPPatientID IN (...)` across parts | `WHERE person_id = ANY(<ids>)` — **`BDSPPatientID` == OMOP `person_id`** |
| Drug match | name/route/concentration parsing from `MedicationDSC` | `drug_source_value ILIKE '%propofol%'` (or RxNorm concept) |
| Administrations vs orders | implicit in the MAR extract | explicit: `drug_type_concept_id = 38000180` ('Inpatient administration') |
| Infusion rate | mL/hr (needs × concentration to get mg) | `drug_exposure.sig` already in **mcg/kg/min** — the unit the Eleveld PK model wants |
| Patient weight | separate covariate file | joinable from `measurement` in the same DB |

`person.person_source_value` holds the **site code** (e.g. `I0001`, `I0002`), not the patient id.

## Verified equivalence

Reproducing **Table 2, RASS cohort, propofol** (patient-level exposure = any administration in the
24 h before the EEG-exam-end time) directly from OMOP:

| | patients | % | denominator |
|---|---|---|---|
| Published (parquet MAR pipeline) | 3,250 | 52.5% | 6,188 |
| OMOP (`reproduce_medication_exposure_omop.py`) | 3,245 | 52.4% | 6,188 |

The **denominator (6,188 patients with MAR data) reproduces exactly**; the exposed count is within
**5 patients (0.1 pp)** — the residual is boundary/rounding between the two snapshots. (As a further
sanity check, the I0002 cohort returns **zero** propofol administrations in OMOP, matching the
manuscript's note that infusion data were unavailable for that site.)

## Connecting

Set these environment variables, then run the script:

```
OMOP_HOST   OMOP_PORT=5432   OMOP_DB=bdsp_omop   OMOP_USER   OMOP_PASSWORD
```

- **External users** open an SSH tunnel through the bastion to Aurora and point `OMOP_HOST` at
  `localhost`, authenticating as your read-only `myelin_<user>` role. See
  `0-BDSP_PRODUCTION_DEPLOYMENT.md` ("The bastion (SSH tunnel into the VPC)") for the tunnel and the
  credentials file. The reader role is `SELECT`-only and cannot read `note`/`note_nlp`.
- **In-VPC code** (e.g. the prod web box) can use the Aurora endpoint directly.

All queries in the script are read-only (`default_transaction_read_only=on`).

## Reproduce the number

```bash
# cohort CSV = the per-cohort lookup shipped with the dataset, e.g.
#   s3://bdsp-opendata-credentialed/yama/NESI/NESI-Medication-Analysis/medication-exposure/data/RASS_NESI_lookup.csv
python reproduce_medication_exposure_omop.py \
    --cohort-csv RASS_NESI_lookup.csv \
    --drug propofol --window-hours 24 \
    --time-col EEGExamEndDTS --filter-col NESI \
    --expected 3250
```

The script supports other sedatives/opioids (`--drug midazolam|ketamine|dexmedetomidine|fentanyl|…`,
patterns mirroring `medication-exposure/drug_configs.py`) and any per-cohort lookup CSV.
