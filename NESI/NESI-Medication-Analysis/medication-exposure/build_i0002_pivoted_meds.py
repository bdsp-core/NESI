# build_i0002_pivoted_meds.py
"""
Build a wide-format I0002 medications parquet equivalent to I0001's
cohort_all_medications.parquet, by pivoting mar_details (EAV) and joining
with mar_nax (event metadata).

Aggregation rules per FieldName (applied within each MarID):
  Dose Given, Product Amount Given          -> SUM (numeric)
  Barcode Type, Product Code,               -> CONCAT (distinct values, ';' separated)
    Product Description, Product Description Other
  Everything else                            -> FIRST

Output columns are renamed to match I0001 conventions where applicable
(MedicationDSC, MedicationRouteDSC, MedicationTakenDTS, InfusionRateNBR, etc.)
so downstream extraction pipelines can use a similar interface.

Paths read from med_config.py.
"""

import duckdb
import pandas as pd

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ─── Aggregation rules per FieldName ──────────────────────────────────────────
SUM_FIELDS = {
    "Dose Given",
    "Product Amount Given",
}

CONCAT_FIELDS = {
    "Barcode Type",
    "Product Code",
    "Product Description",
    "Product Description Other",
}

# All other FieldNames use FIRST aggregation (single value per MarID assumed).

# ─── Output column renames (I0002 FieldName -> I0001-style column) ───────────
FIELDNAME_TO_OUTPUT_COL = {
    "Medication Description":          "MedicationDSC",
    "Route":                           "MedicationRouteDSC",
    "Route of Administration":         "RouteOfAdministrationDSC",
    "Administration Types":            "AdministrationTypesDSC",
    "Infusion Rate":                   "InfusionRateNBR",
    "Infusion Rate Units":             "InfusionRateUnitDSC",
    "Prior Infusion Rate":             "PriorInfusionRateNBR",
    "Infusion Rate Adjustment":        "InfusionRateAdjustment",
    "Infusion Rate Adjustment Amount": "InfusionRateAdjustmentAmount",
    "New IV Bag Hung":                 "NewIVBagHung",
    "Infusion Complete":               "InfusionComplete",
    "Restart Interval":                "RestartInterval",
    "Completion Interval":             "CompletionInterval",
    "Complete Dose Not Given":         "CompleteDoseNotGiven",
    "Remainder of dose will be given?": "RemainderOfDoseWillBeGiven",
    "Site":                            "Site",
    "Side":                            "Side",
    "Visual Verification of Non-Formulary Dose Completed":
        "VisualVerificationOfNonFormularyDoseCompleted",
    "Reason for Unscheduled Stop/Removal": "ReasonForUnscheduledStopRemoval",
    "Continued Infusion started in other location":
        "ContinuedInfusionStartedInOtherLocation",
    "Dose Due":                        "DoseDueNBR",
    "Dose Due Unit":                   "DoseDueUnitDSC",
    "Patient Location":                "PatientLocation",
    "ward":                            "Ward",
    "Dose Given":                      "DiscreteDoseAMT",
    "Dose Given Unit":                 "DoseUnitDSC",
    "Product Amount Given":            "ProductAmountGiven",
    "Product Unit":                    "ProductUnit",
    "Barcode Type":                    "BarcodeType",
    "Product Code":                    "ProductCode",
    "Product Description":             "ProductDescription",
    "Product Description Other":       "ProductDescriptionOther",
}


def build_pivot_sql(det_path):
    """
    Build a SQL pivot of mar_details: one row per DeidentifiedMarID, with one
    column per FieldName. Aggregation per the rules above.
    """
    select_clauses = []
    for fieldname, out_col in FIELDNAME_TO_OUTPUT_COL.items():
        # Escape single quotes in fieldname (none currently expected, but safe)
        fn = fieldname.replace("'", "''")
        if fieldname in SUM_FIELDS:
            # Sum after numeric conversion; non-numeric values become NULL via TRY_CAST
            expr = (
                f"SUM(TRY_CAST(CASE WHEN FieldName = '{fn}' THEN FieldValue END AS DOUBLE)) "
                f"AS \"{out_col}\""
            )
        elif fieldname in CONCAT_FIELDS:
            # Distinct concat, ';' separated. DuckDB's STRING_AGG with DISTINCT.
            expr = (
                f"STRING_AGG(DISTINCT CASE WHEN FieldName = '{fn}' THEN FieldValue END, ';') "
                f"AS \"{out_col}\""
            )
        else:
            # FIRST value per group
            expr = (
                f"ANY_VALUE(CASE WHEN FieldName = '{fn}' THEN FieldValue END) "
                f"AS \"{out_col}\""
            )
        select_clauses.append("    " + expr)

    select_block = ",\n".join(select_clauses)
    return f"""
        SELECT
            DeidentifiedMarID AS MarID,
{select_block}
        FROM '{det_path}'
        GROUP BY DeidentifiedMarID
    """


try:
    nax_path = str(
        med_config.I0002_COHORT_MEDS_DIR / "mar_nax_2025_parquet.parquet"
    ).replace("\\", "/")
    det_path = str(
        med_config.I0002_COHORT_MEDS_DIR / "mar_details_nax_2025_parquet.parquet"
    ).replace("\\", "/")
    out_path = str(med_config.I0002_PIVOTED_MEDS_PARQUET).replace("\\", "/")

    print(f"Reading mar_nax:     {nax_path}")
    print(f"Reading mar_details: {det_path}")
    print(f"Writing pivoted:     {out_path}\n")

    med_config.I0002_PIVOTED_MEDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET threads = 4")

    pivot_sql = build_pivot_sql(det_path)

    # Join pivot with mar_nax for patient/encounter/timestamp/event metadata.
    final_sql = f"""
            WITH pivoted AS (
{pivot_sql}
        )
        SELECT
            n.BDSPPatientID,
            n.BDSPEncounterID,
            n.MarID,
            n.POEID,
            n.EventDate,
            n.EventDateTime AS MedicationTakenDTS,
            n.Event        AS MARActionCode,
            n.EventTXT     AS MARActionDSC,
            n.ScheduleDateTime,
            n.EnterDateTime,
            p.* EXCLUDE (MarID)
        FROM '{nax_path}' n
        LEFT JOIN pivoted p 
          ON CAST(n.MarID AS VARCHAR) = CAST(p.MarID AS VARCHAR)
    """

    print("Building pivot and joining with mar_nax...")
    con.execute(f"COPY ({final_sql}) TO '{out_path}' (FORMAT PARQUET)")

    n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    n_patients = con.execute(
        f"SELECT COUNT(DISTINCT BDSPPatientID) FROM '{out_path}'"
    ).fetchone()[0]
    n_with_med = con.execute(
        f"SELECT COUNT(*) FROM '{out_path}' WHERE MedicationDSC IS NOT NULL"
    ).fetchone()[0]

    print(f"\nFinal pivoted parquet:")
    print(f"  Rows:                       {n_rows:,}")
    print(f"  Unique patients:            {n_patients:,}")
    print(f"  Rows with MedicationDSC:    {n_with_med:,}")

    # Spot-check the column list
    cols = con.execute(f"DESCRIBE SELECT * FROM '{out_path}' LIMIT 0").df()
    print(f"\n  Total columns: {len(cols)}")
    print("  First 15 columns:")
    print(cols.head(15).to_string(index=False))

    con.close()

    print("\nDone.")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_OK)
    else:
        try:
            print("\a")
        except Exception:
            pass

except Exception as e:
    print(f"\nERROR: {e}")
    if HAS_WINSOUND:
        winsound.MessageBeep(winsound.MB_ICONHAND)
    else:
        try:
            print("\a")
        except Exception:
            pass
    raise