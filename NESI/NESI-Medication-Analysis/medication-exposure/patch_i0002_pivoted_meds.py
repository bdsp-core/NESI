# patch_i0002_pivoted_meds.py
"""
Patch the I0002 pivoted medications parquet:
  - When MARActionDSC is null AND MARActionCode is one of the known bolus
    administration codes (ADMMORE, ADMPART), populate MARActionDSC with a
    canonical description so downstream extraction recognizes the event.
  - Leaves all other columns and rows unchanged.

Reads:  I0002_PIVOTED_MEDS_PARQUET
Writes: I0002_PIVOTED_MEDS_PATCHED_PARQUET

Paths read from med_config.py.
"""

import duckdb

import med_config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# Codes that should be treated as bolus administration when MARActionDSC is null.
# Values are the canonical lowercase descriptions we want filled in. Must match
# entries in BOLUS_GIVEN_ACTIONS in infusion_reconstruction.py.
MAR_CODE_DSC_FALLBACK = {
    "ADMMORE": "administered",
    "ADMPART": "partial administered",
}


try:
    src = str(med_config.I0002_PIVOTED_MEDS_PARQUET).replace("\\", "/")
    dst = str(med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET).replace("\\", "/")
    print(f"Reading: {med_config.I0002_PIVOTED_MEDS_PARQUET}")
    print(f"Writing: {med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET}")

    med_config.I0002_PIVOTED_MEDS_PATCHED_PARQUET.parent.mkdir(
        parents=True, exist_ok=True
    )

    con = duckdb.connect()
    temp_dir = str(med_config.DUCKDB_TEMP_DIR).replace("\\", "/")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET preserve_insertion_order = false")

    # Build a CASE WHEN expression for the patch
    case_when_parts = "\n            ".join(
        f"WHEN MARActionCode = '{code}' THEN '{dsc}'"
        for code, dsc in MAR_CODE_DSC_FALLBACK.items()
    )

    sql = f"""
        SELECT
            * EXCLUDE (MARActionDSC),
            CASE
                WHEN MARActionDSC IS NOT NULL THEN MARActionDSC
                {case_when_parts}
                ELSE MARActionDSC
            END AS MARActionDSC
        FROM '{src}'
    """
    con.execute(f"COPY ({sql}) TO '{dst}' (FORMAT PARQUET)")

    # Report on how many rows were patched
    n_total = con.execute(f"SELECT COUNT(*) FROM '{dst}'").fetchone()[0]
    n_patched = con.execute(f"""
        SELECT COUNT(*) FROM '{src}'
        WHERE MARActionDSC IS NULL
          AND MARActionCode IN ({','.join("'" + c + "'" for c in MAR_CODE_DSC_FALLBACK)})
    """).fetchone()[0]
    n_still_null = con.execute(f"""
        SELECT COUNT(*) FROM '{dst}' WHERE MARActionDSC IS NULL
    """).fetchone()[0]

    print(f"\nTotal rows:                 {n_total:,}")
    print(f"Rows patched (DSC filled):  {n_patched:,}")
    print(f"Rows still null after patch:{n_still_null:,}")

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