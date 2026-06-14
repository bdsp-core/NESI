# bolus_route_classify.py
"""
Route classification wrapper for bolus events.

Tags each bolus row with a 'route_bucket' value ('iv', 'non_iv', or 'excluded')
based on MedicationRouteDSC. When MedicationRouteDSC is null, falls back to
keyword matching on MedicationDSC ('injection'/'infusion' → IV;
'tablet'/'capsule'/'oral'/'rectal'/'kit' → non_iv; otherwise non_iv).

Route rules (for sedatives, opiates, and benzos in inpatient/ICU populations):
  iv        — Intravenous (including PCA-as-IV), central/peripheral line,
              or null-route rows whose drug name indicates injectable form
  non_iv    — Oral/PO, sublingual/SL, IM, SQ, rectal, transdermal/patch,
              buccal, intranasal; null-route rows whose drug name indicates
              non-injectable form; and any other null-route rows
  excluded  — Epidural, intrathecal, neuraxial, spinal
"""

import re
import pandas as pd

# Order matters: excluded patterns checked first, then IV, then default non_iv.
EXCLUDED_ROUTE_PATTERNS = [
    r"\bepidural\b",
    r"\bintrathecal\b",
    r"\bneuraxial\b",
    r"\bspinal\b",
]

# "Fast-acting parenteral" routes — IV, IM, intranasal. All have rapid onset
# (within minutes) and clinically equivalent CNS exposure on the 24h analysis
# window for the drugs in scope here. The 'iv' bucket label is retained for
# backward compatibility; describe the bucket accurately in methods.
IV_ROUTE_PATTERNS = [
    # IV
    r"\bintravenous\b",
    r"\biv\b",
    r"\bivp\b",
    r"\biv\s*push\b",
    r"\bpca\b",
    r"patient[\s-]?controlled",
    r"central\s*line",
    r"peripheral\s*line",
    # IM
    r"\bintramuscular\b",
    r"\bim\b",
    # Intranasal (not bare 'in' — too prone to false positives)
    r"\bintranasal\b",
    r"\bnasal\b",
]

# Used only when MedicationRouteDSC is null/missing.
NULL_ROUTE_IV_NAME_PATTERNS = [
    r"\binjection\b",
    r"\binfusion\b",
    r"\binjectable\b",
    r"\bsyringe\b",
    r"\bvial\b",
    r"\bsoln\b",                  # 'Soln' abbreviation = solution, IV context
    r"\bsodium\s+chloride\b",
    r"\bdextrose\b",
    r"\bd5w\b",
    r"\bnormal\s+saline\b",
    r"\bns\b",
    r"\bbag\b",
    r"\biv\s+bag\b",
    # Bare drug-name product descriptions (no form descriptor, null route).
    # In an inpatient/GCS cohort the plausible routes (IV, IM, intranasal,
    # IV-derived bolus) are all fast-acting parenteral within the 24h
    # exposure window. Excludes patches, lozenges, etc. (caught by non-IV
    # form keywords first).
    r"\bmidazolam\b",
    r"\bfentanyl\b",
    r"\bhydromorphone\b",
    r"\bmorphine\s+sulfate\b",
]

NULL_ROUTE_NON_IV_NAME_PATTERNS = [
    r"\btablet\b",
    r"\bcapsule\b",
    r"\boral\b",
    r"\bsyrup\b",
    r"\boral\s+solution\b",     # 'oral solution' rows
    r"\bsuspension\b",
    r"\bsuppository\b",
    r"\brectal\b",
    r"\bkit\b",           # diazepam rectal kit
    r"\bpatch\b",
    r"\btransdermal\b",
    r"\bbuccal\b",
    r"\bsublingual\b",
    r"\bspray\b",
    r"\bnasal\b",
    r"\blozenge\b",
    r"\bfilm\b",
]


def classify_route(route_str, med_dsc=None):
    """
    Return 'iv', 'non_iv', or 'excluded'.

    Primary signal is route_str (MedicationRouteDSC). When that is null,
    fall back to keyword matching on med_dsc (MedicationDSC).
    """
    # Primary classification on route
    if not pd.isna(route_str):
        r = str(route_str).lower()
        for pat in EXCLUDED_ROUTE_PATTERNS:
            if re.search(pat, r):
                return "excluded"
        for pat in IV_ROUTE_PATTERNS:
            if re.search(pat, r):
                return "iv"
        return "non_iv"

    # Fallback to drug name when route is null
    if pd.isna(med_dsc):
        return "non_iv"
    name = str(med_dsc).lower()
    # Non-IV form names take precedence (more specific)
    for pat in NULL_ROUTE_NON_IV_NAME_PATTERNS:
        if re.search(pat, name):
            return "non_iv"
    for pat in NULL_ROUTE_IV_NAME_PATTERNS:
        if re.search(pat, name):
            return "iv"
    return "non_iv"


def add_route_bucket(boluses_df, source_rows_df):
    """
    Given a boluses DataFrame (from extract_boluses_for_cohort) and the source
    rows it was derived from, attach a 'route_bucket' column.

    Pulls MedicationRouteDSC, MedicationDSC, and MedicationDisplayNM (when
    available) from source_rows_df by matching on BDSPPatientID + TakenTime.
    """
    if len(boluses_df) == 0:
        boluses_df = boluses_df.copy()
        for col in ["MedicationRouteDSC", "MedicationDSC", "MedicationDisplayNM"]:
            boluses_df[col] = pd.Series(dtype=object)
        boluses_df["route_bucket"] = pd.Series(dtype=object)
        return boluses_df

    key_cols = ["BDSPPatientID", "MedicationTakenDTS"]
    passthrough_cols = [
        "MedicationRouteDSC", "MedicationDSC", "MedicationDisplayNM",
        "ProductDescription", "ProductDescriptionOther",
    ]
    available = [c for c in passthrough_cols if c in source_rows_df.columns]

    src = (
        source_rows_df[key_cols + available]
        .drop_duplicates(subset=key_cols, keep="first")
        .rename(columns={"MedicationTakenDTS": "TakenTime"})
    )

    merged = boluses_df.merge(src, on=["BDSPPatientID", "TakenTime"], how="left")


    # Apply classification row-wise so we can use both fields together
# Apply classification row-wise so we can use both fields together
    def _drug_name(row):
        # Prefer MedicationDSC (I0001 source), fall back to ProductDescription
        # (I0002 source), then ProductDescriptionOther.
        for col in ("MedicationDSC", "ProductDescription", "ProductDescriptionOther"):
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                return v
        return None

    merged["route_bucket"] = merged.apply(
        lambda row: classify_route(
            row.get("MedicationRouteDSC"),
            _drug_name(row),
        ),
        axis=1,
    )
    return merged