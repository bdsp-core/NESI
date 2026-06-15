#!/usr/bin/env python3
"""
reproduce_medication_exposure_omop.py
=====================================
Reproduce a Table-2-style patient-level medication-exposure number from the BDSP
**OMOP Aurora** database, instead of the original S3 parquet medication-administration
records (MARs). This is the "less wrangling" path: the cohort filter is just a list of
person_ids, and the medication administrations come straight out of `drug_exposure`.

Verified equivalence (NESI manuscript, Table 2, RASS cohort, propofol):
    Published (parquet MAR pipeline):   3,250 / 6,188 = 52.5 %
    OMOP (this script):                 3,245 / 6,188 = 52.4 %   (5-patient / 0.1-pp difference)
The denominator (6,188 patients with MAR data) reproduces exactly.

Key facts about the BDSP OMOP store (schema `omop_prod`):
  - **BDSPPatientID == OMOP `person_id`** (the site code, e.g. I0001, is in
    `person.person_source_value`). So a cohort pull is `WHERE person_id = ANY(<ids>)`.
  - Medications are in `drug_exposure`. The actual MAR (what the original parquet held) is
    `drug_type_concept_id = 38000180` ('Inpatient administration'); `32817` ('EHR') are
    orders/other and are excluded here to match the MAR.
  - Drugs are matched by name on `drug_source_value` (the same `name_patterns` used by the
    original `medication-exposure/drug_configs.py`).
  - Infusion rate is in `drug_exposure.sig` (for propofol, in **mcg/kg/min** — already the
    unit the Eleveld PK model wants; the parquet stored mL/hr, which equals
    `sig * weight_kg * 60 / (1000 * concentration_mg_per_mL)`).
  - Exposure (per Table 2) = any administration within `--window-hours` before the
    **EEG-exam-end** timestamp (`EEGExamEndDTS`); patient-level = any such qualifying EEG.

Connection (set as environment variables):
  OMOP_HOST OMOP_PORT(=5432) OMOP_DB(=bdsp_omop) OMOP_USER OMOP_PASSWORD
  - External users: open an SSH tunnel through the bastion to Aurora and point OMOP_HOST at
    localhost; authenticate as your read-only `myelin_<user>` role. See
    `0-BDSP_PRODUCTION_DEPLOYMENT.md` ("The bastion") for the tunnel + credentials.
  - In-VPC code (e.g. the prod web box) can use the Aurora endpoint directly (the prod
    container exposes the same values as AURORA_HOST/AURORA_DB/AURORA_USER/AURORA_PASSWORD).

The cohort CSV is the per-cohort lookup that ships with the dataset
(`s3://bdsp-opendata-credentialed/yama/NESI/NESI-Medication-Analysis/medication-exposure/data/
RASS_NESI_lookup.csv`), which carries one row per analyzed observation with `BDSPPatientID`,
`EEGExamEndDTS`, and the per-observation `NESI` value.

Example (reproduces the verified number):
  python reproduce_medication_exposure_omop.py --cohort-csv RASS_NESI_lookup.csv \
         --drug propofol --window-hours 24 --time-col EEGExamEndDTS \
         --filter-col NESI --expected 3250
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import psycopg2

# name_patterns mirror medication-exposure/drug_configs.py
DRUG_PATTERNS = {
    'propofol':        ['propofol', 'diprivan'],
    'midazolam':       ['midazolam', 'versed'],
    'ketamine':        ['ketamine', 'ketalar'],
    'dexmedetomidine': ['dexmedetomidine', 'precedex'],
    'fentanyl':        ['fentanyl'],
    'morphine':        ['morphine'],
    'hydromorphone':   ['hydromorphone', 'dilaudid'],
}
ADMIN_TYPE = 38000180  # 'Inpatient administration' == the MAR


def connect():
    return psycopg2.connect(
        host=os.environ['OMOP_HOST'], port=os.environ.get('OMOP_PORT', '5432'),
        dbname=os.environ.get('OMOP_DB', 'bdsp_omop'),
        user=os.environ['OMOP_USER'], password=os.environ['OMOP_PASSWORD'],
        connect_timeout=20,
        options='-c default_transaction_read_only=on -c statement_timeout=600000 -c search_path=omop_prod',
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cohort-csv', required=True, help='per-cohort lookup CSV (e.g. RASS_NESI_lookup.csv)')
    ap.add_argument('--drug', default='propofol', choices=sorted(DRUG_PATTERNS))
    ap.add_argument('--window-hours', type=float, default=24.0)
    ap.add_argument('--id-col', default='BDSPPatientID')
    ap.add_argument('--time-col', default='EEGExamEndDTS', help='window end (24h-before reference)')
    ap.add_argument('--filter-col', default='NESI', help='keep rows where this column is non-null (the analysis set)')
    ap.add_argument('--expected', type=float, default=None, help='published count to diff against')
    a = ap.parse_args()

    df = pd.read_csv(a.cohort_csv, low_memory=False)
    if a.filter_col and a.filter_col in df.columns:
        df = df[df[a.filter_col].notna()]
    df['t'] = pd.to_datetime(df[a.time_col], errors='coerce')
    df = df.dropna(subset=['t'])
    ids = sorted(df[a.id_col].dropna().astype(int).unique())
    print(f"cohort: {len(ids)} patients, {len(df)} analyzed observations")

    pats = DRUG_PATTERNS[a.drug]
    like = ' OR '.join(['drug_source_value ILIKE %s'] * len(pats))
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(DISTINCT person_id) FROM drug_exposure "
                    "WHERE person_id = ANY(%s) AND drug_type_concept_id = %s", (ids, ADMIN_TYPE))
        denom = cur.fetchone()[0]  # patients with any administration ("MAR available")
        cur.execute(f"""SELECT person_id, drug_exposure_start_datetime
                        FROM drug_exposure
                        WHERE person_id = ANY(%s) AND drug_type_concept_id = %s
                          AND drug_exposure_start_datetime IS NOT NULL AND ({like})""",
                    [ids, ADMIN_TYPE] + [f'%{p}%' for p in pats])
        ev = pd.DataFrame(cur.fetchall(), columns=['person_id', 'start'])

    ev['start'] = pd.to_datetime(ev['start'])
    by_pt = {pid: np.sort(g['start'].values) for pid, g in ev.groupby('person_id')}
    win = np.timedelta64(int(a.window_hours * 3600), 's')

    exposed = set()
    for pid, g in df.groupby(a.id_col):
        starts = by_pt.get(int(pid))
        if starts is None:
            continue
        for t in g['t'].values:
            i = np.searchsorted(starts, t - win, side='left')
            if i < len(starts) and starts[i] <= t:
                exposed.add(pid)
                break

    n = len(exposed)
    print(f"\n{a.drug} administrations pulled: {len(ev):,}")
    print(f"{a.drug} patient-level exposure ({a.window_hours:g}h before {a.time_col}): "
          f"{n} / {denom} = {100 * n / denom:.1f}%")
    if a.expected is not None:
        print(f"published: {a.expected:.0f} / {denom} = {100 * a.expected / denom:.1f}%  "
              f"-> diff {n - a.expected:+.0f} patients ({100 * (n - a.expected) / denom:+.2f} pp)")


if __name__ == '__main__':
    main()
