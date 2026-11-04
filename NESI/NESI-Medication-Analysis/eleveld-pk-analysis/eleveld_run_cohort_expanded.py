# eleveld_run_cohort_expanded.py
"""
Run the hand-rolled Eleveld propofol PK/effect-site model on the EXPANDED
cohort: propofol with no other sedative infusion, opioid infusions allowed.

Differences from eleveld_run_cohort.py:
  1. Cohort flags are computed here, per RASS row, from infusion intervals
     overlapping a symmetric +/-12h window around RASSRecordedDTS:
       propofol_no_other_sedative : propofol infusion in window AND no
                                    midazolam / ketamine / dexmedetomidine
                                    infusion in window   (expanded cohort)
       propofol_only_recomputed   : the above AND no fentanyl / morphine /
                                    hydromorphone infusion in window
     propofol_only_recomputed is compared against the existing propofol_only
     column and the agreement is printed. High agreement means the expanded
     flag is the same definition with only the opioid criterion removed.
  2. Simulated patients = patients with >=1 RASS row where
     propofol_no_other_sedative is True (a superset of the original cohort).
  3. Cp/Ce at RASS times are computed in one simulate() call per stay, and the
     infusion rate at each sample time is vectorized (same results, faster).
  4. The time-series parquet is written incrementally (lower memory).
  5. weight_kg_used / height_cm_used are written to the output CSV.
  6. If the primary run's _v2 CSV exists, Ce at shared RASS rows is compared
     and the max difference printed (should be ~0: same model, same inputs).
     The expanded run is self-contained; it does not reuse the primary output.

Opioid coadministration is False throughout, as in the primary run. Setting it
True only for opioid-exposed patients would change how Ce is calculated in
exactly the rows the opioid covariate compares, confounding the PK adjustment
with the opioid effect in the downstream model.

Outputs (OUTPUT_CSV / OUTPUT_PARQUET from med_config.py):
  wt_ht_per_hosp_FULL_with_eleveld_expanded.csv
  eleveld_timeseries_expanded.parquet
"""

import os
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import med_config as mc
from med_config import OUTPUT_CSV, OUTPUT_PARQUET
from eleveld_propofol import eleveld_params, simulate, T2
assert T2[10] < 0, "Stale eleveld_propofol.py: theta10 must be -0.0156 (Eleveld 2018 Table 2)"

import sys
from pathlib import Path
_MC = Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in sys.path:
    sys.path.insert(0, str(_MC))

# ============ PATHS ============
INTERVALS_CSV = os.path.join(mc.PROPOFOL_EXPOSURE_DIR, "propofol_intervals.csv")
BOLUSES_CSV   = os.path.join(mc.PROPOFOL_EXPOSURE_DIR, "propofol_boluses.csv")
WTHT_CSV      = os.path.join(OUTPUT_CSV, 'wt_ht_per_hosp_FULL.csv')
SED_DIR       = str(mc.SEDATIVE_EXPOSURE_DIR)    # midazolam/ketamine/dexmedetomidine_intervals.csv
OPI_DIR       = str(mc.OPIATE_EXPOSURE_DIR)      # fentanyl/morphine/hydromorphone_intervals.csv

OUT_PARQUET   = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries_expanded.parquet')
PRIMARY_V2_CSV = os.path.join(OUTPUT_CSV,    'wt_ht_per_hosp_FULL_with_eleveld_v2.csv')   # for consistency check
OUT_RASS_CSV  = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld_expanded.csv')


# ============ CONSTANTS ============
PROPOFOL_CONC_MG_PER_ML = 10.0   # all patients per build conversation
BOLUS_DURATION_MIN      = 0.1    # 6-second bolus, matches SimTIVA validation
SAMPLE_GRID_MIN         = 1.0    # uniform 1-min sampling on top of events
WINDOW_H                = 12     # symmetric +/- window for cohort flags
OTHER_SEDATIVES         = ['midazolam', 'ketamine', 'dexmedetomidine']
OPIOIDS                 = ['fentanyl', 'morphine', 'hydromorphone']


def norm_id(s):
    return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


# ============ COHORT FLAGS ============
def load_interval_file(path):
    if not os.path.exists(path):
        print(f"    MISSING: {path}  (treated as no infusions)")
        return pd.DataFrame(columns=['BDSPPatientID', 'StartTime', 'EndTime'])
    head = pd.read_csv(path, nrows=0).columns
    iv = pd.read_csv(path, usecols=['BDSPPatientID', 'StartTime', 'EndTime'] +
                     (['Rate'] if 'Rate' in head else []))
    iv['BDSPPatientID'] = norm_id(iv['BDSPPatientID'])
    iv['StartTime'] = pd.to_datetime(iv['StartTime'], errors='coerce')
    iv['EndTime'] = pd.to_datetime(iv['EndTime'], errors='coerce')
    iv = iv.dropna(subset=['StartTime'])
    iv['EndTime'] = iv['EndTime'].fillna(iv['StartTime'])
    if 'Rate' in iv.columns:                                   # titrated to 0 != running
        iv = iv[pd.to_numeric(iv['Rate'], errors='coerce').ne(0)]
    return iv[['BDSPPatientID', 'StartTime', 'EndTime']]


def flag_in_window(con, rows, iv):
    """1 if any interval overlaps [t - WINDOW_H, t + WINDOW_H], else 0 (indexed like rows)."""
    if len(iv) == 0:
        return pd.Series(0, index=rows.index)
    con.register('r', rows[['_rid', 'pid', 't']])
    con.register('iv', iv)
    hit = con.execute(f"""
        SELECT DISTINCT r._rid
        FROM r JOIN iv ON iv.BDSPPatientID = r.pid
         AND CAST(iv.StartTime AS TIMESTAMP) <= CAST(r.t AS TIMESTAMP) + INTERVAL {WINDOW_H} HOUR
         AND CAST(iv.EndTime   AS TIMESTAMP) >= CAST(r.t AS TIMESTAMP) - INTERVAL {WINDOW_H} HOUR
    """).df()['_rid']
    con.unregister('r')
    con.unregister('iv')
    return rows['_rid'].isin(set(hit)).astype(int)


def add_cohort_flags(wtht, prop_iv):
    print(f"\nComputing cohort flags (±{WINDOW_H}h infusion overlap)...")
    con = duckdb.connect()
    rows = pd.DataFrame({'_rid': np.arange(len(wtht)),
                         'pid': norm_id(wtht['BDSPPatientID']),
                         't': wtht['RASSRecordedDTS']}, index=wtht.index).dropna(subset=['t'])
    p = prop_iv.rename(columns={})[['BDSPPatientID', 'StartTime', 'EndTime']].copy()
    p['BDSPPatientID'] = norm_id(p['BDSPPatientID'])
    prop = flag_in_window(con, rows, p)
    other = pd.Series(0, index=rows.index)
    for d in OTHER_SEDATIVES:
        other = other | flag_in_window(con, rows, load_interval_file(os.path.join(SED_DIR, f'{d}_intervals.csv')))
    opi = pd.Series(0, index=rows.index)
    for d in OPIOIDS:
        opi = opi | flag_in_window(con, rows, load_interval_file(os.path.join(OPI_DIR, f'{d}_intervals.csv')))
    con.close()

    wtht['propofol_no_other_sedative'] = False
    wtht['propofol_only_recomputed'] = False
    wtht.loc[rows.index, 'propofol_no_other_sedative'] = (prop == 1) & (other == 0)
    wtht.loc[rows.index, 'propofol_only_recomputed'] = (prop == 1) & (other == 0) & (opi == 0)

    print(f"  propofol_no_other_sedative: {wtht['propofol_no_other_sedative'].sum():,} rows, "
          f"{wtht.loc[wtht['propofol_no_other_sedative'], 'BDSPPatientID'].nunique():,} patients")
    if 'propofol_only' in wtht.columns:
        orig = wtht['propofol_only'] == True
        rec = wtht['propofol_only_recomputed']
        print(f"  propofol_only (existing):   {orig.sum():,} rows")
        print(f"  propofol_only_recomputed:   {rec.sum():,} rows")
        print(f"  agreement: {100 * (orig == rec).mean():.2f}% of rows; "
              f"existing-only {int((orig & ~rec).sum()):,}, recomputed-only {int((~orig & rec).sum()):,}")
        missing = orig & ~wtht['propofol_no_other_sedative']
        print(f"  existing propofol_only rows NOT in expanded cohort: {int(missing.sum()):,} "
              f"(should be ~0; expanded must contain the primary cohort)")
    return wtht


# ============ SEGMENTS (unchanged from eleveld_run_cohort.py) ============
def clean_intervals_for_stay(stay_intervals, admit, discharge):
    df = stay_intervals.copy()
    df = df.sort_values('StartTime').reset_index(drop=True)
    df['StartTime'] = df['StartTime'].clip(lower=admit, upper=discharge)
    df['EndTime']   = df['EndTime'].clip(lower=admit, upper=discharge)
    df = df[df['EndTime'] > df['StartTime']].reset_index(drop=True)
    if df.empty:
        return [], 0
    n_overlaps = 0
    end_col   = df.columns.get_loc('EndTime')
    start_col = df.columns.get_loc('StartTime')
    for i in range(len(df) - 1):
        if df.iat[i, end_col] > df.iat[i + 1, start_col]:
            df.iat[i, end_col] = df.iat[i + 1, start_col]
            n_overlaps += 1
    df = df[df['EndTime'] > df['StartTime']].reset_index(drop=True)
    return list(zip(df['StartTime'], df['EndTime'], df['rate_mg_per_min'])), n_overlaps


def _insert_bolus(segments, t_b, dur, bolus_rate):
    out = []
    t_b_end = t_b + dur
    for (s, e, r) in segments:
        if e <= t_b or s >= t_b_end:
            out.append((s, e, r))
            continue
        if s < t_b:
            out.append((s, t_b, r))
        ov_s = max(s, t_b)
        ov_e = min(e, t_b_end)
        out.append((ov_s, ov_e, r + bolus_rate))
        if e > t_b_end:
            out.append((t_b_end, e, r))
    out.sort(key=lambda x: x[0])
    return out


def build_segments(intervals_for_stay, boluses_for_stay, admit, discharge):
    cleaned, n_ov = clean_intervals_for_stay(intervals_for_stay, admit, discharge)
    t0 = admit
    total_min = (discharge - admit).total_seconds() / 60.0
    events = [(0.0, 0.0)]
    for (start, end, rate) in cleaned:
        events.append(((start - t0).total_seconds() / 60.0, rate))
        events.append(((end - t0).total_seconds() / 60.0, 0.0))
    events.sort(key=lambda x: x[0])
    seg = []
    for i in range(len(events) - 1):
        t_s, r = events[i]
        t_e = events[i + 1][0]
        if t_e > t_s:
            seg.append((t_s, t_e, r))
    if seg and seg[-1][1] < total_min:
        seg.append((seg[-1][1], total_min, 0.0))
    rate_change_times = sorted({t for t, _, _ in seg} | {t for _, t, _ in seg})
    new_seg = list(seg)
    for _, b_row in boluses_for_stay.iterrows():
        t_b = (b_row['TakenTime'] - t0).total_seconds() / 60.0
        if t_b < 0 or t_b > total_min:
            continue
        new_seg = _insert_bolus(new_seg, t_b, BOLUS_DURATION_MIN, b_row['dose_mg'] / BOLUS_DURATION_MIN)
        rate_change_times += [t_b, t_b + BOLUS_DURATION_MIN]
    return new_seg, t0, total_min, sorted(set(rate_change_times)), n_ov


def rate_at(seg_arr, t):
    """Vectorized: rate of the last segment with start <= t <= end (0 if none)."""
    idx = np.searchsorted(seg_arr[:, 0], t, side='right') - 1
    ok = (idx >= 0)
    idx_c = np.clip(idx, 0, len(seg_arr) - 1)
    ok &= t <= seg_arr[idx_c, 1]
    return np.where(ok, seg_arr[idx_c, 2], 0.0)


def main():
    # ============ LOAD ============
    print("Loading wt_ht_per_hosp_FULL.csv ...")
    wtht = pd.read_csv(WTHT_CSV, parse_dates=['HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'])
    print(f"  {len(wtht):,} RASS rows, {wtht['BDSPPatientID'].nunique():,} patients")

    print("Loading propofol_intervals.csv ...")
    intervals_all = pd.read_csv(INTERVALS_CSV, parse_dates=['StartTime', 'EndTime'])
    intervals_all['rate_mg_per_min'] = intervals_all['Rate_converted_per_hr'] / 60.0

    flag_iv = intervals_all[['BDSPPatientID', 'StartTime', 'EndTime']].copy()
    flag_iv = flag_iv[intervals_all['rate_mg_per_min'].fillna(0) > 0]   # titrated to 0 != running
    wtht = add_cohort_flags(wtht, flag_iv)

    cohort_pts = wtht.loc[wtht['propofol_no_other_sedative'], 'BDSPPatientID'].unique()
    print(f"\n  Expanded-cohort patients to simulate: {len(cohort_pts):,}")
    wtht_cohort = wtht[wtht['BDSPPatientID'].isin(cohort_pts)].copy()

    # wt/ht (measured > imputed)
    for df in (wtht, wtht_cohort):
        df['weight_kg_used'] = df['MedianWeightForHospitalization'].fillna(df['ImputedWeight'])
        df['height_cm_used'] = df['MedianHeightForHospitalization'].fillna(df['ImputedHeight'])
    wtht_cohort['wtht_source'] = np.where(
        wtht_cohort['MedianWeightForHospitalization'].notna(), 'measured', 'imputed')

    hosp_attrs = (wtht_cohort
        .drop_duplicates(subset=['BDSPPatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS'])
        [['BDSPPatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
          'AgeAtVisit', 'SexDSC', 'weight_kg_used', 'height_cm_used', 'wtht_source']]
        .reset_index(drop=True))
    print(f"  Cohort hospitalizations: {len(hosp_attrs):,}")

    intervals = intervals_all[intervals_all['BDSPPatientID'].isin(cohort_pts)].copy()
    print("Loading propofol_boluses.csv ...")
    boluses = pd.read_csv(BOLUSES_CSV, parse_dates=['TakenTime'])
    boluses = boluses[boluses['BDSPPatientID'].isin(cohort_pts)].copy()
    boluses['dose_mg'] = boluses['Dose'].astype(float)

    # ============ MAIN LOOP ============
    rass_annotations = []
    n_overlaps_total = n_skipped = n_simulated = 0
    writer = None
    tmp_parquet = OUT_PARQUET + '.tmp'
    print(f"\nSimulating {len(hosp_attrs):,} hospitalizations...")
    t_start = time.time()

    for i, h in hosp_attrs.iterrows():
        pt, admit, disch = h['BDSPPatientID'], h['HospitalAdmitDTS'], h['HospitalDischargeDTS']
        stay_int = intervals[(intervals['BDSPPatientID'] == pt) &
                             (intervals['StartTime'] < disch) & (intervals['EndTime'] > admit)]
        stay_bol = boluses[(boluses['BDSPPatientID'] == pt) &
                           (boluses['TakenTime'] >= admit) & (boluses['TakenTime'] <= disch)]
        if stay_int.empty and stay_bol.empty:
            n_skipped += 1
            continue
        segments, t0, total_min, rate_change_times, n_ov = build_segments(stay_int, stay_bol, admit, disch)
        n_overlaps_total += n_ov
        if not segments:
            n_skipped += 1
            continue

        grid = np.arange(0, total_min + SAMPLE_GRID_MIN, SAMPLE_GRID_MIN)
        sample_min = np.unique(np.concatenate([grid, np.array(rate_change_times)]))
        sample_min = sample_min[(sample_min >= 0) & (sample_min <= total_min)]

        sex_letter = 'm' if str(h['SexDSC']).lower().startswith('m') else 'f'
        params = eleveld_params(age=float(h['AgeAtVisit']), weight=float(h['weight_kg_used']),
                                height=float(h['height_cm_used']), sex=sex_letter,
                                opioid=False, arterial=True)
        cp, ce = simulate(segments, params, sample_min)
        seg_arr = np.array(segments, dtype=float)

        table = pa.Table.from_pandas(pd.DataFrame({
            'BDSPPatientID': pt, 'HospitalAdmitDTS': admit, 'HospitalDischargeDTS': disch,
            't_min_from_admit': sample_min,
            'datetime': admit + pd.to_timedelta(sample_min, unit='m'),
            'Cp_ug_per_mL': cp, 'Ce_ug_per_mL': ce,
            'infusion_rate_mg_per_min': rate_at(seg_arr, sample_min),
        }), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_parquet, table.schema)
        writer.write_table(table.cast(writer.schema))

        # Cp/Ce at all RASS times in this stay, one simulate() call
        rs = wtht_cohort[(wtht_cohort['BDSPPatientID'] == pt) &
                         (wtht_cohort['HospitalAdmitDTS'] == admit) &
                         (wtht_cohort['HospitalDischargeDTS'] == disch)]
        t_rass = ((rs['RASSRecordedDTS'] - admit).dt.total_seconds() / 60.0).to_numpy()
        inside = (t_rass >= 0) & (t_rass <= total_min)
        cp_r = np.full(len(rs), np.nan)
        ce_r = np.full(len(rs), np.nan)
        if inside.any():
            cp_r[inside], ce_r[inside] = simulate(segments, params, t_rass[inside])
        for ridx, ins, a, b in zip(rs.index, inside, cp_r, ce_r):
            rass_annotations.append({'rass_index': ridx, 'Cp_ug_per_mL': a, 'Ce_ug_per_mL': b,
                                     'PropofolSimStatus': 'simulated' if ins else 'rass_outside_stay_window'})
        n_simulated += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(hosp_attrs)} done ({time.time() - t_start:.0f}s elapsed)")

    if writer is not None:
        writer.close()
        os.replace(tmp_parquet, OUT_PARQUET)
    print(f"\nSimulation done in {time.time() - t_start:.0f}s")
    print(f"  Hospitalizations simulated: {n_simulated:,}")
    print(f"  Hospitalizations skipped (no propofol): {n_skipped:,}")
    print(f"  Total interval-overlap fixes applied:   {n_overlaps_total:,}")
    print(f"  Time series → {OUT_PARQUET}")

    # ============ ANNOTATE RASS CSV ============
    ann_df = pd.DataFrame(rass_annotations).set_index('rass_index')
    out = wtht.copy()
    out['Cp_ug_per_mL'] = np.nan
    out['Ce_ug_per_mL'] = np.nan
    out['PropofolSimStatus'] = 'not_in_cohort'
    out.loc[ann_df.index, 'Cp_ug_per_mL'] = ann_df['Cp_ug_per_mL']
    out.loc[ann_df.index, 'Ce_ug_per_mL'] = ann_df['Ce_ug_per_mL']
    out.loc[ann_df.index, 'PropofolSimStatus'] = ann_df['PropofolSimStatus']
    unset = out['BDSPPatientID'].isin(cohort_pts) & (out['PropofolSimStatus'] == 'not_in_cohort')
    out.loc[unset, 'PropofolSimStatus'] = 'no_propofol_in_stay'
    out.loc[unset, ['Cp_ug_per_mL', 'Ce_ug_per_mL']] = 0.0

    tmp = OUT_RASS_CSV + '.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, OUT_RASS_CSV)
    print(f"Annotated RASS CSV → {OUT_RASS_CSV}  ({len(out):,} rows)")
    print("\n  PropofolSimStatus breakdown:")
    print(out['PropofolSimStatus'].value_counts().to_string())

    # ============ CONSISTENCY CHECK vs primary _v2 run ============
    # Patients simulated in both runs must have identical Ce (same model, inputs, opioid=False).
    if os.path.exists(PRIMARY_V2_CSV):
        v2 = pd.read_csv(PRIMARY_V2_CSV, usecols=['BDSPPatientID', 'RASSRecordedDTS', 'Ce_ug_per_mL',
                                                  'PropofolSimStatus'], parse_dates=['RASSRecordedDTS'])
        v2 = v2[v2['PropofolSimStatus'] == 'simulated']
        m = out[out['PropofolSimStatus'] == 'simulated'][['BDSPPatientID', 'RASSRecordedDTS', 'Ce_ug_per_mL']] \
            .merge(v2, on=['BDSPPatientID', 'RASSRecordedDTS'], suffixes=('_exp', '_v2'))
        if len(m):
            d = (m['Ce_ug_per_mL_exp'] - m['Ce_ug_per_mL_v2']).abs()
            print(f"\n  Consistency vs primary _v2 run: {len(m):,} shared rows, "
                  f"max |ΔCe| = {d.max():.2e} µg/mL" + ("  OK" if d.max() < 1e-6 else "  <-- CHECK"))
    else:
        print(f"\n  (Primary _v2 CSV not found; skipping consistency check)")


if __name__ == '__main__':
    try:
        main()
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            print('\a')
    except Exception:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            print('\a')
        raise