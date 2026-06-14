# eleveld_run_cohort.py
"""
Run hand-rolled Eleveld propofol PK/effect-site model on the cohort.

Cohort = patients with ≥1 RASS row where propofol_only=True.

For each (patient, hospitalization) in the cohort:
  - Build piecewise-constant infusion record from propofol_intervals.csv
    (rate in mg/min) plus bolus events from propofol_boluses.csv.
  - Get age, sex, weight, height from wt_ht_per_hosp_FULL.csv.
    Use measured medians if available, otherwise imputed values.
  - Simulate Cp / Ce across the full stay (assuming opioid=False — see notes).
  - Sample at all rate-change times and at every minute (event-aligned + 1-min fill).
  - Sample at every RASSRecordedDTS in the stay.

Outputs:
  - eleveld_timeseries.parquet : full per-patient Cp/Ce time series
  - wt_ht_per_hosp_FULL_with_eleveld.csv : original RASS rows annotated
       with Cp_ug_per_mL, Ce_ug_per_mL, PropofolSimStatus

Notes:
  - Opioid coadministration is set to False for the entire stay (per
    decision in build conversation). Cp/Ce values at propofol_only=True
    RASS rows are computed under the correct assumption (no opioid
    within ±12h); values at other times within the same stay may be
    biased by this assumption but are not used downstream.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from path_configs import OUTPUT_CSV, OUTPUT_PARQUET
from eleveld_propofol import eleveld_params, simulate


# ============ PATHS ============
INTERVALS_CSV = r'C:\Users\chris\Desktop\BDSP\sedative_exposures\propofol_intervals.csv'
BOLUSES_CSV   = r'C:\Users\chris\Desktop\BDSP\sedative_exposures\propofol_boluses.csv'
WTHT_CSV      = os.path.join(OUTPUT_CSV, 'wt_ht_per_hosp_FULL.csv')

OUT_PARQUET   = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries.parquet')
OUT_RASS_CSV  = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld.csv')


# ============ CONSTANTS ============
PROPOFOL_CONC_MG_PER_ML = 10.0   # all patients per build conversation
BOLUS_DURATION_MIN      = 0.1    # 6-second bolus, matches SimTIVA validation
SAMPLE_GRID_MIN         = 1.0    # uniform 1-min sampling on top of events


# ============ LOAD ============
print("Loading wt_ht_per_hosp_FULL.csv ...")
wtht = pd.read_csv(WTHT_CSV, parse_dates=[
    'HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'
])
print(f"  {len(wtht):,} RASS rows, {wtht['BDSPPatientID'].nunique():,} patients")

# Cohort: patients with ≥1 propofol_only=True RASS row
cohort_pts = wtht.loc[wtht['propofol_only'] == True, 'BDSPPatientID'].unique()
print(f"  Cohort patients (≥1 propofol_only=True): {len(cohort_pts):,}")

#cohort_pts = cohort_pts[:10]  # TEST RUN — first 10 patients only
#print(f"  TEST RUN: limiting to first {len(cohort_pts)} patients")

wtht_cohort = wtht[wtht['BDSPPatientID'].isin(cohort_pts)].copy()


# ============ RESOLVE WT/HT (measured > imputed) ============
wtht_cohort['weight_kg_used'] = wtht_cohort['MedianWeightForHospitalization'].fillna(
    wtht_cohort['ImputedWeight'])
wtht_cohort['height_cm_used'] = wtht_cohort['MedianHeightForHospitalization'].fillna(
    wtht_cohort['ImputedHeight'])
wtht_cohort['wtht_source'] = np.where(
    wtht_cohort['MedianWeightForHospitalization'].notna(), 'measured', 'imputed')


# ============ DISTINCT HOSPITALIZATIONS WITH ATTRS ============
# One row per (patient, hospitalization) carrying age/sex/wt/ht.
hosp_attrs = (wtht_cohort
    .drop_duplicates(subset=['BDSPPatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS'])
    [['BDSPPatientID', 'HospitalAdmitDTS', 'HospitalDischargeDTS',
      'AgeAtVisit', 'SexDSC', 'weight_kg_used', 'height_cm_used',
      'wtht_source']]
    .reset_index(drop=True)
)
print(f"  Cohort hospitalizations: {len(hosp_attrs):,}")


# ============ LOAD PROPOFOL INTERVALS ============
print("\nLoading propofol_intervals.csv ...")
intervals = pd.read_csv(INTERVALS_CSV, parse_dates=['StartTime', 'EndTime'])
intervals = intervals[intervals['BDSPPatientID'].isin(cohort_pts)].copy()
print(f"  {len(intervals):,} interval rows for cohort")

# Rate is mg/hr (= rate × concentration; concentration is 10 mg/mL for all).
# Convert to mg/min.
intervals['rate_mg_per_min'] = intervals['Rate_converted_per_hr'] / 60.0


# ============ LOAD PROPOFOL BOLUSES ============
print("Loading propofol_boluses.csv ...")
boluses = pd.read_csv(BOLUSES_CSV, parse_dates=['TakenTime'])
boluses = boluses[boluses['BDSPPatientID'].isin(cohort_pts)].copy()
# DoseUnit = mg per build conversation.
boluses['dose_mg'] = boluses['Dose'].astype(float)
print(f"  {len(boluses):,} bolus rows for cohort")


# ============ SIMULATE ONE HOSPITALIZATION ============
def clean_intervals_for_stay(stay_intervals, admit, discharge):
    """Sort, clip to stay window, resolve overlaps (later wins), return list of
    (start_dt, end_dt, rate_mg_per_min)."""
    df = stay_intervals.copy()
    df = df.sort_values('StartTime').reset_index(drop=True)
    # Clip to stay window
    df['StartTime'] = df['StartTime'].clip(lower=admit, upper=discharge)
    df['EndTime']   = df['EndTime'].clip(lower=admit, upper=discharge)
    df = df[df['EndTime'] > df['StartTime']].reset_index(drop=True)
    if df.empty:
        return [], 0
    # Resolve overlaps: later interval's StartTime truncates earlier's EndTime.
    n_overlaps = 0
    end_col   = df.columns.get_loc('EndTime')
    start_col = df.columns.get_loc('StartTime')
    for i in range(len(df) - 1):
        if df.iat[i, end_col] > df.iat[i + 1, start_col]:
            df.iat[i, end_col] = df.iat[i + 1, start_col]
            n_overlaps += 1
    df = df[df['EndTime'] > df['StartTime']].reset_index(drop=True)
    return list(zip(df['StartTime'], df['EndTime'], df['rate_mg_per_min'])), n_overlaps

def build_segments(intervals_for_stay, boluses_for_stay, admit, discharge):
    """Return (segments, t0, total_min, rate_change_times_min) where segments
    is a list of (t_start_min, t_end_min, rate_mg_per_min) relative to t0
    (= admit). Boluses are added as 6-second high-rate segments at the bolus
    times, layered on top of any active infusion."""
    cleaned, n_ov = clean_intervals_for_stay(intervals_for_stay, admit, discharge)

    # Choose t0 = admit (so rate-change times line up with absolute clock).
    t0 = admit
    total_min = (discharge - admit).total_seconds() / 60.0

    # Build a unified rate timeline at minute resolution events.
    # Approach: collect all transitions (rate-change events), then synthesize
    # piecewise-constant segments. Bolus events are SHORT high-rate inserts
    # that get layered into the segment list.

    events = []   # list of (t_min, rate_change_to)  for infusion only
    events.append((0.0, 0.0))  # rate=0 at t=0
    for (start, end, rate) in cleaned:
        t_s = (start - t0).total_seconds() / 60.0
        t_e = (end   - t0).total_seconds() / 60.0
        events.append((t_s, rate))
        events.append((t_e, 0.0))

    # Sort by time; if two events at the same time, the later in input wins.
    events.sort(key=lambda x: x[0])

    # Build segments from events
    seg = []
    for i in range(len(events) - 1):
        t_s, r = events[i]
        t_e = events[i + 1][0]
        if t_e > t_s:
            seg.append((t_s, t_e, r))
    if seg and seg[-1][1] < total_min:
        seg.append((seg[-1][1], total_min, 0.0))

    # Layer in boluses by splitting any segment they fall into and adding a
    # 6-second high-rate insert. Bolus rate = dose_mg / 0.1 min (1400 mg/min for 140mg).
    rate_change_times = sorted({t for t, _, _ in seg} | {t for _, t, _ in seg})

    new_seg = list(seg)
    for _, b_row in boluses_for_stay.iterrows():
        t_b = (b_row['TakenTime'] - t0).total_seconds() / 60.0
        if t_b < 0 or t_b > total_min:
            continue
        dose = b_row['dose_mg']
        bolus_rate = dose / BOLUS_DURATION_MIN
        # Find the segment containing t_b and split.
        new_seg = _insert_bolus(new_seg, t_b, BOLUS_DURATION_MIN, bolus_rate)
        rate_change_times.append(t_b)
        rate_change_times.append(t_b + BOLUS_DURATION_MIN)

    rate_change_times = sorted(set(rate_change_times))
    return new_seg, t0, total_min, rate_change_times, n_ov


def _insert_bolus(segments, t_b, dur, bolus_rate):
    """Insert a brief bolus segment at t_b, summing its rate with whatever
    infusion was active (so a bolus during an active infusion adds to it)."""
    out = []
    t_b_end = t_b + dur
    for (s, e, r) in segments:
        if e <= t_b or s >= t_b_end:
            out.append((s, e, r))
            continue
        # Overlap with bolus window
        if s < t_b:
            out.append((s, t_b, r))
        # Overlap region itself: rate = r + bolus_rate
        ov_s = max(s, t_b)
        ov_e = min(e, t_b_end)
        out.append((ov_s, ov_e, r + bolus_rate))
        if e > t_b_end:
            out.append((t_b_end, e, r))
    out.sort(key=lambda x: x[0])
    return out


# ============ MAIN LOOP ============
all_ts = []         # list of dataframes for parquet
rass_annotations = []  # list of dicts {RASS_index, cp, ce, status}
n_overlaps_total = 0
n_skipped_no_propofol = 0
n_simulated = 0

print(f"\nSimulating {len(hosp_attrs):,} hospitalizations...")
t_start = time.time()

for i, h in hosp_attrs.iterrows():
    pt    = h['BDSPPatientID']
    admit = h['HospitalAdmitDTS']
    disch = h['HospitalDischargeDTS']

    # Filter events for this stay
    stay_int = intervals[
        (intervals['BDSPPatientID'] == pt) &
        (intervals['StartTime'] < disch) &
        (intervals['EndTime']   > admit)
    ]
    stay_bol = boluses[
        (boluses['BDSPPatientID'] == pt) &
        (boluses['TakenTime'] >= admit) &
        (boluses['TakenTime'] <= disch)
    ]

    # If neither intervals nor boluses, skip (Cp=Ce=0 will be filled later).
    if stay_int.empty and stay_bol.empty:
        n_skipped_no_propofol += 1
        continue

    # Build segments + sampling grid
    segments, t0, total_min, rate_change_times, n_ov = build_segments(
        stay_int, stay_bol, admit, disch)
    n_overlaps_total += n_ov
    if not segments:
        n_skipped_no_propofol += 1
        continue

    # Sample at every minute + every rate change
    grid = np.arange(0, total_min + SAMPLE_GRID_MIN, SAMPLE_GRID_MIN)
    sample_min = np.unique(np.concatenate([grid, np.array(rate_change_times)]))
    sample_min = sample_min[(sample_min >= 0) & (sample_min <= total_min)]

    # Eleveld parameters for this patient
    sex_letter = 'm' if str(h['SexDSC']).lower().startswith('m') else 'f'
    params = eleveld_params(
        age=float(h['AgeAtVisit']),
        weight=float(h['weight_kg_used']),
        height=float(h['height_cm_used']),
        sex=sex_letter,
        opioid=False,
        arterial=True,
    )

    cp, ce = simulate(segments, params, sample_min)

    # Compute infusion rate at each sample time (last segment containing it)
    seg_arr = np.array(segments)  # (n_seg, 3)
    rate_at_sample = np.zeros_like(sample_min)
    for j, t in enumerate(sample_min):
        mask = (seg_arr[:, 0] <= t) & (seg_arr[:, 1] >= t)
        if mask.any():
            rate_at_sample[j] = seg_arr[mask][-1, 2]

    # Build per-patient timeseries dataframe
    ts_df = pd.DataFrame({
        'BDSPPatientID': pt,
        'HospitalAdmitDTS': admit,
        'HospitalDischargeDTS': disch,
        't_min_from_admit': sample_min,
        'datetime': admit + pd.to_timedelta(sample_min, unit='m'),
        'Cp_ug_per_mL': cp,
        'Ce_ug_per_mL': ce,
        'infusion_rate_mg_per_min': rate_at_sample,
    })
    all_ts.append(ts_df)

    # Sample Cp/Ce at RASS times for this stay
    rass_in_stay = wtht_cohort[
        (wtht_cohort['BDSPPatientID'] == pt) &
        (wtht_cohort['HospitalAdmitDTS'] == admit) &
        (wtht_cohort['HospitalDischargeDTS'] == disch)
    ]
    for ridx, rrow in rass_in_stay.iterrows():
        t_rass = (rrow['RASSRecordedDTS'] - admit).total_seconds() / 60.0
        if t_rass < 0 or t_rass > total_min:
            rass_annotations.append({
                'rass_index': ridx,
                'Cp_ug_per_mL': np.nan,
                'Ce_ug_per_mL': np.nan,
                'PropofolSimStatus': 'rass_outside_stay_window',
            })
            continue
        # Sample
        cp_r, ce_r = simulate(segments, params, np.array([t_rass]))
        rass_annotations.append({
            'rass_index': ridx,
            'Cp_ug_per_mL': float(cp_r[0]),
            'Ce_ug_per_mL': float(ce_r[0]),
            'PropofolSimStatus': 'simulated',
        })
    n_simulated += 1

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  {i+1}/{len(hosp_attrs)} done ({elapsed:.0f}s elapsed)")

elapsed = time.time() - t_start
print(f"\nSimulation done in {elapsed:.0f}s")
print(f"  Hospitalizations simulated: {n_simulated:,}")
print(f"  Hospitalizations skipped (no propofol): {n_skipped_no_propofol:,}")
print(f"  Total interval-overlap fixes applied:   {n_overlaps_total:,}")


# ============ WRITE PARQUET ============
print(f"\nWriting time-series parquet to {OUT_PARQUET} ...")
ts_all = pd.concat(all_ts, ignore_index=True)
ts_all.to_parquet(OUT_PARQUET, index=False)
print(f"  {len(ts_all):,} rows")


# ============ ANNOTATE RASS CSV ============
print(f"Writing annotated RASS CSV to {OUT_RASS_CSV} ...")
ann_df = pd.DataFrame(rass_annotations).set_index('rass_index')

out = wtht.copy()
out['Cp_ug_per_mL'] = np.nan
out['Ce_ug_per_mL'] = np.nan
out['PropofolSimStatus'] = 'not_in_cohort'

out.loc[ann_df.index, 'Cp_ug_per_mL']      = ann_df['Cp_ug_per_mL']
out.loc[ann_df.index, 'Ce_ug_per_mL']      = ann_df['Ce_ug_per_mL']
out.loc[ann_df.index, 'PropofolSimStatus'] = ann_df['PropofolSimStatus']

# Cohort patients with no propofol in their stay → status 'no_propofol_in_stay'
cohort_mask = out['BDSPPatientID'].isin(cohort_pts)
unset_mask  = cohort_mask & (out['PropofolSimStatus'] == 'not_in_cohort')
out.loc[unset_mask, 'PropofolSimStatus'] = 'no_propofol_in_stay'
out.loc[unset_mask, 'Cp_ug_per_mL'] = 0.0
out.loc[unset_mask, 'Ce_ug_per_mL'] = 0.0

out.to_csv(OUT_RASS_CSV, index=False)
print(f"  {len(out):,} rows written")
print(f"\n  PropofolSimStatus breakdown:")
print(out['PropofolSimStatus'].value_counts().to_string())


# ============ COMPLETION SOUND ============
try:
    import winsound
    winsound.MessageBeep(winsound.MB_OK)
except Exception:
    print('\a')