"""
infusion_reconstruction.py
==========================

Reconstruct continuous infusion intervals and extract bolus events from raw
EHR medication MAR data.

INFUSION RECONSTRUCTION
-----------------------
Infusion data in the EHR is recorded as a sequence of MAR events ("New Bag",
"Rate Change", "Rate Verify", "Stopped", etc.). Between events, the infusion
runs implicitly at the most recently stated rate. This module walks each
patient's events in order and reconstructs the actual (StartTime, EndTime,
Rate) intervals.

State machine semantics:
  - 'open'     : start a new interval at the stated rate. If an interval is
                 already open AT THE SAME RATE, this is treated as a
                 continuation (e.g., bag swap with no rate change) and does
                 NOT split the interval.
  - 'close'    : end the currently-open interval at the event's time.
  - 'continue' : informational; no state change unless the row carries a rate
                 different from the current one (implicit rate change).
  - 'ignore'   : skip the row entirely.

If the data ends with an interval still open (no 'Stopped'/'Paused' event),
the interval is closed at the last seen MedicationTakenDTS.

BOLUS EXTRACTION
----------------
Bolus rows are identified by having:
  - A parseable numeric DiscreteDoseAMT (rejects range strings like "0-50"
    that come along on infusion rows for titration ranges)
  - DoseUnitDSC matches the expected unit for the drug (case-insensitive)
  - MedicationTakenDTS populated
  - InfusionRateNBR null
  - MARActionDSC indicating actual administration ("Given", "Bolus from Bag")
"""

import pandas as pd
import numpy as np


# ============================================================================
# CONFIG
# ============================================================================

# MAR action vocabulary → state-machine event type for INFUSIONS.
# Keys are lowercased exact matches against MARActionDSC.
# State-machine events for infusion reconstruction. Lowercased exact matches
# against MARActionDSC. Includes both I0001 (Epic-style) and I0002
# (Cerner-style) vocabularies.
EVENT_TYPES = {
    # ─── I0001 (Epic) ────────────────────────────────────────────────────────
    'new bag':                          'open',
    'rate change':                      'open',
    'restarted':                        'open',
    'infusion from other mgb entity':   'open',
    'rate verify':                      'continue',
    'same bag':                         'continue',
    'stopped':                          'close',
    'paused':                           'close',
    'independent double check':         'ignore',
    'held':                             'ignore',
    'discontinued':                     'ignore',
    'completed':                        'ignore',

    # ─── I0002 (Cerner) ──────────────────────────────────────────────────────
    'started':                                    'open',
    'delayed started':                            'open',
    'started in other location':                  'open',
    'in other location':                          'open',   # OTHPSTART variant
    'stopped - unscheduled':                      'close',
    'stopped as directed':                        'close',
    'delayed stopped':                            'close',
    'stopped in other location':                  'close',
    'stopped - unscheduled in other location':    'close',
    'hold dose':                                  'ignore',
    'infusion reconciliation':                    'ignore',
}

# MAR actions that count as actual bolus administration.
# Recognized as bolus administration events. Lowercased exact matches against
# MARActionDSC. Includes both I0001 (Epic-style) and I0002 (Cerner-style)
# vocabularies so the same extraction runs against either source.
BOLUS_GIVEN_ACTIONS = {
    # I0001
    'given',
    'bolus from bag',
    # I0002
    'administered',
    'administered intraoperatively',
    'administered bolus from iv drip',
    'partial administered',
    'delayed administered',
    'administered in other location',
    'in other location',
}

# ============================================================================
# HELPERS
# ============================================================================

def _safe_to_float(x):
    """
    Try to convert a value to float. Returns NaN if it can't be parsed
    (e.g., for range strings like "0-50" which appear in DiscreteDoseAMT
    on infusion rows for titration ranges).
    """
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


# ============================================================================
# INFUSION RECONSTRUCTION
# ============================================================================

def reconstruct_infusions_for_patient(patient_rows):
    """
    Given all infusion-related MAR events for one patient (filtered to one drug),
    build the list of (StartTime, EndTime, Rate, RateUnit) intervals.
    
    Parameters
    ----------
    patient_rows : pd.DataFrame
        Must have columns:
            MedicationTakenDTS  (timestamp)
            InfusionRateNBR     (rate value)
            InfusionRateUnitDSC (rate unit string)
            MARActionDSC        (MAR action — used to classify event type)
    
    Returns
    -------
    pd.DataFrame with columns:
        StartTime, EndTime, Rate, RateUnit
    """
    df = patient_rows.copy().sort_values('MedicationTakenDTS').reset_index(drop=True)
    
    intervals = []
    cur_start = None
    cur_rate = None
    cur_unit = None
    
    for _, row in df.iterrows():
        action = str(row.get('MARActionDSC', '')).lower().strip()
        event_type = EVENT_TYPES.get(action, 'continue')  # unknown actions default to continue
        
        if event_type == 'ignore':
            continue
        
        t = row['MedicationTakenDTS']
        rate = row['InfusionRateNBR']
        unit = row['InfusionRateUnitDSC']
        
        if pd.isna(t):
            continue
        
        if event_type == 'open':
            # If we have an open interval already AND the rate hasn't changed,
            # this event is just a bag swap with the same rate — don't split it.
            if cur_start is not None and not pd.isna(rate) and rate == cur_rate:
                # No change; the interval continues seamlessly
                pass
            else:
                # Real rate change (or starting fresh) — close any open interval
                # and open a new one
                if cur_start is not None:
                    intervals.append({
                        'StartTime': cur_start,
                        'EndTime': t,
                        'Rate': cur_rate,
                        'RateUnit': cur_unit,
                    })
                cur_start = t
                cur_rate = rate
                cur_unit = unit
        
        elif event_type == 'close':
            if cur_start is not None:
                intervals.append({
                    'StartTime': cur_start,
                    'EndTime': t,
                    'Rate': cur_rate,
                    'RateUnit': cur_unit,
                })
                cur_start = None
                cur_rate = None
                cur_unit = None
            # Else: a 'close' with nothing open — orphan event, just skip
        
        elif event_type == 'continue':
            # Informational only. If a rate is provided and differs from current,
            # treat as implicit rate change.
            if cur_start is not None and not pd.isna(rate) and rate != cur_rate:
                intervals.append({
                    'StartTime': cur_start,
                    'EndTime': t,
                    'Rate': cur_rate,
                    'RateUnit': cur_unit,
                })
                cur_start = t
                cur_rate = rate
                cur_unit = unit
    
    # If patient's events end with an interval still open, close it at the last event time
    if cur_start is not None:
        last_time = df['MedicationTakenDTS'].max()
        if last_time > cur_start:
            intervals.append({
                'StartTime': cur_start,
                'EndTime': last_time,
                'Rate': cur_rate,
                'RateUnit': cur_unit,
            })
    
    return pd.DataFrame(intervals)


def reconstruct_infusions_for_cohort(infusion_rows):
    """
    Run reconstruction for every patient in the dataset.
    
    Parameters
    ----------
    infusion_rows : pd.DataFrame
        All MAR events with infusion data, across all patients. Must have
        BDSPPatientID column plus the columns required by the per-patient
        function.
    
    Returns
    -------
    pd.DataFrame with columns BDSPPatientID, StartTime, EndTime, Rate, RateUnit.
    """
    results = []
    for pid, group in infusion_rows.groupby('BDSPPatientID'):
        intervals = reconstruct_infusions_for_patient(group)
        if len(intervals) > 0:
            intervals['BDSPPatientID'] = pid
            results.append(intervals)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ============================================================================
# BOLUS EXTRACTION
# ============================================================================

def extract_boluses_for_patient(patient_rows, expected_unit='mg'):
    """
    Pull bolus events from a patient's medication rows.
    
    A bolus row is identified by:
      - DiscreteDoseAMT is a parseable numeric value (rejects range strings
        like "0-50" that come along on infusion rows)
      - DoseUnitDSC matches `expected_unit` (case-insensitive)
      - MedicationTakenDTS is populated
      - InfusionRateNBR is null
      - MARActionDSC is in BOLUS_GIVEN_ACTIONS (i.e., the dose was given)
    
    Parameters
    ----------
    patient_rows : pd.DataFrame
        Must have columns: DiscreteDoseAMT, DoseUnitDSC, MedicationTakenDTS,
        InfusionRateNBR, MARActionDSC
    expected_unit : str
        The dose unit this drug is expected to be charted in (e.g., 'mg' for 
        propofol/midazolam/ketamine, 'mcg' for dexmedetomidine). Case-insensitive.
        Rows with non-matching units are excluded.
    
    Returns
    -------
    pd.DataFrame with columns: TakenTime, Dose, DoseUnit, MARAction
        Dose values are in the unit specified by DoseUnit (matches expected_unit).
    """
    df = patient_rows.copy()
    
    # Parse dose to numeric — range strings ("0-50") and unparseable values become NaN
    df['_DoseNum'] = df['DiscreteDoseAMT'].apply(_safe_to_float)
    
    given_mask = df['MARActionDSC'].astype(str).str.lower().isin(BOLUS_GIVEN_ACTIONS)
    unit_match = df['DoseUnitDSC'].astype(str).str.lower() == expected_unit.lower()
    
    is_bolus = (
        df['_DoseNum'].notna() &
        unit_match &
        df['MedicationTakenDTS'].notna() &
        df['InfusionRateNBR'].isna() &
        given_mask
    )
    
    boluses = df[is_bolus][[
        'MedicationTakenDTS', '_DoseNum', 'DoseUnitDSC', 'MARActionDSC'
    ]].rename(columns={
        'MedicationTakenDTS': 'TakenTime',
        '_DoseNum': 'Dose',
        'DoseUnitDSC': 'DoseUnit',
        'MARActionDSC': 'MARAction',
    })
    
    return boluses.sort_values('TakenTime').reset_index(drop=True)


def extract_boluses_for_patient(patient_rows, expected_units=('mg',)):
    """
    Pull bolus events from a patient's medication rows.
    
    A bolus row is identified by:
      - DiscreteDoseAMT is a parseable numeric value (rejects range strings)
      - DoseUnitDSC matches one of the expected_units (case-insensitive)
      - MedicationTakenDTS is populated
      - InfusionRateNBR is null
      - MARActionDSC is in BOLUS_GIVEN_ACTIONS
    
    Each row's Dose value stays in its originally-charted unit. The DoseUnit
    column tells you which unit. No conversion is performed.
    
    Parameters
    ----------
    patient_rows : pd.DataFrame
        Required columns: DiscreteDoseAMT, DoseUnitDSC, MedicationTakenDTS,
        InfusionRateNBR, MARActionDSC
    expected_units : tuple of str
        Acceptable dose units (case-insensitive). Examples:
        ('mg',)            — propofol, midazolam
        ('mg', 'mg/kg')    — ketamine
        ('mcg',)           — dexmedetomidine
    
    Returns
    -------
    pd.DataFrame with columns: TakenTime, Dose, DoseUnit, MARAction
        Dose values are in their original units (whatever DoseUnit says for
        that row).
    """
    df = patient_rows.copy()
    
    df['_DoseNum'] = df['DiscreteDoseAMT'].apply(_safe_to_float)
    
    given_mask = df['MARActionDSC'].astype(str).str.lower().isin(BOLUS_GIVEN_ACTIONS)
    expected_units_lower = {u.lower() for u in expected_units}
    unit_match = df['DoseUnitDSC'].astype(str).str.lower().isin(expected_units_lower)
    
    is_bolus = (
        df['_DoseNum'].notna() &
        unit_match &
        df['MedicationTakenDTS'].notna() &
        df['InfusionRateNBR'].isna() &
        given_mask
    )
    
    boluses = df[is_bolus][[
        'MedicationTakenDTS', '_DoseNum', 'DoseUnitDSC', 'MARActionDSC'
    ]].rename(columns={
        'MedicationTakenDTS': 'TakenTime',
        '_DoseNum': 'Dose',
        'DoseUnitDSC': 'DoseUnit',
        'MARActionDSC': 'MARAction',
    })
    
    return boluses.sort_values('TakenTime').reset_index(drop=True)


def extract_boluses_for_cohort(rows, expected_units=('mg',)):
    """
    Run bolus extraction for every patient in the dataset.
    """
    results = []
    for pid, group in rows.groupby('BDSPPatientID'):
        boluses = extract_boluses_for_patient(group, expected_units=expected_units)
        if len(boluses) > 0:
            boluses['BDSPPatientID'] = pid
            results.append(boluses)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()