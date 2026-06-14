# plot_eleveld_one_patient.py
"""
Plot propofol pump rate alongside predicted Cp and Ce for one patient
from the Eleveld time-series parquet.

Picks the first patient in the parquet by default (or set PATIENT_ID).
Saves a PNG. Two y-axes:
  Left: infusion rate (mg/min)
  Right: Cp / Ce (µg/mL)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from path_configs import OUTPUT_PARQUET


# ============ CONFIG ============
PARQUET_FILE = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries.parquet')
PATIENT_ID   = None     # None = pick first patient in file. Set to int to override.
OUTPUT_PNG   = os.path.join(OUTPUT_PARQUET, 'eleveld_one_patient.png')


# ============ LOAD ============
print(f"Loading {PARQUET_FILE} ...")
df = pd.read_parquet(PARQUET_FILE)
print(f"  {len(df):,} rows, {df['BDSPPatientID'].nunique():,} patients")

# Find patients who have bolus-like spikes in their infusion rate.
# Boluses appear as brief (~6 sec) high-rate segments — rate well above
# typical infusion levels (~5-15 mg/min for sedation).
BOLUS_RATE_THRESHOLD = 100  # mg/min — well above any continuous-infusion rate

bolus_pts = (df[df['infusion_rate_mg_per_min'] > BOLUS_RATE_THRESHOLD]
             ['BDSPPatientID'].unique())
print(f"  Patients with at least one bolus: {len(bolus_pts):,}")

#

#if PATIENT_ID is None:
 #   PATIENT_ID = df['BDSPPatientID'].iloc[0]
  #  print(f"  No patient specified — using first: {PATIENT_ID}")

#pt_df = df[df['BDSPPatientID'] == PATIENT_ID].copy()
#if pt_df.empty:
 #   raise SystemExit(f"No rows found for patient {PATIENT_ID}")
# Find patients with at least one bolus (rate spike >> typical infusion rate)
BOLUS_RATE_THRESHOLD = 100  # mg/min
bolus_pts = (df[df['infusion_rate_mg_per_min'] > BOLUS_RATE_THRESHOLD]
             ['BDSPPatientID'].unique())
print(f"  Patients with at least one bolus: {len(bolus_pts):,}")

if PATIENT_ID is None:
    if len(bolus_pts) > 0:
        PATIENT_ID = bolus_pts[0]
        print(f"  No patient specified — picking first patient WITH a bolus: {PATIENT_ID}")
    else:
        PATIENT_ID = df['BDSPPatientID'].iloc[0]
        print(f"  No patient specified — using first patient: {PATIENT_ID}")

pt_df = df[df['BDSPPatientID'] == PATIENT_ID].copy()
if pt_df.empty:
    raise SystemExit(f"No rows found for patient {PATIENT_ID}")

# If the patient has multiple hospitalizations in the file, plot the longest one
hosp_lengths = pt_df.groupby(['HospitalAdmitDTS', 'HospitalDischargeDTS']).size()
admit, disch = hosp_lengths.idxmax()
pt_df = pt_df[(pt_df['HospitalAdmitDTS'] == admit) &
              (pt_df['HospitalDischargeDTS'] == disch)].sort_values('t_min_from_admit')
print(f"  Plotting hospitalization {admit} → {disch} ({len(pt_df):,} samples)")


# ============ PLOT ============
fig, ax_rate = plt.subplots(figsize=(14, 5))
t_hr = pt_df['t_min_from_admit'] / 60.0

# Separate bolus spikes from continuous infusion for plotting purposes.
# Boluses are encoded as 6-second high-rate segments (>100 mg/min); we hide
# them from the rate trace and annotate them as vertical markers instead so
# the continuous-infusion scale is readable.
BOLUS_RATE_THRESHOLD = 100  # mg/min — distinguishes bolus from infusion
infusion_only = pt_df['infusion_rate_mg_per_min'].where(
    pt_df['infusion_rate_mg_per_min'] < BOLUS_RATE_THRESHOLD, 0)

ax_rate.step(t_hr, infusion_only, where='post',
             color='tab:blue', linewidth=1.2, label='Infusion rate')
ax_rate.fill_between(t_hr, 0, infusion_only, step='post',
                     alpha=0.15, color='tab:blue')
ax_rate.set_xlabel('Time from admit (hours)')
ax_rate.set_ylabel('Infusion rate (mg/min)', color='tab:blue')
ax_rate.tick_params(axis='y', labelcolor='tab:blue')

# Set rate axis ylim from the infusion-only data, with a little headroom.
infusion_max = infusion_only.max()
if infusion_max > 0:
    ax_rate.set_ylim(0, infusion_max * 1.3)
else:
    ax_rate.set_ylim(0, 1)

# Detect bolus events: rate >= threshold for ≤ 0.2 min (~12 sec).
# Each bolus segment shows up as 1-2 sample points; collapse to single events.
bolus_mask = pt_df['infusion_rate_mg_per_min'] >= BOLUS_RATE_THRESHOLD
bolus_times = pt_df.loc[bolus_mask, 't_min_from_admit'].values
bolus_rates = pt_df.loc[bolus_mask, 'infusion_rate_mg_per_min'].values

# Collapse adjacent bolus samples (within 0.5 min) into a single event,
# computing the dose as rate * 0.1 min (our standard bolus duration).
bolus_events = []
i = 0
while i < len(bolus_times):
    t_start = bolus_times[i]
    rate_max = bolus_rates[i]
    while i + 1 < len(bolus_times) and (bolus_times[i + 1] - bolus_times[i]) < 0.5:
        rate_max = max(rate_max, bolus_rates[i + 1])
        i += 1
    dose_mg = rate_max * 0.1   # rate × 0.1-min bolus duration
    bolus_events.append((t_start / 60.0, dose_mg))
    i += 1

# Draw bolus markers as vertical lines with dose labels.
for (t_b_hr, dose) in bolus_events:
    ax_rate.axvline(t_b_hr, color='tab:purple', linestyle=':',
                    linewidth=1.2, alpha=0.8)
    ax_rate.annotate(f'{dose:.0f} mg bolus',
                     xy=(t_b_hr, ax_rate.get_ylim()[1] * 0.95),
                     xytext=(3, 0), textcoords='offset points',
                     color='tab:purple', fontsize=8, rotation=90,
                     verticalalignment='top')
print(f"  Detected {len(bolus_events)} bolus event(s)")

# Cp / Ce on right axis
ax_conc = ax_rate.twinx()
ax_conc.plot(t_hr, pt_df['Cp_ug_per_mL'],
             color='tab:red', linewidth=1.4, label='Cp (plasma)')
ax_conc.plot(t_hr, pt_df['Ce_ug_per_mL'],
             color='tab:orange', linewidth=1.4, linestyle='--', label='Ce (effect-site)')
ax_conc.set_ylabel('Concentration (µg/mL)', color='tab:red')
ax_conc.tick_params(axis='y', labelcolor='tab:red')
ax_conc.set_ylim(bottom=0)

# Auto-zoom to the window where propofol is actually running, ±1 hr padding
active = pt_df[pt_df['infusion_rate_mg_per_min'] > 0]
if not active.empty:
    t_first = active['t_min_from_admit'].min() / 60.0
    t_last  = active['t_min_from_admit'].max() / 60.0
    pad_hr = 1.0
    ax_rate.set_xlim(max(0, t_first - pad_hr), t_last + pad_hr)
    print(f"  Zoomed to active window: "
          f"t = {t_first - pad_hr:.1f} – {t_last + pad_hr:.1f} hr")
# Title
plt.title(f'Patient {PATIENT_ID} — propofol infusion vs. Eleveld-predicted Cp / Ce')

# Combined legend
lines_a, labels_a = ax_rate.get_legend_handles_labels()
lines_b, labels_b = ax_conc.get_legend_handles_labels()
ax_rate.legend(lines_a + lines_b, labels_a + labels_b, loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"\nSaved → {OUTPUT_PNG}")
plt.show()