# eleveld_cohort_sanity_checks.py
"""
Sanity checks on the cohort-level Eleveld output, with results captured
to a markdown report.

Reads:
  - wt_ht_per_hosp_FULL_with_eleveld.csv  (RASS rows annotated with Cp/Ce)
  - eleveld_timeseries.parquet            (full per-patient time series)

Produces three checks plus a markdown report:

  CHECK 1: Distribution of Cp / Ce at propofol_only=True RASS rows.
  CHECK 2: Cp / Ce at RASS rows binned by hours-since-last-infusion.
  CHECK 3: RASS score vs predicted Ce. RASS values > 0 are excluded
           because positive RASS (agitation) is dominated by delirium /
           withdrawal / under-sedation, not by propofol exposure.

Writes:
  - sanity_cp_ce_distribution.png
  - sanity_ce_vs_offtime.png
  - sanity_rass_vs_ce.png
  - sanity_report.md
"""

import os
import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from path_configs import OUTPUT_CSV, OUTPUT_PARQUET


# ============ PATHS ============
RASS_CSV     = os.path.join(OUTPUT_CSV,     'wt_ht_per_hosp_FULL_with_eleveld.csv')
TS_PARQUET   = os.path.join(OUTPUT_PARQUET, 'eleveld_timeseries.parquet')

OUT_PNG_DIST     = os.path.join(OUTPUT_PARQUET, 'sanity_cp_ce_distribution.png')
OUT_PNG_OFFTIME  = os.path.join(OUTPUT_PARQUET, 'sanity_ce_vs_offtime.png')
OUT_PNG_RASS     = os.path.join(OUTPUT_PARQUET, 'sanity_rass_vs_ce.png')
OUT_REPORT_MD    = os.path.join(OUTPUT_PARQUET, 'sanity_report.md')


# ============ MARKDOWN BUILDER ============
md_lines = []
def md(s=''):
    md_lines.append(s)


md('# Eleveld Cohort Sanity Checks\n')
md(f'_Generated from `{os.path.basename(RASS_CSV)}` and '
   f'`{os.path.basename(TS_PARQUET)}`._\n')


# ============ LOAD ============
print("Loading annotated RASS CSV ...")
rass = pd.read_csv(RASS_CSV, parse_dates=[
    'HospitalAdmitDTS', 'HospitalDischargeDTS', 'RASSRecordedDTS'
])
print(f"  {len(rass):,} rows")

po = rass[(rass['propofol_only'] == True) &
          (rass['PropofolSimStatus'] == 'simulated')].copy()
print(f"  Simulated propofol_only=True rows: {len(po):,}")
print(f"  Across {po['BDSPPatientID'].nunique():,} patients")

md('## Cohort\n')
md(f'- Total RASS rows in file: **{len(rass):,}**')
md(f'- Simulated propofol_only=True rows used for checks: '
   f'**{len(po):,}** across **{po["BDSPPatientID"].nunique():,}** patients\n')


# ============ CHECK 1: Cp / Ce DISTRIBUTION ============
print("\n=== CHECK 1: Cp / Ce distribution at propofol_only=True RASS rows ===")
md('## Check 1 — Cp / Ce distribution\n')
md('Distribution of predicted Cp and Ce at the simulated, propofol-only '
   'RASS observations.\n')

cp_summary = po['Cp_ug_per_mL'].describe(percentiles=[.05, .25, .5, .75, .95])
ce_summary = po['Ce_ug_per_mL'].describe(percentiles=[.05, .25, .5, .75, .95])

print("\nCp summary (µg/mL):")
print(cp_summary.to_string())
print("\nCe summary (µg/mL):")
print(ce_summary.to_string())

# Markdown table for both
md('| statistic | Cp (µg/mL) | Ce (µg/mL) |')
md('|---|---:|---:|')
for stat in cp_summary.index:
    md(f'| {stat} | {cp_summary[stat]:.3f} | {ce_summary[stat]:.3f} |')
md('')

n = len(po)
in_range_cp = ((po['Cp_ug_per_mL'] >= 0.5) & (po['Cp_ug_per_mL'] <= 5)).sum()
in_range_ce = ((po['Ce_ug_per_mL'] >= 0.5) & (po['Ce_ug_per_mL'] <= 5)).sum()
n_cp_zero = (po['Cp_ug_per_mL'] == 0).sum()
n_ce_zero = (po['Ce_ug_per_mL'] == 0).sum()

print(f"\nCp in 0.5–5 µg/mL: {in_range_cp:,} / {n:,} ({100*in_range_cp/n:.1f}%)")
print(f"Ce in 0.5–5 µg/mL: {in_range_ce:,} / {n:,} ({100*in_range_ce/n:.1f}%)")
print(f"Cp == 0: {n_cp_zero:,}   Ce == 0: {n_ce_zero:,}")

md(f'- Cp in 0.5–5 µg/mL: **{in_range_cp:,} / {n:,} ({100*in_range_cp/n:.1f}%)**')
md(f'- Ce in 0.5–5 µg/mL: **{in_range_ce:,} / {n:,} ({100*in_range_ce/n:.1f}%)**')
md(f'- Cp == 0: {n_cp_zero:,}    Ce == 0: {n_ce_zero:,}\n')

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, col, label in zip(axes, ['Cp_ug_per_mL', 'Ce_ug_per_mL'], ['Cp', 'Ce']):
    vals = po[col].dropna()
    ax.hist(vals, bins=60, range=(0, min(10, vals.max())),
            color='tab:blue', alpha=0.7, edgecolor='white')
    ax.axvline(vals.median(), color='tab:red', linestyle='--',
               label=f'median = {vals.median():.2f}')
    ax.axvspan(0.5, 5, color='tab:green', alpha=0.1, label='ICU sedation reference range')
    ax.set_xlabel(f'{label} (µg/mL)')
    ax.set_ylabel('count (RASS rows)')
    ax.set_title(f'{label} at propofol_only=True RASS rows  (n={len(vals):,})')
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG_DIST, dpi=150)
plt.close()
print(f"Saved → {OUT_PNG_DIST}")
md(f'![Cp / Ce distribution]({os.path.basename(OUT_PNG_DIST)})\n')


# ============ CHECK 2: Ce VS HOURS SINCE INFUSION OFF ============
print("\n=== CHECK 2: Ce vs hours-since-last-infusion ===")
md('## Check 2 — Cp / Ce vs hours since last active infusion\n')
md('Demonstrates expected propofol washout dynamics: Cp drops fast '
   '(redistribution + clearance); Ce drops more slowly because residual '
   'drug returns from V3.\n')

print("Loading time-series parquet ...")
import duckdb
# Pull only the rows where infusion is active, only the three columns we need.
# This keeps memory footprint to ~3 columns × N_active rows.
ts_active_df = duckdb.query(f"""
    SELECT BDSPPatientID, HospitalAdmitDTS, t_min_from_admit
    FROM read_parquet('{TS_PARQUET.replace(os.sep, '/')}')
    WHERE infusion_rate_mg_per_min > 0
    ORDER BY BDSPPatientID, HospitalAdmitDTS, t_min_from_admit
""").df()
print(f"  {len(ts_active_df):,} active-infusion sample rows")

active_by_stay = (
    ts_active_df.groupby(['BDSPPatientID', 'HospitalAdmitDTS'])
                ['t_min_from_admit'].apply(np.array).to_dict()
)

# Free the intermediate now that we have the dict.
del ts_active_df

hours_off = []
for _, r in po.iterrows():
    key = (r['BDSPPatientID'], r['HospitalAdmitDTS'])
    t_rass = (r['RASSRecordedDTS'] - r['HospitalAdmitDTS']).total_seconds() / 60.0
    arr = active_by_stay.get(key)
    if arr is None or len(arr) == 0:
        hours_off.append(np.nan)
        continue
    prior = arr[arr <= t_rass]
    if len(prior) == 0:
        hours_off.append(np.nan)
    else:
        hours_off.append((t_rass - prior.max()) / 60.0)

po['hours_since_active'] = hours_off

bins   = [0, 0.0167, 0.5, 2, 6, 24, 1e6]
labels = ['actively running', '<30 min', '30 min–2 hr',
          '2–6 hr', '6–24 hr', '>24 hr']
po['off_bin'] = pd.cut(po['hours_since_active'], bins=bins, labels=labels,
                       include_lowest=True)

agg = po.groupby('off_bin', observed=True).agg(
    n=('Cp_ug_per_mL', 'size'),
    cp_median=('Cp_ug_per_mL', 'median'),
    cp_p25=('Cp_ug_per_mL', lambda s: s.quantile(0.25)),
    cp_p75=('Cp_ug_per_mL', lambda s: s.quantile(0.75)),
    ce_median=('Ce_ug_per_mL', 'median'),
    ce_p25=('Ce_ug_per_mL', lambda s: s.quantile(0.25)),
    ce_p75=('Ce_ug_per_mL', lambda s: s.quantile(0.75)),
).reindex(labels)

print("\nMedian Cp / Ce by time-since-infusion-active:")
print(agg.round(3).to_string())

md('| time bin | n | Cp median (IQR) | Ce median (IQR) |')
md('|---|---:|---|---|')
for label in labels:
    if label not in agg.index or pd.isna(agg.loc[label, 'n']):
        continue
    row = agg.loc[label]
    md(f"| {label} | {int(row['n']):,} | "
       f"{row['cp_median']:.3f} ({row['cp_p25']:.3f} – {row['cp_p75']:.3f}) | "
       f"{row['ce_median']:.3f} ({row['ce_p25']:.3f} – {row['ce_p75']:.3f}) |")
md('')

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(labels))
agg_p = agg.reindex(labels)
ax.errorbar(x_pos - 0.1, agg_p['cp_median'],
            yerr=[agg_p['cp_median'] - agg_p['cp_p25'],
                  agg_p['cp_p75']    - agg_p['cp_median']],
            fmt='o', color='tab:red', label='Cp (median, IQR)', capsize=4)
ax.errorbar(x_pos + 0.1, agg_p['ce_median'],
            yerr=[agg_p['ce_median'] - agg_p['ce_p25'],
                  agg_p['ce_p75']    - agg_p['ce_median']],
            fmt='s', color='tab:orange', label='Ce (median, IQR)', capsize=4)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('Concentration (µg/mL)')
ax.set_title('Predicted Cp / Ce vs time since infusion was active\n'
             '(at propofol_only=True RASS rows)')
ax.set_yscale('log')
ax.set_ylim(0.001, 10)
ax.grid(True, alpha=0.3, which='both')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG_OFFTIME, dpi=150)
plt.close()
print(f"Saved → {OUT_PNG_OFFTIME}")
md(f'![Cp / Ce vs offtime]({os.path.basename(OUT_PNG_OFFTIME)})\n')


# ============ CHECK 3: RASS SCORE VS CE  (RASS > 0 EXCLUDED) ============
print("\n=== CHECK 3: RASS score vs Ce (excluding RASS > 0) ===")
md('## Check 3 — RASS score vs predicted Ce\n')
md('Tests whether deeper sedation (more negative RASS) associates with '
   'higher predicted effect-site concentration. Positive RASS (+1 to +4) '
   'is excluded because agitation is typically driven by delirium / '
   'withdrawal / under-sedation rather than propofol exposure.\n')

rass_col = 'R PHS IP RASS'
po_r = po[po[rass_col].notna()].copy()
po_r[rass_col] = pd.to_numeric(po_r[rass_col], errors='coerce')
po_r = po_r[po_r[rass_col].between(-5, 0)]    # RASS > 0 excluded
print(f"  {len(po_r):,} rows with RASS in [-5, 0]")
md(f'- Rows with RASS in [−5, 0]: **{len(po_r):,}**\n')

rass_summary = po_r.groupby(rass_col).agg(
    n=('Ce_ug_per_mL', 'size'),
    ce_median=('Ce_ug_per_mL', 'median'),
    ce_p25=('Ce_ug_per_mL', lambda s: s.quantile(0.25)),
    ce_p75=('Ce_ug_per_mL', lambda s: s.quantile(0.75)),
)
print("\nCe by RASS score:")
print(rass_summary.round(3).to_string())

md('| RASS | n | Ce median (IQR) |')
md('|---:|---:|---|')
for r_score, row in rass_summary.iterrows():
    md(f"| {int(r_score)} | {int(row['n']):,} | "
       f"{row['ce_median']:.3f} ({row['ce_p25']:.3f} – {row['ce_p75']:.3f}) |")
md('')

# Boxplot
fig, ax = plt.subplots(figsize=(10, 5))
rass_vals = sorted(po_r[rass_col].unique())
data = [po_r.loc[po_r[rass_col] == r, 'Ce_ug_per_mL'].values for r in rass_vals]
counts = [len(d) for d in data]
bp = ax.boxplot(data, positions=rass_vals, widths=0.6,
                showfliers=False, patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5))
for patch in bp['boxes']:
    patch.set_facecolor('tab:orange')
    patch.set_alpha(0.6)
y_top = max((np.percentile(d, 75) if len(d) else 0) for d in data)
for r, n_b in zip(rass_vals, counts):
    ax.text(r, y_top * 1.05, f'n={n_b}', ha='center', fontsize=8)
ax.set_xlabel('RASS score (more negative = deeper sedation)')
ax.set_ylabel('Predicted Ce (µg/mL)')
ax.set_title('Predicted effect-site concentration vs measured RASS\n'
             '(propofol_only=True RASS rows, RASS > 0 excluded)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PNG_RASS, dpi=150)
plt.close()
print(f"Saved → {OUT_PNG_RASS}")
md(f'![RASS vs Ce]({os.path.basename(OUT_PNG_RASS)})\n')


# ============ WRITE REPORT ============
print(f"\nWriting markdown report to {OUT_REPORT_MD} ...")
tmp = OUT_REPORT_MD + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
os.replace(tmp, OUT_REPORT_MD)
print(f"  done.")


# ============ COMPLETION SOUND ============
try:
    import winsound
    winsound.MessageBeep(winsound.MB_OK)
except Exception:
    print('\a')

print("\nAll sanity checks done.")