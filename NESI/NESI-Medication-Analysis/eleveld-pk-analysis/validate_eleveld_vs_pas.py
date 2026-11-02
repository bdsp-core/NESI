# validate_eleveld_vs_pas.py
"""
Compare eleveld_propofol.py against an independent Eleveld 2018 implementation
(python-anesthesia-simulator, PAS) and SAVE the comparison.

96 scenarios: age (20, 50, 80, 90) x sex x body size (55/155, 80/178, 120/170 kg/cm)
x opioid flag x 2 dosing schedules:
  infusion:          2 mg/kg/h 0-120 min, 1 mg/kg/h 120-240 min, off to 300 min
  bolus_infusion:    1 mg/kg bolus over 6 s at t=0, then the same infusion steps,
                     plus a 0.5 mg/kg bolus over 6 s at t=180 min on top of the infusion
Cp and Ce sampled every 6 s (0.1 min) from both implementations.

Outputs (in ELEVELD_VALIDATION_DIR from med_config.py):
  eleveld_vs_pas_timeseries.csv  every scenario x 6-s sample: Cp/Ce from both + % difference
  eleveld_vs_pas_summary.csv     one row per scenario: MDPE, MDAPE, max |% diff| for Cp and Ce, peak Ce
  eleveld_vs_pas_overlay.png     Ce curves for 4 representative bolus+infusion scenarios
  eleveld_vs_pas_agreement.csv   pooled agreement: MDPE (bias), MDAPE, 95th pct and max |% error|,
                                 for all samples and by schedule and opioid flag
  eleveld_vs_pas_report.md       short text summary for methods/supplement

Setup (once, in bdsp):  python -m pip install -U control matplotlib
                        python -m pip install python-anesthesia-simulator --no-deps
"""
import importlib.util
import itertools
import os
import sys
import sysconfig

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import eleveld_propofol
from eleveld_propofol import eleveld_params, simulate
# Repo layout: med_config.py lives in ../medication-exposure (no effect if already importable)
import sys as _sys
from pathlib import Path as _Path
_MC = _Path(__file__).resolve().parent.parent / "medication-exposure"
if _MC.is_dir() and str(_MC) not in _sys.path:
    _sys.path.insert(0, str(_MC))
from med_config import ELEVELD_VALIDATION_DIR

# (start min, end min, mg/kg/h); boluses are 6-s segments (1 mg/kg over 0.1 min = 600 mg/kg/h)
SCHEDULES = {
    'infusion': [(0, 120, 2.0), (120, 240, 1.0), (240, 300, 0.0)],
    'bolus_infusion': [(0, 0.1, 600.0), (0.1, 120, 2.0), (120, 180, 1.0),
                       (180, 180.1, 300.0 + 1.0), (180.1, 240, 1.0), (240, 300, 0.0)],
}
T_SAMPLE = np.round(np.arange(0.1, 299.95, 0.1), 1)            # every 6 s
AGES = [20, 50, 80, 90]
SEXES = ['m', 'f']
BODIES = [(55, 155), (80, 178), (120, 170)]
OPIOID = [False, True]
CMIN = 0.05            # µg/mL; exclude near-zero values from % difference


def load_pas():
    path = os.path.join(sysconfig.get_paths()['purelib'],
                        'python_anesthesia_simulator', 'pk_models.py')
    if not os.path.exists(path):
        sys.exit(f'PAS not installed. Run: "{sys.executable}" -m pip install '
                 'python-anesthesia-simulator --no-deps')
    spec = importlib.util.spec_from_file_location('pas_pk_models', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atomic_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_scenario(pas, age, sex, wt, ht, opioid, schedule):
    segs = [(a, b, r * wt / 60.0) for a, b, r in schedule]           # mg/min
    cp, ce = simulate(segs, eleveld_params(age, wt, ht, sex, opioid=opioid), T_SAMPLE)
    ref = pas.CompartmentModel([age, ht, wt, 1 if sex == 'm' else 0], lbm=0,
                               drug='Propofol', model='Eleveld', ts=1,
                               opiate=opioid, measurement='arterial')
    u = np.concatenate([np.full(int(round((b - a) * 60)), r * wt / 3600.0)   # mg/s, 1-s steps
                        for a, b, r in schedule])
    x = ref.full_sim(u)
    idx = np.round(T_SAMPLE * 60).astype(int)
    return cp, ce, x[0, idx], x[3, idx]


def pct_diff(mine, ref):
    return np.where(ref > CMIN, 100 * (mine / ref - 1), np.nan)


def main():
    os.makedirs(ELEVELD_VALIDATION_DIR, exist_ok=True)
    pas = load_pas()

    rows = []
    for sid, (sched, age, sex, (wt, ht), op) in enumerate(
            itertools.product(SCHEDULES, AGES, SEXES, BODIES, OPIOID), start=1):
        cp, ce, cp_ref, ce_ref = run_scenario(pas, age, sex, wt, ht, op, SCHEDULES[sched])
        rows.append(pd.DataFrame(dict(
            scenario_id=sid, schedule=sched, age=age, sex=sex,
            weight_kg=wt, height_cm=ht, opioid=op,
            t_min=T_SAMPLE, cp_mine=cp, cp_pas=cp_ref, ce_mine=ce, ce_pas=ce_ref,
            cp_pct_diff=pct_diff(cp, cp_ref), ce_pct_diff=pct_diff(ce, ce_ref))))
    ts = pd.concat(rows, ignore_index=True)

    summary = (ts.groupby(['scenario_id', 'schedule', 'age', 'sex',
                           'weight_kg', 'height_cm', 'opioid'])
                 .agg(cp_mdpe=('cp_pct_diff', 'median'),
                      cp_mdape=('cp_pct_diff', lambda s: s.abs().median()),
                      cp_max_abs_pct_diff=('cp_pct_diff', lambda s: s.abs().max()),
                      ce_mdpe=('ce_pct_diff', 'median'),
                      ce_mdape=('ce_pct_diff', lambda s: s.abs().median()),
                      ce_max_abs_pct_diff=('ce_pct_diff', lambda s: s.abs().max()),
                      ce_peak_mine=('ce_mine', 'max'),
                      ce_peak_pas=('ce_pas', 'max'))
                 .reset_index())

    atomic_csv(ts, os.path.join(ELEVELD_VALIDATION_DIR, 'eleveld_vs_pas_timeseries.csv'))
    atomic_csv(summary, os.path.join(ELEVELD_VALIDATION_DIR, 'eleveld_vs_pas_summary.csv'))

    # Overlay plot: 4 representative scenarios, bolus + infusion schedule
    picks = [(20, 'm', 80, False), (20, 'm', 80, True), (80, 'f', 55, False), (80, 'f', 55, True)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (age, sex, wt, op), color in zip(picks, ['C0', 'C1', 'C2', 'C3']):
        d = ts[(ts.schedule == 'bolus_infusion') & (ts.age == age) & (ts.sex == sex)
               & (ts.weight_kg == wt) & (ts.opioid == op)]
        lab = f"{age}y {sex} {wt} kg, opioid {'on' if op else 'off'}"
        ax.plot(d.t_min, d.ce_pas, color=color, lw=3, alpha=0.35, label=f"{lab} (PAS)")
        ax.plot(d.t_min, d.ce_mine, color=color, lw=1, ls='--', label=f"{lab} (ours)")
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Propofol Ce (µg/mL)')
    ax.set_title('Eleveld 2018 (bolus + infusion): ours vs python-anesthesia-simulator')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png = os.path.join(ELEVELD_VALIDATION_DIR, 'eleveld_vs_pas_overlay.png')
    fig.savefig(png + '.tmp.png', dpi=150)
    os.replace(png + '.tmp.png', png)
    plt.close(fig)

    worst_cp = summary.cp_max_abs_pct_diff.max()
    worst_ce = summary.ce_max_abs_pct_diff.max()

    # Pooled agreement (Varvel 1992 performance-error metrics, PAS as reference):
    #   MDPE  = median % error (bias); MDAPE = median |% error| (inaccuracy)
    #   plus 95th percentile and maximum |% error|, and maximum absolute difference in µg/mL
    def pooled(g, c):
        pe = g[f'{c}_pct_diff'].dropna()
        ad = (g[f'{c}_mine'] - g[f'{c}_pas']).abs()
        return {f'{c}_MDPE_%': pe.median(), f'{c}_MDAPE_%': pe.abs().median(),
                f'{c}_P95_abs_%': pe.abs().quantile(0.95), f'{c}_max_abs_%': pe.abs().max(),
                f'{c}_max_abs_ugml': ad.max(), f'{c}_n': len(pe)}
    agree = pd.DataFrame([{'subset': name, **pooled(g, 'cp'), **pooled(g, 'ce')}
                          for name, g in [('all', ts)] + list(ts.groupby('schedule'))
                          + [(f'opioid={k}', g) for k, g in ts.groupby('opioid')]])
    atomic_csv(agree, os.path.join(ELEVELD_VALIDATION_DIR, 'eleveld_vs_pas_agreement.csv'))
    al = agree.iloc[0]

    per_sched = "".join(
        f"- Max |% diff| ({k}): Cp {g.cp_max_abs_pct_diff.max():.3f}%, "
        f"Ce {g.ce_max_abs_pct_diff.max():.3f}%\n" for k, g in summary.groupby('schedule'))
    report = (
        "# Eleveld implementation verification vs python-anesthesia-simulator\n\n"
        f"- Implementation file: `{eleveld_propofol.__file__}`\n"
        f"- theta10 (V2 age term): {eleveld_propofol.T2[10]}\n"
        f"- Scenarios: {len(summary)} (age {AGES} x sex x {len(BODIES)} body sizes x "
        f"opioid off/on x schedules {list(SCHEDULES)})\n"
        "- infusion: 2 mg/kg/h 0-120 min, 1 mg/kg/h 120-240 min, off to 300 min\n"
        "- bolus_infusion: 1 mg/kg bolus (6 s) at 0, same infusion, "
        "plus 0.5 mg/kg bolus (6 s) at 180 min\n"
        "- Sampled every 6 s\n"
        + per_sched +
        f"- Pooled agreement, Ce (n={al.ce_n:.0f} samples): MDPE {al['ce_MDPE_%']:.3f}%, "
        f"MDAPE {al['ce_MDAPE_%']:.3f}%, 95th pct |error| {al['ce_P95_abs_%']:.3f}%, "
        f"max |error| {al['ce_max_abs_%']:.3f}% ({al.ce_max_abs_ugml:.4f} µg/mL)\n"
        f"- Pooled agreement, Cp (n={al.cp_n:.0f} samples): MDPE {al['cp_MDPE_%']:.3f}%, "
        f"MDAPE {al['cp_MDAPE_%']:.3f}%, 95th pct |error| {al['cp_P95_abs_%']:.3f}%, "
        f"max |error| {al['cp_max_abs_%']:.3f}% ({al.cp_max_abs_ugml:.4f} µg/mL)\n"
        f"- Max |% diff| overall: Cp {worst_cp:.3f}%, Ce {worst_ce:.3f}%\n"
        f"- Differences > 0.5%: {'NONE' if max(worst_cp, worst_ce) < 0.5 else 'YES - INVESTIGATE'}\n")
    md = os.path.join(ELEVELD_VALIDATION_DIR, 'eleveld_vs_pas_report.md')
    with open(md + '.tmp', 'w', encoding='utf-8') as f:
        f.write(report)
    os.replace(md + '.tmp', md)

    print(report)
    print(f"Files saved to: {ELEVELD_VALIDATION_DIR}")
    return max(worst_cp, worst_ce) < 0.5


if __name__ == '__main__':
    try:
        ok = main()
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK if ok else winsound.MB_ICONHAND)
        except Exception:
            print('\a')
    except Exception:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            print('\a')
        raise