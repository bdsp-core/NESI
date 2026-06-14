# eleveld_propofol.py
"""
Eleveld 2018 propofol PK + effect-site model.
Hand-rolled from Eleveld DJ et al., Br J Anaesth 2018; 120:942-959.

Computes Cp (plasma) and Ce (effect-site) concentrations given a patient's
covariates and an infusion record. Uses matrix exponential for fast,
numerically exact integration of the linear 4-compartment system between
rate changes.

Reference patient: 35y male, 70 kg, 170 cm, arterial sampling, no opioids.

Usage:
    params = eleveld_params(age=35, weight=70, height=170, sex='m',
                            opioid=False, arterial=True)
    cp, ce = simulate(infusion_events, params, sample_times)

Where infusion_events is a list of (start_min, rate_mg_per_min) tuples
representing piecewise-constant infusion (use rate=0 for off, and add
boluses as a brief high-rate segment OR use add_bolus() helper).
"""

import numpy as np
from scipy.linalg import expm


# ============ TABLE 2: PK θ ============
T2 = {
    1:  6.28,    # V1 ref (L)
    2:  25.5,    # V2 ref (L)
    3:  273.0,   # V3 ref (L)
    4:  1.79,    # CL ref male (L/min)
    5:  1.75,    # Q2 ref (L/min)
    6:  1.11,    # Q3 ref (L/min)
    8:  42.3,    # CL maturation E50 (weeks PMA)
    9:  9.06,    # CL maturation slope
    10: 0.0156,  # Smaller V2 with age
    11: -0.00286,  # Lower CL with age (sign per equation: f_aging(θ11))
    12: 33.6,    # Weight for 50% of max V1 (kg)
    13: -0.0138, # Smaller V3 with age (V3 uses f_opiates(θ13) per paper)
    14: 68.3,    # Maturation of Q3 (weeks)
    15: 2.10,    # CL ref female (L/min)
    16: 1.30,    # Higher Q2 for maturation of Q3
    17: 1.42,    # V1 venous samples (children)
    18: 0.68,    # Higher Q2 venous samples
}

# ============ TABLE 3: PD θ (we only need ke0) ============
T3 = {
    2: 0.146,    # ke0 arterial (1/min) -- reference value at 35y, 70kg
    6: -0.0517,  # Increase in delay (decrease in ke0) with age (sign in eq)
    8: 1.24,     # ke0 venous
}

# ============ REFERENCE PATIENT ============
AGE_REF       = 35       # years
WGT_REF       = 70       # kg
HGT_REF       = 170      # cm
SEX_REF       = 'm'
PMA_OFFSET_WK = 40       # AGE in weeks + 40 = PMA in weeks (per paper)


# ============ HELPER FUNCTIONS (Table 4 from paper) ============
def f_aging(x, age_years):
    """exp(x * (AGE - AGE_ref))"""
    return np.exp(x * (age_years - AGE_REF))

def f_sigmoid(x, E50, lam):
    return x**lam / (x**lam + E50**lam)

def f_central(weight_kg):
    return f_sigmoid(weight_kg, T2[12], 1)

def f_CL_maturation(pma_weeks):
    return f_sigmoid(pma_weeks, T2[8], T2[9])

def f_Q3_maturation(age_years):
    pma_wk_eq = age_years * 52 + PMA_OFFSET_WK
    return f_sigmoid(pma_wk_eq, T2[14], 1)

def f_opiates(x, age_years, opioid_present):
    """1 if no opioid, else exp(x * AGE)."""
    return np.exp(x * age_years) if opioid_present else 1.0

def f_AlSallami(age_years, weight_kg, height_cm, sex):
    """Al-Sallami fat-free mass (kg)."""
    bmi = weight_kg / (height_cm / 100.0)**2
    if sex == 'm':
        maturation = 0.88 + (1 - 0.88) / (1 + (age_years / 13.4)**(-12.7))
        return maturation * (9270 * weight_kg) / (6680 + 216 * bmi)
    else:
        maturation = 1.11 + (1 - 1.11) / (1 + (age_years / 7.1)**(-1.1))
        return maturation * (9270 * weight_kg) / (8780 + 244 * bmi)


# ============ PARAMETER COMPUTATION ============
def eleveld_params(age, weight, height, sex,
                   opioid=False, arterial=True):
    """
    Compute per-patient Eleveld PK + ke0 parameters.
    Returns dict with V1, V2, V3, CL, Q2, Q3, ke0 (all SI: L, L/min, 1/min).
    """
    sex = sex.lower()
    assert sex in ('m', 'f')

    # --- Reference values (denominators in covariate eqs) ---
    fcent_ref       = f_central(WGT_REF)
    fCLmat_ref      = f_CL_maturation(AGE_REF * 52 + PMA_OFFSET_WK)
    fQ3mat_ref      = f_Q3_maturation(AGE_REF)
    fAlSal_ref      = f_AlSallami(AGE_REF, WGT_REF, HGT_REF, SEX_REF)

    # --- Patient covariate functions ---
    fcent           = f_central(weight)
    fCLmat          = f_CL_maturation(age * 52 + PMA_OFFSET_WK)
    fQ3mat          = f_Q3_maturation(age)
    fAlSal          = f_AlSallami(age, weight, height, sex)

    # --- V1 (arterial; venous adds Q17 term) ---
    V1 = T2[1] * (fcent / fcent_ref)
    if not arterial:
        V1 = V1 * (1 + T2[17] * (1 - fcent))

    # --- V2 ---
    V2 = T2[2] * (weight / WGT_REF) * f_aging(T2[10], age)

    # --- V3 (uses Al-Sallami FFM, opioid-modulated) ---
    V3 = T2[3] * (fAlSal / fAlSal_ref) * f_opiates(T2[13], age, opioid)

    # --- CL (sex-dependent) ---
    CL_ref_typical = T2[4] if sex == 'm' else T2[15]
    CL = (CL_ref_typical
          * (weight / WGT_REF)**0.75
          * (fCLmat / fCLmat_ref)
          * f_opiates(T2[11], age, opioid))

    # --- Q2 (arterial; venous multiplies by Q18) ---
    Q2 = T2[5] * (V2 / T2[2])**0.75 * (1 + T2[16] * (1 - fQ3mat))
    if not arterial:
        Q2 = Q2 * T2[18]

    # --- Q3 ---
    Q3 = T2[6] * (V3 / T2[3])**0.75 * (fQ3mat / fQ3mat_ref)

    # --- ke0 ---
    if arterial:
        # ke0 scales with weight^(-0.25), reference at 70 kg
        ke0 = T3[2] * (weight / WGT_REF)**(-0.25)
    else:
        ke0 = T3[8] * (weight / WGT_REF)**(-0.25)

    return {
        'V1': V1, 'V2': V2, 'V3': V3,
        'CL': CL, 'Q2': Q2, 'Q3': Q3,
        'ke0': ke0,
        'k10': CL / V1,
        'k12': Q2 / V1,
        'k21': Q2 / V2,
        'k13': Q3 / V1,
        'k31': Q3 / V3,
    }


# ============ ODE SYSTEM ============
def _build_A(p):
    """4x4 rate matrix for [A1, A2, A3, Ce] (amounts in mg, Ce in µg/mL)."""
    k10, k12, k21, k13, k31, ke0 = (
        p['k10'], p['k12'], p['k21'], p['k13'], p['k31'], p['ke0'])
    V1 = p['V1']
    A = np.array([
        [-(k10 + k12 + k13),     k21 * (p['V2']/p['V2']),  k31 * (p['V3']/p['V3']), 0],
        [ k12,                   -k21,                      0,                       0],
        [ k13,                    0,                       -k31,                     0],
        [ ke0 / V1,               0,                        0,                      -ke0],
    ])
    # Ce dynamics: dCe/dt = ke0*(Cp - Ce) = ke0*(A1/V1 - Ce)
    # so the [Ce, A1] coupling is ke0/V1, [Ce, Ce] is -ke0. Above is correct.
    # Note: A2, A3 are amounts; rate from comp 2 to 1 is k21*A2 directly.
    return A


def simulate(infusion_segments, params, sample_times):
    """
    Replay piecewise-constant infusion and sample Cp, Ce.

    infusion_segments : list of (t_start_min, t_end_min, rate_mg_per_min)
        Must be contiguous and sorted.
    sample_times : array of times (min) at which to return Cp, Ce.

    Returns (cp_array, ce_array) in µg/mL.
    """
    A = _build_A(params)
    state = np.zeros(4)  # [A1_mg, A2_mg, A3_mg, Ce_ug_per_mL]

    # Build event timeline: segment boundaries + sample times, sorted
    sample_times = np.asarray(sample_times, dtype=float)
    cp_out = np.full_like(sample_times, np.nan, dtype=float)
    ce_out = np.full_like(sample_times, np.nan, dtype=float)

    # Sort segments
    segs = sorted(infusion_segments, key=lambda s: s[0])

    # Walk through time. For each segment, integrate analytically:
    #   x(t+Δt) = expm(A*Δt) @ x(t) + A^{-1} @ (expm(A*Δt) - I) @ b
    # where b is the input vector (rate goes into compartment 1 only).
    t_cur = segs[0][0] if segs else 0.0
    sample_idx_sorted = np.argsort(sample_times)
    next_sample_pos = 0

    # Inverse of A (cached per call; A is non-singular for proper params)
    A_inv = np.linalg.inv(A)
    I = np.eye(4)

    for (t0, t1, rate) in segs:
        # Advance to t0 if sample falls in a gap (treat as zero rate)
        if t0 > t_cur:
            _advance_segment(state, A, A_inv, I, 0.0,
                             t_cur, t0, sample_times, sample_idx_sorted,
                             cp_out, ce_out, params)
            # update next_sample_pos handled inside _advance_segment via cp_out[idx]
            t_cur = t0
        # Now advance through this segment with rate
        _advance_segment(state, A, A_inv, I, rate,
                         t_cur, t1, sample_times, sample_idx_sorted,
                         cp_out, ce_out, params)
        t_cur = t1

    # Any remaining samples after last segment: zero infusion
    _advance_segment(state, A, A_inv, I, 0.0,
                     t_cur, max(t_cur, sample_times.max()) + 1e-9,
                     sample_times, sample_idx_sorted,
                     cp_out, ce_out, params)

    return cp_out, ce_out


def _advance_segment(state, A, A_inv, I, rate,
                     t_start, t_end, sample_times, sample_order,
                     cp_out, ce_out, params):
    """Integrate from t_start to t_end with constant infusion rate (mg/min into A1).
    Sample Cp, Ce at any sample_times falling in (t_start, t_end]."""
    b = np.array([rate, 0.0, 0.0, 0.0])
    V1 = params['V1']

    # Find samples in this interval
    in_interval = np.where(
        (sample_times >  t_start - 1e-12) &
        (sample_times <= t_end   + 1e-12)
    )[0]
    # Sort by time
    in_interval = in_interval[np.argsort(sample_times[in_interval])]

    t_cur = t_start
    cur_state = state.copy()
    for idx in in_interval:
        t_s = sample_times[idx]
        dt = t_s - t_cur
        if dt < 0:
            # Sample is before t_cur (shouldn't happen with proper sorting);
            # report current state.
            cp_out[idx] = cur_state[0] / V1
            ce_out[idx] = cur_state[3]
            continue
        if dt > 0:
            E = expm(A * dt)
            cur_state = E @ cur_state + A_inv @ (E - I) @ b
        cp_out[idx] = cur_state[0] / V1
        ce_out[idx] = cur_state[3]
        t_cur = t_s

    # Advance to end of segment
    dt_remain = t_end - t_cur
    if dt_remain > 0:
        E = expm(A * dt_remain)
        cur_state = E @ cur_state + A_inv @ (E - I) @ b

    # Write final state back
    state[:] = cur_state


# ============ CONVENIENCE: BOLUS HELPER ============
def add_bolus_segment(segments, t_min, dose_mg, duration_min=0.1):
    """Append a brief high-rate segment representing a bolus."""
    rate = dose_mg / duration_min
    segments.append((t_min, t_min + duration_min, rate))
    return segments


# ============ VALIDATION RUN ============
if __name__ == '__main__':
    # Reference scenario: 35y/m/70kg/170cm, no opioid, 2 mg/kg bolus over 6 sec.
    params = eleveld_params(age=35, weight=70, height=170, sex='m',
                            opioid=False, arterial=True)
    print("Per-patient parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    bolus_mg = 2.0 * 70  # 140 mg
    segments = []
    add_bolus_segment(segments, t_min=0.0, dose_mg=bolus_mg, duration_min=0.1)
    # Then nothing for 60 minutes (no further segments needed)

    sample_times = np.linspace(0, 60, 601)  # every 0.1 min
    cp, ce = simulate(segments, params, sample_times)

    # Landmarks
    peak_cp_i = np.nanargmax(cp)
    peak_ce_i = np.nanargmax(ce)
    print(f"\nLandmarks:")
    print(f"  Peak Cp: {cp[peak_cp_i]:.3f} µg/mL at t={sample_times[peak_cp_i]:.2f} min")
    print(f"  Peak Ce: {ce[peak_ce_i]:.3f} µg/mL at t={sample_times[peak_ce_i]:.2f} min")
    for t in [1, 2, 3, 5, 10, 15, 30]:
        i = np.argmin(np.abs(sample_times - t))
        print(f"  Cp at {t:>2} min: {cp[i]:.3f}   Ce: {ce[i]:.3f}")