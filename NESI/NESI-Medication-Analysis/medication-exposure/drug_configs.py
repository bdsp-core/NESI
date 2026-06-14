"""
drug_configs.py
================

Per-drug configuration for sedative exposure pipeline. Each entry defines:
  - name_patterns: substrings to match in MedicationDSC/MedicationDisplayNM
  - excluded_routes: route patterns to filter out (e.g., nasal, sublingual)
  - parse_concentration: function that takes MedicationDSC and returns 
    concentration in the drug's working mass unit per mL
  - mass_unit: 'mg' or 'mcg' — the unit boluses and exposure are tracked in
  - half_life_hours: for the PK decay model
  
The infusion reconstruction logic is drug-agnostic and lives in 
infusion_reconstruction.py. Drug-specific behavior is encoded here.
"""

import re
import numpy as np


# ============================================================================
# CONCENTRATION PARSERS
# ============================================================================

def _extract_mg_per_ml(text, fallback=None):
    """
    Try to find an "X mg/ml" or "X mg/mL" pattern in the text. Returns float.
    Falls back to `fallback` if no match.
    """
    if not isinstance(text, str):
        return fallback
    text_lower = text.lower()
    # Pattern matches: "10 mg/ml", "10mg/ml", "10 mg/mL", with possible decimal
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*ml', text_lower)
    if m:
        return float(m.group(1))
    return fallback


def _extract_mcg_per_ml(text, fallback=None):
    """
    Try to find "X mcg/ml" pattern, OR derive from "X mcg/Y ml" or 
    "X mcg/Y mL" patterns.
    """
    if not isinstance(text, str):
        return fallback
    text_lower = text.lower()
    
    # Direct: "4 mcg/ml"
    m = re.search(r'(\d+(?:\.\d+)?)\s*mcg\s*/\s*ml(?!\s*/)', text_lower)
    if m:
        return float(m.group(1))
    
    # Derived: "200 mcg/2 ml" → 100 mcg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mcg\s*/\s*(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mcg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return mcg / ml
    
    return fallback


def parse_propofol_concentration(med_dsc):
    """All propofol formulations are 10 mg/mL."""
    return 10.0


def parse_midazolam_concentration(med_dsc):
    """Concentration parsed from MedicationDSC; falls back to 1 mg/mL (most common)."""
    return _extract_mg_per_ml(med_dsc, fallback=1.0)


def parse_ketamine_concentration(med_dsc):
    """Concentration parsed from MedicationDSC; falls back to 10 mg/mL."""
    return _extract_mg_per_ml(med_dsc, fallback=10.0)


def parse_precedex_concentration(med_dsc):
    """
    Concentration in MCG/mL (because precedex is dosed in mcg).
    Falls back to 4 mcg/mL (premix, most common).
    """
    return _extract_mcg_per_ml(med_dsc, fallback=4.0)


# ============================================================================
# DRUG CONFIGS
# ============================================================================
DRUG_CONFIGS = {
    'propofol': {
        'name_patterns': ['propofol', 'diprivan'],
        'excluded_route_patterns': [],
        'parse_concentration': parse_propofol_concentration,
        'bolus_units': ('mg',),
        'half_life_hours': 5.0,
    },
    'midazolam': {
        'name_patterns': ['midazolam', 'versed'],
        'excluded_route_patterns': ['nasal', 'intranasal', 'spray', 'oral', 'buccal'],
        'parse_concentration': parse_midazolam_concentration,
        'bolus_units': ('mg',),
        'half_life_hours': 7.0,
    },
    'ketamine': {
        'name_patterns': ['ketamine', 'ketalar'],
        'excluded_route_patterns': ['nasal', 'intranasal'],
        'parse_concentration': parse_ketamine_concentration,
        'bolus_units': ('mg', 'mg/kg'),   # both legitimate
        'half_life_hours': 3.5,
    },
    'dexmedetomidine': {
        'name_patterns': ['dexmedetomidine', 'precedex'],
        'excluded_route_patterns': ['sublingual', 'sl', 'oral'],
        'parse_concentration': parse_precedex_concentration,
        'bolus_units': ('mcg',),
        'half_life_hours': 2.5,
    },
}