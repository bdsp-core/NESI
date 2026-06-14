"""
opiate_configs.py
=================

Per-drug configuration for the opiate INFUSION pipeline. We only process
infusions for now — boluses/injections will be handled later as part of
broader analgesia analysis.

Mirrors drug_configs.py for sedatives. Each entry defines:
  - name_patterns: substrings to match in MedicationDSC/MedicationDisplayNM
  - excluded_route_patterns: route patterns to filter out
  - excluded_name_patterns: name substrings to filter out (catches rows where
    the route field is blank but the medication name reveals the form)
  - parse_concentration: function returning concentration in the drug's mass unit per mL
  - mass_unit: 'mg' or 'mcg' — the canonical mass unit for downstream analysis
  - half_life_hours: for the PK decay model
"""

import re


# ============================================================================
# CONCENTRATION PARSERS
# ============================================================================

def _extract_mg_per_ml(text, fallback=None):
    """
    Find concentration in mg/mL. Tries:
      1. Direct: 'X mg/ml'
      2. Derived: 'X mg/Y ml' (e.g., '10 mg/5 ml' -> 2 mg/mL)
      3. 'X mg in Y ml' bag/bottle phrasing
    """
    if not isinstance(text, str):
        return fallback
    text_lower = text.lower()
    
    # Direct: "10 mg/ml"
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*ml(?!\s*/)', text_lower)
    if m:
        return float(m.group(1))
    
    # Derived: "10 mg/5 ml" -> 2 mg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return mg / ml
    
    # "X mg in Y mL" bag/bottle compound notation
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s+in\s+(?:[\w\s%.]*?\s+)?(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return mg / ml
    
    return fallback


def _extract_mcg_per_ml(text, fallback=None):
    """
    Find concentration in mcg/mL. Tries:
      1. Direct: 'X mcg/ml'
      2. Derived: 'X mcg/Y ml'
      3. Converted from 'X mg/ml': mcg/mL = 1000 × mg/mL
      4. Derived from 'X mg/Y ml': converted to mcg/mL
      5. 'X mcg in Y ml' phrasing
      6. 'X mg in Y ml' phrasing, converted to mcg/mL
    """
    if not isinstance(text, str):
        return fallback
    text_lower = text.lower()
    
    # Direct mcg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mcg\s*/\s*ml(?!\s*/)', text_lower)
    if m:
        return float(m.group(1))
    
    # Derived "X mcg/Y mL"
    m = re.search(r'(\d+(?:\.\d+)?)\s*mcg\s*/\s*(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mcg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return mcg / ml
    
    # Direct mg/mL — convert to mcg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*ml(?!\s*/)', text_lower)
    if m:
        return float(m.group(1)) * 1000
    
    # Derived "X mg/Y mL" — convert to mcg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return (mg / ml) * 1000
    
    # "X mcg in Y mL" bag/bottle notation
    m = re.search(r'(\d+(?:\.\d+)?)\s*mcg\s+in\s+(?:[\w\s%.]*?\s+)?(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mcg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return mcg / ml
    
    # "X mg in Y mL" — convert to mcg/mL
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\s+in\s+(?:[\w\s%.]*?\s+)?(\d+(?:\.\d+)?)\s*ml', text_lower)
    if m:
        mg = float(m.group(1))
        ml = float(m.group(2))
        if ml > 0:
            return (mg / ml) * 1000
    
    return fallback


def parse_morphine_concentration(med_dsc):
    """Morphine concentration in mg/mL. Common values: 1, 2, 4 mg/mL."""
    return _extract_mg_per_ml(med_dsc, fallback=1.0)


def parse_hydromorphone_concentration(med_dsc):
    """
    Hydromorphone concentration in mg/mL. Wide range 0.2 to 25 mg/mL.
    No safe fallback — return None if can't parse so we can drop the row.
    """
    return _extract_mg_per_ml(med_dsc, fallback=None)


def parse_fentanyl_concentration(med_dsc):
    """
    Fentanyl concentration in MCG/mL (since fentanyl is dosed in mcg).
    Handles all formats: mcg/mL, mg/mL (converted), and derived ratios.
    Common values: 10, 50 mcg/mL.
    """
    return _extract_mcg_per_ml(med_dsc, fallback=None)


# ============================================================================
# DRUG CONFIGS
# ============================================================================

OPIATE_CONFIGS = {
    'morphine': {
        'name_patterns': ['morphine'],
        'excluded_route_patterns': [
            'oral', 'po', 'sublingual', 'sl', 'rectal',
            'epidural', 'intrathecal', 'neuraxial', 'spinal',
        ],
        'excluded_name_patterns': [
            'controlled release', 'm/s contin', 'ms contin', 'sustained release',
            ' er ', 'oral', 'tablet', 'capsule', 'suppository',
            'epid', 'intrathecal', 'neuraxial', 'spinal', 'pca',
        ],
        'parse_concentration': parse_morphine_concentration,
        'mass_unit': 'mg',
        'half_life_hours': 3.0,
    },
    'hydromorphone': {
        'name_patterns': ['hydromorphone', 'dilaudid'],
        'excluded_route_patterns': [
            'oral', 'po', 'sublingual', 'sl', 'rectal',
            'epidural', 'intrathecal', 'neuraxial', 'spinal',
        ],
        'excluded_name_patterns': [
            'oral', 'tablet', 'capsule', 'suppository', 'liquid',
            'epid', 'intrathecal', 'neuraxial', 'spinal', 'pca',
        ],
        'parse_concentration': parse_hydromorphone_concentration,
        'mass_unit': 'mg',
        'half_life_hours': 2.5,
    },
    'fentanyl': {
        'name_patterns': ['fentanyl'],
        'excluded_route_patterns': [
            'transdermal', 'patch', 'buccal', 'intramuscular', 'im',
            'sublingual', 'sl', 'lozenge', 'lollipop',
            'epidural', 'intrathecal', 'neuraxial', 'spinal',
        ],
        'excluded_name_patterns': [
            'patch', 'transdermal', 'buccal', 'lozenge', 'lollipop',
            'spray', 'sublingual',
            'epid', 'intrathecal', 'neuraxial', 'spinal', 'pca',
        ],
        'parse_concentration': parse_fentanyl_concentration,
        'mass_unit': 'mcg',
        'half_life_hours': 3.5,
    },
}