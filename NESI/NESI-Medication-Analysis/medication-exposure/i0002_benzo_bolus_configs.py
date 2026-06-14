# i0002_benzo_bolus_configs.py
"""
I0002-specific benzodiazepine bolus configuration.

Same as benzo_bolus_configs.py (lorazepam, diazepam, other_benzo) plus
midazolam. Midazolam is added here because in I0002 it appears only as
discrete doses (no continuous infusions in this cohort), so it functions
as an IV benzo bolus rather than a sedative infusion. Downstream analyses
can still flag midazolam boluses given during midazolam infusions and
handle them separately for cohorts where infusions exist.

This is separate from benzo_bolus_configs.py to preserve I0001 pipeline
reproducibility (where midazolam is handled as a sedative via drug_configs.py).
"""

from benzo_bolus_configs import BENZO_BOLUS_CONFIGS as _BASE

I0002_BENZO_BOLUS_CONFIGS = dict(_BASE)  # copy lorazepam, diazepam, other_benzo
I0002_BENZO_BOLUS_CONFIGS["midazolam"] = {
    "name_patterns": ["midazolam", "versed", "nayzilam"],
    "bolus_units": ("mg",),
}