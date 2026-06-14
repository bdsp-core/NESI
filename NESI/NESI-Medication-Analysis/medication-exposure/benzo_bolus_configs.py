# benzo_bolus_configs.py
"""
Configuration for benzodiazepine bolus extraction.
Lorazepam and diazepam are extracted as their own buckets; all other
benzodiazepines are extracted as a single combined 'other_benzo' bucket.

Not included here (handled elsewhere):
  - Midazolam: covered by drug_configs.py (sedative infusion + bolus pipeline)
  - Clonazepam and clobazam: covered by the long-acting ASM pipeline

These drugs are not typically administered as continuous infusions in the
populations covered by this work, so only bolus extraction is configured.
"""

BENZO_BOLUS_CONFIGS = {
    "lorazepam": {
        "name_patterns": ["lorazepam", "ativan"],
        "bolus_units": ("mg",),
    },
    "diazepam": {
        "name_patterns": ["diazepam", "valium", "diastat", "valtoco"],
        "bolus_units": ("mg",),
    },
    "other_benzo": {
        "name_patterns": [
            "alprazolam",   "xanax",     "niravam",
            "chlordiazepoxide", "librium",
            "clorazepate",  "tranxene",
            "estazolam",    "prosom",
            "flurazepam",   "dalmane",
            "oxazepam",     "serax",
            "quazepam",     "doral",
            "remimazolam",  "byfavo",
            "temazepam",    "restoril",
            "triazolam",    "halcion",
        ],
        "bolus_units": ("mg",),
    },
}