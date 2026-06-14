# i0002_asm_antipsych_configs.py
"""
Name patterns for long-acting ASMs and antipsychotics, used for I0002
ASM/antipsychotic exposure extraction.

Each category is a flat list of generic and brand name keywords. Drug-name
matching uses substring match against ProductDescription in the I0002
pivoted parquet.
"""

LONG_ACTING_ASMS = [
    "levetiracetam", "keppra",
    "lacosamide", "vimpat",
    "phenytoin", "dilantin",
    "fosphenytoin", "cerebyx",
    "valproic", "valproate", "divalproex", "depakote", "depakene", "depacon",
    "phenobarbital", "luminal",
    "brivaracetam", "briviact",
    "carbamazepine", "tegretol",
    "oxcarbazepine", "trileptal",
    "lamotrigine", "lamictal",
    "topiramate", "topamax",
    "zonisamide", "zonegran",
    "gabapentin", "neurontin",
    "pregabalin", "lyrica",
    "clobazam", "onfi",
    "clonazepam", "klonopin",
]

ANTIPSYCHOTICS = [
    "quetiapine", "seroquel",
    "olanzapine", "zyprexa",
    "risperidone", "risperdal",
    "haloperidol", "haldol",
    "aripiprazole", "abilify",
    "ziprasidone", "geodon",
    "chlorpromazine", "thorazine",
    "clozapine", "clozaril",
]

CATEGORY_MAP = {
    "long_acting_asm": LONG_ACTING_ASMS,
    "antipsychotic":   ANTIPSYCHOTICS,
}