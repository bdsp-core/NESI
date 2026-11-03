"""The 24 SPECTRA severity mappings of Supplementary Table 5 (0 good, 1 moderate, 2 bad)."""

RASS = [{-1: 0, 0: 0, -3: 1, -2: 1, -5: 2, -4: 2}, {0: 0, -3: 1, -2: 1, -1: 1, -5: 2, -4: 2}]
GCS = [
    {**{s: 0 for s in (13, 14, 15)}, **{s: 1 for s in range(9, 13)}, **{s: 2 for s in range(3, 9)}},
    {**{s: 0 for s in (14, 15)}, **{s: 1 for s in range(8, 14)}, **{s: 2 for s in range(3, 8)}},
]
CAMS = [
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2},
]
ICANS = [{0: 0, 1: 1, 2: 1, 3: 2, 4: 2}, {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}]


def mapping(n):
    """Return {scale: {raw score: group}} for model Mn, numbered as in Supplementary Table 5."""
    i = n - 1
    return {"ICANS": ICANS[i % 2], "RASS": RASS[(i // 2) % 2], "GCS": GCS[(i // 4) % 2], "CAMS": CAMS[i // 8]}


def regroup(df, n):
    """GroupedScore for rows with Dataset and RawScore columns."""
    m = mapping(n)
    return [m[d][int(s)] for d, s in zip(df.Dataset, df.RawScore)]
