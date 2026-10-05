"""utils/csvio.py — CSV reading that does not eat our category labels.

pandas' default NA set includes the bare strings "null", "NA", "None" and "NaN".
One of our direction families is literally called `null` (the zero-noise
pass-through criterion), so `pd.read_csv` silently turned 20 of 80 rows into NaN
and dropped them from every groupby -- the prior-art formulation, the one whose
weakness is a headline result, quietly vanishing from the analysis.

Empty fields still read as missing, which is what the experiment scripts write
for "not measured" (e.g. the skipped loop-PSR rows in the dimension sweep).
"""
import pandas as pd

_NA = ["", "NaN", "nan", "#N/A", "N/A"]


def read_csv(path, **kw):
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", _NA)
    return pd.read_csv(path, **kw)
