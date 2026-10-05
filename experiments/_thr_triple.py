"""
experiments/_thr_triple.py
==========================
Print one threshold CSV as the comma-separated triple that --thresholds expects.

Exists as a file rather than an inline heredoc because nesting a heredoc inside a
shell function inside a run script is how run_rebase_mismatch_m30.sh got mangled
on its first write. A two-line script is cheaper than that class of bug.

    python experiments/_thr_triple.py experiments/results/thresholds_clean_xsess_tc.csv
    -> 0.8926,0.3358,0.7029

The encoder ORDER is fixed and must match build_encoders(). A KeyError here means
the calibration was produced by a different ensemble than the one about to score,
which is a real error and should not be papered over with .get().
"""
import sys
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.csvio import read_csv

ORDER = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]

if __name__ == "__main__":
    d = read_csv(sys.argv[1]).set_index("encoder").threshold
    print(",".join(f"{d[e]:.4f}" for e in ORDER))
