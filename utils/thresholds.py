"""
utils/thresholds.py
===================
One place that decides which EER threshold belongs to which evaluation channel.

THE BUG THIS EXISTS TO PREVENT (fix S2, 2026-09-17)
---------------------------------------------------
`exp_purifier_aware.py` takes `--thresholds` once on the command line and writes
that constant into every row it produces. The main table was run with
`--eval-t 50,100,200`, so a single t=50 calibration was stamped onto rows evaluated
at t=100 and t=200 as well. The matched calibrations were on disk the whole time and
they differ:

    ECAPA-TDNN   t=50 0.3457   t=100 0.3314   t=200 0.3219
    d-vector     t=50 0.7351   t=100 0.7273   t=200 0.7228
    WavLM-SV     t=50 0.8970   t=100 0.8968   t=200 0.8959

At t=200 the stored ECAPA threshold is **0.024 too permissive, in the defender's
favour** -- the same size as every effect the paper reports (0.011-0.029). Paired
tests are immune because the threshold cancels in a difference, but every ABSOLUTE
binary protection rate at t=100 and t=200 was biased upward, and the draft's own
Sec. 2.3 says thresholds must be calibrated through the attacker's channel.

The foreign-purifier and no-purifier files were NOT affected -- each was run against
its own calibration, verified 2026-09-17.

USAGE
-----
    from utils.thresholds import rethreshold
    d = rethreshold(read_csv(".../main_table_v3.csv"), channel="diffpure_gl_ddim_v3")

`rethreshold` replaces the `threshold` column with the matched one per
(encoder, eval_t) and recomputes `margin`. It RAISES if a matched calibration is
missing rather than silently falling back -- a silent fallback is what produced the
bug in the first place.
"""
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "experiments", "results")

# channel -> (calibration file per noise level t). A channel is the FULL path the
# audio takes to the encoder at evaluation time, not just the purifier.
CHANNELS = {
    # ---- CORRECTED BASE (test-clean, cross-session, fix S1). Use these. -------
    "diffpure_gl_ddim_v7_xsess_tc": {t: f"thresholds_diffpure_gl_t{t}_ddim_v7_xsess_tc.csv"
                                     for t in (50, 100, 200)},
    "diffpure_gl_ddpm_v7_xsess_tc": {50: "thresholds_diffpure_gl_t50_v7_xsess_tc.csv"},
    "diffwave_libri_xsess_tc":      {1: "thresholds_diffwave_t1_libri_xsess_tc.csv",
                                     3: "thresholds_diffwave_t3_libri_xsess_tc.csv"},
    "diffwave_sc09_xsess_tc":       {1: "thresholds_diffwave_t1_sc09_xsess_tc.csv",
                                     3: "thresholds_diffwave_t3_sc09_xsess_tc.csv"},
    "clean_xsess_tc":               {0: "thresholds_clean_xsess_tc.csv"},
    # ---- LEGACY (dev-clean / same-session). Kept ONLY so old-base tables still
    # reproduce. RESULTS 25: a run script's file-existence guard picked up the
    # same-session diffwave files (d-vector EER 0.0009) and scored a corrected-base
    # run with them. The keys are renamed so that cannot happen by name again.
    "diffpure_gl_ddim_v3":          {t: f"thresholds_diffpure_gl_t{t}_ddim_v3.csv"
                                     for t in (50, 100, 200)},
    "diffwave_libri_LEGACY_samesession": {1: "thresholds_diffwave_t1.csv",
                                          3: "thresholds_diffwave_t3.csv"},
    "diffwave_sc09_LEGACY_samesession":  {1: "thresholds_diffwave_t1_sc09.csv",
                                          3: "thresholds_diffwave_t3_sc09.csv"},
    "none":                         {0: "thresholds_gl_only.csv"},
    "clean":                        {0: "thresholds_clean.csv"},
}


def load_thresholds(channel, t):
    """{encoder: threshold} for one channel at one noise level. Raises if absent."""
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; known: {sorted(CHANNELS)}")
    table = CHANNELS[channel]
    if t not in table:
        raise KeyError(
            f"no calibration for channel {channel!r} at t={t}. Available: "
            f"{sorted(table)}. Calibrate it with experiments/calibrate_thresholds.py "
            f"rather than reusing another t -- that is fix S2.")
    path = os.path.join(R, table[t])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing; run calibrate_thresholds.py for "
                                f"channel={channel} t={t}")
    from utils.csvio import read_csv
    return dict(read_csv(path).set_index("encoder").threshold)


def rethreshold(df, channel, t_col="eval_t", verbose=True):
    """Replace `threshold` with the calibration matched to each row's eval_t.

    Recomputes `margin` if the frame carries `cos_after_purify`. Returns a copy.
    Prints what moved, because a silent correction is how the original error
    survived five drafts."""
    d = df.copy()
    if t_col not in d.columns:
        raise KeyError(f"{t_col!r} not in frame; cannot match a calibration to a row")
    changed = []
    for t in sorted(d[t_col].unique()):
        thr = load_thresholds(channel, int(t))
        for enc, new in thr.items():
            m = (d[t_col] == t) & (d.encoder == enc)
            if not m.any():
                continue
            old = d.loc[m, "threshold"].iloc[0]
            if abs(float(old) - float(new)) > 1e-6:
                changed.append((int(t), enc, float(old), float(new)))
            d.loc[m, "threshold"] = new
    if "cos_after_purify" in d.columns:
        d["margin"] = d.cos_after_purify - d.threshold
    if verbose and changed:
        print(f"  [S2] re-thresholded {len(changed)} (t, encoder) groups on "
              f"channel {channel}:")
        for t, enc, old, new in changed:
            print(f"       t={t:<4} {enc:<22} {old:.4f} -> {new:.4f} "
                  f"({new - old:+.4f})")
    elif verbose:
        print(f"  [S2] thresholds already matched for channel {channel}")
    return d
