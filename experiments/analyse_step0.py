"""experiments/analyse_step0.py -- read GATE 1 from the Step 0 responses.

Implements PREREG_step0_listening_frontier.md sections 3-4 (as amended A1,
2026-10-10) and nothing else, in the order the prereg states it:

  1. Unblind: join test_stim against stimuli_key.csv (the page never knew the
     condition).
  2. Exclude listeners who fail either anchor: `original` judged different on
     >= 4 of its 12 trials, or `anchor_diff` judged same on >= 4 of its 12.
  3. Validity: original same-rate >= 0.90, anchor_diff same-rate <= 0.10, and
     >= 18 listeners after exclusions. Any failure -> 1-D and the per-condition
     table is NOT printed (point estimates may not be read; RESULTS 48 is the
     precedent). `--pilot` prints it anyway, under a banner, to *_pilot.csv.
  4. Primary endpoint per condition: same-speaker rate ("same" = definitely or
     probably same), 95 % CI by cluster bootstrap over listeners (2000
     resamples), one-sided bootstrap test of H0: p <= 0.50, Holm step-down over
     the five McAdams conditions. A rung clears when its Holm-adjusted one-sided
     lower bound exceeds 0.50.
  5. Frontier = the lowest alpha that clears. alpha <= 0.85 -> 1-A;
     alpha >= 0.90 -> 1-B; none -> 1-C, with mcadams_100 saying whether the
     vocoder or the warp is responsible.
  6. Secondary readings (prereg 4.1), read by no gate.

Run:
    python experiments/analyse_step0.py
    python experiments/analyse_step0.py --pilot      # development only
Output:
    experiments/results/step0_ratings.csv, step0_conditions.csv, step0_verdict.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

LT = os.path.join(PROJ, "listening_test", "step0")
R = os.path.join(PROJ, "experiments", "results")

CONDITIONS = ["original", "anchor_diff", "tilt_6db", "mcadams_100", "mcadams_095",
              "mcadams_090", "mcadams_085", "mcadams_080", "additive_eot"]
MCADAMS = CONDITIONS[3:8]
ALPHA_OF = {"mcadams_100": 1.00, "mcadams_095": 0.95, "mcadams_090": 0.90,
            "mcadams_085": 0.85, "mcadams_080": 0.80}

# Pre-registered constants. Named so that any change shows in a diff.
MIN_LISTENERS = 18
ANCHOR_HI_MIN = 0.90
ANCHOR_LO_MAX = 0.10
ANCHOR_FAILS_TO_EXCLUDE = 4        # of 12 anchor trials (A1; was 2 of 6)
NULL_RATE = 0.50
N_BOOT = 2000
BOOT_SEED = 20261010
ALPHA_LEVEL = 0.05
FRONTIER_1A_MAX_ALPHA = 0.85
FRONTIER_1B_MIN_ALPHA = 0.90


def is_same(x):
    """1 definitely same, 2 probably same, 3 probably different, 4 definitely different."""
    return (pd.to_numeric(x, errors="coerce") <= 2).astype(float)


def cluster_boot(rates, seed=BOOT_SEED, n_boot=N_BOOT):
    x = np.asarray(rates, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    return x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)


def holm_step_down(pvals, alpha=ALPHA_LEVEL):
    """Holm (1979). {key: (level this hypothesis is tested at, rejected)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, stop = len(items), {}, False
    for rank, (k, p) in enumerate(items):
        thr = alpha / (m - rank)
        rej = (not stop) and (p <= thr)
        if not rej:
            stop = True
        out[k] = (thr, rej)
    return out


def load_responses(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as exc:
            print(f"[step0] SKIP {os.path.basename(f)}: {exc}")
            continue
        need = {"listener", "trial_id", "test_stim", "same"}
        if not need <= set(d.columns):
            print(f"[step0] SKIP {os.path.basename(f)}: missing {sorted(need - set(d.columns))}")
            continue
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    if not frames:
        return files, None
    raw = pd.concat(frames, ignore_index=True)
    raw["listener"] = raw.listener.astype(str)
    dup = raw.groupby("listener")["source_file"].nunique() > 1
    if dup.any():
        print(f"[step0] WARNING listener id(s) {list(dup[dup].index)} appear in several files; "
              f"each submission is kept separately")
        raw["listener"] = raw.listener + "@" + raw.source_file
    return files, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=os.path.join(LT, "responses"))
    ap.add_argument("--key", default=os.path.join(LT, "stimuli_key.csv"))
    ap.add_argument("--pilot", action="store_true",
                    help="print the condition table even when validity fails. DEVELOPMENT "
                         "ONLY: outputs are suffixed _pilot and no verdict is written.")
    ap.add_argument("--out-prefix", default=os.path.join(R, "step0"), dest="prefix")
    a = ap.parse_args()
    sfx = "_pilot" if a.pilot else ""

    files, raw = load_responses(a.responses)
    if raw is None:
        print(f"[step0] no usable response files in {a.responses}; nothing to analyse yet.")
        return

    key = pd.read_csv(a.key, dtype={"stim_id": str})
    raw["test_stim"] = raw.test_stim.astype(str)
    unknown = sorted(set(raw.test_stim) - set(key.stim_id))
    if unknown:
        raise SystemExit(f"[step0] {len(unknown)} test stimuli not in the key (first {unknown[0]}): "
                         f"collected against a different build. Refusing to unblind.")
    if "build_seed" in raw.columns and raw.build_seed.nunique() > 1:
        raise SystemExit(f"[step0] responses span builds {sorted(raw.build_seed.unique())}")

    df = raw.merge(key[["stim_id", "speaker", "condition"]]
                   .rename(columns={"stim_id": "test_stim"}), on="test_stim", how="left")
    df["same_bin"] = is_same(df.same)

    # -------------------------------------------------------------- exclusions
    excluded, reasons = [], {}
    for lis, g in df.groupby("listener"):
        o, d = g[g.condition == "original"], g[g.condition == "anchor_diff"]
        n_o, n_d = int((o.same_bin == 0).sum()), int((d.same_bin == 1).sum())
        why = []
        if n_o >= ANCHOR_FAILS_TO_EXCLUDE:
            why.append(f"original judged different {n_o}/{len(o)}")
        if n_d >= ANCHOR_FAILS_TO_EXCLUDE:
            why.append(f"anchor_diff judged same {n_d}/{len(d)}")
        if why:
            excluded.append(lis)
            reasons[lis] = "; ".join(why)
    kept = df[~df.listener.isin(excluded)].copy()
    n_sub, n_kept = df.listener.nunique(), kept.listener.nunique()

    print("=" * 78)
    print("STEP 0 -- THE LISTENING FRONTIER (GATE 1)")
    print("=" * 78)
    print(f"{len(files)} file(s), {n_sub} listener(s), {len(df)} trials")
    print(f"excluded {len(excluded)} on the anchor rule (>= {ANCHOR_FAILS_TO_E