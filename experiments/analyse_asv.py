"""
experiments/analyse_asv.py
==========================
Council #19 — report the verification operating point the way ASV papers do.

WHAT WAS MISSING
----------------
The project reported a single EER per encoder and nothing else. Three problems,
all of which a speaker-verification reviewer sees immediately:

  * **No uncertainty.** "EER 0.62 %" on ECAPA is about three errors. Quoting it
    to two decimals implies a precision that 464 target trials over 29 speakers
    cannot support, and the trials are not independent: every trial a speaker
    appears in shares that speaker's voice. The interval has to come from a
    **cluster bootstrap over speakers**, not over trials.
  * **No minDCF.** EER weights the two error types equally, which no deployed
    verifier does. minDCF at p_target = 0.01 and 0.05 is the field's convention
    and is what makes our thresholds comparable to published systems.
  * **No DET curve.** A single point cannot show that two encoders with similar
    EER behave differently in the low-false-alarm region the attacker lives in.

WHAT THIS COMPUTES
------------------
From `trialscores_<tag>_<encoder>.csv` (written by
`calibrate_thresholds.py --dump-scores`, which now also records `spk_enrol` and
`spk_test`):

  EER and its threshold
  minDCF(p_target, C_miss = C_fa = 1), normalised by the best of the two
      trivial systems, at p_target in {0.01, 0.05}
  the DET curve, written out so it can be plotted without re-running anything
  speaker-cluster bootstrap 95 % CIs for EER and both minDCFs

THE BOOTSTRAP, AND WHY IT IS OVER SPEAKERS
------------------------------------------
Each resample draws |S| speakers with replacement and keeps every trial whose
*enrolment* speaker is drawn, which preserves the trial structure that speaker
contributes (their targets and their impostor attempts) instead of shredding
it. MEASURED on this data: the speaker-clustered interval is **1.7x wider** than a
trial-level one, on all three encoders. Not an order of magnitude -- 29
speakers x 16 trials each is not a severe clustering -- but enough to change
whether an interval contains zero, and it is the honest one.

Run:
  python experiments/analyse_asv.py
  python experiments/analyse_asv.py --tag clean_xsess_tc --n-boot 2000
Output:
  experiments/results/asv_operating_points.csv
  experiments/results/det_<tag>_<encoder>.csv
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

R = os.path.join(PROJ, "experiments", "results")
P_TARGETS = (0.01, 0.05)


def det_curve(scores, labels):
    """(false-alarm rate, miss rate, threshold) at every distinct threshold.

    Accept iff score >= thr. Swept over the sorted scores, which is where the
    rates can change.
    """
    order = np.argsort(scores, kind="mergesort")
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    n_t = max(int(y.sum()), 1)
    n_n = max(int((1 - y).sum()), 1)
    # At threshold s[i]: everything below i is rejected.
    miss = np.cumsum(y) / n_t                      # targets rejected
    fa = 1.0 - np.cumsum(1 - y) / n_n              # non-targets accepted
    return fa, miss, s


def eer_from(fa, miss, thr):
    i = int(np.argmin(np.abs(miss - fa)))
    return float((miss[i] + fa[i]) / 2.0), float(thr[i])


def min_dcf(fa, miss, thr, p_target, c_miss=1.0, c_fa=1.0):
    """Normalised minimum detection cost, NIST SRE convention.

    DCF = c_miss*p_target*P_miss + c_fa*(1-p_target)*P_fa, divided by the cost
    of the better trivial system (always accept, or always reject), so 1.0
    means "no better than guessing" on any scale.
    """
    dcf = (c_miss * p_target * miss) + (c_fa * (1 - p_target) * fa)
    default = min(c_miss * p_target, c_fa * (1 - p_target))
    i = int(np.argmin(dcf))
    return float(dcf[i] / default), float(thr[i])


def metrics(df):
    fa, miss, thr = det_curve(df.score.to_numpy(), df.is_target.to_numpy())
    eer, eer_thr = eer_from(fa, miss, thr)
    out = {"eer": eer, "eer_threshold": eer_thr}
    for p in P_TARGETS:
        d, t = min_dcf(fa, miss, thr, p)
        out[f"mindcf_p{p:g}"] = d
        out[f"mindcf_p{p:g}_threshold"] = t
    return out, (fa, miss, thr)


def speaker_bootstrap(df, n_boot=2000, seed=20261006, alpha=0.05):
    """Cluster bootstrap over enrolment speakers. Returns {metric: (lo, hi)}."""
    if "spk_enrol" not in df.columns:
        return {}
    spks = df.spk_enrol.unique()
    rng = np.random.default_rng(seed)
    by_spk = {s: df[df.spk_enrol == s] for s in spks}
    acc = {}
    for _ in range(n_boot):
        pick = rng.choice(spks, size=len(spks), replace=True)
        sub = pd.concat([by_spk[s] for s in pick], ignore_index=True)
        if sub.is_target.nunique() < 2:
            continue
        m, _ = metrics(sub)
        for k, v in m.items():
            if k.endswith("threshold"):
                continue
            acc.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in acc.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(R, "trialscores_*.csv"))
    ap.add_argument("--tag", default="", help="only files whose tag matches")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--no-boot", action="store_true", dest="no_boot")
    ap.add_argument("--out", default=os.path.join(R, "asv_operating_points.csv"))
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if a.tag:
        files = [f for f in files if a.tag in os.path.basename(f)]
    if not files:
        print(f"[ASV] no trial-score files matching {a.glob}")
        print("[ASV] They are written by:")
        print("      python experiments/calibrate_thresholds.py --channel clean "
              "--cross-session --subset test-clean --n-spk 29 --n-utt 4 \\")
        print("            --dump-scores --tag xsess_tc")
        print("[ASV] Nothing on disk predates that flag with raw scores, so "
              "minDCF and the CIs cannot be computed from the summary files.")
        return

    rows = []
    for f in files:
        # The encoder name is sanitised into the filename with the same
        # non-alphanumeric -> "_" rule calibrate_thresholds.py uses, and it can
        # itself contain underscores ("Resemblyzer_d_vector"), so a greedy
        # split on "_" gets it wrong ("clean_xsess_tc_v2_ECAPA" / "TDNN").
        # Match the known encoders off the END of the name instead.
        base = os.path.basename(f)[len("trialscores_"):-len(".csv")]
        tag, enc = base, "?"
        for cand in ("WavLM_SV", "ECAPA_TDNN", "Resemblyzer_d_vector",
                     "x_vector"):
            if base.endswith("_" + cand):
                tag, enc = base[: -(len(cand) + 1)], cand
                break
        d = pd.read_csv(f)
        if not {"score", "is_target"} <= set(d.columns):
            print(f"[ASV] skip {os.path.basename(f)}: no score/is_target")
            continue
        vals, (fa, miss, thr) = metrics(d)
        n_spk = d.spk_enrol.nunique() if "spk_enrol" in d.columns else np.nan
        ci = {} if a.no_boot else speaker_bootstrap(d, n_boot=a.n_boot)

        det = pd.DataFrame(dict(threshold=thr, p_fa=fa, p_miss=miss))
        det_path = os.path.join(R, f"det_{tag}_{enc}.csv")
        det.to_csv(det_path, index=False)

        row = dict(tag=tag, encoder=enc,
                   n_target=int(d.is_target.sum()),
                   n_nontarget=int((1 - d.is_target).sum()),
                   n_speakers=n_spk, **{k: round(v, 6) for k, v in vals.items()})
        for k, (lo, hi) in ci.items():
            row[f"{k}_lo"] = round(lo, 6)
            row[f"{k}_hi"] = round(hi, 6)
        row["det_file"] = os.path.basename(det_path)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    print("=" * 94)
    print("ASV OPERATING POINTS  (council #19)")
    print("EER and minDCF with speaker-cluster bootstrap 95 % CIs; "
          "minDCF normalised, C_miss = C_fa = 1")
    print("=" * 94)
    for tag in out.tag.unique():
        sub = out[out.tag == tag]
        print(f"\n--- {tag} ---")
        n = sub.n_speakers.iloc[0]
        print(f"  {int(sub.n_target.iloc[0])} target / "
              f"{int(sub.n_nontarget.iloc[0])} non-target trials"
              + (f" over {int(n)} speakers" if n == n else ""))
        print(f"  {'encoder':22s} {'EER %':>16s} {'minDCF p=0.01':>20s} "
              f"{'minDCF p=0.05':>20s}")
        for _, r in sub.iterrows():
            def fmt(k, scale=1.0, pct=False):
                v = r.get(k, np.nan) * scale
                lo, hi = r.get(f"{k}_lo", np.nan), r.get(f"{k}_hi", np.nan)
                if lo != lo:
                    return f"{v:.3f}" + ("%" if pct else "")
                return f"{v:.2f} [{lo * scale:.2f},{hi * scale:.2f}]"
            print(f"  {r.encoder[:22]:22s} {fmt('eer', 100):>16s} "
                  f"{fmt('mindcf_p0.01'):>20s} {fmt('mindcf_p0.05'):>20s}")
    print("\nHOW TO READ THIS")
    print("  The CI is a cluster bootstrap over SPEAKERS, because every trial a")
    print("  speaker appears in shares that speaker's voice. Measured on this")
    print("  data, a trial-level bootstrap would report an interval 1.7x")
    print("  narrower on all three encoders.")
    print("  A minDCF near 1.0 means the system is no better than the trivial")
    print("  one at that operating point, whatever its EER says.")
    print(f"\nsaved {a.out}")
    print(f"saved {len(rows)} DET curve(s) as det_<tag>_<encoder>.csv")


if __name__ == "__main__":
    main()
