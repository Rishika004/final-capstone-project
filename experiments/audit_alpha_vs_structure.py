"""
experiments/audit_alpha_vs_structure.py
=======================================
Audit 2, risk 3 (RESULTS section 58): does alpha track how CONCENTRATED a
direction is, across every direction the mel path has measured?

Every alpha section equalises the global per-element RMS of the directions in
a cell. `audit_direction_stability.py` measured what that leaves unequal: an
encoder direction keeps most of its energy in a few per cent of its elements,
its band-matched control does not, isotropic noise least of all. If alpha is
monotone in that concentration across ALL direction kinds -- controls and
targets alike -- then "the encoder direction survives worse than its control"
and "the spikier direction survives worse" are the same observation, and the
C1 contrast has a confound that equal RMS does not remove.

Joins:
  results/encoder_subspace_controlb.csv                 alpha, 29 speakers x 18 directions
  results/audit_direction_stability_mel29_structure.csv the same directions' statistics
                                                       (same seeds, so the same vectors)
Reports Spearman rank correlations of alpha with the concentration statistics
over all cells, within each direction (across speakers) and within each
speaker (across directions), plus the per-direction means side by side.

    python experiments/audit_alpha_vs_structure.py
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "experiments", "results")


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan"), float("nan")
    r = spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def main():
    a = pd.read_csv(os.path.join(R, "encoder_subspace_controlb.csv"))
    s = pd.read_csv(os.path.join(R, "audit_direction_stability_mel29_structure.csv"))
    s = s[s.path == "mel"].copy()
    s["direction"] = np.where(s.kind == "target", s.encoder,
                              np.where(s.kind == "band_matched", "band_matched:" + s.encoder,
                                       s.kind))
    # pca_span / random were re-drawn once per encoder but are the same vector
    # (same seed) -- keep one copy
    s = s.drop_duplicates(["speaker", "direction"])
    s["log_kurt"] = np.log10(s["kurtosis"])
    j = a.merge(s[["speaker", "direction", "kind", "top5_energy", "frame_cv",
                   "band_cv", "log_kurt", "peak_over_rms", "corr_with_logmel"]],
                on=["speaker", "direction"], how="inner")
    print(f"joined {len(j)} cells: {j.speaker.nunique()} speakers x "
          f"{j.direction.nunique()} directions "
          f"(alpha rows {len(a)}, structure rows {len(s)})\n")

    stats = ["top5_energy", "log_kurt", "frame_cv", "peak_over_rms"]
    for col in ("alpha_t50", "alpha_t200"):
        print("=" * 100)
        print(f"{col}: Spearman rho of alpha with each concentration statistic")
        print("=" * 100)
        print(f"  {'scope':46s} " + " ".join(f"{c:>16s}" for c in stats))
        line = f"  {'ALL cells (n=%d)' % len(j):46s} "
        for c in stats:
            r, p = spearman(j[col], j[c])
            line += f" {r:+.3f} (p={p:.0e})".rjust(17)
        print(line)
        # within each direction, across speakers
        rows = []
        for d, g in j.groupby("direction", sort=False):
            rows.append([d] + [spearman(g[col], g[c])[0] for c in stats])
        print(f"\n  within each direction, across {j.speaker.nunique()} speakers:")
        for r_ in rows:
            print(f"  {r_[0][:46]:46s} " + " ".join(f"{v:+16.3f}" for v in r_[1:]))
        print(f"  {'MEAN within-direction rho':46s} " + " ".join(
            f"{np.nanmean([r_[k + 1] for r_ in rows]):+16.3f}" for k in range(len(stats))))
        # within each speaker, across the 18 directions
        ws = {c: [] for c in stats}
        for spk, g in j.groupby("speaker"):
            for c in stats:
                ws[c].append(spearman(g[col], g[c])[0])
        print(f"  {'MEAN within-speaker rho (18 directions)':46s} " + " ".join(
            f"{np.nanmean(ws[c]):+16.3f}" for c in stats))
        print(f"  {'  ... speakers with rho < 0':46s} " + " ".join(
            f"{int(np.sum(np.array(ws[c]) < 0)):>13d}/{len(ws[c])}" for c in stats))
        print()

    print("=" * 100)
    print("per-direction means, sorted by concentration (top-5 % energy share)")
    print("=" * 100)
    m = j.groupby("direction").agg(alpha_t50=("alpha_t50", "mean"), alpha_t200=("alpha_t200", "mean"),
                                   top5=("top5_energy", "mean"), log_kurt=("log_kurt", "mean"),
                                   frame_cv=("frame_cv", "mean"), peak_rms=("peak_over_rms", "mean"),
                                   kind=("kind", "first")).sort_values("top5")
    print(m.to_string(float_format=lambda v: f"{v:.3f}"))
    m.to_csv(os.path.join(R, "audit_alpha_vs_structure.csv"))
    print(f"\nwritten: {os.path.join(R, 'audit_alpha_vs_structure.csv')}")


if __name__ == "__main__":
    main()
