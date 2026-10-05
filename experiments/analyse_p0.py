"""
experiments/analyse_p0.py
=========================
Did segment-level crafting close the Resemblyzer gap?

THE PREDICTION, RECORDED BEFORE THE RESULT
------------------------------------------
Resemblyzer sat at 35-50 % clone success in the DEPLOYMENT setting while being
crafted against, where WavLM-SV and ECAPA reached 0 %. The diagnostic traced that
to the crafting objective being ~0.13 cosine away from the scoring function
(pooled embedding vs partial average). After P0 the two agree at 0.977.

  If the gap was the cause  -> the deployment Resemblyzer column drops sharply,
                               toward the 0 % the other two already reach.
  If it was not             -> the column barely moves, and the remaining cause is
                               something else (proxy front-end, or the encoder is
                               genuinely harder to attack).

Both answers are worth having. The second would mean P0 was a correctness fix
with no headline benefit, which is still worth keeping but must not be sold as a
result.

WHAT MUST NOT MOVE
------------------
WavLM-SV and ECAPA are unaffected by the partial fix (it only changes the
d-vector path) but ARE affected by --seg-partials if they expose partials. They
do not, so their numbers should be close to the pooled run. A large shift there
means something unintended changed, and the run should be investigated rather
than reported.

Run:  python experiments/analyse_p0.py
"""
import sys, io, os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import pandas as pd
from utils.csvio import read_csv

R = os.path.join(PROJ, "experiments", "results")
ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
SHORT = {"WavLM-SV": "WavLM", "ECAPA-TDNN": "ECAPA", "Resemblyzer d-vector": "d-vector"}


def get(n):
    p = os.path.join(R, n)
    return read_csv(p) if os.path.isfile(p) else None


def pre(df):
    """AntiFake pre-filter, per (encoder, purifier)."""
    out = []
    for n in df.encoder.unique():
        for p in df.purifier.unique():
            ch = df[(df.encoder == n) & (df.purifier == p)]
            ok = ch[ch.variant == "unprotected"]
            out.append(ch[ch.speaker.isin(set(ok[ok.clone_succeeds].speaker))])
    return pd.concat(out, ignore_index=True) if out else df


def rate(df, enc, pur):
    s = df[(df.encoder == enc) & (df.purifier == pur) & (df.variant != "unprotected")]
    return (100 * s.clone_succeeds.mean(), len(s)) if len(s) else (float("nan"), 0)


def block(title, pooled_file, seg_file):
    a, b = get(pooled_file), get(seg_file)
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    if a is None or b is None:
        missing = pooled_file if a is None else seg_file
        print(f"  ({missing} not present yet)")
        return
    A, B = pre(a), pre(b)
    print("  clone success %, lower = better protection\n")
    print(f"  {'purifier':<11}{'encoder':<11}{'pooled':>9}{'segments':>11}"
          f"{'change':>9}{'n':>5}")
    for p in ("none", "ours", "diffwave"):
        for e in ENC:
            ra, na = rate(A, e, p)
            rb, nb = rate(B, e, p)
            if na == 0 or nb == 0:
                continue
            mark = "  <-- the target" if e == "Resemblyzer d-vector" else ""
            print(f"  {p:<11}{SHORT[e]:<11}{ra:>8.0f}%{rb:>10.0f}%"
                  f"{rb - ra:>+8.0f}{nb // 1:>5}{mark}")


if __name__ == "__main__":
    print("P0 — did segment-level crafting close the Resemblyzer gap?")
    print("Prediction recorded before the run: the deployment d-vector column")
    print("should fall sharply; WavLM and ECAPA should barely move.")
    block("A. DEPLOYMENT (all encoders crafted) — the setting P0 targets",
          "cloning_tc_m30_deploy.csv", "cloning_tc_m30_deploy_seg.csv")
    block("B. STRICT (cloner's encoder held out) — re-run for comparability",
          "cloning_tc_m30_nodvec.csv", "cloning_tc_m30_nodvec_seg.csv")
    print("\nReading it:")
    print("  d-vector falls a lot        -> P0 was the cause. Link 5 closes in")
    print("                                 the deployment setting.")
    print("  d-vector barely moves       -> P0 was a correctness fix only. Keep it,")
    print("                                 do not sell it, and look elsewhere.")
    print("  WavLM/ECAPA move a lot      -> unintended change. Investigate before")
    print("                                 reporting anything from this run.")
