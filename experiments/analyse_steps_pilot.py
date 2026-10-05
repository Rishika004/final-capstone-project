"""
experiments/analyse_steps_pilot.py
==================================
Was the d-vector margin limited by the optimisation budget, or by the method?

THE OBSERVATION THAT PROMPTED THIS
----------------------------------
After fix P0, post-purification margins (distance below the accept threshold,
more negative = better protected):

    WavLM-SV   -0.161
    ECAPA      -0.213
    d-vector   -0.073     <- clears the line by less than half

And the crafting log shows the d-vector cosine still falling at the final step:
0.72, 0.68, 0.65, 0.62, 0.61, 0.59 at steps 100..200. The optimiser was cut off
mid-descent, and because the threshold-aware LSE works on whichever encoder is
closest to accepting, the d-vector IS the binding constraint -- so its budget
limits the whole run.

WHAT THIS TELLS US
------------------
  d-vector margin improves a lot at 500 steps -> it was budget-limited. Raise
      steps for the real runs; cheap, no method change.
  margin flat                                 -> the optimiser has converged and
      more steps are wasted. The d-vector is genuinely hard to move and the fix
      is elsewhere (ensemble weights, or more encoders).

WATCH THE PERCEPTUAL COLUMNS. More steps can buy protection by spending the
perceptual budget. The projection caps loudness every step, so it should not --
but if PESQ/SNR degrade, the gain is not free and must be reported as a cost.

Run:  python experiments/analyse_steps_pilot.py
"""
import sys, io, os

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from utils.csvio import read_csv

ENC = ["WavLM-SV", "ECAPA-TDNN", "Resemblyzer d-vector"]
SH = {"WavLM-SV": "WavLM", "ECAPA-TDNN": "ECAPA", "Resemblyzer d-vector": "d-vector"}
p = os.path.join(PROJ, "experiments", "results", "_steps_pilot.csv")
if not os.path.isfile(p):
    raise SystemExit("pilot has not written results yet")
d = read_csv(p)
tags = sorted(d.variant.unique(), key=lambda t: int(t.replace("STEPS", "")))

print("POST-PURIFICATION MARGIN (cos_after - threshold). More negative = better.\n")
print(f"  {'encoder':<11}" + "".join(f"{t:>12}" for t in tags) + f"{'change':>10}")
for e in ENC:
    vals = []
    for t in tags:
        s = d[(d.variant == t) & (d.encoder == e) & (d.eval_t == 50)]
        vals.append((s.cos_after_purify - s.threshold).mean() if len(s) else float("nan"))
    ch = vals[-1] - vals[0]
    mark = "  <-- the target" if e == "Resemblyzer d-vector" else ""
    print(f"  {SH[e]:<11}" + "".join(f"{v:>12.3f}" for v in vals)
          + f"{ch:>+10.3f}{mark}")

print("\nPROTECTION RATE (margin < 0), and PERCEPTUAL COST\n")
print(f"  {'metric':<11}" + "".join(f"{t:>12}" for t in tags))
for e in ENC:
    vals = []
    for t in tags:
        s = d[(d.variant == t) & (d.encoder == e) & (d.eval_t == 50)]
        vals.append(100 * ((s.cos_after_purify - s.threshold) < 0).mean() if len(s) else float("nan"))
    print(f"  {SH[e]:<11}" + "".join(f"{v:>11.0f}%" for v in vals))
for col, lbl in (("pesq", "PESQ"), ("stoi", "STOI"), ("snr_db", "SNR dB")):
    vals = []
    for t in tags:
        s = d[d.variant == t].drop_duplicates("file")
        vals.append(s[col].mean() if len(s) else float("nan"))
    print(f"  {lbl:<11}" + "".join(f"{v:>12.3f}" for v in vals))

print("\n  Protection up AND perceptual columns flat -> the gain is free; raise")
print("  steps everywhere. Perceptual columns worse -> it was bought, report it.")
