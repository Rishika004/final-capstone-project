"""
experiments/check_direction_sanity.py
=====================================
The instrument check RESULTS §57 makes mandatory for any new α path.

WHY IT EXISTS
-------------
#13c (§55) measured α on a new code path and produced a confident 20-of-20
verdict that was applied to the paper's framing within the hour. The path was
broken: it evaluated the identity direction at the global maximum of
cosine-to-self, where the gradient is analytically zero, so what it normalised
to unit scale was numerical residue. Its directions flip sign under an input
jitter of one part in 100 000.

The mel path, which every other α result uses, happens to avoid this because
it differentiates through a mel→waveform reconstruction rather than at the
original signal. That was an accident of the design, not a decision, and
nothing tested it for two weeks.

So: **a direction is not a measurement until it has passed these two checks.**

  stationarity   is the evaluation point at a maximum of the objective?
                 cos(f(x), f(x0)) ~ 1 and a vanishing gradient means yes, and
                 the gradient is then residue.
  stability      does the direction reproduce, (a) run to run, and (b) under a
                 tiny input jitter? A genuine descent direction is a smooth
                 function of its input; residue is not.

A trained encoder is the positive control: whatever threshold is read, it must
pass, or the check is measuring something other than what it claims.

WHAT THE SECOND AUDIT ADDED (RESULTS 58)
----------------------------------------
* Run the gate PER SPEAKER, not on one speaker. It was calibrated on speaker
  61, where every mel-path direction passes. On speaker 121 the three
  Kaldi-fbank directions (CAM++, ERes2Net, untrained CAM++) and the fbank
  front-end control fail it, the untrained one with a sign flip (-0.23).
* The cause is the front end, not precision: `torchaudio.compliance.kaldi.
  fbank` floors mel energies at float32 epsilon before the log, a clip with
  DIGITAL silence (121: 8637 consecutive zero samples, 22 % of fbank bins on
  the floor) puts the gradient on a knife-edge there, and a 1e-5 relative
  input jitter is not small for a log of zero. The float64 twin of the same
  direction agrees with float32 at cosine 0.998-0.9999, so the gradient is
  computed exactly; it is the FUNCTION that is discontinuous.
* A failed gate therefore means "no verdict from this cell", and the cell is
  reported beside any mean that includes it (`exp_encoder_subspace.py`
  records `gate_passed` on every row). It does not by itself mean the whole
  path is void: the same directions pass on the 27 speakers without digital
  silence. Only two cohort speakers have it (121, 260;
  `results/audit_digital_silence.csv`), and 260 passes at 1e-5.

USE
---
    from experiments.check_direction_sanity import check_direction

    ok, report = check_direction(lambda jitter: my_direction_fn(jitter),
                                 name="WavLM-SV / waveform path")
    print(report)
    if not ok:
        raise SystemExit("direction failed its sanity check; no verdict read")

`direction_fn(jitter)` must return `(direction_tensor, cosine_at_eval_point,
grad_norm)` and apply a relative input jitter of the given size.
"""
import os
import sys

import torch
import torch.nn.functional as F

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# Thresholds. Calibrated against the mel path, which scores repeat 1.0000 and
# jitter 0.9986-0.9999, and against the broken waveform path, which scores
# jitter -0.9975..+0.9958. Anything in between is a judgement call the caller
# should be forced to make explicitly rather than have made for it.
MIN_REPEAT = 0.99
MIN_JITTER = 0.95
MAX_COS_AT_EVAL = 0.9995      # above this, we are sitting at the maximum
JITTER = 1e-5


def _cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1).double(),
                               b.reshape(1, -1).double()).item()


def scale_to_rms(v, rms, min_rms=1e-9):
    """`v` rescaled to per-element RMS `rms`, exactly -- or a loud refusal.

    Every alpha script scales its directions with ``v / (rms(v) + 1e-12) * rms``.
    The additive epsilon is harmless for a real gradient (RMS 1e-5 .. 1e-4 in
    log-mel units) and was NOT harmless for the residue #13c measured: the
    d-vector's x0 "gradient" had RMS 4.6e-11, so the epsilon cost that arm
    1.2 % of its RMS and its band-matched control inherited the deficit
    (RESULTS 58, risk 3). A direction whose RMS is below `min_rms` is residue,
    not a measurement, and this refuses it instead of scaling it up.
    """
    cur = float(v.pow(2).mean().sqrt())
    if not cur > min_rms:
        raise ValueError(f"scale_to_rms: input RMS {cur:.3e} <= {min_rms:g}; "
                         f"this is numerical residue, not a direction")
    return v * (float(rms) / cur)


def check_direction(direction_fn, name="direction", jitter=JITTER):
    """Returns (ok, report_string). See module docstring for direction_fn."""
    g1, c1, n1 = direction_fn(0.0)
    g2, _, _ = direction_fn(0.0)
    g3, _, _ = direction_fn(jitter)
    rep, jit = _cos(g1, g2), _cos(g1, g3)

    fails = []
    if c1 > MAX_COS_AT_EVAL:
        fails.append(f"STATIONARY: cosine at the evaluation point is "
                     f"{c1:.7f} > {MAX_COS_AT_EVAL}, i.e. this is the maximum "
                     f"of the objective and the gradient there is zero by "
                     f"construction")
    if rep < MIN_REPEAT:
        fails.append(f"UNSTABLE (repeat): {rep:.4f} < {MIN_REPEAT}")
    if jit < MIN_JITTER:
        fails.append(f"UNSTABLE (jitter {jitter:g}): {jit:.4f} < {MIN_JITTER}"
                     + ("  -- the direction FLIPS SIGN" if jit < 0 else ""))

    lines = [f"[sanity] {name}",
             f"           cosine at eval point : {c1:.7f}",
             f"           gradient norm        : {n1:.3e}",
             f"           repeat reproducibility: {rep:.4f}",
             f"           jitter reproducibility: {jit:.4f}"]
    if fails:
        lines.append("           VERDICT: NOT A MEASUREMENT")
        lines += [f"             - {f}" for f in fails]
        lines.append("           Do not read a verdict from this path.")
    else:
        lines.append("           VERDICT: usable")
    return (not fails), "\n".join(lines)


if __name__ == "__main__":
    print(__doc__)
    print("This module is a library. Import check_direction from an α script.")
