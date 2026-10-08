"""
experiments/exp_encoder_subspace_wav.py
=======================================
Council #13c -- alpha measured through DiffWave, in the waveform domain.

THE HOLE THIS CLOSES
--------------------
C1's alpha is measured on OUR purifier, a mel-domain denoiser (v7). The
phenomenon C1 exists to explain -- the invariance constraint's encoder-
dependent effect, and the d-vector reversal -- lives on DiffWave at t = 1 and
t = 3, which is a WAVEFORM model. Explanation and explanandum sit on different
purifiers, in different representations. `PLAN_V3` W7 and council v4 §E.7 both
call this the only structural hole left in the lead contribution, and R1's
phrasing is that a reviewer sees it in thirty seconds.

WHAT IS MEASURED
----------------
Exactly the quantity RESULTS §32 measures, one domain over:

    alpha(g_e) = <R(x + d) - R(x), d> / ||d||^2

with x a WAVEFORM, R a DiffWave purifier at its evaluation timestep, and d the
encoder's identity direction **in the waveform domain** -- the gradient of
cos(E(x), E(x_0)) with respect to the samples, which every encoder here
supports because all four front ends (WavLM raw audio, ECAPA fbank, Resemblyzer
mel, 3D-Speaker Kaldi fbank) are differentiable to the waveform.

The same `survival()` used by §32 does the arithmetic, so the two alphas are
computed by identical code and differ only in R and in the domain of d.

CONTROLS
--------
`band_matched` -- white noise given the direction's per-STFT-bin magnitude
profile and its exact norm (core.matched_random.band_matched_wav). This is the
waveform analogue of the control that carries C1, and it is the one that
answers "the purifier just removes high frequencies, and delta is bandlimited
to 6 kHz".

`random` -- isotropic, norm-matched.

`pca_span` has NO honest waveform analogue (the mel basis is over log-mel
frames), so it is deliberately absent rather than faked. #13c reports the two
controls it can construct.

WHAT THIS CANNOT DO
-------------------
DiffWave at t = 1-3 is close to the identity plus a little noise, so alpha
here will sit much nearer 1 than the mel-purifier alphas. **The levels are not
comparable across purifiers and no row may be quoted as if they were.** What
IS comparable, and what #13c is for, is the CONTRAST within this purifier:
does each encoder's direction still sit below its own band-matched control?

Run:
    python experiments/exp_encoder_subspace_wav.py --help
    bash experiments/run_13c.sh
"""
import argparse
import os
import sys

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from core.matched_random import band_matched_wav  # noqa: E402
from experiments.exp_nullspace_survival import survival  # noqa: E402
from experiments.librispeech import (cross_session_split,  # noqa: E402
                                     list_librispeech, load_flac,
                                     reference_items)
from utils.provenance import add_prereg_arg, provenance  # noqa: E402


def append_rows(path, rows):
    """Flush incrementally, so a stall leaves usable partial output.

    The mel version writes once at the end; this run is longer per speaker
    (two purifier checkpoints x two timesteps x two draws), and the GPU here
    has a documented habit of stalling, so partial output is worth the
    re-read.
    """
    import pandas as pd
    if not rows:
        return
    df = pd.DataFrame(rows)
    hdr = not os.path.isfile(path)
    df.to_csv(path, mode="a", header=hdr, index=False)

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(R, "encoder_subspace_wav.csv")
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--cross-session", action="store_true",
                    dest="cross_session", default=True)
    ap.add_argument("--utt-index", type=int, default=0, dest="utt_index")
    ap.add_argument("--max-sec", type=float, default=4.0, dest="max_sec")
    ap.add_argument("--diffwave-ckpt", required=True, dest="diffwave_ckpt",
                    help="foreign waveform purifier checkpoint")
    ap.add_argument("--purifier-name", default="", dest="purifier_name",
                    help="label for the CSV, e.g. DiffWave-LibriSpeech")
    ap.add_argument("--t", default="1,3",
                    help="DiffWave timesteps; these are the EVALUATION "
                         "timesteps the protection results use")
    ap.add_argument("--seeds", default="2026,7",
                    help="purifier draws averaged per measurement")
    ap.add_argument("--rel-rms", type=float, default=0.05, dest="rel_rms",
                    help="direction RMS as a fraction of the utterance RMS. "
                         "alpha is only scale-free for a linear operator, and "
                         "DiffWave is not, so this is recorded on every row.")
    ap.add_argument("--with-extra", action="store_true", dest="with_extra",
                    help="also measure CAM++ and ERes2Net (eval-only)")
    ap.add_argument("--out", default=OUT)
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in a.t.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]
    pname = a.purifier_name or os.path.basename(a.diffwave_ckpt)
    print(f"[13c] code_sha={PROV['code_sha']} prereg_sha={PROV['prereg_sha'] or '-'}")
    print(f"[13c] purifier={pname} t={ts} seeds={seeds} rel_rms={a.rel_rms}")

    from experiments.exp_purifier_aware import build_encoders
    from models.diffwave_purifier import load_diffwave

    encs = list(build_encoders(device, with_xvector=False))
    if a.with_extra:
        from experiments.extra_encoders import build_extra
        encs = encs + list(build_extra(device))
    print(f"[13c] encoders: {[e.name for e in encs]}")

    dws = {t: load_diffwave(a.diffwave_ckpt, t=t, device=str(device))
           for t in ts}

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=a.max_sec)
    ref = reference_items(items, min_sec=a.max_sec, utt_index=a.utt_index)
    ok = set(cross_session_split(items, min_sec=a.max_sec)) \
        if a.cross_session else set(ref)
    spks = sorted(s for s in ref if s in ok)[:a.n_files]
    print(f"[13c] {len(spks)} speakers from {a.subset} "
          f"(cross_session={a.cross_session})")

    rows = []
    for n, spk in enumerate(spks, 1):
        wav = load_flac(ref[spk]["path"], a.max_sec).to(device)   # [1, L]
        rms = float(wav.pow(2).mean().sqrt())
        scale = a.rel_rms * rms
        print(f"[13c] {n}/{len(spks)} spk{spk}  rms={rms:.4f}", flush=True)

        def unit(v):
            return v / (v.pow(2).mean().sqrt() + 1e-12) * scale

        # ===================================================================
        # DISABLED 2026-10-08 -- RESULTS §57. THIS PATH IS BROKEN.
        #
        # The block below evaluated the gradient AT `wav`, which is the global
        # MAXIMUM of cos(embed(w), embed(wav)). For any L2-normalised
        # embedding ||f||^2 == 1, so 2*J^T f == 0 and the gradient there is
        # ANALYTICALLY ZERO. Measured: 1.2e-08 to 5.7e-07, against 0.14-0.57
        # one small step away, and the resulting "directions" FLIP SIGN under
        # an input jitter of 1e-5 (cosines -0.98, -0.74, -0.11).
        #
        # Everything RESULTS §55 reported from this path is withdrawn. The
        # script refuses to run rather than being silently fixed, because a
        # fix is a REDESIGN that needs its own pre-registration (§57e):
        #   1. evaluate off the maximum -- a perturbed point, or a different
        #      utterance's embedding, so cosine-to-self < 1;
        #   2. a positive control that must pass BEFORE a verdict is read;
        #   3. a control that is not noise-like by construction.
        #
        # Until that exists, this measurement does not.
        # ===================================================================
        sys.exit(
            "[13c] REFUSING TO RUN: this path is void (RESULTS 57).\n"
            "  The identity direction was taken at the maximum of\n"
            "  cosine-to-self, where the gradient is analytically zero, so it\n"
            "  measured numerical residue. Directions from it flip sign under\n"
            "  a 1e-5 input jitter.\n"
            "  A valid version is a REDESIGN needing its own pre-registration;\n"
            "  see RESULTS 57e. Use experiments/check_direction_sanity.py to\n"
            "  gate the replacement before reading any verdict from it.")

        dirs = {}
        for e in encs:
            with torch.no_grad():
                o = e.embed(wav).detach()
            w = wav.clone().requires_grad_(True)
            c = (e.embed(w) * o).sum()
            g, = torch.autograd.grad(c, w)
            # steepest DESCENT of similarity to self: the direction that moves
            # this encoder's embedding away fastest. Same convention as §32.
            dirs[e.name] = unit(-g).detach()

        gen = torch.Generator(device=device).manual_seed(spk)
        dirs["random"] = unit(torch.randn(wav.shape, generator=gen,
                                          device=device)).detach()
        for e in encs:
            gb = torch.Generator(device=device).manual_seed(spk + 101)
            dirs[f"band_matched:{e.name}"] = band_matched_wav(
                dirs[e.name], generator=gb).detach()

        for name, d in dirs.items():
            r = dict(speaker=spk, direction=name, purifier=pname,
                     rel_rms=a.rel_rms,
                     rms=round(float(d.pow(2).mean().sqrt()), 6),
                     is_control=name.startswith(("band_matched", "random")),
                     controls_for=(name.split(":", 1)[1]
                                   if name.startswith("band_matched:") else ""))
            for t in ts:
                psr, cosv, alpha, nsr = survival(dws[t], wav, d, seeds,
                                                 full=True)
                r[f"alpha_t{t}"] = round(alpha, 4)
                r[f"nsr_t{t}"] = round(nsr, 4)
                r[f"psr_t{t}"] = round(psr, 4)
                r[f"dircos_t{t}"] = round(cosv, 4)
            r.update(PROV)
            rows.append(r)

        if n % 5 == 0 or n == len(spks):
            append_rows(a.out, rows)
            rows = []
            print(f"[13c]   -> flushed through spk{spk}", flush=True)
    if rows:
        append_rows(a.out, rows)
    print(f"[13c] done -> {a.out}")


if __name__ == "__main__":
    main()
