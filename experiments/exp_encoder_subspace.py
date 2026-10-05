"""
experiments/exp_encoder_subspace.py
===================================
A2 — WHY does the purification-invariant constraint help WavLM-SV and ECAPA but
REVERSE on the d-vector? And which encoder should we add to fix it?

THE QUESTION, AND WHY IT IS NOT ANSWERED BY WHAT WE HAVE
--------------------------------------------------------
`diag_directional_survival.py` measures how much of the CRAFTED perturbation's
encoder-relevant component survives. That is a property of a particular crafted
file. This script measures something upstream and encoder-INTRINSIC: for each
encoder, take the mel-space direction that moves it most, and ask how well THAT
DIRECTION ITSELF survives the purifier.

    g_e  = d/d(mel) cos( E_e(V(mel)), E_e(x) )     (unit norm, scaled to --rms)
    then survival of g_e through the real purification loop:
        alpha = <P(m + g_e) - P(m), g_e> / ||g_e||^2      1.0 = untouched
        nsr   = ||residual|| / ||alpha * g_e||            0.0 = clean survival

alpha(g_e) is a property of the ENCODER and the PURIFIER, not of any crafting
run. It is therefore usable as a SELECTION CRITERION for a new ensemble member
before spending GPU hours crafting with it (item A1).

THE HYPOTHESIS
--------------
Protection must satisfy two things at once: move the encoder, and survive the
purifier. Those are compatible only if the encoder's identity-relevant
directions lie inside the purifier's invariant subspace. If
alpha(g_WavLM) is high and alpha(g_dvector) is low, then for the d-vector the
two requirements CONFLICT -- the constraint steers the perturbation toward
directions that survive, which are precisely the ones the d-vector does not
read -- and the measured reversal (RESULTS 27a: +0.071, +0.075 at t=1, 3/29
speakers improved) is the predicted consequence, not an anomaly.

PRE-REGISTERED READINGS, written before the run
-----------------------------------------------
  * alpha(WavLM), alpha(ECAPA) HIGH and alpha(d-vector) LOW
        -> the hypothesis holds. The paper gains a mechanism for the whole
           encoder pattern, and A1 becomes "add the encoder with the highest
           alpha" instead of "add more encoders".
  * all alphas SIMILAR
        -> the hypothesis FAILS. The d-vector story lies elsewhere (its partial
           averaging, or its threshold geometry -- note its cross-session EER is
           3.5-5 % against ECAPA's 0.6-0.9 %, so its accept region is far
           looser). Report as a negative result and do not pick encoders this
           way.
  * alpha(d-vector) HIGH
        -> the hypothesis is refuted outright and §27's reversal needs a
           different explanation entirely.

THE RETRO-VALIDATION THAT MAKES THIS CREDIBLE
---------------------------------------------
x-vector was tested as a fourth crafting member and REJECTED on measured grounds
(RESULTS 24b: it moved 0.08 against ECAPA's 1.16; separation 0.062). It is
included here with `--with-xvector`. If this criterion is sound it must score
x-vector LOW *without being told that outcome*. A criterion that cannot predict
the one case we already know is not a criterion.

Baselines, so the alphas are interpretable rather than free-floating:
  * random  -- a Gaussian mel direction of identical norm.
  * (reference, not recomputed here) the OPTIMISED pass-through direction and
    random, measured on this purifier in survival_tc_v7.csv: alpha 0.867 vs
    0.713 at t=50, 0.525 vs 0.466 at t=200. An encoder direction near the
    optimised value lives in the invariant subspace; near random, it does not.

Run:
  python experiments/exp_encoder_subspace.py --n-files 29 --with-xvector
Output:
  experiments/results/encoder_subspace.csv
"""
import sys
import io
import os
import argparse

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import torch
import torch._dynamo   # noqa: F401  (import-order gotcha; see exp_purifier_aware)
import transformers    # noqa: F401

OUT = os.path.join(PROJ, "experiments", "results", "encoder_subspace.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--t", default="50,200", help="purifier strengths")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--rms", type=float, default=0.2,
                    help="direction norm in log-mel units; matches "
                         "exp_nullspace_survival.py so alphas are comparable")
    ap.add_argument("--purifier-ckpt", default=os.path.join(
        PROJ, "checkpoints", "purifier_v7.pt"), dest="ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--with-xvector", action="store_true", dest="with_xvector",
                    help="include x-vector, the retro-validation case (RESULTS 24b)")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in a.t.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]

    from models.real_diffpure import load_real_diffpure, MelDiffPure
    from evaluation.evaluate import (wav_to_logmel, N_FFT, N_MELS, HOP_LENGTH,
                                     SAMPLE_RATE)
    from core.optimizer import _build_mel_to_wav, _mel_to_wav_differentiable
    from experiments.exp_purifier_aware import build_encoders
    from experiments.exp_nullspace_survival import survival
    from experiments.librispeech import list_librispeech, reference_items, load_flac

    dp0 = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=ts[0],
                             device=device, sampler=a.sampler)
    den = dp0.denoiser
    for p in den.parameters():
        p.requires_grad_(False)
    dps = {t: MelDiffPure(den, n_steps=dp0.n_steps, t_diffuse=t,
                          sampler=a.sampler).to(device).eval() for t in ts}
    encs = build_encoders(device, with_xvector=a.with_xvector)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    spks = sorted(ref)[: a.n_files]
    print(f"[A2] {len(spks)} speakers | purifier {os.path.basename(a.ckpt)} "
          f"({a.sampler}) t={ts} | encoders: {[e.name for e in encs]}")
    print("[A2] alpha = how much of the encoder's own identity direction the "
          "purifier passes through (1.0 = untouched).")
    print("[A2] reference on this purifier (survival_tc_v7.csv): optimised "
          "pass-through 0.867 / random 0.713 at t=50.\n")

    rows = []
    for i, spk in enumerate(spks):
        it = ref[spk]
        wav_o = load_flac(it["path"], 4.0).to(device)
        mel_o = wav_to_logmel(wav_o, device).detach()
        inv, phase, win, wav_len, T_frames = _build_mel_to_wav(
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            sample_rate=SAMPLE_RATE, x_wav=wav_o, device=device)

        def unit(v):
            """Scale a mel direction to the protocol's per-element RMS."""
            return v / (v.pow(2).mean().sqrt() + 1e-12) * a.rms

        dirs = {}
        for e in encs:
            with torch.no_grad():
                o = e.embed(wav_o).detach()
            m = mel_o.clone().requires_grad_(True)
            w = _mel_to_wav_differentiable(m, inv, phase, win, N_FFT,
                                           HOP_LENGTH, T_frames, wav_len)
            w = w / (w.abs().max() + 1e-6)
            c = (e.embed(w) * o).sum()
            g, = torch.autograd.grad(c, m)
            # The identity direction: steepest DESCENT of similarity to self,
            # i.e. the direction that moves this encoder's embedding away fastest.
            dirs[e.name] = unit(-g).detach()
        gen = torch.Generator(device=device).manual_seed(spk)
        dirs["random"] = unit(torch.randn(mel_o.shape, generator=gen,
                                          device=device)).detach()

        for name, d in dirs.items():
            r = dict(speaker=spk, direction=name,
                     rms=round(float(d.pow(2).mean().sqrt()), 4))
            for t in ts:
                psr, cosv, alpha, nsr = survival(dps[t], mel_o, d, seeds,
                                                 full=True)
                r[f"alpha_t{t}"] = round(alpha, 4)
                r[f"nsr_t{t}"] = round(nsr, 4)
                r[f"psr_t{t}"] = round(psr, 4)
                r[f"dircos_t{t}"] = round(cosv, 4)
            rows.append(r)
        print(f"  [{i + 1}/{len(spks)}] spk{spk}  " + "  ".join(
            f"{n.split()[0][:9]}={[x for x in rows if x['speaker'] == spk and x['direction'] == n][0][f'alpha_t{ts[0]}']:+.3f}"
            for n in dirs), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("MEAN alpha / nsr per direction  (higher alpha = survives purification)")
    print("=" * 74)
    cols = [c for c in df.columns if c.startswith(("alpha", "nsr"))]
    print(df.groupby("direction")[cols].mean().round(3).to_string())
    print(f"\n[A2] saved {a.out}")
    print("[A2] Read against the pre-registered criteria in this file's docstring.")


if __name__ == "__main__":
    main()
