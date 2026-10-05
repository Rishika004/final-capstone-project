"""
experiments/exp_vocoder_control.py
==================================
E6 — how much of the measured "survival" is the PURIFIER, and how much is the
mel + Griffin-Lim round trip?

Every survival number we report is a mel-domain PSR measured across the
purifier. But the attacker's full chain is mel -> purify -> Griffin-Lim -> wav,
and the protection numbers are measured at the end of that chain. If the
vocoder round trip alone already destroys (or already preserves) perturbations,
then attributing the outcome to the purifier is unsound. We diagnosed exactly
this class of confound for purifier v2 in RESULTS_ICASSP.md section 1b and never
added the control. This is the control.

Three pipelines, same PSR metric, same directions:

  vocoder only    mel -> GL -> wav -> mel                (no purifier at all)
  purifier only   mel -> purify -> mel                   (what we have reported)
  full chain      mel -> purify -> GL -> wav -> mel      (what the attacker does)

If "vocoder only" is near 1.0, the round trip is transparent and the purifier
numbers stand. If it is low, part of what we have been calling purification is
really the vocoder, and the paper must say so.

Run:
  python experiments/exp_vocoder_control.py --n-files 5 --t 200
Output:
  experiments/results/vocoder_control.csv
"""
import sys, io, os, argparse
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from evaluation.evaluate import wav_to_logmel
from utils.vocoder import invert_mel
from models.real_diffpure import load_real_diffpure, MelDiffPure
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_nullspace_survival import find_direction, r_t_score
from experiments.exp_purifier_aware import attacker_resynth

OUT = os.path.join(PROJ, "experiments", "results", "vocoder_control.csv")


def psr(d_in, d_out):
    """Decompose the output shift into the part along the input perturbation and
    the part orthogonal to it:  d_out = alpha * d_in + residual.

    A bare norm ratio ||d_out||/||d_in|| conflates two very different things. An
    operation that PRESERVES the perturbation and one that DESTROYS it but adds
    its own uncorrelated error of similar size both score ~1. Griffin-Lim is
    firmly in the second category, so the norm ratio alone would have told us the
    round trip is transparent when it is not.

      alpha  signal gain     : 1 = the perturbation came through at full strength
      nsr    noise-to-signal : ||residual|| / ||alpha * d_in||, the junk injected
                               by the operation, relative to what survived
      cos    = alpha / (||d_out||/||d_in||)
    """
    d_in_f, d_out_f = d_in.reshape(1, -1), d_out.reshape(1, -1)
    n2 = float(d_in.pow(2).sum()) + 1e-12
    alpha = float((d_out * d_in).sum()) / n2
    resid = float((d_out - alpha * d_in).norm())
    nsr = resid / (abs(alpha) * float(d_in.norm()) + 1e-12)
    return (float(d_out.norm() / (d_in.norm() + 1e-12)),
            float(F.cosine_similarity(d_in_f, d_out_f)),
            alpha, nsr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v3.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--vocoder", default="gl", choices=["gl", "phase"],
                    help="gl = Griffin-Lim from scratch; phase = magnitude from the "
                         "purified mel + phase of the waveform the attacker holds")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dp = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=args.t,
                            device=device, sampler=args.sampler)
    model = dp.denoiser
    for p in model.parameters():
        p.requires_grad_(False)
    dp_t = MelDiffPure(model, n_steps=dp.n_steps, t_diffuse=args.t,
                       sampler=args.sampler).to(device).eval()
    ab_t = dp.alpha_bar[args.t - 1]

    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=args.utt_index + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]

    rows = []
    print(f"vocoder = {args.vocoder}   (a = signal gain, nsr = injected noise / signal)")
    print(f"{'spk':>6} {'direction':>14} | {'vocoder only':>19} | {'purifier only':>19} "
          f"| {'full chain':>19}")
    for spk in spks:
        it = ps[spk][args.utt_index]
        x = wav_to_logmel(load_flac(it["path"], 4.0).to(device), device)
        torch.manual_seed(0)
        d0 = torch.randn_like(x)
        d0 = d0 / d0.pow(2).mean().sqrt() * args.rms
        dirs = {"random": d0,
                "null_zero": find_direction(model, x, d0, args.steps, args.lr, +1),
                "null_matched": find_direction(model, x, d0, args.steps, args.lr, +1, ab_t=ab_t)}
        wav_o = load_flac(it["path"], 4.0).to(device)

        def resynth(mel):
            if args.vocoder == "gl":
                return invert_mel(mel.cpu()).to(device)
            return attacker_resynth(mel, wav_o, method="phase")

        for name, d in dirs.items():
            with torch.no_grad():
                # 1. vocoder round trip only
                w_o = resynth(x)
                w_p = resynth(x + d)
                v = psr(d, wav_to_logmel(w_p, device) - wav_to_logmel(w_o, device))
                # 2. purifier only
                px, pxd = dp_t(x, seed=args.seed), dp_t(x + d, seed=args.seed)
                pur = psr(d, pxd - px)
                # 3. full chain
                w_o2 = resynth(px)
                w_p2 = resynth(pxd)
                full = psr(d, wav_to_logmel(w_p2, device) - wav_to_logmel(w_o2, device))
                rt = r_t_score(model, x, d, ab_t, seed=spk)
            rows.append(dict(speaker=spk, direction=name, eval_t=args.t, R_t=round(rt, 5),
                             vocoder=args.vocoder,
                             psr_vocoder=round(v[0], 4), cos_vocoder=round(v[1], 4),
                             alpha_vocoder=round(v[2], 4), nsr_vocoder=round(v[3], 4),
                             psr_purifier=round(pur[0], 4), cos_purifier=round(pur[1], 4),
                             alpha_purifier=round(pur[2], 4), nsr_purifier=round(pur[3], 4),
                             psr_full=round(full[0], 4), cos_full=round(full[1], 4),
                             alpha_full=round(full[2], 4), nsr_full=round(full[3], 4)))
            print(f"{spk:>6} {name:>14} | a={v[2]:>5.2f} nsr={v[3]:>5.2f} | "
                  f"a={pur[2]:>5.2f} nsr={pur[3]:>5.2f} | a={full[2]:>5.2f} nsr={full[3]:>5.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n" + "=" * 86)
    print(f"MEANS over {len(spks)} speakers, t={args.t}, vocoder={args.vocoder}")
    print("=" * 86)
    print(df.groupby("direction")[["R_t", "alpha_vocoder", "nsr_vocoder",
                                   "alpha_purifier", "nsr_purifier",
                                   "alpha_full", "nsr_full"]].mean().round(3).to_string())

    g = df.groupby("direction")[["alpha_purifier", "alpha_vocoder", "alpha_full",
                                 "nsr_vocoder", "nsr_purifier"]].mean()
    print("\n" + "-" * 86)
    print("VERDICT")
    print("-" * 86)
    av, nv = g.loc["random", "alpha_vocoder"], g.loc["random", "nsr_vocoder"]
    print(f"  The {args.vocoder} round trip alone passes {100*av:.0f} % of a perturbation through,")
    print(f"  while injecting uncorrelated error {nv:.1f}x the size of what survives.")
    if "null_matched" in g.index:
        ap_, af = g.loc["null_matched", "alpha_purifier"], g.loc["null_matched", "alpha_full"]
        rp = g.loc["random", "alpha_purifier"]
        print("")
        print("  Noise-matched pass-through direction:")
        print(f"    across the purifier alone : signal gain {ap_:.3f}  (random: {rp:.3f})")
        print(f"    across the full chain     : signal gain {af:.3f}")
        print("")
        if ap_ > 0.9 and af < 0.75 * ap_:
            print("  -> The subspace is REAL but it is a property of the PURIFIER only.")
            print("     The perturbation survives purification essentially intact and is then")
            print("     degraded by the vocoder, a separate additive channel the null-space")
            print("     construction does not and cannot control. Report survival across the")
            print("     purifier, and report this vocoder loss as the reason subspace-")
            print("     constrained attacks do not convert survival into protection.")
        elif ap_ > 0.9:
            print("  -> The subspace survives the full attacker chain, not just the purifier.")
        else:
            print("  -> The subspace does not survive even the purifier at this setting.")
    if nv > 1.0:
        print("")
        print(f"  NOTE: norm-ratio PSR is not a safe metric here (vocoder nsr = {nv:.1f}).")
        print("        Report the signal gain alpha instead: a norm ratio near 1 can mean")
        print("        'preserved' or 'destroyed and replaced by equal-sized noise'.")
    print("")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
