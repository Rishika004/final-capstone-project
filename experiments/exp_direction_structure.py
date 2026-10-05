"""
experiments/exp_direction_structure.py
======================================
E8 — WHAT are the pass-through directions, actually?

E7 found that directions optimised on different utterances transfer to each other
(16.5x better than random) but have mean |cos| = 0.69 between them. That is high.
Two readings, with very different consequences for the paper:

  (a) They share a trivial common component -- a global spectral tilt, or a
      concentration in silent/low-energy frames. We ALREADY know a global tilt
      passes through the purifier (R_t ~ 1e-4) and protects 0/480. If the
      optimised directions are mostly tilt, then the expensive white-box
      construction is rediscovering a one-parameter filter, the "subspace" is
      largely a line, and the negative result is explained trivially.

  (b) They share genuine structure but are not the same vector: a large common
      subspace, with the individual directions differing inside it.

This is a load-bearing distinction and it must be settled before the claim is
written. The test decomposes each optimised direction into interpretable parts:

  time-constant   d_bar[f] = mean_t d[f,t], broadcast back over time.
                  This is EXACTLY the family of static EQ curves; a global
                  spectral tilt is one element of it. Energy fraction here is
                  the "is it just a filter?" number.
  tilt            projection onto a single linear-in-mel-index ramp, constant in
                  time -- the specific transform from the on-manifold probe.
  low-energy      fraction of the direction's energy sitting in frames whose
                  log-mel energy is in the bottom quartile (silence/pauses).
  residual        what is left: genuinely time-varying, non-tilt structure.

Reported against a random direction, which fixes the scale: a random direction in
a space of T frames has ~1/T of its energy in any time-constant component, so
"large" must be judged relative to that, not against zero.

Run:
  python experiments/exp_direction_structure.py --n-files 6
Output:
  experiments/results/direction_structure.csv
"""
import sys, io, os, argparse

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
import torch.nn.functional as F

from models.real_diffpure import load_real_diffpure, MelDiffPure
from evaluation.evaluate import wav_to_logmel
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_nullspace_survival import find_direction, r_t_score, survival

OUT = os.path.join(PROJ, "experiments", "results", "direction_structure.csv")


def decompose(d, x):
    """Energy fractions of d in interpretable components. d, x: [1, n_mels, T]."""
    d = d.squeeze(0)                       # [F, T]
    tot = float(d.pow(2).sum()) + 1e-12
    F_, T = d.shape

    # 1. time-constant part = the static-EQ family (a global tilt lives here)
    dbar = d.mean(dim=1, keepdim=True)     # [F, 1]
    e_const = float((dbar.pow(2) * T).sum()) / tot

    # 2. the specific linear-in-mel ramp used by the on-manifold tilt transform
    ramp = torch.linspace(0, 1, F_, device=d.device) - 0.5
    ramp = ramp / ramp.norm()
    e_tilt = float((dbar.squeeze(1) * ramp).sum() ** 2 * T) / tot

    # 3. energy sitting in the quietest frames
    fr = x.squeeze(0).mean(dim=0)          # [T] mean log-mel energy per frame
    quiet = fr <= torch.quantile(fr, 0.25)
    e_quiet = float(d[:, quiet].pow(2).sum()) / tot
    frac_quiet = float(quiet.float().mean())

    return dict(e_time_const=round(e_const, 4), e_tilt=round(e_tilt, 4),
                e_quiet_frames=round(e_quiet, 4), frac_quiet_frames=round(frac_quiet, 3),
                e_residual=round(1.0 - e_const, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=6, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--null-t", type=int, default=200, dest="null_t")
    ap.add_argument("--t", type=int, default=200)
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v3.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(v) for v in args.seeds.split(",")]

    dp = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=args.t,
                            device=device, sampler=args.sampler)
    model = dp.denoiser
    for p in model.parameters():
        p.requires_grad_(False)
    dp_t = MelDiffPure(model, n_steps=dp.n_steps, t_diffuse=args.t,
                       sampler=args.sampler).to(device).eval()
    ab_t = dp.alpha_bar[args.null_t - 1]

    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=args.utt_index + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]

    rows, ds, dr = [], [], []
    for spk in spks:
        x = wav_to_logmel(load_flac(ps[spk][args.utt_index]["path"], 4.0).to(device), device)
        # Seed PER SPEAKER. A shared seed is correct within a speaker (it makes
        # `random` a controlled baseline against the other direction kinds), but
        # across speakers it hands every optimisation the SAME starting point, so
        # any pairwise similarity between the results is inherited from the
        # initialisation rather than from shared structure. That confound is fatal
        # for a cross-example claim: it showed up as random directions having
        # pairwise cosine 1.000.
        torch.manual_seed(1000 + int(spk))
        d0 = torch.randn_like(x)
        d0 = d0 / d0.pow(2).mean().sqrt() * args.rms
        d = find_direction(model, x, d0, args.steps, args.lr, +1, ab_t=ab_t)
        ds.append(d); dr.append(d0)
        for name, v in (("pass_through", d), ("random", d0)):
            row = dict(speaker=spk, kind=name, **decompose(v, x))
            row["R_t"] = round(r_t_score(model, x, v, ab_t, seed=spk), 5)
            row["psr"], row["dircos"] = [round(q, 4) for q in survival(dp_t, x, v, seeds)]
            rows.append(row)
        print(f"  spk{spk}: time-const {rows[-2]['e_time_const']:.3f} "
              f"(random {rows[-1]['e_time_const']:.4f})  tilt {rows[-2]['e_tilt']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # pairwise cosine, before and after removing the time-constant part
    def pcos(vs):
        out = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                n = min(vs[i].shape[-1], vs[j].shape[-1])
                a, b = vs[i][..., :n].reshape(1, -1), vs[j][..., :n].reshape(1, -1)
                out.append(abs(float(F.cosine_similarity(a, b))))
        return float(np.mean(out))

    def strip_const(v):
        w = v.squeeze(0)
        return (w - w.mean(dim=1, keepdim=True)).unsqueeze(0)

    g = df.groupby("kind")[["e_time_const", "e_tilt", "e_quiet_frames",
                            "R_t", "psr", "dircos"]].mean()
    print("\n" + "=" * 76)
    print("WHAT ARE THE PASS-THROUGH DIRECTIONS MADE OF?")
    print("=" * 76)
    print(g.round(4).to_string())
    pt, rd = g.loc["pass_through"], g.loc["random"]
    print(f"\n  energy in the static-EQ (time-constant) family:")
    print(f"     pass-through {pt.e_time_const:>7.4f}     random {rd.e_time_const:>7.4f}"
          f"   ({pt.e_time_const/max(rd.e_time_const,1e-9):.0f}x random)")
    print(f"  energy in the single linear tilt ramp:")
    print(f"     pass-through {pt.e_tilt:>7.4f}     random {rd.e_tilt:>7.4f}")
    print(f"\n  mean |cos| between directions from different utterances:")
    print(f"     as found                      {pcos(ds):.3f}")
    print(f"     after removing the time-constant part  {pcos([strip_const(v) for v in ds]):.3f}")
    print(f"     random directions (reference)  {pcos(dr):.3f}")
    print("\n" + "-" * 76)
    if pt.e_time_const > 0.5:
        print("  -> The directions ARE mostly a static filter. The white-box construction")
        print("     is largely rediscovering an EQ curve, and the paper must say so:")
        print("     report the tilt baseline as the honest comparison, not as a foil.")
    elif pt.e_tilt > 0.25:
        print("  -> A single linear tilt ramp explains a large share. Same conclusion.")
    else:
        print("  -> NOT reducible to a static filter. The time-constant family accounts")
        print(f"     for only {100*pt.e_time_const:.1f} % of the direction's energy and the single")
        print(f"     tilt ramp for {100*pt.e_tilt:.2f} %, so the construction is not rediscovering")
        print("     an EQ curve. Read the pairwise cosine above: near zero means the")
        print("     directions are DIFFERENT vectors, and any transfer between them is")
        print("     evidence of a shared high-dimensional subspace rather than of one")
        print("     shared vector.")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
