"""
experiments/exp_crossexample_transfer.py
========================================
E7 — is the pass-through subspace a property of the INPUT, or of the DATA?

E2 showed the subspace transfers across independently trained purifiers. That
rules out "idiosyncrasy of one checkpoint" but leaves a second, equally
deflationary reading open: the subspace might be local to the utterance the
direction was optimised on, in which case it is a per-example curiosity rather
than structure in speech.

The theory says otherwise. By Tweedie/Miyasawa the denoiser's Jacobian is

    J_m(x_t) = ( I + (1 - ab_t) * H_t ) / sqrt(ab_t),     H_t = grad^2 log p_t(x_t)

so the noise-matched pass-through condition m(x_t + sqrt(ab_t) d) - m(x_t) = d
is, to first order, exactly

    H_t d = 0.

The criterion R_t is therefore measuring ||H_t d||: **the pass-through subspace
IS the null space of the log-density Hessian.** Two consequences worth testing,
because they are what separate a real claim from a coincidence:

  (a) it should transfer across models   -> E2, confirmed (76x on an unseen net)
  (b) it should transfer across INPUTS   -> this experiment

If (b) holds, the subspace is a property of the density that speech lives on,
which is a statement about data and not about our purifier at all.

Measured here, for directions optimised on speaker i and evaluated at speaker j:

    R_t(d_i @ x_j)        pass-through error of i's direction at j's utterance
    cos(d_i, d_j)         are they literally the same direction, or merely both
                          inside a shared subspace?

The second column is the one that decides the interpretation. If the transfer
matrix is flat AND the pairwise cosines are near zero, the directions are
different vectors drawn from a common high-dimensional subspace -- which is a far
stronger result than every utterance yielding the same vector.

Run:
  python experiments/exp_crossexample_transfer.py --n-files 6 --null-t 200
Output:
  experiments/results/crossexample_transfer.csv
"""
import sys, io, os, argparse, time

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

OUT = os.path.join(PROJ, "experiments", "results", "crossexample_transfer.csv")


def crop_to(d, shape):
    """Utterances differ in length; compare on the overlapping frames only."""
    n = min(d.shape[-1], shape[-1])
    if d.shape[-1] == shape[-1]:
        return d
    out = torch.zeros(shape, device=d.device, dtype=d.dtype)
    out[..., :n] = d[..., :n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=6, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--null-t", type=int, default=200, dest="null_t")
    ap.add_argument("--t", type=int, default=200, help="purification strength for loop PSR")
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

    # 1. one pass-through direction per utterance, each optimised only on its own
    xs, ds, rands = [], [], []
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
        t0 = time.time()
        d = find_direction(model, x, d0, args.steps, args.lr, +1, ab_t=ab_t)
        xs.append(x); ds.append(d); rands.append(d0)
        print(f"  optimised spk{spk}  R_t(own) = {r_t_score(model, x, d, ab_t, seed=0):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 2. every direction against every utterance
    rows = []
    for j, (spk_j, x_j) in enumerate(zip(spks, xs)):
        r_rand = r_t_score(model, x_j, crop_to(rands[j], x_j.shape), ab_t, seed=0)
        for i, spk_i in enumerate(spks):
            d = crop_to(ds[i], x_j.shape)
            if d.norm() < 1e-8:
                continue
            d = d / d.pow(2).mean().sqrt() * args.rms      # renormalise after cropping
            rt = r_t_score(model, x_j, d, ab_t, seed=0)
            psr, dcos = survival(dp_t, x_j, d, seeds)
            rows.append(dict(dir_from=spk_i, eval_at=spk_j, own=(i == j),
                             R_t=round(rt, 5), R_t_random=round(r_rand, 5),
                             psr=round(psr, 4), dircos=round(dcos, 4),
                             null_t=args.null_t, eval_t=args.t))
        print(f"  evaluated all directions at spk{spk_j}", flush=True)

    # 3. are the directions the same vector, or different vectors in one subspace?
    cs = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            n = min(ds[i].shape[-1], ds[j].shape[-1])
            a, b = ds[i][..., :n].reshape(1, -1), ds[j][..., :n].reshape(1, -1)
            cs.append(abs(float(F.cosine_similarity(a, b))))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    own = df[df.own].R_t.mean()
    other = df[~df.own].R_t.mean()
    rnd = df.R_t_random.mean()
    print("\n" + "=" * 74)
    print("CROSS-EXAMPLE TRANSFER OF THE PASS-THROUGH SUBSPACE")
    print("=" * 74)
    print("\nR_t (lower = passes through better), mean over pairs:")
    print(f"  random direction                      {rnd:.5f}   (1.0x)")
    print(f"  direction optimised ON this utterance  {own:.5f}   ({rnd/max(own,1e-9):.1f}x better)")
    print(f"  direction optimised on ANOTHER one     {other:.5f}   ({rnd/max(other,1e-9):.1f}x better)")
    print(f"\n  loop PSR at t={args.t}: own {df[df.own].psr.mean():.3f}, "
          f"transferred {df[~df.own].psr.mean():.3f}")
    print(f"  direction cosine  : own {df[df.own].dircos.mean():.3f}, "
          f"transferred {df[~df.own].dircos.mean():.3f}")
    mc = float(np.mean(cs)) if cs else float("nan")
    print(f"\n  mean |cos| between two directions from different utterances: {mc:.3f}")
    print("\n" + "-" * 74)
    keep = rnd / max(other, 1e-9)
    if keep > 5 and mc < 0.3:
        print("  -> The subspace is a property of the DATA, not of the utterance.")
        print("     Directions optimised on one utterance pass through another almost")
        print("     as well, while being close to ORTHOGONAL to each other. They are")
        print("     therefore different vectors drawn from one shared, high-dimensional")
        print("     subspace -- consistent with the Tweedie prediction that the")
        print("     criterion is measuring the null space of the log-density Hessian.")
    elif keep > 5:
        print(f"  -> Transfers, but the directions are largely the SAME vector")
        print(f"     (|cos| = {mc:.2f}). Check for a trivial common component, e.g. a")
        print("     global tilt or a silence/low-energy region, before claiming a subspace.")
    else:
        print("  -> Does NOT transfer across utterances. The subspace is local to the")
        print("     example, and the paper must say the effect is per-utterance.")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
