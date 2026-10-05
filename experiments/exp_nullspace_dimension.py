"""
experiments/exp_nullspace_dimension.py
======================================
E5 — how BIG is the pass-through subspace?

Everything so far characterises the subspace by two optimised extreme directions
(best pass-through, worst). That is two anecdotes, not a geometry. This measures
its effective dimension by greedy orthogonal deflation: find the best
pass-through direction, then the best one orthogonal to it, and so on.

    d_k = argmin R_t(d)   s.t.  ||d|| = r  and  d _|_ span(d_1..d_{k-1})

R_t(d) = E_eps || m(x_t + sqrt(ab_t) d) - m(x_t) - d ||^2 / ||d||^2, the
noise-matched pass-through error (0 = passes through untouched).

Reading the output:
  * R_t(k) stays flat out to large k  -> the subspace is high-dimensional; a
    perturbation is not forced onto a knife edge by living in it.
  * R_t(k) rises quickly              -> the subspace retains few degrees of
    freedom, which MECHANISTICALLY EXPLAINS why constraining an attack into it
    costs protection: it discards most of the attack's search space.

Either way this is the structural statement the paper currently cannot make.
Ambient dimension is n_mels x frames ~ 80 x 400 = 32000.

Run:
  python experiments/exp_nullspace_dimension.py --n-files 5 --n-dirs 48 --null-t 200 --t 200
Output:
  experiments/results/nullspace_dimension.csv
"""
import sys, io, os, argparse, time
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

from models.real_diffpure import load_real_diffpure, MelDiffPure
from evaluation.evaluate import wav_to_logmel
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_nullspace_survival import r_t_score, survival

OUT = os.path.join(PROJ, "experiments", "results", "nullspace_dimension.csv")


def orthogonalise(d, basis):
    """Project d out of the span of `basis` (list of unit vectors)."""
    for b in basis:
        d = d - (d * b).sum() * b
    return d


def find_orthogonal_direction(model, x, r, ab_t, basis, n_steps, lr, n_eps=4, seed=0):
    """Minimise the noise-matched pass-through error, staying orthogonal to
    everything already found and on the sphere ||d|| = r."""
    device = x.device
    g = torch.Generator(device=device).manual_seed(seed)
    eps = [torch.randn(x.shape, generator=g, device=device) for _ in range(n_eps)]
    with torch.no_grad():
        xts = [ab_t.sqrt() * x + (1 - ab_t).sqrt() * e for e in eps]
        m0 = [model(xt) for xt in xts]
    d = torch.randn(x.shape, generator=g, device=device)
    with torch.no_grad():
        d = orthogonalise(d, basis)
        d = d / (d.norm() + 1e-12) * r
    d = d.detach().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)
    for s in range(n_steps):
        opt.zero_grad()
        k = s % n_eps
        loss = (model(xts[k] + ab_t.sqrt() * d) - m0[k] - d).pow(2).sum() / r.pow(2)
        loss.backward()
        opt.step()
        with torch.no_grad():                       # re-project onto the constraint set
            d.data.copy_(orthogonalise(d.data, basis))
            d.data.mul_(r / (d.data.norm() + 1e-12))
    return d.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--n-dirs", type=int, default=48, dest="n_dirs")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--null-t", type=int, default=200, dest="null_t")
    ap.add_argument("--t", type=int, default=200, help="purification strength for loop PSR")
    ap.add_argument("--psr-every", type=int, default=4, dest="psr_every",
                    help="run the (expensive) full loop every k-th direction")
    ap.add_argument("--seeds", default="2026")
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

    rows = []
    for spk in spks:
        it = ps[spk][args.utt_index]
        x = wav_to_logmel(load_flac(it["path"], 4.0).to(device), device)
        ambient = int(np.prod(x.shape))
        r = torch.tensor(args.rms, device=device) * np.sqrt(ambient)
        r = (torch.ones((), device=device) * args.rms * np.sqrt(ambient))
        basis, t0 = [], time.time()
        # random baseline at the same norm, for reference
        gg = torch.Generator(device=device).manual_seed(7)
        d_rand = torch.randn(x.shape, generator=gg, device=device)
        d_rand = d_rand / d_rand.norm() * r
        r_rand = r_t_score(model, x, d_rand, ab_t, seed=spk)
        psr_rand, cos_rand = survival(dp_t, x, d_rand, seeds)
        print(f"\nspk{spk}  ambient dim {ambient}  | random: R_t={r_rand:.4f} PSR={psr_rand:.3f}",
              flush=True)
        for k in range(1, args.n_dirs + 1):
            d = find_orthogonal_direction(model, x, r, ab_t, basis, args.steps, args.lr, seed=spk + k)
            rt = r_t_score(model, x, d, ab_t, seed=spk)
            row = dict(speaker=spk, k=k, ambient=ambient, rms=args.rms,
                       R_t=round(rt, 5), R_t_random=round(r_rand, 5),
                       psr=np.nan, dircos=np.nan,
                       psr_random=round(psr_rand, 4), null_t=args.null_t, eval_t=args.t)
            if k % args.psr_every == 0 or k == 1:
                psr, dcos = survival(dp_t, x, d, seeds)
                row["psr"], row["dircos"] = round(psr, 4), round(dcos, 4)
                print(f"  k={k:3d}  R_t={rt:.5f}  PSR={psr:.3f}  cos={dcos:+.3f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            elif k % 8 == 0:
                print(f"  k={k:3d}  R_t={rt:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            rows.append(row)
            with torch.no_grad():
                basis.append((d / d.norm()).detach())
        pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint each speaker

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n" + "=" * 72)
    print("PASS-THROUGH ERROR vs SUBSPACE DIMENSION  (mean over speakers)")
    print("=" * 72)
    g = df.groupby("k")[["R_t", "psr", "dircos"]].mean()
    ref = df.R_t_random.mean()
    print(f"  random direction at the same norm: R_t = {ref:.5f}, "
          f"PSR = {df.psr_random.mean():.3f}\n")
    for k, row in g.iterrows():
        if k % max(args.psr_every, 1) == 0 or k == 1:
            frac = row.R_t / ref if ref > 0 else np.nan
            psr = "" if np.isnan(row.psr) else f"  PSR {row.psr:.3f}"
            print(f"  k={k:3d}  R_t {row.R_t:.5f}  ({100*frac:.1f} % of random){psr}")
    below = g[g.R_t < 0.25 * ref]
    if len(below):
        print(f"\n  R_t stays below 25 % of the random baseline out to k = {int(below.index.max())} "
              f"of an ambient {int(df.ambient.iloc[0])}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
