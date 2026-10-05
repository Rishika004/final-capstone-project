"""
experiments/exp_nullspace_survival.py
=====================================
E1 — does a perturbation in the pass-through (null) space of the purifier's
CORRECTION network survive the full stochastic DDPM purification loop?

Background (2026-09-11). The original repo projected mel perturbations onto
"null(J)" of the x0-predicting denoiser. Two problems:
  (a) null space of an x0-predictor = directions the purifier IGNORES, i.e.
      erases (output = clean x). The subspace that survives purification is
      the null space of the CORRECTION C(x) = P(x) - x  (pass-through:
      P(x+d) - P(x) ~= d), which for a score model at low noise is the tangent
      space of the data manifold (Stanczuk et al. 2022).
  (b) `compute_null_space_factored` returns the complement of K random probe
      directions, not of J's row space, so the projection was ~identity.

Here we estimate pass-through directions properly, by optimisation:
      d_null  = argmin ||C(x+d) - C(x)||^2   s.t. ||d|| = r   (Adam, renorm)
      d_range = argmax  ...                  (same constraint)
and compare with a random d of the same norm and the LINEAR jvp ||J_C d||/||d||.

Survival through the real purifier (residual UNet v2, fixed loop), seeds x t:
      PSR     = ||P(x+d) - P(x)|| / ||d||          (1 = fully survives)
      dir_cos = cos( P(x+d) - P(x), d )            (1 = same direction)

Run:
  python experiments/exp_nullspace_survival.py --n-files 5
Output:
  experiments/results/nullspace_survival.csv
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
from torch.autograd.functional import jvp

from models.real_diffpure import load_real_diffpure, MelDiffPure
from evaluation.evaluate import wav_to_logmel
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET

OUT = os.path.join(PROJ, "experiments", "results", "nullspace_survival.csv")


def correction(model, x):
    return model(x) - x


def find_direction(model, x, d0, n_steps, lr, sign, ab_t=None, n_eps=4, seed=0):
    """sign=+1: minimise the pass-through error (null); -1: maximise (range).
    ab_t=None: zero-noise single pass  ||C(x+d)-C(x)||^2.
    ab_t given: NOISE-MATCHED  E_eps || m(x_t + sqrt(ab) d) - m(x_t) - d ||^2 with
    x_t = sqrt(ab) x + sqrt(1-ab) eps  (fixed set of n_eps noise draws)."""
    r = d0.norm()
    g = torch.Generator(device=x.device).manual_seed(seed)
    eps = [torch.randn(x.shape, generator=g, device=x.device) for _ in range(n_eps)]
    with torch.no_grad():
        if ab_t is None:
            c0 = correction(model, x)
        else:
            xts = [ab_t.sqrt() * x + (1 - ab_t).sqrt() * e for e in eps]
            m0 = [model(xt) for xt in xts]
    d = d0.clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=lr)
    for s in range(n_steps):
        opt.zero_grad()
        if ab_t is None:
            loss = (correction(model, x + d) - c0).pow(2).sum() / r.pow(2)
        else:
            k = s % n_eps
            loss = (model(xts[k] + ab_t.sqrt() * d) - m0[k] - d).pow(2).sum() / r.pow(2)
        (sign * loss).backward()
        opt.step()
        with torch.no_grad():
            d.mul_(r / (d.norm() + 1e-12))
    return d.detach()


@torch.no_grad()
def r_t_score(model, x, d, ab_t, n_eps=4, seed=0):
    """R_t = E_eps || m(x_t + sqrt(ab) d) - m(x_t) - d ||^2 / ||d||^2.
    One denoiser pass per noise draw: the cheap first-order pass-through
    criterion, recorded here so every direction has (R_t, loop-PSR) paired."""
    nrm = d.pow(2).sum() + 1e-12
    g = torch.Generator(device=x.device).manual_seed(seed)
    acc = 0.0
    for _ in range(n_eps):
        eps = torch.randn(x.shape, generator=g, device=x.device)
        xt = ab_t.sqrt() * x + (1 - ab_t).sqrt() * eps
        shift = model(xt + ab_t.sqrt() * d) - model(xt)
        acc += float((shift - d).pow(2).sum() / nrm)
    return acc / n_eps


@torch.no_grad()
def survival(dp, x, d, seeds, full=False):
    """PSR and direction cosine through the full purification loop.

    With full=True also returns the signal gain and injected-noise ratio from the
    decomposition  P(x+d) - P(x) = alpha*d + residual. The norm ratio PSR cannot
    separate "the perturbation survived" from "it was destroyed and replaced by
    uncorrelated noise of similar size" -- see RESULTS_ICASSP.md section 11a, where
    that ambiguity nearly inverted the reading of the vocoder control. alpha is the
    quantity the paper reports; PSR is kept for continuity with earlier tables."""
    psr, cosv, alph, nsr = [], [], [], []
    n2 = d.pow(2).sum() + 1e-12
    for sd in seeds:
        px = dp(x, seed=sd)
        pxd = dp(x + d, seed=sd)
        diff = pxd - px
        psr.append((diff.norm() / d.norm()).item())
        cosv.append(F.cosine_similarity(diff.reshape(1, -1), d.reshape(1, -1)).item())
        a = ((diff * d).sum() / n2).item()
        alph.append(a)
        nsr.append(((diff - a * d).norm() / (abs(a) * d.norm() + 1e-12)).item())
    if full:
        return (float(np.mean(psr)), float(np.mean(cosv)),
                float(np.mean(alph)), float(np.mean(nsr)))
    return float(np.mean(psr)), float(np.mean(cosv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: EVAL_SUBSET). Pin "
                         "it to the corpus of the run this describes.")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="use the SHARED reference selector and the cross-session "
                         "speaker set, so this matches the crafting/cloning runs.")
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--rms", type=float, default=0.2, help="perturbation RMS in log-mel units")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--t", default="50,100", help="purification strengths to test")
    ap.add_argument("--seeds", default="2026,7,99")
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v2.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    ap.add_argument("--eval-ckpt", default="", dest="eval_ckpt",
                    help="evaluate survival through a DIFFERENT purifier than the one "
                         "used to find the directions (cross-purifier transfer test)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--null-t", default="", dest="null_t",
                    help="comma list of noise levels for NOISE-MATCHED pass-through directions")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in args.t.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    dp50 = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=ts[0], device=device)
    model = dp50.denoiser
    for p in model.parameters():
        p.requires_grad_(False)
    if args.eval_ckpt:
        dp_ev = load_real_diffpure(checkpoint_path=args.eval_ckpt, t_diffuse=ts[0], device=device)
        eval_model = dp_ev.denoiser
        for p_ in eval_model.parameters():
            p_.requires_grad_(False)
        print(f"[E1] directions from {os.path.basename(args.purifier_ckpt)} -> "
              f"evaluated through {os.path.basename(args.eval_ckpt)}")
    else:
        eval_model = model
    dps = {t: MelDiffPure(eval_model, n_steps=dp50.n_steps, t_diffuse=t,
                          sampler=args.sampler).to(device).eval() for t in ts}

    # Corpus and speaker selection go through the SAME shared selector the
    # crafting and cloning scripts use (fix 2026-09-19). Without --subset these
    # scripts silently followed EVAL_SUBSET, which moved from dev-clean to
    # test-clean, and without --cross-session they picked a different utterance
    # AND a different speaker set than the run they are meant to describe.
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             args.subset or EVAL_SUBSET, 4.0)
    if args.cross_session:
        from experiments.librispeech import reference_items, cross_session_split
        ref = reference_items(items, min_sec=4.0, utt_index=args.utt_index)
        ok = set(cross_session_split(items, min_sec=4.0))
        spks = [s for s in sorted(ref) if s in ok][
            args.spk_offset:args.spk_offset + args.n_files]
        ps = {s: {args.utt_index: ref[s]} for s in spks}
        print(f"[sel] cross-session: {len(spks)} speakers from "
              f"{args.subset or EVAL_SUBSET} (shared selector)")
    else:
        ps = per_speaker(items, n_utt=args.utt_index + 1)
        spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]
    rows = []
    for spk in spks:
        it = ps[spk][args.utt_index]
        x = wav_to_logmel(load_flac(it["path"], 4.0).to(device), device)
        torch.manual_seed(0)
        d_rand = torch.randn_like(x)
        d_rand = d_rand / d_rand.pow(2).mean().sqrt() * args.rms
        t0 = time.time()
        d_null = find_direction(model, x, d_rand, args.steps, args.lr, +1)
        d_range = find_direction(model, x, d_rand, args.steps, args.lr, -1)
        kinds = [("random", d_rand), ("null", d_null), ("range", d_range)]
        for nt in [int(v) for v in args.null_t.split(",") if v]:
            abt = dp50.alpha_bar[nt - 1]
            kinds.append((f"null_t{nt}", find_direction(model, x, d_rand, args.steps, args.lr, +1, ab_t=abt)))
        t_opt = time.time() - t0
        for name, d in kinds:
            # linear (jvp) response of the correction network
            _, jd = jvp(lambda z: correction(model, z), (x,), (d,))
            lin = (jd.norm() / d.norm()).item()
            with torch.no_grad():
                single = ((model(x + d) - model(x)).norm() / d.norm()).item()
            r = dict(speaker=spk, kind=name, rms=args.rms, lin_gain=round(lin, 4),
                     single_pass_psr=round(single, 4), opt_s=round(t_opt, 1),
                     dirs_from=os.path.basename(args.purifier_ckpt),
                     eval_through=os.path.basename(args.eval_ckpt or args.purifier_ckpt))
            if eval_model is not model:
                # The transfer question is asked at the Jacobian level too, not only
                # at the loop level: a direction that is pass-through for the source
                # purifier need not be pass-through for the target one.
                _, jd_e = jvp(lambda z: correction(eval_model, z), (x,), (d,))
                r["lin_gain_eval"] = round((jd_e.norm() / d.norm()).item(), 4)
                with torch.no_grad():
                    r["single_pass_psr_eval"] = round(
                        ((eval_model(x + d) - eval_model(x)).norm() / d.norm()).item(), 4)
            for nt in [int(v) for v in args.null_t.split(",") if v] or ts:
                r[f"R_t{nt}"] = round(r_t_score(model, x, d, dp50.alpha_bar[nt - 1], seed=spk), 5)
                if eval_model is not model:
                    r[f"R_t{nt}_eval"] = round(
                        r_t_score(eval_model, x, d, dp50.alpha_bar[nt - 1], seed=spk), 5)
            for t in ts:
                psr, c, alpha, nsr = survival(dps[t], x, d, seeds, full=True)
                r[f"psr_t{t}"] = round(psr, 4)
                r[f"dircos_t{t}"] = round(c, 4)
                r[f"alpha_t{t}"] = round(alpha, 4)
                r[f"nsr_t{t}"] = round(nsr, 4)
            rows.append(r)
            print(f"spk{spk} {name:6s} | lin |Jd|/|d|={lin:.3f} | single PSR={single:.3f} | "
                  + " ".join(f"t{t}: PSR={r[f'psr_t{t}']:.3f} cos={r[f'dircos_t{t}']:.3f}" for t in ts),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nMEAN over files:")
    cols = [c for c in df.columns if c.startswith(("lin", "single", "psr", "dircos", "R_t", "alpha", "nsr"))]
    print(df.groupby("kind")[cols].mean().round(3).to_string())

    if eval_model is not model:
        # The transfer verdict. A pass-through direction found on the source purifier
        # transfers if it is still far better than a random direction on the TARGET.
        print("\n" + "=" * 72)
        print(f"TRANSFER  {os.path.basename(args.purifier_ckpt)} -> "
              f"{os.path.basename(args.eval_ckpt)}")
        print("=" * 72)
        g = df.groupby("kind")[cols].mean()
        for nt in [int(v) for v in args.null_t.split(",") if v]:
            src, tgt = f"R_t{nt}", f"R_t{nt}_eval"
            if src not in g or tgt not in g or "random" not in g.index:
                continue
            k = f"null_t{nt}"
            if k not in g.index:
                continue
            print(f"  t={nt}  pass-through error R_t (lower = survives better)")
            print(f"    random direction : source {g.loc['random', src]:.4f}   "
                  f"target {g.loc['random', tgt]:.4f}")
            print(f"    optimised {k:9s}: source {g.loc[k, src]:.4f}   "
                  f"target {g.loc[k, tgt]:.4f}")
            gain_s = g.loc['random', src] / max(g.loc[k, src], 1e-9)
            gain_t = g.loc['random', tgt] / max(g.loc[k, tgt], 1e-9)
            print(f"    advantage over random: source {gain_s:.2f}x   target {gain_t:.2f}x")
            if gain_t >= 0.5 * gain_s and gain_t > 1.5:
                print("    -> TRANSFERS: the subspace is a property of the purifier family.")
            elif gain_t > 1.5:
                print("    -> PARTIAL transfer.")
            else:
                print("    -> DOES NOT TRANSFER: the subspace is specific to the "
                      "purifier it was found on.")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
