"""
experiments/exp_foreign_purifier.py
===================================
E9 — does the pass-through subspace transfer to a purifier we did NOT train?

This is the experiment that decides how far the title claim can reach.

E2 showed transfer from purifier v3 to v4. But v3 and v4 share an architecture, a
corpus, a beta schedule and a training script; they differ in seed, data order and
validation split. A reviewer is entitled to read that as *seed stability*, which
Zhang et al. (ICML 2024) already predict, and our own related-work section
concedes the prediction. To say anything about "diffusion purifiers of speech" we
need a purifier that is foreign in every respect: someone else's architecture,
someone else's training data, someone else's domain (raw waveform rather than mel).

THE MEASUREMENT, AND THE TRAP IN IT
-----------------------------------
Our perturbation d lives in log-mel space. A waveform-domain purifier cannot be
handed d directly; we have to render mel+d to audio, purify, and come back:

    w_o = vocode(mel)          w_p = vocode(mel + d)
    d_out = mel(P(w_p)) - mel(P(w_o))

The trap: the mel -> waveform -> mel round trip is NOT lossless. E6 measured it
at signal gain 0.57-0.68 with injected noise 3-4x the surviving signal, for
Griffin-Lim and phase-preserving inversion respectively. So **even a perfectly
transparent purifier would score well below 1.0 here.** An absolute alpha from
this pipeline is uninterpretable on its own.

The quantity that IS interpretable is the ADVANTAGE OVER RANDOM measured through
the same pipeline, because the vocoder loss applies to both arms and cancels:

    advantage = alpha(pass-through direction) / alpha(random direction)

E2 reported 109.8x on the source purifier and 76.2x on the unseen one in R_t
terms. Here the comparison to make is between:

    advantage through OUR purifier      (the reference cell, same pipeline)
    advantage through the FOREIGN one   (the test cell)

If the foreign advantage is a substantial fraction of ours, the subspace is a
property of speech diffusion purifiers rather than of our recipe. If it collapses
to ~1x, it is a property of our recipe -- and that is a publishable negative too,
considerably more interesting than another seed.

Every run therefore reports three pipelines, exactly as E6 did:
    vocoder only    (the floor: what the round trip alone costs)
    ours            (reference)
    foreign         (the test)

BACKEND SEAM
------------
`PURIFIERS` maps a name to a factory returning a callable
`purify(wav[1, T] float32 @16 kHz, seed:int) -> wav[1, T]`, which must perform a
DiffPure-style forward-noise-to-t then reverse-denoise -- NOT generation from pure
noise. Register a foreign model there and nothing else changes.

Run (once a foreign backend is registered):
  python experiments/exp_foreign_purifier.py --foreign diffwave --n-files 5
Output:
  experiments/results/foreign_purifier.csv
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

from utils.csvio import read_csv as read_csv_safe  # noqa: F401  (kept for symmetry)
from evaluation.evaluate import wav_to_logmel, SAMPLE_RATE
from utils.vocoder import invert_mel
from models.real_diffpure import load_real_diffpure, MelDiffPure
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_nullspace_survival import find_direction, r_t_score
from experiments.exp_purifier_aware import attacker_resynth

OUT = os.path.join(PROJ, "experiments", "results", "foreign_purifier.csv")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def decompose(d_in, d_out):
    """d_out = alpha * d_in + residual.  See RESULTS section 11a: a norm ratio
    cannot separate 'survived' from 'destroyed and replaced by equal-sized
    noise', and the vocoder round trip is firmly the second case."""
    n2 = float(d_in.pow(2).sum()) + 1e-12
    alpha = float((d_out * d_in).sum()) / n2
    resid = float((d_out - alpha * d_in).norm())
    nsr = resid / (abs(alpha) * float(d_in.norm()) + 1e-12)
    cos = float(torch.nn.functional.cosine_similarity(
        d_in.reshape(1, -1), d_out.reshape(1, -1)))
    return alpha, nsr, cos


# ---------------------------------------------------------------------------
# Purifier backends.  Each factory returns purify(wav, seed) -> wav.
# ---------------------------------------------------------------------------
def _ours(ckpt, t, sampler, device):
    """Our mel-domain purifier, wrapped to a waveform interface so it goes
    through the identical vocoder path as the foreign one. Without this the
    reference cell would enjoy a lossless pipeline and the comparison would be
    rigged in our favour."""
    dp = load_real_diffpure(checkpoint_path=ckpt, t_diffuse=t, device=device,
                            sampler=sampler)
    loop = MelDiffPure(dp.denoiser, n_steps=dp.n_steps, t_diffuse=t,
                       sampler=sampler).to(device).eval()

    @torch.no_grad()
    def purify(wav, seed=0):
        mel = wav_to_logmel(wav, device)
        return invert_mel(loop(mel, seed=seed).cpu()).to(wav.device)
    return purify


def _diffwave(ckpt, t, device):
    """Foreign backend: unconditional DiffWave on raw 16 kHz waveform.

    Two public checkpoints share this architecture (see models/diffwave_purifier.py):
    philsyn's SC09 weights and De-AntiFake's LibriSpeech weights. Its t is NOT our
    t -- its schedule is its own and the noise is added in the waveform domain,
    so t is calibrated by matching damage to unprotected audio, not by integer."""
    from models.diffwave_purifier import load_diffwave
    if not ckpt:
        raise SystemExit("--foreign-ckpt is required for the diffwave backend")
    net = load_diffwave(ckpt, t=t, device=device)

    @torch.no_grad()
    def purify(wav, seed=0):
        return net(wav, seed=seed)
    return purify


PURIFIERS = {"ours": _ours, "diffwave": _diffwave}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foreign", default="diffwave", help="key in PURIFIERS")
    ap.add_argument("--foreign-ckpt", default="", dest="foreign_ckpt")
    ap.add_argument("--foreign-t", type=int, default=0, dest="foreign_t",
                    help="diffusion strength for the foreign purifier; its "
                         "schedule differs from ours, so this is NOT our t")
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: EVAL_SUBSET)")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="shared reference selector + cross-session speaker set, "
                         "so this matches the crafting and cloning runs")
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--null-t", type=int, default=200, dest="null_t")
    ap.add_argument("--t", type=int, default=200, help="our purifier's strength")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v3.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--vocoder", default="gl", choices=["gl", "phase"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the model the directions are FOUND against (always ours -- that is the point)
    dp = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=args.t,
                            device=device, sampler=args.sampler)
    src = dp.denoiser
    for p in src.parameters():
        p.requires_grad_(False)
    ab_t = dp.alpha_bar[args.null_t - 1]

    ours = _ours(args.purifier_ckpt, args.t, args.sampler, device)
    foreign = PURIFIERS[args.foreign](args.foreign_ckpt, args.foreign_t, device)

    # Shared selector, same as crafting / cloning / eval_saved (fix 2026-09-19).
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             args.subset or EVAL_SUBSET, 4.0)
    if args.cross_session:
        from experiments.librispeech import reference_items, cross_session_split
        ref = reference_items(items, min_sec=4.0, utt_index=args.utt_index)
        ok = set(cross_session_split(items, min_sec=4.0))
        spks = [x for x in sorted(ref) if x in ok][
            args.spk_offset:args.spk_offset + args.n_files]
        ps = {x: {args.utt_index: ref[x]} for x in spks}
        print(f"[sel] cross-session: {len(spks)} speakers from "
              f"{args.subset or EVAL_SUBSET} (shared selector)")
    else:
        ps = per_speaker(items, n_utt=args.utt_index + 1)
        spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]

    rows = []
    for spk in spks:
        wav = load_flac(ps[spk][args.utt_index]["path"], 4.0).to(device)
        x = wav_to_logmel(wav, device)
        torch.manual_seed(1000 + int(spk))          # per speaker; see E7/section 15a
        d0 = torch.randn_like(x)
        d0 = d0 / d0.pow(2).mean().sqrt() * args.rms
        d_null = find_direction(src, x, d0, args.steps, args.lr, +1, ab_t=ab_t)

        def render(mel):
            if args.vocoder == "gl":
                return invert_mel(mel.cpu()).to(device)
            return attacker_resynth(mel, wav, method="phase")

        for name, d in (("random", d0), ("pass_through", d_null)):
            with torch.no_grad():
                w_o, w_p = render(x), render(x + d)
                # 1. the floor: what the vocoder round trip alone costs
                voc = decompose(d, wav_to_logmel(w_p, device) - wav_to_logmel(w_o, device))
                # 2. reference: our purifier, through the SAME pipeline
                o = decompose(d, wav_to_logmel(ours(w_p, seed=args.seed), device)
                                 - wav_to_logmel(ours(w_o, seed=args.seed), device))
                # 3. test: the foreign purifier
                f = decompose(d, wav_to_logmel(foreign(w_p, seed=args.seed), device)
                                 - wav_to_logmel(foreign(w_o, seed=args.seed), device))
                rt = r_t_score(src, x, d, ab_t, seed=spk)
            rows.append(dict(speaker=spk, direction=name, R_t_source=round(rt, 5),
                             foreign=args.foreign, vocoder=args.vocoder,
                             alpha_vocoder=round(voc[0], 4), nsr_vocoder=round(voc[1], 4),
                             alpha_ours=round(o[0], 4), nsr_ours=round(o[1], 4),
                             cos_ours=round(o[2], 4),
                             alpha_foreign=round(f[0], 4), nsr_foreign=round(f[1], 4),
                             cos_foreign=round(f[2], 4)))
            print(f"  spk{spk} {name:>13}: vocoder a={voc[0]:.3f} | "
                  f"ours a={o[0]:.3f} | foreign a={f[0]:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    g = df.groupby("direction")[["alpha_vocoder", "alpha_ours", "alpha_foreign"]].mean()
    print("\n" + "=" * 76)
    print(f"FOREIGN-PURIFIER TRANSFER  ({args.foreign}, vocoder={args.vocoder})")
    print("=" * 76)
    print(g.round(4).to_string())

    def adv(col):
        r = g.loc["random", col]
        return g.loc["pass_through", col] / r if abs(r) > 1e-9 else float("nan")

    a_voc, a_ours, a_for = adv("alpha_vocoder"), adv("alpha_ours"), adv("alpha_foreign")
    print("")
    print("  advantage of the pass-through direction over random,")
    print("  measured through each pipeline (the vocoder loss cancels in a ratio):")
    print(f"    vocoder round trip only : {a_voc:.2f}x   <- the FLOOR, and it is not 1x")
    print(f"    OUR purifier            : {a_ours:.2f}x   <- reference")
    print(f"    FOREIGN purifier        : {a_for:.2f}x   <- the test")

    # The vocoder floor is NOT 1x. E6 showed the round trip already favours the
    # pass-through direction (alpha 0.92 vs 0.68) because that direction is
    # smoother and survives mel inversion better. Crediting the purifier with that
    # head start would overstate the transfer, so the quantity to read is the
    # EXCESS over the floor: what the purifier contributes beyond the vocoder.
    e_ours = a_ours / max(a_voc, 1e-9)
    e_for = a_for / max(a_voc, 1e-9)
    frac = e_for / e_ours if e_ours > 1e-9 else float("nan")
    print("")
    print("  excess over the vocoder floor (what the PURIFIER contributes):")
    print(f"    OUR purifier            : {e_ours:.2f}x")
    print(f"    FOREIGN purifier        : {e_for:.2f}x")
    print(f"    the foreign purifier retains {100*frac:.0f} % of ours")
    print("")
    print("-" * 76)
    if e_for > 1.15 and frac >= 0.5:
        print("  -> TRANSFERS to a purifier we did not train, on someone else's data,")
        print("     in a different domain (raw waveform). The subspace is a property of")
        print("     speech diffusion purifiers, not only of our training recipe.")
    elif e_for > 1.15:
        print("  -> PARTIAL transfer: present but attenuated. State the attenuation")
        print("     explicitly and scope the claim to what the number supports.")
    else:
        print("  -> DOES NOT TRANSFER beyond the vocoder floor. The subspace is a")
        print("     property of OUR recipe. Publishable, and sharper than another seed:")
        print("     it bounds how far a purifier-specific construction generalises, and")
        print("     it directly qualifies E2.")
    print("")
    print("  NOTE: this pipeline is lossy by construction (mel -> wav -> mel), so its")
    print("  dynamic range is far smaller than E2's mel-domain R_t ratios. Compare the")
    print("  foreign cell to the OURS cell in this table, never to E2's 76x.")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
