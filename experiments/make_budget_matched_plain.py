"""
experiments/make_budget_matched_plain.py
========================================
N5a -- rescale the plain arm's perturbation, per speaker, to the exact
perturbation energy of the purifier-aware arm.

PREREG: paper/PREREG_N5_budget_matched_plain.md

WHY A RESCALE AND NOT A RE-CRAFT. RESULTS 59f found the purifier-aware arm
spends 2.89 dB more of the same budget allowance, which confounds "protects
better" with "perturbs more". The obvious fix -- re-craft plain at a bigger
budget -- cannot be relied on, because a per-frame diagnostic shows the plain
held-out arm is only 63 % saturated against the --snr-floor gate (the
purifier-aware arm is 84 %): in a third of its frames the optimiser has
already converged, and no budget increase makes a converged optimiser spend
more. That re-craft is N5b and runs separately.

The rescale answers the question exactly and cheaply:

    y' = x + s * (y - x),   s chosen per speaker so SNR(y') == SNR(eot arm)

and it is deliberately GENEROUS TO PLAIN. Scaling pushes frames above the
psychoacoustic masking cap the crafting respected, so this arm is more
audible than any legitimate plain arm at the same SNR. If it still loses, the
magnitude explanation is dead; if it wins, the explanation is live and §59c
is withdrawn (prereg 5-C).

Because it is no longer masking-constrained, the script REPORTS what the
rescale broke: PESQ, and the share of frames pushed past the local SNR floor.
Those numbers belong in the paper beside the result.

Run:
  python experiments/make_budget_matched_plain.py
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
import soundfile as sf
import torch

from experiments.librispeech import load_flac
from experiments.exp_purifier_aware import WAV_DIR

PROMPTS = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")
OUT_CSV = os.path.join(PROJ, "experiments", "results", "n5a_rescale.csv")
SR = 16000
NFFT, HOP = 1024, 256
FLOOR_DB = 15.0          # the --snr-floor the arms were crafted under


def snr_db(ref, y):
    d = y - ref
    return 10.0 * np.log10((ref ** 2).sum() / max((d ** 2).sum(), 1e-20))


def floor_violation(ref, y, floor_db=FLOOR_DB):
    """Share of non-silent frames whose delta exceeds frame_energy - floor.

    The crafting enforces this per frame; a global rescale does not, so this
    is exactly what the rescale breaks and it has to be reported.
    """
    win = torch.hann_window(NFFT)
    D = torch.stft(torch.from_numpy((y - ref).astype(np.float32)), NFFT, HOP,
                   window=win, return_complex=True)
    S = torch.stft(torch.from_numpy(ref.astype(np.float32)), NFFT, HOP,
                   window=win, return_complex=True)
    ed = D.abs().pow(2).sum(0)
    es = S.abs().pow(2).sum(0)
    keep = es > es.max() * 1e-6
    if keep.sum() == 0:
        return float("nan")
    local = 10 * torch.log10((es[keep] + 1e-20) / (ed[keep] + 1e-20))
    return float((local < floor_db - 1e-6).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--plain", default="plain_m30_f15_nodvec")
    ap.add_argument("--target", default="fullnsT_m30_f15_nodvec",
                    help="arm whose realised SNR the plain arm is matched to")
    ap.add_argument("--tag", default="plain_m30_f15_nodvec_rescaled")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    try:
        from pesq import pesq as pesq_fn
    except Exception as e:                                   # noqa: BLE001
        print(f"[n5a] pesq unavailable ({e}); SNR only")
        pesq_fn = None

    prompts = pd.read_csv(args.prompts).drop_duplicates("speaker")
    rows, n_written, n_clip = [], 0, 0
    for _, r in prompts.iterrows():
        name = r["name"]
        o = load_flac(os.path.join(PROJ, r["ref_path"]), 4.0).squeeze(0).numpy()
        pp = os.path.join(WAV_DIR, f"{name}_{args.plain}.wav")
        tp = os.path.join(WAV_DIR, f"{name}_{args.target}.wav")
        if not (os.path.isfile(pp) and os.path.isfile(tp)):
            print(f"  missing arm for {name}")
            continue
        yp, _ = sf.read(pp, dtype="float32")
        yt, _ = sf.read(tp, dtype="float32")
        n = min(len(o), len(yp), len(yt))
        ref, yp, yt = o[:n], yp[:n], yt[:n]

        snr_p, snr_t = snr_db(ref, yp), snr_db(ref, yt)
        d = yp - ref
        e_d = (d ** 2).sum()
        e_target = (ref ** 2).sum() / (10.0 ** (snr_t / 10.0))
        s = float(np.sqrt(e_target / max(e_d, 1e-20)))
        y_new = ref + s * d

        peak = float(np.abs(y_new).max())
        clipped = peak > 1.0
        n_clip += clipped
        if clipped:                       # preserve the SNR, report the peak
            print(f"  {name}: peak {peak:.3f} > 1.0 after rescale "
                  f"(NOT normalised; would change the matched SNR)")

        snr_new = snr_db(ref, y_new)
        rows.append(dict(
            speaker=r["speaker"], name=name, scale=s,
            snr_plain=snr_p, snr_target=snr_t, snr_rescaled=snr_new,
            snr_err=snr_new - snr_t, peak=peak, clipped=bool(clipped),
            pesq_plain=(pesq_fn(SR, ref, yp, "wb") if pesq_fn else np.nan),
            pesq_target=(pesq_fn(SR, ref, yt, "wb") if pesq_fn else np.nan),
            pesq_rescaled=(pesq_fn(SR, ref, y_new.astype(np.float32), "wb")
                           if pesq_fn else np.nan),
            floorviol_plain=floor_violation(ref, yp),
            floorviol_target=floor_violation(ref, yt),
            floorviol_rescaled=floor_violation(ref, y_new)))

        if not args.dry_run:
            sf.write(os.path.join(WAV_DIR, f"{name}_{args.tag}.wav"),
                     y_new.astype(np.float32), SR)
            n_written += 1

    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    print(f"\n[n5a] {len(d)} speakers -> {args.out}")
    if not args.dry_run:
        print(f"[n5a] {n_written} wavs written as *_{args.tag}.wav")

    print("\n=== THE MATCHING CHECK (prereg 4: must be within 0.5 dB) ===")
    print(f"  plain      mean SNR {d.snr_plain.mean():7.3f} dB")
    print(f"  target     mean SNR {d.snr_target.mean():7.3f} dB")
    print(f"  rescaled   mean SNR {d.snr_rescaled.mean():7.3f} dB")
    err = d.snr_err.abs().max()
    print(f"  worst per-speaker error {err:.4f} dB  -> "
          f"{'MATCHED' if err < 0.5 else 'MATCH FAILED (prereg 5-D)'}")
    print(f"  scale factor: mean {d.scale.mean():.3f}, "
          f"range {d.scale.min():.3f}-{d.scale.max():.3f}")
    if n_clip:
        print(f"  {n_clip} speakers clip after rescale (peak > 1.0)")

    print("\n=== WHAT THE RESCALE BROKE (report beside the result) ===")
    print(f"  PESQ        plain {d.pesq_plain.mean():.3f} -> "
          f"rescaled {d.pesq_rescaled.mean():.3f}   "
          f"(purifier-aware target {d.pesq_target.mean():.3f})")
    print(f"  frames past the {FLOOR_DB:.0f} dB floor: "
          f"plain {100*d.floorviol_plain.mean():.1f} % -> "
          f"rescaled {100*d.floorviol_rescaled.mean():.1f} %   "
          f"(target {100*d.floorviol_target.mean():.1f} %)")
    print("\n  A rescaled arm above the target's PESQ and floor-violation rate")
    print("  is MORE perturbed than any legitimate plain arm at this SNR,")
    print("  which is the generous-to-plain direction the prereg intends.")


if __name__ == "__main__":
    main()
