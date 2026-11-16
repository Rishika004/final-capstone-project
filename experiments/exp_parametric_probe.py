"""
experiments/exp_parametric_probe.py
===================================
P1 — can ON-MANIFOLD parametric transformations protect where additive
perturbations cannot?

Motivation (2026-09-15). Additive imperceptible perturbations are noise-like,
hence off the speech manifold, hence exactly what a diffusion purifier removes.
Our own data: survival and protection anti-correlate (r = -0.86). A perturbation
that survives purification must look like plausible speech to the purifier.

Instead of searching the tangent space for pass-through directions (which found
directions that pass through but do not attack), we CONSTRUCT points that are on
the manifold by construction: speech -> speech transformations an attacker's
purifier has no basis to undo, because the output already is valid speech.

    tilt    broadband spectral tilt (dB across the band)
    warp    formant / spectral-envelope frequency warping (factor)
    eq      a few broad peaking-EQ bumps (dB)
    reverb  short synthetic room impulse response (RT60 in ms)
    pitch   small pitch shift (semitones)

For each transform and magnitude we measure, on held-out speakers, all three
quantities that matter at once:

  survival    mel PSR = ||P(mel(x')) - P(mel(x))|| / ||mel(x') - mel(x)||  and
              the direction cosine, through the real purifier (v3 + DDIM).
              ~1.0 = the purifier cannot undo the transformation.
  attack      cosine to the clean enrolment embedding on WavLM-SV / ECAPA /
              d-vector, BEFORE and AFTER the attacker's purification, against
              EER thresholds calibrated through that same channel.
  cost        SNR, PESQ, STOI (all reference-based, and all expected to look
              terrible for an inaudible formant warp -- that mismatch is itself
              a finding; no-reference MOS and a listening test come next).

Run:
  python experiments/exp_parametric_probe.py --n-files 10
Output:
  experiments/results/parametric_probe.csv
"""
import sys, io, os, argparse, math, time
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
import torch._dynamo  # noqa: F401  (before speechbrain; see exp_purifier_aware.py)
import transformers  # noqa: F401
import torch.nn.functional as F

from evaluation.evaluate import wav_to_logmel, save_wav, SAMPLE_RATE, N_FFT, HOP_LENGTH
from models.real_diffpure import load_real_diffpure, MelDiffPure
from experiments.exp_purifier_aware import (build_encoders, real_embed_all, cs,
                                            attacker_resynth, quality)
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET

OUT = os.path.join(PROJ, "experiments", "results", "parametric_probe.csv")
WAV_DIR = os.path.join(PROJ, "experiments", "results", "parametric")
os.makedirs(WAV_DIR, exist_ok=True)

# EER thresholds through the attacker's channel (purifier v3 + DDIM)
THR = {50:  {"WavLM-SV": 0.897, "ECAPA-TDNN": 0.346, "Resemblyzer d-vector": 0.735},
       200: {"WavLM-SV": 0.896, "ECAPA-TDNN": 0.322, "Resemblyzer d-vector": 0.723}}


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------
def _stft(wav, win):
    return torch.stft(wav.squeeze(0), N_FFT, HOP_LENGTH, window=win, return_complex=True)


def _istft(spec, win, length):
    return torch.istft(spec, N_FFT, HOP_LENGTH, window=win, length=length).unsqueeze(0)


def _apply_gain_curve(wav, gain_lin, win):
    """Multiply the STFT magnitude by a per-bin gain, keep phase."""
    S = _stft(wav, win)
    return _istft(S * gain_lin.unsqueeze(1), win, wav.shape[-1])


# ---------------------------------------------------------------------------
# On-manifold transformations
# ---------------------------------------------------------------------------
def t_tilt(wav, db, win, **kw):
    """Broadband spectral tilt: `db` dB difference between DC and Nyquist."""
    F_ = N_FFT // 2 + 1
    f = torch.linspace(0, 1, F_, device=wav.device)
    return _apply_gain_curve(wav, 10 ** (db * f / 20.0), win)


def t_eq(wav, db, win, seed=0, n_peaks=3, **kw):
    """A few broad peaking-EQ bumps of +-db, at fixed pseudo-random centres."""
    F_ = N_FFT // 2 + 1
    f = torch.linspace(0, 1, F_, device=wav.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    curve_db = torch.zeros(F_, device=wav.device)
    for k in range(n_peaks):
        c = float(torch.rand(1, generator=g)) * 0.8 + 0.05      # centre
        w = float(torch.rand(1, generator=g)) * 0.12 + 0.06     # width
        s = 1.0 if k % 2 == 0 else -1.0
        curve_db = curve_db + s * db * torch.exp(-0.5 * ((f - c) / w) ** 2)
    return _apply_gain_curve(wav, 10 ** (curve_db / 20.0), win)


def t_warp(wav, alpha, win, **kw):
    """Frequency-axis warp of the STFT magnitude: mag'[f] = mag[f/alpha].
    alpha > 1 pushes formants up. Phase is kept (mild artefacts, fine here)."""
    S = _stft(wav, win)
    mag, phase = S.abs(), S / (S.abs() + 1e-8)
    F_ = mag.shape[0]
    src = torch.arange(F_, device=wav.device, dtype=torch.float32) / alpha
    lo = src.floor().clamp(0, F_ - 1).long()
    hi = (lo + 1).clamp(0, F_ - 1)
    frac = (src - lo.float()).clamp(0, 1).unsqueeze(1)
    mag_w = mag[lo] * (1 - frac) + mag[hi] * frac
    return _istft(mag_w * phase, win, wav.shape[-1])


def t_reverb(wav, rt60_ms, win, seed=0, drr_db=12.0, **kw):
    """Short synthetic room response: direct path + exponentially decaying noise
    tail at `drr_db` below the direct sound."""
    sr = SAMPLE_RATE
    n = max(int(sr * rt60_ms / 1000.0), 8)
    g = torch.Generator(device="cpu").manual_seed(seed)
    tail = torch.randn(n, generator=g).to(wav.device)
    decay = torch.exp(-6.9 * torch.arange(n, device=wav.device) / n)   # -60 dB at n
    tail = tail * decay
    tail = tail / (tail.norm() + 1e-9) * (10 ** (-drr_db / 20.0))
    rir = torch.cat([torch.ones(1, device=wav.device), tail])
    y = F.conv1d(wav.unsqueeze(0), rir.flip(0).view(1, 1, -1), padding=rir.numel() - 1)
    y = y.squeeze(0)[:, : wav.shape[-1]]
    return y / (y.abs().max() + 1e-8) * wav.abs().max()


def t_mcadams(wav, alpha, win, frame_len=320, hop=160, order=16, **kw):
    """McAdams-coefficient formant shifting -- the VoicePrivacy B2 anonymisation
    baseline (Patino et al., arXiv 2011.01130), and the PROPER way to warp
    formants: per-frame LPC analysis, raise each pole's phase to the power
    `alpha`, resynthesise by filtering the LPC residual through the modified
    filter. Unlike `t_warp` (which warps the STFT magnitude and keeps the
    original phase, introducing artefacts) this stays a valid source-filter
    speech signal, so it is the fair test of whether formant shifting can be
    both purification-invariant and natural.

    2026-10-10: the implementation moved to `utils/onmanifold.py` so that the
    Step 0 listening stimuli and the Step 1 arms are produced by the SAME code
    (PLAN_V4). This wrapper keeps the probe's signature and behaviour; the
    silent / failed / unstable frame counts are available through `stats=`.
    """
    from utils.onmanifold import t_mcadams as _mc
    return _mc(wav, alpha, win, frame_len=frame_len, hop=hop, order=order, **kw)


def t_pitch(wav, semitones, win, **kw):
    import torchaudio
    return torchaudio.functional.pitch_shift(wav, SAMPLE_RATE, float(semitones),
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)


TRANSFORMS = {
    # Magnitudes widened after the 2-speaker smoke run (2026-09-15): at the mild
    # end every transform passes through the purifier (R_t ~ 0.005, PSR ~ 0.95)
    # but protects on ~0 % of speakers, so the sweep has to reach the magnitudes
    # where protection actually appears -- that is the whole point of the curve.
    "tilt":   (t_tilt,   [-18.0, -12.0, -6.0, 6.0, 12.0, 18.0], "dB across band"),
    "eq":     (t_eq,     [4.0, 8.0, 12.0, 16.0],                "dB peaks"),
    "warp":   (t_warp,   [0.88, 0.91, 0.94, 1.06, 1.09, 1.12],  "envelope factor"),
    "reverb": (t_reverb, [50.0, 150.0, 300.0, 600.0],           "RT60 ms"),
    # McAdams: the proper LPC formant shift (VoicePrivacy B2). alpha<1 and >1
    # both move formants; 0.8-0.9 is the published anonymisation range.
    "mcadams": (t_mcadams, [0.80, 0.85, 0.90, 0.95, 1.05, 1.10], "McAdams alpha"),
    # "pitch" needs torchaudio.functional.pitch_shift, which is unavailable
    # while Smart App Control blocks libtorchaudio.pyd. Excluded by default.
    "pitch":  (t_pitch,  [-0.75, -0.35, 0.35, 0.75],    "semitones"),
}


# ---------------------------------------------------------------------------
# The null-space instrument: how far inside the purifier's pass-through
# subspace does this protection sit?  One denoiser pass per noise draw, no
# 50-step loop -- so it is a CHEAP PREDICTOR of whether purification will
# strip the protection.
#   R_t(d) = E_eps || m(x_t + sqrt(ab_t) d) - m(x_t) - d ||^2 / ||d||^2
#   x_t    = sqrt(ab_t) mel(x) + sqrt(1-ab_t) eps
# R_t -> 0 : the denoiser passes d through untouched (d in null(J_C) at this
#            noise level).  R_t -> 1 : the denoiser removes d entirely.
# ---------------------------------------------------------------------------
@torch.no_grad()
def pass_through_score(model, mel_o, mel_p, ab_t, n_eps=4, seed=0):
    d = mel_p - mel_o
    nrm = d.pow(2).sum() + 1e-12
    g = torch.Generator(device=mel_o.device).manual_seed(seed)
    acc = 0.0
    for _ in range(n_eps):
        eps = torch.randn(mel_o.shape, generator=g, device=mel_o.device)
        x_t = ab_t.sqrt() * mel_o + (1 - ab_t).sqrt() * eps
        shift = model(x_t + ab_t.sqrt() * d) - model(x_t)
        acc += float((shift - d).pow(2).sum() / nrm)
    return acc / n_eps


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=10, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--t", default="50,200")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--transforms", default="tilt,eq,warp,reverb")
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v3.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    ap.add_argument("--save-wavs", type=int, default=2, dest="save_wavs",
                    help="save audio for the first N speakers (for listening)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in args.t.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    win = torch.hann_window(N_FFT, device=device)
    names = [n for n in args.transforms.split(",") if n in TRANSFORMS]
    print(f"[P1] device={device} | transforms={names} | t={ts} | seeds={seeds}")

    encs = build_encoders(device)
    base = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=ts[0],
                              device=device, sampler=args.sampler)
    dps = {t: MelDiffPure(base.denoiser, n_steps=base.n_steps, t_diffuse=t,
                          sampler=args.sampler).to(device).eval() for t in ts}

    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=args.utt_index + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]

    rows = []
    for si, spk in enumerate(spks):
        it = ps[spk][args.utt_index]
        wav_o = load_flac(it["path"], 4.0).to(device)
        mel_o = wav_to_logmel(wav_o, device)
        emb_o = real_embed_all(encs, wav_o)
        if si < args.save_wavs:
            save_wav(wav_o.cpu(), os.path.join(WAV_DIR, f"spk{spk}_original.wav"))

        # unprotected control through the attacker's channel (once per t/seed)
        ctrl = {}
        for t in ts:
            acc = {n: [] for n in THR[t]}
            for sd in seeds:
                with torch.no_grad():
                    mel_op = dps[t](mel_o, seed=sd)
                e = real_embed_all(encs, attacker_resynth(mel_op, wav_o, "gl"))
                for n in THR[t]:
                    acc[n].append(cs(emb_o[n], e[n]))
            ctrl[t] = {n: float(np.mean(v)) for n, v in acc.items()}

        for tname in names:
            fn, mags, unit = TRANSFORMS[tname]
            for mag in mags:
                t0 = time.time()
                try:
                    with torch.no_grad():
                        wav_p = fn(wav_o, mag, win, seed=spk).clamp(-1, 1)
                except Exception as exc:
                    print(f"  spk{spk} {tname} {mag}: FAILED ({exc})", flush=True)
                    continue
                wav_p = wav_p[:, : wav_o.shape[-1]]
                mel_p = wav_to_logmel(wav_p, device)
                emb_p = real_embed_all(encs, wav_p)
                snr, pesq_v, stoi_v = quality(wav_o, wav_p)
                if si < args.save_wavs:
                    save_wav(wav_p.cpu(), os.path.join(WAV_DIR, f"spk{spk}_{tname}_{mag}.wav"))

                for t in ts:
                    after = {n: [] for n in THR[t]}
                    psr, dcos, alph, nsr = [], [], [], []
                    rt = pass_through_score(base.denoiser, mel_o, mel_p,
                                            base.alpha_bar[t - 1], seed=spk)
                    for sd in seeds:
                        with torch.no_grad():
                            mel_pp = dps[t](mel_p, seed=sd)
                            mel_op = dps[t](mel_o, seed=sd)
                            d_in = (mel_p - mel_o).reshape(1, -1)
                            d_out = (mel_pp - mel_op).reshape(1, -1)
                            psr.append((d_out.norm() / (d_in.norm() + 1e-12)).item())
                            dcos.append(F.cosine_similarity(d_in, d_out).item())
                            # signal gain: d_out = alpha * d_in + residual. A norm
                            # ratio cannot separate "survived" from "destroyed and
                            # replaced by equal-sized noise" (RESULTS section 11a).
                            n2 = d_in.pow(2).sum() + 1e-12
                            a = ((d_out * d_in).sum() / n2).item()
                            alph.append(a)
                            nsr.append(((d_out - a * d_in).norm()
                                        / (abs(a) * d_in.norm() + 1e-12)).item())
                        e = real_embed_all(encs, attacker_resynth(mel_pp, wav_p, "gl"))
                        for n in THR[t]:
                            after[n].append(cs(emb_o[n], e[n]))
                    for n, tau in THR[t].items():
                        ca, cc = float(np.mean(after[n])), ctrl[t][n]
                        rows.append(dict(
                            speaker=spk, transform=tname, magnitude=mag, unit=unit,
                            eval_t=t, encoder=n, threshold=tau,
                            pass_through_R=round(rt, 4),
                            mel_psr=round(float(np.mean(psr)), 4),
                            mel_dircos=round(float(np.mean(dcos)), 4),
                            mel_alpha=round(float(np.mean(alph)), 4),
                            mel_nsr=round(float(np.mean(nsr)), 4),
                            cos_before=round(cs(emb_o[n], emb_p[n]), 4),
                            cos_after_purify=round(ca, 4),
                            cos_ctrl_unprot=round(cc, 4),
                            clone_fails_after=bool(ca < tau),
                            ctrl_identity_kept=bool(cc >= tau),
                            defence_valid=bool(ca < tau and cc >= tau),
                            snr_db=round(snr, 2), pesq=round(pesq_v, 3),
                            stoi=round(stoi_v, 4)))
                print(f"  spk{spk} {tname:6s} {mag:>6} | R_t={rt:.3f} "
                      f"PSR={np.mean(psr):.3f} cos={np.mean(dcos):.3f} | "
                      f"PESQ={pesq_v:.2f} SNR={snr:.1f} | {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 90)
    print("SURVIVAL through the purifier (1.0 = the purifier cannot undo it)")
    print("=" * 90)
    print(df.drop_duplicates(["speaker", "transform", "magnitude", "eval_t"])
            .groupby(["transform", "magnitude", "eval_t"])[["pass_through_R", "mel_alpha", "mel_nsr", "mel_psr", "mel_dircos"]]
            .mean().round(3).to_string())

    print("\n" + "=" * 90)
    print("PROTECTION after the attacker's purification (% of speakers)")
    print("=" * 90)
    piv = df.pivot_table(index=["transform", "magnitude", "eval_t"], columns="encoder",
                         values="defence_valid", aggfunc=lambda v: 100 * np.mean(v))
    print(piv.round(0).to_string())

    print("\n" + "=" * 90)
    print("PERCEPTUAL COST (reference-based; expected to understate naturalness)")
    print("=" * 90)
    print(df.drop_duplicates(["speaker", "transform", "magnitude"])
            .groupby(["transform", "magnitude"])[["pesq", "stoi", "snr_db"]]
            .mean().round(2).to_string())
    print(f"\nsaved {args.out}  |  audio in {WAV_DIR}")


if __name__ == "__main__":
    main()
