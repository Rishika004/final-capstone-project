"""
experiments/exp_robust_craft.py
===============================
Prototype of the real fix: craft the perturbation so it is adversarial AFTER
the attacker purifies (EOT / BPDA over the purifier), instead of only before.

Baseline  (current):  minimise  cos( E(x+δ),                 E(x) )
Robust    (proposed): minimise  cos( E(purify(x+δ)),         E(x) )  + λ·cos( E(x+δ), E(x) )

"purify" here is the single-pass surrogate denoiser (fast, differentiable) — the
same UNet backbone the real DiffPure DDPM loop is built on, so robustness to it
should transfer to the full loop. Both variants use the psychoacoustic masking
projection so audio quality stays high.

We then evaluate BOTH variants end to end against the REAL DiffPure loop
(t=50) + Griffin-Lim, exactly as an attacker would, and report WavLM speaker
similarity after purification (the number that decides the paper's claim).

Run:
  python experiments/exp_robust_craft.py --file sample_001.wav --steps 120
Output:
  experiments/results/robust_craft.csv
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
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pathlib import Path

from core.psychoacoustic import (PsychoacousticMasker, project_delta_to_masking,
                                 waveform_snr_db)
from core.optimizer import _build_mel_to_wav, _mel_to_wav_differentiable
from evaluation.evaluate import load_wav, wav_to_logmel, save_wav, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH
from utils.vocoder import invert_mel
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import (build_all, ECAPAEnc, DvecEnc,
                                        real_embed_wavlm, real_embed_ecapa, real_embed_dvec)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(OUT_DIR, "robust_craft")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(WAV_DIR, exist_ok=True)
DIFFPURE_SEED = 2026


def craft(wav_o, encoders, masker, cap, denoiser, bandlimit_hz,
          robust, n_steps, lr, lam_before=0.3, seed=42):
    """
    Ensemble masked PGD. If robust=True, the loss also passes (x+δ) through the
    single-pass surrogate denoiser before the encoders, so δ must fool the
    encoder on the PURIFIED signal too.
    """
    device = wav_o.device
    orig = [e.embed(wav_o).detach() for e in encoders]

    # differentiable mel->wav (fixed original phase) for the purified branch
    inv_mel_scale, phase_orig, win, wav_len, T_frames = _build_mel_to_wav(
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE, x_wav=wav_o, device=device)
    mel_tf = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                              hop_length=HOP_LENGTH, n_mels=N_MELS).to(device)

    def purified_wav(wav_p):
        mel_p = torch.log(mel_tf(wav_p) + 1e-9)
        mel_pure = denoiser(mel_p).clamp(-10, 10)
        return _mel_to_wav_differentiable(mel_pure, inv_mel_scale, phase_orig,
                                          win, N_FFT, HOP_LENGTH, T_frames, wav_len)

    torch.manual_seed(seed)
    delta = ((torch.rand_like(wav_o) * 2 - 1) * 2e-3).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_score, best_delta = float("inf"), delta.detach().clone()

    for s in range(1, n_steps + 1):
        opt.zero_grad()
        wav_p = wav_o.detach() + delta
        if robust:
            wav_pure = purified_wav(wav_p)
            coss_after = torch.stack([(e.embed(wav_pure) * o).sum()
                                      for e, o in zip(encoders, orig)])
            coss_before = torch.stack([(e.embed(wav_p) * o).sum()
                                       for e, o in zip(encoders, orig)])
            loss = (torch.logsumexp(coss_after * 4.0, dim=0) / 4.0
                    + lam_before * torch.logsumexp(coss_before * 4.0, dim=0) / 4.0)
            score = coss_after.max().item()
        else:
            coss = torch.stack([(e.embed(wav_p) * o).sum()
                                for e, o in zip(encoders, orig)])
            loss = torch.logsumexp(coss * 4.0, dim=0) / 4.0
            score = coss.max().item()
        loss.backward()
        with torch.no_grad():
            if score < best_score:
                best_score, best_delta = score, delta.detach().clone()
        opt.step()
        with torch.no_grad():
            delta.data.copy_(project_delta_to_masking(
                delta.data, cap, masker, bandlimit_hz=bandlimit_hz).to(device))
        if s % 30 == 0 or s == 1:
            print(f"    {'robust' if robust else 'base '} step {s:3d}/{n_steps} | "
                  f"train-score={score:+.3f}")
    best_delta = project_delta_to_masking(best_delta, cap, masker,
                                          bandlimit_hz=bandlimit_hz).to(device)
    return best_delta


def cs(a, b):
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()


def eval_e2e(tag, wav_o, wav_p, diffpure, wavlm, ecapa_clf, dvec_model, device):
    mel_p = wav_to_logmel(wav_p, device)
    mel_o = wav_to_logmel(wav_o, device)
    with torch.no_grad():
        mel_pp = diffpure(mel_p, seed=DIFFPURE_SEED)
        mel_op = diffpure(mel_o, seed=DIFFPURE_SEED)
    wav_pp = invert_mel(mel_pp.cpu()).to(device)
    wav_op = invert_mel(mel_op.cpu()).to(device)

    def trio(w):
        return (cs(real_embed_wavlm(wavlm, wav_o), real_embed_wavlm(wavlm, w)),
                cs(real_embed_ecapa(ecapa_clf, wav_o), real_embed_ecapa(ecapa_clf, w)),
                cs(real_embed_dvec(dvec_model, wav_o), real_embed_dvec(dvec_model, w)))
    wb = trio(wav_p)     # before purify
    wa = trio(wav_pp)    # after purify (protected)
    wc = trio(wav_op)    # after purify (control, unprotected)
    snr = waveform_snr_db(wav_o.cpu(), wav_p.cpu())
    x = wav_o.squeeze().cpu().numpy(); y = wav_p.squeeze().cpu().numpy()[:x.shape[0]]
    try:
        from pesq import pesq as pf; pesq_v = pf(16000, x, y, "wb")
    except Exception: pesq_v = float("nan")
    print(f"  [{tag}] WavLM  before={wb[0]:+.3f} after_purify={wa[0]:+.3f} ctrl={wc[0]:+.3f} (τ0.50)")
    print(f"  [{tag}] ECAPA  before={wb[1]:+.3f} after_purify={wa[1]:+.3f} ctrl={wc[1]:+.3f} (τ0.25)")
    print(f"  [{tag}] dvec   before={wb[2]:+.3f} after_purify={wa[2]:+.3f} ctrl={wc[2]:+.3f} (τ0.75)")
    print(f"  [{tag}] PESQ={pesq_v:.2f} SNR={snr:.1f}dB")
    return dict(variant=tag,
                wavlm_before=round(wb[0], 4), wavlm_after=round(wa[0], 4), wavlm_ctrl=round(wc[0], 4),
                ecapa_before=round(wb[1], 4), ecapa_after=round(wa[1], 4), ecapa_ctrl=round(wc[1], 4),
                dvec_before=round(wb[2], 4), dvec_after=round(wa[2], 4), dvec_ctrl=round(wc[2], 4),
                pesq=round(pesq_v, 3), snr_db=round(snr, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="sample_001.wav")
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--margin-db", type=float, default=30.0, dest="margin_db")
    ap.add_argument("--bandlimit", type=float, default=6000.0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RobustCraft] device={device} | file={args.file} | steps={args.steps}")

    wav_o = load_wav(os.path.join(args.input, args.file)).to(device)
    diffpure = load_real_diffpure(t_diffuse=50, device=device)
    denoiser = diffpure.denoiser          # single-pass surrogate (fast, differentiable)
    wavlm = load_wavlm(device=device)
    encoders = build_all(device, wavlm, SAVEDIR)
    ecapa_clf = [e for e in encoders if isinstance(e, ECAPAEnc)][0].clf
    dvec_model = [e for e in encoders if isinstance(e, DvecEnc)][0].dvec
    masker = PsychoacousticMasker(margin_db=args.margin_db, device=device)
    cap = masker.magnitude_cap(wav_o)

    rows = []
    for robust in (False, True):
        tag = "robust" if robust else "baseline"
        print(f"\n--- craft: {tag} ---")
        delta = craft(wav_o, encoders, masker, cap, denoiser, args.bandlimit,
                      robust=robust, n_steps=args.steps, lr=args.lr)
        wav_p = (wav_o + delta).clamp(-1.0, 1.0)
        save_wav(wav_p.cpu(), os.path.join(WAV_DIR, f"{Path(args.file).stem}_{tag}.wav"))
        rows.append(eval_e2e(tag, wav_o, wav_p, diffpure, wavlm, ecapa_clf, dvec_model, device))
        rows[-1]["file"] = args.file

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "robust_craft.csv")
    df.to_csv(csv, index=False)
    print(f"\n[RobustCraft] saved {csv}")
    print("=" * 70)
    print("Does crafting THROUGH the purifier keep WavLM broken after purify?")
    print(df[["variant", "wavlm_after", "wavlm_ctrl", "ecapa_after", "dvec_after",
              "pesq"]].to_string(index=False))
    print("=" * 70)
    print("Goal: robust wavlm_after << baseline wavlm_after, and << wavlm_ctrl.")


if __name__ == "__main__":
    main()
