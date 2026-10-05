"""
experiments/exp_perceptual.py
=============================
A/B experiment: flat-budget ensemble PGD (current baseline) vs
psychoacoustically-masked ensemble PGD (proposed fix for audible noise).

  flat : signed-gradient PGD, |delta_wav|_inf <= 0.05  (exp_transfer_multi.py)
         -> strong protection but audible broadband hiss (~24 dB SNR,
            half the noise energy above 4 kHz).

  psy  : Adam PGD where after EVERY step the delta's STFT magnitude is
         clamped below the frequency-masking threshold of the original
         speech (core/psychoacoustic.py). The perturbation hides under the
         voice's own spectral energy -> sounds like the original recording.
         --margin-db raises the cap (louder, stronger protection);
         --bandlimit additionally zeroes delta energy above a cutoff, which
         also restores robustness to resample/low-pass attacks.

Both variants are evaluated identically and honestly:
  * real (non-proxy) cos-sim on WavLM + ECAPA + Resemblyzer d-vector
  * PRR through the real DiffPure loop — for the null-projected mel delta
    (the paper's protocol) AND for the mel delta of the audio actually
    delivered (prr_actual)
  * waveform SNR, noise level in silent frames, PESQ-WB, STOI

Run:
  python experiments/exp_perceptual.py --n-files 3 --steps 300
Output:
  experiments/results/perceptual.csv
  experiments/results/perceptual/<file>_{flat,psy}.wav
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
from pathlib import Path

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import compute_null_space_factored, project_onto_null_space
from core.psychoacoustic import (PsychoacousticMasker, project_delta_to_masking,
                                 waveform_snr_db, silence_noise_db)
from evaluation.evaluate import load_wav, wav_to_logmel, save_wav
from evaluation.metrics import purification_resistance_rate
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import (build_all, ECAPAEnc, DvecEnc,
                                        real_embed_wavlm, real_embed_ecapa,
                                        real_embed_dvec)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(OUT_DIR, "perceptual")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(WAV_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Attack variants
# ---------------------------------------------------------------------------

def ensemble_loss(wav_p, encoders, orig_embs):
    coss = [(e.embed(wav_p) * o).sum() for e, o in zip(encoders, orig_embs)]
    coss_t = torch.stack(coss)
    # smooth worst-case: targets the least-fooled encoder but keeps all in grad
    loss = torch.logsumexp(coss_t * 4.0, dim=0) / 4.0
    return loss, coss_t


def flat_pgd(wav_o, encoders, epsilon_wav=0.05, n_steps=80, seed=42):
    """Baseline: signed-gradient PGD with a flat L-inf budget (current method)."""
    orig = [e.embed(wav_o).detach() for e in encoders]
    torch.manual_seed(seed)
    delta = ((torch.rand_like(wav_o) * 2 - 1) * epsilon_wav * 0.1)
    delta = delta.detach().requires_grad_(True)
    step = epsilon_wav / max(n_steps // 2, 1)

    best_score, best_delta = float("inf"), delta.detach().clone()
    for s in range(1, n_steps + 1):
        if delta.grad is not None:
            delta.grad.zero_()
        loss, coss_t = ensemble_loss(wav_o.detach() + delta, encoders, orig)
        loss.backward()
        with torch.no_grad():
            score = coss_t.max().item()
            if score < best_score:
                best_score, best_delta = score, delta.detach().clone()
            delta.data -= step * delta.grad.sign()
            delta.data.clamp_(-epsilon_wav, epsilon_wav)
        if s % 20 == 0 or s == 1:
            print(f"    [flat] step {s:3d}/{n_steps} | max-cos={score:+.3f}")
    print(f"    [flat] done | best max-cos={best_score:+.3f}")
    return best_delta


def psy_pgd(wav_o, encoders, masker, cap, n_steps=300, lr=1e-3,
            bandlimit_hz=None, seed=42):
    """
    Proposed: Adam PGD projected onto the masking constraint after every step.
    The delta can only live where the voice itself masks it.
    """
    device = wav_o.device
    orig = [e.embed(wav_o).detach() for e in encoders]
    torch.manual_seed(seed)
    # Small random init (NOT zeros): at delta=0 every cos=1.0 is a maximum, so
    # the gradient vanishes and Adam can stall at the origin.
    delta = ((torch.rand_like(wav_o) * 2 - 1) * 2e-3).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)

    best_score, best_delta = float("inf"), delta.detach().clone()
    for s in range(1, n_steps + 1):
        opt.zero_grad()
        loss, coss_t = ensemble_loss(wav_o.detach() + delta, encoders, orig)
        loss.backward()
        with torch.no_grad():
            # score corresponds to the CURRENT (already projected) delta
            score = coss_t.max().item()
            if score < best_score:
                best_score, best_delta = score, delta.detach().clone()
        opt.step()
        with torch.no_grad():
            proj = project_delta_to_masking(
                delta.data, cap, masker, bandlimit_hz=bandlimit_hz,
            ).to(device)
            delta.data.copy_(proj)
        if s % 50 == 0 or s == 1:
            print(f"    [psy]  step {s:3d}/{n_steps} | max-cos={score:+.3f} "
                  f"| ||d||_inf={delta.data.abs().max().item():.4f}")
    print(f"    [psy]  done | best max-cos={best_score:+.3f}")
    return best_delta


# ---------------------------------------------------------------------------
# Evaluation of one protected waveform
# ---------------------------------------------------------------------------

def eval_variant(tag, wav_o, wav_p, mel_o, purifier, range_basis,
                 wavlm, ecapa_clf, dvec_model, device, sr=16000):
    def cs(a, b):
        a = F.normalize(a.float().reshape(1, -1), dim=-1)
        b = F.normalize(b.float().reshape(1, -1), dim=-1)
        return (a * b).sum().item()

    cos_wavlm = cs(real_embed_wavlm(wavlm, wav_o), real_embed_wavlm(wavlm, wav_p))
    cos_ecapa = cs(real_embed_ecapa(ecapa_clf, wav_o), real_embed_ecapa(ecapa_clf, wav_p))
    cos_dvec = cs(real_embed_dvec(dvec_model, wav_o), real_embed_dvec(dvec_model, wav_p))

    # PRR — paper protocol (null-projected mel delta) and honest actual delta
    mel_p = wav_to_logmel(wav_p, device)
    dmel = (mel_p - mel_o).cpu()
    dmel_proj = project_onto_null_space(dmel, range_basis).to(device)
    prr_proj = purification_resistance_rate(mel_o, mel_o + dmel_proj, purifier)
    prr_actual = purification_resistance_rate(mel_o, mel_p, purifier)

    snr = waveform_snr_db(wav_o.cpu(), wav_p.cpu())
    sil_db, n_sil = silence_noise_db(wav_o.cpu(), wav_p.cpu(), sample_rate=sr)

    x = wav_o.squeeze().cpu().numpy()
    y = wav_p.squeeze().cpu().numpy()[: x.shape[0]]
    try:
        from pesq import pesq as pesq_fn
        pesq_v = pesq_fn(sr, x, y, "wb")
    except Exception:
        pesq_v = float("nan")
    try:
        from pystoi import stoi as stoi_fn
        stoi_v = stoi_fn(x, y, sr, extended=False)
    except Exception:
        stoi_v = float("nan")

    row = dict(variant=tag,
               cos_wavlm=round(cos_wavlm, 4), cos_ecapa=round(cos_ecapa, 4),
               cos_dvec=round(cos_dvec, 4),
               fails_wavlm=cos_wavlm < 0.50, fails_ecapa=cos_ecapa < 0.25,
               fails_dvec=cos_dvec < 0.75,
               prr_proj=round(prr_proj, 4), prr_actual=round(prr_actual, 4),
               snr_wav_db=round(snr, 2), silence_noise_db=round(sil_db, 1),
               n_silent_frames=n_sil,
               pesq_wb=round(pesq_v, 3), stoi=round(stoi_v, 4))
    print(f"  [{tag}] WavLM={cos_wavlm:+.3f}(<0.50) ECAPA={cos_ecapa:+.3f}(<0.25) "
          f"dvec={cos_dvec:+.3f}(<0.75)")
    print(f"  [{tag}] SNR={snr:.1f}dB PESQ={pesq_v:.2f} STOI={stoi_v:.3f} "
          f"| PRR proj={prr_proj:.3f} actual={prr_actual:.3f}")
    return row


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=3, dest="n_files")
    ap.add_argument("--steps", type=int, default=300, help="Adam steps (psy)")
    ap.add_argument("--flat-steps", type=int, default=80, dest="flat_steps")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin-db", type=float, default=0.0, dest="margin_db",
                    help="raise masking cap: 0=inaudible, +6dB=2x amplitude")
    ap.add_argument("--bandlimit", type=float, default=None,
                    help="zero delta above this Hz (e.g. 4000) for resample robustness")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--skip-flat", action="store_true", dest="skip_flat")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Perceptual] device={device} | margin_db={args.margin_db} "
          f"| bandlimit={args.bandlimit}")

    wavs = sorted(Path(args.input).glob("*.wav"))[: args.n_files]
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    wavlm = load_wavlm(device=device)
    encoders = build_all(device, wavlm, SAVEDIR)
    ecapa_clf = [e for e in encoders if isinstance(e, ECAPAEnc)][0].clf
    dvec_model = [e for e in encoders if isinstance(e, DvecEnc)][0].dvec

    masker = PsychoacousticMasker(margin_db=args.margin_db, device=device)

    rows = []
    for wp in wavs:
        print(f"\n{'=' * 60}\n[Perceptual] {wp.name}")
        wav_o = load_wav(str(wp)).to(device)
        mel_o = wav_to_logmel(wav_o, device)

        print("  computing DiffPure Jacobian / null space ...")
        Z, V = estimate_jacobian_randomized(purifier, mel_o, K=args.K, seed=42)
        range_basis, _ = compute_null_space_factored(Z, V, threshold=1e-3)

        variants = {}
        if not args.skip_flat:
            print("  --- flat PGD (baseline) ---")
            d_flat = flat_pgd(wav_o, encoders, n_steps=args.flat_steps)
            variants["flat"] = (wav_o + d_flat).clamp(-1.0, 1.0)

        print("  --- psychoacoustic PGD (proposed) ---")
        cap = masker.magnitude_cap(wav_o)
        d_psy = psy_pgd(wav_o, encoders, masker, cap, n_steps=args.steps,
                        lr=args.lr, bandlimit_hz=args.bandlimit)
        # final tightening projection (Adam's last step may overshoot the cap)
        d_psy = project_delta_to_masking(d_psy, cap, masker,
                                         bandlimit_hz=args.bandlimit).to(device)
        variants["psy"] = (wav_o + d_psy).clamp(-1.0, 1.0)

        for tag, wav_p in variants.items():
            row = eval_variant(tag, wav_o, wav_p, mel_o, purifier, range_basis,
                               wavlm, ecapa_clf, dvec_model, device)
            row["file"] = wp.name
            rows.append(row)
            out = os.path.join(WAV_DIR, f"{wp.stem}_{tag}.wav")
            save_wav(wav_p.cpu(), out)
            print(f"  saved {out}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "perceptual.csv")
    df.to_csv(csv, index=False)
    print(f"\n[Perceptual] saved {csv}")

    print("\n" + "=" * 78)
    print("MEAN PER VARIANT")
    print("=" * 78)
    num_cols = ["cos_wavlm", "cos_ecapa", "cos_dvec", "prr_proj", "prr_actual",
                "snr_wav_db", "silence_noise_db", "pesq_wb", "stoi"]
    g = df.groupby("variant")[num_cols].mean()
    print(g.round(3).to_string())
    for v in g.index:
        sub = df[df.variant == v]
        all_fail = (sub.fails_wavlm & sub.fails_ecapa & sub.fails_dvec).mean()
        print(f"  {v}: cloning fails on ALL 3 encoders for {100 * all_fail:.0f}% of files")
    print("=" * 78)


if __name__ == "__main__":
    main()
