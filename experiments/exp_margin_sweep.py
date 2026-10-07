"""
experiments/exp_margin_sweep.py
===============================
Find the operating point where masked PGD both PROTECTS (cloning fails on
WavLM+ECAPA+d-vector) AND sounds like the original (high PESQ/STOI).

margin_db raises the masking-threshold cap: 0 dB = strictly inaudible (too
weak to fool robust encoders), higher = louder but still SPECTRALLY SHAPED to
the voice, so even when audible it reads as natural coloration rather than the
flat broadband hiss of the old L-inf method.

Computes the DiffPure Jacobian ONCE (slow) and reuses it across all margins.

Run:
  python experiments/exp_margin_sweep.py --file sample_001.wav \
      --margins 12,20,28,36 --steps 200
Output:
  experiments/results/margin_sweep.csv
  experiments/results/margin_sweep/<file>_m<margin>[_bl<khz>].wav
"""
import sys, io, os, argparse
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from core.jacobian import estimate_jacobian_randomized
from core.nullspace import compute_null_space_factored, project_onto_null_space
from core.psychoacoustic import (PsychoacousticMasker, project_delta_to_masking,
                                 waveform_snr_db)
from evaluation.evaluate import load_wav, wav_to_logmel, save_wav
from evaluation.metrics import purification_resistance_rate
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import (build_all, ECAPAEnc, DvecEnc,
                                        real_embed_wavlm, real_embed_ecapa,
                                        real_embed_dvec)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(OUT_DIR, "margin_sweep")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(WAV_DIR, exist_ok=True)


def psy_pgd(wav_o, encoders, masker, cap, n_steps, lr, bandlimit_hz, seed=42):
    device = wav_o.device
    orig = [e.embed(wav_o).detach() for e in encoders]
    torch.manual_seed(seed)
    # Small random init (NOT zeros): delta=0 is a vanishing-gradient stationary
    # point of the cosine loss.
    delta = ((torch.rand_like(wav_o) * 2 - 1) * 2e-3).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_score, best_delta = float("inf"), delta.detach().clone()
    for s in range(1, n_steps + 1):
        opt.zero_grad()
        coss = torch.stack([(e.embed(wav_o.detach() + delta) * o).sum()
                            for e, o in zip(encoders, orig)])
        loss = torch.logsumexp(coss * 4.0, dim=0) / 4.0
        loss.backward()
        with torch.no_grad():
            score = coss.max().item()
            if score < best_score:
                best_score, best_delta = score, delta.detach().clone()
        opt.step()
        with torch.no_grad():
            proj = project_delta_to_masking(delta.data, cap, masker,
                                            bandlimit_hz=bandlimit_hz).to(device)
            delta.data.copy_(proj)
    # final tightening
    best_delta = project_delta_to_masking(best_delta, cap, masker,
                                          bandlimit_hz=bandlimit_hz).to(device)
    return best_delta, best_score


def evaluate(wav_o, wav_p, mel_o, purifier, range_basis, wavlm, ecapa_clf,
             dvec_model, device, sr=16000):
    def cs(a, b):
        a = F.normalize(a.float().reshape(1, -1), dim=-1)
        b = F.normalize(b.float().reshape(1, -1), dim=-1)
        return (a * b).sum().item()
    cw = cs(real_embed_wavlm(wavlm, wav_o), real_embed_wavlm(wavlm, wav_p))
    ce = cs(real_embed_ecapa(ecapa_clf, wav_o), real_embed_ecapa(ecapa_clf, wav_p))
    cd = cs(real_embed_dvec(dvec_model, wav_o), real_embed_dvec(dvec_model, wav_p))
    mel_p = wav_to_logmel(wav_p, device)
    dmel_proj = project_onto_null_space((mel_p - mel_o).cpu(), range_basis, kind="range").to(device)
    prr = purification_resistance_rate(mel_o, mel_o + dmel_proj, purifier)
    snr = waveform_snr_db(wav_o.cpu(), wav_p.cpu())
    x = wav_o.squeeze().cpu().numpy(); y = wav_p.squeeze().cpu().numpy()[:x.shape[0]]
    try:
        from pesq import pesq as pf; pesq_v = pf(sr, x, y, "wb")
    except Exception: pesq_v = float("nan")
    try:
        from pystoi import stoi as sf_; stoi_v = sf_(x, y, sr, extended=False)
    except Exception: stoi_v = float("nan")
    return dict(cos_wavlm=round(cw, 4), cos_ecapa=round(ce, 4), cos_dvec=round(cd, 4),
                fail_wavlm=cw < 0.50, fail_ecapa=ce < 0.25, fail_dvec=cd < 0.75,
                prr=round(prr, 4), snr_wav_db=round(snr, 2),
                pesq_wb=round(pesq_v, 3), stoi=round(stoi_v, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="sample_001.wav")
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--margins", default="12,20,28,36")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bandlimit-khz", type=float, default=None, dest="bandlimit_khz",
                    help="also run each margin with this low-pass cutoff (kHz)")
    ap.add_argument("--K", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    margins = [float(m) for m in args.margins.split(",")]
    print(f"[MarginSweep] device={device} | margins={margins} | steps={args.steps}")

    wav_o = load_wav(os.path.join(args.input, args.file)).to(device)
    mel_o = wav_to_logmel(wav_o, device)
    purifier = load_real_diffpure(t_diffuse=50, device=device)
    wavlm = load_wavlm(device=device)
    encoders = build_all(device, wavlm, SAVEDIR)
    ecapa_clf = [e for e in encoders if isinstance(e, ECAPAEnc)][0].clf
    dvec_model = [e for e in encoders if isinstance(e, DvecEnc)][0].dvec

    print("  computing DiffPure Jacobian / null space (once) ...")
    Z, V = estimate_jacobian_randomized(purifier, mel_o, K=args.K, seed=42)
    range_basis, _ = compute_null_space_factored(Z, V, threshold=1e-3)

    configs = [(m, None) for m in margins]
    if args.bandlimit_khz is not None:
        configs += [(m, args.bandlimit_khz * 1000.0) for m in margins]

    rows = []
    for margin, bl in configs:
        tag = f"m{int(margin)}" + (f"_bl{int(args.bandlimit_khz)}" if bl else "")
        print(f"\n--- margin={margin:+.0f} dB | bandlimit={bl} ---")
        masker = PsychoacousticMasker(margin_db=margin, device=device)
        cap = masker.magnitude_cap(wav_o)
        best_delta, best_cos = psy_pgd(wav_o, encoders, masker, cap,
                                       args.steps, args.lr, bl)
        wav_p = (wav_o + best_delta).clamp(-1.0, 1.0)
        row = evaluate(wav_o, wav_p, mel_o, purifier, range_basis,
                       wavlm, ecapa_clf, dvec_model, device)
        row.update(margin_db=margin, bandlimit_hz=bl, best_train_cos=round(best_cos, 4))
        rows.append(row)
        out = os.path.join(WAV_DIR, f"{Path(args.file).stem}_{tag}.wav")
        save_wav(wav_p.cpu(), out)
        n_fail = int(row["fail_wavlm"]) + int(row["fail_ecapa"]) + int(row["fail_dvec"])
        print(f"  cos WavLM={row['cos_wavlm']:+.3f} ECAPA={row['cos_ecapa']:+.3f} "
              f"dvec={row['cos_dvec']:+.3f} | {n_fail}/3 encoders fooled")
        print(f"  PESQ={row['pesq_wb']:.2f} STOI={row['stoi']:.3f} "
              f"SNR={row['snr_wav_db']:.1f}dB PRR={row['prr']:.3f} | saved {out}")

    df = pd.DataFrame(rows)[["margin_db", "bandlimit_hz", "cos_wavlm", "cos_ecapa",
                             "cos_dvec", "fail_wavlm", "fail_ecapa", "fail_dvec",
                             "prr", "snr_wav_db", "pesq_wb", "stoi"]]
    csv = os.path.join(OUT_DIR, "margin_sweep.csv")
    df.to_csv(csv, index=False)
    print(f"\n[MarginSweep] saved {csv}")
    print("=" * 92)
    print(df.to_string(index=False))
    print("=" * 92)
    print("Target: all three fail_* = True (cloning fails) AND PESQ > 3.5 (sounds like original)")


if __name__ == "__main__":
    main()
