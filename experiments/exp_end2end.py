"""
experiments/exp_end2end.py
==========================
The security experiment an IEEE reviewer will ask for: does protection hold
AFTER the attacker purifies with DiffPure and re-extracts the speaker identity?

Threat model (matches THREAT_MODEL.md):
    x        clean utterance
    x_p      = protect(x)                         defender publishes this
    x_r      = GriffinLim(DiffPure(mel(x_p)))     ATTACKER purifies to strip δ
    accept  iff  cos(E(x_r), E(x)) >= τ           clone would succeed

We report, per encoder (WavLM / ECAPA / d-vector), on the REAL eval paths:
    cos_before   cos(E(x_p),  E(x))    protected, no purification
    cos_after    cos(E(x_r),  E(x))    protected + attacker DiffPure  ← THE metric
    cos_ctrl     cos(E(x0_r), E(x))    UNPROTECTED through the same DiffPure+GL

Defence is valid on a file iff, for the targeted encoders,
    cos_after < τ   (clone still fails)   AND   cos_ctrl >= τ   (purification
alone preserves identity, so the drop is caused by δ, not by the vocoder).
The cos_ctrl column is the Griffin-Lim control that review finding #7 requires.

No Jacobian / null projection is needed here — this measures the delivered
waveform end to end, which is the honest object of the security claim.

Run:
  python experiments/exp_end2end.py --n-files 5 --margin-db 30 --bandlimit 6000 --steps 250
Output:
  experiments/results/end2end.csv
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

from core.psychoacoustic import (PsychoacousticMasker, project_delta_to_masking,
                                 waveform_snr_db)
from evaluation.evaluate import load_wav, wav_to_logmel, save_wav
from utils.vocoder import invert_mel
from models.real_diffpure import load_real_diffpure
from models.speaker_encoder import load_pretrained as load_wavlm
from experiments.multi_encoders import (build_all, ECAPAEnc, DvecEnc,
                                        real_embed_wavlm, real_embed_ecapa,
                                        real_embed_dvec)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(OUT_DIR, "end2end")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(WAV_DIR, exist_ok=True)

THRESH = {"WavLM (target)": 0.50, "ECAPA-TDNN": 0.25, "Resemblyzer d-vector": 0.75}
DIFFPURE_SEED = 2026


def ensemble_masked_pgd(wav_o, encoders, masker, cap, n_steps, lr, bandlimit_hz, seed=42):
    device = wav_o.device
    orig = [e.embed(wav_o).detach() for e in encoders]
    torch.manual_seed(seed)
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
            delta.data.copy_(project_delta_to_masking(
                delta.data, cap, masker, bandlimit_hz=bandlimit_hz).to(device))
    best_delta = project_delta_to_masking(best_delta, cap, masker,
                                          bandlimit_hz=bandlimit_hz).to(device)
    return best_delta


def cs(a, b):
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()


def embed_all(wav, wavlm, ecapa_clf, dvec_model):
    return {
        "WavLM (target)": real_embed_wavlm(wavlm, wav),
        "ECAPA-TDNN": real_embed_ecapa(ecapa_clf, wav),
        "Resemblyzer d-vector": real_embed_dvec(dvec_model, wav),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--margin-db", type=float, default=30.0, dest="margin_db")
    ap.add_argument("--bandlimit", type=float, default=6000.0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weights", default="1,1,1.5",
                    help="ensemble weights wavlm,ecapa,dvec (dvec up-weighted: hardest)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[End2End] device={device} | margin={args.margin_db} "
          f"| bandlimit={args.bandlimit} | steps={args.steps}")

    wavs = sorted(Path(args.input).glob("*.wav"))[: args.n_files]
    diffpure = load_real_diffpure(t_diffuse=50, device=device)
    wavlm = load_wavlm(device=device)
    encoders = build_all(device, wavlm, SAVEDIR)
    ecapa_clf = [e for e in encoders if isinstance(e, ECAPAEnc)][0].clf
    dvec_model = [e for e in encoders if isinstance(e, DvecEnc)][0].dvec
    masker = PsychoacousticMasker(margin_db=args.margin_db, device=device)

    rows = []
    for wp in wavs:
        print(f"\n{'=' * 60}\n[End2End] {wp.name}")
        wav_o = load_wav(str(wp)).to(device)
        mel_o = wav_to_logmel(wav_o, device)

        cap = masker.magnitude_cap(wav_o)
        delta = ensemble_masked_pgd(wav_o, encoders, masker, cap, args.steps,
                                    args.lr, args.bandlimit)
        wav_p = (wav_o + delta).clamp(-1.0, 1.0)

        # ---- attacker purification (same DiffPure seed for protected & control)
        with torch.no_grad():
            mel_p = wav_to_logmel(wav_p, device)
            mel_p_pure = diffpure(mel_p, seed=DIFFPURE_SEED)
            mel_o_pure = diffpure(mel_o, seed=DIFFPURE_SEED)
        wav_p_pure = invert_mel(mel_p_pure.cpu()).to(device)   # attacker's cleaned audio
        wav_o_pure = invert_mel(mel_o_pure.cpu()).to(device)   # GL+DiffPure control

        emb_o = embed_all(wav_o, wavlm, ecapa_clf, dvec_model)
        emb_p = embed_all(wav_p, wavlm, ecapa_clf, dvec_model)
        emb_pp = embed_all(wav_p_pure, wavlm, ecapa_clf, dvec_model)
        emb_op = embed_all(wav_o_pure, wavlm, ecapa_clf, dvec_model)

        snr = waveform_snr_db(wav_o.cpu(), wav_p.cpu())
        x = wav_o.squeeze().cpu().numpy(); y = wav_p.squeeze().cpu().numpy()[:x.shape[0]]
        try:
            from pesq import pesq as pf; pesq_v = pf(16000, x, y, "wb")
        except Exception: pesq_v = float("nan")
        try:
            from pystoi import stoi as sf_; stoi_v = sf_(x, y, 16000, extended=False)
        except Exception: stoi_v = float("nan")

        for name in THRESH:
            tau = THRESH[name]
            cb = cs(emb_o[name], emb_p[name])
            ca = cs(emb_o[name], emb_pp[name])
            cc = cs(emb_o[name], emb_op[name])
            valid = (ca < tau) and (cc >= tau)
            rows.append(dict(file=wp.name, encoder=name, threshold=tau,
                             cos_before=round(cb, 4), cos_after_purify=round(ca, 4),
                             cos_ctrl_unprot=round(cc, 4),
                             clone_fails_after=ca < tau,
                             ctrl_identity_kept=cc >= tau,
                             defence_valid=valid,
                             pesq=round(pesq_v, 3), stoi=round(stoi_v, 4),
                             snr_db=round(snr, 2)))
            print(f"  {name:22s} τ={tau} | before={cb:+.3f} "
                  f"after_purify={ca:+.3f} ctrl={cc:+.3f} | "
                  f"{'VALID' if valid else 'weak'}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "end2end.csv")
    df.to_csv(csv, index=False)
    print(f"\n[End2End] saved {csv}")

    print("\n" + "=" * 84)
    print("MEAN PER ENCODER — cloning AFTER attacker's DiffPure purification")
    print("=" * 84)
    g = df.groupby("encoder", sort=False)
    for name, sub in g:
        tau = THRESH[name]
        print(f"  {name:22s} τ={tau} | before={sub.cos_before.mean():+.3f} "
              f"after_purify={sub.cos_after_purify.mean():+.3f} "
              f"ctrl={sub.cos_ctrl_unprot.mean():+.3f} | "
              f"clone fails after purify on {100*sub.clone_fails_after.mean():.0f}% "
              f"(control identity kept on {100*sub.ctrl_identity_kept.mean():.0f}%)")
    print(f"  quality: PESQ={df.pesq.mean():.2f}  STOI={df.stoi.mean():.3f}  "
          f"SNR={df.snr_db.mean():.1f}dB")
    print("=" * 84)
    print("Read: 'clone fails after purify' high AND 'control identity kept' high")
    print("      => the drop is caused by the perturbation, not the vocoder.")


if __name__ == "__main__":
    main()
