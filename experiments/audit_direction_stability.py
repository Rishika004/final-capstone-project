"""
experiments/audit_direction_stability.py
========================================
Independent re-check of RESULTS section 57 (the audit of sections 54 and 55),
on more than one speaker and more than one jitter magnitude, plus the
instrument checks that section 57 did not run. Written 2026-10-08 for the
second audit (RESULTS section 58). Nothing here touches a pre-registered
threshold, endpoint or decision rule; this script measures DIRECTIONS, not
alpha, and never reads a verdict on a claim.

WHAT IS CHECKED, AND HOW
------------------------
For each speaker and each encoder, two code paths build "the identity
direction":

  waveform path  (exp_encoder_subspace_wav.py, section 55, now disabled)
      d = -grad_w <E(w), E(x0)>  evaluated at  w = x0
  mel path       (exp_encoder_subspace.py, every other alpha section)
      d = -grad_m <E(V(m)), E(x0)>  evaluated at  m = logmel(x0),
      V = the fixed-phase mel->wav inverse, then peak-normalised

Section 57 says the first is residue (the gradient is analytically zero at x0)
and the second is sound. Both claims are re-measured here, each by FOUR
independent instruments rather than one:

  1. stationarity   cos at the evaluation point and |grad| there, against |grad|
                    at a nearby point;
  2. stability      cosine between the direction and the direction recomputed
                    (a) run-to-run, (b) after jittering the INPUT x0 (o is
                    recomputed, so the waveform path stays at its maximum),
                    (c) after jittering only the EVALUATION POINT (o fixed,
                    so the waveform path is moved off its maximum), at several
                    magnitudes and with two independent jitter draws per
                    magnitude -- if two draws give two different directions,
                    there is no direction, only a response to the jitter;
  3. float64        the same direction recomputed in double precision on CPU
                    (CAM++ family only; the others do not cast cleanly). A
                    rounding residue collapses by ~1e8; a genuine gradient
                    does not move;
  4. structure      what the residue IS, if it is anything: its cosine with
                    the gradient of the raw embedding NORM (which the L2
                    normalisation discards and which rounding re-admits), and
                    with the TOP SINGULAR DIRECTION of the encoder Jacobian
                    (power iteration on J^T J). This decides whether a sign
                    flip is "unrelated noise" or "the same direction, sign
                    lost", which matters because alpha is nearly sign-symmetric.

Also recorded, because the second audit asks for them:

  * the official gate `check_direction_sanity.check_direction`, run at 1e-5,
    1e-4 and 1e-3 input jitter, on every path including the two FRONT-END
    directions (a fixed random linear projection of Kaldi fbank features, and
    of the raw waveform) that `audit_frontend_alpha.py` will measure alpha on;
    those may not be read until this gate passes;
  * concentration statistics of every direction and of its controls (energy
    fraction in the loudest 5 % of elements, per-frame energy variation,
    correlation with the signal's own loudness), which is what "equal RMS"
    does and does not equalise;
  * the exact per-element RMS every direction receives after `unit()`.

Run (GPU, ~10 min for 3 speakers):
    python experiments/audit_direction_stability.py --speakers 3
Output:
    experiments/results/audit_direction_stability.csv   one row per (speaker,
                                                        path, encoder, check)
    experiments/results/audit_direction_structure.csv   concentration stats
    experiments/results/audit_direction_gate.txt        the gate reports
"""
import argparse
import io
import math
import os
import sys

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
import torch._dynamo   # noqa: F401  (import-order gotcha; see exp_purifier_aware)
import transformers    # noqa: F401
import torch.nn.functional as F

from core.matched_random import (band_matched, band_matched_wav, clean_mel_basis,
                                 pca_span)
from experiments.check_direction_sanity import check_direction
from utils.provenance import provenance

R = os.path.join(PROJ, "experiments", "results")
SR = 16000


# --------------------------------------------------------------------------
# The two front-end directions (council audit 2, risk 2). Same front end as
# CAM++ (Kaldi fbank + CMN, `extra_encoders.fbank`), NO network: a fixed
# random linear projection, L2-normalised like every encoder here. The second
# one projects the raw waveform, i.e. it isolates the mel->wav inverse V alone.
# --------------------------------------------------------------------------
class FbankLinearEnc:
    """normalize(W . vec(fbank_cmn(x))), W fixed Gaussian, seed 2026."""
    name = "frontend:fbank-linear"
    threshold = 0.0

    def __init__(self, device, n_frames, n_mels=80, dim=512, seed=2026):
        g = torch.Generator().manual_seed(seed)
        self.n_frames = int(n_frames)
        self.W = (torch.randn(dim, self.n_frames * n_mels, generator=g)
                  / math.sqrt(self.n_frames * n_mels)).to(device)

    def embed(self, wav):
        from experiments.extra_encoders import fbank
        f = fbank(wav)                                  # [frames, 80], CMN'd
        if f.shape[0] < self.n_frames:
            raise SystemExit(f"[frontend] {f.shape[0]} frames < {self.n_frames}")
        f = f[: self.n_frames]
        return F.normalize((self.W @ f.reshape(-1)).reshape(1, -1), dim=-1)


class WavLinearEnc:
    """normalize(W . x), W fixed Gaussian over the 4 s waveform, seed 2027."""
    name = "frontend:wav-linear"
    threshold = 0.0

    def __init__(self, device, n_samples, dim=512, seed=2027):
        g = torch.Generator().manual_seed(seed)
        self.n = int(n_samples)
        self.W = (torch.randn(dim, self.n, generator=g) / math.sqrt(self.n)).to(device)

    def embed(self, wav):
        x = wav.reshape(-1)[: self.n]
        return F.normalize((self.W @ x).reshape(1, -1), dim=-1)


# --------------------------------------------------------------------------
# direction builders, mirroring the two scripts line for line
# --------------------------------------------------------------------------
def _noise(shape, seed, device):
    """Drawn on CPU and moved, so the float32/GPU and float64/CPU twins of a
    direction see the SAME jitter and their cosine compares precision only."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=g).to(device)


def wav_direction(e, wav, jit_in=0.0, jit_ev=0.0, seed=0):
    """exp_encoder_subspace_wav.py lines 193-201, with two optional jitters.

    jit_in : relative jitter of the INPUT x0 (o is recomputed from it, so the
             evaluation point is still the maximum of cosine-to-self).
    jit_ev : relative jitter of the EVALUATION POINT only (o is fixed to the
             unjittered input, so the evaluation point moves off the maximum).
    Returns (-grad, cosine at the evaluation point, |grad|).
    """
    dev = wav.device
    rms = wav.pow(2).mean().sqrt()
    x_in = wav + jit_in * rms * _noise(wav.shape, 1000 + seed, dev)
    with torch.no_grad():
        o = e.embed(x_in).detach()
    w = (x_in + jit_ev * rms * _noise(wav.shape, 2000 + seed, dev)).clone()
    w.requires_grad_(True)
    c = (e.embed(w) * o).sum()
    g, = torch.autograd.grad(c, w)
    return (-g).detach(), float(c), float(g.norm())


class MelPath:
    """Everything exp_encoder_subspace.py does per speaker, as a reusable object."""

    def __init__(self, device):
        from evaluation.evaluate import (wav_to_logmel, N_FFT, N_MELS,
                                         HOP_LENGTH, SAMPLE_RATE)
        from core.optimizer import _build_mel_to_wav, _mel_to_wav_differentiable
        self.device = device
        self.wav_to_logmel = wav_to_logmel
        self.build = _build_mel_to_wav
        self.inv = _mel_to_wav_differentiable
        self.N_FFT, self.N_MELS, self.HOP, self.SR = N_FFT, N_MELS, HOP_LENGTH, SAMPLE_RATE

    def setup(self, x_in):
        mel_o = self.wav_to_logmel(x_in, self.device).detach()
        inv, phase, win, wav_len, T_frames = self.build(
            n_fft=self.N_FFT, hop_length=self.HOP, n_mels=self.N_MELS,
            sample_rate=self.SR, x_wav=x_in, device=self.device)
        return mel_o, (inv, phase, win, wav_len, T_frames)

    def recon(self, m, parts):
        inv, phase, win, wav_len, T_frames = parts
        w = self.inv(m, inv, phase, win, self.N_FFT, self.HOP, T_frames, wav_len)
        return w / (w.abs().max() + 1e-6)

    def direction(self, e, wav_o, jit_in=0.0, jit_ev=0.0, seed=0):
        """exp_encoder_subspace.py lines 337-347, with the two jitters.

        Returns (-grad, cosine at the evaluation point, |grad| / |mel|)."""
        dev = wav_o.device
        rms = wav_o.pow(2).mean().sqrt()
        x_in = wav_o + jit_in * rms * _noise(wav_o.shape, 1000 + seed, dev)
        mel_o, parts = self.setup(x_in)
        with torch.no_grad():
            o = e.embed(x_in).detach()
        m = mel_o + jit_ev * mel_o.pow(2).mean().sqrt() * _noise(mel_o.shape, 2000 + seed, dev)
        m = m.clone().requires_grad_(True)
        w = self.recon(m, parts)
        c = (e.embed(w) * o).sum()
        g, = torch.autograd.grad(c, m)
        return (-g).detach(), float(c), float(g.norm() / mel_o.norm())


# --------------------------------------------------------------------------
# float64 twins, CAM++ family only (fbank + TDNN cast cleanly; the others
# carry float32 buffers inside third-party code and are not worth forcing)
# --------------------------------------------------------------------------
def _embed64(m64, wav64):
    from experiments.extra_encoders import fbank
    f = fbank(wav64).unsqueeze(0)
    e = m64(f)
    if isinstance(e, (tuple, list)):
        e = e[-1]
    return F.normalize(e.reshape(1, -1), dim=-1)


def wav_direction64(m64, wav, jit_ev=0.0, seed=0):
    cpu = torch.device("cpu")
    x = wav.detach().cpu().double()
    rms = x.pow(2).mean().sqrt()
    with torch.no_grad():
        o = _embed64(m64, x)
    w = (x + jit_ev * rms * _noise(x.shape, 2000 + seed, cpu).double()).clone()
    w.requires_grad_(True)
    c = (_embed64(m64, w) * o).sum()
    g, = torch.autograd.grad(c, w)
    return (-g).detach(), float(c), float(g.norm())


def mel_direction64(mp, m64, wav_o, jit_ev=0.0, seed=0):
    """The mel path in double on CPU: our mel front end, V, fbank, CAM++."""
    import torchaudio.transforms as T
    x = wav_o.detach().cpu().double()
    cpu = torch.device("cpu")
    melT = T.MelSpectrogram(sample_rate=mp.SR, n_fft=mp.N_FFT, hop_length=mp.HOP,
                            n_mels=mp.N_MELS).double()
    mel_o = torch.log(melT(x) + 1e-9).detach()
    inv, phase, win, wav_len, T_frames = mp.build(
        n_fft=mp.N_FFT, hop_length=mp.HOP, n_mels=mp.N_MELS, sample_rate=mp.SR,
        x_wav=x.float(), device=cpu)
    parts = (inv.double(), phase.to(torch.complex128), win.double(), wav_len, T_frames)
    with torch.no_grad():
        o = _embed64(m64, x)
    m = mel_o + jit_ev * mel_o.pow(2).mean().sqrt() * _noise(mel_o.shape, 2000 + seed, cpu).double()
    m = m.clone().requires_grad_(True)
    w = mp.recon(m, parts)
    c = (_embed64(m64, w) * o).sum()
    g, = torch.autograd.grad(c, m)
    return (-g).detach(), float(c), float(g.norm() / mel_o.norm())


# --------------------------------------------------------------------------
# the top singular direction of the encoder Jacobian at x0 (waveform path),
# by power iteration on J^T J with a finite-difference JVP and an autograd VJP
# --------------------------------------------------------------------------
def top_singular_direction(e, wav, iters=12, h_rel=1e-2, seed=0):
    dev = wav.device
    v = _noise(wav.shape, 3000 + seed, dev)
    v = v / v.norm()
    h = h_rel * wav.norm()
    hist = []
    for _ in range(iters):
        with torch.no_grad():
            u = (e.embed(wav + h * v) - e.embed(wav - h * v)) / (2 * h)   # J v
        w = wav.clone().requires_grad_(True)
        c = (e.embed(w) * u.detach()).sum()
        gv, = torch.autograd.grad(c, w)                                   # J^T J v
        lam = float(gv.norm())
        v_new = gv / gv.norm()
        hist.append((lam, float(F.cosine_similarity(v_new.reshape(1, -1),
                                                    v.reshape(1, -1)).item())))
        v = v_new.detach()
    return v, hist


def raw_norm_gradient(e, wav):
    """grad_w ||u(w)|| at x0, where u is the embedding BEFORE L2 normalisation.

    Each wrapper normalises inside embed(); the raw embedding is reached through
    the wrapper's own model. This is the direction float32 rounding re-admits
    when (I - f f^T) o is computed at o == f."""
    from experiments.multi_encoders import ECAPAEnc, DvecEnc, WavLMSVEnc
    from experiments.extra_encoders import _Base, fbank
    w = wav.clone().requires_grad_(True)
    if isinstance(e, WavLMSVEnc):
        x = w.reshape(1, -1)
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        u = e.m(input_values=x).embeddings
    elif isinstance(e, ECAPAEnc):
        u = e.clf.encode_batch(w.reshape(1, -1))
    elif isinstance(e, DvecEnc):
        u = e.embed_partials(w).mean(dim=0)
    elif isinstance(e, _Base):
        u = e.m(fbank(w).unsqueeze(0))
        if isinstance(u, (tuple, list)):
            u = u[-1]
    else:
        return None
    n = u.float().reshape(-1).norm()
    g, = torch.autograd.grad(n, w)
    return g.detach()


# --------------------------------------------------------------------------
# concentration / structure statistics
# --------------------------------------------------------------------------
def _cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1).double(),
                                     b.reshape(1, -1).double()).item())


def mel_structure(d, mel_o):
    x = d.reshape(-1).double()
    e2 = x.pow(2)
    k = max(1, int(0.05 * x.numel()))
    top5 = float(e2.topk(k).values.sum() / e2.sum())
    frames = d.reshape(d.shape[-2], d.shape[-1]).double().pow(2).sum(dim=0)
    frame_cv = float(frames.std() / frames.mean().clamp_min(1e-30))
    bands = d.reshape(d.shape[-2], d.shape[-1]).double().pow(2).sum(dim=1)
    band_cv = float(bands.std() / bands.mean().clamp_min(1e-30))
    loud = mel_o.reshape(-1).double()
    corr = float(torch.corrcoef(torch.stack([e2, loud]))[0, 1])
    kurt = float((x - x.mean()).pow(4).mean() / x.var().pow(2))
    return dict(top5_energy=top5, frame_cv=frame_cv, band_cv=band_cv,
                corr_with_logmel=corr, kurtosis=kurt,
                peak_over_rms=float(x.abs().max() / x.pow(2).mean().sqrt()))


def wav_structure(d, wav):
    x = d.reshape(-1).double()
    e2 = x.pow(2)
    k = max(1, int(0.05 * x.numel()))
    top5 = float(e2.topk(k).values.sum() / e2.sum())
    win = torch.hann_window(512, device=x.device, dtype=x.dtype)
    S = torch.stft(x, 512, 128, window=win, return_complex=True).abs().pow(2)
    frames = S.sum(dim=0)
    frame_cv = float(frames.std() / frames.mean().clamp_min(1e-30))
    Sx = torch.stft(wav.reshape(-1).double(), 512, 128, window=win,
                    return_complex=True).abs().pow(2).sum(dim=0)
    corr = float(torch.corrcoef(torch.stack([frames, Sx]))[0, 1])
    kurt = float((x - x.mean()).pow(4).mean() / x.var().pow(2))
    return dict(top5_energy=top5, frame_cv=frame_cv, band_cv=float("nan"),
                corr_with_signal_env=corr, kurtosis=kurt,
                peak_over_rms=float(x.abs().max() / x.pow(2).mean().sqrt()))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=3,
                    help="first N speakers of the cross-session test-clean cohort")
    ap.add_argument("--jitters", default="1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--no-float64", action="store_true", dest="no_f64")
    ap.add_argument("--no-power", action="store_true", dest="no_power",
                    help="skip the top-singular-direction power iteration")
    ap.add_argument("--paths", default="wav,mel",
                    help="which paths to check: wav, mel, or both")
    ap.add_argument("--first", type=int, default=0,
                    help="skip the first N cohort speakers (to resume / extend a run)")
    ap.add_argument("--out", default=os.path.join(R, "audit_direction_stability.csv"))
    a = ap.parse_args()
    paths = set(a.paths.split(","))
    PROV = provenance()
    jits = [float(v) for v in a.jitters.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[audit] code_sha={PROV['code_sha']} device={device} jitters={jits}")

    from experiments.exp_purifier_aware import build_encoders
    from experiments.extra_encoders import build_extra, build_untrained
    from experiments.librispeech import (cross_session_split, list_librispeech,
                                         load_flac, reference_items)

    encs = list(build_encoders(device, with_xvector=False)) + list(build_extra(device))
    unt = list(build_untrained(device, seeds=(0, 1, 2)))
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0, utt_index=0)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][a.first: a.first + a.speakers]
    print(f"[audit] speakers {spks}")

    mp = MelPath(device)
    basis, _ = clean_mel_basis(k=20, subset="dev-clean", device=device, verbose=False)

    # the front-end directions need the fixed geometry of a 4 s clip
    wav0 = load_flac(ref[spks[0]]["path"], 4.0).to(device)
    from experiments.extra_encoders import fbank
    n_frames = int(fbank(wav0).shape[0])
    fe = [FbankLinearEnc(device, n_frames), WavLinearEnc(device, wav0.shape[-1])]
    print(f"[audit] front-end directions: fbank frames={n_frames}, "
          f"samples={wav0.shape[-1]}")

    rows, struct_rows, gate_lines = [], [], []

    def add(**kw):
        kw.update(PROV)
        rows.append(kw)

    for si, spk in enumerate(spks, 1):
        wav = load_flac(ref[spk]["path"], 4.0).to(device)
        print(f"\n[audit] ===== speaker {spk} ({si}/{len(spks)}) =====", flush=True)

        # ------------------------------------------------ waveform path
        for e in (encs if "wav" in paths else []):
            nm = e.name
            d0, c0, n0 = wav_direction(e, wav)
            d0b, _, _ = wav_direction(e, wav)
            rec = dict(speaker=spk, path="wav", encoder=nm, cos_at_eval=c0,
                       grad_norm=n0, repeat_cos=_cos(d0, d0b))
            for j in jits:
                di1, ci1, ni1 = wav_direction(e, wav, jit_in=j, seed=1)
                di2, _, _ = wav_direction(e, wav, jit_in=j, seed=2)
                de1, ce1, ne1 = wav_direction(e, wav, jit_ev=j, seed=1)
                de2, _, _ = wav_direction(e, wav, jit_ev=j, seed=2)
                rec[f"jin{j:g}_cos"] = _cos(d0, di1)
                rec[f"jin{j:g}_pair"] = _cos(di1, di2)
                rec[f"jin{j:g}_gn"] = ni1
                rec[f"jev{j:g}_cos"] = _cos(d0, de1)
                rec[f"jev{j:g}_pair"] = _cos(de1, de2)
                rec[f"jev{j:g}_c"] = ce1
                rec[f"jev{j:g}_gn"] = ne1
            q = raw_norm_gradient(e, wav)
            rec["abscos_with_rawnorm_grad"] = abs(_cos(d0, q)) if q is not None else float("nan")
            if not a.no_power:
                v1, hist = top_singular_direction(e, wav)
                rec["abscos_with_v1"] = abs(_cos(d0, v1))
                rec["abscos_rawnorm_v1"] = abs(_cos(q, v1)) if q is not None else float("nan")
                de_big, _, _ = wav_direction(e, wav, jit_ev=jits[-1], seed=1)
                rec["abscos_jev_big_v1"] = abs(_cos(de_big, v1))
                rec["power_sigma2"] = hist[-1][0]
                rec["power_last_cos"] = hist[-1][1]
            ok_, rep = check_direction(lambda j_: wav_direction(e, wav, jit_in=j_, seed=1),
                                       name=f"spk{spk} wav {nm}")
            rec["gate_1e-5"] = ok_
            gate_lines.append(rep)
            for j in (1e-4, 1e-3):
                ok_j, _ = check_direction(lambda j_: wav_direction(e, wav, jit_in=j_, seed=1),
                                          name=f"spk{spk} wav {nm}", jitter=j)
                rec[f"gate_{j:g}"] = ok_j
            add(**rec)
            # structure of the direction and its controls
            scale = 0.05 * wav.pow(2).mean().sqrt()
            dd = d0 / (d0.pow(2).mean().sqrt() + 1e-12) * scale
            bm = band_matched_wav(dd, generator=torch.Generator(device=device).manual_seed(spk + 101))
            rn = _noise(wav.shape, spk, device)
            rn = rn / rn.pow(2).mean().sqrt() * scale
            for kind, vec in (("target", dd), ("band_matched", bm), ("random", rn)):
                st = wav_structure(vec, wav)
                st.update(speaker=spk, path="wav", encoder=nm, kind=kind,
                          rms=float(vec.pow(2).mean().sqrt()),
                          cos_to_target=_cos(vec, dd))
                struct_rows.append(st)
            print(f"  wav {nm:28s} c={c0:.7f} |g|={n0:.2e} rep={rec['repeat_cos']:+.4f} "
                  f"jin1e-5={rec['jin1e-05_cos']:+.3f} jev1e-2 pair={rec['jev0.01_pair']:+.3f} "
                  f"|cos(q)|={rec['abscos_with_rawnorm_grad']:.3f}"
                  + (f" |cos(v1)|={rec['abscos_with_v1']:.3f}" if not a.no_power else ""),
                  flush=True)

        # float64 twin of the waveform path: CAM++ trained and untrained seed 0
        if not a.no_f64 and "wav" in paths:
            for e in [x for x in encs if x.name == "CAM++"] + unt[:1]:
                m64 = __import__("copy").deepcopy(e.m).cpu().double()
                d32, c32, n32 = wav_direction(e, wav)
                d64, c64, n64 = wav_direction64(m64, wav)
                de32, _, ne32 = wav_direction(e, wav, jit_ev=1e-2, seed=1)
                de64, _, ne64 = wav_direction64(m64, wav, jit_ev=1e-2, seed=1)
                add(speaker=spk, path="wav-f64", encoder=e.name, cos_at_eval=c64,
                    grad_norm=n64, grad_norm_f32=n32,
                    f64_vs_f32_cos_at_x0=_cos(d32, d64.to(device)),
                    f64_vs_f32_cos_offmax=_cos(de32, de64.to(device)),
                    grad_norm_jev1e_2_f64=ne64, grad_norm_jev1e_2_f32=ne32)
                print(f"  wav-f64 {e.name:24s} |g| f32={n32:.2e} f64={n64:.2e}  "
                      f"off-max |g| f32={ne32:.2e} f64={ne64:.2e}  "
                      f"cos(f32,f64) at x0={rows[-1]['f64_vs_f32_cos_at_x0']:+.3f} "
                      f"off-max={rows[-1]['f64_vs_f32_cos_offmax']:+.4f}", flush=True)
                del m64

        # ------------------------------------------------ mel path
        mel_o, _ = mp.setup(wav)
        for e in (encs + unt + fe if "mel" in paths else []):
            nm = e.name
            d0, c0, n0 = mp.direction(e, wav)
            d0b, _, _ = mp.direction(e, wav)
            rec = dict(speaker=spk, path="mel", encoder=nm, cos_at_eval=c0,
                       grad_norm=n0, repeat_cos=_cos(d0, d0b))
            for j in jits:
                di1, _, ni1 = mp.direction(e, wav, jit_in=j, seed=1)
                di2, _, _ = mp.direction(e, wav, jit_in=j, seed=2)
                de1, ce1, ne1 = mp.direction(e, wav, jit_ev=j, seed=1)
                de2, _, _ = mp.direction(e, wav, jit_ev=j, seed=2)
                rec[f"jin{j:g}_cos"] = _cos(d0, di1)
                rec[f"jin{j:g}_pair"] = _cos(di1, di2)
                rec[f"jin{j:g}_gn"] = ni1
                rec[f"jev{j:g}_cos"] = _cos(d0, de1)
                rec[f"jev{j:g}_pair"] = _cos(de1, de2)
                rec[f"jev{j:g}_c"] = ce1
                rec[f"jev{j:g}_gn"] = ne1
            ok_, rep = check_direction(lambda j_: mp.direction(e, wav, jit_in=j_, seed=1),
                                       name=f"spk{spk} mel {nm}")
            rec["gate_1e-5"] = ok_
            gate_lines.append(rep)
            for j in (1e-4, 1e-3):
                ok_j, _ = check_direction(lambda j_: mp.direction(e, wav, jit_in=j_, seed=1),
                                          name=f"spk{spk} mel {nm}", jitter=j)
                rec[f"gate_{j:g}"] = ok_j
            add(**rec)
            dd = d0 / (d0.pow(2).mean().sqrt() + 1e-12) * 0.2
            bm = band_matched(dd, generator=torch.Generator(device=device).manual_seed(spk + 101))
            ps = pca_span(basis, dd, generator=torch.Generator(device=device).manual_seed(spk + 202))
            rn = _noise(mel_o.shape, spk, device)
            rn = rn / rn.pow(2).mean().sqrt() * 0.2
            for kind, vec in (("target", dd), ("band_matched", bm),
                              ("pca_span", ps), ("random", rn)):
                st = mel_structure(vec, mel_o)
                st.update(speaker=spk, path="mel", encoder=nm, kind=kind,
                          rms=float(vec.pow(2).mean().sqrt()),
                          cos_to_target=_cos(vec, dd))
                struct_rows.append(st)
            print(f"  mel {nm:28s} c={c0:.7f} |g|/|m|={n0:.2e} rep={rec['repeat_cos']:+.4f} "
                  f"jin1e-5={rec['jin1e-05_cos']:+.4f} jin1e-3={rec['jin0.001_cos']:+.4f} "
                  f"jev1e-2 pair={rec['jev0.01_pair']:+.4f} gate={'ok' if ok_ else 'FAIL'}",
                  flush=True)

        if not a.no_f64 and "mel" in paths:
            for e in [x for x in encs if x.name == "CAM++"] + unt:
                m64 = __import__("copy").deepcopy(e.m).cpu().double()
                d32, c32, n32 = mp.direction(e, wav)
                d64, c64, n64 = mel_direction64(mp, m64, wav)
                add(speaker=spk, path="mel-f64", encoder=e.name, cos_at_eval=c64,
                    grad_norm=n64, grad_norm_f32=n32,
                    f64_vs_f32_cos_at_x0=_cos(d32, d64.to(device)))
                print(f"  mel-f64 {e.name:24s} c f32={c32:.6f} f64={c64:.6f} "
                      f"|g|/|m| f32={n32:.2e} f64={n64:.2e}  cos(f32,f64)="
                      f"{rows[-1]['f64_vs_f32_cos_at_x0']:+.5f}", flush=True)
                del m64

        pd.DataFrame(rows).to_csv(a.out, index=False)
        stem = os.path.splitext(a.out)[0]
        pd.DataFrame(struct_rows).to_csv(stem + "_structure.csv", index=False)
        with open(stem + "_gate.txt", "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(gate_lines))

    print(f"\n[audit] saved {a.out}")
    print(f"[audit] saved {stem}_structure.csv and {stem}_gate.txt")


if __name__ == "__main__":
    main()
