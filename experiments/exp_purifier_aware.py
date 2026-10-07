"""
experiments/exp_purifier_aware.py
=================================
Purifier-aware perturbation crafting + honest end-to-end evaluation.

Variants (--variant):
  plain      minimise cos(E(x+d), E(x))                       (old end2end)
  surrogate  + cos(E(V(denoiser(mel(x+d)))), E(x))            single-pass UNet
  full       + cos(E(V(DiffPure_s(mel(x+d)))), E(x))          full DDPM loop,
             fresh noise seed each step (EOT), grad-checkpointed

V = fixed-original-phase differentiable mel->wav. All variants share the
psychoacoustic masking projection and the encoder ensemble
(WavLM-SV, ECAPA-TDNN, Resemblyzer d-vector proxy).

--craft-encoders: WHICH encoders the PGD loss sees. Evaluation always uses all
three, so dropping one here makes it a HELD-OUT encoder and the result a transfer
result rather than a white-box one. This exists because of a specific problem:
DiffVC's speaker conditioner is a GE2E checkpoint BYTE-IDENTICAL to the
Resemblyzer encoder (md5 a2e2a0bf4cc3ac59053d3fe4d67b0c4b), so crafting against
Resemblyzer makes every cloning number a matched-encoder number.

AntiFake (CCS 2023) Table 4 is the prior for what that is worth. Rows are the
crafting encoder, columns are five evaluated SYNTHESIZERS (not five encoders --
the match is implicit, via each synthesizer's internal encoder). Single-encoder
crafting protects the synthesizer built on that encoder at 99.6-100 %, and the
other four at 0-20.6 %. The row that matches our situation exactly is GE2E, the
same encoder family as Resemblyzer: 100 % on SV2TTS, which uses it, against
4.8 / 1.2 / 20.6 / 16.8 % on the four that do not. A cloning claim crafted against
the cloner's own encoder is therefore not evidence about deployment.

  --craft-encoders all      WavLM-SV + ECAPA + d-vector   (the published setting)
  --craft-encoders no-dvec  WavLM-SV + ECAPA              (d-vector held out)

Evaluation is the attacker's real pipeline: real DiffPure (t*=50, several
seeds) -> Griffin-Lim -> real encoder paths.

Run:
  python experiments/exp_purifier_aware.py --variant full --n-files 5 --steps 200
  python experiments/exp_purifier_aware.py --variant full --dataset librispeech --n-files 40
Output:
  experiments/results/purifier_aware.csv  (appends rows tagged by variant)
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
from utils.csvio import read_csv as read_csv_safe
from utils.provenance import provenance, add_prereg_arg
import torch
import torch._dynamo  # noqa: F401  (must load BEFORE speechbrain: its lazy k2 import breaks dynamo's module scan)
import transformers  # noqa: F401  (same reason: transformers registers torch ops on import)
import torch.nn.functional as F
try:
    import torchaudio.transforms as T
except Exception:   # Smart App Control blocks libtorchaudio.pyd (2026-09-15)
    from utils import tacompat as T
from torch.utils.checkpoint import checkpoint
from pathlib import Path

from core.psychoacoustic import (PsychoacousticMasker, project_delta_to_masking,
                                 waveform_snr_db)
from core.optimizer import _build_mel_to_wav, _mel_to_wav_differentiable
from evaluation.evaluate import (load_wav, wav_to_logmel, save_wav,
                                 SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH)
from utils.vocoder import invert_mel
from models.real_diffpure import load_real_diffpure
from experiments.multi_encoders import (ECAPAEnc, DvecEnc, WavLMSVEnc,
                                        real_embed_ecapa, real_embed_dvec,
                                        real_embed_wavlm_sv)

OUT_DIR = os.path.join(PROJ, "experiments", "results")
WAV_DIR = os.path.join(OUT_DIR, "purifier_aware")
SAVEDIR = os.path.join(PROJ, "experiments", "sb_models")
os.makedirs(WAV_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
def build_encoders(device, with_xvector=False):
    """The evaluation ensemble. x-vector is OPT-IN and APPENDED LAST.

    Appended rather than inserted so the existing indices 0,1,2 keep meaning
    WavLM-SV / ECAPA / d-vector. Ten scripts call this function; defaulting the
    fourth encoder off means none of them change behaviour, and no CSV written
    before today acquires a new column.
    """
    sv = WavLMSVEnc(device)          # transformers model FIRST (before speechbrain import)
    from speechbrain.inference.speaker import EncoderClassifier
    from resemblyzer import VoiceEncoder
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(SAVEDIR, "ecapa"), run_opts={"device": str(device)})
    for p in ecapa.mods.parameters():
        p.requires_grad_(False)
    dvec = VoiceEncoder(device=device)
    for p in dvec.parameters():
        p.requires_grad_(False)
    encs = [sv, ECAPAEnc(ecapa), DvecEnc(dvec, device)]
    if with_xvector:
        from experiments.multi_encoders import build_xvector
        encs.append(build_xvector(device, SAVEDIR))
    return encs


# Selected BY ENCODER NAME, not by index. Index-based sets break silently the
# moment the ensemble grows: "no-dvec": (0, 1) would still return two encoders
# after a fourth is added, just not the intended ones.
# Each entry is (include, exclude). `include=None` means "everything
# build_encoders returned", so a set defined by EXCLUSION keeps working as the
# ensemble grows. That matters: "no-dvec" must mean "hold out the cloner's
# encoder", whatever else is present. Spelling it as a fixed include-list broke
# the existing strict runs the moment a fourth encoder was defined, because they
# do not pass --with-xvector and the list then demanded a model that was absent.
CRAFT_SETS = {
    "all":      (None, ()),                             # every encoder built
    "no-dvec":  (None, ("Resemblyzer d-vector",)),      # hold out the cloner's
    "wavlm":    (("WavLM-SV",), ()),
    "ecapa":    (("ECAPA-TDNN",), ()),
}


def select_craft(encoders, which):
    """Encoders the optimiser is allowed to see. The rest are held out.

    Returns (subset, indices). Never mutates `encoders` -- evaluation keeps all
    three, which is the whole point: a held-out encoder only means anything if it
    is still scored."""
    include, exclude = CRAFT_SETS[which]
    names = [e.name for e in encoders]
    if include is not None:
        missing = set(include) - set(names)
        if missing:
            raise SystemExit(
                f"--craft-encoders {which} needs {sorted(missing)}, which "
                f"build_encoders did not return (pass --with-xvector?)")
    idx = tuple(i for i, e in enumerate(encoders)
                if (include is None or e.name in include) and e.name not in exclude)
    if not idx:
        raise SystemExit(f"--craft-encoders {which} selected no encoders")
    return [encoders[i] for i in idx], idx


def real_embed_all(encs, wav):
    """Real (eval-time) embeddings, keyed by encoder name.

    Was `sv, ec, dv = encs`, which silently fixed the ensemble at three and would
    raise the moment a fourth was added. Dispatches on type instead, so adding an
    encoder is a one-line change in build_encoders.
    """
    out = {}
    for e in encs:
        if isinstance(e, WavLMSVEnc):
            out[e.name] = real_embed_wavlm_sv(e, wav)
        elif isinstance(e, ECAPAEnc):
            out[e.name] = real_embed_ecapa(e.clf, wav)
        elif isinstance(e, DvecEnc):
            out[e.name] = real_embed_dvec(e.dvec, wav)      # preprocess + partials
        else:
            with torch.no_grad():                           # x-vector and future ones
                out[e.name] = e.embed(wav)
    return out


def cs(a, b):
    a = F.normalize(a.float().reshape(1, -1), dim=-1)
    b = F.normalize(b.float().reshape(1, -1), dim=-1)
    return (a * b).sum().item()


# ----------------------------------------------------------------------------
# Ensemble aggregation and ship-iterate selection.
#
# Extracted from craft()'s closures so they can be golden-tested (see
# tests/test_selection.py). Extraction is behaviour-preserving: with
# scales=None and legacy=True these compute exactly what the closures computed
# for every arm crafted up to tag recipe-v1.
#
# The two defects they now also implement fixes for, both OFF by default:
#
#   2.1 / PLAN_V3 2a  the ship-iterate was chosen by RAW COSINE, not by margin
#        to threshold, although thresholds span 0.59 across encoders
#        (ECAPA 0.336, d-vector 0.703, WavLM-SV 0.893). Raw-cosine selection can
#        prefer an iterate with a worse d-vector margin.
#
#   2e / RESULTS 35    the LSE smooth-maxes RAW margins, which are not
#        commensurable across encoders: the identity of the binding encoder
#        changes for 4 of 29 speakers once margins are divided by each encoder's
#        impostor SD. Dividing first makes any tau_lse discussion mean what it
#        says.
#
# RESULTS 36d is why they are off: applying either changes v1, and PLAN_V3
# freezes the base. They ship as the §36c disclosure paragraph and are the
# candidates for the W6 --n-eot 5 parity ablation.
# ----------------------------------------------------------------------------
def agg_lse(c, thr_t, w, tau_lse, scales=None):
    """Weighted smooth-max of the threshold margins. scales=None -> legacy."""
    m = c - thr_t
    if scales is not None:
        m = m / scales
    return torch.logsumexp(m * w * tau_lse, dim=0) / tau_lse


def ship_score(c, thr_t, scales=None, legacy=True):
    """The scalar the ship-iterate is selected on. Lower is better.

    legacy=True reproduces `c.max().item()` exactly, including its blindness to
    the per-encoder threshold.
    """
    if legacy:
        return float(c.max().item())
    m = c - thr_t
    if scales is not None:
        m = m / scales
    return float(m.max().item())


# ----------------------------------------------------------------------------
# Differentiable DiffPure (full DDPM loop, gradient-checkpointed)
# ----------------------------------------------------------------------------
def diffpure_grad(dp, x_mel, gen, stride=1):
    """Same maths as MelDiffPure.forward but with an explicit generator and
    torch.utils.checkpoint around every denoiser call (memory ~ 1 pass).

    STRIDED DDIM (--craft-stride, added 2026-09-19).
    ------------------------------------------------
    This loop is the entire cost of crafting: 200 optimisation steps x t denoiser
    passes = 10,000 passes per speaker at t=50. But DDIM is deterministic and
    non-Markovian by construction, so its trajectory can be subsampled -- that is
    the property the DDIM paper exists to provide. `stride=5` visits
    50,45,40,...,5,1 instead of every integer, cutting the dominant cost ~5x.

    De-AntiFake's App. C names exactly this ("accelerated sampling as a surrogate
    to reduce computational-graph depth") as what would make adaptive attacks on
    diffusion purification tractable in speech, where memory is the obstacle.

    IMPORTANT, and the reason `stride` defaults to 1: the gradient through a
    strided trajectory APPROXIMATES the gradient through the full one. It is a
    crafting-time speedup only. Evaluation always runs the attacker's real,
    full-resolution purifier, so the number we report is never the approximation.
    Validate before trusting it: experiments/validate_stride.py.

    DDPM is left unstrided -- it re-injects noise at every step, so skipping steps
    changes the marginal distribution rather than just the discretisation.
    """
    t = dp.t_diffuse
    abt = dp.alpha_bar[t - 1]                       # same (fixed) indexing as MelDiffPure
    eps = torch.randn(x_mel.shape, generator=gen, device=x_mel.device)
    x = abt.sqrt() * x_mel + (1.0 - abt).sqrt() * eps
    is_ddim = getattr(dp, "sampler", "ddpm") == "ddim"
    if is_ddim and stride > 1:
        # Descending timesteps, always landing on 1 so the final step reaches x0.
        steps = list(range(t, 0, -stride))
        if steps[-1] != 1:
            steps.append(1)
    else:
        steps = list(reversed(range(1, t + 1)))
    for k, i in enumerate(steps):
        ab_i = dp.alpha_bar[i - 1]
        # The DDIM update needs alpha_bar at the NEXT timestep we will visit, not
        # at i-1. With stride=1 these coincide; with a stride they do not, and
        # using i-1 would apply a one-step update across a multi-step jump.
        nxt = steps[k + 1] if k + 1 < len(steps) else 0
        ab_prev = (dp.alpha_bar[nxt - 1] if nxt >= 1
                   else torch.ones((), device=x.device))
        beta_i = dp.betas[i - 1]
        x0 = checkpoint(dp.denoiser, x, use_reentrant=False).clamp(-10.0, 10.0)
        if is_ddim:
            eps_pred = (x - ab_i.sqrt() * x0) / (1.0 - ab_i).sqrt()
            x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps_pred
            continue
        coef1 = (ab_prev.sqrt() * beta_i) / (1.0 - ab_i)
        coef2 = ((1.0 - ab_prev) * (1.0 - beta_i).sqrt()) / (1.0 - ab_i)
        mean = coef1 * x0 + coef2 * x
        if i > 1:
            var = beta_i * (1.0 - ab_prev) / (1.0 - ab_i)
            sigma = var.clamp(min=1e-20).sqrt()
            x = mean + sigma * torch.randn(x.shape, generator=gen, device=x.device)
        else:
            x = mean
    return x


# ----------------------------------------------------------------------------
# Differentiable Griffin-Lim surrogate (matches the attacker's vocoder better
# than the fixed-original-phase inversion: random phase init + momentum, as in
# torchaudio.transforms.GriffinLim, but only a few iterations, all
# differentiable STFT/iSTFT ops, and peak-normalised like utils/vocoder.py).
# ----------------------------------------------------------------------------
def griffin_lim_differentiable(mag, n_iter, window, n_fft, hop, length, gen,
                               momentum=0.99):
    F_, T_ = mag.shape
    ang = torch.rand(F_, T_, generator=gen, device=mag.device) * 2 * np.pi
    spec = mag * torch.polar(torch.ones_like(mag), ang)
    prev = torch.zeros_like(spec)
    for _ in range(n_iter):
        wav = torch.istft(spec, n_fft, hop, window=window, length=length)
        S = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
        S = S - (momentum / (1 + momentum)) * prev
        prev = S
        spec = mag * (S / (S.abs() + 1e-8))
    wav = torch.istft(spec, n_fft, hop, window=window, length=length)
    return wav / (wav.abs().max() + 1e-6)


# ----------------------------------------------------------------------------
# Crafting
# ----------------------------------------------------------------------------
def craft(wav_o, encoders, weights, masker, cap, diffpure, variant, n_steps, lr,
          bandlimit_hz, lam_before, n_eot, tau_lse=4.0, seed=42, log_every=20,
          vocoder="phase", gl_iters=8, thr=None, lam_null=0.0, null_t=0,
          null_mode="full", seg_partials=False, craft_stride=1,
          grad_estimator="exact", seg_tau=8.0, legacy_selection=True,
          enc_scales=None, sel_window=10):
    device = wav_o.device
    orig = [e.embed(wav_o).detach() for e in encoders]
    w = torch.tensor(weights, device=device)
    # Threshold-aware ensemble loss: LSE over (cos_e - tau_e) so the optimiser
    # works on whichever encoder is closest to ACCEPTING the clone, instead of
    # the one with the largest raw cosine (which ignores that ECAPA accepts at
    # 0.32 while WavLM-SV needs 0.89). thr=None -> raw cosines (legacy).
    thr_t = (torch.tensor(thr, device=device) if thr is not None
             else torch.zeros(len(encoders), device=device))
    # 2e. Per-encoder score scale, so the LSE and the selection statistic
    # compare margins in commensurable units. None -> legacy raw margins.
    # FIX 2026-10-07 (council v5 code review, "Trap"). This read
    #     if enc_scales is not None and not legacy_selection:
    # which WELDED two independent fixes to one flag: 2e (per-encoder score
    # scaling in the LSE) and 2.1/2d (margin-based, trailing-mean ship
    # selection). Passing enc_scales with legacy selection on was silently
    # ignored, so the W6 parity ablation could not vary them separately and a
    # caller who asked for one and got neither had no way to tell.
    #
    # They are now independent: enc_scales applies whenever it is supplied.
    # The legacy default is unchanged, because the call site only supplies
    # enc_scales under --no-legacy-selection, so no existing run moves.
    scales_t = None
    if enc_scales is not None:
        scales_t = torch.tensor([float(enc_scales[e.name]) for e in encoders],
                                device=device)

    inv_mel_scale, phase_orig, win, wav_len, T_frames = _build_mel_to_wav(
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE, x_wav=wav_o, device=device)
    mel_tf = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                              hop_length=HOP_LENGTH, n_mels=N_MELS).to(device)
    gl_gen = torch.Generator(device=device)
    gl_gen.manual_seed(seed + 2000)

    def to_wav(mel_log):
        if vocoder == "phase":
            wav = _mel_to_wav_differentiable(mel_log, inv_mel_scale, phase_orig,
                                             win, N_FFT, HOP_LENGTH, T_frames, wav_len)
            return wav / (wav.abs().max() + 1e-6)
        mag = inv_mel_scale(torch.exp(mel_log).clamp(min=1e-9).squeeze(0))
        return griffin_lim_differentiable(mag, gl_iters, win, N_FFT, HOP_LENGTH,
                                          wav_len, gl_gen).unsqueeze(0)

    # FIX P0, part 2 (2026-09-19). Segment-level scoring, per AntiFake Sec. C2.
    #
    # Part 1 made `embed()` return the same quantity the evaluator computes (the
    # average over 1.6 s partials). That fixes FIDELITY -- we now optimise the
    # right function. This fixes OPTIMISATION QUALITY, which is a separate thing.
    #
    # AntiFake C2: "directly deviating the final embedding is susceptible to local
    # optima, as the impact of a partial embedding not close to the target is
    # mitigated by other partials that may have attained better optimal points."
    #
    # So for any encoder exposing per-partial embeddings, we take a SMOOTH MAX of
    # the per-partial cosines instead of the cosine of the average. The optimiser
    # then works on the partial that still matches the speaker best -- the weak
    # link the verifier would find -- rather than being satisfied by a good mean.
    # seg_tau controls the softness; the limit seg_tau -> inf is a hard max.
    def ens_cos(wav, seg_tau=seg_tau):
        out = []
        for e, o in zip(encoders, orig):
            if seg_partials and hasattr(e, "embed_partials"):
                p = e.embed_partials(wav)                  # [n_partials, d]
                c = (p * o.reshape(1, -1)).sum(dim=-1)     # cosine per partial
                out.append(torch.logsumexp(c * seg_tau, dim=0) / seg_tau)
            else:
                out.append((e.embed(wav) * o).sum())
        return torch.stack(out)

    def lse(c):
        return agg_lse(c, thr_t, w, tau_lse, scales_t)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1000)
    torch.manual_seed(seed)
    delta = ((torch.rand_like(wav_o) * 2 - 1) * 2e-3).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_score, best_delta = float("inf"), delta.detach().clone()
    sel_hist = []          # trailing window of scores, for the 2d selection fix
    t0 = time.time()

    # Null-space (purification-invariance) regulariser: first-order surrogate
    # of "the purifier passes delta through". C(m) = P(m) - m is the
    # correction network (single pass, cheap: 1 UNet fwd/bwd per step vs the
    # full loop). Penalise the part of the mel delta that C reacts to:
    #     R = ||C(mel(x+d)) - C(mel(x))||^2 / ||mel(x+d) - mel(x)||^2
    # R -> 0 means delta lies in null(J_C) = the pass-through subspace.
    # null_t > 0: NOISE-MATCHED version. The purifier only ever sees delta after
    # the forward process has added noise at level t, and removal happens in
    # the first (noisiest) denoising steps. So require pass-through of the
    # x0-predictor AT that noise level (same noise draw for x and x+d, fresh
    # draw each step = EOT-lite, one denoiser pass):
    #     R_t = || m(x_t + sqrt(ab_t) d) - m(x_t) - d ||^2 / ||d||^2 ,
    #     x_t = sqrt(ab_t) mel(x) + sqrt(1-ab_t) eps.
    mel_o_log = torch.log(mel_tf(wav_o.detach()) + 1e-9)
    with torch.no_grad():
        c_o = (diffpure.denoiser(mel_o_log) - mel_o_log) if (lam_null > 0 and null_t == 0) else None
    ab_t = diffpure.alpha_bar[null_t - 1] if null_t > 0 else None
    null_gen = torch.Generator(device=device)
    null_gen.manual_seed(seed + 3000)

    for s in range(1, n_steps + 1):
        opt.zero_grad()
        wav_p = wav_o.detach() + delta
        c_before = ens_cos(wav_p)
        reg = 0.0
        if lam_null > 0:
            mel_pn = torch.log(mel_tf(wav_p) + 1e-9)
            dmel = mel_pn - mel_o_log
            if null_t > 0:
                eps = torch.randn(mel_o_log.shape, generator=null_gen, device=device)
                x_t = ab_t.sqrt() * mel_o_log + (1 - ab_t).sqrt() * eps
                out_shift = diffpure.denoiser(x_t + ab_t.sqrt() * dmel) - diffpure.denoiser(x_t)
                if null_mode == "directed":
                    # DIRECTED pass-through. "full" asks the WHOLE perturbation to
                    # survive -- which is exactly the quantity this paper shows is
                    # the wrong objective (purification keeps delta large while
                    # absorbing its speaker-discriminative part). Instead require
                    # only the component along g to survive, where g is the mel
                    # direction that actually moves the speaker embeddings:
                    #     g = d/d(mel) LSE_e[cos(E_e(V(mel)), E_e(x))]
                    #     reg = (<out_shift, g> - <d, g>)^2 / <d, g>^2
                    # One extra encoder backward per step (vs t denoiser passes for
                    # EOT), so this is a cheap first-order surrogate for EOT.
                    mel_req = mel_pn.detach().clone().requires_grad_(True)
                    g = torch.autograd.grad(lse(ens_cos(to_wav(mel_req))), mel_req)[0].detach()
                    g = g / (g.norm() + 1e-8)
                    pin = (dmel * g).sum()
                    pout = (out_shift * g).sum()
                    reg = (pout - pin).pow(2) / (pin.pow(2) + 1e-8)
                else:
                    reg = (out_shift - dmel).pow(2).sum() / (dmel.pow(2).sum() + 1e-8)
            else:
                reg = ((diffpure.denoiser(mel_pn) - mel_pn) - c_o).pow(2).sum() / (dmel.pow(2).sum() + 1e-8)
        if variant == "plain":
            loss = lse(c_before) + lam_null * reg
            score = ship_score(c_before, thr_t, scales_t, legacy_selection)
        else:
            mel_p = torch.log(mel_tf(wav_p) + 1e-9)
            c_after_acc = 0.0
            score_acc = 0.0
            for _ in range(n_eot):
                if variant == "surrogate":
                    mel_pure = diffpure.denoiser(mel_p).clamp(-10, 10)
                elif grad_estimator == "bpda":
                    # BPDA (Athalye et al. 2018): real purifier forward, identity
                    # backward. mel_pure == P(mel_p) in value; d(mel_pure)/d(mel_p) == I.
                    with torch.no_grad():
                        pur = diffpure_grad(diffpure, mel_p, gen, stride=craft_stride)
                    mel_pure = mel_p + (pur - mel_p).detach()
                else:
                    mel_pure = diffpure_grad(diffpure, mel_p, gen,
                                             stride=craft_stride)
                c_after = ens_cos(to_wav(mel_pure))
                c_after_acc = c_after_acc + lse(c_after) / n_eot
                score_acc += ship_score(c_after, thr_t, scales_t,
                                        legacy_selection) / n_eot
            loss = c_after_acc + lam_before * lse(c_before) + lam_null * reg
            score = score_acc
        loss.backward()
        with torch.no_grad():
            # 2d / RESULTS 36. `score` is a SINGLE purification draw at n_eot=1
            # (diffpure_grad draws fresh eps each call), so argmin over 200 steps
            # ships the iterate whose one draw was luckiest -- a winner's curse,
            # against an attacker who selects best-of-5 (§28c: worth
            # +0.020-0.037 to them). Selecting on a trailing mean estimates the
            # quality of the TRAJECTORY instead of the luck of one draw, and
            # costs no extra purifier passes.
            #
            # Legacy keeps the single-draw argmin, because that is what crafted
            # every arm up to recipe-v1.
            if legacy_selection:
                if score < best_score:
                    best_score, best_delta = score, delta.detach().clone()
            else:
                sel_hist.append(score)
                if len(sel_hist) > sel_window:
                    sel_hist.pop(0)
                if len(sel_hist) == sel_window:
                    sma = sum(sel_hist) / len(sel_hist)
                    if sma < best_score:
                        best_score, best_delta = sma, delta.detach().clone()
        opt.step()
        with torch.no_grad():
            delta.data.copy_(project_delta_to_masking(
                delta.data, cap, masker, bandlimit_hz=bandlimit_hz).to(device))
        if s % log_every == 0 or s == 1:
            if lam_null > 0:
                print(f"    [{variant}] null-reg R={float(reg):.4f}", flush=True)
            print(f"    [{variant}] step {s:3d}/{n_steps} | score={score:+.3f} "
                  f"| before={[round(v, 2) for v in c_before.tolist()]} "
                  f"| {time.time() - t0:.0f}s", flush=True)
    best_delta = project_delta_to_masking(best_delta, cap, masker,
                                          bandlimit_hz=bandlimit_hz).to(device)
    return best_delta


# ----------------------------------------------------------------------------
# Attacker resynthesis (mel -> waveform) after purification.
#   gl    : Griffin-Lim from scratch (utils/vocoder.py, 64 iters, random init)
#   phase : magnitude from the purified mel + STFT phase of the waveform the
#           attacker already holds (x_p, or x for the control), + `refine`
#           Griffin-Lim iterations. This is how mel-domain enhancement is
#           normally applied to a waveform and gives far better identity
#           retention than Griffin-Lim from scratch.
# ----------------------------------------------------------------------------
_INV_MEL = {}


def attacker_resynth(mel_log, ref_wav, method="gl", refine=8):
    if method == "gl":
        return invert_mel(mel_log.cpu()).to(ref_wav.device)
    device = ref_wav.device
    key = str(device)
    if key not in _INV_MEL:
        _INV_MEL[key] = T.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS,
                                          sample_rate=SAMPLE_RATE, driver="gels").to(device)
    win = torch.hann_window(N_FFT, device=device)
    with torch.no_grad():
        S = torch.stft(ref_wav.squeeze(0), N_FFT, HOP_LENGTH, window=win, return_complex=True)
        phase = S / (S.abs() + 1e-8)
        mag = _INV_MEL[key](torch.exp(mel_log.to(device)).clamp(min=1e-9).squeeze(0))
        Tn = min(mag.shape[-1], phase.shape[-1])
        mag, phase = mag[..., :Tn], phase[..., :Tn]
        length = ref_wav.shape[-1]
        spec = mag * phase
        for _ in range(refine):
            wav = torch.istft(spec, N_FFT, HOP_LENGTH, window=win, length=length)
            S2 = torch.stft(wav, N_FFT, HOP_LENGTH, window=win, return_complex=True)
            spec = mag * (S2 / (S2.abs() + 1e-8))
        wav = torch.istft(spec, N_FFT, HOP_LENGTH, window=win, length=length)
        return (wav / (wav.abs().max() + 1e-6)).unsqueeze(0)


# ----------------------------------------------------------------------------
# Evaluation: attacker's real pipeline
# ----------------------------------------------------------------------------
def quality(wav_o, wav_p):
    snr = waveform_snr_db(wav_o.cpu(), wav_p.cpu())
    x = wav_o.squeeze().cpu().numpy()
    y = wav_p.squeeze().cpu().numpy()[: x.shape[0]]
    try:
        from pesq import pesq as pf
        pesq_v = pf(16000, x, y, "wb")
    except Exception:
        pesq_v = float("nan")
    try:
        from pystoi import stoi as sf_
        stoi_v = sf_(x, y, 16000, extended=False)
    except Exception:
        stoi_v = float("nan")
    return snr, pesq_v, stoi_v


def eval_e2e(wav_o, wav_p, diffpure, encs, thresholds, eval_seeds, device,
             attacker_vocoder="gl", eval_t=None):
    if eval_t is not None and eval_t != diffpure.t_diffuse:
        from models.real_diffpure import MelDiffPure
        diffpure = MelDiffPure(diffpure.denoiser, n_steps=diffpure.n_steps,
                               t_diffuse=eval_t, sampler=diffpure.sampler).to(device).eval()
    mel_o = wav_to_logmel(wav_o, device)
    mel_p = wav_to_logmel(wav_p, device)
    emb_o = real_embed_all(encs, wav_o)
    emb_p = real_embed_all(encs, wav_p)
    after = {n: [] for n in thresholds}
    ctrl = {n: [] for n in thresholds}
    for sd in eval_seeds:
        with torch.no_grad():
            mel_pp = diffpure(mel_p, seed=sd)
            mel_op = diffpure(mel_o, seed=sd)
        wav_pp = attacker_resynth(mel_pp, wav_p, attacker_vocoder)   # attacker holds x_p
        wav_op = attacker_resynth(mel_op, wav_o, attacker_vocoder)   # control holds x
        e_pp = real_embed_all(encs, wav_pp)
        e_op = real_embed_all(encs, wav_op)
        for n in thresholds:
            after[n].append(cs(emb_o[n], e_pp[n]))
            ctrl[n].append(cs(emb_o[n], e_op[n]))
    rows = {}
    for n, tau in thresholds.items():
        ca = float(np.mean(after[n])); cc = float(np.mean(ctrl[n]))
        rows[n] = dict(threshold=tau,
                       cos_before=round(cs(emb_o[n], emb_p[n]), 4),
                       cos_after_purify=round(ca, 4),
                       cos_after_max=round(float(np.max(after[n])), 4),
                       cos_ctrl_unprot=round(cc, 4),
                       clone_fails_after=bool(ca < tau),
                       fails_all_seeds=bool(np.max(after[n]) < tau),
                       ctrl_identity_kept=bool(cc >= tau),
                       defence_valid=bool(ca < tau and cc >= tau))
    return rows


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def list_files(args):
    if args.dataset == "samples":
        return [(p.stem, str(p), None) for p in
                sorted(Path(args.input).glob("*.wav"))[: args.n_files]]
    # librispeech dev-clean: one utterance per speaker, >= min_sec long.
    # --utt-index picks which utterance of each speaker (calibration uses 0..2,
    # so the protection set defaults to index 3 = disjoint from calibration).
    from experiments.librispeech import list_librispeech, per_speaker, EVAL_SUBSET
    root = os.path.join(PROJ, "data", "librispeech")
    # --subset pins the corpus. It exists so an ABLATION can be run against the
    # same speakers as the run it is being compared to: EVAL_SUBSET moved from
    # dev-clean to test-clean on 2026-09-17, and re-crafting under the new default
    # would confound the thing being ablated with a change of corpus.
    subset = args.subset or EVAL_SUBSET
    items = list_librispeech(root, subset, min_sec=args.min_sec)

    # --speaker-set restricts to a NAMED, published speaker list so one row of our
    # table can be placed directly beside a baseline's rather than argued to be
    # comparable. "deantifake" is their 25 published test-clean speakers.
    #
    # Note the conflict this exposes: only 18 of those 25 have two chapters in
    # test-clean, so cross-session thresholds cannot cover the other seven. We
    # therefore craft the UNION and score two rows -- their 25 under the protocol
    # they used, and the 18 that overlap under ours. The gap between the rows
    # measures what the protocol is worth, which nobody in this literature reports.
    if args.speaker_set:
        from experiments import librispeech as _ls
        keep = set(getattr(_ls, args.speaker_set.upper() + "_SPEAKERS"))
        items = [it for it in items if it["speaker"] in keep]
        print(f"[PA] speaker-set={args.speaker_set}: restricted to "
              f"{len({i['speaker'] for i in items})} of {len(keep)} named speakers "
              f"present in {subset}")

    # Reference selection goes through ONE shared function (librispeech.
    # reference_items) that the cloning evaluation also calls, so the file we
    # protect and the file the cloner is later handed cannot diverge. They did
    # diverge on 2026-09-19 -- two code paths, 9 of 18 speakers silently skipped.
    #
    # Note this is independent of --cross-session, which controls the THRESHOLD
    # protocol and which speakers survive it. Reference choice and threshold
    # protocol are separate decisions and are now coded that way.
    if args.cross_session or args.speaker_set:
        from experiments.librispeech import reference_items, cross_session_split
        ref = reference_items(items, min_sec=args.min_sec, utt_index=args.utt_index)
        keep = sorted(ref)
        if args.cross_session:
            ok = set(cross_session_split(items, min_sec=args.min_sec))
            keep = [s for s in keep if s in ok]
        out = []
        for spk in keep[args.spk_offset:]:
            it = ref[spk]
            out.append((f"spk{spk}_{it['chapter']}_{it['utt']:04d}",
                        ("flac", it["path"]), spk))
            if len(out) >= args.n_files:
                break
        print(f"[PA] {len(out)} speakers from {subset}; reference = largest chapter "
              f"(shared selector){', cross-session only' if args.cross_session else ''}")
        return out

    per_spk = per_speaker(items, n_utt=args.utt_index + 1, n_spk=None)
    out = []
    for spk in sorted(per_spk)[args.spk_offset:]:
        lst = per_spk[spk]
        if len(lst) <= args.utt_index:
            continue
        it = lst[args.utt_index]
        out.append((f"spk{spk}_{it['chapter']}_{it['utt']:04d}", ("flac", it["path"]), spk))
        if len(out) >= args.n_files:
            break
    return out


def load_item(item, args, device):
    name, src, _ = item
    if isinstance(src, str):
        wav = load_wav(src, max_seconds=args.max_sec)
    else:
        from experiments.librispeech import load_flac
        wav = load_flac(src[1], max_sec=args.max_sec)
    return wav.to(device)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["plain", "surrogate", "full"])
    ap.add_argument("--dataset", default="samples", choices=["samples", "librispeech"])
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="FIX S1: take the reference from the same chapter the "
                         "cross-session thresholds use as the reference side, and "
                         "drop single-chapter speakers, so crafting and calibration "
                         "agree on the speaker population.")
    ap.add_argument("--craft-stride", type=int, default=1, dest="craft_stride",
                    help="subsample the DDIM trajectory during CRAFTING only "
                         "(1 = every step, the default). Cuts the dominant cost "
                         "~linearly. Evaluation always uses the full-resolution "
                         "purifier. Validate with experiments/validate_stride.py "
                         "before using in a reported run.")
    ap.add_argument("--seg-partials", action="store_true", dest="seg_partials",
                    help="FIX P0: score every 1.6 s partial and optimise the WORST "
                         "one (smooth max) instead of the pooled embedding. Adopts "
                         "AntiFake Sec. C2, which shows pooled deviation hits local "
                         "optima because good partials mask bad ones.")
    ap.add_argument("--speaker-set", default="", dest="speaker_set",
                    choices=["", "deantifake", "cross_session"],
                    help="restrict to a named published speaker list, so a row can "
                         "sit directly beside a baseline's table. 'deantifake' = "
                         "their 25 published test-clean speakers.")
    ap.add_argument("--with-xvector", action="store_true", dest="with_xvector",
                    help="add SpeechBrain x-vector to the ensemble (appended "
                         "LAST, so existing encoder order is unchanged). Off by "
                         "default: every result before 2026-09-20 was produced "
                         "without it.")
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: librispeech.EVAL_SUBSET, "
                         "currently test-clean). Pin it to match a run you are "
                         "comparing against.")
    ap.add_argument("--input", default=os.path.join(PROJ, "data", "samples"))
    ap.add_argument("--n-files", type=int, default=5, dest="n_files")
    ap.add_argument("--min-sec", type=float, default=4.0, dest="min_sec")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--max-sec", type=float, default=4.0, dest="max_sec")
    ap.add_argument("--margin-db", type=float, default=30.0, dest="margin_db")
    ap.add_argument("--snr-floor", type=float, default=None, dest="snr_floor",
                    help="per-frame local SNR floor in dB (delta <= frame energy - floor); "
                         "kills audible noise in pauses")
    ap.add_argument("--bandlimit", type=float, default=6000.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-before", type=float, default=0.3, dest="lam_before")
    ap.add_argument("--lam-null", type=float, default=0.0, dest="lam_null",
                    help="weight of the null-space (purification-invariance) regulariser")
    ap.add_argument("--null-mode", default="full", choices=["full", "directed"],
                    dest="null_mode",
                    help="full: require the whole perturbation to pass through; "
                         "directed: require only its speaker-discriminative component to")
    ap.add_argument("--null-t", type=int, default=0, dest="null_t",
                    help="noise level (t) at which the null-space regulariser is evaluated; "
                         "0 = zero-noise single pass")
    ap.add_argument("--n-eot", type=int, default=1, dest="n_eot")
    ap.add_argument("--craft-seed", type=int, default=42, dest="craft_seed",
                    help="FIX S6 (2026-09-30). Seed for crafting randomness: delta "
                         "init, EOT purifier noise, Griffin-Lim, null-reg noise. Every "
                         "run before this date used the craft() default of 42. Seed "
                         "variance across crafting runs was unmeasured and visibly "
                         "larger than some reported effects (RESULTS 29e caveat).")
    ap.add_argument("--grad-estimator", default="exact", choices=["exact", "bpda"],
                    dest="grad_estimator",
                    help="HEAD-TO-HEAD CONTROL (council review, 2026-09-22). 'exact' "
                         "differentiates through the full DDIM reverse process "
                         "(checkpointed) -- our method. 'bpda' runs the SAME purifier "
                         "forward but treats it as the identity in the backward pass "
                         "(straight-through), which is the estimator De-AntiFake "
                         "Sec. 5.4 and VocalBridge Sec. V-E used when they concluded "
                         "adaptive protection fails. Everything else identical, so a "
                         "difference between the two arms is the gradient estimator "
                         "and nothing else.")
    ap.add_argument("--weights", default="1,1,1.5")
    ap.add_argument("--craft-encoders", default="all", choices=list(CRAFT_SETS),
                    dest="craft_encoders",
                    help="encoders the PGD loss sees; the rest are HELD OUT but "
                         "still evaluated. 'no-dvec' makes the DiffVC cloning "
                         "result a transfer result (see module docstring).")
    ap.add_argument("--thr-aware", action="store_true", dest="thr_aware",
                    help="ensemble loss on (cos - threshold) per encoder")
    # 2.3. Both sharpness knobs were hard-coded (tau_lse in craft()'s signature,
    # seg_tau in an inner closure's default). Defaults are unchanged, so no
    # existing arm moves; they are exposed because §23c/§27a showed that
    # changing where the budget concentrates reallocates protection ACROSS
    # encoders, which makes them ablation knobs rather than constants.
    ap.add_argument("--tau-lse", type=float, default=4.0, dest="tau_lse",
                    help="sharpness of the cross-encoder smooth-max (default 4.0, "
                         "the value every arm up to recipe-v1 used)")
    ap.add_argument("--seg-tau", type=float, default=8.0, dest="seg_tau",
                    help="sharpness of the per-partial smooth-max used by "
                         "--seg-partials (default 8.0, as crafted)")
    # 2.1 / 2d / 2e, all three off by default. RESULTS 36d: applying any of them
    # changes v1, and the base never moves. --no-legacy-selection turns on
    # margin-based selection, the per-encoder margin scale, and selection on a
    # trailing mean instead of one purification draw.
    ap.add_argument("--legacy-selection", action=argparse.BooleanOptionalAction,
                    default=True, dest="legacy_selection",
                    help="keep v1's ship-iterate selection: argmin over steps of "
                         "the single-draw MAX RAW COSINE. --no-legacy-selection "
                         "applies the 2.1/2d/2e fixes together (margin scale, "
                         "per-encoder units, trailing-mean selection).")
    ap.add_argument("--sel-window", type=int, default=10, dest="sel_window",
                    help="trailing window of steps the non-legacy selection "
                         "averages over before comparing (2d)")
    ap.add_argument("--scale-channel", default="diffpure_gl_ddim_v7_xsess_tc",
                    dest="scale_channel",
                    help="channel whose calibration supplies the 2e per-encoder "
                         "margin scales (only read with --no-legacy-selection)")
    ap.add_argument("--scale-t", type=int, default=50, dest="scale_t",
                    help="timestep of that calibration (2e)")
    ap.add_argument("--vocoder", default="phase", choices=["phase", "gl"],
                    help="differentiable mel->wav used INSIDE crafting")
    ap.add_argument("--gl-iters", type=int, default=8, dest="gl_iters")
    ap.add_argument("--attacker-vocoder", default="gl", choices=["gl", "phase"],
                    dest="attacker_vocoder", help="mel->wav used by the ATTACKER at eval")
    ap.add_argument("--t-diffuse", type=int, default=50, dest="t_diffuse")
    ap.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    ap.add_argument("--purifier-ckpt", default="", dest="purifier_ckpt",
                    help="denoiser checkpoint (default: checkpoints/purifier_trained.pt)")
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset",
                    help="librispeech: skip the first N speakers. DEFAULT CHANGED "
                         "2026-09-19 from 20 to 0. The old default existed because "
                         "the purifier trained on dev-clean speakers 0-19, so they "
                         "had to be skipped. Purifier v7 trains on train-clean-100, "
                         "disjoint from every evaluation speaker, so an offset of 20 "
                         "now silently discards most of a run: it cut a 29-speaker "
                         "test-clean job to 9 and left the cloning step looking for "
                         "wavs that were never crafted. Older dev-clean scripts pass "
                         "20 explicitly and are unaffected.")
    ap.add_argument("--eval-seeds", default="2026,7,99", dest="eval_seeds")
    ap.add_argument("--eval-t", default="", dest="eval_t",
                    help="comma list of purification strengths to EVALUATE at "
                         "(default: the crafting --t-diffuse only)")
    ap.add_argument("--thresholds", default="", help="wavlmsv,ecapa,dvec override")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "purifier_aware.csv"))
    add_prereg_arg(ap)
    args = ap.parse_args()
    PROV = provenance(args.prereg)   # code_sha / prereg_sha / argv, stored per row (W0)
    print(f"[PA] code_sha={PROV['code_sha']} prereg_sha={PROV['prereg_sha'] or '-'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PA] device={device} variant={args.variant} dataset={args.dataset} "
          f"n={args.n_files} steps={args.steps} margin={args.margin_db} "
          f"bl={args.bandlimit} lam_before={args.lam_before} n_eot={args.n_eot}")

    dp_kwargs = dict(t_diffuse=args.t_diffuse, device=device, sampler=args.sampler)
    if args.purifier_ckpt:
        dp_kwargs["checkpoint_path"] = args.purifier_ckpt
    diffpure = load_real_diffpure(**dp_kwargs)
    for p in diffpure.parameters():
        p.requires_grad_(False)
    encoders = build_encoders(device, with_xvector=args.with_xvector)
    thresholds = {e.name: e.threshold for e in encoders}
    if args.thresholds:
        for e, v in zip(encoders, args.thresholds.split(",")):
            thresholds[e.name] = float(v)
    elif args.thr_aware:
        # 2.2. The per-encoder `threshold` class attributes in multi_encoders.py
        # are MODEL-CARD PLACEHOLDERS (0.86 / 0.25 / 0.75), not calibrations of
        # this channel or this speaker population; the calibrated cross-session
        # values are 0.893 / 0.336 / 0.703. A threshold-aware loss built on the
        # placeholders optimises against the wrong accept region -- 0.047 of
        # false slack on the d-vector, twice any effect this paper reports --
        # and it would do so silently. Every run script already passes
        # --thresholds, so this refuses a mistake rather than changing a result.
        raise SystemExit(
            "[PA] --thr-aware without --thresholds: the ensemble loss would use "
            "the model-card placeholder thresholds in multi_encoders.py, not a "
            "calibration of this channel. Pass --thresholds (see "
            "experiments/_thr_triple.py and utils/thresholds.py), or drop "
            "--thr-aware.")
    # 2e. Per-encoder impostor SD, the divisor that makes cross-encoder margins
    # commensurable. Only consulted when --no-legacy-selection is passed; when
    # the calibration file carries no SD columns utils.thresholds falls back to
    # inferring sigma from the EER and says so.
    enc_scales = None
    if not args.legacy_selection:
        from utils.thresholds import encoder_scales
        enc_scales, how = encoder_scales(args.scale_channel, args.scale_t)
        print(f"[PA] 2e per-encoder margin scales ({how}, "
              f"channel={args.scale_channel} t={args.scale_t}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in enc_scales.items()))
        missing = [e.name for e in encoders if e.name not in enc_scales]
        if missing:
            raise SystemExit(f"[PA] no 2e scale for {missing} on channel "
                             f"{args.scale_channel}; calibrate it first")
    craft_encs, craft_idx = select_craft(encoders, args.craft_encoders)
    held_out = [e.name for i, e in enumerate(encoders) if i not in craft_idx]
    weights = [float(v) for v in args.weights.split(",")]
    if len(weights) == len(encoders) and len(craft_encs) != len(encoders):
        # --weights was given for the full ensemble; take the crafting subset so
        # existing run scripts keep working unchanged.
        weights = [weights[i] for i in craft_idx]
    if len(weights) != len(craft_encs):
        raise SystemExit(f"--weights has {len(weights)} values but the crafting "
                         f"ensemble has {len(craft_encs)} encoders "
                         f"({', '.join(e.name for e in craft_encs)})")
    print(f"[PA] crafting against: {', '.join(e.name for e in craft_encs)}")
    print(f"[PA] HELD OUT (evaluated, never optimised against): "
          f"{', '.join(held_out) if held_out else 'none -- this is a white-box result'}")
    masker = PsychoacousticMasker(margin_db=args.margin_db, device=device,
                                  local_snr_floor_db=args.snr_floor)
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    eval_ts = [int(v) for v in args.eval_t.split(",")] if args.eval_t else [args.t_diffuse]
    tag = args.tag or args.variant

    rows = []
    for item in list_files(args):
        name = item[0]
        print(f"\n{'=' * 64}\n[PA] {name}", flush=True)
        wav_o = load_item(item, args, device)
        cap = masker.magnitude_cap(wav_o)
        t0 = time.time()
        delta = craft(wav_o, craft_encs, weights, masker, cap, diffpure, args.variant,
                      args.steps, args.lr, args.bandlimit, args.lam_before, args.n_eot,
                      vocoder=args.vocoder, gl_iters=args.gl_iters,
                      thr=[thresholds[e.name] for e in craft_encs] if args.thr_aware else None,
                      lam_null=args.lam_null, null_t=args.null_t,
                      null_mode=args.null_mode, seg_partials=args.seg_partials,
                      craft_stride=args.craft_stride,
                      grad_estimator=args.grad_estimator, seed=args.craft_seed,
                      tau_lse=args.tau_lse, seg_tau=args.seg_tau,
                      legacy_selection=args.legacy_selection,
                      enc_scales=enc_scales, sel_window=args.sel_window)
        craft_s = time.time() - t0
        wav_p = (wav_o + delta).clamp(-1.0, 1.0)
        save_wav(wav_p.cpu(), os.path.join(WAV_DIR, f"{name}_{tag}.wav"))
        snr, pesq_v, stoi_v = quality(wav_o, wav_p)
        for et in eval_ts:
          res = eval_e2e(wav_o, wav_p, diffpure, encoders, thresholds, eval_seeds, device,
                         attacker_vocoder=args.attacker_vocoder, eval_t=et)
          if len(eval_ts) > 1:
              print(f"  -- evaluated at t_diffuse={et}")
          for enc_name, r in res.items():
            rows.append(dict(variant=tag, file=name, encoder=enc_name, eval_t=et, **r,
                             pesq=round(pesq_v, 3), stoi=round(stoi_v, 4),
                             snr_db=round(snr, 2), craft_s=round(craft_s, 1),
                             steps=args.steps, margin_db=args.margin_db,
                             lam_before=args.lam_before, n_eot=args.n_eot,
                             grad_estimator=args.grad_estimator,
                             vocoder=args.vocoder, weights=args.weights,
                             attacker_vocoder=args.attacker_vocoder,
                             snr_floor=args.snr_floor, thr_aware=args.thr_aware,
                             lam_null=args.lam_null, null_t=args.null_t,
                             null_mode=args.null_mode, sampler=args.sampler,
                             craft_encoders=args.craft_encoders,
                             subset=(args.subset or "default"),
                             cross_session=bool(args.cross_session),
                             seg_partials=bool(args.seg_partials),
                             craft_stride=args.craft_stride,
                             held_out=("|".join(held_out) or "none"),
                             purifier=os.path.basename(args.purifier_ckpt),
                             **PROV))
            print(f"  {enc_name:12s} tau={r['threshold']:.2f} | before={r['cos_before']:+.3f} "
                  f"after={r['cos_after_purify']:+.3f} (max {r['cos_after_max']:+.3f}) "
                  f"ctrl={r['cos_ctrl_unprot']:+.3f} | "
                  f"{'VALID' if r['defence_valid'] else 'weak'}", flush=True)
        print(f"  PESQ={pesq_v:.2f} STOI={stoi_v:.3f} SNR={snr:.1f}dB craft={craft_s:.0f}s")
        if device.type == "cuda":
            print(f"  peak GPU mem={torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    df = pd.DataFrame(rows)
    if os.path.isfile(args.out):
        old = read_csv_safe(args.out)
        if "eval_t" not in old.columns:
            old["eval_t"] = old.get("steps", 0) * 0 + 50
        old = old[~((old.variant == tag) & (old.file.isin(df.file)) & (old.eval_t.isin(df.eval_t)))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\n[PA] saved {args.out}")

    sub = df[df.variant == tag]
    print("\n" + "=" * 84)
    print(f"MEAN PER ENCODER — variant={tag} — cloning AFTER attacker's DiffPure+GL")
    print("=" * 84)
    for (et, enc_name), g in sub.groupby(["eval_t", "encoder"], sort=False):
        print(f"  t={et:3d} {enc_name:12s} tau={g.threshold.iloc[0]:.2f} | before={g.cos_before.mean():+.3f} "
              f"after={g.cos_after_purify.mean():+.3f} ctrl={g.cos_ctrl_unprot.mean():+.3f} | "
              f"fails after purify {100 * g.clone_fails_after.mean():.0f}% | "
              f"ctrl kept {100 * g.ctrl_identity_kept.mean():.0f}% | "
              f"VALID {100 * g.defence_valid.mean():.0f}%")
    q = sub.drop_duplicates("file")
    print(f"  quality: PESQ={q.pesq.mean():.2f} STOI={q.stoi.mean():.3f} SNR={q.snr_db.mean():.1f}dB")


if __name__ == "__main__":
    main()
