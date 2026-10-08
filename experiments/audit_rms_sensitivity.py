"""
experiments/audit_rms_sensitivity.py
====================================
Audit 2, risk 3 (RESULTS section 58): is alpha sensitive to the perturbation
scale on a non-linear purifier, and does every direction in a cell really
receive equal RMS?

alpha = <P(x + d) - P(x), d> / ||d||^2 is scale-free only for a linear P.
Both purifiers are non-linear, so alpha(s . d) may depend on s, and -- the
part that matters for a CONTRAST -- it may depend on s differently for an
encoder direction than for its control. Every alpha section equalises the
GLOBAL per-element RMS of all directions in a cell (`unit()`), which is the
right thing to equalise only if alpha's scale dependence is the same for all
of them. Nothing had measured that.

WHAT IS MEASURED
----------------
Part A  purifier v7 DDIM t = 50 (the primary-endpoint cell of every alpha
        section), 3 speakers, seeds 2026,7. Each direction is built once at
        unit RMS and then scaled to rms in {0.05, 0.1, 0.2, 0.4}; 0.2 is the
        protocol value. Directions: WavLM-SV, d-vector, CAM++, CAM++
        untrained seed 0, frontend:fbank-linear, with band-matched controls
        for three of them, plus random and pca_span. At rms 0.2 the NEGATED
        direction is also measured: alpha(-d) == alpha(d) for a linear P,
        and the size of the asymmetry is the size of the non-linearity at
        the operating point.
Part B  DiffWave-LibriSpeech t = 1 (the cell section 55 tabulated), 3
        speakers, rel_rms in {0.025, 0.05, 0.1, 0.2}; 0.05 is the protocol
        value. Directions: random, the section-55 residue direction for
        WavLM-SV (VOID per section 57 -- included only as the fixed vector
        the run actually used, so the scale question is answered on the
        same object), a genuine gradient taken 1e-2 off the maximum
        (exploratory, not a C1 measurement), and the band-matched control of
        each. Sign symmetry at 0.05.

Every row records the exact per-element RMS the direction received, so
"equal RMS" is checked rather than assumed.

Run (GPU, ~25 min):
    python experiments/audit_rms_sensitivity.py
Output:
    experiments/results/audit_rms_sensitivity.csv
"""
import argparse
import io
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
import torch._dynamo   # noqa: F401
import transformers    # noqa: F401

from core.matched_random import band_matched, band_matched_wav, clean_mel_basis, pca_span
from utils.provenance import provenance

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(R, "audit_rms_sensitivity.csv")


def _unit(v):
    return v / (v.pow(2).mean().sqrt() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=3)
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--mel-scales", default="0.05,0.1,0.2,0.4", dest="mel_scales")
    ap.add_argument("--wav-scales", default="0.025,0.05,0.1,0.2", dest="wav_scales")
    ap.add_argument("--skip-wav", action="store_true", dest="skip_wav")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    PROV = provenance()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(v) for v in a.seeds.split(",")]
    mel_scales = [float(v) for v in a.mel_scales.split(",")]
    wav_scales = [float(v) for v in a.wav_scales.split(",")]

    from models.real_diffpure import load_real_diffpure, MelDiffPure
    from experiments.exp_nullspace_survival import survival
    from experiments.exp_purifier_aware import build_encoders
    from experiments.extra_encoders import CAMPPlusEnc, UntrainedCAMPPlusEnc, fbank
    from experiments.librispeech import (cross_session_split, list_librispeech,
                                         load_flac, reference_items)
    from experiments.audit_direction_stability import (FbankLinearEnc, MelPath,
                                                       wav_direction)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"), "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][: a.speakers]
    print(f"[rms] code_sha={PROV['code_sha']} speakers {spks} seeds {seeds}")

    base = build_encoders(device, with_xvector=False)
    wavlm = base[0]
    dvec = base[2]
    wav0 = load_flac(ref[spks[0]]["path"], 4.0).to(device)
    encs = [wavlm, dvec, CAMPPlusEnc(device), UntrainedCAMPPlusEnc(device, seed=0),
            FbankLinearEnc(device, int(fbank(wav0).shape[0]))]
    ctl_for = {"WavLM-SV", "CAM++", "frontend:fbank-linear"}
    mp = MelPath(device)
    basis, _ = clean_mel_basis(k=20, subset="dev-clean", device=device, verbose=False)
    rows = []

    # ------------------------------------------------------------ Part A
    ck = os.path.join(PROJ, "checkpoints", "purifier_v7.pt")
    dp0 = load_real_diffpure(checkpoint_path=ck, t_diffuse=50, device=device, sampler="ddim")
    for p in dp0.denoiser.parameters():
        p.requires_grad_(False)
    dp = MelDiffPure(dp0.denoiser, n_steps=dp0.n_steps, t_diffuse=50, sampler="ddim").to(device).eval()
    print("[rms] Part A: purifier v7 DDIM t=50")
    for spk in spks:
        wav = load_flac(ref[spk]["path"], 4.0).to(device)
        mel_o, _ = mp.setup(wav)
        units = {}
        for e in encs:
            g, _, _ = mp.direction(e, wav)
            units[e.name] = _unit(g).detach()
        g = torch.Generator(device=device).manual_seed(spk)
        units["random"] = _unit(torch.randn(mel_o.shape, generator=g, device=device))
        for nm in list(units):
            if nm in ctl_for:
                gb = torch.Generator(device=device).manual_seed(spk + 101)
                units[f"band_matched:{nm}"] = _unit(band_matched(units[nm], generator=gb))
        gp = torch.Generator(device=device).manual_seed(spk + 202)
        units["pca_span"] = _unit(pca_span(basis, units[encs[0].name], generator=gp))
        for nm, u in units.items():
            for s in mel_scales:
                for sign in ((1, -1) if abs(s - 0.2) < 1e-9 else (1,)):
                    d = (sign * s * u).detach()
                    psr, cosv, alpha, nsr = survival(dp, mel_o, d, seeds, full=True)
                    rows.append(dict(part="A-mel-v7-t50", speaker=spk, direction=nm,
                                     scale=s, sign=sign,
                                     rms=float(d.pow(2).mean().sqrt()),
                                     peak=float(d.abs().max()),
                                     alpha=round(alpha, 5), psr=round(psr, 5),
                                     nsr=round(nsr, 5), dircos=round(cosv, 5), **PROV))
            got = [r for r in rows if r["speaker"] == spk and r["direction"] == nm and r["sign"] == 1]
            print(f"  spk{spk} {nm:34s} " + "  ".join(
                f"s={r['scale']:.2f}:{r['alpha']:+.3f}" for r in got), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    del dp, dp0
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ Part B
    if not a.skip_wav:
        from models.diffwave_purifier import load_diffwave
        dw = load_diffwave(os.path.join(PROJ, "checkpoints", "foreign",
                                        "diffwave_libri_deantifake.pkl"),
                           t=1, device=str(device))
        print("[rms] Part B: DiffWave-LibriSpeech t=1")
        for spk in spks:
            wav = load_flac(ref[spk]["path"], 4.0).to(device)
            wrms = float(wav.pow(2).mean().sqrt())
            units = {}
            g0, _, _ = wav_direction(wavlm, wav)                       # the void 55 vector
            units["wav-residue:WavLM-SV (VOID, 57)"] = _unit(g0)
            g1, _, _ = wav_direction(wavlm, wav, jit_ev=1e-2, seed=1)  # off the maximum
            units["wav-offmax1e-2:WavLM-SV (exploratory)"] = _unit(g1)
            g = torch.Generator(device=device).manual_seed(spk)
            units["random"] = _unit(torch.randn(wav.shape, generator=g, device=device))
            for nm in list(units):
                if nm != "random":
                    gb = torch.Generator(device=device).manual_seed(spk + 101)
                    units[f"band_matched:{nm}"] = _unit(band_matched_wav(units[nm], generator=gb))
            for nm, u in units.items():
                for s in wav_scales:
                    for sign in ((1, -1) if abs(s - 0.05) < 1e-9 else (1,)):
                        d = (sign * s * wrms * u).detach()
                        psr, cosv, alpha, nsr = survival(dw, wav, d, seeds, full=True)
                        rows.append(dict(part="B-diffwave-libri-t1", speaker=spk, direction=nm,
                                         scale=s, sign=sign,
                                         rms=float(d.pow(2).mean().sqrt()),
                                         peak=float(d.abs().max()),
                                         alpha=round(alpha, 5), psr=round(psr, 5),
                                         nsr=round(nsr, 5), dircos=round(cosv, 5), **PROV))
                got = [r for r in rows if r["part"].startswith("B") and r["speaker"] == spk
                       and r["direction"] == nm and r["sign"] == 1]
                print(f"  spk{spk} {nm:42s} " + "  ".join(
                    f"s={r['scale']:.3f}:{r['alpha']:+.3f}" for r in got), flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    # ------------------------------------------------------------ summary
    df = pd.DataFrame(rows)
    for part, g in df.groupby("part"):
        print("\n" + "=" * 96 + f"\n{part}: mean alpha over {g.speaker.nunique()} speakers, "
              f"by scale (sign +1)\n" + "=" * 96)
        piv = g[g.sign == 1].pivot_table(index="direction", columns="scale", values="alpha")
        print(piv.round(3).to_string())
        ps = g[g.sign == 1].pivot_table(index="direction", columns="scale", values="peak")
        print("\npeak |d| (local amplitude) at each scale:")
        print(ps.round(4).to_string())
        s0 = 0.2 if part.startswith("A") else 0.05
        both = g[(g.scale - s0).abs() < 1e-9].pivot_table(index=["direction", "speaker"],
                                                          columns="sign", values="alpha")
        if -1 in both and 1 in both:
            asym = (both[1] - both[-1]).groupby(level=0).agg(["mean", "max"])
            print(f"\nsign asymmetry alpha(+d) - alpha(-d) at the protocol scale {s0}:")
            print(asym.round(4).to_string())
    print(f"\n[rms] saved {a.out}")


if __name__ == "__main__":
    main()
