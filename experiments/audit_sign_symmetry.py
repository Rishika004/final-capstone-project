"""
experiments/audit_sign_symmetry.py
==================================
Audit 2, risk 3 (RESULTS section 58e): is alpha a property of a DIRECTION or
of a SIGNED VECTOR?

C1 is phrased as a subspace claim -- encoder identity directions "live outside
the purifier's invariant subspace" -- and a subspace does not have a sign.
Every alpha section measures one sign: d = -grad <E(V(m)), E(x0)>, the
steepest DESCENT of similarity to self (section 32's convention). On three
speakers `audit_rms_sensitivity.py` found alpha(+d) and alpha(-d) equal to
within 0.02 for every control and for random, and different by 0.04-0.16 for
the gradient directions, with alpha(-d) landing at the noise level. This
measures that on the whole cohort, at the primary-endpoint cell.

WHAT IS MEASURED
----------------
purifier v7 DDIM, t = 50, the 29 cross-session test-clean speakers, --rms 0.2,
seeds 2026,7, `survival()`, `controls_for` -- the protocol of every alpha
section. For each of the eight gradient directions (WavLM-SV, ECAPA-TDNN,
d-vector, CAM++, ERes2Net, CAM++ untrained seed 0, frontend:fbank-linear,
frontend:wav-linear): alpha(+d), alpha(-d), and its band-matched control
(sign-symmetric to 0.02, so measured once). Plus random and pca_span. Every
gradient direction gated per speaker first.

READING, fixed here before the run. Primary endpoint: for each direction,
mean paired per-speaker alpha(-d) - alpha(band_matched control), n = gate-
passing speakers, BCa 95 % CI, Wilcoxon, Holm over the eight directions; and
the paired asymmetry alpha(+d) - alpha(-d) with the same machinery.
  * asymmetry within +-0.02 and alpha(-d) still significantly below its control
        -> alpha is a property of the direction; C1's subspace wording stands.
  * asymmetry beyond 0.02 and alpha(-d) NOT below its control
        -> alpha is a property of the signed vector; the C1 contrast is the
           purifier removing a particular signed, concentrated perturbation,
           and "subspace" must leave the claim. The measurement stands as a
           measurement of +d.
  * mixed by encoder -> report per encoder; no mechanism language.

Run (GPU, ~1 h):
    python experiments/audit_sign_symmetry.py
    python experiments/audit_sign_symmetry.py --analyse-only
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

from core.matched_random import clean_mel_basis, controls_for, DEFAULT_K
from experiments.check_direction_sanity import check_direction, scale_to_rms
from utils.provenance import provenance

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(R, "audit_sign_symmetry.csv")


def analyse(src):
    from experiments.analyse_matched_control import bca_ci, holm, wilcoxon_p
    d = pd.read_csv(src)
    n = d.speaker.nunique()
    targets = [x for x in d.direction.unique()
               if not str(x).startswith(("band_matched", "pca_span", "random"))]
    gate = d[d.direction.isin(targets) & (d.sign == 1)].pivot_table(
        index="speaker", columns="direction", values="gate_passed", aggfunc="first")
    print("=" * 104)
    print(f"SIGN SYMMETRY -- alpha(+d) vs alpha(-d), purifier v7 DDIM t=50, n = {n}")
    print("=" * 104)
    for spk, row in gate.iterrows():
        bad = [c for c in gate.columns if not bool(row[c])]
        if bad:
            print(f"  gate failed  spk{spk}: {bad}")
    piv = d.pivot_table(index="speaker", columns=["direction", "sign"], values="alpha_t50")
    rnd = float(piv[("random", 1)].mean())
    print(f"\n  random {rnd:.3f}   pca_span {float(piv[('pca_span', 1)].mean()):.3f}"
          f"   (PRIMARY = gate-passing speakers)")
    print(f"  {'direction':26s} {'n':>3s} {'a(+d)':>6s} {'a(-d)':>6s} {'ctl':>6s} "
          f"{'+d - ctl':>9s} {'p':>8s} {'-d - ctl':>9s} {'95% CI':>18s} {'p':>8s} {'below':>6s} "
          f"{'asym +d - -d':>13s} {'95% CI':>18s} {'p':>8s}")
    rows = []
    for t in targets:
        keep = gate.index[gate[t].astype(bool)] if t in gate else piv.index
        sub = piv.loc[piv.index.isin(keep)]
        ap, am = sub[(t, 1)], sub[(t, -1)]
        ct = sub[(f"band_matched:{t}", 1)]
        dp = (ap - ct).dropna().to_numpy()
        dm = (am - ct).dropna().to_numpy()
        da = (ap - am).dropna().to_numpy()
        lo_m, hi_m, _, _ = bca_ci(dm, alpha=0.05, n_boot=4000)
        lo_a, hi_a, _, _ = bca_ci(da, alpha=0.05, n_boot=4000)
        p_p, p_m, p_a = wilcoxon_p(dp), wilcoxon_p(dm), wilcoxon_p(da)
        print(f"  {t[:26]:26s} {len(dm):3d} {ap.mean():6.3f} {am.mean():6.3f} {ct.mean():6.3f} "
              f"{dp.mean():+9.3f} {p_p:8.1e} {dm.mean():+9.3f} [{lo_m:+.3f},{hi_m:+.3f}] {p_m:8.1e} "
              f"{int((dm < 0).sum()):3d}/{len(dm):<2d} {da.mean():+13.3f} [{lo_a:+.3f},{hi_a:+.3f}] {p_a:8.1e}")
        rows.append(dict(direction=t, n=len(dm), alpha_plus=float(ap.mean()),
                         alpha_minus=float(am.mean()), control=float(ct.mean()),
                         plus_minus_ctl=float(dp.mean()), p_plus=p_p,
                         minus_minus_ctl=float(dm.mean()), lo=lo_m, hi=hi_m, p_minus=p_m,
                         below=int((dm < 0).sum()), asym=float(da.mean()),
                         asym_lo=lo_a, asym_hi=hi_a, p_asym=p_a))
    c = pd.DataFrame(rows)
    c["p_minus_holm"] = holm(c.p_minus.to_numpy())
    c["p_asym_holm"] = holm(c.p_asym.to_numpy())
    c.to_csv(os.path.join(R, "audit_sign_symmetry_cells.csv"), index=False)
    print("\n  Holm over the eight directions:")
    print(c[["direction", "n", "alpha_plus", "alpha_minus", "control", "minus_minus_ctl",
             "p_minus_holm", "asym", "p_asym_holm"]].round(4).to_string(index=False))
    print("\n  READING (fixed in the docstring): per direction, 'signed vector' if |asym| > 0.02 "
          "and alpha(-d) is not significantly below its control (Holm); 'direction' if "
          "|asym| <= 0.02 and alpha(-d) is below its control.")
    for r in c.itertuples():
        if abs(r.asym) > 0.02 and r.p_minus_holm >= 0.05:
            v = "SIGNED VECTOR: alpha(-d) is not below its control"
        elif abs(r.asym) <= 0.02 and r.p_minus_holm < 0.05:
            v = "DIRECTION: sign-symmetric and below its control either way"
        elif abs(r.asym) > 0.02 and r.p_minus_holm < 0.05:
            v = "MIXED: asymmetric, but alpha(-d) is still below its control"
        else:
            v = "MIXED: symmetric, but not below its control either way"
        print(f"    {r.direction[:26]:26s} {v}")
    print(f"\nwritten: {os.path.join(R, 'audit_sign_symmetry_cells.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--t", type=int, default=50)
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v7.pt"),
                    dest="ckpt")
    ap.add_argument("--analyse-only", action="store_true", dest="analyse_only")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.analyse_only:
        analyse(a.out)
        return
    if os.path.isfile(a.out):
        raise SystemExit(f"[sign] REFUSING: {a.out} exists. Move it aside for a clean run.")
    PROV = provenance()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(v) for v in a.seeds.split(",")]

    from models.real_diffpure import load_real_diffpure, MelDiffPure
    from experiments.exp_nullspace_survival import survival
    from experiments.exp_purifier_aware import build_encoders
    from experiments.extra_encoders import build_extra, UntrainedCAMPPlusEnc, fbank
    from experiments.librispeech import (cross_session_split, list_librispeech,
                                         load_flac, reference_items)
    from experiments.audit_direction_stability import (FbankLinearEnc, MelPath,
                                                       WavLinearEnc)

    dp0 = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=a.t, device=device, sampler="ddim")
    for p in dp0.denoiser.parameters():
        p.requires_grad_(False)
    dp = MelDiffPure(dp0.denoiser, n_steps=dp0.n_steps, t_diffuse=a.t, sampler="ddim").to(device).eval()
    basis, _ = clean_mel_basis(k=DEFAULT_K, subset="dev-clean", device=device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"), "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][: a.n_files]
    mp = MelPath(device)
    wav0 = load_flac(ref[spks[0]]["path"], 4.0).to(device)
    encs = (list(build_encoders(device, with_xvector=False)) + list(build_extra(device))
            + [UntrainedCAMPPlusEnc(device, seed=0),
               FbankLinearEnc(device, int(fbank(wav0).shape[0])),
               WavLinearEnc(device, wav0.shape[-1])])
    print(f"[sign] code_sha={PROV['code_sha']} {len(spks)} speakers t={a.t} rms={a.rms} "
          f"seeds={seeds} directions {[e.name for e in encs]}")

    rows, gate_log = [], []
    for i, spk in enumerate(spks, 1):
        wav = load_flac(ref[spk]["path"], 4.0).to(device)
        mel_o, _ = mp.setup(wav)
        gate, dirs = {}, {}
        for e in encs:
            ok_, rep = check_direction(lambda j, e=e: mp.direction(e, wav, jit_in=j, seed=1),
                                       name=f"spk{spk} mel {e.name}")
            gate[e.name] = bool(ok_)
            gate_log.append(rep)
            g, _, _ = mp.direction(e, wav)
            dirs[e.name] = scale_to_rms(g, a.rms).detach()
        failed = [k for k, v in gate.items() if not v]
        if failed:
            print(f"  spk{spk} GATE FAILED for {failed}")
        gen = torch.Generator(device=device).manual_seed(spk)
        dirs["random"] = scale_to_rms(torch.randn(mel_o.shape, generator=gen, device=device), a.rms)
        ctl = {f"band_matched:{e.name}": controls_for(dirs[e.name], None, seed=spk)["band_matched"]
               for e in encs}
        ctl["pca_span"] = controls_for(dirs[encs[0].name], basis, seed=spk)["pca_span"]
        dirs.update(ctl)
        for name, d in dirs.items():
            tgt = name.split(":", 1)[1] if name.startswith("band_matched:") else name
            signs = (1, -1) if name in gate else (1,)
            for sign in signs:
                psr, cosv, alpha, nsr = survival(dp, mel_o, sign * d, seeds, full=True)
                rows.append(dict(speaker=spk, direction=name, sign=sign,
                                 rms=round(float(d.pow(2).mean().sqrt()), 6),
                                 is_control=name.startswith(("band_matched", "pca_span")),
                                 gate_passed=gate.get(tgt, ""),
                                 **{f"alpha_t{a.t}": round(alpha, 4), f"nsr_t{a.t}": round(nsr, 4),
                                    f"psr_t{a.t}": round(psr, 4), f"dircos_t{a.t}": round(cosv, 4)},
                                 **PROV))
        got = {(r["direction"], r["sign"]): r[f"alpha_t{a.t}"] for r in rows if r["speaker"] == spk}
        print(f"  [{i}/{len(spks)}] spk{spk}  " + "  ".join(
            f"{e.name.split(':')[-1][:8]}:{got[(e.name, 1)]:+.2f}/{got[(e.name, -1)]:+.2f}"
            for e in encs) + f"  rnd {got[('random', 1)]:+.2f}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + "_gate.txt", "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(gate_log))
    print(f"[sign] saved {a.out}\n")
    analyse(a.out)


if __name__ == "__main__":
    main()
