"""
experiments/audit_frontend_alpha.py
===================================
Audit 2, risk 2 (RESULTS section 58): is alpha measuring the NETWORK or the
FRONT END?

THE QUESTION
------------
Control B (section 54) found an untrained CAM++ as purifier-hostile as a
trained one and read that as "the effect does not need learned identity". An
untrained network still has a network. The sharper control is NO network:
the gradient direction of a fixed random linear projection of the same
front-end features, through the same measurement path. If that direction also
lands inside the trained band, alpha is a property of the front end (Kaldi
fbank + the mel->wav inverse V the mel path differentiates through), and both
the lead claim and Control B change meaning.

WHAT IS MEASURED
----------------
Exactly `exp_encoder_subspace.py`'s protocol -- purifier v7 DDIM, t = 50 and
200, 29 cross-session test-clean speakers, --rms 0.2, seeds 2026,7, the same
`survival()` -- on these directions per speaker:

  frontend:fbank-linear   normalize(W . vec(fbank_cmn(x))), W fixed Gaussian
                          [512 x 398*80], seed 2026, the same W for every
                          speaker. Same front end as CAM++, no network.
  frontend:wav-linear     normalize(W . x) on the raw waveform, W fixed
                          Gaussian [512 x 64000], seed 2027. No front end at
                          all except V: isolates the mel->wav inverse the
                          measurement itself differentiates through.
  CAM++                   trained, measured IN THIS RUN as the positive
                          reference (section 54 has it at 0.631).
  CAM++ (untrained, seed 0)   Control B's direction, in this run (0.638).
  band_matched:<each>, random, pca_span   the council-#3 controls, built by
                          `controls_for` exactly as every alpha section does.

GATE. The two front-end directions are new measurement paths. Every gradient
direction here (both front-end ones, CAM++, untrained CAM++) is run through
`check_direction_sanity.check_direction` on EVERY speaker before its alpha is
measured, and the result is stored on the row (`gate_passed`; a band-matched
control inherits its target's gate). `audit_direction_stability.py` found,
before this script ran, that the Kaldi-fbank front end fails the gate on
clips containing digital silence (its log floor is a knife-edge there), and
`audit_digital_silence.csv` says exactly two cohort speakers have it: 121 and
260. So the PREDICTION, written before the run, is: fbank-linear, CAM++ and
untrained CAM++ fail the gate on 121 and 260 and pass elsewhere; wav-linear
passes everywhere. The PRIMARY endpoint is read on gate-passing speakers (no
verdict is read from a cell that failed its gate); the same statistic on all
29 is reported beside it as the sensitivity, and if the two give different
branches that is said, not resolved.

READING, FIXED HERE BEFORE THE RUN
----------------------------------
Primary endpoint, as in every alpha section: mean paired per-speaker
alpha(frontend:fbank-linear) - alpha(its band-matched control) at t = 50,
n = 29, BCa 95 % CI, Wilcoxon signed-rank p, Holm over the four cells read
(two front-end directions x two timesteps). Reference points are section 54's
on the same cohort and purifier: trained 0.388-0.631, random 0.714, and the
same 0.63 / 0.70 cut points PREREG_council5_controls B.3 fixed for Control B,
reused unchanged because the question is the same one asked of a weaker object.

  F-1  alpha(fbank-linear, t=50) <= 0.63, and significantly below its own
       band-matched control
       -> the FRONT END alone reproduces the C1 contrast. "The gradient
          direction of a deep speech network ... survives worse" (54d) is
          then true of a direction with no network in it; the honest
          statement becomes "directions that pass through a speech front
          end's Jacobian survive worse than noise with the same spectrum".
          Section 54's disjunction "identity or architecture" gains a third,
          dominant term: the front end. C1's per-encoder ORDERING may still
          be reported, but its interpretation as a property of the network
          is gone.
  F-2  alpha(fbank-linear, t=50) >= 0.70, i.e. not distinguishable from its
       control / from random
       -> the network matters. The front end alone is not purifier-hostile,
          section 54's reading ("trained or untrained network") stands as
          written, and the lead claim keeps its wording.
  F-3  between
       -> report the number and the share of the trained-vs-random gap
          (0.714 - 0.388..0.631) that the front end explains; no mechanism
          language.

  Secondary: if wav-linear ALSO lands <= 0.63, the effect sits in V, the
  measurement's own reconstruction Jacobian, and alpha is measuring the
  instrument. That is reported as such, and it invalidates the interpretation
  of every mel-path alpha section (32, 37, 43, 44, 46b, 54), not their
  arithmetic.

Run (GPU, ~45 min):
    python experiments/audit_frontend_alpha.py
    python experiments/audit_frontend_alpha.py --analyse-only
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
from experiments.check_direction_sanity import check_direction
from utils.provenance import provenance

R = os.path.join(PROJ, "experiments", "results")
OUT = os.path.join(R, "audit_frontend_alpha.csv")
B1_MAX, B2_MIN = 0.63, 0.70            # PREREG_council5_controls B.3, reused


def analyse(src):
    from experiments.analyse_matched_control import bca_ci, holm, wilcoxon_p
    d = pd.read_csv(src)
    if "gate_passed" in d and not bool(d.gate_passed.all()):
        nbad = int((~d.gate_passed.astype(bool) & ~d.is_control).sum())
        print(f"[frontend] {nbad} direction cell(s) failed the gate; the PRIMARY set "
              f"below excludes them, 'all' keeps them as the sensitivity.")
    n = d.speaker.nunique()
    print("=" * 100)
    print(f"FRONT-END CONTROL -- is alpha the network or the front end?   n = {n}")
    print("=" * 100)
    targets = ["frontend:fbank-linear", "frontend:wav-linear", "CAM++",
               "CAM++ (untrained, seed 0)"]
    gate = d[~d.is_control & d.direction.isin(targets)].pivot_table(
        index="speaker", columns="direction", values="gate_passed", aggfunc="first")
    print("  gate failures (speaker: directions):")
    for spk, row in gate.iterrows():
        bad = [c for c in gate.columns if not bool(row[c])]
        if bad:
            print(f"    {spk}: {bad}")
    cells, summary = [], {}
    for col, tlab in (("alpha_t50", "t = 50"), ("alpha_t200", "t = 200")):
        piv = d.pivot_table(index="speaker", columns="direction", values=col)
        rnd, pca = float(piv["random"].mean()), float(piv["pca_span"].mean())
        print(f"\n  {tlab}    random {rnd:.3f}    pca_span {pca:.3f}    "
              f"(PRIMARY = gate-passing speakers; all = all {n}, the sensitivity)")
        print(f"  {'direction':28s} {'set':>7s} {'n':>3s} {'alpha':>7s} {'vs own ctl':>11s} "
              f"{'95% CI':>20s} {'p':>9s} {'below':>6s} {'vs random':>10s}")
        for t in targets:
            if t not in piv:
                continue
            for setname in ("primary", "all"):
                keep = piv.index if setname == "all" else gate.index[gate[t].astype(bool)]
                sub = piv.loc[piv.index.isin(keep)]
                dd = (sub[t] - sub[f"band_matched:{t}"]).dropna().to_numpy()
                lo, hi, _, _ = bca_ci(dd, alpha=0.05, n_boot=4000)
                p = wilcoxon_p(dd)
                vr = float((sub[t] - sub["random"]).mean())
                print(f"  {t:28s} {setname:>7s} {len(dd):3d} {sub[t].mean():7.3f} {dd.mean():+11.3f} "
                      f"[{lo:+8.3f},{hi:+8.3f}] {p:9.2e} {int((dd < 0).sum()):3d}/{len(dd):<2d} "
                      f"{vr:+10.3f}")
                summary[(t, tlab, setname)] = (float(sub[t].mean()), p, len(dd))
                if t.startswith("frontend") and setname == "primary":
                    cells.append(dict(timestep=tlab, direction=t, n=len(dd),
                                      alpha=float(sub[t].mean()),
                                      delta_own_control=float(dd.mean()), lo=lo, hi=hi, p=p,
                                      below=int((dd < 0).sum()), vs_random=vr))
    c = pd.DataFrame(cells)
    c["p_holm"] = holm(c.p.to_numpy())
    c.to_csv(os.path.join(R, "audit_frontend_cells.csv"), index=False)
    print("\n  Holm over the four front-end cells (primary set):")
    print(c[["timestep", "direction", "n", "alpha", "delta_own_control", "p", "p_holm"]]
          .round(4).to_string(index=False))

    af = summary[("frontend:fbank-linear", "t = 50", "primary")][0]
    af_all = summary[("frontend:fbank-linear", "t = 50", "all")][0]
    aw = summary.get(("frontend:wav-linear", "t = 50", "primary"), (float("nan"),))[0]
    sig = bool(c[(c.timestep == "t = 50") & (c.direction == "frontend:fbank-linear")].p_holm.iloc[0] < 0.05)
    print("\n" + "=" * 100)
    print("READING -- fixed in this file's docstring before the run")
    print("=" * 100)
    print(f"  alpha(frontend:fbank-linear, t=50) = {af:.4f} on the primary set, "
          f"{af_all:.4f} on all {n}   (own control significant below, Holm: {sig})")
    print(f"  alpha(frontend:wav-linear,   t=50) = {aw:.4f}")
    print(f"  cut points reused from B.3: F-1 if <= {B1_MAX}, F-2 if >= {B2_MIN}, else F-3")
    if af <= B1_MAX and sig:
        print("  *** F-1: the FRONT END alone reproduces the C1 contrast. ***")
    elif af >= B2_MIN:
        print("  *** F-2: the front end alone is NOT purifier-hostile; the network matters. ***")
    else:
        print("  *** F-3: intermediate. Report the number; no mechanism language. ***")
    if aw <= B1_MAX:
        print("  *** SECONDARY: wav-linear (V alone) is ALSO inside the trained band: "
              "alpha is measuring the measurement's own reconstruction Jacobian. ***")
    print(f"\nwritten: {os.path.join(R, 'audit_frontend_cells.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--t", default="50,200")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--rms", type=float, default=0.2)
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v7.pt"),
                    dest="ckpt")
    ap.add_argument("--sampler", default="ddim")
    ap.add_argument("--analyse-only", action="store_true", dest="analyse_only")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.analyse_only:
        analyse(a.out)
        return
    if os.path.isfile(a.out):
        raise SystemExit(f"[frontend] REFUSING: {a.out} exists. Move it aside for a clean run.")
    PROV = provenance()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in a.t.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]

    from models.real_diffpure import load_real_diffpure, MelDiffPure
    from experiments.exp_nullspace_survival import survival
    from experiments.librispeech import (cross_session_split, list_librispeech,
                                         load_flac, reference_items)
    from experiments.extra_encoders import CAMPPlusEnc, UntrainedCAMPPlusEnc, fbank
    from experiments.audit_direction_stability import (FbankLinearEnc, MelPath,
                                                       WavLinearEnc)

    dp0 = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=ts[0], device=device,
                             sampler=a.sampler)
    den = dp0.denoiser
    for p in den.parameters():
        p.requires_grad_(False)
    dps = {t: MelDiffPure(den, n_steps=dp0.n_steps, t_diffuse=t,
                          sampler=a.sampler).to(device).eval() for t in ts}
    basis, pca_info = clean_mel_basis(k=DEFAULT_K, subset="dev-clean", device=device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"), "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][: a.n_files]
    print(f"[frontend] code_sha={PROV['code_sha']} {len(spks)} speakers, purifier "
          f"{os.path.basename(a.ckpt)} ({a.sampler}) t={ts} rms={a.rms} seeds={seeds}")

    mp = MelPath(device)
    wav0 = load_flac(ref[spks[0]]["path"], 4.0).to(device)
    n_frames = int(fbank(wav0).shape[0])
    encs = [FbankLinearEnc(device, n_frames), WavLinearEnc(device, wav0.shape[-1]),
            CAMPPlusEnc(device), UntrainedCAMPPlusEnc(device, seed=0)]
    print(f"[frontend] directions: {[e.name for e in encs]}  (fbank frames {n_frames})")

    rows, gate_log = [], []
    for i, spk in enumerate(spks, 1):
        wav = load_flac(ref[spk]["path"], 4.0).to(device)
        mel_o, _ = mp.setup(wav)
        # ---- the gate, per speaker and per gradient direction, BEFORE alpha
        gate = {}
        for e in encs:
            ok_, rep = check_direction(lambda j: mp.direction(e, wav, jit_in=j, seed=1),
                                       name=f"spk{spk} mel {e.name}")
            gate[e.name] = bool(ok_)
            gate_log.append(rep)
        failed = [k for k, v in gate.items() if not v]
        if failed:
            print(f"  spk{spk} GATE FAILED for {failed} -- rows stamped, no verdict from them")
        dirs = {}
        for e in encs:
            g, c, _ = mp.direction(e, wav)
            dirs[e.name] = (g / (g.pow(2).mean().sqrt() + 1e-12) * a.rms).detach()
        gen = torch.Generator(device=device).manual_seed(spk)
        rn = torch.randn(mel_o.shape, generator=gen, device=device)
        dirs["random"] = (rn / (rn.pow(2).mean().sqrt() + 1e-12) * a.rms).detach()
        ctl = {}
        for e in encs:
            ctl[f"band_matched:{e.name}"] = controls_for(dirs[e.name], None, seed=spk)["band_matched"]
        ctl["pca_span"] = controls_for(dirs[encs[0].name], basis, seed=spk)["pca_span"]
        dirs.update(ctl)
        for name, d in dirs.items():
            tgt = name.split(":", 1)[1] if name.startswith("band_matched:") else name
            r = dict(speaker=spk, direction=name, rms=round(float(d.pow(2).mean().sqrt()), 6),
                     is_control=name.startswith(("band_matched", "pca_span")),
                     controls_for=(tgt if name.startswith("band_matched:") else ""),
                     gate_passed=gate.get(tgt, True))
            for t in ts:
                psr, cosv, alpha, nsr = survival(dps[t], mel_o, d, seeds, full=True)
                r[f"alpha_t{t}"] = round(alpha, 4)
                r[f"nsr_t{t}"] = round(nsr, 4)
                r[f"psr_t{t}"] = round(psr, 4)
                r[f"dircos_t{t}"] = round(cosv, 4)
            r.update(PROV)
            rows.append(r)
        got = {x["direction"]: x[f"alpha_t{ts[0]}"] for x in rows if x["speaker"] == spk}
        print(f"  [{i}/{len(spks)}] spk{spk}  " + "  ".join(
            f"{k.split(':')[-1][:10]}={v:+.3f}" for k, v in got.items()
            if not k.startswith("band_matched")), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
        with open(os.path.splitext(a.out)[0] + "_gate.txt", "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(gate_log))
    print(f"[frontend] saved {a.out}\n")
    analyse(a.out)


if __name__ == "__main__":
    main()
