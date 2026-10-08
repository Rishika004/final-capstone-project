"""
experiments/exp_encoder_subspace.py
===================================
A2 — WHY does the purification-invariant constraint help WavLM-SV and ECAPA but
REVERSE on the d-vector? And which encoder should we add to fix it?

THE QUESTION, AND WHY IT IS NOT ANSWERED BY WHAT WE HAVE
--------------------------------------------------------
`diag_directional_survival.py` measures how much of the CRAFTED perturbation's
encoder-relevant component survives. That is a property of a particular crafted
file. This script measures something upstream and encoder-INTRINSIC: for each
encoder, take the mel-space direction that moves it most, and ask how well THAT
DIRECTION ITSELF survives the purifier.

    g_e  = d/d(mel) cos( E_e(V(mel)), E_e(x) )     (unit norm, scaled to --rms)
    then survival of g_e through the real purification loop:
        alpha = <P(m + g_e) - P(m), g_e> / ||g_e||^2      1.0 = untouched
        nsr   = ||residual|| / ||alpha * g_e||            0.0 = clean survival

alpha(g_e) is a property of the ENCODER and the PURIFIER, not of any crafting
run. It is therefore usable as a SELECTION CRITERION for a new ensemble member
before spending GPU hours crafting with it (item A1).

THE HYPOTHESIS
--------------
Protection must satisfy two things at once: move the encoder, and survive the
purifier. Those are compatible only if the encoder's identity-relevant
directions lie inside the purifier's invariant subspace. If
alpha(g_WavLM) is high and alpha(g_dvector) is low, then for the d-vector the
two requirements CONFLICT -- the constraint steers the perturbation toward
directions that survive, which are precisely the ones the d-vector does not
read -- and the measured reversal (RESULTS 27a: +0.071, +0.075 at t=1, 3/29
speakers improved) is the predicted consequence, not an anomaly.

PRE-REGISTERED READINGS, written before the run
-----------------------------------------------
  * alpha(WavLM), alpha(ECAPA) HIGH and alpha(d-vector) LOW
        -> the hypothesis holds. The paper gains a mechanism for the whole
           encoder pattern, and A1 becomes "add the encoder with the highest
           alpha" instead of "add more encoders".
  * all alphas SIMILAR
        -> the hypothesis FAILS. The d-vector story lies elsewhere (its partial
           averaging, or its threshold geometry -- note its cross-session EER is
           3.5-5 % against ECAPA's 0.6-0.9 %, so its accept region is far
           looser). Report as a negative result and do not pick encoders this
           way.
  * alpha(d-vector) HIGH
        -> the hypothesis is refuted outright and §27's reversal needs a
           different explanation entirely.

THE RETRO-VALIDATION THAT MAKES THIS CREDIBLE
---------------------------------------------
x-vector was tested as a fourth crafting member and REJECTED on measured grounds
(RESULTS 24b: it moved 0.08 against ECAPA's 1.16; separation 0.062). It is
included here with `--with-xvector`. If this criterion is sound it must score
x-vector LOW *without being told that outcome*. A criterion that cannot predict
the one case we already know is not a criterion.

Baselines, so the alphas are interpretable rather than free-floating:
  * random  -- a Gaussian mel direction of identical norm.
  * band_matched:<enc>  -- MATCHED CONTROL, one per encoder (council #3,
    PLAN_V3 W1): white noise carrying that encoder direction's own per-mel-band
    power profile, same norm. Matched on the spectral envelope, random in
    direction. If an encoder direction survives no worse than its own matched
    control, alpha is measuring spectral smoothness, not a per-encoder subspace,
    and C1 is a spectral claim. See core/matched_random.py for the
    pre-registered reading of each branch.
  * pca_span  -- MATCHED CONTROL: every frame inside the span of the top-k
    principal components of CLEAN log-mel frames, built from a subset disjoint
    from the evaluation speakers. As smooth and speech-like as the corpus
    allows, and blind to every encoder.
  * (reference, not recomputed here) the OPTIMISED pass-through direction and
    random, measured on this purifier in survival_tc_v7.csv: alpha 0.867 vs
    0.713 at t=50, 0.525 vs 0.466 at t=200. An encoder direction near the
    optimised value lives in the invariant subspace; near random, it does not.

Run:
  python experiments/exp_encoder_subspace.py --n-files 29 --with-xvector
Output:
  experiments/results/encoder_subspace.csv
"""
import sys
import io
import os
import argparse

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

from core.matched_random import clean_mel_basis, controls_for, DEFAULT_K
from experiments.check_direction_sanity import check_direction, scale_to_rms
from utils.provenance import provenance, add_prereg_arg


class FrozenVADDvec:
    """The d-vector behind a FROZEN voice-activity trim, differentiably.

    PLAN_V3 W2's C1 robustness row. alpha(g_dvec) in §32/§37 is computed
    through `DvecEnc.embed` -- the torch-mel partial average with NO
    `preprocess_wav`. The real evaluation path trims silence first, so a
    reviewer can ask whether the d-vector's low alpha is a property of the
    encoder or of the front end it was measured without.

    `preprocess_wav` is not differentiable because the VAD decision is a hard
    threshold. Freezing that decision on the CLEAN signal makes the rest of it
    differentiable -- the trim becomes a fixed index selection, and §38
    measured the volume-normalisation gain at exactly 0.000 dB on all 29
    speakers of this cohort, so there is nothing else to model. Freezing on the
    clean signal is legitimate here precisely because the direction being
    measured is a property of the clean utterance.
    """

    def __init__(self, inner, keep_idx):
        self.inner = inner
        self.keep = keep_idx
        self.name = inner.name + " (frozen VAD)"
        self.threshold = getattr(inner, "threshold", 0.0)

    def embed(self, wav):
        w = wav if wav.dim() == 2 else wav.unsqueeze(0)
        k = self.keep[self.keep < w.shape[-1]]
        return self.inner.embed(w[:, k])

OUT = os.path.join(PROJ, "experiments", "results", "encoder_subspace.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--t", default="50,200", help="purifier strengths")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--rms", type=float, default=0.2,
                    help="direction norm in log-mel units; matches "
                         "exp_nullspace_survival.py so alphas are comparable")
    ap.add_argument("--purifier-ckpt", default=os.path.join(
        PROJ, "checkpoints", "purifier_v7.pt"), dest="ckpt")
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--with-xvector", action="store_true", dest="with_xvector",
                    help="include x-vector, the retro-validation case (RESULTS 24b)")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="restrict to the CROSS-SESSION speaker set, which is "
                         "what every other script in this project reports. "
                         "Without it this script takes sorted(ref)[:n_files], a "
                         "different cohort -- see the overlap line it prints.")
    ap.add_argument("--pca-k", type=int, default=DEFAULT_K, dest="pca_k",
                    help="components the pca_span control is confined to")
    ap.add_argument("--pca-subset", default="dev-clean", dest="pca_subset",
                    help="corpus for the clean-mel PCA basis; MUST be disjoint "
                         "from --subset or the control sees the eval speakers")
    ap.add_argument("--with-extra", action="store_true", dest="with_extra",
                    help="ALSO measure alpha for CAM++ and ERes2Net, the two "
                         "eval-only encoders of council #13b. They are appended "
                         "to the measurement list and NEVER to build_encoders, "
                         "so no crafted file can change "
                         "(PREREG_13b_prospective_encoders.md section 6.2).")
    ap.add_argument("--dvec-frozen-vad", action="store_true",
                    dest="dvec_frozen_vad",
                    help="ALSO measure alpha for the d-vector's identity "
                         "direction computed through a FROZEN VAD trim, i.e. "
                         "its real evaluation front end made differentiable. "
                         "PLAN_V3 W2's C1 robustness row: is alpha(g_dvec) an "
                         "artefact of scoring it without preprocess_wav?")
    ap.add_argument("--with-untrained", action="store_true",
                    dest="with_untrained",
                    help="CONTROL B (PREREG_council5_controls.md section B): "
                         "ALSO measure alpha for an UNTRAINED CAM++ -- same "
                         "architecture, same front end, no learned weights, so "
                         "its top gradient direction reads no identity. Tests "
                         "whether the word 'speaker' in C1 is doing any work, "
                         "as opposed to 'a denoiser removes directions any deep "
                         "speech network is sensitive to'. Eval-only: appended "
                         "to the measurement list, never to build_encoders.")
    ap.add_argument("--untrained-seeds", default="0,1,2",
                    dest="untrained_seeds",
                    help="init seeds for --with-untrained. A single random "
                         "init is a single draw; the mean over these is the "
                         "pre-declared primary and each is also reported.")
    ap.add_argument("--no-controls", action="store_true", dest="no_controls",
                    help="omit the council #3 matched controls, reproducing the "
                         "2026-10-01 run exactly")
    ap.add_argument("--no-gate", action="store_true", dest="no_gate",
                    help="skip the RESULTS 57f instrument check (check_direction_"
                         "sanity) that is otherwise run on every gradient direction "
                         "of every speaker and recorded as gate_passed on the row. "
                         "RESULTS 58 found the Kaldi-fbank directions (CAM++, "
                         "ERes2Net, untrained CAM++) fail it on clips with digital "
                         "silence; a row with gate_passed=False may not carry a "
                         "verdict.")
    ap.add_argument("--out", default=OUT)
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in a.t.split(",")]
    seeds = [int(v) for v in a.seeds.split(",")]

    from models.real_diffpure import load_real_diffpure, MelDiffPure
    from evaluation.evaluate import (wav_to_logmel, N_FFT, N_MELS, HOP_LENGTH,
                                     SAMPLE_RATE)
    from core.optimizer import _build_mel_to_wav, _mel_to_wav_differentiable
    from experiments.exp_purifier_aware import build_encoders
    from experiments.exp_nullspace_survival import survival
    from experiments.librispeech import list_librispeech, reference_items, load_flac

    dp0 = load_real_diffpure(checkpoint_path=a.ckpt, t_diffuse=ts[0],
                             device=device, sampler=a.sampler)
    den = dp0.denoiser
    for p in den.parameters():
        p.requires_grad_(False)
    dps = {t: MelDiffPure(den, n_steps=dp0.n_steps, t_diffuse=t,
                          sampler=a.sampler).to(device).eval() for t in ts}
    encs = build_encoders(device, with_xvector=a.with_xvector)

    # The matched controls (council #3). Built once, before the speaker loop:
    # the basis is a property of the corpus, not of any evaluation speaker.
    basis, pca_info = (None, {})
    if not a.no_controls:
        if a.pca_subset == a.subset:
            raise SystemExit(
                f"[A2] --pca-subset == --subset ({a.subset}): the clean-PCA "
                "control would be built from the speakers it controls for. "
                "Pass a disjoint subset.")
        basis, pca_info = clean_mel_basis(k=a.pca_k, subset=a.pca_subset,
                                          device=device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)

    # Speaker-cohort check (found 2026-10-05, PLAN_V3 W1). This script took
    # sorted(ref)[:n_files] while every other script takes the CROSS-SESSION
    # filtered set, so the alpha table was measured on a different cohort than
    # RESULTS 27 and 33: 21 of its 29 speakers overlap, 8 do not. alpha is pure
    # geometry -- no enrolment, no thresholds, no second session -- so the
    # measurement is valid on its own terms, but per-speaker pairing with any
    # other section is NOT, and "the same 29 speakers" would be false.
    # The overlap is printed every run so this cannot hide again.
    from experiments.librispeech import cross_session_split
    ok = set(cross_session_split(items, min_sec=4.0))
    if a.cross_session:
        spks = [s for s in sorted(ref) if s in ok][: a.n_files]
    else:
        spks = sorted(ref)[: a.n_files]
    _ov = sum(1 for s in spks if s in ok)
    print(f"[A2] cohort: {len(spks)} speakers, cross_session="
          f"{bool(a.cross_session)}; {_ov}/{len(spks)} are in the cross-session "
          f"set the rest of the paper reports")
    if _ov < len(spks):
        print(f"[A2] WARNING: {len(spks) - _ov} speaker(s) here are NOT in the "
              f"reported cohort. Valid for alpha (pure geometry), but do NOT "
              f"pair these rows per-speaker with RESULTS 27/33. Pass "
              f"--cross-session to match.")
    print(f"[A2] {len(spks)} speakers | purifier {os.path.basename(a.ckpt)} "
          f"({a.sampler}) t={ts} | encoders: {[e.name for e in encs]}")
    print(f"[A2] code_sha={PROV['code_sha']} prereg_sha={PROV['prereg_sha'] or '-'}")
    if basis is None:
        print("[A2] matched controls DISABLED (--no-controls): white noise is the "
              "only baseline, which is council defect #3.")
    else:
        print(f"[A2] matched controls ON: band_matched per encoder + pca_span "
              f"(k={a.pca_k} from {a.pca_subset}, "
              f"{pca_info.get('var_explained', float('nan')):.1%} of clean "
              f"log-mel variance)")
    print("[A2] alpha = how much of the encoder's own identity direction the "
          "purifier passes through (1.0 = untouched).")
    print("[A2] reference on this purifier (survival_tc_v7.csv): optimised "
          "pass-through 0.867 / random 0.713 at t=50.\n")

    rows = []
    for i, spk in enumerate(spks):
        it = ref[spk]
        wav_o = load_flac(it["path"], 4.0).to(device)
        mel_o = wav_to_logmel(wav_o, device).detach()
        inv, phase, win, wav_len, T_frames = _build_mel_to_wav(
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            sample_rate=SAMPLE_RATE, x_wav=wav_o, device=device)

        def unit(v):
            """Scale a mel direction to the protocol's per-element RMS, exactly.

            `scale_to_rms` refuses residue-sized input instead of scaling it up
            (RESULTS 58, risk 3: an additive epsilon cost #13c's d-vector arm
            1.2 % of its RMS because its "gradient" was numerical residue)."""
            return scale_to_rms(v, a.rms)

        def direction_of(e, jitter=0.0, seed=1):
            """The identity direction, at an input jittered by `jitter` x RMS.

            jitter = 0 is the measurement, unchanged from the 2026-10-01 code.
            jitter > 0 is the RESULTS 57f instrument check: everything that
            depends on the input -- mel, phase, the reference embedding -- is
            recomputed from the jittered input, and a direction that does not
            reproduce under a jitter of 1e-5 is not a measurement."""
            if jitter:
                g_ = torch.Generator().manual_seed(seed)
                x_in = wav_o + jitter * wav_o.pow(2).mean().sqrt() * \
                    torch.randn(wav_o.shape, generator=g_).to(device)
                mel_in = wav_to_logmel(x_in, device).detach()
                inv_, phase_, win_, len_, T_ = _build_mel_to_wav(
                    n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
                    sample_rate=SAMPLE_RATE, x_wav=x_in, device=device)
            else:
                x_in, mel_in = wav_o, mel_o
                inv_, phase_, win_, len_, T_ = inv, phase, win, wav_len, T_frames
            with torch.no_grad():
                o = e.embed(x_in).detach()
            m = mel_in.clone().requires_grad_(True)
            w = _mel_to_wav_differentiable(m, inv_, phase_, win_, N_FFT,
                                           HOP_LENGTH, T_, len_)
            w = w / (w.abs().max() + 1e-6)
            c = (e.embed(w) * o).sum()
            g, = torch.autograd.grad(c, m)
            # The identity direction: steepest DESCENT of similarity to self,
            # i.e. the direction that moves this encoder's embedding away fastest.
            return (-g).detach(), float(c), float(g.norm() / mel_in.norm())

        # The frozen-VAD twin is per speaker: the keep-set is a property of
        # this utterance. Built here so it sees the same wav_o the directions
        # are measured on.
        enc_list = list(encs)
        if a.with_extra:
            # Built once, outside the speaker loop in spirit -- cached on the
            # function object so the checkpoints load exactly once.
            if not hasattr(main, "_extra"):
                from experiments.extra_encoders import build_extra
                main._extra = build_extra(device)
                print(f"[A2] #13b eval-only encoders: "
                      f"{[e.name for e in main._extra]}")
            enc_list = enc_list + list(main._extra)
        if a.with_untrained:
            # CONTROL B. Built once and cached, exactly like the #13b pair.
            if not hasattr(main, "_untrained"):
                from experiments.extra_encoders import (build_untrained,
                                                        speaker_separation)
                sds = [int(v) for v in a.untrained_seeds.split(",")]
                main._untrained = build_untrained(device, seeds=sds)
                print(f"[A2] CONTROL B directions: "
                      f"{[e.name for e in main._untrained]}")
                # Verify the premise rather than asserting it: an untrained
                # net that DID carry identity would void the whole control,
                # and nothing else in this run would reveal it.
                for e in main._untrained:
                    sep, s_same, s_diff = speaker_separation(e, device)
                    print(f"[A2]   {e.name}: speaker separation "
                          f"{sep:+.4f}  (same {s_same:+.4f}, "
                          f"diff {s_diff:+.4f})")
                    if sep > 0.20:
                        print("[A2]   WARNING: this untrained network carries "
                              "non-trivial identity; Control B's premise is "
                              "violated and B.4's disjunction applies.")
            enc_list = enc_list + list(main._untrained)
        if a.dvec_frozen_vad:
            from experiments.diag_dvec_proxy import vad_keep_mask
            from resemblyzer.audio import normalize_volume
            from resemblyzer.hparams import audio_norm_target_dBFS
            from experiments.multi_encoders import DvecEnc as _Dvec
            arr = wav_o.detach().squeeze().cpu().numpy().astype(np.float32)
            _, mask = vad_keep_mask(normalize_volume(
                arr, audio_norm_target_dBFS, increase_only=True))
            keep = torch.from_numpy(np.flatnonzero(mask)).long().to(device)
            for e in encs:
                if isinstance(e, _Dvec):
                    enc_list.append(FrozenVADDvec(e, keep))
                    break

        dirs, gate = {}, {}
        for e in enc_list:
            g, _, _ = direction_of(e)
            dirs[e.name] = unit(g)
            if not a.no_gate:
                ok_, rep = check_direction(lambda j, e=e: direction_of(e, j),
                                           name=f"spk{spk} {e.name}")
                gate[e.name] = bool(ok_)
                if not ok_:
                    print(rep)
        gen = torch.Generator(device=device).manual_seed(spk)
        dirs["random"] = unit(torch.randn(mel_o.shape, generator=gen,
                                          device=device)).detach()

        # One band-matched control per encoder direction (its own envelope),
        # and a single pca_span control -- every direction here is scaled to the
        # same per-element RMS, so one norm-matched span draw serves them all.
        controls = {}
        if basis is not None:
            for e in enc_list:
                c = controls_for(dirs[e.name], None, seed=spk)
                controls[f"band_matched:{e.name}"] = c["band_matched"]
            controls["pca_span"] = controls_for(
                dirs[encs[0].name], basis, seed=spk)["pca_span"]
        dirs.update(controls)

        for name, d in dirs.items():
            tgt = name.split(":", 1)[1] if name.startswith("band_matched:") else name
            r = dict(speaker=spk, direction=name,
                     rms=round(float(d.pow(2).mean().sqrt()), 4),
                     is_control=name.startswith(("band_matched", "pca_span")),
                     controls_for=(tgt if name.startswith("band_matched:") else ""),
                     # a control inherits its target's gate; random / pca_span
                     # are not gradients and have no gate ("")
                     gate_passed=("" if a.no_gate else gate.get(tgt, "")),
                     pca_k=(a.pca_k if name == "pca_span" else ""),
                     pca_subset=(a.pca_subset if name == "pca_span" else ""),
                     pca_var=(pca_info.get("var_explained", "")
                              if name == "pca_span" else ""))
            for t in ts:
                psr, cosv, alpha, nsr = survival(dps[t], mel_o, d, seeds,
                                                 full=True)
                r[f"alpha_t{t}"] = round(alpha, 4)
                r[f"nsr_t{t}"] = round(nsr, 4)
                r[f"psr_t{t}"] = round(psr, 4)
                r[f"dircos_t{t}"] = round(cosv, 4)
            r.update(PROV)
            rows.append(r)
        got = {x["direction"]: x[f"alpha_t{ts[0]}"] for x in rows
               if x["speaker"] == spk}
        print(f"  [{i + 1}/{len(spks)}] spk{spk}")
        print("        encoder " + "  ".join(
            f"{e.name.split()[0][:9]}={got[e.name]:+.3f}" for e in enc_list)
            + f"  random={got['random']:+.3f}")
        if basis is not None:
            print("        control " + "  ".join(
                f"{e.name.split()[0][:9]}={got[f'band_matched:{e.name}']:+.3f}"
                for e in enc_list) + f"  pca_span={got['pca_span']:+.3f}")
        sys.stdout.flush()
        pd.DataFrame(rows).to_csv(a.out, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("MEAN alpha / nsr per direction  (higher alpha = survives purification)")
    print("=" * 74)
    cols = [c for c in df.columns if c.startswith(("alpha", "nsr"))]
    print(df.groupby("direction")[cols].mean().round(3).to_string())

    if basis is not None:
        # The council #3 gate, paired per speaker. Printed here so the verdict
        # is in the log beside the numbers it is read from; the CI and the
        # Wilcoxon p are computed in analyse_matched_control.py.
        a_t = f"alpha_t{ts[0]}"
        piv = df.pivot_table(index="speaker", columns="direction", values=a_t)
        print("\n" + "=" * 74)
        print(f"COUNCIL #3 GATE -- paired alpha({a_t}) vs the MATCHED control, "
              f"n={len(piv)}")
        print("=" * 74)
        print(f"  {'encoder':22s} {'alpha':>7s} {'matched':>8s} {'delta':>8s} "
              f"{'below':>7s}  {'vs pca_span':>11s}")
        for e in enc_list:
            bm = f"band_matched:{e.name}"
            if e.name not in piv or bm not in piv:
                continue
            d = piv[e.name] - piv[bm]
            dp = piv[e.name] - piv["pca_span"] if "pca_span" in piv else None
            print(f"  {e.name[:22]:22s} {piv[e.name].mean():+7.3f} "
                  f"{piv[bm].mean():+8.3f} {d.mean():+8.3f} "
                  f"{int((d < 0).sum()):3d}/{len(d):<3d} "
                  f"{(dp.mean() if dp is not None else float('nan')):+11.3f}")
        print("\n  delta < 0 = the encoder direction survives WORSE than a random")
        print("  direction with its own spectral envelope, which is what C1 claims.")
        print("  Read against the pre-registered branches in core/matched_random.py;")
        print("  the CI and Wilcoxon p decide, not these means.")
    if not a.no_gate and "gate_passed" in df:
        bad = df[(df.gate_passed.astype(str) == "False") & (~df.is_control)]
        print(f"\n[A2] INSTRUMENT GATE (RESULTS 57f/58): {len(bad)} direction cell(s) "
              f"failed check_direction_sanity" + (":" if len(bad) else "."))
        for spk_, grp in bad.groupby("speaker"):
            print(f"       spk{spk_}: {sorted(grp.direction)}")
        if len(bad):
            print("       No verdict may be read from those cells; report them beside any "
                  "mean that includes them.")
    print(f"\n[A2] saved {a.out}")
    print("[A2] Read against the pre-registered criteria in this file's docstring.")


if __name__ == "__main__":
    main()
