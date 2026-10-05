"""
experiments/eval_frontend.py
============================
Council #10 — the front-end confound, settled by evaluation alone.

THE OBJECTION
-------------
R1: "WavLM/ECAPA are evaluated through the exact crafting function; only the
d-vector has a different eval path (VAD + partials) — and it's the one that
fails."

That is a real asymmetry. `real_embed_wavlm_sv` and `real_embed_ecapa` take the
published waveform straight, which is byte-for-byte the function the optimiser
differentiated through. `real_embed_dvec` first runs Resemblyzer's
`preprocess_wav` (volume normalise + webrtcvad silence trim) and then averages
1.6 s partials. So the one encoder whose protection collapses under
purification is also the one scored through a path the optimiser never saw, and
on that evidence nobody can tell which fact causes which.

Crucially, the objection survives §23a and §38. Those measured how far the
d-vector's own embedding moves under its front end (cos 0.977 / 0.985) — a
statement about the d-vector. #10 asks the complementary question, about the
OTHER TWO: if WavLM and ECAPA were scored through the same awkward front end,
would their protection collapse too?

THE DESIGN — one variable, both directions
------------------------------------------
Evaluation only. Nothing is re-crafted; the base does not move.

Every encoder is scored twice, under its own native front end and under the
other kind, so the comparison runs in both directions rather than only handing
the d-vector's handicap to its neighbours:

    native   WavLM-SV / ECAPA : the raw published waveform (status quo)
             d-vector         : preprocess_wav + partial average (status quo)

    swapped  WavLM-SV / ECAPA : preprocess_wav applied first, so they inherit
                                the VAD trim and the volume normalisation
             d-vector         : the torch-mel partial average with NO
                                preprocess_wav — i.e. scored the way its two
                                neighbours are

    rs+2 / rs-2  every encoder, native path, after a +-2 % resample. A benign
             channel perturbation no encoder was crafted against, and the one a
             real upload pipeline would apply. It tests robustness rather than
             the asymmetry, and it is the cheaper half of what 2.4 would have
             cost.

Both the clean original and the protected file go through the SAME front end on
each row, because that is what a verifier does.

PRE-REGISTERED READING, written before the run
----------------------------------------------
Primary endpoint: the paired per-speaker protection effect
`cos_e(original, protected)` minus `cos_e(original, unprotected)` is not used --
the unprotected file IS the original, so the quantity is simply
`cos_e(original, protected)` under front end f, paired across speakers against
the same quantity under `native`. Unit: speaker. Estimator: mean paired
difference with a BCa 95 % CI (2000 resamples) and a Wilcoxon signed-rank p,
Holm-corrected within each encoder.

  * **WavLM / ECAPA protection survives the swapped front end** (their paired
    shift is small beside the d-vector's native-vs-swapped shift)
        -> the confound is DEAD. The d-vector's collapse is not attributable to
           its evaluation path, because giving the same path to the other two
           does not break them. C1's per-encoder ordering stands.
  * **WavLM / ECAPA collapse under the swapped front end**
        -> the confound is REAL and the paper must say so: the per-encoder
           pattern is partly an artefact of who is scored through what, and
           §32/§37's ordering needs restating with a matched front end.
  * **The d-vector IMPROVES markedly when its front end is removed**
        -> the front end was the cause after all, contradicting §38's 0.985,
           and 2.4 should be reinstated despite PLAN_V3 cutting it.

§38 predicts the first and third branches: the partial average absorbed the
window misalignment at cos 0.985, so removing the front end should move the
d-vector little. That prediction is on record before this run.

Run:
  python experiments/eval_frontend.py --n-files 29
  python experiments/eval_frontend.py --n-files 29 --purifiers none
Output:
  experiments/results/frontend_rescore.csv
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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch._dynamo   # noqa: F401
import transformers    # noqa: F401

OUT = os.path.join(PROJ, "experiments", "results", "frontend_rescore.csv")
SR = 16000


# --------------------------------------------------------------- front ends
def fe_identity(wav):
    return wav


def fe_preprocess(wav):
    """Resemblyzer's preprocess_wav: volume normalise + webrtcvad silence trim.

    This is the d-vector's own front end, applied here to whatever encoder is
    being scored. Non-differentiable (the VAD decision is a hard threshold),
    which is exactly why the optimiser never saw it.
    """
    from resemblyzer import preprocess_wav
    arr = wav.detach().squeeze().cpu().numpy().astype(np.float32)
    pp = preprocess_wav(arr, source_sr=SR)
    return torch.from_numpy(np.ascontiguousarray(pp)).reshape(1, -1).to(wav.device)


def fe_resample(wav, pct):
    """Resample by (1 + pct/100) and back to 16 kHz.

    A real upload path resamples; nothing in the crafting loop is invariant to
    it by construction. Done as a round trip so the duration is preserved and
    the only change is the interpolation, which is the part that could plausibly
    disturb a perturbation tuned to a particular time-frequency grid.
    """
    import torchaudio.functional as AF
    inter = int(round(SR * (1.0 + pct / 100.0)))
    up = AF.resample(wav, SR, inter)
    return AF.resample(up, inter, SR)


def build_front_ends(pcts=(2.0, -2.0)):
    fes = {"native": ("as the paper scores it", None),
           "swapped": ("preprocess_wav for WavLM/ECAPA; none for the d-vector",
                       None)}
    for p in pcts:
        fes[f"rs{p:+.0f}"] = (f"{p:+.0f} % resample round trip, native paths",
                              lambda w, p=p: fe_resample(w, p))
    return fes


def embed_under(encs, wav, fe_name, resampler):
    """Every encoder's embedding of `wav` under one front end.

    `native` and `swapped` are per-encoder by definition, so they are resolved
    here rather than by transforming the waveform once.
    """
    from experiments.exp_purifier_aware import real_embed_all
    from experiments.multi_encoders import (WavLMSVEnc, ECAPAEnc, DvecEnc,
                                            real_embed_wavlm_sv,
                                            real_embed_ecapa, real_embed_dvec)
    if resampler is not None:
        wav = resampler(wav)
    if fe_name != "swapped":
        return real_embed_all(encs, wav)

    out = {}
    pp = None
    for e in encs:
        if isinstance(e, DvecEnc):
            # The d-vector WITHOUT preprocess_wav: the torch-mel partial
            # average, i.e. scored the way its two neighbours are.
            with torch.no_grad():
                out[e.name] = e.embed(wav)
        else:
            if pp is None:
                pp = fe_preprocess(wav)
            if isinstance(e, WavLMSVEnc):
                out[e.name] = real_embed_wavlm_sv(e, pp)
            elif isinstance(e, ECAPAEnc):
                out[e.name] = real_embed_ecapa(e.clf, pp)
            else:
                with torch.no_grad():
                    out[e.name] = e.embed(pp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=0, dest="utt_index")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--tags", default="plain_tc_m30_seg,fullnsT_tc_m30_seg",
                    help="saved protected arms to re-score")
    ap.add_argument("--purifiers", default="none,diffwave",
                    help="attacker purifier applied BEFORE the front end")
    ap.add_argument("--dw-t", type=int, default=3, dest="dw_t")
    ap.add_argument("--dw-ckpt", default=os.path.join(
        PROJ, "checkpoints", "foreign", "diffwave_libri_deantifake.pkl"),
        dest="dw_ckpt")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--resample-pcts", default="2,-2", dest="resample_pcts")
    ap.add_argument("--out", default=OUT)
    from utils.provenance import add_prereg_arg, provenance
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from experiments.exp_purifier_aware import build_encoders, cs, WAV_DIR
    from experiments.librispeech import (list_librispeech, reference_items,
                                         load_flac, cross_session_split)

    seeds = [int(v) for v in a.seeds.split(",")]
    tags = [t for t in a.tags.split(",") if t]
    purs = [p for p in a.purifiers.split(",") if p]
    pcts = tuple(float(v) for v in a.resample_pcts.split(",") if v)
    fes = build_front_ends(pcts)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=4.0)
    ref = reference_items(items, min_sec=4.0, utt_index=a.utt_index)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][a.spk_offset:a.spk_offset + a.n_files]

    encs = build_encoders(device)
    dw = None
    if "diffwave" in purs:
        if not os.path.isfile(a.dw_ckpt):
            raise SystemExit(f"[FE] DiffWave checkpoint missing: {a.dw_ckpt}")
        from models.diffwave_purifier import load_diffwave
        dw = load_diffwave(a.dw_ckpt, t=a.dw_t, device=device)

    print("=" * 78)
    print("COUNCIL #10 -- the front-end confound, evaluation only")
    print("=" * 78)
    print(f"{len(spks)} cross-session speakers from {a.subset} | arms: "
          f"{', '.join(tags)} | purifiers: {', '.join(purs)}"
          + (f" (DiffWave t={a.dw_t})" if dw is not None else ""))
    print(f"front ends: {', '.join(fes)}")
    for k, (desc, _) in fes.items():
        print(f"  {k:9s} {desc}")
    print(f"code_sha={PROV['code_sha']} prereg_sha={PROV['prereg_sha'] or '-'}")
    print("Nothing is re-crafted. Read against the pre-registered branches in "
          "this file's docstring.\n")

    rows = []
    for i, spk in enumerate(spks):
        it = ref[spk]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0).to(device)

        arms = {"unprotected": wav_o}
        for t in tags:
            p = os.path.join(WAV_DIR, f"{name}_{t}.wav")
            if not os.path.isfile(p):
                print(f"  missing {p}")
                continue
            y, _ = sf.read(p, dtype="float32")
            arms[t] = torch.from_numpy(y).reshape(1, -1).to(device)[
                :, : wav_o.shape[-1]]

        for pname in purs:
            # The attacker purifies first; the verifier's front end comes after.
            # Averaged over seeds, as everywhere else, so one lucky draw of the
            # purifier does not decide a cell.
            pur_arms = {}
            for arm, w in arms.items():
                if pname == "none":
                    pur_arms[arm] = [w]
                else:
                    with torch.no_grad():
                        pur_arms[arm] = [dw(w, seed=sd) for sd in seeds]

            for fe_name, (_, resampler) in fes.items():
                # Reference: the CLEAN original, UNPURIFIED, through this front
                # end. That is the project's convention (eval_saved.py's
                # `emb_o`) and it is what the protocol actually does -- the
                # enrolment utterance is clean, only the published clip passes
                # through the attacker. Referencing the purified original
                # instead would make the `unprotected` row 1.000 by
                # construction and throw away the control that says the
                # purifier plus front end still preserves identity.
                ref_emb = embed_under(encs, wav_o, fe_name, resampler)
                for arm, ws in pur_arms.items():
                    embs = [embed_under(encs, w, fe_name, resampler) for w in ws]
                    for n in ref_emb:
                        vals = [cs(ref_emb[n], e[n]) for e in embs]
                        rows.append(dict(
                            speaker=spk, file=name, variant=arm,
                            purifier=pname, front_end=fe_name, encoder=n,
                            cos_to_clean=round(float(np.mean(vals)), 4),
                            n_seeds=len(vals), **PROV))
        print(f"  [{i + 1}/{len(spks)}] spk{spk}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 78)
    print("mean cos(clean, arm) by encoder x front end  "
          "(lower on a protected arm = more protection)")
    print("=" * 78)
    for pname in purs:
        d = df[df.purifier == pname]
        if not len(d):
            continue
        print(f"\n--- purifier: {pname} ---")
        piv = d.pivot_table(index=["encoder", "variant"], columns="front_end",
                            values="cos_to_clean")
        cols = [c for c in fes if c in piv.columns]
        print(piv[cols].round(3).to_string())
    print(f"\nsaved {a.out}")
    print("Decide with: python experiments/analyse_frontend.py")


if __name__ == "__main__":
    main()
