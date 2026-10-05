"""
experiments/eval_saved.py
=========================
Re-evaluate ALREADY-CRAFTED protected files (experiments/results/purifier_aware/
<name>_<tag>.wav) through the attacker's channel at several purification
strengths, including t=0 (Griffin-Lim resynthesis only, no diffusion).
No crafting -> cheap. Separates "removed by the purifier" from "removed by the
mel/vocoder bottleneck".

Run:
  python experiments/eval_saved.py --tags plain_m20_f20,full_m20_f20 --t 0,50,200
Output:
  experiments/results/eval_saved.csv
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
import torch._dynamo  # noqa: F401
import transformers  # noqa: F401
import soundfile as sf

from evaluation.evaluate import wav_to_logmel
from models.real_diffpure import load_real_diffpure, MelDiffPure
from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs, attacker_resynth, WAV_DIR
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET

OUT = os.path.join(PROJ, "experiments", "results", "eval_saved.csv")
# EER thresholds calibrated through each channel (see thresholds_*.csv)
THR = {0:   {"WavLM-SV": 0.888, "ECAPA-TDNN": 0.490, "Resemblyzer d-vector": 0.749},   # GL only
       50:  {"WavLM-SV": 0.889, "ECAPA-TDNN": 0.324, "Resemblyzer d-vector": 0.717},
       100: {"WavLM-SV": 0.883, "ECAPA-TDNN": 0.328, "Resemblyzer d-vector": 0.703},
       200: {"WavLM-SV": 0.857, "ECAPA-TDNN": 0.292, "Resemblyzer d-vector": 0.665}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="plain_m20_f20,full_m20_f20,plain_m30_f15,full_m30_f15")
    ap.add_argument("--t", default="0,50,200")
    ap.add_argument("--seeds", default="2026,7,99")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    ap.add_argument("--subset", default="",
                    help="librispeech subset override (default: EVAL_SUBSET). Pin "
                         "it to the corpus of the run this describes.")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="use the SHARED reference selector and the cross-session "
                         "speaker set, so this matches the crafting/cloning runs.")
    ap.add_argument("--n-files", type=int, default=20, dest="n_files")
    ap.add_argument("--purifier-ckpt", default=os.path.join(PROJ, "checkpoints", "purifier_v2.pt"),
                    dest="purifier_ckpt")
    ap.add_argument("--attacker-vocoder", default="gl", dest="attacker_vocoder")
    ap.add_argument("--diffwave-ckpt", default="", dest="diffwave_ckpt",
                    help="evaluate through a FOREIGN waveform purifier (DiffWave) "
                         "instead of our mel purifier; --t then means ITS timestep")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    ap.add_argument("--thresholds", default="",
                    help="wavlmsv,ecapa,dvec EER thresholds for this channel; "
                         "applied to every t>0 (t=0 keeps the Griffin-Lim-only values)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = [int(v) for v in args.t.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    tags = args.tags.split(",")
    if args.thresholds:
        v = [float(x) for x in args.thresholds.split(",")]
        for t in ts:
            if t > 0:
                THR[t] = {"WavLM-SV": v[0], "ECAPA-TDNN": v[1], "Resemblyzer d-vector": v[2]}

    encs = build_encoders(device)
    dw = None
    if args.diffwave_ckpt:
        # Foreign waveform purifier. It operates directly on the published wav --
        # no mel, no vocoder -- which makes it a STRICTLY STRONGER attacker than
        # our chain, since it never pays the Griffin-Lim cost that E6 measured at
        # alpha 0.57-0.68. Thresholds must come from the matching
        # thresholds_diffwave_t*.csv; reusing our v3 thresholds here would be
        # invalid, because the channel is different.
        from models.diffwave_purifier import load_diffwave
        dw = {t: (load_diffwave(args.diffwave_ckpt, t=t, device=device) if t > 0 else None)
              for t in ts}
        dps = {t: None for t in ts}
    else:
        base = load_real_diffpure(checkpoint_path=args.purifier_ckpt, t_diffuse=50, device=device)
        dps = {t: (MelDiffPure(base.denoiser, n_steps=base.n_steps, t_diffuse=t, sampler=args.sampler).to(device).eval()
                   if t > 0 else None) for t in ts}

    # Corpus and speaker selection go through the SAME shared selector the
    # crafting and cloning scripts use (fix 2026-09-19). Without --subset these
    # scripts silently followed EVAL_SUBSET, which moved from dev-clean to
    # test-clean, and without --cross-session they picked a different utterance
    # AND a different speaker set than the run they are meant to describe.
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             args.subset or EVAL_SUBSET, 4.0)
    if args.cross_session:
        from experiments.librispeech import reference_items, cross_session_split
        ref = reference_items(items, min_sec=4.0, utt_index=args.utt_index)
        ok = set(cross_session_split(items, min_sec=4.0))
        spks = [s for s in sorted(ref) if s in ok][
            args.spk_offset:args.spk_offset + args.n_files]
        ps = {s: {args.utt_index: ref[s]} for s in spks}
        print(f"[sel] cross-session: {len(spks)} speakers from "
              f"{args.subset or EVAL_SUBSET} (shared selector)")
    else:
        ps = per_speaker(items, n_utt=args.utt_index + 1)
        spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]
    rows = []
    for spk in spks:
        it = ps[spk][args.utt_index]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0).to(device)
        mel_o = wav_to_logmel(wav_o, device)
        emb_o = real_embed_all(encs, wav_o)
        for tag in tags:
            path = os.path.join(WAV_DIR, f"{name}_{tag}.wav")
            if not os.path.isfile(path):
                print(f"  missing {path}"); continue
            y, sr = sf.read(path, dtype="float32")
            wav_p = torch.from_numpy(y).reshape(1, -1).to(device)[:, : wav_o.shape[-1]]
            mel_p = wav_to_logmel(wav_p, device)
            emb_p = real_embed_all(encs, wav_p)
            for t in ts:
                after = {n: [] for n in THR[t]}; ctrl = {n: [] for n in THR[t]}
                psr, dcos = [], []
                for sd in (seeds if t > 0 else [seeds[0]]):
                    if dw is not None:
                        with torch.no_grad():
                            w_pp = dw[t](wav_p, seed=sd) if t > 0 else wav_p
                            w_op = dw[t](wav_o, seed=sd) if t > 0 else wav_o
                            d_in = (mel_p - mel_o).reshape(1, -1)
                            d_out = (wav_to_logmel(w_pp, device)
                                     - wav_to_logmel(w_op, device)).reshape(1, -1)
                            psr.append((d_out.norm() / (d_in.norm() + 1e-12)).item())
                            dcos.append(torch.nn.functional.cosine_similarity(d_in, d_out).item())
                        e_pp = real_embed_all(encs, w_pp)
                        e_op = real_embed_all(encs, w_op)
                    else:
                        with torch.no_grad():
                            mel_pp = dps[t](mel_p, seed=sd) if t > 0 else mel_p
                            mel_op = dps[t](mel_o, seed=sd) if t > 0 else mel_o
                            d_in = (mel_p - mel_o).reshape(1, -1); d_out = (mel_pp - mel_op).reshape(1, -1)
                            psr.append((d_out.norm() / (d_in.norm() + 1e-12)).item())
                            dcos.append(torch.nn.functional.cosine_similarity(d_in, d_out).item())
                        e_pp = real_embed_all(encs, attacker_resynth(mel_pp, wav_p, args.attacker_vocoder))
                        e_op = real_embed_all(encs, attacker_resynth(mel_op, wav_o, args.attacker_vocoder))
                    for n in THR[t]:
                        after[n].append(cs(emb_o[n], e_pp[n])); ctrl[n].append(cs(emb_o[n], e_op[n]))
                for n, tau in THR[t].items():
                    ca, cc = float(np.mean(after[n])), float(np.mean(ctrl[n]))
                    # RESULTS 28b. The mean over seeds is the AVERAGE attacker. A
                    # patient attacker keeps the seed that recovered the most
                    # identity, so the best-of-seeds cosine is recorded too, mirroring
                    # cos_after_max / fails_all_seeds in exp_purifier_aware.py.
                    cmax = float(np.max(after[n]))
                    rows.append(dict(variant=tag, file=name, eval_t=t, encoder=n, threshold=tau,
                                     mel_psr=round(float(np.mean(psr)), 4),
                                     mel_dircos=round(float(np.mean(dcos)), 4),
                                     cos_before=round(cs(emb_o[n], emb_p[n]), 4),
                                     cos_after_purify=round(ca, 4), cos_ctrl_unprot=round(cc, 4),
                                     cos_after_max=round(cmax, 4),
                                     clone_fails_after=bool(ca < tau), ctrl_identity_kept=bool(cc >= tau),
                                     fails_all_seeds=bool(cmax < tau),
                                     defence_valid=bool(ca < tau and cc >= tau),
                                     n_seeds=len(after[n])))
        print(f"{name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\nperturbation survival through the purifier (mel PSR = |P(x_p)-P(x)| / |x_p-x|, direction cos)")
    print(df.drop_duplicates(["variant", "file", "eval_t"]).groupby(["variant", "eval_t"])[["mel_psr", "mel_dircos"]]
          .mean().round(3).to_string())
    print("\nPROTECTION (% files defence valid) by variant x t x encoder")
    piv = df.pivot_table(index=["variant", "eval_t"], columns="encoder", values="defence_valid",
                         aggfunc=lambda v: 100 * np.mean(v)).round(0)
    print(piv.to_string())
    print("\nmean cos_after / cos_ctrl")
    print(df.pivot_table(index=["variant", "eval_t"], columns="encoder",
                         values=["cos_after_purify", "cos_ctrl_unprot"]).round(3).to_string())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
