"""
experiments/score_f5_clones.py
==============================
N2 stage C -- score the F5-TTS clones with the project's encoders and the
cross-session thresholds already in use.

PREREG: paper/PREREG_N2_modern_cloner.md

The verdict is exactly the one exp_cloning_protection.py computes:

    cos(E(clone), E(enrolment_clean)) >= tau ?

with tau the per-encoder clean-channel EER threshold from the cross-session
calibration (WavLM-SV 0.8926, ECAPA 0.3358, d-vector 0.7029), plus the ECAPA
0.25 field convention alongside. The threshold does NOT vary with the
attacker's purifier, for the reason set out in exp_cloning_protection's
docstring: the purifier acts on the reference, upstream of the cloner, so the
verifier's channel is the same in every condition.

Output schema is cloning_tc_*.csv's, plus a `cloner` column so the DiffVC and
F5-TTS rows can never be pooled by accident.

Run:
  python experiments/score_f5_clones.py --prereg paper/PREREG_N2_modern_cloner.md
"""
import sys, io, os, argparse, time

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import torch._dynamo   # noqa: F401
import transformers    # noqa: F401
import soundfile as sf
import pandas as pd

from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs
from experiments.librispeech import load_flac
from experiments.exp_cloning_protection import load_thresholds
from evaluation.cloning_attack import FIELD_CONVENTION_THRESHOLD
from utils.provenance import provenance, add_prereg_arg

CLONE_DIR = os.path.join(PROJ, "experiments", "results", "f5_clones")
PROMPTS = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")
OUT = os.path.join(PROJ, "experiments", "results", "cloning_f5tts.csv")
THR = os.path.join(PROJ, "experiments", "results", "thresholds_clean_xsess_tc.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--clone-dir", default=CLONE_DIR, dest="clone_dir")
    ap.add_argument("--thresholds", default=THR)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--cloner", default="F5-TTS")
    ap.add_argument("--out", default=OUT)
    add_prereg_arg(ap)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROV = provenance(args.prereg)

    thr = load_thresholds(args.thresholds, cross_session=True, subset=args.subset)
    print("per-encoder EER thresholds (clean channel): "
          + ", ".join(f"{k}={v:.4f}" for k, v in thr.items()))

    encs = build_encoders(device)
    prompts = pd.read_csv(args.prompts)
    print(f"[score] {len(prompts)} prompt rows")

    # Enrolment embedding once per speaker: it is the clean held-back utterance
    # and does not depend on arm or channel.
    e_enr, rows, t0, n_missing = {}, [], time.time(), 0
    for spk, grp in prompts.groupby("speaker"):
        p = os.path.join(PROJ, grp.iloc[0]["enrol_path"])
        enr = load_flac(p, 4.0).to(device)
        e_enr[spk] = real_embed_all(encs, enr)

    for i, r in prompts.iterrows():
        name, arm, pname = r["name"], r["variant"], r["purifier"]
        cp = os.path.join(args.clone_dir, f"{name}__{arm}__{pname}__f5.wav")
        if not os.path.isfile(cp):
            print(f"  missing clone {cp}")
            n_missing += 1
            continue
        y, sr = sf.read(cp, dtype="float32")
        clone = torch.from_numpy(y).reshape(1, -1).to(device)
        e_cl = real_embed_all(encs, clone)
        for n in e_cl:
            sim = cs(e_cl[n], e_enr[r["speaker"]][n])
            tau = thr[n]
            rows.append(dict(
                speaker=r["speaker"], source_speaker=r["source_speaker"],
                variant=arm, purifier=pname, encoder=n,
                clone_sim=round(sim, 4), threshold=round(tau, 4),
                margin=round(sim - tau, 4),
                clone_succeeds=bool(sim > tau),
                succeeds_at_025=(bool(sim > FIELD_CONVENTION_THRESHOLD)
                                 if n == "ECAPA-TDNN" else None),
                cloner=args.cloner,
                clone_sec=round(len(y) / sr, 2),
                **PROV))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[score] {len(df)} rows -> {args.out} ({time.time()-t0:.0f} s)")
    if n_missing:
        print(f"[score] {n_missing} clones MISSING")

    if len(df):
        print("\nclone success rate (EER thresholds):")
        piv = (df.groupby(["purifier", "variant", "encoder"])["clone_succeeds"]
                 .agg(["mean", "sum", "count"]))
        print(piv.to_string())
        print("\nmean margin:")
        print(df.groupby(["purifier", "variant", "encoder"])["margin"]
                .mean().to_string())


if __name__ == "__main__":
    main()
