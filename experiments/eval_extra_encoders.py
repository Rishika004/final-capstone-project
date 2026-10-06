"""
experiments/eval_extra_encoders.py
==================================
The outcome half of council #13b: does the constraint behave on two unseen
encoders the way α predicted it would?

Evaluation only. It scores the **existing** `full_tc_m30_seg` and
`fullnsT_tc_m30_seg` wavs — the frozen v1 arms — through CAM++ and ERes2Net,
after the attacker's foreign purifier. Nothing is crafted, and neither encoder
is in `build_encoders`, so no file and no previously reported number can move
(`PREREG_13b_prospective_encoders.md` §6.2, §6.4).

ENDPOINT, fixed in the pre-registration §4
------------------------------------------
    Δ = mean over speakers of  cos(clean, fullnsT) − cos(clean, full)
    unit        speaker, n = 29
    sign        NEGATIVE = the constraint improves protection
                POSITIVE = the reversal
    cells       2 purifiers × t ∈ {1,3} × 2 encoders = 8
    estimator   mean paired difference, BCa 95 % CI (2000 resamples),
                Wilcoxon signed-rank, Holm over the 8 cells
    thresholds  NONE. The endpoint is a paired difference of cosines to the
                clean original, so no operating point enters it — and §45g
                showed a margin difference cancels τ anyway. That is deliberate:
                these two encoders have no calibration on this corpus, and
                inventing one would add a free parameter to a prospective test.

Both sides of every comparison pass through the same purifier and the same
encoder; only the arm changes.

Run (after the α measurement and the §7 prediction commit):
  python experiments/eval_extra_encoders.py --n-files 29
Output:
  experiments/results/extra_encoder_protection.csv
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

OUT = os.path.join(PROJ, "experiments", "results",
                   "extra_encoder_protection.csv")
ARMS = ("full_tc_m30_seg", "fullnsT_tc_m30_seg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=0, dest="utt_index")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--dw-libri", default=os.path.join(
        PROJ, "checkpoints", "foreign", "diffwave_libri_deantifake.pkl"),
        dest="dw_libri")
    ap.add_argument("--dw-sc09", default=os.path.join(
        PROJ, "checkpoints", "foreign", "diffwave_sc09_philsyn.pkl"),
        dest="dw_sc09")
    ap.add_argument("--out", default=OUT)
    from utils.provenance import add_prereg_arg, provenance
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from experiments.exp_purifier_aware import cs, WAV_DIR
    from experiments.extra_encoders import build_extra
    from experiments.librispeech import (list_librispeech, reference_items,
                                         load_flac, cross_session_split)
    from models.diffwave_purifier import load_diffwave

    seeds = [int(v) for v in a.seeds.split(",")]
    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=4.0)
    ref = reference_items(items, min_sec=4.0, utt_index=a.utt_index)
    ok = set(cross_session_split(items, min_sec=4.0))
    spks = [s for s in sorted(ref) if s in ok][
        a.spk_offset:a.spk_offset + a.n_files]

    encs = build_extra(device)
    purs = []
    for ck, tag in ((a.dw_libri, "DiffWave-LibriSpeech"),
                    (a.dw_sc09, "DiffWave-SC09")):
        if not os.path.isfile(ck):
            raise SystemExit(f"[13b] missing purifier checkpoint {ck}")
        for t in (1, 3):
            purs.append((tag, t, load_diffwave(ck, t=t, device=device)))

    print("=" * 78)
    print("COUNCIL #13b -- the outcome half, evaluation only")
    print("=" * 78)
    print(f"{len(spks)} cross-session speakers | arms: {', '.join(ARMS)}")
    print(f"encoders (eval-only, never crafted against): "
          f"{[e.name for e in encs]}")
    print(f"cells: {len(purs)} purifier x t  ×  {len(encs)} encoders = "
          f"{len(purs) * len(encs)}")
    print(f"code_sha={PROV['code_sha']} prereg_sha={PROV['prereg_sha'] or '-'}")
    print("Endpoint: paired cos(clean, fullnsT) - cos(clean, full). "
          "NEGATIVE = constraint helps.\n")

    rows = []
    for i, spk in enumerate(spks):
        it = ref[spk]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0).to(device)

        arms = {}
        for arm in ARMS:
            p = os.path.join(WAV_DIR, f"{name}_{arm}.wav")
            if not os.path.isfile(p):
                print(f"  missing {p}")
                continue
            y, _ = sf.read(p, dtype="float32")
            arms[arm] = torch.from_numpy(y).reshape(1, -1).to(device)[
                :, : wav_o.shape[-1]]
        if len(arms) != len(ARMS):
            print(f"  skipping spk{spk}: not both arms present")
            continue

        for pname, t, dw in purs:
            with torch.no_grad():
                ref_e = {e.name: e.embed(wav_o) for e in encs}
                for arm, w in arms.items():
                    vals = {e.name: [] for e in encs}
                    for sd in seeds:
                        wp = dw(w, seed=sd)
                        for e in encs:
                            vals[e.name].append(cs(ref_e[e.name], e.embed(wp)))
                    for e in encs:
                        rows.append(dict(
                            speaker=spk, file=name, variant=arm,
                            purifier=pname, t=t, encoder=e.name,
                            cos_to_clean=round(float(np.mean(vals[e.name])), 4),
                            n_seeds=len(seeds), **PROV))
        print(f"  [{i + 1}/{len(spks)}] spk{spk}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\nmean cos(clean, arm) by encoder x purifier x t")
    print(df.pivot_table(index=["encoder", "purifier", "t"], columns="variant",
                         values="cos_to_clean").round(4).to_string())
    print(f"\nsaved {a.out}")
    print("Decide with: python experiments/analyse_extra_encoders.py")


if __name__ == "__main__":
    main()
