"""
experiments/gate_cloning.py
===========================
GATE: does UNPROTECTED audio actually clone through DiffVC?

Nothing downstream means anything until this passes. If clean audio does not
clone, then a protected sample that "fails to clone" has demonstrated nothing --
the attack was broken before the protection was applied. This is the cloning
analogue of the purifier denoising-gain gate, and this project has already been
bitten twice by exactly this shape of error: a purifier that did not purify, and a
channel that destroyed the identity of unprotected audio.

DESIGN -- three distinct utterances per speaker, which is the point
------------------------------------------------------------------
    reference   the victim's published clip. DiffVC takes its speaker identity.
    source      a DIFFERENT speaker's utterance, supplying the words. This is the
                attacker's own speech.
    enrollment  a THIRD clip of the victim, held back. All similarity is measured
                against this.

Measuring the clone against the *reference* would be near-circular: the reference
is literally the model's conditioning input. Using a held-back enrollment utterance
asks the question that matters -- did the attacker obtain the victim's IDENTITY, or
just echo one file?

FOUR NUMBERS, AND WHAT EACH IS FOR
----------------------------------
    sim(reference,  enrollment)   same-speaker ceiling. What "this is the victim"
                                  looks like with no cloning involved at all.
    sim(source,     enrollment)   different-speaker floor, i.e. chance.
    sim(clone,      enrollment)   THE GATE. Must sit near the ceiling, far above
                                  the floor.
    sim(clone, source_enrollment) the diagnostic for the most likely failure: if
                                  conversion silently did nothing, the clone still
                                  carries the SOURCE speaker's identity.

Run:
  python experiments/gate_cloning.py --n-files 10
"""
import sys, io, os, argparse, time

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
import torch._dynamo  # noqa: F401
import transformers   # noqa: F401

from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from evaluation.cloning_attack import FIELD_CONVENTION_THRESHOLD

OUT = os.path.join(PROJ, "experiments", "results", "gate_cloning.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=10, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--ref-utt", type=int, default=3, dest="ref_utt")
    ap.add_argument("--enroll-utt", type=int, default=0, dest="enroll_utt")
    ap.add_argument("--n-timesteps", type=int, default=30, dest="n_timesteps")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.diffvc_clone import DiffVCCloner
    vc = DiffVCCloner(device=str(device), n_timesteps=args.n_timesteps)
    encs = build_encoders(device)

    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=max(args.ref_utt,
                                                                   args.enroll_utt) + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_files]

    rows = []
    for i, spk in enumerate(spks):
        # the attacker's content comes from the NEXT speaker in the list
        src_spk = spks[(i + 1) % len(spks)]
        ref = load_flac(ps[spk][args.ref_utt]["path"], 4.0).to(device)
        enr = load_flac(ps[spk][args.enroll_utt]["path"], 4.0).to(device)
        src = load_flac(ps[src_spk][args.ref_utt]["path"], 4.0).to(device)
        src_enr = load_flac(ps[src_spk][args.enroll_utt]["path"], 4.0).to(device)

        t0 = time.time()
        clone = vc.clone(src, ref).to(device)
        dt = time.time() - t0

        e_enr, e_ref = real_embed_all(encs, enr), real_embed_all(encs, ref)
        e_src, e_srcenr = real_embed_all(encs, src), real_embed_all(encs, src_enr)
        e_cl = real_embed_all(encs, clone)
        for n in e_enr:
            rows.append(dict(
                speaker=spk, source_speaker=src_spk, encoder=n, clone_s=round(dt, 1),
                ceiling_ref_vs_enroll=round(cs(e_ref[n], e_enr[n]), 4),
                floor_src_vs_enroll=round(cs(e_src[n], e_enr[n]), 4),
                gate_clone_vs_enroll=round(cs(e_cl[n], e_enr[n]), 4),
                leak_clone_vs_srcenroll=round(cs(e_cl[n], e_srcenr[n]), 4)))
        print(f"  spk{spk} <- content from spk{src_spk}  ({dt:.0f}s)  " +
              "  ".join(f"{n.split('-')[0]}:{rows[-len(e_enr)+j]['gate_clone_vs_enroll']:+.3f}"
                        for j, n in enumerate(e_enr)), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    g = df.groupby("encoder")[["ceiling_ref_vs_enroll", "floor_src_vs_enroll",
                               "gate_clone_vs_enroll", "leak_clone_vs_srcenroll"]].mean()
    print("\n" + "=" * 78)
    print(f"GATE: does UNPROTECTED audio clone through DiffVC?  (n={len(spks)})")
    print("=" * 78)
    print(g.round(4).to_string())

    print("\n" + "-" * 78)
    ok = True
    for n, r in g.iterrows():
        span = r.ceiling_ref_vs_enroll - r.floor_src_vs_enroll
        # where does the clone sit between chance and the same-speaker ceiling?
        frac = (r.gate_clone_vs_enroll - r.floor_src_vs_enroll) / max(span, 1e-9)
        leaks = r.leak_clone_vs_srcenroll > r.gate_clone_vs_enroll
        verdict = ("LEAKS SOURCE" if leaks else
                   "clones" if frac > 0.5 else
                   "WEAK" if frac > 0.2 else "FAILS")
        if leaks or frac <= 0.2:
            ok = False
        print(f"  {n:<22} clone sits {100*frac:5.1f} % of the way from chance "
              f"({r.floor_src_vs_enroll:+.3f}) to same-speaker ({r.ceiling_ref_vs_enroll:+.3f})"
              f"   -> {verdict}")
    print(f"\n  (field convention: a clone counts as successful at "
          f"cos > {FIELD_CONVENTION_THRESHOLD} on ECAPA)")
    ec = g.loc["ECAPA-TDNN"] if "ECAPA-TDNN" in g.index else None
    if ec is not None:
        n_ok = int((df[df.encoder == "ECAPA-TDNN"].gate_clone_vs_enroll
                    > FIELD_CONVENTION_THRESHOLD).sum())
        print(f"  ECAPA: {n_ok}/{len(spks)} clones exceed 0.25 "
              f"(mean {ec.gate_clone_vs_enroll:.3f})")

    print("\n" + "=" * 78)
    if ok:
        print("  GATE PASSED -- unprotected audio clones. Protection numbers measured")
        print("  against this attacker are meaningful. Proceed to 2.1.")
    else:
        print("  GATE FAILED -- STOP. Do not run the protection experiment: a")
        print("  protected sample that fails to clone would prove nothing, because")
        print("  clean audio does not clone either. Diagnose first. Likely causes:")
        print("    * resampling 16 <-> 22.05 kHz damaging the reference")
        print("    * 4 s clips too short for the speaker encoder")
        print("    * the spectral-subtraction step (try --spectral-subtract off)")
        print("    * source/reference passed the wrong way round")
    print("=" * 78)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
