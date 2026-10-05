"""
experiments/make_demo_audio.py
==============================
Produce a listening set for one speaker, so the result can be HEARD rather than
only read off a table. Five files:

    1_original.wav          the victim's real voice, unmodified
    2_protected.wav         the same recording after our perturbation  <- listen: same person?
    3_source.wav            the attacker's own voice (what they say)
    4_clone_unprotected.wav DiffVC clone built from file 1  <- the attack WORKING
    5_clone_protected.wav   DiffVC clone built from file 2  <- the attack DEFEATED

The pair that matters is 4 vs 5. File 4 should sound like the victim; that is the
threat. File 5 is the same attack against the protected clip and should sound like
somebody else. Files 1 and 2 should be indistinguishable to the ear; if they are
not, the perceptual constraint failed and the table numbers are moot.

Speaker 8297 is the default: its unprotected clone succeeded (similarity 0.409,
above the 0.376 threshold) and its protected clone failed (0.271). Chosen from the
results, not cherry-picked by listening.

    python experiments/make_demo_audio.py --speaker 8297
"""
import sys, io, os, argparse

if not getattr(sys.stdout, "_nvp_utf8", False):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import soundfile as sf
import torch

OUT = os.path.join(PROJ, "experiments", "results", "demo_audio")
WAV_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=int, default=8297)
    ap.add_argument("--variant", default="full_m20_f20_nodvec",
                    help="which protected arm to demo (default: the honest, "
                         "held-out-encoder crafting)")
    ap.add_argument("--subset", default="dev-clean")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from experiments.librispeech import list_librispeech, per_speaker, load_flac
    per = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                       a.subset, min_sec=4.0), n_utt=4)
    spks = sorted(per)
    if a.speaker not in per:
        raise SystemExit(f"speaker {a.speaker} not in {a.subset}")

    # same pairing the experiment uses: source speaker is the NEXT one in order
    i = spks.index(a.speaker)
    src_spk = spks[(i + 1) % len(spks)]
    it = per[a.speaker][3]
    name = f"spk{a.speaker}_{it['chapter']}_{it['utt']:04d}"
    print(f"[demo] victim spk{a.speaker}, reference {name}")
    print(f"[demo] attacker's source voice: spk{src_spk}")

    victim = load_flac(it["path"], 4.0).to(device)
    source = load_flac(per[src_spk][3]["path"], 4.0).to(device)

    prot_path = os.path.join(WAV_DIR, f"{name}_{a.variant}.wav")
    if not os.path.isfile(prot_path):
        raise SystemExit(f"missing protected wav: {prot_path}")
    y, sr = sf.read(prot_path, dtype="float32")
    protected = torch.from_numpy(y).reshape(1, -1).to(device)[:, : victim.shape[-1]]

    def save(n, w):
        p = os.path.join(OUT, n)
        sf.write(p, w.squeeze(0).detach().cpu().numpy(), 16000)
        print(f"  wrote {n}")
        return p

    save("1_original.wav", victim)
    save("2_protected.wav", protected)
    save("3_source.wav", source)

    print("[demo] loading DiffVC ...")
    from models.diffvc_clone import DiffVCCloner
    vc = DiffVCCloner(device=str(device))

    print("[demo] cloning from the UNPROTECTED reference (the attack working) ...")
    save("4_clone_unprotected.wav", vc.clone(source, victim).to(device))
    print("[demo] cloning from the PROTECTED reference (the attack defeated) ...")
    save("5_clone_protected.wav", vc.clone(source, protected).to(device))

    # score them so the listening test has the numbers beside it
    from experiments.exp_purifier_aware import build_encoders, real_embed_all, cs
    from utils.csvio import read_csv
    encs = build_encoders(device)
    thr = {r.encoder: float(r.threshold)
           for r in read_csv(os.path.join(PROJ, "experiments", "results",
                                          "thresholds_clean.csv")).itertuples()}
    enrol = load_flac(per[a.speaker][0]["path"], 4.0).to(device)
    e_enr = real_embed_all(encs, enrol)
    print("\nsimilarity of each clone to a HELD-BACK recording of the victim")
    print("(above threshold = the clone would pass as the victim)\n")
    print(f"  {'file':<24}{'encoder':<22}{'sim':>7}{'thr':>8}   verdict")
    for lbl, f in (("clone_unprotected", "4_clone_unprotected.wav"),
                   ("clone_protected", "5_clone_protected.wav")):
        w, _ = sf.read(os.path.join(OUT, f), dtype="float32")
        e = real_embed_all(encs, torch.from_numpy(w).reshape(1, -1).to(device))
        for n in e_enr:
            sim = cs(e[n], e_enr[n])
            ok = "PASSES as victim" if sim >= thr[n] else "rejected"
            print(f"  {lbl:<24}{n:<22}{sim:>7.3f}{thr[n]:>8.3f}   {ok}")
    print(f"\n[demo] files in {OUT}")


if __name__ == "__main__":
    main()
