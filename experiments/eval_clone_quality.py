"""
experiments/eval_clone_quality.py
=================================
Council #6, both halves, on the shipping arm and with real references.

WHAT WAS WRONG WITH THE QUALITY EVIDENCE
----------------------------------------
`eval_quality_nr.py` answers "how does OUR PUBLISHED audio sound", on
dev-clean, non-segment arms, with UTMOS + SQUIM, and with a WER whose reference
is **Whisper-small's own transcript of the original**. Three defects, all named
by the 2026-10-04 council:

  #6a  it never ran on the arm the paper ships (`fullnsT_tc_m30_seg`, 29
       cross-session test-clean speakers). The shipping arm has only
       PESQ 2.2 / SNR 15 dB and no no-reference score at all.
  #6b  no NISQA. SQUIM is a reference-free PESQ/STOI *predictor* and was being
       read as a naturalness score, which it is not.
  #6c  the WER reference was an ASR transcript, so ASR errors cancel and the
       number flatters every condition equally. LibriSpeech ground truth plus
       MFA word timings are on disk; there is no reason to use a proxy.

And a fourth defect that is not about our audio at all:

  #6d  **there is no quality evidence on what the ATTACKER produces.** Every
       number in the paper is speaker similarity of the clone. A protection
       that merely makes the clone sound broken is a different and weaker claim
       than one that moves the speaker identity, and the two cannot be told
       apart from similarity alone. If protected clones score far below
       unprotected clones on naturalness, then part of what we report as
       "protection" is really "the attacker's output is damaged", and a
       reviewer will say so.

This script answers all four on one cohort, in one table.

CONDITIONS
----------
  original              the clean 4-s crop                       (anchor)
  protected:<tag>       what the victim PUBLISHES                (#6a)
  clone:<arm>:<pur>     what the ATTACKER gets out of DiffVC     (#6d)

`clone` rows come from the wavs written by `exp_cloning_protection.py
--save-clones`.

THE WER REFERENCE, AND A TRAP IN IT
-----------------------------------
A DiffVC clone carries the ATTACKER's words in the VICTIM's voice:
`clone = DiffVC(source=attacker's speech, reference=victim's published clip)`.
So a clone's transcript must be scored against the **source** speaker's
utterance, not the victim's. Scoring it against the victim's transcript would
report ~100 % WER for every clone, protected or not, and look like a finding.
The source speaker is read from the cloning CSV's `source_speaker` column so
the mapping cannot drift from the run that produced the audio.

References come from `gilkeyio/librispeech-alignments` (MFA `english_us_arpa`,
already on disk for §29a), clipped to the 4-s crop by word midpoint, and
normalised with Whisper's own `EnglishTextNormalizer` on both sides.

Run (after the --save-clones cloning pass):
  python experiments/eval_clone_quality.py --n-files 29
  python experiments/eval_clone_quality.py --n-files 29 --asr openai/whisper-small.en
Output:
  experiments/results/clone_quality.csv
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

os.environ["TRANSFORMERS_NO_TF"] = "1"      # see eval_quality_nr.py
os.environ["USE_TF"] = "0"
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

OUT = os.path.join(PROJ, "experiments", "results", "clone_quality.csv")
CLONE_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware", "clones")
WAV_DIR = os.path.join(PROJ, "experiments", "results", "purifier_aware")
ALIGN = os.path.join(PROJ, "external", "De-AntiFake", "data",
                     "librispeech_alignments",
                     "test_clean_phoneme_alignments.parquet")
NISQA_DIR = os.path.join(PROJ, "external", "NISQA")
SR = 16000
CROP_SEC = 4.0


def load_for_scoring(path, match_lufs=-23.0):
    """Read, mono-mix, crop, and (by default) loudness-match before scoring.

    LEVEL IS CONFOUNDED WITH CONDITION HERE, exactly as utils/loudness.py says
    for the listening test: the protection is an SNR-constrained perturbation,
    so a protected file is quieter than its original by construction, and the
    clean original read straight off the FLAC is louder again than anything
    load_flac wrote. NISQA reports a `loud` dimension and UTMOS is not level
    invariant, so scoring unmatched audio would let a gain difference show up
    as a naturalness difference between conditions.

    Matching every condition to the same integrated loudness removes that,
    and is what the listening-test stimuli already do (-23 LUFS). The measured
    level before matching is kept in the CSV so the gain difference itself
    stays visible rather than being quietly discarded.
    """
    from utils.loudness import integrated_lufs, normalise
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: int(CROP_SEC * SR)]
    if match_lufs is None:
        return (np.ascontiguousarray(y, dtype=np.float32),
                round(float(integrated_lufs(y, sr)), 2), 0.0, False)
    # normalise() returns (scaled, info); info carries the measured input
    # loudness, the gain applied, and whether the peak ceiling clamped it. A
    # clamp means the file did NOT reach the target, so it is carried through to
    # the CSV rather than dropped -- an unreached target is exactly the case
    # where level is still confounded with condition.
    y2, info = normalise(y, sr, target_lufs=match_lufs)
    return (np.ascontiguousarray(y2, dtype=np.float32),
            float(info["lufs_before"]), float(info["gain_db"]),
            bool(info["clamped"]))


# --------------------------------------------------------------- references
def ground_truth_words(align, utt_id, crop=CROP_SEC):
    """Ground-truth words of `utt_id` whose MIDPOINT falls inside the crop.

    Midpoint rather than `end <= crop`: a word straddling the 4-s boundary is
    audible for part of its duration and an ASR may or may not emit it, so
    neither including nor excluding it is clearly right. The midpoint rule is
    stated here, applied to every condition identically, and the count is
    written to the CSV so the choice is auditable.
    """
    if utt_id not in align.index:
        return None
    out = []
    for wd in align.loc[utt_id]["words"]:
        s, e = float(wd["start"]), float(wd["end"])
        w = str(wd["word"]).strip()
        if not w or w in ("<eps>", "sil", "sp", "spn"):
            continue
        if 0.5 * (s + e) < crop:
            out.append(w)
    return out


# ------------------------------------------------------------------ scorers
def load_asr(device, name):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained(name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        name, torch_dtype=dtype).to(device).eval()
    forced = None
    if not name.endswith(".en"):
        forced = proc.get_decoder_prompt_ids(language="english",
                                             task="transcribe")

    @torch.no_grad()
    def transcribe(wav_np):
        feats = proc(wav_np, sampling_rate=SR,
                     return_tensors="pt").input_features.to(device, dtype)
        kw = dict(max_new_tokens=128)
        if forced is not None:
            kw["forced_decoder_ids"] = forced
        ids = model.generate(feats, **kw)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]

    return transcribe, model


def load_utmos(device):
    """UTMOS22 via torch.hub (already cached here). torch 2.9's
    _validate_not_a_forked_repo raises KeyError('Authorization') on
    unauthenticated requests; it only guards against forks."""
    import torch.hub as hub
    orig = getattr(hub, "_validate_not_a_forked_repo", None)
    hub._validate_not_a_forked_repo = lambda *a, **k: True
    try:
        m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True, verbose=False)
        return m.to(device).eval()
    finally:
        if orig is not None:
            hub._validate_not_a_forked_repo = orig


def run_nisqa(files, workdir, python=None):
    """NISQA v2 (Mittag et al., Interspeech 2021) over a flat directory.

    Vendored at external/NISQA (MIT, commit fe84f0f2). Invoked as a subprocess
    because its entry point is directory-driven and loads its own model; that
    also keeps its torch.load of a v1-era checkpoint out of this process.

    `files` maps a flat basename -> source path. Returns a DataFrame indexed by
    that basename with mos_pred and the four NISQA dimensions, or None if NISQA
    is unavailable.
    """
    weights = os.path.join(NISQA_DIR, "weights", "nisqa.tar")
    if not os.path.isfile(weights):
        print(f"[Q] NISQA not vendored at {NISQA_DIR}; skipping its columns")
        return None
    # A FRESH input directory every time. NISQA scores whatever is in the
    # directory, so a leftover file from an earlier attempt would silently add
    # rows -- and if it were an unmatched copy of a file scored here, the merge
    # would attach the wrong loudness condition to a row.
    d_in = os.path.join(workdir, "nisqa_in")
    if os.path.isdir(d_in):
        shutil.rmtree(d_in)
    os.makedirs(d_in, exist_ok=True)
    res = os.path.join(workdir, "NISQA_results.csv")
    if os.path.isfile(res):
        os.remove(res)
    for flat, src in files.items():
        shutil.copyfile(src, os.path.join(d_in, flat))
    cmd = [python or sys.executable, "run_predict.py", "--mode", "predict_dir",
           "--pretrained_model", os.path.join("weights", "nisqa.tar"),
           "--data_dir", d_in, "--output_dir", workdir,
           "--num_workers", "0", "--bs", "4"]
    print(f"[Q] NISQA on {len(files)} files ...", flush=True)
    r = subprocess.run(cmd, cwd=NISQA_DIR, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(res):
        # Loud, and the whole message. A silent None here is how the first run
        # produced a quality table with no NISQA columns and no explanation.
        log = os.path.join(workdir, "nisqa_failed.log")
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            fh.write("cmd: %s\ncwd: %s\nexit: %s\n\n--- stdout ---\n%s"
                     "\n\n--- stderr ---\n%s"
                     % (cmd, NISQA_DIR, r.returncode, r.stdout, r.stderr))
        raise SystemExit(
            "[Q] NISQA failed (exit %s). Full output in %s.\n"
            "    stderr tail: %s\n"
            "    Re-run with --no-nisqa to produce the table without it, and "
            "say so in the write-up."
            % (r.returncode, log, r.stderr[-800:]))
    df = pd.read_csv(res)
    # NISQA has written `deg` as a bare basename and as a full path across
    # versions; normalise so the merge key is always the basename we chose.
    df["deg"] = df["deg"].astype(str).map(os.path.basename)
    df = df.set_index("deg")
    keep = [c for c in ["mos_pred", "noi_pred", "dis_pred", "col_pred",
                        "loud_pred"] if c in df.columns]
    return df[keep].rename(columns={c: "nisqa_" + c.replace("_pred", "")
                                    for c in keep})


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=29, dest="n_files")
    ap.add_argument("--spk-offset", type=int, default=0, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=0, dest="utt_index")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--protected-tags", default="fullnsT_tc_m30_seg",
                    dest="protected_tags",
                    help="saved protected arms to score (council #6a)")
    ap.add_argument("--cloning-csv",
                    default=os.path.join(PROJ, "experiments", "results",
                                         "cloning_tc_m30_seg_qual.csv"),
                    dest="cloning_csv",
                    help="the run that wrote the clones; its source_speaker "
                         "column supplies each clone's WER reference")
    ap.add_argument("--asr", default="openai/whisper-large-v3",
                    help="council #6 asks for whisper-large-v3; pass a smaller "
                         "checkpoint to trade accuracy for VRAM")
    ap.add_argument("--no-nisqa", action="store_true", dest="no_nisqa")
    ap.add_argument("--match-lufs", type=float, default=-23.0, dest="match_lufs",
                    help="integrated loudness every file is matched to before "
                         "scoring, so level is not confounded with condition "
                         "(utils/loudness.py). Pass 0 to disable.")
    ap.add_argument("--workdir", default=os.environ.get("TMPDIR") or
                    os.path.join(PROJ, "experiments", "results", "_qual_tmp"))
    ap.add_argument("--out", default=OUT)
    from utils.provenance import add_prereg_arg, provenance
    add_prereg_arg(ap)
    a = ap.parse_args()
    PROV = provenance(a.prereg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.workdir, exist_ok=True)
    if a.match_lufs == 0:
        a.match_lufs = None
        print("[Q] WARNING: loudness matching OFF. Level is confounded with "
              "condition here; see load_for_scoring.")

    from experiments.librispeech import (list_librispeech, reference_items,
                                         load_flac, cross_session_split)
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    import jiwer

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             a.subset, min_sec=CROP_SEC)
    ref = reference_items(items, min_sec=CROP_SEC, utt_index=a.utt_index)
    ok = set(cross_session_split(items, min_sec=CROP_SEC))
    spks = [s for s in sorted(ref) if s in ok][a.spk_offset:a.spk_offset + a.n_files]

    align = pd.read_parquet(ALIGN).set_index("id")
    norm = EnglishTextNormalizer({})

    def utt_id(spk):
        it = ref[spk]
        return f"{spk}-{it['chapter']}-{it['utt']:04d}"

    def name_of(spk):
        it = ref[spk]
        return f"spk{spk}_{it['chapter']}_{it['utt']:04d}"

    # Who spoke the words in each clone. Read from the run that made them.
    src_of = {}
    if os.path.isfile(a.cloning_csv):
        cdf = pd.read_csv(a.cloning_csv)
        if "source_speaker" in cdf.columns:
            src_of = (cdf.drop_duplicates("speaker")
                      .set_index("speaker")["source_speaker"].to_dict())
    if not src_of:
        # exp_cloning_protection's own rule, as a fallback.
        src_of = {s: spks[(i + 1) % len(spks)] for i, s in enumerate(spks)}
        print("[Q] WARNING: no source_speaker column; re-deriving the "
              "attacker-source mapping from the loop rule")

    # --------------------------------------------------- collect the wav set
    # (condition, speaker) -> (path, wer_reference_speaker)
    work = {}
    for spk in spks:
        nm = name_of(spk)
        src = int(src_of.get(spk, spk))
        work[("original", spk)] = (ref[spk]["path"], spk)
        for tag in [t for t in a.protected_tags.split(",") if t]:
            p = os.path.join(WAV_DIR, f"{nm}_{tag}.wav")
            if os.path.isfile(p):
                work[(f"protected:{tag}", spk)] = (p, spk)
        if os.path.isdir(CLONE_DIR):
            for f in os.listdir(CLONE_DIR):
                if not f.startswith(nm + "__") or not f.endswith("__clone.wav"):
                    continue
                parts = f[len(nm) + 2:-len("__clone.wav")].split("__")
                if len(parts) != 2:
                    continue
                arm, pur = parts
                # The clone speaks the ATTACKER's words, so its reference is
                # the SOURCE speaker's transcript. This is the trap in the
                # module docstring.
                work[(f"clone:{arm}:{pur}", spk)] = (
                    os.path.join(CLONE_DIR, f), src)

    if not work:
        raise SystemExit("[Q] nothing to score")
    n_clone = sum(1 for k in work if k[0].startswith("clone:"))
    print(f"[Q] {len(spks)} speakers | {len(work)} files "
          f"({n_clone} clones) | device={device}")
    print(f"[Q] code_sha={PROV['code_sha']}")

    # ------------------------------------------------------ pass 1: ASR / WER
    # Done first and then unloaded: whisper-large-v3 in fp16 is ~3.1 GB and the
    # MOS predictors plus NISQA do not fit beside it on a 6 GB card.
    print(f"[Q] pass 1/3  ASR {a.asr}", flush=True)
    transcribe, asr_model = load_asr(device, a.asr)
    hyp, refs, levels = {}, {}, {}
    for i, (key, (path, ref_spk)) in enumerate(sorted(work.items(), key=str)):
        y, lufs_before, gain_db, clamped = load_for_scoring(path, a.match_lufs)
        levels[key] = (lufs_before, gain_db, clamped)
        hyp[key] = transcribe(y)
        gt = ground_truth_words(align, utt_id(ref_spk))
        refs[key] = gt
        if (i + 1) % 20 == 0:
            print(f"[Q]   transcribed {i + 1}/{len(work)}", flush=True)
    del asr_model, transcribe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------ pass 2: UTMOS and SQUIM
    print("[Q] pass 2/3  UTMOS + SQUIM", flush=True)
    utmos = load_utmos(device)
    try:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        squim = SQUIM_OBJECTIVE.get_model().to(device).eval()
    except Exception as exc:
        print(f"[Q] SQUIM unavailable ({exc}); continuing without it")
        squim = None

    rows = []
    for key, (path, ref_spk) in sorted(work.items(), key=str):
        cond, spk = key
        y, lufs_before, gain_db, clamped = load_for_scoring(path, a.match_lufs)
        w = torch.from_numpy(y).reshape(1, -1).float().to(device)
        with torch.no_grad():
            mos = float(utmos(w, SR))
            sq = {}
            if squim is not None:
                st, pe, si = squim(w)
                sq = dict(squim_stoi=round(float(st), 4),
                          squim_pesq=round(float(pe), 4),
                          squim_sisdr=round(float(si), 3))
        gt = refs[key]
        h = norm(hyp[key])
        r = norm(" ".join(gt)) if gt else ""
        wer = (float(jiwer.wer(r, h)) if r and h else float("nan"))
        rows.append(dict(
            condition=cond, speaker=spk, wer_reference_speaker=ref_spk,
            is_clone=cond.startswith("clone:"),
            utmos=round(mos, 3), **sq,
            lufs_before=round(lufs_before, 2), gain_db=round(gain_db, 2),
            lufs_clamped=clamped, matched_lufs=a.match_lufs,
            wer=round(wer, 4) if wer == wer else np.nan,
            n_ref_words=(len(gt) if gt else 0), n_hyp_words=len(h.split()),
            flat=f"{cond.replace(':', '-')}__spk{spk}.wav", path=path, **PROV))

    df = pd.DataFrame(rows)

    # ----------------------------------------------------------- pass 3: NISQA
    if not a.no_nisqa:
        print("[Q] pass 3/3  NISQA", flush=True)
        del utmos
        if squim is not None:
            del squim
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # NISQA reads files, so it must read the LOUDNESS-MATCHED audio, not the
        # originals -- otherwise its loud_pred (and its MOS) would be scored on
        # a different signal than UTMOS was.
        matched_dir = os.path.join(a.workdir, "matched")
        os.makedirs(matched_dir, exist_ok=True)
        flat_to_matched = {}
        for flat, src in zip(df.flat, df.path):
            y, _, _, _ = load_for_scoring(src, a.match_lufs)
            mp = os.path.join(matched_dir, flat)
            sf.write(mp, y, SR)
            flat_to_matched[flat] = mp
        nis = run_nisqa(flat_to_matched, a.workdir)
        if nis is not None:
            df = df.merge(nis, left_on="flat", right_index=True, how="left")

    df.drop(columns=["path"]).to_csv(a.out, index=False)

    cols = [c for c in ["utmos", "nisqa_mos", "nisqa_noi", "nisqa_dis",
                        "nisqa_col", "wer", "squim_pesq", "squim_stoi"]
            if c in df.columns]
    summ = df.groupby("condition")[cols].agg(["mean", "count"]).round(3)
    print("\n" + "=" * 86)
    print("QUALITY -- what we publish, and what the attacker gets out")
    print("UTMOS / NISQA 1-5 higher = more natural; WER vs LibriSpeech ground "
          "truth (MFA-clipped)")
    print("=" * 86)
    print(df.groupby("condition")[cols].mean().round(3).to_string())
    print("\nn per condition:")
    print(df.groupby("condition").size().to_string())

    # The #6d comparison, stated explicitly: protected clone vs unprotected
    # clone, paired per speaker. If protection shows up here as a big
    # naturalness drop, part of the reported protection is damage.
    cl = df[df.is_clone].copy()
    if len(cl):
        cl["arm"] = cl.condition.str.split(":").str[1]
        cl["pur"] = cl.condition.str.split(":").str[2]
        print("\n" + "-" * 86)
        print("#6d  protected clone MINUS unprotected clone, paired per speaker")
        print("-" * 86)
        for pur in sorted(cl.pur.unique()):
            sub = cl[cl.pur == pur]
            arms = [x for x in sorted(sub.arm.unique()) if x != "unprotected"]
            base = sub[sub.arm == "unprotected"].set_index("speaker")
            for arm in arms:
                got = sub[sub.arm == arm].set_index("speaker")
                both = base.index.intersection(got.index)
                if not len(both):
                    continue
                print(f"  purifier={pur}  {arm} vs unprotected  n={len(both)}")
                for m in [c for c in ["utmos", "nisqa_mos", "wer"] if c in cl]:
                    d = (got.loc[both, m] - base.loc[both, m]).dropna()
                    if not len(d):
                        continue
                    print(f"      {m:10s} {d.mean():+7.3f}  "
                          f"(SD {d.std():.3f}, n={len(d)})")
        print("\n  A large NEGATIVE utmos/nisqa delta would mean the protected")
        print("  clone is audibly worse, i.e. part of what the similarity")
        print("  numbers call protection is really damage to the attacker's")
        print("  output. A delta near zero is the result the paper needs.")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
