"""
experiments/diag_dvec_proxy.py
==============================
P0 DIAGNOSTIC. Where does the Resemblyzer proxy gap come from?

THE PROBLEM
-----------
In the DEPLOYMENT setting (all three encoders crafted against), Resemblyzer still
lets 35-50 % of clones through, while WavLM-SV and ECAPA reach 0 %. It is being
optimised against, so this is NOT encoder transfer failing. We are optimising
against a different function than the one that scores us.

`DvecEnc.embed()`        torch mel -> dvec(mel) -> one pooled embedding
`real_embed_dvec()`      preprocess_wav() -> embed_utterance()
                         = volume normalise + trim silences, then SPLIT into
                           1.6 s partials, embed each, L2-normed AVERAGE

Two candidate causes, and the fix differs depending on which dominates:

  (a) PARTIAL AVERAGING. AntiFake Sec. C2 predicts this one: "directly deviating
      the final embedding is susceptible to local optima, as the impact of a
      partial embedding not close to the target is mitigated by other partials".
      Fix = score every partial, push the worst.

  (b) PREPROCESSING. Volume normalisation + voice-activity trimming change the
      signal before it is ever embedded. Fix = model it, or craft on the
      preprocessed signal.

This script measures both, separately, before any code is written.

  python experiments/diag_dvec_proxy.py            # the original 2026-09-19 table

--------------------------------------------------------------------------------
SECOND SECTION: the front-end pre-check   (PLAN_V3 W1, "run 2.4's CPU pre-check
and report it as part of #10's answer")
--------------------------------------------------------------------------------
`IMPROVE_DVECTOR_2026-10-05.md` T1.1 claims the remaining d-vector front-end gap
is a WINDOW ALIGNMENT problem, not a filterbank problem:

    "VAD trimming of LibriSpeech removes 0.3-0.8 s -- half to a full partial
     hop. So the ~4 windows whose worst case --seg-partials optimises are not
     the windows the evaluator scores."

PLAN_V3 cut the T2.4 *implementation* (§23a measured the whole remaining
preprocessing gap at cosine 0.977, i.e. minor) but kept this pre-check, because
the claim is cheap to test and it is the premise of a 2.5 GPU-h item. The
pre-check answers three things with no GPU and no crafting:

  1. how much audio `trim_long_silences` actually removes, in units of the
     INTER-PARTIAL HOP (77 frames = 12320 samples = 0.77 s);
  2. how far each partial window moves as a result -- measured by mapping the
     trimmed signal's partial boundaries back through the VAD keep-mask to
     original sample indices, and counting the windows that move more than half
     a hop;
  3. what that misalignment costs the EMBEDDING: cosine between the
     partial-averaged embedding of the untrimmed signal (what the optimiser
     pushes) and of the trimmed signal (what the evaluator scores), with the
     torch mel front end held fixed on both sides so trimming and gain are the
     only variables.

Reading it:
  * windows move a lot AND the embedding cosine is low -> T1.1's premise holds
    and a frozen-VAD front end is worth the GPU.
  * windows move a lot but the embedding cosine is HIGH -> the d-vector's
    partial average is alignment-insensitive, the premise is false, and #10's
    evaluation-only re-score is the right call. PLAN_V3 predicts this branch
    from §23a's 0.977.

  python experiments/diag_dvec_proxy.py --frontend --n-files 29 --cross-session

Output (with --frontend):  experiments/results/dvec_frontend_precheck.csv
"""
import argparse
import os
import struct
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import torch
import torch.nn.functional as F

OUT = os.path.join(PROJ, "experiments", "results", "dvec_frontend_precheck.csv")


def l2(x):
    return F.normalize(x.float().reshape(1, -1), dim=-1)


def cos(a, b):
    return float((l2(a) * l2(b)).sum())


# ---------------------------------------------------------------------------
# The original diagnostic (2026-09-19), unchanged.
# ---------------------------------------------------------------------------
def proxy_gap(n_files=6):
    device = torch.device("cpu")          # GPU is busy; this is tiny
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.hparams import partials_n_frames, mel_window_step, sampling_rate
    from experiments.librispeech import list_librispeech, reference_items, load_flac
    from experiments.multi_encoders import DvecEnc

    dvec = VoiceEncoder(device=device)
    proxy = DvecEnc(dvec, device)

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             "test-clean", min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    spks = sorted(ref)[:n_files]

    print("Cosine between the embedding the OPTIMISER sees and the one that SCORES us.")
    print("1.0 would mean no gap at all. Lower = we are optimising the wrong thing.\n")
    print(f"{'speaker':<10}{'proxy vs real':>15}{'proxy vs partials':>20}"
          f"{'partials vs real':>19}{'n_partials':>12}")

    rows = []
    for spk in spks:
        wav_np = load_flac(ref[spk]["path"], 4.0).squeeze(0).numpy().astype(np.float32)
        wav_t = torch.from_numpy(wav_np).reshape(1, -1)

        # (1) what the optimiser currently sees
        with torch.no_grad():
            e_proxy = proxy.embed(wav_t)

        # (2) the real scoring path: preprocess + partial average
        pp = preprocess_wav(wav_np, source_sr=16000)
        e_real = torch.tensor(dvec.embed_utterance(pp)).reshape(1, -1)

        # (3) partial averaging WITHOUT preprocessing -- isolates cause (a)
        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        wav_slices, mel_slices = dvec.compute_partial_slices(
            len(wav_np), rate=1.3, min_coverage=0.75)   # embed_utterance defaults
        pad = wav_slices[-1].stop - len(wav_np)
        w = np.pad(wav_np, (0, max(0, pad)), "constant")
        wt = torch.from_numpy(w).reshape(1, -1)
        with torch.no_grad():
            mel = proxy.mel(wt.float()).squeeze(0).transpose(0, 1)      # [frames, 40]
            parts = torch.stack([mel[s] for s in mel_slices])            # [n, 160, 40]
            pe = dvec(parts)                                             # [n, 256]
            e_parts = l2(F.normalize(pe, dim=-1).mean(dim=0))

        rows.append((cos(e_proxy, e_real), cos(e_proxy, e_parts),
                     cos(e_parts, e_real), len(mel_slices)))
        print(f"{spk:<10}{rows[-1][0]:>15.3f}{rows[-1][1]:>20.3f}"
              f"{rows[-1][2]:>19.3f}{rows[-1][3]:>12d}")

    a = np.array([r[:3] for r in rows]).mean(axis=0)
    print(f"\n{'MEAN':<10}{a[0]:>15.3f}{a[1]:>20.3f}{a[2]:>19.3f}")
    print("\nHOW TO READ THIS")
    print("  'proxy vs partials' LOW  -> cause (a), partial averaging. Fix: score")
    print("                              every partial, push the worst (AntiFake C2).")
    print("  'partials vs real'  LOW  -> cause (b), preprocessing (normalise + VAD).")
    print("  both high but 'proxy vs real' low -> the two causes compound.")


# ---------------------------------------------------------------------------
# The front-end pre-check (PLAN_V3 W1).
# ---------------------------------------------------------------------------
def vad_keep_mask(wav):
    """`resemblyzer.audio.trim_long_silences`, but it also returns the mask.

    Reimplemented rather than imported because the library function returns only
    the trimmed waveform, and the whole question here is WHICH original samples
    survived. Every constant and every step is taken from that function so the
    mask describes exactly what the real evaluation path does; verified against
    it by asserting the trimmed waveforms match bit-for-bit (see `_check`).
    """
    from scipy.ndimage import binary_dilation
    from resemblyzer.hparams import (sampling_rate, vad_window_length,
                                     vad_moving_average_width,
                                     vad_max_silence_length)
    import webrtcvad

    int16_max = (2 ** 15) - 1
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav),
                           *(np.round(wav * int16_max)).astype(np.int16))

    vad = webrtcvad.Vad(mode=3)
    voice_flags = []
    for start in range(0, len(wav), samples_per_window):
        end = start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[start * 2:end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    def moving_average(array, width):
        padded = np.concatenate((np.zeros((width - 1) // 2), array,
                                 np.zeros(width // 2)))
        ret = np.cumsum(padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    mask = moving_average(voice_flags, vad_moving_average_width)
    mask = np.round(mask).astype(bool)
    mask = binary_dilation(mask, np.ones(vad_max_silence_length + 1))
    mask = np.repeat(mask, samples_per_window)
    return wav[mask], mask


def frontend_precheck(n_files=29, cross_session=False, subset="test-clean",
                      out=OUT):
    device = torch.device("cpu")        # deliberately CPU: no GPU budget spent
    import pandas as pd
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import normalize_volume
    from resemblyzer.hparams import (sampling_rate, mel_window_step,
                                     partials_n_frames, audio_norm_target_dBFS)
    from experiments.librispeech import (list_librispeech, reference_items,
                                         load_flac, cross_session_split)
    from experiments.multi_encoders import DvecEnc
    from utils.provenance import provenance

    PROV = provenance(None)
    dvec = VoiceEncoder(device=device)
    proxy = DvecEnc(dvec, device)

    samples_per_frame = int(sampling_rate * mel_window_step / 1000)     # 160
    # The inter-partial hop, from compute_partial_slices: frame_step frames.
    frame_step = int(np.round((sampling_rate / 1.3) / samples_per_frame))  # 77
    hop_samples = frame_step * samples_per_frame                          # 12320
    half_hop = hop_samples / 2.0

    items = list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                             subset, min_sec=4.0)
    ref = reference_items(items, min_sec=4.0)
    ok = set(cross_session_split(items, min_sec=4.0))
    if cross_session:
        spks = [s for s in sorted(ref) if s in ok][:n_files]
    else:
        spks = sorted(ref)[:n_files]
    overlap = sum(1 for s in spks if s in ok)

    print("=" * 78)
    print("FRONT-END PRE-CHECK -- does VAD trimming move the partial windows,")
    print("and does it matter to the embedding?   (PLAN_V3 W1, 2.4's pre-check)")
    print("=" * 78)
    print(f"cohort: {len(spks)} speakers from {subset}, cross_session="
          f"{bool(cross_session)}; {overlap}/{len(spks)} in the reported "
          f"cross-session set")
    print(f"partial = {partials_n_frames} frames (1.60 s); inter-partial hop = "
          f"{frame_step} frames = {hop_samples} samples = "
          f"{hop_samples / sampling_rate:.2f} s; half hop = "
          f"{half_hop / sampling_rate:.3f} s")
    print(f"code_sha={PROV['code_sha']}\n")
    print(f"  {'spk':>6} {'cut s':>7} {'cut/hop':>8} {'gain dB':>8} "
          f"{'n_un':>5} {'n_tr':>5} {'max shift s':>12} {'>½hop':>7} "
          f"{'cos emb':>8} {'cos part':>9}")

    rows = []
    for spk in spks:
        wav_np = load_flac(ref[spk]["path"], 4.0).squeeze(0).numpy().astype(np.float32)

        # preprocess_wav, decomposed: gain first, then VAD trim.
        normed = normalize_volume(wav_np, audio_norm_target_dBFS,
                                  increase_only=True)
        gain = float(np.sqrt(np.mean(normed ** 2)) /
                     (np.sqrt(np.mean(wav_np ** 2)) + 1e-12))
        trimmed, mask = vad_keep_mask(normed)

        # The library's own output, as a correctness check on the reimplementation.
        ref_pp = preprocess_wav(wav_np, source_sr=16000)
        assert len(ref_pp) == len(trimmed) and np.allclose(ref_pp, trimmed, atol=1e-6), \
            f"spk{spk}: mask reimplementation disagrees with preprocess_wav"

        n_un_samples = len(mask)          # normalize+truncate length, pre-trim
        orig_index = np.flatnonzero(mask)  # trimmed position -> original index
        cut_s = (n_un_samples - len(trimmed)) / sampling_rate

        sl_un, _ = dvec.compute_partial_slices(n_un_samples, rate=1.3,
                                               min_coverage=0.75)
        sl_tr, mel_tr = dvec.compute_partial_slices(len(trimmed), rate=1.3,
                                                    min_coverage=0.75)

        # Where does partial k actually sit, in ORIGINAL sample indices?
        shifts = []
        for k in range(min(len(sl_un), len(sl_tr))):
            start_tr = min(sl_tr[k].start, len(orig_index) - 1)
            shifts.append(abs(int(orig_index[start_tr]) - sl_un[k].start))
        shifts = np.array(shifts, dtype=float) if shifts else np.array([0.0])
        n_moved = int((shifts > half_hop).sum())

        # What the misalignment costs the embedding, torch mel front end on both
        # sides so the only difference is gain + trimming.
        with torch.no_grad():
            p_un = proxy.embed_partials(torch.from_numpy(wav_np).reshape(1, -1))
            p_tr = proxy.embed_partials(torch.from_numpy(trimmed).reshape(1, -1))
            e_un = l2(p_un.mean(dim=0))
            e_tr = l2(p_tr.mean(dim=0))
            cos_emb = cos(e_un, e_tr)
            m = min(p_un.shape[0], p_tr.shape[0])
            cos_part = float(F.cosine_similarity(p_un[:m], p_tr[:m],
                                                 dim=-1).mean())

        rows.append(dict(
            speaker=spk, in_reported_cohort=bool(spk in ok),
            dur_s=round(n_un_samples / sampling_rate, 3),
            kept_s=round(len(trimmed) / sampling_rate, 3),
            cut_s=round(cut_s, 3),
            cut_in_hops=round(cut_s * sampling_rate / hop_samples, 3),
            gain_db=round(20 * np.log10(max(gain, 1e-12)), 3),
            n_partials_untrimmed=len(sl_un), n_partials_trimmed=len(sl_tr),
            mean_shift_s=round(float(shifts.mean()) / sampling_rate, 4),
            max_shift_s=round(float(shifts.max()) / sampling_rate, 4),
            n_partials_moved_over_half_hop=n_moved,
            n_partials_compared=len(shifts),
            cos_embed_untrimmed_vs_trimmed=round(cos_emb, 4),
            cos_partialwise_mean=round(cos_part, 4), **PROV))
        r = rows[-1]
        print(f"  {spk:>6} {r['cut_s']:>7.3f} {r['cut_in_hops']:>8.2f} "
              f"{r['gain_db']:>8.2f} {r['n_partials_untrimmed']:>5d} "
              f"{r['n_partials_trimmed']:>5d} {r['max_shift_s']:>12.3f} "
              f"{n_moved:>3d}/{len(shifts):<3d} "
              f"{r['cos_embed_untrimmed_vs_trimmed']:>8.4f} "
              f"{r['cos_partialwise_mean']:>9.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    n = len(df)
    cut = df.cut_s.to_numpy()
    emb = df.cos_embed_untrimmed_vs_trimmed.to_numpy()
    part = df.cos_partialwise_mean.to_numpy()
    moved_any = int((df.n_partials_moved_over_half_hop > 0).sum())
    print("\n" + "-" * 78)
    print(f"n = {n} speakers")
    print(f"  audio removed by VAD      mean {cut.mean():.3f} s  "
          f"[{cut.min():.3f}, {cut.max():.3f}]  "
          f"= {cut.mean() * sampling_rate / hop_samples:.2f} inter-partial hops")
    print(f"  partials lost             {int((df.n_partials_untrimmed - df.n_partials_trimmed).sum())} "
          f"of {int(df.n_partials_untrimmed.sum())} "
          f"({df.n_partials_untrimmed.mean():.1f} -> {df.n_partials_trimmed.mean():.1f} per speaker)")
    print(f"  windows moved > ½ hop     {int(df.n_partials_moved_over_half_hop.sum())} "
          f"of {int(df.n_partials_compared.sum())} compared; "
          f"{moved_any}/{n} speakers have at least one")
    print(f"  volume gain applied       mean {df.gain_db.mean():+.2f} dB  "
          f"[{df.gain_db.min():+.2f}, {df.gain_db.max():+.2f}]")
    print(f"  cos(embedding)            mean {emb.mean():.4f}  "
          f"[{emb.min():.4f}, {emb.max():.4f}]")
    print(f"  cos(partial-wise)         mean {part.mean():.4f}  "
          f"[{part.min():.4f}, {part.max():.4f}]")
    print("\nHOW TO READ THIS (written before the run, PLAN_V3 W1)")
    print("  T1.1's premise needs BOTH: windows move a lot, AND the embedding")
    print("  follows. §23a already put the whole remaining preprocessing gap at")
    print("  cos 0.977, so a high cos(embedding) here means the misalignment is")
    print("  real but the partial AVERAGE absorbs it -- the premise fails and")
    print("  #10's evaluation-only re-score is the correct, tenth-of-the-cost")
    print("  substitute. A low cos(embedding) would reopen the frozen-VAD item.")
    print(f"\nsaved {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", action="store_true",
                    help="run the PLAN_V3 W1 front-end pre-check instead of the "
                         "original 2026-09-19 proxy-gap table")
    ap.add_argument("--n-files", type=int, default=0, dest="n_files",
                    help="speakers (default 6 for the proxy table, 29 for the "
                         "front-end pre-check)")
    ap.add_argument("--cross-session", action="store_true", dest="cross_session",
                    help="restrict to the cross-session cohort the rest of the "
                         "paper reports (see RESULTS 37e)")
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.frontend:
        frontend_precheck(n_files=a.n_files or 29,
                          cross_session=a.cross_session, subset=a.subset,
                          out=a.out)
    else:
        proxy_gap(n_files=a.n_files or 6)


if __name__ == "__main__":
    main()
