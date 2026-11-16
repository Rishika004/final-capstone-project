"""
experiments/f5_clone.py
=======================
N2 stage B -- generate the clones with F5-TTS.

RUN THIS WITH THE ISOLATED INTERPRETER, NOT THE PROJECT'S:
    external/f5tts_venv/Scripts/python.exe experiments/f5_clone.py

PREREG: paper/PREREG_N2_modern_cloner.md

It deliberately imports NOTHING from this project. Its whole input is
results/f5_prompts.csv plus the wavs that CSV names, both written by
prep_f5_prompts.py. That is what lets F5-TTS keep its own torch/transformers
pins without touching the project's environment (same arrangement as
external/phonepure_venv).

Protocol note (prereg 2.4). DiffVC is voice conversion and takes the attacker's
waveform as content; F5-TTS is TTS and takes text. So:

    DiffVC:  clone = VC (source_wav, Purify(reference))
    F5-TTS:  clone = TTS(gen_text  , Purify(reference), ref_text)

ref_text is Whisper's transcript of the victim's CLEAN crop, held fixed across
arms, so a protection that merely confuses the attacker's ASR earns nothing
here. gen_text comes from the same source-speaker rotation the VC run used.

Sampling is seeded per (speaker, arm, channel) with the same CRC mixing
exp_cloning_protection.py uses for --clone-seed, so each generation is
reproducible on its own and independent of loop order.
"""
import os, sys, argparse, time, zlib, csv

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

import torch
import torchaudio
import soundfile as sf
import numpy as np

SR = 16000


def _install_torchaudio_shim():
    """Route torchaudio.load/save through soundfile.

    torchaudio 2.9 dispatches load() to torchcodec, whose DLLs need FFmpeg,
    which is not installed on this machine -- every F5-TTS generation dies at
    utils_infer.py:403 `audio, sr = torchaudio.load(ref_audio)` with
    "Could not load this library: ...libtorchcodec_image.dll".

    This is the same gap the project already works around twice: see
    utils/tacompat.py and eval_quality_nr.load_asr's docstring. soundfile reads
    these WAVs correctly (verified), and pydub -- which F5-TTS uses for its
    other two audio paths -- handles WAV natively without FFmpeg.

    Patched on the module object because F5-TTS calls `torchaudio.load(...)` as
    an attribute, so the shim is picked up whatever the import order. Returns
    (Tensor[channels, frames], sr), torchaudio's own contract.
    """
    def _load(path, *a, **kw):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    def _save(path, tensor, sample_rate, *a, **kw):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(path), arr, int(sample_rate))

    try:
        torchaudio.load("")
    except OSError:
        torchaudio.load = _load
        torchaudio.save = _save
        print("[f5] torchaudio.load shimmed to soundfile (no FFmpeg on this box)")
    except Exception:
        # Any other error means load() itself is reachable; leave it alone.
        pass
OUT_DIR = os.path.join(PROJ, "experiments", "results", "f5_clones")
PROMPTS = os.path.join(PROJ, "experiments", "results", "f5_prompts.csv")


def _to_sr(wav, sr):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr == SR:
        return wav
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(wav), sr, SR).numpy()


def load_f5(device, model_name, ckpt, vocab):
    """Return gen(ref_wav_path, ref_text, gen_text, seed) -> np.float32 @ SR.

    Prefers f5_tts.api.F5TTS, the package's own high-level entry point: it
    resolves the config, downloads the released checkpoint and vocoder, and
    takes an explicit seed. Using the model exactly as released matters here --
    any reimplementation of the sampler would make the column a measurement of
    our code rather than of the attacker's tool.

    Falls back to the low-level utils_infer path if the api module is absent in
    the installed version.
    """
    try:
        from f5_tts.api import F5TTS
    except Exception as e:                            # noqa: BLE001
        print(f"[f5] f5_tts.api unavailable ({type(e).__name__}: {e}); "
              f"falling back to utils_infer")
        return _load_f5_lowlevel(device, model_name, ckpt, vocab)

    kw = dict(model=model_name, device=device)
    if ckpt:
        kw["ckpt_file"] = ckpt
    if vocab:
        kw["vocab_file"] = vocab
    f5 = F5TTS(**kw)
    print(f"[f5] F5TTS api loaded (model={model_name})")

    def gen(ref_wav_path, ref_text, gen_text, seed):
        # Seed both: the api seeds torch itself, but the reference preprocessing
        # and any vocoder dither are not covered by its seed in every version.
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        out = f5.infer(ref_file=ref_wav_path, ref_text=ref_text,
                       gen_text=gen_text, seed=seed, remove_silence=False,
                       show_info=lambda *a, **k: None)
        wav, sr = out[0], out[1]
        return _to_sr(wav, sr)

    return gen


def _load_f5_lowlevel(device, model_name, ckpt, vocab):
    """Direct use of utils_infer, for versions without f5_tts.api."""
    from f5_tts.infer.utils_infer import (
        load_model, load_vocoder, infer_process, preprocess_ref_audio_text,
    )
    from f5_tts.model import DiT
    from importlib.resources import files
    from omegaconf import OmegaConf

    cfg_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    cfg = OmegaConf.load(cfg_path)
    arch = cfg.model.arch
    vocoder_name = cfg.model.mel_spec.mel_spec_type
    print(f"[f5] config {cfg_path} (vocoder {vocoder_name})")

    vocoder = load_vocoder(vocoder_name=vocoder_name, device=device)
    model = load_model(DiT, arch, ckpt, mel_spec_type=vocoder_name,
                       vocab_file=vocab or "", device=device)

    def gen(ref_wav_path, ref_text, gen_text, seed):
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        rp, rt = preprocess_ref_audio_text(ref_wav_path, ref_text,
                                           show_info=lambda *a, **k: None)
        wav, sr, _ = infer_process(rp, rt, gen_text, model, vocoder,
                                   mel_spec_type=vocoder_name, device=device,
                                   show_info=lambda *a, **k: None)
        return _to_sr(wav, sr)

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--out-dir", default=OUT_DIR, dest="out_dir")
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint path; empty = download the released one")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only the first N rows")
    ap.add_argument("--speakers", default="",
                    help="smoke test: comma-separated speaker ids only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    _install_torchaudio_shim()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[f5] torch {torch.__version__} device={device}")
    if device == "cuda":
        print(f"[f5] {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    with open(args.prompts, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.speakers:
        keep = {s.strip() for s in args.speakers.split(",") if s.strip()}
        rows = [r for r in rows if r["speaker"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[f5] {len(rows)} generations to do")

    os.makedirs(args.out_dir, exist_ok=True)
    # Empty --ckpt means "let F5-TTS fetch the released checkpoint itself",
    # which is what we want: the released weights, resolved by the package.
    print(f"[f5] ckpt {args.ckpt or '(released, downloaded by f5_tts)'}")
    gen = load_f5(device, args.model, args.ckpt, args.vocab)

    t0, n_done, n_skip, n_fail = time.time(), 0, 0, 0
    for i, r in enumerate(rows):
        name, arm, pname = r["name"], r["variant"], r["purifier"]
        out = os.path.join(args.out_dir, f"{name}__{arm}__{pname}__f5.wav")
        if os.path.isfile(out) and not args.overwrite:
            n_skip += 1
            continue
        h = zlib.crc32(f"{r['speaker']}|{arm}|{pname}".encode()) & 0xFFFFFFFF
        seed = (args.seed * 1000003 + h) % (2 ** 31)
        rp = os.path.join(PROJ, r["ref_wav"])
        try:
            wav = gen(rp, r["ref_text"], r["gen_text"], seed)
        except Exception as e:                      # noqa: BLE001
            print(f"  FAIL {name} {arm} {pname}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        sf.write(out, wav, SR)
        n_done += 1
        el = time.time() - t0
        print(f"[f5] {i+1}/{len(rows)} {name} {arm}/{pname} "
              f"{len(wav)/SR:.1f}s audio ({el:.0f}s elapsed, "
              f"{el/max(n_done,1):.1f}s/gen)")

    print(f"\n[f5] done: {n_done} written, {n_skip} already present, "
          f"{n_fail} failed -> {args.out_dir}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
