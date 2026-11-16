"""
demo/f5_once.py — clone ONE clip with F5-TTS. Runs in external/f5tts_venv.

The demo's "attacker" step. Kept apart from experiments/f5_clone.py because
that one is CSV-driven for the 261-cell experiment; this takes a single
reference and prints the output path. Same model, same released weights, and
the same torchaudio shim (torchcodec needs FFmpeg, which this machine lacks).

  <venv>/python.exe demo/f5_once.py --ref r.wav --ref-text "..." \
      --gen-text "..." --out clone.wav [--seed 2026]
"""
import os, sys, argparse
import torch, numpy as np, soundfile as sf, torchaudio

SR = 16000


def _shim():
    """torchaudio 2.9 sends load() to torchcodec, whose DLLs want FFmpeg."""
    def _load(path, *a, **k):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr
    try:
        torchaudio.load("")
    except OSError:
        torchaudio.load = _load
        def _save(path, t, sr, *a, **k):
            arr = t.detach().cpu().numpy()
            sf.write(str(path), arr.T if arr.ndim == 2 else arr, int(sr))
        torchaudio.save = _save
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="",
                    help="JSON file: [{ref,ref_text,gen_text,out}, ...]. Both "
                         "clones in ONE process, so the model loads once "
                         "instead of twice -- ~30 s a run.")
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref-text", default="", dest="ref_text")
    ap.add_argument("--gen-text", default="", dest="gen_text")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--model", default="F5TTS_v1_Base")
    a = ap.parse_args()

    _shim()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from f5_tts.api import F5TTS
    f5 = F5TTS(model=a.model, device=device)

    if a.jobs:
        import json
        with open(a.jobs, encoding="utf-8") as fh:
            jobs = json.load(fh)
    else:
        jobs = [dict(ref=a.ref, ref_text=a.ref_text, gen_text=a.gen_text,
                     out=a.out)]

    for j in jobs:
        torch.manual_seed(a.seed); np.random.seed(a.seed % (2 ** 32))
        out = f5.infer(ref_file=j["ref"], ref_text=j["ref_text"],
                       gen_text=j["gen_text"], seed=a.seed,
                       remove_silence=False, show_info=lambda *x, **k: None)
        wav, sr = out[0], out[1]
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != SR:
            import torchaudio.functional as AF
            wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
        sf.write(j["out"], wav, SR)
        print("OK " + j["out"], flush=True)


if __name__ == "__main__":
    main()
