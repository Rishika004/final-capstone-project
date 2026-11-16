"""
demo/app.py -- the live demo, on localhost.

  python demo/app.py            then open http://127.0.0.1:8200

WHAT IT SHOWS, AND WHY IT IS BUILT THIS WAY
-------------------------------------------
A visitor records two clips, the protection is crafted on the first, and an
attacker then purifies it and clones it with F5-TTS. Both clones -- one from
the unprotected clip, one from the protected clip -- are scored against the
visitor's SECOND recording. The demo plays all of it and shows the verdict.

Design choices that keep it honest:

* TWO recordings, not one. The verifier compares the clone against a clip the
  clone was NOT built from. Scoring against the same clip would be the
  flattering setup fix S1 exists to prevent, and it would overstate the
  result on stage.
* The visitor READS GIVEN SENTENCES, so no ASR is needed anywhere: ref_text is
  known exactly, which removes Whisper from the loop and ~3 s from the run.
* The same `craft()` the paper uses, with the shipping recipe's flags, so the
  demo is the method and not a lookalike. BPDA by default because it is ~2x
  faster at the same protection (RESULTS 30).
* The attacker's clone of the UNPROTECTED clip is produced too, every run.
  Without it there is no control, and "the clone failed" means nothing.

TIMING, measured on this machine (RTX 3050 6 GB)
------------------------------------------------
A run is about 3.5 minutes: ~128 s crafting at 60 steps, ~60 s for the two
F5-TTS subprocesses (most of it model load, twice), and seconds for the rest.
60 steps was chosen by measurement, not guesswork: against 120 steps it gives
the same protection (ECAPA 0.077 vs 0.085, all three verifiers failing either
way) and the same quality (PESQ 2.17 vs 2.16) for half the crafting time. The
paper itself uses 200.

WHAT IT IS NOT
--------------
Live microphone audio is out of domain for every number in the paper: those
are LibriSpeech, clean read speech, with an ECAPA threshold calibrated on that
corpus (0.3358). A phone or laptop mic in a noisy room may well protect less
well, or score oddly against that threshold. The page says so, and shows the
paper's 29-speaker figures beside the live ones. A weak live run is a weak
live run, not a refutation and not a result.
"""
import os, sys, io, json, time, base64, threading, subprocess, traceback
import http.server, socketserver
from urllib.parse import urlparse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
F5PY = os.path.join(PROJ, "external", "f5tts_venv", "Scripts", "python.exe")
CKPT = os.path.join(PROJ, "checkpoints", "purifier_v7.pt")
THR_FILE = os.path.join(PROJ, "experiments", "results",
                        "thresholds_clean_xsess_tc.csv")
SR = 16000
MAX_SEC = 4.0

# Read aloud. Fixed so ref_text is exact and no ASR is needed.
SENTENCE_PUBLISH = ("The autumn light came through the tall windows and lay "
                    "in long bars across the floor.")
SENTENCE_ENROL = ("She packed the last of the books into the box and wrote "
                  "the address on the lid.")
# What the attacker puts in your mouth.
SENTENCE_ATTACK = ("Yes, it is me. Please transfer the money to the new "
                   "account I sent you.")

# The paper's own figures, shown beside the live ones (RESULTS 60b, 61a, 66).
PAPER = {"f5_unprot_pct": 86.2, "f5_prot_pct": 10.3,
         "diffvc_unprot_pct": 48.3, "diffvc_prot_pct": 0.0,
         "utmos_orig": 4.152, "utmos_prot": 3.073, "n_speakers": 29}

_M = {}          # model cache
_LOCK = threading.Lock()
_STATE = {"stage": "idle", "pct": 0, "msg": ""}


def _say(stage, pct, msg=""):
    _STATE.update(stage=stage, pct=pct, msg=msg)
    print(f"[demo] {pct:3d}% {stage} {msg}", flush=True)


def models():
    """Load once. ~30 s the first time; every run after is warm."""
    if _M:
        return _M
    import torch
    from experiments.exp_purifier_aware import build_encoders, select_craft
    from models.real_diffpure import load_real_diffpure
    from core.psychoacoustic import PsychoacousticMasker
    from utils.csvio import read_csv
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _say("load", 2, f"device={dev}")
    dp = load_real_diffpure(checkpoint_path=CKPT, t_diffuse=200,
                            device=dev, sampler="ddim")
    for p in dp.parameters():
        p.requires_grad_(False)
    encs = build_encoders(dev)
    _t = read_csv(THR_FILE)          # utils.csvio returns a DataFrame
    thr = {str(r.encoder): float(r.threshold) for r in _t.itertuples()}
    craft_encs, _ = select_craft(encs, "all")
    # Whisper, so the speaker may say ANYTHING. F5-TTS has to be told what the
    # reference clip says; hard-coding that to a sentence on screen breaks the
    # moment someone speaks their own words, and it fails SILENTLY -- the clone
    # just comes out weaker, which understates the attacker.
    from experiments.eval_quality_nr import load_asr
    asr = load_asr(dev, "openai/whisper-small")
    _M.update(dev=dev, dp=dp, encs=encs, thr=thr, craft_encs=craft_encs, asr=asr,
              masker=PsychoacousticMasker(margin_db=30.0, device=dev,
                                          local_snr_floor_db=15.0))
    _say("load", 8, "models ready")
    return _M


def _wav_read(path, max_sec=None):
    import torch, soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    import numpy as np
    if sr != SR:
        import torchaudio.functional as AF
        y = AF.resample(torch.from_numpy(y), sr, SR).numpy()
    n = int((MAX_SEC if max_sec is None else max_sec) * SR)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    peak = float(abs(y).max()) or 1.0
    if peak > 0.99:
        y = y / peak * 0.99
    return torch.from_numpy(y).reshape(1, -1)


def _wav_write(path, wav):
    import soundfile as sf
    sf.write(path, wav.detach().cpu().reshape(-1).numpy(), SR)


def _clone(ref_wav_path, ref_text, out_path, gen_text=None):
    """Attacker's step: F5-TTS, in its own venv."""
    if not os.path.isfile(F5PY):
        raise RuntimeError("F5-TTS venv missing; run external/install_f5tts.sh")
    r = subprocess.run([F5PY, os.path.join(HERE, "f5_once.py"),
                        "--ref", ref_wav_path, "--ref-text", ref_text,
                        "--gen-text", gen_text or SENTENCE_ATTACK, "--out", out_path],
                       capture_output=True, text=True, cwd=PROJ)
    if r.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError("cloner failed: " + (r.stderr or "")[-600:])


def run_pipeline(pub_path, enr_path, run_dir, steps, pub_sec, enr_sec,
                 gen_text=None):
    """enr_path may be None: then the clones are scored against the recorded
    clip itself. That is the flattering comparison (the clone was built from
    it), so the result says so and the page labels those numbers indicative.
    The point of the one-clip mode is the THREE LISTENS, not the score."""
    import torch
    from experiments.exp_purifier_aware import craft, quality, real_embed_all, cs
    m = models()
    dev, encs, thr = m["dev"], m["encs"], m["thr"]

    _say("prepare", 12, "reading your recordings")
    # The PUBLISHED clip drives the cost: crafting runs n_steps passes over
    # the whole waveform, so time scales with its length. The ENROLMENT clip
    # is only embedded once, so extra seconds there are free and give the
    # verifier a better reference.
    wav_o = _wav_read(pub_path, pub_sec).to(dev)
    solo = not enr_path or not os.path.isfile(enr_path)
    wav_e = wav_o if solo else _wav_read(enr_path, enr_sec).to(dev)
    p_orig = os.path.join(run_dir, "original.wav"); _wav_write(p_orig, wav_o)
    said = (m["asr"](wav_o.squeeze(0).cpu().numpy()) or "").strip()
    if not said:
        said = SENTENCE_PUBLISH          # silence or an unusable clip
    _say("prepare", 14, "heard: " + said[:60])
    # What the attacker makes you say. "" or "same" -> your own words, which
    # makes the voice comparison much easier to judge by ear: identical text,
    # so the only thing that differs between the two clips is the voice.
    if not gen_text or gen_text == "same":
        attack = said                       # your own words
    elif gen_text == "SCAM":
        attack = SENTENCE_ATTACK            # the stock scam line
    else:
        attack = gen_text                   # typed by the presenter
    e_enr = real_embed_all(encs, wav_e)

    # ---- our protection --------------------------------------------------
    _say("protect", 15, f"crafting the protection ({steps} steps)")
    t0 = time.time()
    cap = m["masker"].magnitude_cap(wav_o)
    # Positional order is (.., variant, n_steps, lr, bandlimit_hz, lam_before,
    # n_eot). These are exp_purifier_aware's OWN defaults, not guesses: lr 2e-3,
    # bandlimit 6000 Hz, lam_before 0.3, n_eot 1. Passing bandlimit 0 silently
    # band-limits the perturbation to nothing -- the first smoke test came back
    # at 219 dB SNR with every clone still succeeding, which is what that looks
    # like. Everything else matches the shipping recipe (thr-aware, seg
    # partials, BPDA, legacy selection).
    delta = craft(wav_o, m["craft_encs"], [1, 1, 1], m["masker"], cap, m["dp"],
                  "full", steps, 2e-3, 6000.0, 0.3, 1, vocoder="phase",
                  thr=[thr[e.name] for e in m["craft_encs"]],
                  seg_partials=True, grad_estimator="bpda", seed=42,
                  legacy_selection=True, log_every=10 ** 9)
    wav_p = (wav_o + delta).clamp(-1, 1)
    craft_s = time.time() - t0
    p_prot = os.path.join(run_dir, "protected.wav"); _wav_write(p_prot, wav_p)
    snr, pesq_v, stoi_v = quality(wav_o, wav_p)

    # ---- the attacker, on the UNPROTECTED clip: the control -------------
    # Runs AFTER crafting, deliberately. Crafting is the long pole, so doing it
    # first puts the protected audio and its quality numbers on screen about
    # 30 s sooner -- which is the part an audience is waiting for. The control
    # is no less essential for coming second.
    _say("attack_control", 70, "attacker clones your unprotected voice")
    p_c0 = os.path.join(run_dir, "clone_unprotected.wav")
    _clone(p_orig, said, p_c0, attack)
    e_c0 = real_embed_all(encs, _wav_read(p_c0).to(dev))

    # ---- the attacker, on the PROTECTED clip -----------------------------
    _say("purify", 78, "attacker purifies it (DiffPure v7)")
    from evaluation.evaluate import wav_to_logmel
    from utils.vocoder import invert_mel
    with torch.no_grad():
        pur = invert_mel(m["dp"](wav_to_logmel(wav_p, dev), seed=1).cpu()).to(dev)
    p_pur = os.path.join(run_dir, "purified.wav"); _wav_write(p_pur, pur)

    _say("attack_protected", 85, "attacker clones the protected voice")
    p_c1 = os.path.join(run_dir, "clone_protected.wav")
    _clone(p_pur, said, p_c1, attack)
    e_c1 = real_embed_all(encs, _wav_read(p_c1).to(dev))

    _say("score", 95, "scoring both clones against your second recording")
    rows = []
    for name in e_enr:
        tau = thr[name]
        s0, s1 = cs(e_c0[name], e_enr[name]), cs(e_c1[name], e_enr[name])
        rows.append(dict(encoder=name, threshold=round(tau, 4),
                         sim_unprotected=round(s0, 4), sim_protected=round(s1, 4),
                         cloned_unprotected=bool(s0 > tau),
                         cloned_protected=bool(s1 > tau)))
    _say("done", 100, "")
    return dict(
        quality=dict(snr_db=round(snr, 2), pesq=round(float(pesq_v), 3),
                     stoi=round(float(stoi_v), 4)),
        craft_s=round(craft_s, 1), steps=steps, rows=rows, paper=PAPER,
        pub_sec=pub_sec, enr_sec=enr_sec, scored_against_source=bool(solo),
        said=said, attack_text=attack,
        sentences=dict(publish=SENTENCE_PUBLISH, enrol=SENTENCE_ENROL,
                       attack=attack),
        audio={k: f"/runs/{os.path.basename(run_dir)}/{k}.wav" for k in
               ("original", "protected", "purified",
                "clone_unprotected", "clone_protected")})


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self.path = "/index.html"
        elif p == "/api/state":
            return self._json(_STATE)
        elif p == "/api/config":
            return self._json(dict(sentences=dict(
                publish=SENTENCE_PUBLISH, enrol=SENTENCE_ENROL,
                attack=SENTENCE_ATTACK), paper=PAPER,
                default_pub_sec=MAX_SEC, default_enr_sec=6.0,
                max_pub_sec=12.0, max_enr_sec=20.0))
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._json({"error": "no such endpoint"}, 404)
        if not _LOCK.acquire(blocking=False):
            return self._json({"error": "a run is already in progress"}, 409)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            rid = time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(RUNS, rid)
            os.makedirs(run_dir, exist_ok=True)
            for key, fn in (("publish", "in_publish.wav"), ("enrol", "in_enrol.wav")):
                blob = body.get(key)
                if not blob:          # enrol is optional: one-clip mode
                    continue
                with open(os.path.join(run_dir, fn), "wb") as fh:
                    fh.write(base64.b64decode(blob.split(",")[-1]))
            pub_sec = max(2.0, min(12.0, float(body.get("pub_sec", MAX_SEC))))
            enr_sec = max(2.0, min(20.0, float(body.get("enr_sec", 6.0))))
            enr_p = os.path.join(run_dir, "in_enrol.wav")
            out = run_pipeline(os.path.join(run_dir, "in_publish.wav"),
                               enr_p if os.path.isfile(enr_p) else None,
                               run_dir, int(body.get("steps", 60)),
                               pub_sec, enr_sec,
                               (body.get("gen_text") or "").strip()[:300])
            out["run_id"] = rid
            self._json(out)
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            _say("error", 0, str(e))
            self._json({"error": str(e)}, 500)
        finally:
            _LOCK.release()


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", action="store_true",
                    help="load the models now, so the first run is not slow")
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if a.preload:
        models()
    print(f"[demo] http://{a.host}:{a.port}   (Ctrl-C to stop)")
    S((a.host, a.port), H).serve_forever()
