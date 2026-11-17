"""
echosec/app.py -- the EchoSec site and its protection service.

  python echosec/app.py --preload      then  http://127.0.0.1:8300

WHAT THIS IS
------------
A product-shaped front end over the same method the paper measures. The service
is deliberately narrow:

  POST /api/protect   audio in (recorded or uploaded) -> protected audio out
  POST /api/verify    optional: run the attacker on that result, both ways

Protecting is the product. The attack is a demonstration you can ask for, not
part of the flow, because a person who wants a protected file should not have
to wait for a clone to be generated first.

DELIBERATE CHOICES
------------------
* Nothing leaves the machine. No upload to a third party, no account, no
  telemetry. The page says so and it is true.
* Uploads accept WAV / MP3 / FLAC / OGG, decoded by libsndfile. There is no
  FFmpeg on this machine, so anything it cannot read is refused with a clear
  message rather than a stack trace.
* One job at a time. The GPU holds ~5 GB during crafting on a 6 GB card, so a
  second concurrent run would fail on memory, not queue politely.
* Run output goes OUTSIDE the project directory: it lives in OneDrive, and
  bulk writes there slow every later GPU run by 2-3x.

HONESTY
-------
The site states the quality cost and the measured ceiling (72.4 % -> 72.4 %
against a held-out verifier behind a foreign purifier) on the page itself, not
in a footnote. A protection tool that reports only its wins teaches a user
nothing about when to trust it.
"""
import os, sys, io, json, time, base64, threading, subprocess, traceback, tempfile
import http.server, socketserver
from urllib.parse import urlparse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("ECHOSEC_RUNS") or os.path.join(
    tempfile.gettempdir(), "echosec_runs")
F5PY = os.path.join(PROJ, "external", "f5tts_venv", "Scripts", "python.exe")
CKPT = os.path.join(PROJ, "checkpoints", "purifier_v7.pt")
THR_FILE = os.path.join(PROJ, "experiments", "results",
                        "thresholds_clean_xsess_tc.csv")
SR = 16000
STEPS, STRIDE = 60, 5          # see demo/app.py for why these, and not 200/1
ATTACK_LINE = ("Yes, it is me. Please transfer the money to the new account "
               "I sent you.")

_M, _LOCK = {}, threading.Lock()
_STATE = {"stage": "idle", "pct": 0, "msg": ""}


def _say(stage, pct, msg=""):
    _STATE.update(stage=stage, pct=pct, msg=msg)
    print(f"[echosec] {pct:3d}% {stage} {msg}", flush=True)


def models():
    if _M:
        return _M
    import torch
    from experiments.exp_purifier_aware import build_encoders, select_craft
    from models.real_diffpure import load_real_diffpure
    from core.psychoacoustic import PsychoacousticMasker
    from utils.csvio import read_csv
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _say("load", 2, str(dev))
    dp = load_real_diffpure(checkpoint_path=CKPT, t_diffuse=50,
                            device=dev, sampler="ddim")
    dp_atk = load_real_diffpure(checkpoint_path=CKPT, t_diffuse=200,
                                device=dev, sampler="ddim")
    for m in (dp, dp_atk):
        for p in m.parameters():
            p.requires_grad_(False)
    encs = build_encoders(dev)
    t = read_csv(THR_FILE)
    thr = {str(r.encoder): float(r.threshold) for r in t.itertuples()}
    from experiments.eval_quality_nr import load_asr
    _M.update(dev=dev, dp=dp, dp_atk=dp_atk, encs=encs, thr=thr,
              craft_encs=select_craft(encs, "all")[0],
              asr=load_asr(dev, "openai/whisper-small"),
              masker=PsychoacousticMasker(margin_db=30.0, device=dev,
                                          local_snr_floor_db=15.0))
    _say("load", 8, "ready")
    return _M


def _read(path, max_sec):
    """Decode whatever libsndfile can read; refuse the rest in plain words."""
    import torch, numpy as np, soundfile as sf
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        raise ValueError("That file could not be read. WAV, MP3, FLAC and OGG "
                         "work; some phone recordings (m4a) do not.")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import torchaudio.functional as AF
        y = AF.resample(torch.from_numpy(y), sr, SR).numpy()
    if len(y) < SR:
        raise ValueError("That clip is under a second. Four seconds or more "
                         "gives the protection something to work with.")
    n = int(max_sec * SR)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    peak = float(abs(y).max()) or 1.0
    if peak > 0.99:
        y = y / peak * 0.99
    return torch.from_numpy(y).reshape(1, -1)


def _write(path, wav):
    import soundfile as sf
    sf.write(path, wav.detach().cpu().reshape(-1).numpy(), SR)


def protect(in_path, run_dir, sec):
    import torch
    from experiments.exp_purifier_aware import craft, quality
    m = models()
    dev = m["dev"]
    _say("prepare", 12, "")
    wav_o = _read(in_path, sec).to(dev)
    p_orig = os.path.join(run_dir, "original.wav"); _write(p_orig, wav_o)
    said = (m["asr"](wav_o.squeeze(0).cpu().numpy()) or "").strip()

    _say("protect", 20, f"{STEPS} steps")
    t0 = time.time()
    cap = m["masker"].magnitude_cap(wav_o)
    delta = craft(wav_o, m["craft_encs"], [1, 1, 1], m["masker"], cap, m["dp"],
                  "full", STEPS, 2e-3, 6000.0, 0.3, 1, vocoder="phase",
                  thr=[m["thr"][e.name] for e in m["craft_encs"]],
                  seg_partials=True, grad_estimator="bpda", seed=42,
                  legacy_selection=True, craft_stride=STRIDE,
                  log_every=10 ** 9)
    wav_p = (wav_o + delta).clamp(-1, 1)
    craft_s = time.time() - t0
    p_prot = os.path.join(run_dir, "protected.wav"); _write(p_prot, wav_p)

    _say("quality", 92, "")
    snr, pesq_v, stoi_v = quality(wav_o, wav_p)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"said": said, "sec": sec}, fh)
    _say("done", 100, "")
    rid = os.path.basename(run_dir)
    return dict(run_id=rid, craft_s=round(craft_s, 1),
                quality=dict(snr_db=round(snr, 2), pesq=round(float(pesq_v), 3),
                             stoi=round(float(stoi_v), 4)),
                audio={"original": f"/runs/{rid}/original.wav",
                       "protected": f"/runs/{rid}/protected.wav"})


def verify(run_dir):
    """The optional demonstration: clone both versions, score both."""
    import torch
    from experiments.exp_purifier_aware import real_embed_all, cs
    from evaluation.evaluate import wav_to_logmel
    from utils.vocoder import invert_mel
    m = models(); dev = m["dev"]
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    said = meta.get("said") or "This is a recording of my voice."
    wav_o = _read(os.path.join(run_dir, "original.wav"), meta["sec"]).to(dev)
    wav_p = _read(os.path.join(run_dir, "protected.wav"), meta["sec"]).to(dev)

    _say("purify", 20, "")
    with torch.no_grad():
        pur = invert_mel(m["dp_atk"](wav_to_logmel(wav_p, dev), seed=1).cpu()).to(dev)
    p_pur = os.path.join(run_dir, "purified.wav"); _write(p_pur, pur)

    _say("attack", 35, "")
    c0 = os.path.join(run_dir, "clone_unprotected.wav")
    c1 = os.path.join(run_dir, "clone_protected.wav")
    jobs = [dict(ref=os.path.join(run_dir, "original.wav"), ref_text=said,
                 gen_text=ATTACK_LINE, out=c0),
            dict(ref=p_pur, ref_text=said, gen_text=ATTACK_LINE, out=c1)]
    spec = os.path.join(run_dir, "_jobs.json")
    json.dump(jobs, open(spec, "w", encoding="utf-8"))
    if not os.path.isfile(F5PY):
        raise RuntimeError("the cloner is not installed on this machine")
    r = subprocess.run([F5PY, os.path.join(PROJ, "demo", "f5_once.py"),
                        "--jobs", spec], capture_output=True, text=True, cwd=PROJ)
    if any(not os.path.isfile(j["out"]) for j in jobs):
        raise RuntimeError("the cloner failed: " + (r.stderr or "")[-400:])

    _say("score", 85, "")
    e_ref = real_embed_all(m["encs"], wav_o)
    e0 = real_embed_all(m["encs"], _read(c0, meta["sec"]).to(dev))
    e1 = real_embed_all(m["encs"], _read(c1, meta["sec"]).to(dev))
    rows = []
    for n in e_ref:
        tau = m["thr"][n]
        s0, s1 = cs(e0[n], e_ref[n]), cs(e1[n], e_ref[n])
        rows.append(dict(encoder=n, threshold=round(tau, 4),
                         sim_unprotected=round(s0, 4), sim_protected=round(s1, 4),
                         cloned_unprotected=bool(s0 > tau),
                         cloned_protected=bool(s1 > tau)))
    _say("done", 100, "")
    rid = os.path.basename(run_dir)
    return dict(rows=rows, attack_line=ATTACK_LINE,
                audio={"clone_unprotected": f"/runs/{rid}/clone_unprotected.wav",
                       "clone_protected": f"/runs/{rid}/clone_protected.wav",
                       "purified": f"/runs/{rid}/purified.wav"})


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def log_message(self, *a):
        pass

    def _json(self, o, code=200):
        b = json.dumps(o).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/api/state":
            return self._json(_STATE)
        if p.startswith("/runs/"):
            full = os.path.normpath(os.path.join(RUNS, p[len("/runs/"):]))
            if not full.startswith(os.path.normpath(RUNS)) or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(os.path.getsize(full)))
            self.end_headers()
            with open(full, "rb") as fh:
                self.wfile.write(fh.read())
            return
        if p == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        p = urlparse(self.path).path
        if p not in ("/api/protect", "/api/verify"):
            return self._json({"error": "no such endpoint"}, 404)
        if not _LOCK.acquire(blocking=False):
            return self._json({"error": "Another request is being processed. "
                                        "This runs one at a time."}, 409)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if p == "/api/protect":
                sec = max(2.0, min(12.0, float(body.get("sec", 8))))
                rid = time.strftime("%Y%m%d-%H%M%S")
                run_dir = os.path.join(RUNS, rid); os.makedirs(run_dir, exist_ok=True)
                blob = body.get("audio")
                if not blob:
                    raise ValueError("No audio was sent. Record a clip or "
                                     "choose a file first.")
                try:
                    raw = base64.b64decode(str(blob).split(",")[-1])
                except Exception:
                    raise ValueError("That audio could not be decoded.")
                src = os.path.join(run_dir, "upload.bin")
                with open(src, "wb") as fh:
                    fh.write(raw)
                return self._json(protect(src, run_dir, sec))
            rid = body.get("run_id")
            if not rid:
                raise ValueError("Protect a clip before asking for the demo.")
            run_dir = os.path.join(RUNS, os.path.basename(str(rid)))
            if not os.path.isdir(run_dir):
                return self._json({"error": "that result has expired"}, 404)
            return self._json(verify(run_dir))
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:                                    # noqa: BLE001
            traceback.print_exc(); _say("error", 0, str(e))
            self._json({"error": str(e)}, 500)
        finally:
            _LOCK.release()


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if a.preload:
        models()
    print(f"[echosec] http://{a.host}:{a.port}")
    S((a.host, a.port), H).serve_forever()
