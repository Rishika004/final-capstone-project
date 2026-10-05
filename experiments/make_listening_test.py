"""
experiments/make_listening_test.py
==================================
Build a self-contained listening test for the two questions no automatic metric
can answer:

  BLOCK A  naturalness  -- "does this sound like natural speech?" (MOS 1-5),
           single stimulus, no reference.
  BLOCK B  identity     -- "is this the same person?" (1-5), original then
           protected. A protection is only usable if the owner can still publish
           it as their own voice, and our speaker encoders cannot judge this:
           disagreeing with them is the method's entire purpose.

Design
------
* 6 held-out speakers x 8 conditions, chosen to span the measured
  quality/protection frontier (see RESULTS_ICASSP.md section 8).
* Two anchors for listener screening, per ITU-T P.808 practice:
    - `original`      : must be rated high (>= 4) on naturalness
    - `anchor_noise`  : original at 0 dB SNR white noise, must be rated low (<= 2)
  A listener who fails either anchor is excluded before analysis.
* Trial order is shuffled per listener from a seed recorded in the results file,
  so every listener sees a different order and the order is reproducible.
* Output is a folder you can zip and send: audio + one HTML page that runs
  entirely offline and exports a CSV at the end.

Run:
  python experiments/make_listening_test.py
  # then open listening_test/index.html in a browser
Analyse returned CSVs:
  python experiments/analyse_listening.py listening_test/responses/*.csv
"""
import sys, io, os, json, argparse, random
if not getattr(sys.stdout, "_nvp_utf8", False):   # guard: re-wrapping a second
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")   # time closes the first wrapper
    sys.stdout._nvp_utf8 = True
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import numpy as np
import torch
import soundfile as sf

from evaluation.evaluate import save_wav
from experiments.librispeech import list_librispeech, per_speaker, load_flac, EVAL_SUBSET
from experiments.exp_parametric_probe import TRANSFORMS
from experiments.exp_purifier_aware import WAV_DIR as ADD_WAV_DIR

OUT_DIR = os.path.join(PROJ, "listening_test")

# (condition id, kind, parameter, human-readable note kept OUT of the UI)
CONDITIONS = [
    ("original",      "clean",     None,              "high anchor"),
    ("anchor_noise",  "noise",     0.0,               "low anchor, 0 dB SNR"),
    ("tilt",          "transform", ("tilt", -6.0),    "free quality, 0 % protection"),
    ("mcadams_095",   "transform", ("mcadams", 0.95), "18 % protection"),
    ("mcadams_085",   "transform", ("mcadams", 0.85), "38 % protection"),
    ("mcadams_080",   "transform", ("mcadams", 0.80), "58 % protection"),
    ("additive_plain", "saved",    "plain_m30_f15",   "70 % protection"),
    ("additive_eot",  "saved",     "full_m30_f15",    "90 % protection"),
]


def build_audio(args):
    device = torch.device("cpu")
    win = torch.hann_window(1024)
    ps = per_speaker(list_librispeech(os.path.join(PROJ, "data", "librispeech"),
                                      EVAL_SUBSET, 4.0), n_utt=args.utt_index + 1)
    spks = sorted(ps)[args.spk_offset:args.spk_offset + args.n_speakers]
    audio_dir = os.path.join(OUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    made, skipped = [], []
    for spk in spks:
        it = ps[spk][args.utt_index]
        name = f"spk{spk}_{it['chapter']}_{it['utt']:04d}"
        wav_o = load_flac(it["path"], 4.0)
        for cid, kind, param, _note in CONDITIONS:
            out = os.path.join(audio_dir, f"{spk}_{cid}.wav")
            if kind == "clean":
                y = wav_o
            elif kind == "noise":
                g = torch.Generator().manual_seed(spk)
                n = torch.randn(wav_o.shape, generator=g)
                n = n / n.pow(2).mean().sqrt() * wav_o.pow(2).mean().sqrt()
                y = (wav_o + n).clamp(-1, 1)
            elif kind == "transform":
                tname, mag = param
                fn = TRANSFORMS[tname][0]
                y = fn(wav_o, mag, win, seed=spk).clamp(-1, 1)[:, : wav_o.shape[-1]]
            else:  # saved additive protection
                p = os.path.join(ADD_WAV_DIR, f"{name}_{param}.wav")
                if not os.path.isfile(p):
                    skipped.append((spk, cid)); continue
                d, _ = sf.read(p, dtype="float32")
                y = torch.from_numpy(d).reshape(1, -1)[:, : wav_o.shape[-1]]
            save_wav(y, out)
            made.append((spk, cid))
    return spks, made, skipped


def build_trials(spks, args):
    """Block A: every speaker x condition, single stimulus.
       Block B: every speaker x condition except `original`, paired with it."""
    a = [{"block": "A", "speaker": int(s), "condition": c, "file": f"audio/{s}_{c}.wav"}
         for s in spks for c, *_ in CONDITIONS]
    b = [{"block": "B", "speaker": int(s), "condition": c,
          "ref": f"audio/{s}_original.wav", "file": f"audio/{s}_{c}.wav"}
         for s in spks for c, *_ in CONDITIONS if c != "original"]
    return a, b


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Voice protection listening test</title><style>
*{box-sizing:border-box} body{font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
margin:0;background:#f6f7f9;color:#1d2229} .wrap{max-width:720px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:26px 0 8px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:20px;margin:16px 0}
.muted{color:#68707a;font-size:13.5px} button{font:inherit;padding:9px 16px;border-radius:8px;
border:1px solid #c9ced5;background:#fff;cursor:pointer} button:hover{background:#f0f2f5}
button.primary{background:#1f6feb;color:#fff;border-color:#1f6feb} button.primary:hover{filter:brightness(1.07)}
button.play{font-size:15px;padding:11px 20px} .scale{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.scale label{flex:1;min-width:112px;border:1px solid #c9ced5;border-radius:8px;padding:10px 8px;
text-align:center;cursor:pointer;background:#fff} .scale input{display:none}
.scale input:checked+span{font-weight:650} .scale label:has(input:checked){border-color:#1f6feb;
background:#eaf2fe;box-shadow:inset 0 0 0 1px #1f6feb} .bar{height:6px;background:#e3e6ea;border-radius:3px;overflow:hidden}
.bar>div{height:100%;background:#1f6feb;transition:width .2s} .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
code{background:#eef0f3;padding:1px 5px;border-radius:4px;font-size:13px}
</style></head><body><div class="wrap">
<div id="app"></div></div><script>
const TRIALS_A = __TRIALS_A__, TRIALS_B = __TRIALS_B__;
let seed = Math.floor(Math.random()*1e9), listener = "", idx = 0, trials = [], results = [];
function rng(s){return function(){s=(s*1664525+1013904223)%4294967296;return s/4294967296}}
function shuffle(arr,r){for(let i=arr.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[arr[i],arr[j]]=[arr[j],arr[i]]}return arr}
const app = document.getElementById("app");
function intro(){app.innerHTML=`<h1>Voice protection listening test</h1>
<p class="muted">About 20 minutes. Please use headphones in a quiet room, and keep the volume
constant once you have set it.</p>
<div class="card"><h2>What you will do</h2>
<p><b>Part 1 &mdash; naturalness.</b> You hear one clip. Rate how natural it sounds as speech.
Ignore what is being said and ignore the recording volume.</p>
<p><b>Part 2 &mdash; same person.</b> You hear a reference clip and then a second clip.
Rate how confident you are that both are <i>the same speaker</i>.</p>
<p class="muted">Some clips are deliberately degraded. There are no trick questions and no
right answers &mdash; we want your impression.</p>
<div class="row"><label>Your name or ID &nbsp;<input id="who" style="font:inherit;padding:8px;
border:1px solid #c9ced5;border-radius:7px"></label>
<button class="primary" onclick="start()">Start</button></div></div>`}
function start(){listener=(document.getElementById("who").value||"anon").trim();
const r=rng(seed);trials=shuffle(TRIALS_A.slice(),r).concat(shuffle(TRIALS_B.slice(),r));idx=0;render()}
function render(){
 if(idx>=trials.length) return done();
 const t=trials[idx], pct=Math.round(100*idx/trials.length);
 const isA=t.block==="A";
 const q=isA?"How natural does this sound as speech?":"How confident are you that these are the same speaker?";
 const labels=isA?["1 &mdash; bad","2 &mdash; poor","3 &mdash; fair","4 &mdash; good","5 &mdash; excellent"]
                 :["1 &mdash; definitely different","2","3 &mdash; unsure","4","5 &mdash; definitely same"];
 app.innerHTML=`<h1>${isA?"Part 1 &mdash; naturalness":"Part 2 &mdash; same person?"}</h1>
 <div class="bar"><div style="width:${pct}%"></div></div>
 <p class="muted">Trial ${idx+1} of ${trials.length}</p>
 <div class="card">
 ${isA?`<button class="play" onclick="play('a')">&#9654; Play clip</button>`
      :`<div class="row"><button class="play" onclick="play('r')">&#9654; Reference</button>
        <button class="play" onclick="play('a')">&#9654; Clip to judge</button></div>`}
 <h2>${q}</h2><div class="scale">${labels.map((L,i)=>
   `<label><input type="radio" name="s" value="${i+1}" onchange="pick(${i+1})"><span>${L}</span></label>`).join("")}</div>
 <p class="muted" style="margin-top:14px">You may replay as often as you like. Choose a rating to continue.</p>
 </div>
 <audio id="au"></audio><audio id="ref"></audio>`;
 const au=document.getElementById("au");au.src=t.file;
 if(!isA){document.getElementById("ref").src=t.ref}
 setTimeout(()=>play(isA?'a':'r'),250)}
function play(which){const el=document.getElementById(which==='r'?"ref":"au");el.currentTime=0;el.play()}
function pick(v){const t=trials[idx];
 results.push({listener,seed,block:t.block,speaker:t.speaker,condition:t.condition,rating:v,order:idx});
 idx++;setTimeout(render,180)}
function done(){
 const hdr="listener,seed,block,speaker,condition,rating,order";
 const csv=[hdr].concat(results.map(r=>[r.listener,r.seed,r.block,r.speaker,r.condition,r.rating,r.order].join(","))).join("\n");
 const blob=new Blob([csv],{type:"text/csv"});const url=URL.createObjectURL(blob);
 app.innerHTML=`<h1>Finished &mdash; thank you</h1><div class="card">
 <p>Please save the results file and send it back.</p>
 <p><a class="primary" style="display:inline-block;text-decoration:none;padding:10px 18px;
 border-radius:8px;background:#1f6feb;color:#fff" href="${url}" download="listening_${listener||"anon"}_${seed}.csv">
 Download results CSV</a></p>
 <p class="muted">Filename: <code>listening_${listener||"anon"}_${seed}.csv</code></p></div>`}
intro();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-speakers", type=int, default=6, dest="n_speakers")
    ap.add_argument("--spk-offset", type=int, default=20, dest="spk_offset")
    ap.add_argument("--utt-index", type=int, default=3, dest="utt_index")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    spks, made, skipped = build_audio(args)
    a, b = build_trials(spks, args)
    html = (HTML.replace("__TRIALS_A__", json.dumps(a))
                .replace("__TRIALS_B__", json.dumps(b)))
    open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8").write(html)
    os.makedirs(os.path.join(OUT_DIR, "responses"), exist_ok=True)

    key = [{"condition": c, "kind": k, "param": str(p), "note": n} for c, k, p, n in CONDITIONS]
    json.dump({"speakers": [int(s) for s in spks], "conditions": key,
               "n_trials_A": len(a), "n_trials_B": len(b)},
              open(os.path.join(OUT_DIR, "design.json"), "w"), indent=2)

    print(f"speakers      : {spks}")
    print(f"conditions    : {len(CONDITIONS)}")
    print(f"audio written : {len(made)} files")
    if skipped:
        print(f"SKIPPED (no saved protected wav): {skipped}")
    print(f"trials        : {len(a)} naturalness + {len(b)} identity = {len(a)+len(b)}")
    print(f"\nopen {os.path.join(OUT_DIR, 'index.html')} in a browser")
    print("zip the listening_test folder to send it to listeners")


if __name__ == "__main__":
    main()
