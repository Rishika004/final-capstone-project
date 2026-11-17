# Voice protection that survives diffusion purification

Everything behind the paper: code, results, the full lab record, and the one
model checkpoint you cannot download. Target submission date: 9 February 2027.

---

## What the paper claims

**C1 — the defence works.** A perturbation crafted with the purifier inside
the optimisation loop survives two diffusion purifiers and defeats two
diffusion-based cloners. Clone success through the purifier:

| cloner | unprotected | protected |
|---|---|---|
| DiffVC | 48.3 % | **0.0 %** |
| F5-TTS (2024, no speaker encoder) | 86.2 % | **10.3 %** |

Nine of nine cells each, all Holm-corrected. Quality cost: 4.15 → 3.07 on a
1–5 naturalness scale, with intelligibility unchanged.

**C2 — the purifier-in-the-loop step buys coverage, not strength.** At equal
perturbation energy it helps against an attacker that was *not* targeted and
does not help against the one that was. Two budget-matched routes agree.

**C3 — the limit, measured with the same protocol.** Protection does not
transfer to a speaker encoder held out of the crafting ensemble. Against a
held-out encoder behind a purifier we never trained on it is
**72.4 % → 72.4 %** — none at all.

C3 is not an afterthought. It is why C2 is interesting.

---

## What is in here

```
paper/        the brief (DEFENSE_BRIEF.md -- READ THIS FIRST), the plan,
              every pre-registration, references.bib, related_work.md, figures
results/      RESULTS_ICASSP.md (the full lab record, 66 numbered sections),
              THREAT_MODEL.md, and csv/ with every result file (334)
code/         all of our own code: experiments, core, models, utils,
              evaluation, training, tests, and the live demo
checkpoints/  purifier_v7.pt -- we trained this one, it is not downloadable
listening_test/  the test page and its stimulus key (audio excluded, 6 MB)
```

**Start with `paper/DEFENSE_BRIEF.md` §I.1.** It states the topic, the title,
the three claims, the conditions every number carries, and the wordings that
are forbidden because earlier work falsified them.

---

## What is NOT in here, and why

| missing | why | how to get it |
|---|---|---|
| `data/` (LibriSpeech) | gigabytes, and public | `openslr.org/12` — test-clean and dev-clean |
| other `checkpoints/` | 1.1 GB; only purifier_v7 is ours | retrainable from `training/`, or ask |
| speaker-encoder weights | downloaded on first run | automatic (speechbrain / HuggingFace) |
| `external/` | De-AntiFake, NISQA, Coqui TTS — third-party code under their own licences | clone from their own repositories |
| generated audio (~90 MB) | reproducible from the code | re-run the scripts |
| the AntiFake paper PDF | copyrighted, not ours to pass on | ACM DL, DOI 10.1145/3576915.3623209 |

---

## Running it

```
pip install -r code/requirements.txt
```

Then LibriSpeech into `data/librispeech/`, and `purifier_v7.pt` into
`checkpoints/`. The headline cloning run:

```
python experiments/exp_cloning_protection.py --subset test-clean --cross-session \
  --n-files 29 --spk-offset 0 --ref-utt 0 --enroll-utt 0 \
  --purifiers none,ours,diffwave --dw-t 3 --ours-ckpt checkpoints/purifier_v7.pt \
  --thresholds experiments/results/thresholds_clean_xsess_tc.csv \
  --variants plain_tc_m30_seg,fullnsT_tc_m30_seg --clone-seed 2026 \
  --out experiments/results/my_run.csv
```

The live demo, which records a voice and shows a clone failing in about two
minutes:

```
python demo/app.py --preload      # then http://127.0.0.1:8200
```

---

## Where each claim's evidence lives

| claim | section | file |
|---|---|---|
| C1, DiffVC | RESULTS §61a | `cloning_tc_m30_seg_n5.csv` |
| C1, F5-TTS | RESULTS §60 | `cloning_f5tts.csv` |
| C1, unseen purifier | RESULTS §33 | transfer CSVs |
| C2, varying the cloner | RESULTS §60d, §61c | both of the above |
| C2, varying the ensemble | RESULTS §64 | `cloning_heldout_seg_n7a.csv` |
| C2, decomposition | RESULTS §62 | `cloning_tc_m30_seg_n5b.csv` |
| C3, held-out encoder | RESULTS §63a | `cloning_heldout_seg_n7.csv` |
| quality | RESULTS §65, §66 | `f5_clone_quality.csv` |

---

## Rules this project works under — please keep them

These were learned the hard way and several exist because something went
wrong once.

* **Current evaluation base only.** test-clean, 29 cross-session speakers,
  purifier v7. Nothing from before RESULTS §21 is quotable as a number.
* **Every cloning comparison comes from a single run** (§62b). DiffVC's clones
  are unseeded and the arm list shifts the RNG; the same contrast measured
  twice moved 0.0137 and two speakers. Pass `--clone-seed` on new runs.
* **Pre-register endpoint, decision rule and branches before running**, in
  `paper/`, in the style of `PREREG_N2_modern_cloner.md`.
* **A result that contradicts the hypothesis is a finding.** Do not re-run to
  get a different answer. Several sections here record predictions that
  failed, including two of three scored ones.
* **Print clone-success rates beside any margin contrast.** Saturated cells
  (both arms near 0 %) are *not identifiable*, not null — §59b.
* **Never write "nullspace"** in the paper. The repository is named after a
  programme §54–§58 closed; no α number survives.
* **Do not write bulk output into the OneDrive folder.** It costs 2–3x on
  later GPU runs and looks exactly like thermal throttling.

---

## Honest state

Done: the experimental programme is closed. Two cloners, two purifiers, a
stochastic attacker, the budget-matched attribution, the measured limit, and
quality on both the published audio and the attacker's output.

Open: **the listening test has no responses yet** — it is live and needs
10–15 naive listeners on headphones; **no draft exists**; and the method has
no name. n = 29 speakers, one corpus, English only, which caps what the
statistics can resolve (§52, §64).
