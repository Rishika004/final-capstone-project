# Voice protection that survives diffusion purification

Protecting a published voice recording against **diffusion-based voice
cloning**, when the attacker uses **diffusion purification** to try to wash the
protection off. Research code for a paper under preparation.

> **About the repository name.** The directory and the early commits say
> *null-space*. That was the project's founding hypothesis and **it was
> falsified by our own audits** (RESULTS §54–§58): the geometric quantity the
> claim rested on turned out to be reproduced by a random projection of the
> front end with no network in it. The name is kept so the history stays
> readable. No null-space or α claim survives, and none appears in the paper.
> The original README is preserved at `docs/README_v1_nullspace.md`.

---

## What we claim, and what we do not

**The defence works.** A perturbation crafted with the purifier *inside* the
optimisation loop survives two diffusion purifiers and defeats two
diffusion-based cloners:

| cloner | clone success, unprotected | protected |
|---|---|---|
| DiffVC | 48.3 % | **0.0 %** |
| F5-TTS (2024, no speaker encoder) | 86.2 % | **10.3 %** |

Nine of nine cells each, Holm-corrected, 29 cross-session LibriSpeech
test-clean speakers. It also holds against a purifier never crafted against
(89–128 % retention) and a best-of-5 stochastic attacker (90–97 %).

**What the purifier-in-the-loop step actually buys is coverage, not strength.**
At equal perturbation energy it helps against an attacker that was *not*
targeted, and does not help against the one that was.

**And the limit, measured with the same protocol.** Protection does **not**
transfer to a speaker encoder held out of the crafting ensemble. Against a
held-out encoder behind a purifier we never trained on, it is
**72.4 % → 72.4 %** — none at all.

**The cost is real.** 4.15 → 3.07 on a 1–5 naturalness scale, intelligibility
unchanged. This is not an imperceptible perturbation and we do not claim one.

---

## Layout

```
experiments/   the runs and their analyses; RESULTS_ICASSP.md is the lab record
paper/         DEFENSE_BRIEF.md (start here), plans, pre-registrations, figures
demo/          a localhost demo: record a voice, watch a clone fail (~2 min)
core/ models/ utils/ evaluation/ training/   the method
listening_test/  blinded same-speaker + naturalness test
```

**Read `paper/DEFENSE_BRIEF.md` §I.1 first.** It carries the claims, the
conditions each number depends on, and the wordings forbidden because earlier
work falsified them.

---

## Running it

```bash
pip install -r requirements.txt
```

You also need LibriSpeech (`openslr.org/12`, test-clean and dev-clean) in
`data/librispeech/`. Speaker-encoder weights download on first run.
`external/` holds third-party code (De-AntiFake, NISQA, Coqui TTS) and is not
redistributed here — clone it from the upstream projects.

The demo:

```bash
python demo/app.py --preload      # http://127.0.0.1:8200
```

---

## How this project works

Rules adopted after things went wrong, kept because they keep going right:

* **Pre-register** endpoint, decision rule and outcome branches before running
  — see `paper/PREREG_*.md`.
* **A result that contradicts the hypothesis is a finding.** This repository
  contains more refuted claims than confirmed ones, including the one it is
  named after. Predictions are written down before runs and scored afterwards;
  two of the three scored so far were wrong.
* **Current evaluation base only** — test-clean, 29 cross-session speakers,
  purifier v7. Earlier numbers are not quotable.
* **Every cloning comparison comes from a single run.** The cloner is
  stochastic; the same contrast measured twice moved by two speakers.
* **Report clone-success rates beside any margin contrast.** A cell where both
  arms sit near zero is *not identifiable*, not null.

---

## Status

The experimental programme is closed. Open: the listening test needs
participants, the paper is not drafted, and the method has no name. Scope is
29 speakers, one corpus, English.
