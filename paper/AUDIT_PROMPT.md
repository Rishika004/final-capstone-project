# Prompt for an independent audit session

Copy everything below the line into a fresh Claude Code session opened on
`C:\Users\Rishika\OneDrive\Desktop\Phantom\nullspace_voice_protection`.

It asks for an honest audit with the criteria fixed in advance. It deliberately
does **not** ask anyone to make an unwelcome result go away — that would be
research misconduct, and it would destroy the thing that makes this project's
record worth anything.

---

You are auditing a speech-research repository for **correctness bugs in a
measurement**. Judge the code against what it should do, not against which
answer would be convenient. Reporting "no bug, the result stands" is a good
outcome, not a failed audit.

## Background

The project measures a survival coefficient α: how much of a direction in
audio survives a diffusion purifier. The key comparison is each speaker
encoder's own gradient direction against a random direction matched to it in
norm and spectral envelope. Lower α for the encoder direction than for its
control is the paper's lead claim (C1).

Two experiments ran on 2026-10-08 and both went against C1:

* **Control B** (RESULTS §54, `experiments/run_control_b.sh`): an *untrained*
  CAM++ scored α = 0.597, inside the trained band 0.388–0.631. Conclusion
  drawn: the effect is not about speaker identity, so the word "speaker" came
  out of the paper's title.
* **#13c** (RESULTS §55, `experiments/exp_encoder_subspace_wav.py`): on a
  waveform purifier the effect reversed sign in all 20 cells.

A partial audit has already been done. **Verify everything below
independently — do not take it on trust.**

## What the partial audit found

**(a) A confirmed bug in #13c.** `exp_encoder_subspace_wav.py` computes the
direction as the gradient of `cos(embed(w), embed(wav))` evaluated at
`w = wav`. For any L2-normalised embedding that point is the global maximum of
cosine-to-self, so the gradient is analytically zero:
`d/dx ||f||² = 2·Jᵀf = 0`. Measured gradient norms there were 1.2e−08 to
5.7e−07, against 0.14–0.57 one small step away — a ratio of 6e5 to 1e7.

**(b) The mel path is sound for trained encoders.**
`experiments/exp_encoder_subspace.py` (used by §32, §37, §43, §44, §46 and
Control B §54) takes the gradient through a mel→waveform reconstruction while
the reference embedding comes from the original waveform, so the evaluation
point is genuinely off the maximum: cosine 0.64–0.95, relative gradient
5e−05 to 1.2e−04. Healthy.

**(c) A concern about Control B that was raised and then REFUTED.** The
untrained network's embeddings are near-degenerate, so its evaluation point
sits at cosine 0.995–0.999 with a relative gradient of ~3e−07 — 200–400×
weaker than any trained encoder's, near float32 epsilon. That looked like it
might make the direction numerical residue. **A stability test refuted it**:
recomputing the direction gave cosine 1.0000 against itself and 0.9991–0.9999
under a 1e−5 input jitter, matching the trained positive control exactly. A
small gradient is not an ill-defined one — backprop computes it to full
relative precision. **So Control B's direction is well defined and §54 appears
to stand.** Re-verify this; it is the single most consequential claim here.

## Your tasks

1. **Reproduce or refute (a), (b) and (c)** with your own checks, on more than
   one speaker.

2. **Finish the test that did not complete**: run the same two stability tests
   (repeat-run, and 1e−5 input jitter) on the **waveform** path of
   `exp_encoder_subspace_wav.py`. The mel path passes both. If the waveform
   directions fail them, (a) is confirmed as fatal and §55 is void. If they
   pass, the vanishing-gradient argument does not by itself void §55 and the
   reversal needs another explanation.

3. **Audit the remaining risks**, reporting each as confirmed, refuted or
   undetermined:
   - `core/matched_random.py` `band_matched_wav` randomises phase, which in
     the waveform domain may make the control noise-like by construction while
     the target is not. Diagnostic already seen: these controls behave almost
     identically to plain white noise (psr 0.52 vs 0.53, nsr 1.02 vs 0.93),
     i.e. matching the spectrum bought them nothing. Confirm or refute, and
     say whether the mel-domain `band_matched` shares the problem — it has no
     phase to randomise.
   - **Is α measuring the network or the front end?** A decisive check:
     measure α for the gradient of a *fixed random linear projection of fbank
     features* — same front end, no trained network, no untrained network
     either. If that also lands near 0.6, the effect is the front end and both
     C1 and Control B change meaning.
   - `exp_encoder_subspace_wav.py` scales directions by `--rel-rms` relative
     to the utterance. Check whether α is sensitive to that scale on a
     non-linear purifier, and that every direction in a cell really receives
     equal RMS.
   - Anything else you find.

## Rules, fixed before you start

- A bug must be **demonstrated** with a check someone else can re-run, not
  argued from plausibility.
- **Do not re-run an experiment to obtain a different verdict.** Re-running is
  legitimate only after you have shown the instrument was broken, and must be
  labelled a fixed-instrument re-measurement.
- **Do not change a pre-registered threshold, endpoint or decision rule.**
  They are in `paper/PREREG_council5_controls.md` and `experiments/run_13c.sh`.
  If a rule now looks wrong, say so; do not edit it.
- **Do not modify RESULTS sections before §48.** It is an append-only lab
  notebook; rewriting history to match a new belief destroys the record.
- `python -m pytest tests/ -q` must stay at 70 passed.
- One GPU job at a time — a second job beside a running one has OOMed this
  machine before.

## Deliverable

Append a new numbered section to `experiments/RESULTS_ICASSP.md` containing,
for each risk: the verdict, the evidence, and — for every confirmed bug —
which existing RESULTS sections it invalidates and what would have to be
re-run. Then fix the confirmed bugs in code with tests green, and commit with
a message stating what was broken and what it cost.

Do not state a conclusion the evidence does not support, in either direction.
A finding that the original result stands is as valuable as a finding that it
does not.
