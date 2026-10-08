# Next steps — the on-manifold route

Written 2026-10-09, after RESULTS §54–§58 closed out the additive/geometry
programme. This is the working brief for the next phase. A session picking this
up should read it, then `paper/FRAME_v7.md`, `experiments/THREAT_MODEL.md` and
RESULTS §54–§58 before proposing anything.

## 1. What this project is

Speech privacy. We study how a person can publish a recording of their own
voice without it being usable to train or prompt a voice-conversion model
against them. The published line of work is AntiFake (CCS 2023), De-AntiFake
(ICML 2025), VocalBridge (2026) and ClearMask (ASIA CCS 2025). Our role so far
has been measurement: we evaluate how well these protections hold up, and we
report the limits we find, including our own.

## 2. Where the additive approach stopped, in measured numbers

A small additive signal optimised against a speaker-verification model works
well **on the models it was optimised against** and poorly otherwise:

| condition | voice-conversion success |
|---|---|
| verification models included in the optimisation, no restoration | 12–14 % |
| a verification model **not** included | **75–92 %** |
| after a learned restoration stage (PhonePuRe) | 38 / 27 / 71 % |

The second row is the problem and it is not about restoration at all — no
cleaning step is applied there. A perturbation derived from one model's
gradients simply does not transfer to a different model.

§54–§58 also closed off the shortcut we had hoped for: the direction-survival
analysis that was meant to identify structure shared across verification models
turned out to be measuring the analysis apparatus itself. There is no gradient
geometry to exploit.

## 3. The hypothesis for this phase

A **parametric transform of the speech itself** — formant warping — changes an
acoustic property of the voice that any speaker-verification model has to read.
So it should transfer across models rather than being tied to one. And because
its output is ordinary speech, a restoration model has nothing to subtract.

> **THESE NUMBERS MAY NOT APPEAR IN THE PAPER.**
> `experiments/results/parametric_probe.csv` is 20 speakers on the
> **superseded** base, which `FRAME_v6`'s base rule forbids quoting anywhere,
> including the abstract and figures. They are here to *motivate Step 1* and
> for no other purpose. Step 1 re-measures them on the current base; until it
> does, there is no quotable number for this hypothesis at all.

| warp strength | WavLM-SV | ECAPA | d-vector | survival loss R |
|---|---|---|---|---|
| 0.88 | 0.95 | 0.43 | **0.55** | 0.019 |

55 % against 8–25 % for the additive method on the same model, and essentially
no loss through the purifier.

**Which column counts, and the complication.** The d-vector is untargeted by
the additive system *and* is the encoder DiffVC conditions on, so breaking it
blocks cloning causally rather than only by proxy. But §42h rates it a mediocre
verifier (minDCF 0.39–0.57), so a protection number scored on it inherits that
weakness, exactly as the WavLM-SV claims do. ECAPA is the only verifier §42h
rates as trustworthy, and it reads 0.43 here.

So **report ECAPA as the headline and the d-vector separately as the cloner's
own conditioner**, each with its operating point stated. No single column is
both the untargeted condition and the trustworthy instrument, and any sentence
implying otherwise is wrong.

**The untested part** is that ClearMask *optimises* its transform parameters
while our probe used fixed magnitudes, and ClearMask never evaluates under any
restoration model (verified: zero mentions). So: optimised parameters,
evaluated under restoration, measured on untargeted models.

## 4. The cost, and the gate that decides everything

The probe also contains an existence proof of a trade-off. Spectral tilt at any
magnitude is inaudible, perfectly survives purification, and gives **zero**
protection. Warp gives protection and costs naturalness — roughly 2.4–2.6 on a
MOS predictor against 4.15 for the original.

So: inaudible, survives restoration, actually protects — the evidence says pick
two.

**If a listener hears warped audio as a different person, this is voice
anonymisation, not protection, and the phase should stop.** That question is
cheap to answer and nothing else should be built until it is.

## 5. Steps

**Step 0 — the listening gate. No GPU. Do this first.**
Reuse the blinded-stimulus and loudness-matching code from W1
(`experiments/make_listening_stimuli.py`, `utils/loudness.py`). Build a
same/different task on warped versus original audio. **Write the decision rule
and the sample size before collecting a single response.**

**Run it as a FRONTIER, not a single point** — several warp strengths, plus the
additive arm and both anchors, in one panel. A single-point test at 0.88
returns one bit and that bit is probably "different person", which ends the
phase while telling you nothing about whether an acceptable operating point
exists lower down. The frontier costs the same and answers both questions.
(Improvement contributed by the review session, 2026-10-09.)

If no warp strength is heard as the same speaker, write that up and stop.

**Step 1 — re-measure the baseline on the current base.**
The probe is 20 speakers on the superseded base, which `FRAME_v6`'s base rule
forbids quoting. Re-run fixed-magnitude warp on test-clean, 29 cross-session
speakers, purifier v7, and add the two conditions it never had: the untargeted
model, and PhonePuRe. A phoneme-conditioned restorer cannot know the original
formants, so warping may survive the stage that defeated the additive method.
That is worth knowing early.

**Step 2 — optimise the transform. NOT FUNDED YET; Step 1 decides.**
Parameters, not a 64 000-sample perturbation, so the search is low-dimensional.
Grid search or CMA-ES; do not assume PGD is the right tool. Optimise against an
ensemble, with the purifier in the loop, under an explicit naturalness budget.

**Why it waits.** Optimisation only pays if there is headroom between the
fixed-magnitude frontier and the naturalness budget, and the measured frontier
says there may be none: 58 % protection at UTMOS 2.41, against a floor of 3.0
that the shipping additive arm clears by 0.07. If Step 0 rejects 0.88 and
Step 1 confirms that frontier on the current base, Step 2 is a search inside an
empty box. **Let Step 1 set the budget rather than fixing it now.**

**One idea worth recording before it is lost.** A fixed warp moves every
speaker's formants the same way, which is plausibly why it reads as "different
person" rather than "same person, odd recording". Warping *toward each
speaker's own distribution edge* instead of a global constant is cheap, is not
what ClearMask does, and is the one route that might buy transfer without the
full perceptual cost. It is a Step 2 variant, not a replacement for the gate.
(Contributed by the review session, 2026-10-09.)

**Step 3 — evaluate.**
Untargeted models, all four restoration settings, against both the additive
system and fixed-magnitude warp, with quality measured and Step 0's listening
data attached.

**Step 4 — write.** The measurement results in §42h, §29, §40c and §22b remain
a fallback paper if Steps 1–3 do not deliver.

## 6. Rules carried over

* Pre-register each experiment's endpoint, decision rule and outcome branches
  in `paper/` before running it, in the style of
  `paper/PREREG_council5_controls.md`.
* A result that contradicts the hypothesis is a finding. Do not re-run an
  experiment to obtain a different answer.
* Every claim names its condition: which models were targeted, which
  restoration setting, what perceptual budget.
* Current evaluation base only.
* `python -m pytest tests/ -q` stays green.
* One GPU job at a time — 6 GB laptop card.
* Any new measurement path is gated by
  `experiments/check_direction_sanity.py` before a verdict is read from it
  (§58's rule, learned the hard way).

## 7. Honest odds

Step 0 is close to a coin flip. A warp strong enough to defeat an untargeted
model may well be strong enough to hear. The novelty overlap with ClearMask is
real, so the contribution is the optimisation and the restoration evaluation,
not the transform.

Say so if a step looks not worth its cost.
