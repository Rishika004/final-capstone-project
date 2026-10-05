# Roadmap to publication

**Written 2026-09-19.** Supersedes nothing; sits beside `FIXES_TO_PUBLISHABLE.md`,
which tracks correctness defects. This file answers two different questions:

1. **What would drive the cloning test toward 0 % clone success?**
2. **What is still required before this is submittable to a good venue?**

Ordered by expected impact per hour of work, not by category. Every claim about
prior work here was verified against the primary source during the session that
produced this file.

---

## PART 0 — The honest starting position

| Claim | Status | Number |
|---|---|---|
| Perturbation survives purification | **solid** | α 0.87 vs 0.74 random, n=20 |
| Survives purifiers we did not train | **solid, strongest result** | 86–107 % retention, 20/20 in all 4 cells, p=1.9e-06 |
| Constraint helps only under purifier mismatch | **solid** | 7/12 cells after Holm, and the no-purifier control loses 0/20 |
| Costs nothing perceptually | **solid** | PESQ 3.44, STOI 0.965, quieter than plain PGD |
| Stops cloning | **weak, and honest** | 100 % → **38–60 %** with the cloner's encoder held out |

The first four are publishable as they stand. **Row five is the problem**, and
Part 1 is about it.

---

## PART 1 — Getting the cloning number down

### Is 20/20 fail (0 % clone success) actually reachable?

Probably not at the current perceptual budget with a held-out encoder, and it is
worth being clear-eyed about that before spending GPU hours. But **90 %+
protection is reachable**, and there is published evidence for it.

AntiFake (CCS 2023, Table 4) is the relevant precedent. Crafting against **one**
speaker encoder, their protection reached only **0–20.6 %** on synthesizers built
around a *different* encoder. Crafting against **all four**, they reached
**97–100 %** across every synthesizer tested — including ElevenLabs, described in
their §6.1 as "a completely black-box system" whose encoder they did not know.

**We currently craft against two.** We are running the small-ensemble version of an
experiment whose large-ensemble version is already published as working.

---

### P1.1 — Enlarge the crafting ensemble ⛔ NOW THE CRITICAL PATH
**Effort:** ~2 h coding, ~4 h GPU · **Expected impact:** very large

Add three or four more speaker encoders to `CRAFT_SETS` in
`experiments/exp_purifier_aware.py`. Candidates, all public and architecturally
distinct from the current pair:

| Encoder | Why this one |
|---|---|
| **x-vector** (SpeechBrain `spkrec-xvect-voxceleb`) | TDNN lineage, different from both current encoders; also De-AntiFake's metric |
| **CAM++** (WeSpeaker / 3D-Speaker) | context-aware masking, a genuinely different inductive bias |
| **ERes2Net** | multi-scale residual; used as an evaluation encoder by Enkidu and FreeTalk |
| **ResNet-34 / H\_ASP** | the encoder YourTTS uses; AntiFake has it in its ensemble |

**Design rule, so this does not quietly become cheating:** Resemblyzer stays held
out. The point is to attack what encoders have *in common*, which is what a
stranger's encoder will also have — not to add the one the cloner uses.

**A cheap diagnostic worth running first (~1 h):** add a GE2E model with the *same
architecture as Resemblyzer but different weights*, if one can be found or trained.
If protection transfers to it but not to Resemblyzer, the barrier is weights. If it
transfers to neither, the barrier is architecture. That single result tells you
whether ensemble size or ensemble *diversity* is the lever.

---

### P1.2 — Attack segment embeddings, not the pooled embedding ⭐ UNDER-RATED
**Effort:** ~3 h coding, ~4 h GPU · **Expected impact:** large, and it is free evidence

This is a published improvement we are not using, and it comes from the paper we
are already citing.

AntiFake §C2 states the problem directly: speaker encoders build partial embeddings
from segments and then pool them, and *"directly deviating the final embedding is
susceptible to local optima, as the impact of a partial embedding not close to the
target is mitigated by other partials that may have attained better optimal
points."* Their fix is a **weighted segment-based raw embedding deviation**.

**We attack the pooled embedding.** In `craft()`, `ens_cos()` calls `e.embed(wav)`,
which returns the final L2-normalised vector for the whole utterance. Every encoder
wrapper in `multi_encoders.py` does one forward pass over the full clip.

So we are in exactly the local-optimum regime AntiFake described and fixed. The
change is to score windowed segments and combine with a weighted loss that pushes
the *worst* segment rather than letting good segments mask bad ones.

This is also defensible in review: adopting a published refinement of the baseline
is not a novelty claim, it is doing the baseline properly.

---

### P1.3 — Re-run the LOUD budget with the encoder held out ✅ DONE 2026-09-19 — ANSWER: TRANSFER
**Effort:** ~0 coding, ~4 h GPU · **Expected impact:** potentially decisive

**Verified 2026-09-19, and this changes the priority.** The loud budget `m30_f15`
has already been run against DiffVC, and it produces **1 % and 5 % clone success**
(`cloning_protection.csv`, `cloning_protection_ours.csv`). That is the near-zero
number we want.

**But those runs used all three crafting encoders, so they are white-box.** The
loud budget has **never** been run with Resemblyzer held out.

That makes this the most direct test of "can we reach a strong cloning number
honestly?", and it needs no new code — only `--craft-encoders no-dvec` at
`--margin-db 30 --snr-floor 15`, then the cloning evaluation.

Two outcomes, both worth having:

* **Held-out loud budget stays near zero** → the 38–60 % figure was a budget
  artefact, not an encoder-transfer failure, and the headline is recoverable.
* **It degrades like the tight budget did** → encoder mismatch is the real barrier,
  which sharpens the case for P1.1 and is itself a clean finding.

#### RESULT: the second outcome. Full write-up in RESULTS §22.

Unpurified, loud budget, encoder held out:

| encoder | clone success |
|---|---|
| WavLM-SV (crafted) | **14 %** |
| ECAPA (crafted) | **12 %** |
| **d-vector (held out)** | **88 %** |

Same speakers, same perturbation files, same thresholds, same run. The only
difference between the 12 % column and the 88 % column is whether that encoder was
in the crafting ensemble.

**The perturbation is not weak — it is narrow.** Loudness protects what you
optimised against and does not generalise. So P1.1 is no longer the
highest-value item among several; it is **the only remaining lever on the cloning
claim**, and this run has ruled out the cheaper alternative.

All cloning numbers currently reported use `m20_f20`, the tighter budget, because
the loud one **saturates** and cannot distinguish between arms. That is right for
the *ablation* and wrong for the headline. **Report both**: loud for "how well does
this protect", tight for "which component does the work".

For calibration: AntiFake's own reported quality sits at MOS 3.37–3.84, which is
not a quiet perturbation. We are being stricter than the baseline and then
reporting a weaker number than it.

**Caveat to state:** perceptual cost must be reported for the loud budget too, or
this is just turning up the volume.

---

### P1.4 — Three sampling seeds per cell (fix S6)
**Effort:** ~1 h coding, ~2 h GPU · **Expected impact:** small on the mean, large on credibility

DiffVC is a stochastic sampler and `cloning_protection_*.csv` is exactly one draw
per cell.

**This is not theoretical.** On 2026-09-19 we regenerated the demo clone for
speaker 3170 and a cell flipped: the stored result had Resemblyzer *rejecting* the
protected clone, the fresh draw had it *accepting* at 0.792 against a threshold of
0.779. A borderline cell is currently a coin flip.

With n = 5–12 speakers per cell, single-draw sampling is the second-largest source
of noise after sample size. Three seeds is the minimum defensible number.

---

### P1.5 — Multi-utterance protection
**Effort:** ~2 h coding, ~3 h GPU · **Expected impact:** moderate

Right now one 4-second clip is protected and the attacker clones from it. A real
victim posts many clips. Two sub-cases, and they pull in opposite directions:

* **All clips protected** — the defender's realistic best case. Protection should
  improve, because the attacker has no clean reference anywhere.
* **Attacker concatenates several protected clips** — the attacker's realistic
  best case, and it will make our numbers *worse*, because averaging over clips
  cancels per-clip perturbation noise.

Run the second one. It is the harder test and a reviewer will ask for it.

---

### What will NOT move the cloning number, and should be done anyway

* **F3, running DiffWave at t=3 or t=10.** It currently runs at t=1, which is close
  to a no-op, so the "foreign purifier" cloning column is nearly "no purifier".
  Fixing it makes results *worse* and is required for honesty.
* **A second cloner.** Also makes results worse, also required. See P2.3.

---

## PART 2 — Publication blockers

Ordered by how badly each one hurts if a reviewer finds it first.

### P2.1 — Corpus split: move to `test-clean` (fix S1b) ⛔
**Effort:** ~1 h coding, ~6 h GPU (purifier retrain), download already done

This is the single biggest comparability gap and it is already half-built.

| | now | required |
|---|---|---|
| Evaluation corpus | `dev-clean` | **`test-clean`** |
| Purifier training | `dev-clean` (same split!) | **`train-clean-100`** |
| Speakers available | 40, of which 20 usable | 40, all usable |

Three separate problems, all fixed by the same move:

1. **Zero of seven baseline papers evaluate on `dev-clean`.** AntiFake and
   De-AntiFake both name `test-clean`. Our numbers currently cannot be placed
   beside theirs at all.
2. **n = 20 is a ceiling, not a choice.** `dev-clean` has 40 speakers and the
   purifier ate half of them.
3. **Same-split train/eval.** De-AntiFake trains its purifier on
   `train-clean-100` and says so. We trained on the split we evaluate on.

`experiments/librispeech.py` already carries `EVAL_SUBSET`, `TRAIN_SUBSET`,
`DEANTIFAKE_SPEAKERS` (their published 25) and `CROSS_SESSION_SPEAKERS`. The
tarballs are downloaded and extracted. **The purifier retrain has not run.**

**Bonus available for free:** `DEANTIFAKE_SPEAKERS` gives one table row directly
comparable to De-AntiFake's Table 1 and AntiFake's Table 3. That single row is
worth more to a reviewer than any amount of prose about comparability.

---

### P2.2 — Cross-session thresholds (fix S1) ⛔
**Effort:** ~1 h coding, ~1 h GPU · folds into P2.1

Our ECAPA error rate on clean audio is **0.03 %**. That is not a good result, it is
an implausible one. Every held-out speaker's first four utterances come from **one
LibriSpeech chapter**: same room, same microphone, same session. Thresholds are
calibrated on same-session trials, which makes verification far easier than reality.

VocalBridge, with the same limitation, reports 0.6 %. Twenty times higher.

`cross_session_split()` is written in `librispeech.py` and **is not called
anywhere**. Wire it in: draw enrolment from a different chapter than the reference,
and exclude scored utterances from the calibration pool.

Expect every absolute number to get worse. That is the point.

---

### P2.3 — A second cloning system ⛔
**Effort:** ~4 h coding, ~4 h GPU

One cloner is not enough for a venue that has seen AntiFake (five synthesizers) and
De-AntiFake (six). Pick one that is **not** a diffusion voice converter, so the
result is not about DiffVC's family:

* **OpenVoice V2** — De-AntiFake uses it, and their paper notes its encoder is *not*
  in AntiFake's ensemble, so it is genuinely held out for us too.
* **YourTTS** — a TTS system, uses the H/ASP encoder, tested by both baselines.
* **SeedVC** — De-AntiFake's hardest case; AttackVC and VoiceGuard both *failed*
  against it, which makes it a strong stress test.

---

### P2.4 — Resolve the Shallow Diffuse comparison (fix T3) ⛔
**Effort:** ~2 h reading, 0 GPU

Shallow Diffuse (NeurIPS 2025) constructs a perturbation in a Jacobian eigenspace
of a diffusion model and evaluates it under diffusion purification. That is four of
the five words in our contribution.

Our defence is that they target the λ≈0 *erased* end and we the λ≈1 *preserved*
end. **That defence rests on a sign convention we could not verify from their
text.** Settle it before submission, not in a rebuttal. Re-read v5 (Feb 2026); our
prior-art note predates it.

---

### P2.5 — Missing citations (fix T4)
**Effort:** ~1 h, 0 GPU

* **Tramèr et al., ICML 2020** (arXiv:2002.04599) — *"defenses against
  sensitivity-based attacks actively harm a model's accuracy on invariance-based
  attacks."* A closer antecedent to our trade mechanism than Stutz.
* **arXiv:2607.00899** — adversarial purification for speaker verification with
  defender-aware adaptive attacks, gradients through the whole pipeline. Partially
  undercuts any "nobody used exact gradients" phrasing.

Already added this session: `pu2025freetalk`, `kim2025rovo`, `wang2025clearmask`,
`chen2025songbsab`, `yang2024rwvoiceshield`, `yu2025mimicblocker`,
`vanle2023antidreambooth`, `zhao2026purify`, `yu2023antifake`, `popov2022diffvc`.
**23 entries, all verified against primary sources.**

---

### P2.6 — A listening test
**Effort:** ~4 h to run · **Impact:** closes a gap both baselines report

We have **no** human evaluation. AntiFake reports MOS and a perceptual speaker
dissimilarity panel. De-AntiFake reports NISQA-based objective MOS and a four-point
human similarity judgement.

Minimum viable version: 10–15 listeners, two questions per clip pair — "same person
or different?" and "rate the audio quality 1–5". Even a small panel beats none.

**Cheap partial substitute if time runs out:** run **NISQA**, which is what
De-AntiFake actually reports. It is automatic, takes an hour, and gives a directly
comparable number. This is the single cheapest comparability win left.

---

## PART 3 — Already fixed this session, for the record

| Fix | What it changed |
|---|---|
| **F1** held-out encoder | Cloning claim restated: 4–19 % → **38–60 %**. The old number was white-box. |
| **F2** pre-filter per (encoder, purifier) | Matched-purifier protected rate 15/21/11 % → 40/50/32 % |
| **S2** per-channel thresholds | Absolute protection rates were overstated by up to **7.3 points** |
| **S3** no "best arm" | A minimum over four arms at n=5–14 is a biased estimator |
| **S4** Holm correction | Positive claim survives (7/12); the negative claim does **not** |
| **S5** pseudoreplication | "Significantly harmful when matched" now rests on d-vector alone |
| **S7** encoder dependence | The 3 cloning encoders are correlated readouts (ρ 0.39–0.59), not 3 tests |
| **T1/T2/T5** | Draft no longer contradicts the results of record |

Also corrected: our own repeated misquote of AntiFake's ablation (we had written
"99.6 % matched → 0 % mismatched"; all three parts were wrong), and an overclaim
that the DiffVC/Resemblyzer shared checkpoint was our discovery (it is publicly
verifiable from the repos).

---

## PART 4 — Re-check: what else could improve the results

Written after the list above, deliberately asking "what did I miss?"

### Crafting hyperparameters never swept
* **`--n-eot 1`.** Every EOT step uses **one** noise draw. The gradient estimate is
  as noisy as it can be. `n_eot=4` costs 4× compute per step but may buy more than
  4× in gradient quality. **Never tested.** Cheap to test at n=5 speakers.
* **`--steps 200`.** Chosen early, never re-examined after the loss changed.
* **`--lam-null 10`** and **`--null-t 200`.** Verified coverage: the loud budget
  was swept at 0/1/3/10/30, the tight budget only at **0/3/10**. And `null_t` takes
  exactly two values anywhere in the results, **0 and 200** — so "noise-matched
  beats zero-noise" is established, but *where* in the noise schedule to measure it
  has never been swept. That is the parameter closest to the paper's core idea and
  it has one data point.

### A targeted attack, as AntiFake uses
We push *away* from the victim. AntiFake pushes *toward* a selected real speaker,
and gives a reason worth taking seriously: landing on a real embedding keeps the
synthesis natural, whereas pushing into nowhere produces artefacts. Their §5.3 notes
the trade — targeting narrows the solution space but improves naturalness.

**Untested by us.** It could plausibly improve both the cloning number and the
perceptual number at once.

### Attack DiffVC's own pathway, not only speaker encoders
Every arm attacks speaker encoders and hopes the damage reaches the cloner. DiffVC
also has a mel encoder and an average-voice pathway. Nothing stops us adding a term
that directly disrupts the conversion, rather than only the identity summary. This
is a bigger change and arguably a different paper, but worth noting.

### Waveform-domain crafting
We craft in mel space with a phase-preserving inverse, so some perturbation is lost
in the mel → waveform conversion. RW-VoiceShield (Interspeech 2024) crafts on the
raw waveform for exactly this reason. Worth one comparison at n=5.

### Report the m30 budget's perceptual cost
If P1.3 is done, `m30_f15` quality numbers must be reported beside it. They exist in
`main_table_v3.csv` and are simply not in the paper.

---

## Suggested order of work

| Order | Item | GPU | Why first |
|---|---|---|---|
| 1 | P2.1 + P2.2 corpus and thresholds | ~7 h | Everything else must be re-run on the final split anyway. **Do not run new experiments before this.** |
| 2 | **P1.3 loud budget, held out** | ~4 h | **No new code.** Directly tests whether the weak cloning number is a budget artefact or a transfer failure — and that answer decides how much P1.1 is worth |
| 3 | P1.1 larger ensemble | ~4 h | Highest expected impact if P1.3 says transfer is the barrier |
| 4 | P1.4 three seeds | ~2 h | Cheap, and we have live evidence it matters |
| 5 | P1.2 segment embeddings | ~4 h | Published improvement, we are not using it |
| 6 | P2.3 second cloner | ~4 h | Required for the venue |
| 7 | F3 stronger purifier | ~2 h | Honesty; makes numbers worse |
| 8 | P2.4 Shallow Diffuse | 0 | Must be settled before writing |
| 9 | P2.6 NISQA, then a listening test | ~1 h + 4 h | Cheapest comparability win |

**The ordering point that matters most:** items 2 through 7 all produce numbers that
go in the paper. Running them before item 1 means running them twice. The corpus
move is not the most interesting task on this list, but it is the one that gates
everything else.
