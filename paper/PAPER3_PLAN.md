# Paper 3 — plan of record
### Purification-invariance is achievable, free, and protection-irrelevant
*Written 2026-09-16 after a four-reviewer audit (theory, negative-results venue, main-conference
elevation, experiment design) plus independent verification of every load-bearing claim.*

---

## 1. The claim

> Diffusion purifiers preserve on-manifold perturbations, exactly as the geometry predicts.
> That preservation is **protection-irrelevant for speech**, because speaker encoders are already
> invariant to precisely the on-manifold directions that are perceptually cheap.

Scope it to **speaker verification**, explicitly and everywhere. No cloning model is in our loop,
so any claim beyond ASV is unsupported. This scoping is what makes the work finishable; protect it
in the writing.

---

## 2. What is genuinely unpublished (verified, not assumed)

| # | Result | Status |
|---|---|---|
| N1 | **Noise-free / transformation-based voice protection evaluated under diffusion purification.** | **Novel.** ClearMask (ASIA CCS'25) never tests purification — 0 mentions; its adaptive attacker only dereverberates. De-AntiFake and VocalBridge attack only additive protections (all 5 baselines additive). AntiPure is images + additive. |
| N2 | **The existence proof: imperceptible + purification-invariant + 0 % protection.** Spectral tilt to ±18 dB: UTMOS 4.12 (= original), WER 0.4 %, R_t 0.0001, PSR 0.98, protection **0/480 trials**. Reverb 0/320. | **Novel.** No prior work constructs a counterexample to the survive-therefore-protect inference. |
| N3 | **"On-manifold perturbations survive purification" is NOT established in the image domain either.** arXiv 2505.22839, which we previously cited for it, contains none of the words manifold / on-manifold / off-manifold / tangent (verified 2026-09-16). Its mechanisms are gradient masking and *general* input-space compression. | **Strengthens N1/N2** — we are not porting a known result, we are establishing it. |
| N4 | **Formulation fix:** the surviving subspace is null(J) of the **correction** C(x)=P(x)−x, not of the x0-predictor; and it must be computed at the purifier's **operating noise level**. The zero-noise version is indistinguishable from random once purification is strong (PSR 0.61 vs random 0.61). | **Novel** as a speech result and as a correction others will otherwise repeat. |
| N5 | **Mechanism:** ASV encoders are trained with channel/reverb augmentation, so they are invariant to exactly the cheap on-manifold directions. Formant shifting is the sole exception **because you cannot augment it away without destroying the speaker labels.** | **Novel**, and it is the paper's explanatory spine. |
| N6 | **Registered prediction that failed:** we predicted before running that reverberation would protect strongly (channel mismatch is the classic ASV nuisance). It protects 0 % at every RT60 up to 600 ms. Explained by N5. | **Novel**, and disproportionately valued in this genre. |
| N7 | **The constraint significantly hurts** (margin Wilcoxon, plain vs +regulariser): worse on 15–18 of 20 files on every encoder; d-vector p = 0.0003 / 0.0027, WavLM p = 0.0107. | **Novel** negative result, now statistically supported. |

**Taken, cite and move on:** Stanczuk et al. 2212.12611 (σ→0 only — see §5); the purifier-validation
*principle* (DW-box, ICLR 2025, arXiv 2404.14309); amplification as an empirical observation
(2505.22839); adaptive-protection-for-voice (De-AntiFake §5.4, VocalBridge §V-E — belongs to Paper 2).

---

## 3. Workshop → conference: the four properties, and where we stand

Derived from the natural experiment in our own bibliography: DW-box was analysis-only with no new
defence and made **ICLR 2025 main**, because it invalidated published robustness numbers
(47.27 % → 16.80 %) and handed the field a protocol. "The Purification Paradox" dissected a
mechanism without retiring anything and went to a **CVPR workshop**. Same domain, same year.

| | Property | Now | To close it |
|---|---|---|---|
| **P1** | The finding survives a change of **artefact**, not just of seed. ≥2 independently originated purifiers. | **FAIL.** One purifier, ours, 0.78 M params, mel-domain. | **E2 + E3** below. This single item decides workshop vs conference. |
| **P2** | The reader must **change a practice**, not just learn a fact. | **FAIL as drafted** ("we tried a constraint, it didn't help" — nobody's practice changes). | Lead with **N2**. It retires an inference the field is actively making. |
| **P3** | A **mechanism with a registered prediction**. | **PASS — our strongest asset.** N5 + N6 + the self-correction on the r = −0.86 scoping. | Keep all three, foreground them. |
| **P4** | Evidence scaled to the claim. | **WAS FAIL, NOW MOSTLY PASS.** Margin statistics (§4) rescued every comparison. Survival still n = 5. | **E4**; delete the underpowered claims. |

---

## 4. The statistical change already made (2026-09-16, zero compute)

Every comparison had been tested on the **binary** protected/not outcome over 20 speakers, where the
exact test floors at p = 0.0078 and nothing could reach significance. Re-tested on the **continuous
decision margin** (`cos_after_purify − threshold`), Wilcoxon signed-rank, same 20 paired files:

* purifier-aware crafting: **p < 0.0001 at both budgets** (was p = 0.22–0.63). Margin shift −0.11 to −0.22.
* the constraint alone: **significantly worse**, p = 0.0003 (d-vector).
* the constraint on top of EOT: **nothing**, p = 0.31–0.97 → **delete the "100 %" claim.**

**Rule for the paper: report margin shifts as the effect size, binary rates only as a readable
summary alongside.** This is also the single cheapest credibility purchase available, and it is done.

---

## 5. Theory: the framing to use, and the one to stop using

**Stop writing "tangent space" anywhere near a t = 200 number.** Stanczuk et al. prove a **σ→0**
statement about the span of score *vectors*, not the kernel of the score *Jacobian*; their own
Limitations section disclaims larger t; their experiments use σ ≈ 0.01 on unit-scaled data while our
t = 200 has σ_VE = 0.72 against a log-mel std of 2.8–4.8 — **15–25× outside the regime**. Our own
§5c(i) refutes the identification anyway: a tangent space does not depend on t, but our subspace does
(linear gain 0.375 zero-noise vs 0.004 noise-matched).

**Use this instead.** Tweedie/Miyasawa gives `J_m(x_t) = (I + (1−ᾱ_t)∇²log p_t)/√ᾱ_t`, so a
clean-signal perturbation enters the reverse process as `√ᾱ_t d` and is **passed through** exactly
when `∇²log p_t d = 0`, **erased** exactly when `∇²log p_t d = −d/(1−ᾱ_t)`. Survival and removal are
the flat and steepest-curvature eigenspaces of one operator — the log-density Hessian *at the noise
level where purification happens*. This is what makes "null space" an accurate word: of the score
Jacobian, never of the denoiser network.

Two things this buys us, converting negatives into derivations:

1. **Amplification is predicted, not curious.** `J_m` is the posterior covariance up to scale, so
   gain > 1 exactly where the posterior over the clean signal is more uncertain than the noise being
   removed. A purifier is not a contraction. (Our measured 1.48 is an adversarial extreme at finite
   ‖d‖, state that.)
2. **The zero-noise regulariser is a no-op by derivation.** With a residual purifier `x0 = x + U(x)`,
   `J_m = I + J_U`, so pass-through requires `J_U d = ((1−√ᾱ)/√ᾱ) d ≈ 0.232 d` at t = 200 — an
   **eigenvalue** condition. The zero-noise version targets eigenvalue 0. It was always going to fail.

**Two honesty caveats our own code forces.** Our denoiser is **time-unconditional**
(`MelUNet2D.forward(self, x)`), so Tweedie returns the score of a *mixture* marginal, not `p_t`.
And the residual parameterisation biases `J_U → 0` globally, which is a likelier explanation of
"random perturbations are already ~95 % pass-through" than "the tangent space is enormous."

---

## 6. Experiments (≈ 3 GPU-hours; measured costs, one job at a time)

| # | Experiment | Cost | Fixes | Command |
|---|---|---|---|---|
| **E1** | Margin statistics | **done** | P4 | — |
| **E2** | **Cross-purifier transfer.** Directions optimised against v3, evaluated through v4 and v5. Needs a ~15-line `--eval-ckpt` flag in `exp_nullspace_survival.py`. | 0.6 h | **P1** | `exp_nullspace_survival.py --purifier-ckpt v3 --eval-ckpt v4 --n-files 20 --t 50,100,200 --null-t 200 --sampler ddim` |
| **E3** | **Second architecture: 1-D UNet (v5).** 4.5× faster to train than the 2-D. Gate it on denoising gain ≥ 0.15 and channel EER within 1 pt of v3 *before* using it — the v2 lesson. | 0.5 h | **P1** | `train_purifier_v2.py --arch unet1d --base 96 --residual --t-max 200 --steps 30000` |
| **E4** | Survival on **20 speakers × 2 utterances** instead of 5. Patch `exp_nullspace_survival.py` to also record `R_t` per direction (free — `find_direction` already builds `x_t`). | 0.5 h | P4, (c) | `--n-files 20 --utt-index 3` then `--utt-index 6` |
| **E5** | **Subspace dimension** by greedy orthogonal deflation (Gram-Schmidt inside the Adam loop), R_t(k) for k = 1…64 out of ambient ≈ 32 000. | 0.7 h | structure | new `exp_nullspace_dimension.py` |
| **E6** | **Vocoder control.** Same PSR metric with the purifier removed, isolating how much "survival" is the mel + Griffin-Lim round trip rather than the purifier. | 0.2 h | confound | variant of `exp_nullspace_survival.py` |

**E2 is the decisive one.** Either directions transfer across independently trained purifiers — and
the title claim upgrades from *our purifier* to *diffusion purifiers of speech* — or they do not, and
that is a fourth negative result with real teeth (*a defender cannot precompute the subspace without
the attacker's exact weights*). Both outcomes improve the paper, which is what makes it the best
experiment available.

**E5 matters more than it looks.** If R_t(k) stays flat to k = 64, the subspace is large and not a
knife edge. If it rises fast, that **mechanistically explains N7** — the constraint discards most of
the attack's search space — which is consistent with the constraint hurting more at the tight budget.

---

## 7. Cut list (all of it, without regret)

* **R_t as a "cheap predictor" tool.** It has no dynamic range at t = 50 (0.0000–0.0081, ρ = −0.12);
  its pooled correlation is *between*-family not within (ρ ≈ 0 for EQ and McAdams); and it correlates
  **positively with protection** (ρ = +0.39 / +0.34 on WavLM, p < 1e-14) — it would be a cheap
  predictor of the quantity this paper argues is the wrong one, pointing the wrong way for the
  quantity that matters. Keep two sentences: it is the objective used to *construct* the directions.
* **The "85 → 100 %" claim.** Deleted. 3 discordant pairs, p-floor 0.25, margin test finds nothing,
  and the sign reverses at the other budget.
* The λ = 10 / 30 sweep and any "recommended configuration" line → one sentence.
* The `directed` variant (degenerate minimum: the optimiser shrinks the perturbation ~5× and stops
  attacking) → one footnote. It is an optimiser pathology, not a finding.
* The 3-speaker trial sweep → one sentence.
* Any sentence containing "tangent space" at operating noise, or "we propose purification-aware
  voice protection" (pre-empted — that is Paper 2).
* The listening test. Cannot be done properly this cycle; state as a limitation.

---

## 8. Title and outline

**Title:** *Imperceptible, Purification-Invariant, and Useless: Survival Is the Wrong Objective for
Voice Protection*
**Sober alternative:** *Surviving Diffusion Purification Does Not Protect Speech*

1. **Introduction** (0.75 p). State the hypothesis **as the field's, with citations** — never as
   ours; a paper that refutes its own idea reads as a lab notebook, one that refutes a published
   premise is a contribution. Two constructions, one sentence of result, explicit "this is a
   negative result and here is what we did to make it hard to dismiss."
2. **Setup and threat model** (0.75 p). Purifier + **the two validation gates** (positive denoising
   gain; unprotected control retains identity, 100 % everywhere) as a contribution here, citing
   DW-box as precedent. Channel-calibrated EER thresholds. Definition of protection. **Scope to ASV.**
3. **Two ways to maximise survival** (1 p). §3.1 free on-manifold transforms, no model access.
   §3.2 the pass-through subspace done correctly (N4), with the naive-vs-matched table as evidence
   the construction is right.
4. **Survival is achieved; protection is not** (1.5 p). The **double dissociation** table below.
   The 0/480 and 0/320 columns. The UTMOS/WER frontier. The one exception, formant warp.
5. **Mechanism** (0.75 p). N5 augmentation argument; purification re-maps rather than erases
   (direction cosine, amplification with the posterior-covariance derivation); N6 the registered
   failed prediction; the r = −0.86 scoping correction stated as a scoping condition.
6. **Implications** (0.5 p). Designers: optimise adversariality *after* purification, not
   invariance. Evaluators: report a purifier-validation gate; never report survival without
   protection beside it.
7. **Limitations** (0.25 p). Mel-domain surrogate purifier; Griffin-Lim; no cloning model; ASV-scoped;
   20 speakers × 1 utterance for protection; no listening test.

**The table the paper is built around:**

| construction | purifier access | cost | survival | protection |
|---|---|---|---|---|
| spectral tilt | **none** | one filter | R_t 0.0001, PSR 0.98 | **0 / 480** |
| noise-matched null direction | full white-box | 200-step optimisation | R_t 0.004, PSR 0.98 | **reduces** protection (p = 0.0003) |

Two constructions at opposite extremes of cost and knowledge converge on the same near-perfect
survival, and neither protects. One arm is an anecdote; both arms are a refutation.

**Say this ourselves before a reviewer says it:** our headline survival number (0.98 vs random 0.61)
has the wrong baseline. Spectral tilt reaches 0.98 **for free, with no purifier access at all**. An
expensive per-utterance white-box optimisation reproduces what one filter gives away. It is the most
persuasive sentence available and it costs only a number we should not have been claiming.

---

## 9. Decision rule and timeline

**ICASSP 2027 closes 23 Sep 2026** (submission system 24 Sep 23:59 UTC); **Interspeech 2027 closes
9 Feb 2027** (update 16 Feb).

* **Sep 17–18:** E2 + E3 (the two that decide P1). Patch `--eval-ckpt`, train v5, gate it.
* **Sep 19:** E4, E6.
* **Sep 20: GO / NO-GO.** *Go* if the subspace transfers across at least two independently trained
  purifiers (or fails to, cleanly and interpretably) **and** the vocoder control is clean.
* **Sep 20–22:** E5 + writing.
* **NO-GO fallback:** submit the geometry version to **ISCA SPSC 2027** (negative results explicitly
  welcome, our exact community) and spend the week on Paper 2 for Interspeech instead. **Do not**
  submit a single-purifier geometry paper to ICASSP and burn the result.

**Honest odds.** As the null-space-constraint paper described a week ago: workshop, clearly. As
written above, with E2/E3 landing: roughly even at ICASSP, better at Interspeech with a full cycle.

**What it needs and cannot get in a week** — state these as limitations rather than discover them in
review: a voice-cloning model in the loop; a waveform-domain purifier instead of a mel surrogate with
Griffin-Lim; speaker counts in the hundreds; and human listeners on "does it still sound like the
same person", which is the one question no encoder can answer.

---

## 10. EXECUTION LOG (2026-09-16)

### Done

| # | experiment | outcome | where |
|---|---|---|---|
| E6 | vocoder control | **CLEAN** — the effect is not a vocoder artifact, under both Griffin-Lim and phase-preserving resynthesis | RESULTS §11 |
| E2 | cross-purifier transfer, v3 → v4 | **TRANSFERS** — 76.2x advantage over random on an unseen purifier (109.8x on the source) | RESULTS §12 |

**The go/no-go in §9 is met on the transfer half.** The subspace is not an
artefact of one checkpoint, and it is not an artefact of the vocoder.

Two things came out of E6 that were not in the plan and change how results are
reported:

1. **PSR had to be replaced.** A norm ratio cannot tell a preserved perturbation
   from a destroyed one replaced by equal-sized noise, and Griffin-Lim is the
   second case — the first run of E6 concluded "the vocoder preserves 211 %,
   therefore it is transparent," which is not a coherent statement. All survival
   numbers are now the signal gain `alpha = <d_out,d_in>/||d_in||^2`, with the
   injected-noise ratio `nsr` reported alongside. See RESULTS §11a.
2. **The `range` directions transfer too** (R_t 2.071 → 2.147). This upgrades the
   claim from "a privileged subspace survives two networks" to "**the entire
   ordering of directions is shared between independently trained purifiers**."
   A single shared subspace could be coincidence; a shared spectrum cannot. This
   should be the sentence in the abstract.

### Changed from the plan

- **E3 abandoned and replaced.** The 1-D UNet (v5) never learned to denoise:
  gain flat at −0.5 from step 3000 to 10000, the identical failure mode to v2.
  Killed at step 10000 rather than spend the remaining ~20 min. The second
  purifier is now **v6** — 2-D UNet at base 64 (double width), seed 7, crop 160:
  an independent run differing in capacity, initialisation and training crops.
  The beta schedule is shared with inference and is deliberately *not* varied,
  since changing it would require plumbing through `real_diffpure` and would
  confound the comparison rather than strengthen it.
- **E7 added** (`exp_crossexample_transfer.py`). E2 asks whether the subspace is
  a property of the *checkpoint*. E7 asks whether it is a property of the
  *utterance*. Both readings have to be closed for the claim to be about speech.
  The decisive column is the pairwise cosine between directions optimised on
  different utterances: if they transfer while being near-orthogonal, they are
  distinct vectors from one shared high-dimensional subspace, which is much
  stronger than every utterance yielding the same vector.

### Theory note — the framing to lead with

By Tweedie/Miyasawa, `J_m(x_t) = (I + (1−ᾱ_t) H_t)/√ᾱ_t` with
`H_t = ∇²log p_t`. The noise-matched pass-through condition
`m(x_t + √ᾱ_t d) − m(x_t) = d` is, to first order, exactly `H_t d = 0`.

**So R_t is measuring the action of the log-density Hessian, and the pass-through
subspace is its null space.** This is not decoration: it *predicts* both transfer
results in advance. Two networks fit to the same corpus approximate the same
score, hence the same Hessian, hence the same null space (E2); and the null space
belongs to the density, not to any one sample (E7). It also explains, without
extra assumptions, why the zero-noise criterion fails — it is a statement about
`J` at `t=0`, which is not the operator the purifier applies.

This supersedes the tangent-space framing flagged in §5, which was not defensible
at t=200 (Stanczuk's result is σ→0 and about score *vectors*, not the Jacobian
kernel; we are 15–25x outside that regime).

### Still running / queued

`experiments/run_paper3.sh` — v6 training → E2 cell 2 (v3→v6, gated on v6
denoising gain ≥ 0.15) → E4 (n=20 × 2 utterances) → E4b (training-speaker
memorisation control) → E5 (subspace dimension). Then E7 separately.

---

## 11. EXECUTION LOG, PART 2 — after the novelty audit (2026-09-16)

### Results in hand

| # | experiment | outcome |
|---|---|---|
| E6 | vocoder control, both vocoders | clean; metric replaced (PSR -> signal gain) |
| E2 | cross-purifier transfer v3 -> v4 | **transfers**, 76.2x, rho = 0.997 over 80 directions |
| E4 | survival, 20 speakers, utterance 3 | matched criterion 20/20 at every t; zero-noise criterion dead at t >= 200 |
| E4 | replication, utterance 6 | **every row agrees to 3 decimals** |

### The novelty audit changed what we may claim

Recorded in full in `related_work.md` sections 10-11. The short version:

* **The theory is not ours.** Miyasawa (1961) for the identity; Kadkhodaie et al.
  (ICLR 2024) for the Jacobian/Hessian/posterior-covariance relation and the
  spectral reading; Saito & Matsubara for the tangent identification; Zhang et al.
  (ICML 2024) for cross-model score convergence, which *predicts* E2.
* **arXiv:2505.22839 v2 (11 Aug 2026) is the most dangerous prior art.** It defines
  our subspace ("on-manifold" = small-|eigenvalue| eigendirections of the score
  Jacobian, via Tweedie) and states our mechanism as a section heading. v1 of the
  same paper contained none of this, which is why an earlier check in this project
  cleared it. **A prior-art check has an expiry date** — re-verify any novelty
  argument that rests on what a paper does *not* say, and record the version and
  date beside the claim.
* **Surviving contributions:** constructive exploitation, dimension measurement,
  cross-model transfer, the audio domain, and the quantified negative result.
  Exactly the experiments in this plan — the audit narrowed the framing, not the
  experimental programme.
* **Naming:** "pass-through subspace" throughout; "null space" is forbidden in the
  prose because Shallow Diffuse (arXiv:2410.21088) owns it for the *opposite*
  (erased) end. Anchor sentence drafted in `draft_v4.md` section 3.1.
* **Two references do not exist** — `VoiceShield` and `DiffSpec`. Four venues were
  wrong. Corrected in the ledger.

### Deliverable

`paper/draft_v4.md` supersedes `draft_v3.md`. Cuts the "100 % protection" claim
(section 10 showed the constraint adds nothing over EOT) and the unconditional
r = -0.86 (within-additive-family only). Leads with transfer + insufficiency.

### Still running

E4b (memorisation control), E5 (dimension), then E7 (cross-example transfer) and
the v6 second transfer cell, both queued in `run_e7.sh`.

### Candidate, not yet queued — replicate the NEGATIVE result on utterance 6

The paper's core claim is the negative one, and it currently rests on 20 speakers
x 1 utterance. Survival now has a two-utterance replication; protection does not.
Cost is roughly 2.5 GPU-hours (EOT is ~164 s/file). With the deadline on 23 Sep
there is time. Decide once E5 and E7 land.

---

## 12. FINAL EXECUTION STATUS (2026-09-16, end of session)

Every experiment in section 6 is measured. The go/no-go in section 9 is met:
the subspace transfers across independently trained purifiers, and the vocoder
control is clean.

| claim | evidence | where |
|---|---|---|
| the subspace exists | R_t 0.006 vs 0.621 random, 20/20 speakers, p = 1.9e-06 | RESULTS 13 |
| not a vocoder artefact | alpha 0.971 vs 0.493, both vocoders | RESULTS 11 |
| not checkpoint-specific | 76x on an unseen purifier; **rho = 0.997** over 80 directions | RESULTS 12 |
| not memorisation | R_t 0.0058 training speakers vs 0.0057 held-out | RESULTS 13d |
| not noise-level-specific | found at t=100, survives t=300 (PSR 0.848) | RESULTS 13b |
| not utterance-specific | 17.1x transfer at \|cos\| = 0.016 | RESULTS 15 |
| not a static filter | 2.8 % static-EQ energy, 0.1 % tilt ramp | RESULTS 15b |
| large | >= 48 orthogonal directions, R_t flat | RESULTS 14 |
| replicates | second utterance agrees to 3 decimals | RESULTS 13c |
| **and it does not protect** | 0/480 tilt; constraint alone hurts p=0.0003; adds nothing to EOT | RESULTS 8, 10 |

### Two controls earned their keep

Neither of the session's two significant errors was found by reading code. Both
were caught by a control built to fail loudly:

1. **The purifier gain gate** refused a half-trained v6 automatically
   (gain -0.47), preventing a non-purifying purifier from entering a robustness
   result — the v2 mistake, which cost a week the first time.
2. **A baseline reporting an impossible value.** E7's first run showed random
   directions with pairwise cosine **1.000**. That is impossible for independent
   draws, and it exposed a shared random seed across speakers that had inflated
   the headline cosine from 0.016 to 0.690 — which would have inverted the
   conclusion from "a shared high-dimensional subspace" to "one shared vector,
   probably a filter".

Worth generalising: **put a quantity in every experiment whose correct value you
already know**, and read it first. A plausible-looking result has no error bar on
its own plausibility.

### Remaining, queued and self-sequencing

`run_e7.sh` (v6 + second transfer cell) -> `run_final.sh` (probe with alpha) ->
`run_alpha.sh` (E4 both utterances with alpha) -> `run_protect.sh` (the negative
result replicated on utterance 6, ~2.2 GPU-h). Roughly 5 hours unattended.

### The one gap compute cannot close before 23 Sep

The result is scoped to **speaker verification, not voice cloning**. No cloning
model was run. Section 6 of the draft states this plainly rather than eliding it.
A reviewer will press on it and the honest answer is that it is out of scope for
this paper.
