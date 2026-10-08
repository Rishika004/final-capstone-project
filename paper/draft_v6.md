# draft v6 — written against `FRAME_v7`, 2026-10-08

**Status: first full pass, W9 of `PLAN_V3`.** Supersedes `draft_v5.md`, which is
old-base and must not be quoted. Target: Interspeech 2027, 4 pages + 1 of
references.

**Two slots are open and marked `[PENDING]` rather than filled.** Nothing in
this draft may be submitted while either is open:
* **Control B** (`PREREG_council5_controls.md` §B) — can remove the word
  *speaker* from the title.
* **#13c** — α through DiffWave, the one structural hole in C1.

Every number below comes from RESULTS §21 onward. Page estimates in brackets.

---

## Title [provisional — see `FRAME_v7` §2]

**Every Speaker Encoder's Identity Direction Is Purifier-Hostile: Encoder
Geometry and the Measurement Limits of Purification-Robust Voice Protection**

---

## Abstract [~160 words]

Protective perturbations are added to published speech so that voice-cloning
models cannot reproduce the speaker, and diffusion-based speech purifiers
largely remove them. We ask a narrower question than whether protection
survives: *which directions does a purifier erase, and how does that relate to
what a speaker encoder reads?* Measuring a survival coefficient α for each
encoder's own identity direction through a mel-domain purifier, we find that
for five encoders spanning four architecture families — two of them never used
during optimisation — each encoder's identity direction survives **worse than a
random direction matched to it in norm and per-mel-band spectral envelope**
(10 of 10 cells, Holm-adjusted p ≤ 4·10⁻⁶). Matching a random direction's
spectral envelope to an encoder buys it nothing, so the effect is not spectral.
We then show the same purification degrades the *verifiers* used to score
protection — WavLM-SV reaches minDCF 0.77–0.91 in every cell we report — so
part of what this literature measures as a limit of protection is a property of
its instruments. We report the paradigm's measured ceiling under one-variable
controls.

---

## 1. Introduction [0.55 pp]

Anti-cloning perturbations (AntiFake [cite], and the protective-perturbation
line more broadly) add a small signal to published audio so that a zero-shot
voice-conversion or TTS model conditioned on it produces a voice an automatic
speaker verifier will reject. Two recent results have largely closed that
paradigm: De-AntiFake [ICML 2025] and VocalBridge [2026] show that running the
published audio through a diffusion purifier first recovers the speaker.

The usual response is an arms-race framing: better perturbations, stronger
purifiers. We think the interesting question is structural and comes first.
A purifier is a learned projection toward the clean-speech manifold. A speaker
encoder reads a particular direction in the same space. **Are those two
geometries related, and does the relationship differ by encoder?**

We measure it directly, with no crafting in the loop: take the gradient
direction of an encoder's identity score at a clean utterance, push it through
the purifier, and record the fraction α that survives. The comparison that
makes this a measurement rather than an anecdote is the control — a random
direction matched to the encoder's direction in **both** norm and per-mel-band
spectral envelope.

**Contributions.**

* **C1.** For five speaker encoders across four architecture families, each
  encoder's own identity direction is **purifier-hostile**: α falls below its
  own band-matched control in every cell we measure. The gap is not spectral,
  and a random direction inside the clean-speech principal subspace survives
  *best*. (§3, Fig. 1)
* **C2.** The same purification **degrades the verifiers used to score
  protection** to near-trivial operating points; automatic-verification
  acceptance overstates what a cloner actually produces; and part of what is
  reported as protection is measurable damage to the clone. Three ways this
  literature's instruments distort its findings. (§4)
* **C3.** Under one-variable controls, the paradigm's ceiling is **coverage,
  not generalisation**; the gradient estimator is a small lever, not the lever;
  and a learned refinement stage recovers most of what purification leaves.
  (§5)

We are not proposing a protection method. The perturbation recipe we analyse is
described in §2 as the artefact under study, and it is reported throughout in
the condition that flatters it — with the cloner's own encoder in the
optimisation ensemble — which we label as an upper bound.

---

## 2. Setup [0.60 pp]

**Corpus and cohort.** LibriSpeech `test-clean`, 29 speakers, 4 s clips.
Enrolment is a **different session** from the published utterance, so no
verification trial reuses a recording. A second, disjoint cohort of 31
`dev-clean` speakers is used once, for the out-of-sample check in §5.

**Encoders.** WavLM-SV (`wavlm-base-plus-sv`), ECAPA-TDNN
(`spkrec-ecapa-voxceleb`), and a Resemblyzer GE2E d-vector — the last is the
family DiffVC conditions on. Two further encoders, **CAM++** and **ERes2Net**,
are used for measurement only and never enter any optimisation. Thresholds are
calibrated at EER on the clean cross-session channel: 0.8926 (EER 5.2 %),
0.3358 (0.62 %), 0.7029 (3.45 %). **Cosine scales differ per encoder and a
single threshold is never applied across them.**

> **Reporting rule (see §4).** Every protection table in this paper prints
> minDCF in the column header, because the three encoders are not three
> comparable verifiers. Where WavLM-SV carries an identity claim we call it an
> **embedding probe**, not a verifier.

**Purifiers.** The defender optimises through an iterated mel-domain denoiser
in a DiffPure-style loop (denoise gain 0.19, DDIM, t = 50). Every headline
number is then reported under purifiers the defender did **not** train
against: **DiffWave** at two checkpoints (LibriSpeech, from De-AntiFake; and
SC09) at t ∈ {1, 3}, and De-AntiFake's full **PhonePuRe** (stage 1 + a
phoneme-conditioned refiner). The attacker's set is enumerated — eight
configurations, three architectures — and every "minimum over purifiers"
statement names that set rather than quantifying over an unbounded one.

**The protection we analyse — not a contribution.** Purifier-in-the-loop
optimisation at EOT size 1, a threshold-aware ensemble loss, segment-level
crafting after AntiFake §C2, under a psychoacoustic masking budget (margin
30 dB, SNR floor 15 dB, bandlimited to 6 kHz). An invariance-constrained
variant adds a pass-through regulariser. **Measured cost:** UTMOS 3.07 against
4.15 clean, NISQA 3.27 against 4.05, WER 8.8 % against 7.9 % (n.s.). The
protection is **audible**; no inaudibility is claimed.

**Encoder match is an axis, not a setting.** DiffVC's conditioner is a GE2E
network. Whether that encoder is in the optimisation ensemble changes results
by a lot, so two conditions are named and never collapsed: **matched-encoder
(upper bound)** and **held-out**. Every headline in §5 is matched; §4 and §5
say so in the sentence that states the number.

**Metric.** The primary endpoint everywhere is the **continuous margin**
cos(·) − τ, paired per speaker, with BCa bootstrap intervals and Wilcoxon
tests; binary rates are secondary because at the available cell sizes they move
by tens of points between identical runs. The unit is always the speaker.

**Protection, in the primary currency.** For the arm analysed throughout, the
paired clone margin (protected − unprotected, unfiltered, n = 29, pooled over
purifier conditions) is **−0.102 (WavLM-SV), −0.182 (ECAPA), −0.098
(d-vector)**, 27–29 of 29 speakers, Holm-adjusted p ≤ 1.1·10⁻⁸. Across three
crafting seeds these move by at most 0.012 — an order of magnitude below the
effects — whereas binary rates in the same cells swing by tens of points.

---

## 3. C1 — encoder identity directions are purifier-hostile [0.95 pp]

**Measurement.** For encoder *e* and clean utterance *x*, let g_e be the
unit-norm mel-domain gradient of the identity score at *x*. We push x + ρ·g_e
through the purifier R and record

  α(g_e) = ⟨R(x + ρ g_e) − R(x), g_e⟩ / ρ,

the fraction of the injected direction that survives. α is **crafting-free**:
no optimisation, no perturbation budget, no cloner.

**Controls.** Three, and the second is the one that matters.
1. `random` — isotropic, norm-matched.
2. **`band_matched`** — a random direction given the **encoder's own
   per-mel-band power envelope**. This is the control that separates "identity
   directions are special" from "the purifier removes high-frequency energy".
3. `pca_span` — random inside the top-20 principal subspace of clean mel
   (95.5 % variance, basis from a disjoint corpus).

### 3.1 Result [Fig. 1]

| direction | α (t = 50) | α (t = 200) |
|---|---|---|
| `pca_span` | **0.844** | 0.620 |
| `random` | 0.714 | 0.465 |
| band-matched controls | 0.699–0.711 | 0.453–0.465 |
| CAM++ † | 0.631 | 0.385 |
| ERes2Net † | 0.576 | 0.355 |
| WavLM-SV | 0.568 | 0.369 |
| ECAPA-TDNN | 0.463 | 0.302 |
| d-vector | 0.388 | 0.262 |

† never in the optimisation ensemble.

**Every encoder sits below its own band-matched control, in 10 of 10 cells**
(5 encoders × 2 timesteps), Holm-adjusted within timestep, p ≤ 4.0·10⁻⁶,
24–28 of 29 speakers per cell. Paired differences run −0.068 to −0.314.

**Three readings, in order of how much weight they carry.**

1. **The gap is not spectral.** Band-matched controls land at 0.699–0.711
   against isotropic random's 0.714. Giving a random direction an encoder's
   exact per-mel-band envelope buys it **nothing**. Whatever makes identity
   directions fragile is not their spectrum.
2. **It is not an artefact of our optimiser.** CAM++ and ERes2Net were never
   optimised against, and sit in the same band.
3. **The purifier preserves the subspace clean speech occupies, and identity
   points out of it.** `pca_span` survives *best* of anything we measure.

### 3.2 Ordering — the fragile half

At t = 50, all ten pairwise comparisons under one Holm family give a
**three-group** separation (WavLM-SV > ECAPA > {x-vector ≈ d-vector}), with
both middle steps at p_adj = .032. At t = 200 the middle collapses and **only
WavLM-SV separates**. x-vector and the d-vector are never distinguishable
(p = .54, .47). We therefore never state a four-way ranking, and never quote
the t = 50 ordering without its timestep.

### 3.3 What C1 explains, and the correction that goes with it

The invariance constraint's effect on protection is **encoder-dependent**, and
C1 predicts the direction: the constraint trades attack strength for purifier
survival, so it should cost most on the encoder whose identity direction
survives worst. Under mismatched purifiers at the loud budget it improves the
paired margin on WavLM-SV in all four foreign cells (−0.026 to −0.078), does
the same on ECAPA, and **reverses on the d-vector** (+0.071, +0.075).

**This must be reported with its correction in the same breath.** Holding the
d-vector *out* of the optimisation ensemble — one variable, everything else
identical — **the reversal disappears and changes sign**: +0.074/+0.079 matched
against −0.008/−0.012 held out, with non-overlapping intervals. A mixed model
agrees from a second direction: the `recipe × encoder` interaction is
p = 2·10⁻¹³ in the matched condition and **p = 0.19 held out**, on a cohort
with power to detect a matched-sized interaction.

So the reversal is a property of **optimising against the d-vector while
constrained**, not of the constraint. It is reported as a **matched-encoder
observation**. C1 still explains it, and explains why it is conditional on
ensemble membership — a sharper claim than the one it replaces, but one fitted
after a contradicted prediction, and labelled as such.

The interaction is carried **entirely by the d-vector**: ECAPA and WavLM-SV are
statistically indistinguishable in all three cohorts (p = .26, .82, .80), so
this is a two-group claim.

> **[PENDING #13c]** α is measured on the mel purifier while the phenomenon it
> explains lives on DiffWave. The waveform-domain measurement is the one
> structural hole in C1.
> **[PENDING Control B]** whether an *untrained* network of the same
> architecture is also purifier-hostile — the control that earns the word
> "speaker".

---

## 4. C2 — the measurement is not neutral [0.70 pp]

Three independent ways this literature's instruments distort its findings. All
are measured on the same base; none requires our protection to be good.

### 4.1 The purifier degrades the verifier, not just the perturbation

minDCF at p_target = 0.01 (1.0 = no better than the trivial system), 464 target
and 12 992 non-target trials per channel:

| channel | ECAPA | d-vector | WavLM-SV |
|---|---|---|---|
| clean | **0.02** | 0.39 | 0.77 |
| DiffWave-Libri t=1 | 0.11 | 0.44 | 0.81 |
| DiffWave-Libri t=3 | 0.30 | 0.51 | **0.90** |
| matched purifier t=50 | 0.10 | 0.54 | **0.91** |

**WavLM-SV sits at 0.77–0.91 in every cell this paper reports** — effectively
not better than the trivial system at a realistic operating point — while its
EER of 5.2–9.5 % reads as respectable. Its target and impostor distributions
overlap heavily (d′ = 2.48 against ECAPA's 6.16), which also explains a
long-standing oddity: it moves by large cosine amounts for small behavioural
changes.

**The consequence cuts against us, and we state it that way.** "Protection
holds on WavLM-SV" is the **weakest** of our three per-encoder claims, not the
strongest. "Protection holds on ECAPA" is the strongest, because ECAPA is the
only one of the three whose accept/reject decision is trustworthy at this
operating point. And our central *negative* result — the d-vector — is measured
on a mediocre verifier, which weakens it symmetrically.

### 4.2 Verification acceptance overstates what a cloner produces

Running De-AntiFake's full PhonePuRe against our protected audio restores the
speaker on **29 of 29 files** by ASV cosine. End-to-end DiffVC cloning from the
same purified audio tells a different story.

Primary endpoint, the paired per-speaker clone margin (protected − unprotected,
**unfiltered**, n = 29, lower is better for the defender):

| attacker's purifier | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| none | −0.150 | −0.242 | −0.136 |
| De-AntiFake stage 1 | −0.050 | −0.110 | −0.042 |
| **full PhonePuRe (1 + 2)** | **−0.030** | **−0.060** | **−0.034** |

Protection is **cut to about a fifth and does not reach zero**: pooled over
purifiers, every encoder is still significantly protected (−0.076 / −0.137 /
−0.071, 28–29 of 29 speakers, Holm-adjusted p ≤ 1.5·10⁻⁸).

The binary rate tells the same story and is quoted with its denominator and its
filter, never as a bare percentage: after the AntiFake pre-filter, clone success
under full PhonePuRe is 6/16, 3/11 and 17/24 (WavLM / ECAPA / d-vector) against
1/16, 0/11 and 7/24 unpurified. **Reduced, not removed.**

A paper scoring only the verification proxy would have reported a total defeat
where the victim suffers a partial one.

### 4.3 Part of "protection" is damage to the clone

The clone produced from protected audio is **0.32 UTMOS and 0.44 NISQA** less
natural than one from unprotected audio, with WER untouched; the gap halves
under purification. Some of what similarity metrics score as protection is the
cloner producing a worse artefact, not a different speaker.

---

## 5. C3 — the paradigm's measured ceiling [0.70 pp]

**Coverage, not generalisation.** Same files, same thresholds, one variable —
whether the cloner's encoder was in the optimisation ensemble. With **no
purifier**, clone success is **12–14 % on the two crafted-against encoders
against 75–92 % on the held-out one** (post-pre-filter, 29 speakers). Under a
purifier the contrast narrows but does not close: 21–60 % against 76–91 %.
Protection is **coverage of the encoders you happened to optimise against**,
not a property that generalises to an encoder you did not — which measures how
much any result in this literature depends on encoder overlap.

**The gradient estimator is a small lever, not the lever.** We compared BPDA
against exact gradients through the purifier under one change. Tested as the
equivalence claim it is (TOST, speaker as unit), **the pre-specified margin of
0.01 fails** — it holds in 7 of 24 cells — and the data support equivalence
only at **0.032** (median cell 0.011). The direction is systematic: exact
gradients protect better in **19 of 24 cells across both arms** (10 of 12 in
the arm we analyse), by **0.005** on average. That is an order of magnitude
smaller than the effects at stake (0.076), so the estimator cannot explain why
previously published adaptive attacks failed; but it is not zero, and we report
the bound rather than a null.

**Out-of-sample.** On 31 disjoint dev-clean speakers with independently
calibrated thresholds, the pattern reproduces directionally — all twelve cells
keep their predicted sign, and all six mild-purifier cells are significant
under Holm. We call this **directionally reproduced, not replicated**: at
n = 31, **no t = 3 cell could have resolved**, so their non-significance is
uninformative rather than disconfirming.

**A learned refinement stage.** §4.2's PhonePuRe result is also a ceiling
result: the collapse concentrates on the d-vector, C1's lowest-α encoder,
which is what C1 predicts.

---

## 6. Limitations and conclusion [0.50 pp]

**One cloner.** Every end-to-end number uses DiffVC.

**Two checkpoints of one purifier architecture** carry the transfer result;
the one architecturally different attacker (PhonePuRe) breaks it.

**Granted and unmeasured.** Codec and channel transport — our perturbation is
bandlimited to 6 kHz, exactly where a low-bitrate codec spends least — plus
multi-clip aggregation, cropping, and non-diffusion speech enhancement. These
are granted to the attacker in the threat model and not evaluated.

**A prospective test returned inconclusive, not null.** We pre-registered a
prediction for two unseen encoders and all eight cells had the predicted sign,
but all eight sit below their 80 %-power detectable effect (median power 21 %),
so the test could not have resolved them. We report it as inconclusive. The
pre-registration also under-counted its own underpowered cells, naming three
where the arithmetic says six.

**Crafting is stochastic at fixed seed.** Two runs of identical code on
identical input differ by up to 0.03 in post-purification cosine on a single
utterance, because CUDA reduction order is not replayable. Every comparison
here is paired with the speaker as the unit, where that term enters only as
√n-reduced variance — but a purifier-in-the-loop study reporting one crafting
run per arm and a within-run interval is understating its uncertainty.

**Human listening data** is not yet in. `[PENDING]`

**Conclusion.** A speech purifier does not remove an arbitrary perturbation; it
removes the directions a speaker encoder reads, more readily than it removes
directions matched to them in spectrum — for every encoder we measured, across
four architecture families, including two we never optimised against. The same
purification degrades the verifiers this field scores protection with. Both
facts are measurements, neither requires a protection method to work, and
together they say that the ceiling this paradigm is hitting is partly a
property of its instruments.

---

## Appendix — what is not in this draft and why

| | |
|---|---|
| the α ordering as a four-way ranking | §44: never separable at that resolution |
| "replicated" for the dev-clean result | two samples, read separately, never pooled |
| "BPDA ≡ exact" | the pre-specified TOST failed (§51) |
| "every effect null" for the unseen-encoder test | inconclusive at 21 % median power (§48) |
| sign-count p-values; the r = 0.961 p-value | cells are not independent (§48c) |
| "C1 gives a criterion for choosing ensemble members" | §46 killed it; council §A.4 |
| "deployment" for the matched-encoder setting | it is an upper bound |
| "realistic" for the held-out setting | withdrawn; nothing measures which is realistic |
| any MOS from the colleague panel | UTMOS/NISQA are predictors, and the panel is not naive |
