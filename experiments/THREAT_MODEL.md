# Threat model

*Rewritten 2026-10-05 for council #7. The July version is superseded; it is
kept only in git history. Every number here comes from the corrected base
(LibriSpeech test-clean, cross-session calibration, n = 29, purifier v7,
budget m30_f15, `--seg-partials`), i.e. RESULTS §21–40.*

**What changed, and why.** The July document described a different system than
the one we now have, in four ways that each flattered it:

| July said | Why it was wrong |
|---|---|
| the defender's advantage is **white-box access to the purifier** | the contribution lives in the *mismatch*: the attacker's purifier is one we did not train against (§33), and our own §29 shows a purifier we did not anticipate recovers the speaker |
| protection parameters are **secret** from the attacker | violates Kerckhoffs's principle, and nothing in the method depends on secrecy |
| success requires **PESQ > 3.0** | the shipping arm measures 2.2, so the stated criterion was never met by the thing being shipped; PESQ is also reference-based and penalises a deliberate voice change (§40) |
| δ lies in the **null space** of the purifier's Jacobian | measured α is 0.57–0.41 per encoder, not 0; it is a *purification-invariant subspace*, approximately the subspace clean speech occupies (§37d) |

---

## 1. Parties

**User (defender).** Holds a clean utterance **x** and wants to publish it
without enabling unauthorised cloning. Publishes **x_p = P(x)**.

**Attacker.** Collects x_p, optionally purifies it with **R**, and clones with
**G**:  x_r = R(x_p),  x̃ = G(x_r).

**Verifier.** A speaker encoder **E** with an operating threshold τ. The clone
is *accepted* when cos(E(x̃), E(x_enrol)) > τ, where x_enrol is a **clean,
different-session** utterance of the user that the attacker never sees.

Three encoders are scored, with thresholds calibrated at EER on the clean
channel, cross-session, on the evaluation population:

| encoder | τ | EER |
|---|---|---|
| WavLM-SV (`microsoft/wavlm-base-plus-sv`) | 0.8926 | 5.2 % |
| ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | 0.3358 | 0.62 % |
| Resemblyzer d-vector (GE2E) | 0.7029 | 3.45 % |

The cosine scale differs per encoder, so **a single threshold must never be
applied across them**; the field's 0.25 convention is reported for ECAPA only.

---

## 2. Knowledge, under Kerckhoffs's principle

| | defender knows | attacker knows |
|---|---|---|
| clean audio x | yes | **no** (only x_p) |
| the protection P, its code and its parameters ψ | yes | **yes** |
| the per-utterance random seed of P | yes | no |
| the defender's in-loop purifier R_d | yes (chooses it) | yes |
| the attacker's purifier R_a | **no** | yes (chooses it) |
| the cloning model G | **no** | yes |
| the verifier's encoder E | the public ones | yes |

**The protection is public.** P, ψ and the budget are assumed published; only
the instance seed is unknown, and §31 shows the seed changes continuous
quantities by ≈0.01, so secrecy buys nothing and is not claimed.

**The purifiers are not the same model.** The defender optimises through a
purifier of their choosing (ours: an iterated mel denoiser in a DiffPure-style
loop, v7, denoise gain 0.19, DDIM, t = 50). The attacker uses whatever they
like. Every headline number is reported under a purifier the defender did
**not** train against — DiffWave (LibriSpeech, De-AntiFake's checkpoint; and
SC09) at t = 1 and t = 3. Results under the matched purifier are an upper bound
and are labelled as such.

---

## 3. Encoder match is an explicit axis, not a setting

The cloner used throughout is DiffVC, whose speaker conditioner is a GE2E
network — the same family as the Resemblyzer d-vector. Whether that encoder is
in the crafting ensemble changes the result by a lot, so it is reported as two
named conditions and never collapsed into one:

* **matched-encoder (upper bound)** — the ensemble includes the encoder the
  cloner conditions on. Comparable to AntiFake, which crafts against four
  encoders including GE2E and evaluates on a GE2E-based synthesiser. It is
  *not* called "deployment".
* **held-out** — the cloner's encoder is excluded from crafting. Stronger than
  anything published, because it models an attacker whose identity encoder we
  do not possess.
  *(The word "realistic" is withdrawn, council v4 B1. It was a claim about
  which condition deployment resembles, and nothing here measures that. The
  conditions are named by what they do, not by how plausible they are.)*

**Which condition each reported result lives in — council v4 B1.** This table
is the honest answer and it is not flattering. `--craft-encoders all` is the
default in `run_rebase_mismatch_m30_seg.sh:70` and in the dev-clean runner, so
**the matched-encoder arm is where almost everything sits**:

| result | condition |
|---|---|
| §27, §30e (C2's constraint grid) | **matched** |
| §29e (PhonePuRe end-to-end), §40 (quality), §41 (front end) | **matched** |
| §45 (#5 dev-clean replication) | **matched** |
| §42g/h (operating points) | purifier channels, no crafting arm |
| §22b (the coverage contrast) | **both**, one variable — this is the point of it |
| §47 (Control A, the held-out C2 grid) | **held-out** |
| §32, §37, §43, §44, §46b (α) | crafting-free; the condition does not apply |

So every verdict sentence drawn from §27, §29e, §40, §41 or §45 carries
"matched-encoder (upper bound)" or it is wrong. §47 is the one C2 cell measured
held out, and it reverses the headline (the d-vector reversal disappears), which
is why the distinction is a reported quantity rather than a caveat.

The **gap between them is a reported quantity**, not an inconvenience: clone
success is 12–14 % on crafted-against encoders against 75–92 % on the held-out
one (§22b). That gap measures how much any protection result in this literature
depends on encoder overlap, which almost nobody states.

---

## 4. Attacker capabilities, and the ones we grant for free

The attacker may:

1. **Collect** published utterances.
2. **Purify** with any R, including models released after the protection.
3. **Refine.** Apply a learned repair stage. §29: De-AntiFake's full PhonePuRe
   (stage 1 + 2) restores the speaker on 29/29 files; end-to-end clone success
   rises from 6/0/29 % to 38/27/71 %. Reduced, not removed.
4. **Retry.** Draw the purifier several times and keep the best. §28c measures
   best-of-five as worth +0.020–0.037 to the attacker. We grant this: our own
   ship-iterate was selected on a *single* purification draw (§36), so the
   asymmetry runs against us and the reported protection is a lower bound.
5. **Differentiate.** §30: BPDA and exact gradients through the purifier give
   the same protection (1 of 21 Holm cells, Δ −0.009). The gradient estimator
   is not what limits an adaptive attacker.
6. **Clone** with any G.

The attacker may **not** obtain x, and may not read the per-utterance seed.

---

## 5. Defender objectives, stated so they can fail

**(a) Imperceptibility — a measured budget, not a PESQ claim.**
δ is constrained by a psychoacoustic magnitude cap (masking margin 30 dB) and
an SNR floor of 15 dB, bandlimited to 6 kHz. The honest perceptual statement is
the measured one (§40, n = 29, no-reference, loudness-matched):

| | shipping arm | clean original |
|---|---|---|
| UTMOS | 3.07 | 4.15 |
| NISQA MOS | 3.27 | 4.05 |
| WER (ground truth) | 8.8 % | 7.9 % (n.s.) |

So: content is preserved, naturalness costs about one point, and the protection
is **audible**. No claim of inaudibility is made, and the old PESQ > 3.0
criterion is withdrawn rather than restated — it was never met. Whether 3.07 is
acceptable to publish is a question only the listening test can answer, and it
is open.

**(b) Purification-invariance — a subspace, not a null space.**
The requirement is that the component of δ the encoder reads survives R. It is
not satisfied exactly: the survival coefficient α of each encoder's own
identity direction through v7 at t = 50, on the reported cross-session cohort,
is 0.568 (WavLM-SV), 0.463 (ECAPA), 0.388 (d-vector), against 0.714 for
isotropic noise and 0.844 for a direction inside the clean-speech principal
subspace (§43, §46). Two encoders that were never crafted against sit in the
same band — CAM++ 0.631, ERes2Net 0.576 — so this now holds for **five
encoders across four architecture families**, each below its own band-matched
control (§46b). Every encoder's identity direction is *purifier-hostile*; they
differ in degree, and that difference is the paper's subject.

*(Supersedes the 0.573 / 0.490 / 0.406 triple, which is the §37 cohort. Only
one α table may be quoted — two tables in two files is the §25 failure mode.)*

**(c) Inimitability.** After R and G, cos(E(x̃), E(x_enrol)) < τ.

---

## 6. Success, defined as the minimum over the attacker's options

A defence that holds against one attacker configuration and fails against
another has failed. So, for a fixed utterance:

> **Defence succeeds** iff **an external quality floor** is met **and**
> max over { purifier R ∈ 𝓡₇, seed s ∈ S, refinement on/off } of
> cos(E(G(R_s(x_p))), E(x_enrol)) < τ,
> reported **per encoder E**, never pooled.

**The quality conjunct, rewritten so it can fail — council v4 B8.** The
previous version read "the budget in (a) is met", and (a) *is* the crafting
constraint: the psychoacoustic cap and the SNR floor are enforced by projection
inside the optimiser, so every file it emits satisfies them **by
construction**. Withdrawing PESQ > 3.0 (rightly, it was never met) deleted the
only conjunct that could ever be false, leaving a success criterion with a
vacuous half. That is not a conservative error; it makes the definition
unfalsifiable on the axis reviewers care most about.

The floor is therefore **external to the optimiser and stated as a number the
shipping arm might miss**:

| | floor | shipping arm (§40) | met? |
|---|---|---|---|
| WER, ground-truth transcript | ≤ +3 points over clean | 8.8 % vs 7.9 % (+0.9, n.s.) | **yes** |
| UTMOS | ≥ 3.0 | 3.07 | **yes, by 0.07** |
| NISQA MOS | ≥ 3.0 | 3.27 | **yes** |
| naive-listener "same speaker?" | majority still same speaker | **not yet measured** | **open** |

Three of the four are met and the margin on UTMOS is 0.07, which is inside
nothing — it is a single point estimate with no interval, on a predictor, not
an opinion score. The fourth is open and only the listening panel can close it.
So the criterion is now one the shipping arm **could** fail, and on one row it
has not yet been shown to pass.

Two consequences we accept:

* Taking the max over seeds is why `fails_all_seeds` is recorded beside the
  mean (§28).
* Taking the max over 𝓡₇ means §29's PhonePuRe row is part of the verdict, not
  a footnote: on the d-vector the defence fails there.
* Reporting per encoder is why there is no single headline number. The honest
  summary is that the defence holds on WavLM-SV and ECAPA under a mismatched
  purifier and fails on the d-vector, and the paper's contribution is the
  *explanation* of that split rather than the split itself.

### 6a. 𝓡₇ enumerated — council v4 B9

The old wording quantified over "purifier R ∈ 𝓡" without ever saying what 𝓡
contains. Under Kerckhoffs with permanently published audio the set an attacker
may draw from is **unbounded and includes models released after the
protection**, so a max over it is not something any paper can report. What we
measure is a max over a **seven-element set**, named here, and the verdict is
labelled that way everywhere:

| # | R | representation | ours? | sections |
|---|---|---|---|---|
| 1 | identity (no purifier) | — | — | §26b, §27c, §41c |
| 2 | purifier v7, DDIM, t = 50 | mel | **matched** | §21–28, §30 |
| 3 | purifier v7, DDIM, t = 200 | mel | **matched** | §21–28, §30d |
| 4 | DiffWave-LibriSpeech (De-AntiFake ckpt), t = 1 | waveform | foreign | §27, §33, §45, §47 |
| 5 | DiffWave-LibriSpeech, t = 3 | waveform | foreign | §27, §33, §45, §47 |
| 6 | DiffWave-SC09 (philsyn ckpt), t = 1 | waveform | foreign | §27, §33, §45, §47 |
| 7 | DiffWave-SC09, t = 3 | waveform | foreign | §27, §33, §45, §47 |
| 8 | PhonePuRe stage 1 + 2 (stage 1 = #5 at their t = 3) | waveform + learned refiner | foreign | §29 |

Eight configurations, **three architectures** (one mel denoiser, one waveform
diffusion model at two checkpoints, one phoneme-conditioned refiner), and the
DiffWave rows are two checkpoints of a single architecture (§33e). So:

> **Write:** "the minimum over an eight-configuration, three-architecture
> measured set, the strongest of which (PhonePuRe stage 1 + 2) defeats the
> defence on the d-vector."
> **Never:** "min over purifiers", "robust to purification", or any phrasing
> that implies the set is the attacker's rather than ours.

### 6b. Capabilities granted in principle and not measured — council v4 B10

R4 asked for this list at two consecutive councils. It is now part of the
document rather than an answer given verbally:

| capability | why it matters here | status |
|---|---|---|
| **Codec / channel transport** (Opus, AAC, telephony) | δ is bandlimited to 6 kHz, which is exactly where a low-bitrate codec spends least; a codec is also the single most likely thing to happen to published audio | **unmeasured.** R4's first question for two councils. |
| **Multi-clip aggregation** | the attacker who collects ten clips can average speaker embeddings; our per-utterance seed is independent across clips, so the perturbation may average away while identity does not | **unmeasured** |
| **Cropping / re-segmentation** | segment crafting optimises the worst 1.6 s partial; an attacker who crops may discard it | **unmeasured** |
| **Speech-enhancement preprocessing** (not diffusion: spectral subtraction, RNNoise, Demucs) | a cheaper and far more widely deployed denoiser than anything in 𝓡₇ | **unmeasured** |
| **Resampling ±2 %** | front-end confound | **measured**, §41e |
| **Purifier retry / best-of-five** | attacker patience | **measured**, §28c |
| **Exact gradients through the purifier** | adaptive attack | **measured**, §30 |
| **A second cloner** | every end-to-end number is DiffVC | **unmeasured**; mandatory if the paper moves to SPSC |

The four unmeasured rows are **granted to the attacker in the threat model and
not evaluated**, which is a limitation of the evaluation and not a property of
the defence. None of them is cheap enough to add before the GPU stop, and
claiming robustness to any of them would be unsupported.

The **continuous** margin `cos − τ`, paired per speaker, is the primary
endpoint everywhere; binary rates are secondary because they discard most of
the evidence and, at the cell sizes available, move by tens of points between
identical runs (§31b).

---

## 7. Scope

**In scope.** Mel-domain and waveform-domain diffusion purification; a learned
phoneme-conditioned refinement stage (PhonePuRe); zero-shot voice conversion
(DiffVC); three public speaker encoders; English read speech, 4-second clips,
LibriSpeech test-clean.

**Out of scope, and stated as limitations rather than future work.**

* **Purifiers in other representations.** EnCodec-latent purification
  (VocalBridge) is untested here. The invariant subspace we measure is a
  property of a mel denoiser and need not transfer.
* **One cloner.** Every end-to-end number uses DiffVC. A second cloner is the
  first thing to add, and is mandatory if the paper moves to SPSC.
* **Two checkpoints of one purifier architecture.** The transfer result (§33)
  is DiffWave twice, not a property of speech purifiers in general — and §29's
  architecturally different attacker breaks it.
* **Clip length and language.** 4 s, English, read speech. Longer clips give
  the partial-averaged encoders more windows and would change the d-vector
  result in an unknown direction.
* **No adaptive attacker that targets the invariance constraint itself.** §30
  removes the gradient estimator as the explanation for prior failures, but an
  attacker who optimises against our published objective is not evaluated.
