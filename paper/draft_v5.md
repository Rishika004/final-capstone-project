> **RETIRED 2026-10-04 — OLD BASE. DO NOT QUOTE ANY NUMBER FROM THIS FILE.**
> Written on dev-clean / purifier v3 / same-session thresholds / n=20, the base
> RESULTS §21 superseded. The live framing is `FRAME_v6.md`; the live numbers are
> RESULTS §21-32 (test-clean / v7 / cross-session / n=29). Kept for history only.

# Purification-Invariance Is a Transfer Mechanism, Not a Protection Mechanism

*ICASSP 2027 draft, 4 pages + references. Supersedes `draft_v4.md`, which predates
E9 (foreign-purifier transfer) and E10 (protection through a foreign purifier) and
whose thesis — "survival is the wrong objective" — is now too coarse. Every number
is on disk and reproduced in `experiments/RESULTS_ICASSP.md`; section numbers below
in the form (R§n) point there.*

> **Open items before submission.** (a) ~~Citations~~ **DONE** — all 13 verified
> against primary sources; see `paper/references.bib`. (b) The protection result is
> 20 speakers × 1 utterance; survival has two. (c) Protection is now tested
> against a real cloner (DiffVC, §5.5), but the purifier-mismatch *effect* is
> speaker-verification-only — stated in §5.5 and §7, not hidden.

---

## Abstract (~150 words)

Protective perturbations against voice cloning are removed by diffusion
purification, and the field's stated fix is to make them survive it. We show in
speech that survival is a **trade**, and that which side of it wins depends on
whether the attacker's purifier is the one the defender assumed — a variable
studied for image protections (Zhao et al. 2026) but not, to our knowledge, for
speech. We construct perturbation directions that pass a mel-domain speech
purifier essentially untouched (signal gain 0.97 vs 0.49 random), and show the
subspace they span transfers to **two published purifiers we did not train** at
86–107% of its strength on our own (20/20 speakers, four cells). Constraining a
perturbation into this subspace never protects alone. Added to purifier-aware
crafting it costs raw attack strength and improves protection against an unassumed
purifier: 7 of 12 tests after Holm correction, all on WavLM-SV and ECAPA-TDNN, with
the Resemblyzer cells null. **Against a DiffVC cloning attack the effect does not
reappear (0 of 9 tests), so it is scoped to speaker verification.** Protection
itself does transfer to cloning, but far less than a matched-encoder evaluation
suggests: excluding the cloner's own speaker encoder from crafting raises clone
success from 4–19% to 38–60%, on every encoder rather than only the excluded one.

---

## 1. Introduction

**Framing rule: state the hypothesis as the field's, with citations — never as
ours.** A paper that refutes its own idea reads as a lab notebook; one that refutes
a published premise is a contribution.

- Voice cloning from seconds of public audio. Protective perturbations are the
  deployable countermeasure: AntiFake (CCS 2023), SafeSpeech (USENIX Sec 2025),
  Voice Guard (IJCAI 2023), POP, RoVo.
- **They are broken by diffusion purification.** De-AntiFake (ICML 2025) — whose
  PhonePuRe is itself diffusion-based — and VocalBridge both strip these
  protections and both close by calling for protections that survive purification.
  That call is the premise we test.
- **In images this is already a programme:** AntiPure (ICCV 2025) formalises the
  anti-purification task; Shallow Diffuse (NeurIPS 2025 Spotlight) embeds watermarks in
  a purifier-invariant subspace; and Zhao et al. (2026) show across 2,100 editing
  tasks that protections break when the attacker's pipeline differs from the
  defender's surrogate, and call for defenses robust to that mismatch. In audio, *no protection claims robustness to diffusion purification at
  all* — existing claims target noise reduction, augmentation, speech enhancement,
  or signal recovery.
- **We test the premise directly, in audio.** It is wrong as stated, and wrong in
  an informative way.

### Contributions

1. **A constructive spectral attack on a purifier** (§3). Every existing adaptive
   attack on diffusion purification is gradient-based optimisation — DiffPure's
   adjoint method, Lee & Kim, DiffAttack, DiffBreak. We derive directions from the
   purifier's spectrum instead, and show the criterion must be evaluated at the
   noise level where purification acts.
2. **The subspace transfers to purifiers we did not train** (§4.2) — two public
   DiffWave checkpoints, foreign in architecture, weights, corpus and domain.
3. **Its effective dimension** is at least 48 orthogonal directions (§4.3), so the
   constraint is nearly free in degrees of freedom.
4. **Purification-invariance is a trade, not a protection mechanism** (§5): it
   costs raw adversarial strength, and repays that cost only under purifier
   mismatch — in speaker verification. Against a cloning model the trade's upside
   does not reappear (§5.5), and we report that rather than scoping it away.
5. **Validation gates** any purification-robustness claim should report (§2.2).

### What is ours and what is not — in §1, not §7

The geometry we build on is published and we claim none of it. The second-order
Tweedie identity is Miyasawa (1961). Relating the denoiser Jacobian to the
log-density Hessian and posterior covariance, and reading the eigenspectrum as
local invariance, is Kadkhodaie et al. (ICLR 2024). Identifying the
small-eigenvalue end with the manifold tangent space is Saito & Matsubara. That
**independently trained diffusion models converge to the same score function** —
which predicts our transfer results — is Zhang et al. (ICML 2024). Most directly,
Liu & Wei (arXiv:2505.22839 **v2**, 11 Aug 2026) define the small-|eigenvalue|
eigendirections of the score Jacobian as "on-manifold" and state that *"diffusion
model purifies off-manifold while preserves on-manifold perturbations."*

**Our step is constructive and adversarial.** They characterise the subspace; we
optimise perturbations into it, measure its dimension, test transfer across
independently trained *and* foreign purifiers, and ask what living in it buys a
defender. None of that is attempted in the prior work, and none of it has been done
in audio, where no geometric analysis of any diffusion purifier exists.

---

## 2. Setup

### 2.1 Threat model
The defender publishes `x_p = x + d`. The attacker purifies and clones. A verifier
accepts iff `cos(E(x), E(x_r)) ≥ τ`. **The defender does not know the attacker's
purifier** — §5 shows this is decisive for speaker verification (though not for
cloning, §5.5). We evaluate two verifier populations: speaker verification with
three encoders (WavLM-SV, ECAPA-TDNN, Resemblyzer d-vector), and a real cloner
(DiffVC) scored against a held-back enrolment utterance through the same three.

### 2.2 Purifiers, and the gates that validate them (Table 1)

*Ours:* residual 2-D UNet over (mel × time), DDPM-consistent training noise,
deterministic DDIM reverse process. Three independently trained instances (v3, v4,
v6 — the last at double width), denoising gains 0.189 / 0.188 / 0.192.

*Foreign:* two public unconditional DiffWave checkpoints (WaveNet on raw 16 kHz
waveform) — De-AntiFake's `purification.pkl`, trained on LibriSpeech, and philsyn's
SC09 model. Purification is DiffPure-style partial diffusion, written from the
published schedule rather than vendored.

**Two gates, reported for every purifier used:**
- **denoising gain** — an earlier purifier of ours was effectively an identity map
  (gain 0.06); everything measured against it was really measuring the mel+vocoder
  bottleneck.
- **channel EER on unprotected audio** — an earlier one destroyed the identity of
  *unprotected* speech (EER 14–23%), making any protection against it look good for
  the wrong reason.

| channel | WavLM / ECAPA / d-vector EER |
|---|---|
| clean | 3.33 / 0.03 / 0.09 % |
| ours v3 + GL, t=50 | 6.7 / 0.2 / 0.8 % |
| DiffWave-LibriSpeech t=1 | 4.16 / 0.84 / 0.09 % |
| DiffWave-SC09 t=1 | 3.98 / 0.84 / 0.10 % |

All foreign channels preserve unprotected identity at least as well as ours, so
protection measured against them is meaningful. **We recommend both gates become
standard practice.**

### 2.3 Evaluation
Thresholds at EER **through each attacker channel** — reusing thresholds across
channels is invalid. Statistics on the continuous decision margin
(`cos_after_purify − τ`), paired by file and encoder: binarising at the threshold
discards most of the evidence, and here the binary rates saturate at 95%.

---

## 3. The pass-through subspace

### 3.1 Definition, and the four names it already has

> **Anchor sentence — verbatim on first mention.** We call this the **pass-through
> subspace**: the near-zero eigenspace of `∇²log p_t`, equivalently the
> unit-eigenvalue eigenspace of the denoiser Jacobian. It is the set Kadkhodaie et
> al. call the *preserved subspace*, that Saito & Matsubara identify with the
> manifold's *tangent space*, and that Liu & Wei term the *on-manifold* directions.
> It is the complement of the null space of the denoiser Jacobian used by Shallow
> Diffuse for watermarking, which targets the *erased* directions instead.

**"Null space" must not denote our object anywhere in the paper.** Shallow Diffuse
owns that phrase for the opposite end of the same spectrum.

By Tweedie/Miyasawa, `J_m(x_t) = (I + (1−ᾱ_t)H_t)/√ᾱ_t`, so first-order
pass-through is exactly `H_t d = 0`. *(Preliminaries — attributed, not claimed.)*

### 3.2 The criterion must be noise-matched

The naive reading evaluates the Jacobian at the clean input; the purifier acts at
`x_t`. We use `R_t(d) = E_ε‖m(x_t + √ᾱ_t d) − m(x_t) − d‖²/‖d‖²`. One denoiser pass
per step, against t passes for EOT.

---

## 4. The subspace is real, large, and shared

### 4.1 Survival, 20 held-out speakers (R§13)

Signal gain `α` (§4.5), n = 20 held-out speakers. `t = 300` is **six times** the
strength the defender assumes and beyond the purifier's training range. `nsr` is
the injected-noise ratio at t = 200 — how much of the output is *not* the input
perturbation.

| direction | t=50 | t=100 | t=200 | t=300 | nsr @ t=200 |
|---|---|---|---|---|---|
| range (worst case, maximised) | 0.215 | 0.226 | 0.191 | 0.138 | 5.81 |
| random | 0.736 | 0.646 | 0.486 | 0.367 | 0.74 |
| null — zero-noise criterion | 0.871 | 0.737 | 0.533 | 0.395 | 0.53 |
| **null_t100 — noise-matched** | 0.980 | 0.969 | 0.920 | **0.829** | 0.15 |
| **null_t200 — noise-matched** | 0.986 | 0.984 | 0.975 | **0.930** | **0.11** |

Every noise-matched cell is 20/20 speakers at p = 1.9e-06 (the floor at n = 20).

**The criterion is the result, not a detail.** At t = 200 the zero-noise criterion —
the naive reading of "the subspace the purifier ignores", and what the first
implementation of this project used — buys a **1.10×** advantage over random
(0.533 vs 0.486). The noise-matched criterion buys **2.01×** (0.975). The
difference is §3.2: the purifier acts at `x_t`, not at `x`, and evaluating the
Jacobian at the clean input identifies nearly the wrong subspace.

A second reading of the same rows: the noise-matched direction is not merely larger
at the output, it is *cleaner*. Its nsr of 0.11 means almost everything that comes
out is the perturbation that went in, against 0.74 for random — three-quarters of
which is junk the purifier introduced.

Replicates on a second, disjoint utterance (R§13c) and holds on the purifier's own
training speakers (R§13d), ruling out memorisation.

> **Note on a metric change, stated because it moved a claim.** An earlier version
> of this table reported PSR, a norm ratio, and concluded the zero-noise criterion
> was *statistically indistinguishable* from random at t ≥ 200 (6/20, p = 0.12).
> Under `α` it is distinguishable — 20/20, p = 1.9e-06 — but by only 1.10×. The two
> metrics disagree because at t = 200 the zero-noise direction and a random one have
> near-identical PSR (0.604 vs 0.605) while differing in nsr (0.53 vs 0.74): a norm
> ratio counts injected noise as survival and hides the difference. `α` is the
> correct measure (R§11a) and we report it, at the cost of the sharper claim.

### 4.2 It transfers — to our purifiers, and to purifiers we did not train

*Three of ours* (v4 and v6 independently trained; v6 at double width):

| | ρ(R_t source, R_t target) | null_t200 vs random |
|---|---|---|
| v3 → v4 | **0.997** | 20/20, p=1.9e-06 |
| v3 → v6 | **0.994** | 20/20, p=1.9e-06 |

Not only do the good directions transfer — **the entire ordering does**, including
the `range` directions each network destroys most violently. A shared subspace
could be coincidence; a shared spectrum cannot.

*What this ρ is and is not.* It correlates `R_t` computed on one purifier against
`R_t` computed on another, over the same set of directions. It is a statement about
two purifiers agreeing, not about `R_t` predicting protection. We make no claim of
the latter: `R_t` is the construction objective and nothing more, and across
direction families it correlates with protection in the wrong sign.

*Two foreign purifiers* (R§16), n=20, signal gain α:

| foreign purifier | t | α pass | α random | better | retention of ours |
|---|---|---|---|---|---|
| DiffWave-LibriSpeech | 1 | 0.771 | 0.350 | 20/20 | 86% |
| DiffWave-LibriSpeech | 3 | 0.576 | 0.212 | 20/20 | 106% |
| DiffWave-SC09 | 1 | 0.788 | 0.355 | 20/20 | 87% |
| DiffWave-SC09 | 3 | 0.592 | 0.215 | 20/20 | 107% |

Retention holds in an 86–107% band across two independently trained models, two
strengths, and a large corpus mismatch (LibriSpeech vs spoken digits), crossing an
architecture and domain boundary. *Attribution:* Zhang et al. and Kadkhodaie et al.
predict score-function agreement; what is not implied is that it holds for
**optimised adversarial** directions over the **fine-grained ordering** of the
spectrum.

### 4.3 The subspace is wide enough not to constrain the attacker (R§14)

*Two sentences, not a contribution.* Applying Tramèr et al.'s greedy orthogonal
deflation — their algorithm, which also supplies the published subspace-size
explanation for transferability — the pass-through error stays flat at 1.3–1.5% of
a random direction's out to 48 mutually orthogonal directions (5 speakers, ambient
dimension 20,080). We report this as a **lower bound, not a dimension**: the sweep
stopped for budget reasons, not because the criterion degraded.

Its only role here is to foreclose the obvious defence of §5's negative half —
the constraint is not starving the optimiser of room.

### 4.4 It is not the utterance, and not a filter (R§15)

Directions optimised on one utterance pass through another 17.1× better than random
while being **near-orthogonal to each other** (|cos| = 0.016). Different vectors
from one shared subspace, not one vector reused. By energy they are 2.8%
static-EQ family and **0.1%** tilt ramp, and put *less* energy in silent frames
than random — so the construction is not rediscovering an EQ curve.

### 4.5 The measurement is not a vocoder artefact (R§11)

A norm ratio cannot distinguish a preserved perturbation from a destroyed one
replaced by equal-sized noise, and Griffin-Lim is the second case. We therefore
report the **signal gain** `α` from `d_out = α·d_in + residual`. Across the
purifier alone, α = 0.971 (nsr 0.11) vs 0.493 (0.70) for random — this is the
separate n = 5 vocoder-control experiment, which is why it differs in the third
decimal from the n = 20 figures in §4.1; the effect holds
through the full chain under both vocoders.

---

## 5. Purification-invariance is a trade, in speaker verification (R§17)

All numbers below are the decision margin, paired, n=20 **per encoder**, and use
the *same saved perturbations* evaluated through different purifiers — so nothing
varies but the attacker. Every family is Holm-corrected; we report raw and adjusted
p together, and we do not pool encoders (three measurements of one utterance are
not three samples).

**Prior work on this variable, and how ours differs.** Zhao et al.
(arXiv:2603.13028, Mar 2026) vary the gap between a defender's surrogate and the
attacker's pipeline across 2,100 editing tasks and six protection methods, and
report that protections *"optimized for a surrogate model may fail when attackers
use mismatched diffusion pipelines."* They establish mismatch as a **failure mode**
for image protections and stop there, closing with a call to *design defenses
robust to heterogeneous attackers*. We answer that call in the opposite polarity
and a different modality: a construction whose whole value is that it pays off
precisely when the attacker's purifier is not the assumed one. We also carry the
test through to an actual cloning model, where it does not hold (§5.5).

### 5.1 On its own, the constraint never protects

Constraint alone vs plain, per encoder, Holm-corrected within the family of 21
(positive = worse). Delta is the mean shift in decision margin; n=20 per cell.

| attacker's purifier | WavLM-SV | ECAPA-TDNN | d-vector |
|---|---|---|---|
| ours v3, t=50 | +0.025, p_adj 0.16 | +0.023, p_adj 1.00 | **+0.034, p_adj 0.007** |
| ours v3, t=100 | +0.020, p_adj 0.63 | +0.020, p_adj 1.00 | **+0.031, p_adj 0.007** |
| ours v3, t=200 | +0.013, p_adj 0.77 | +0.002, p_adj 1.00 | **+0.019, p_adj 0.046** |
| DiffWave-LS t=1 / t=3 | +0.002 / +0.001, p_adj 1.00 | −0.032 / +0.011, p_adj 0.64 / 1.00 | **+0.042, p_adj 0.013** / +0.022, p_adj 0.31 |
| DiffWave-SC09 t=1 / t=3 | +0.013 / +0.005, p_adj 1.00 | −0.052 / −0.009, p_adj 0.078 / 1.00 | **+0.041, p_adj 0.011** / +0.015, p_adj 1.00 |

**The constraint alone never protects, in any of 21 cells.** Its point estimate is
positive (harmful) in 18 of 21. But the stronger claim we made in earlier drafts —
that it is *significantly* harmful under the purifier it was computed on — does not
survive correction on the two strong encoders: after Holm, 5 of the 6 surviving
cells are Resemblyzer d-vector, the encoder we discount as a proxy front-end
whenever it is null elsewhere. **Reported as: never helpful, and harmful on one
encoder.** The earlier "significantly harmful" phrasing rested on pooling three
encoders over the same 20 files, which is pseudoreplication; that pooling is now
blocked in code.

### 5.2 On top of EOT, it depends entirely on whose purifier

`EOT + constraint` vs `EOT alone` (negative = protects better). The 12 foreign
cells form one Holm family; the matched rows and the no-purifier control are
separate families. Delta is the margin shift, n=20 per cell, per encoder.

| attacker's purifier | WavLM-SV | ECAPA-TDNN | d-vector |
|---|---|---|---|
| **none at all** | **+0.162, 0/20, p<10⁻⁴** | **+0.292, 0/20, p<10⁻⁴** | +0.025, 5/20, p=0.006 |
| ours v3 (the assumed one), t=200 | −0.006, 11/20, p_adj 1.00 | +0.021, 8/20, p_adj 1.00 | +0.005, 11/20, p_adj 1.00 |
| DiffWave-LibriSpeech t=3 | −0.025, 15/20, p_adj 0.16 | −0.037, 15/20, **p_adj 0.029** | +0.000, 9/20, p_adj 1.00 |
| DiffWave-SC09 t=3 | −0.045, 18/20, **p_adj 0.008** | −0.044, 17/20, **p_adj 0.012** | −0.007, 14/20, p_adj 0.61 |
| DiffWave-LibriSpeech t=1 | **−0.082, 19/20, p_adj<10⁻⁴** | **−0.088, 18/20, p_adj 0.003** | +0.009, 7/20, p_adj 1.00 |
| DiffWave-SC09 t=1 | **−0.091, 19/20, p_adj 10⁻⁴** | **−0.095, 17/20, p_adj 0.001** | +0.007, 10/20, p_adj 1.00 |


**7 of 12 foreign tests survive Holm correction, all in the same direction**,
across two independently trained purifiers. Two purifiers sharing nothing but
architecture family give effect sizes within 10% of each other. The 5 that do not
survive are **all four d-vector cells** plus WavLM at DiffWave-LibriSpeech t=3
(p=0.033, p_adj 0.16). The d-vector nulls are not noise to be rounded off: that
encoder is attacked through a proxy front-end, and it is null in every foreign cell
in both directions. So the effect is real on the two strong encoders and absent on
the third, which is how we state it.

### 5.3 The mechanism: a trade, established by the no-purifier control

Read the rows in order and the pattern is monotonic in *how far the attacker's
purifier is from the assumption*. The top row is decisive: **with nothing
purifying, the constraint loses on every single file.** So this is not generic
robustness. The constraint **spends raw adversarial strength to buy purification
survival** — a loss when nothing purifies, a wash against a known purifier (EOT's
gradients already cover that one), a win when the attacker's differs.

That is exactly what §4.2 predicts, and is the mechanism rather than an analogy:
EOT gradients encode the specific model they were computed on; subspace membership
is shared across purifiers.

**The trade is perceptually free.** Equal PESQ (2.239 vs 2.233, p=0.50), and the
constrained perturbation is *less* audible: +0.96 dB SNR (p<0.001), +0.008 STOI
(p=0.003). It cannot be dismissed as a larger perturbation.

### 5.4 Survival alone is still not sufficient

A spectral tilt is perceptually free (UTMOS 4.12 vs 4.12, 0.4% WER), passes the
purifier untouched, and protects **0 of 20 speakers across 480 trials**. Its energy
overlaps our optimised directions by 0.1% (§4.4), so these are mechanistically
unrelated constructions converging on the same near-perfect survival — and neither
protects on its own.


### 5.5 The cloning threat model: protection transfers, the mismatch effect does not

Everything above is speaker verification. The threat that motivates the field is
voice cloning, so we ran it: **DiffVC** (Popov et al., ICLR 2022) clones each
victim from a protected reference, and we score the clone against a held-back
enrolment utterance of the same speaker through all three encoders, at
EER-calibrated per-encoder thresholds.

**Pre-filter.** Following AntiFake, we score only speakers whose *unprotected*
audio successfully cloned — an attack that already fails is not protection. We
apply the filter per (encoder, attacker-purifier) cell rather than pooled; pooling
retains speakers whose unprotected attack failed in that cell and counts them as
protection successes. This leaves n=5–12 per cell and is the single largest
correction we made to these numbers.

**The cloner shares an encoder with us, so we report both settings.** DiffVC's
conditioner is the same file as our Resemblyzer encoder (§7), so crafting against
all three encoders is white-box with respect to the cloner. We therefore re-crafted
every arm against WavLM-SV and ECAPA only, holding Resemblyzer out while still
scoring it, with corpus, speakers, purifier, thresholds and budget pinned.

Clone success rate, mean across the four protected arms (lower = better; the
unprotected row is 100% by construction of the filter; n=5–12 per cell):

| attacker's purifier | ECAPA-TDNN | d-vector | WavLM-SV |
|---|---|---|---|
| none, all three crafted | 6 % | 19 % | 4 % |
| **none, cloner's encoder held out** | **50 %** | **60 %** | **38 %** |
| DiffWave, all three | 12 % | 45 % | 29 % |
| **DiffWave, held out** | **53 %** | **68 %** | **57 %** |
| ours (matched), all three | 40 % | 50 % | 32 % |
| **ours, held out** | **45 %** | **71 %** | **61 %** |

**Protection transfers to cloning, at roughly half the strength a matched-encoder
evaluation reports.** Against an unpurified attacker with all three encoders
crafted, clone success falls from 100% to 4–19% and every cell is significant
(p=0.031 to 0.0005). Hold the cloner's encoder out and the same cells read 38–60%.
Clone success rises in **9 of 9 encoder × purifier cells**. Against a matched
purifier *and* a held-out encoder, protection is largely gone.

**It collapses on all three encoders, not only the held-out one, and that is the
informative part.** WavLM-SV and ECAPA remain in the crafting ensemble yet lose 29
and 44 points. In verification the three encoders are independent verifiers. In
cloning they are partly redundant readouts of one upstream event — whether DiffVC's
conditioning embedding was moved off the victim — and that conditioner *is* the
held-out encoder. We measure the redundancy rather than assert it: pairwise Spearman
correlation of the decision margin is 0.39 to 0.59, with 73–82% binary agreement
(n=300). Dependent, not identical. **Per-encoder cloning tables therefore carry less
independent evidence than their count suggests**, so we designate ECAPA-TDNN primary
and report the others as secondary, and we do not correct across them as a family.
The same caution applies elsewhere: De-AntiFake scores with a Resemblyzer d-vector
while two of its cloning methods condition on that encoder, and RoVo crafts with one
system's encoder while reporting Resemblyzer as an evaluator.

**What survives is the ordering.** Held out and unpurified, the EOT arms still beat
plain PGD on every encoder (`full` 38/58/17%, `fullnsT` 25/50/17% against `plain`
75/58/67%). Purifier-aware crafting does real work that does not depend on encoder
match; it is the magnitude of the cloning claim that was inflated, not its
existence.

**The mismatch effect does not replicate here.** Testing `EOT+constraint` vs `EOT`
in the cloning setting, exactly as §5.2 does for ASV, gives **0 of 9 cells
significant in the predicted direction**. The one cell that reaches p<0.05
(WavLM-SV under DiffWave, p=0.047) has a *positive* delta, meaning the constraint
protects worse. So §5.2's effect is scoped to speaker verification, and we do not
claim it for cloning.

Why the two disagree is open. The most likely reason is power: n=5–12 speakers
after the filter against n=20 files in ASV, and cloning adds a whole generative
model between perturbation and score. A 0% cell at n=5 has a 95% binomial upper
bound near 45%, so these numbers bound the effect loosely in both directions. We
report the disagreement rather than the ASV result alone.

---

## 6. Implications

- **For designers:** purification-invariance is worth buying *only* as insurance
  against purifier mismatch, and it is not free — it costs attack strength. Buy it
  when you cannot know the attacker's purifier, which is the deployed case. The
  evidence for that insurance is speaker verification; §5.5 does not reproduce it
  against a cloner, so treat it as unproven for cloning rather than established.
- **What does hold end-to-end:** purifier-aware crafting cuts DiffVC clone success
  from 100% to 4–19% when the cloner's speaker encoder is in the crafting ensemble,
  and to 38–60% when it is held out. The EOT arms beat plain PGD in both settings,
  so the purification problem is real in the cloning threat model even where our
  subspace constraint adds nothing to it.
- **For anyone reporting per-encoder cloning results:** check whether your cloner's
  conditioning network is one of your encoders, and whether your evaluation
  encoders are readouts of one event rather than independent tests (§5.5). We found
  both problems in our own results and in two published pipelines.
- **For evaluators:** report a purifier-validation gate (denoising gain + channel
  EER on unprotected audio); never report survival without protection beside it;
  and **evaluate against a purifier you did not train**, since §5.2 shows matched
  and mismatched give opposite answers.

## 7. Limitations

Mel-domain surrogate for crafting (foreign purifiers used only for evaluation).
Griffin-Lim and phase-preserving vocoders; the foreign-purifier arm needs no
vocoder and is correspondingly stronger.

**Cloning is evaluated against one voice-conversion model (DiffVC) and no TTS.**
Two consequences we state rather than leave to be found. First, DiffVC's speaker
conditioner is **the same file** as the Resemblyzer encoder in our crafting
ensemble. We claim no discovery here: DiffVC's repository vendors the
Real-Time-Voice-Cloning speaker-encoder package verbatim, with that project's
licence beside the checkpoint, and the two files carry the same git blob SHA-1
(`47e405a4…`, 17,090,379 bytes), so a reader can confirm the byte-identity from
public metadata alone. What is absent is any sentence, in either project or in the
DiffVC paper, that says so — the paper attributes its speaker network only to Jia
et al. (2018). That arm is a matched-encoder result, not a transfer result,
and no prior work using this cloner has said so. **We do not merely disclose it, we
measure it** (§5.5): re-crafting with that encoder held out raises clone success
from 4–19% to 38–60%. Cross-encoder transfer numbers exist in this literature
(FreeTalk reports a full crafting × evaluation encoder matrix; RoVo excludes one
system's encoder), but no paper makes a designated held-out encoder its reported
setting, and AntiFake's discussion names systematic cross-encoder study as future
work while conceding the same overlap in its own evaluation. AntiFake's
Table 4 is the prior for how much that matters: crafting against a single encoder
protects the synthesizer built on that encoder at 99.6–100% and the other four at
0–20.6%, and their GE2E row is our case exactly. Second, the pre-filter leaves only
**5–12 speakers per (encoder, purifier) cell**, so a 0% has a 95% binomial upper
bound near 45% and no cloning number here should be read to better than ±20
points. The held-out-encoder conclusion rests on all 9 cells moving the same way,
not on any single cell. Third, WavLM-SV and ECAPA are still white-box for the
verification results; F1 removes the confound from the cloning arm only.

The mismatch effect of §5 is **speaker-verification-scoped**: it does not reappear
against cloning at either perceptual budget, and the d-vector cells are null
throughout. Protection numbers are 20 speakers × 1 utterance (survival has two);
LibriSpeech `dev-clean` is the whole corpus, which caps n and makes threshold
calibration same-session. **Perceptual quality is objective only** — PESQ, STOI,
SNR and UTMOS. We have run no listening test, so there is no human judgement of
whether protected audio still sounds like the same person, and no human check that
a clone we score as failed actually sounds unlike the victim.

---

## References

**Do not cite `VoiceShield` or `DiffSpec` — neither could be verified to exist.**
Corrected venues: Attack-VC is IEEE SLT 2021 (not AAAI); AudioPure is ICLR 2023
(not AAAI); "Voice Guard" is two words; VoiceMe is TTS personalization, not a
protection; AntiFake has no arXiv preprint — cite the ACM DOI.

**All entries verified 2026-09-16** against primary sources; the machine-readable
list is `paper/references.bib`, which records provenance and the traps in each.
Corrections made during verification, every one of which would have shipped wrong:

| item | was | is |
|---|---|---|
| DiffWave | ICLR 2021 | ICLR 2021 **oral** |
| Kadkhodaie et al. | ICLR 2024 oral | oral **+ Outstanding Paper Award** |
| Shallow Diffuse | bare preprint, "Low-Dim" | **NeurIPS 2025 Spotlight**, "Low-**Dimensional**" |
| AntiFake | short title | "AntiFake: **Using Adversarial Audio to Prevent Unauthorized Speech Synthesis**", pp. 460–474 |
| Liu & Wei | cited as "Yuezhang & Wei" | arXiv prints **"Liu Yuezhang"** (surname first) — "Yuezhang" is not the surname |
| Miyasawa | 38**(4)**:181–188 | vol. 38, pp. 181–188 — **no issue number is supported** |
| Tramèr et al. | assumed published | **preprint only**, never formally published |

Also: **do not cite `VoiceShield` or `DiffSpec`** — neither could be verified to
exist. Attack-VC is IEEE SLT 2021 (not AAAI); AudioPure is ICLR 2023 (not AAAI);
"Voice Guard" is two words; VoiceMe is TTS personalization, not a protection.

Two residual uncertainties, both low-risk and both flagged rather than hidden:
AudioPure's *track* ("Poster") rests on the ICLR virtual site and an OpenReview
mirror, since openreview.net and dblp now sit behind bot challenges — the *venue*
is solid from the camera-ready header. And Miyasawa (1961) is dated 1960 by one
source; 1961 is what the peer-reviewed reference lists print.
