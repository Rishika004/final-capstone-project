# Surviving Diffusion Purification Is Not Enough: The Pass-Through Subspace of a Speech Purifier

*ICASSP 2027 draft, 4 pages + references. Supersedes `draft_v3.md`, which was
written before the §9/§10 corrections, the E2/E6 experiments, and the novelty
audit in `related_work.md` §10–11. Every number below is on disk and reproduced
in `experiments/RESULTS_ICASSP.md`.*

> **Status markers.** `[v6]` marks the optional second cross-purifier cell,
> still training. Everything else is measured.
>
> **Two numbers to refresh before submission.** The abstract's signal gains
> (0.97 / 0.49) are from E6, n = 5 speakers at t = 200 — the only run that had the
> alpha metric when the abstract was written. The n = 20 rerun with alpha at
> t ∈ {50,100,200,300} is queued (`run_alpha.sh`); substitute those, and the
> t = 300 alpha, when it lands. The §4.1 table is still in PSR for the same
> reason. Do not ship a paper that reports survival in two different metrics —
> see RESULTS §11a for why PSR is the weaker one.

---

## Abstract (~150 words)

Protective perturbations that prevent voice cloning are removed by diffusion
purification, and the natural fix is to make the perturbation survive the
purifier. We show that survival is achievable, transferable, and insufficient.
For a mel-domain speech diffusion purifier we construct perturbation directions
that pass through it essentially untouched — signal gain 0.97 against 0.49 for a
random perturbation of equal norm, holding at six times the purification strength
the defender assumes. These directions transfer to an independently trained
purifier they were never optimised against (76x advantage over random), and the
transfer covers the *entire* spectrum: the two networks agree on which directions
survive **and** which are destroyed, with rank correlation 0.997 over 80
directions. Yet constraining a protective perturbation to this subspace does not
improve protection of speaker verification, and alone it measurably hurts.
Purification does not erase a perturbation; it re-maps it, and what survives is
no longer adversarial.

---

## 1. Introduction

**Framing rule for this section: state the hypothesis as the field's, with
citations — never as ours.** A paper that refutes its own idea reads as a lab
notebook; one that refutes a published premise is a contribution.

- Voice cloning from seconds of public audio. Protective perturbations are the
  deployable countermeasure: AntiFake (CCS 2023), SafeSpeech (USENIX Sec 2025),
  Voice Guard (IJCAI 2023), POP, RoVo.
- **They are broken by diffusion purification.** De-AntiFake (ICML 2025) — whose
  PhonePuRe is itself diffusion-based — and VocalBridge both strip these
  protections and both close by calling for protections that survive purification.
  That call is the premise we test.
- **The obvious reading of that call is "make the perturbation survive."** In the
  image domain this has become a research programme: AntiPure (ICCV 2025)
  formalises the anti-purification task; Shallow Diffuse embeds watermarks in a
  purifier-invariant subspace. In audio, *no protection claims robustness to
  diffusion purification at all* — existing robustness claims target noise
  reduction, augmentation, speech enhancement, or signal recovery.
- **We test the reading directly, in audio, and it fails.**

### Contributions

1. **A constructive spectral attack on a purifier.** Every existing adaptive
   attack on diffusion purification is gradient-based optimisation — DiffPure's
   adjoint method, Lee & Kim, DiffAttack, DiffBreak. We instead *derive* the
   perturbation directions from the purifier's own spectrum, and show the
   criterion must be evaluated at the noise level where purification acts (§3.2).
2. **The subspace transfers across independently trained purifiers**, over the
   whole spectrum, not just the surviving end (§4.2).
3. **Its effective dimension**: it admits at least 48 mutually orthogonal
   directions, all surviving, so the constraint is nearly free in degrees of
   freedom (§4.5). The deflation method is Tramèr et al. (2017); ours is its
   first application to a purifier.
4. **Survival does not confer protection** — a quantified negative result, with a
   double dissociation that separates the two axes (§4.6).
5. **A purifier that provably purifies**, with the two validation gates any
   purification-robustness claim should report (§2.2).

### What is ours and what is not — say this in §1, not §6

The geometry we build on is published and we claim none of it. The second-order
Tweedie identity is Miyasawa (1961). Its use to relate the denoiser Jacobian to
the log-density Hessian and posterior covariance, and the reading of the
eigenspectrum as local invariance, is Kadkhodaie et al. (ICLR 2024). The
identification of the small-eigenvalue end with the manifold tangent space is
Saito & Matsubara. That **independently trained diffusion models converge to the
same score function** — which predicts our transfer result — is Zhang et al.
(ICML 2024) and Kadkhodaie et al. Most directly, Yuezhang & Wei (arXiv:2505.22839
v2, 11 Aug 2026) define the small-|eigenvalue| eigendirections of the score
Jacobian as "on-manifold" and state that *"diffusion model purifies off-manifold
while preserves on-manifold perturbations."*

**Our step is constructive.** They characterise the subspace; we optimise
perturbations into it, measure its dimension, test whether it transfers across
independently trained purifiers, and ask whether living in it protects anything.
None of that is attempted in the prior work, and none of it has been done in
audio, where no geometric analysis of any diffusion model or purifier exists.

---

## 2. Setup

### 2.1 Threat model
The defender publishes `x_p = x + d`. The attacker computes
`x_r = V(P_t(mel(x_p)))` and clones from `x_r`. A verifier accepts iff
`cos(E(x), E(x_r)) >= tau`. The defender assumes `t = 50`; we evaluate
`t ∈ {50, 100, 200, 300}`, i.e. up to **6x** the assumed strength. Scope: ASV,
three encoders (WavLM-SV, ECAPA-TDNN, Resemblyzer d-vector).

### 2.2 A purifier that purifies — and the two gates (Table 1)

Residual 2-D UNet over (mel × time), DDPM-consistent training noise
`t ~ U{0..200}`, deterministic DDIM reverse process. We report:

- **denoising gain** (0.17 / 0.35 / 0.57 at t = 50/100/200), and
- **channel EER** on *unprotected* audio (6.7 / 0.2 / 0.8 % at t = 50).

*Methodological note, and a contribution in its own right.* An earlier purifier in
this project, trained on one speaker with an off-by-one in the noise schedule,
destroyed the identity of **unprotected** audio (channel EER 14–23 %). Any
protection measured against it looks robust for the wrong reason. A second one
was effectively an identity map (gain 0.06), so everything measured against it was
really measuring the mel+vocoder bottleneck. Both were caught only by these gates.
**Reporting a denoising gain and a channel EER should be standard practice for any
purification-robustness claim**, and we recommend it as such.

### 2.3 Evaluation
Thresholds at EER computed **through the attacker's own channel**. Success =
clone fails after purification **and** the unprotected control retains identity
(100 % throughout). LibriSpeech dev-clean; speakers 0–19 train the purifier,
20–39 held out. Statistics on the continuous decision margin
(`cos_after_purify − tau`), paired by speaker — binarising at the threshold throws
away most of the evidence (§4.6).

---

## 3. The pass-through subspace

### 3.1 Definition, and the four names it already has

> **Anchor sentence — use verbatim on first mention.** We call this the
> **pass-through subspace**: the near-zero eigenspace of `∇²log p_t`,
> equivalently the unit-eigenvalue eigenspace of the denoiser Jacobian. It is the
> set Kadkhodaie et al. call the *preserved subspace*, that Saito & Matsubara
> identify with the manifold's *tangent space*, and that Yuezhang & Wei term the
> *on-manifold* directions. It is the complement of the null space of the denoiser
> Jacobian used by Shallow Diffuse for watermarking, which targets the *erased*
> directions instead.

**Do not write "null space" for our object anywhere in the paper.** Shallow
Diffuse owns that phrase for the opposite end of the same spectrum; a reviewer who
knows it will read our claim backwards. (The repository name and the CSV label
`null_t200` are legacy and stay in code only.)

By Tweedie/Miyasawa, `J_m(x_t) = (I + (1−ᾱ_t) H_t)/√ᾱ_t` with `H_t = ∇²log p_t`,
so first-order pass-through is exactly `H_t d = 0`. *(Preliminaries — attributed,
not claimed.)*

### 3.2 The criterion must be noise-matched — and this is the whole result

The naive reading of "the subspace the purifier ignores" evaluates the Jacobian at
the clean input. That is the wrong operator: the purifier acts at `x_t`, not `x`.
We use

```
R_t(d) = E_eps || m(x_t + √ᾱ_t d) − m(x_t) − d ||² / ||d||²,
x_t = √ᾱ_t·mel(x) + √(1−ᾱ_t)·eps
```

One denoiser pass per step, against `t` passes for EOT. Directions are found by
Adam on the sphere `||d|| = r`.

**Table 2 is the evidence that the construction is right** (§4.1): the zero-noise
criterion is statistically indistinguishable from a random perturbation at exactly
the purification strengths that matter.

---

## 4. Results

### 4.1 Survival, 20 held-out speakers (Table 2)

`nullspace_survival_20spk_u3.csv`. PSR = norm ratio; direction cosine in brackets.

| direction | R_t(200) | t=50 | t=100 | t=200 | t=300 |
|---|---|---|---|---|---|
| range (worst case) | 2.071 | 1.513 | 1.434 | 1.115 | 0.613 (0.23) |
| random | 0.621 | 0.828 | 0.746 | 0.605 | 0.473 (0.78) |
| null, zero-noise criterion | 0.497 | 0.896 | 0.782 | 0.604 | 0.477 (0.83) |
| **pass-through, matched t=100** | 0.036 | 0.988 | 0.978 | 0.933 | **0.848 (0.98)** |
| **pass-through, matched t=200** | 0.006 | 0.992 | 0.991 | 0.982 | **0.938 (0.99)** |

Paired Wilcoxon vs random, n = 20 (1.9e-06 is the floor at n = 20):

| criterion | t=50 | t=100 | t=200 | t=300 |
|---|---|---|---|---|
| zero-noise | 20/20, 1.9e-06 | 20/20, 1.9e-06 | **6/20, p=0.12** | **13/20, p=0.57** |
| matched | 20/20, 1.9e-06 | 20/20, 1.9e-06 | 20/20, 1.9e-06 | 20/20, 1.9e-06 |

Two points. First, the zero-noise criterion — what "null space of the denoiser"
naturally means, and what the first implementation of this project used — **is no
better than a random perturbation** at t ≥ 200. It appears to win at t = 50 only
because at t = 50 almost everything survives. Second, a direction found at t = 100
still passes at t = 300, beyond the purifier's training range, so the subspace is
not an artefact of one noise level.

### 4.2 It transfers across independently trained purifiers (Figure 2)

Directions optimised against purifier v3, evaluated through v4 (independent run,
different seed, data order, validation split):

| on v4 (never optimised against) | pass-through | random | better | p |
|---|---|---|---|---|
| R_t(200) | 0.0082 | 0.6242 | 20/20 | 1.9e-06 |
| PSR (t=200) | 0.980 | 0.604 | 20/20 | 1.9e-06 |
| direction cosine | 0.992 | 0.803 | 20/20 | 1.9e-06 |

Advantage over random: **109.8x** on the source purifier, **76.2x** on the unseen
one. `[v6]` second cell at different capacity.

**The sentence for the abstract:** Spearman correlation between pass-through error
on v3 and on v4, over all 80 directions, is **ρ = 0.997 (p = 5e-87)**. The two
networks agree not only on which directions survive but on the entire ordering —
including `range`, the directions each destroys most violently (2.07 → 2.15). A
single shared subspace could be coincidence; a shared spectrum cannot.

*Attribution.* Zhang et al. and Kadkhodaie et al. already show independently
trained diffusion models converge to the same score function, so agreement is
*predicted*. What is not implied by that work: the agreement holds for **optimised
adversarial** directions rather than random or isotropic ones, and over the
fine-grained ordering of the whole spectrum rather than the coarse score field.

### 4.3 The measurement is not a vocoder artefact (§11 of RESULTS)

A norm ratio cannot distinguish a preserved perturbation from a destroyed one
replaced by uncorrelated noise of equal size, and Griffin-Lim is the second case.
We therefore decompose `d_out = α·d_in + residual` and report the **signal gain α**
with the injected-noise ratio `nsr`:

| t = 200 | vocoder only | purifier only | full chain |
|---|---|---|---|
| random | 0.675 (3.30) | 0.493 (0.70) | 0.362 (5.02) |
| pass-through | 0.918 (1.93) | **0.971 (0.11)** | 0.871 (2.55) |

The round trip alone passes only 57–68 % of a random perturbation while injecting
error 3–4x what survives, so it is not transparent — but it degrades all
directions roughly equally and cannot manufacture the gap. The separation happens
at the purifier. Confirmed under both Griffin-Lim and phase-preserving
resynthesis.

### 4.4 It is shared across utterances, and it is not a filter

Directions optimised on one utterance, evaluated on another (6 speakers):

| | R_t | vs random | PSR t=200 |
|---|---|---|---|
| random | 0.624 | 1.0x | — |
| optimised **on** this utterance | 0.0042 | **148.8x** | 0.983 |
| optimised on **another** | 0.0365 | **17.1x** | 0.939 |

Mean |cos| between directions from different utterances: **0.016** — near
orthogonal (independent random draws give 0.005). They are *different vectors*
that each pass through the other's utterance 17x better than random, not one
shared vector reused. Combined with §4.5 this is a shared, high-dimensional
subspace.

**The directions are not a static filter.** By energy, the time-constant family
(all static EQ curves, the tilt among them) accounts for 2.8 % of an optimised
direction and the single tilt ramp for 0.1 %; a random direction gives 0.4 % and
0.0 %. They also put *less* energy in the quietest quartile of frames than random
does (0.198 vs 0.250), so they are not hiding in pauses. This matters for §4.6:
the tilt and the optimised direction reach the same survival by mechanistically
**different** means, which is what makes the double dissociation an argument
rather than two variants of one trick.

### 4.5 How large is the subspace? At least 48 dimensions (Figure 3)

Greedy orthogonal deflation (Tramèr et al. 2017), 5 speakers, ambient dimension
20,080:

| k | 1 | 12 | 24 | 36 | 48 |
|---|---|---|---|---|---|
| R_t | 0.0088 | 0.0085 | 0.0091 | 0.0090 | **0.0082** |
| % of random | 1.4 | 1.4 | 1.5 | 1.5 | **1.3** |
| loop PSR | 0.973 | 0.975 | 0.975 | 0.973 | **0.977** |

**Flat.** R_t at k = 48 is lower than at k = 1; the fitted slope is +4e-06 per
direction. We report this as a **lower bound, not a dimension** — the sweep stopped
at 48 for budget reasons, not because the criterion degraded.

**This forecloses the obvious defence of the null-space idea.** If the subspace
were narrow, "it did not protect because the constraint left the optimiser no
room" would explain §4.6 away. It is not narrow: there are at least 48 orthogonal
directions to work in, all surviving. The constraint is cheap and still buys
nothing, so the failure is not a budget problem.

### 4.6 Survival does not protect — the double dissociation

**The table the paper is built around:**

| construction | purifier access | cost | survival | protection |
|---|---|---|---|---|
| spectral tilt | **none** | one filter | R_t 0.0001, PSR 0.98 | **0 / 480** |
| pass-through direction | full white-box | 200-step optimisation | R_t 0.006, PSR 0.98 | **reduces** protection (p = 0.0003) |

Two constructions at opposite extremes of cost and knowledge converge on the same
near-perfect survival, and neither protects. One arm alone is an anecdote; both
arms are a refutation.

- The tilt is **perceptually free** — UTMOS 4.12 against 4.12 for the original,
  0.4 % word errors — passes the purifier untouched, and protects **0 of 20
  speakers across 480 trials**.
- The pass-through constraint **alone** makes protection *worse* (15–18 of 20
  files worse; d-vector p = 0.0003).
- Purifier-aware crafting (EOT through the reverse process, exact gradients)
  **does** work: margin Wilcoxon p < 0.0001 at both perceptual budgets. Adding
  the pass-through constraint on top of it changes nothing (p = 0.31–0.97).

**Say this before a reviewer does.** Our headline survival number has the wrong
baseline: a spectral tilt reaches the same 0.98 for free, with no purifier access.
An expensive white-box optimisation reproduces what one filter gives away.

### 4.7 Mechanism

ASV encoders are trained with channel, reverb and codec augmentation, so they are
*already invariant* to the cheap on-manifold directions that survive purification.
Survival and adversariality are therefore selecting for opposite things. The one
exception is formant/McAdams warping — you cannot augment it away without
destroying the speaker label — and it is the one on-manifold transform that
protects.

**A registered failed prediction:** we predicted reverberation would protect
strongly. It protects 0 %. We state this as a failed prediction rather than
omitting it.

This is a concrete instance of Stutz et al. (CVPR 2019) — on-manifold perturbations
behave as generalization errors rather than adversarial examples — not a new
principle. On the temporal axis, Kamble et al. observe the complementary failure:
local sensitivity does not imply attack success because later denoising absorbs
the perturbation. Ours is the spectral axis: a perturbation that survives intact
still does not protect.

---

## 5. Implications

- **For designers:** optimise adversariality *after* purification, not invariance
  *to* it. Invariance is cheap, transferable, and orthogonal to the objective.
- **For evaluators:** report a purifier-validation gate (denoising gain + channel
  EER on unprotected audio). Never report survival without protection beside it.

## 6. Limitations

Mel-domain surrogate purifier, not an image-domain DiffPure score model.
Griffin-Lim and phase-preserving vocoders, no neural vocoder. No cloning model —
ASV-scoped. 20 speakers × 2 utterances. v3 and v4 share an architecture and
corpus, differing in initialisation, data order and validation split, so the
demonstrated claim is that the subspace is a property of the *training procedure*
rather than of a checkpoint — not yet of speech diffusion purifiers in general.
Objective quality metrics plus a small listening test.

---

## References

**Do not cite `VoiceShield` or `DiffSpec` — neither could be verified to exist.**
Corrected venues: Attack-VC is IEEE SLT 2021 (not AAAI); AudioPure is ICLR 2023
(not AAAI); "Voice Guard" is two words; VoiceMe is TTS personalization, not a
protection. AntiFake has no arXiv preprint — cite the ACM DOI.

Miyasawa 1961 · Kadkhodaie et al. ICLR 2024 (2310.02557) · Saito & Matsubara
(2510.05509) · Zhang et al. ICML 2024 (2310.05264) · Yuezhang & Wei (2505.22839
**v2**) · Shallow Diffuse (2410.21088) · LOCO Edit NeurIPS 2024 (2409.02374) ·
Stanczuk et al. ICML 2024 (2212.12611) · Kamkari et al. NeurIPS 2024 (2406.03537) ·
Tramèr et al. 2017 (1704.03453) · Stutz et al. CVPR 2019 (1812.00740) ·
Kamble et al. CVPR 2026 W (SAFE) · DiffPure ICML 2022 (2205.07460) · Lee & Kim
ICCV 2023 (2303.09051) · DiffAttack NeurIPS 2023 (2311.16124) · DiffBreak NeurIPS
2025 (2411.16598) · Athalye et al. ICML 2018 · AntiPure ICCV 2025 (2509.13922) ·
AntiFake CCS 2023 (10.1145/3576915.3623209) · De-AntiFake ICML 2025 (2507.02606) ·
VocalBridge (2601.02444) · SafeSpeech USENIX Sec 2025 (2504.09839) · RoVo
(2505.12686) · ClearMask AsiaCCS 2025 (2508.17660) · VoiceCloak AAAI 2026
(2505.12332) · AudioPure ICLR 2023 (2303.01507) · DAP IEEE SPL (2310.14270) ·
WavLM · ECAPA-TDNN · Resemblyzer · Johnston masking · UTMOS · LibriSpeech
