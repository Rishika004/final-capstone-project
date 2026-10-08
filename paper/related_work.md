# Related work and running bibliography

Maintained per the standing rule: novelty-check every method before building it,
and record the citation and the delta at the time. Last updated 2026-09-15.

---

## 0. Novelty ledger (what is ours, what is taken)

| Idea | Status | Closest prior work |
|---|---|---|
| Additive perturbation in the "null space" of a purifier | **taken / misframed** | null space of an x0-predictor is what a purifier *erases*; the surviving subspace is the tangent space (Stanczuk 2022; Demystifying 2505.22839) |
| On-manifold parametric protection (filtering + style transfer + reverb) | **TAKEN** | **ClearMask, ASIA CCS 2025** — same mechanism, same motivation |
| Echo/reverb perturbation targeting formant bands | **taken** | EchoMark, APIT 2026 |
| Formant/VTLN warping to hide speaker identity | **taken (different goal)** | McAdams anonymisation, VoicePrivacy B2 baseline |
| "On-manifold perturbations survive diffusion purification" as theory | **SUPERSEDED — see section 11.2. This row was right about arXiv:2505.22839 v1 and is WRONG about v2** (posted 11 Aug 2026), which says exactly this: *"Diffusion model purifies off-manifold while preserves on-manifold perturbations"* (`on-manifold` 27x, `eigen` 22x, vs 0 and 0 in v1). The theory IS now published for images. This does not strengthen our novelty; it is our most dangerous prior art | see 11.2 |
| Purification amplifies some perturbations | **taken (empirically)**, but we can now *derive* it | 2505.22839 reports purification increases l2 distance to the clean sample, rejecting the "denoises toward clean" hypothesis. Our addition: J_m = posterior covariance, so gain > 1 exactly where the posterior is more uncertain than the noise removed |
| Purification-aware (adaptive) protection **for voice** | **PUBLISHED TWICE, AND THEY REPORT IT FAILS** — see below. ~~Our delta is the *gradient estimator*~~ — **RETRACTED 2026-09-22 (RESULTS 30): BPDA and exact gradients give the same protection on our purifier; the estimator is not the delta.** What differs from their attempts is untested (their attacker was the full pipeline, RESULTS 29; budget/objective). Our conclusion against their STAGE 1 is the opposite of theirs; against their full pipeline protection is reduced, not removed | **De-AntiFake §5.4** (BPDA+EOT sizes 1/5/10/15, 150 steps, AttackVC on DiffVC; also adjoint) and **VocalBridge §V-E** (BPDA+EOT over 1/5/10/15 runs, StyleTTS2 + x-vector) |
| Survival of a perturbation anti-correlates with protection | **PARTIALLY NOVEL** — the qualitative insight is published in two domains; the *signed, quantified* version in speech is ours | Purification Paradox (CVPRW 2026); **SyncBreaker 2604.08405** coins "pseudorobustness"; Adversarial Purification of Information Masking 2311.15339 |
| Purifier validation before a robustness claim | **PARTIALLY NOVEL** — the principle is published; only our speech-specific two-gate instantiation is new | **DW-box, ICLR 2025 (arXiv 2404.14309)** separates stochasticity-driven from purification-driven robustness; Removability of Speaker-Adversarial Perturbations (2510.09504); Lee & Kim ICCV 2023 |
| Purification-resistant protective perturbation (method) | **taken (image domain)** | **AntiPure, arXiv 2509.13922** — anti-purification guidance; ours is the speech analogue |
| **Noise-free / on-manifold protection evaluated under adaptive diffusion purification** | **OPEN — nobody has run it** | ClearMask never tests purification (0 mentions); De-AntiFake / VocalBridge only attack *additive* protections; AntiPure is images and additive |
| Residual perturbation scale vs attack strength | **compatible, not contested** | Purification Paradox measures *within* one crafting family; we measure *across* families. State this explicitly rather than claiming a conflict |

---

### 0b. THE PIVOT: prior work already tried adaptive protection and concluded it fails

Verified directly from both papers on 2026-09-16.

* **De-AntiFake, ICML 2025, §5.4 "Adaptive Protection".** Verbatim: *"treat the purification
  process as an identity transformation during backpropagation"* (BPDA), with *"Expectation Over
  Transformation (EOT) on BPDA adaptive protection, using EOT sizes of 1, 5, 10, and 15"*,
  150 steps, AttackVC protection against DiffVC. They also try the **adjoint method**. Conclusion:
  *"even at an EOT size of 15, our method maintains a dSVA above 0.8 ... even under white-box
  conditions, designing adaptive protection for our purification method remains a challenge for
  protectors."*
* **VocalBridge, 2026, §V-E "Adaptive Protection".** Independently repeats it: BPDA
  (*"treats the purifier as an identity mapping during backpropagation"*) + EOT averaged over
  *"1, 5, 10, and 15 stochastic runs"*, StyleTTS2 cloning with x-vector verification. Conclusion:
  *"Even with white-box knowledge, developing effective adaptive protections against our
  purification method is still difficult."*

**Why this is an opportunity rather than a scoop.** Both use BPDA — the purifier approximated as
the identity in the backward pass — and De-AntiFake additionally uses the adjoint method.
**DiffBreak (arXiv 2411.16598, NeurIPS 2025) showed the adjoint method is a flawed gradient
estimator for diffusion purification**, and BPDA-as-identity discards the purifier's Jacobian
entirely, which is precisely the object that determines what survives. We differentiate *exactly*
through the full reverse process with gradient checkpointing, and reach the opposite conclusion
(70 % -> 90 %).

**The paper's claim must therefore be reworded** from "we propose purification-aware voice
protection" (published) to: *two independent papers concluded adaptive protection against
purification is futile; both estimated gradients with BPDA or the adjoint method; with exact
gradients the conclusion reverses.* That is a correction of a published negative result, which is
stronger than a new method — **but it raises the evidential bar**: we must show our 90 % is not an
artefact of a weaker purifier. **Highest-value remaining experiment: re-run against PhonePuRe
(De-AntiFake's purifier, code at https://github.com/cyberrrange/de-antifake).**

## 1. Voice protection (defence side)

- **AntiFake** — Yu et al., *CCS 2023*. Additive adversarial perturbation that pushes
  speaker embeddings away; the canonical baseline.
  https://cybersecurity.seas.wustl.edu/paper/AntiFakeCCS23.pdf
- **ClearMask: Noise-Free and Naturalness-Preserving Protection Against Voice Deepfake
  Attacks** — Wang, Chen, Guo, Wang, Ding, Yan. *ASIA CCS 2025*, arXiv 2508.17660.
  **The direct precedent for on-manifold protection.** Three stages: (i) mel-spectrogram
  masking / selective frequency filtering, (ii) audio style transfer, (iii) optimised
  artificial reverberation. Plus *LiveMask* for streaming. Surrogate ensemble encoder for
  transferability. Evaluated on YourTTS, DiffVC, AGAIN-VC and black-box APIs; ECAPA for
  verification; MOS for quality (no PESQ).
  **Its adaptive attacker attempts to remove the reverberation — dereverberation only.**
  Zero mentions of purification, DiffPure, or denoising-based recovery. Its only
  "diffusion" references are to DiffVC, a *synthesis* model.
  https://arxiv.org/abs/2508.17660
- **EchoMark** — *APIT 2026*. Structured echo perturbations over the formant range
  500 Hz - 4 kHz, echo delays 25-800 ms; no model access required.
  https://dl.acm.org/doi/10.1145/3796130.3796142
- **Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection** —
  arXiv 2507.12932.
- **SafeSpeech: Robust and Universal Voice Protection** — arXiv 2504.09839.
- **VoiceCloak: Multi-Dimensional Defense against Diffusion-based Voice Cloning** —
  arXiv 2505.12332.
- **SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background
  Noise** — arXiv 2511.16114.
- **Mitigating Unauthorized Speech Synthesis for Voice Protection** — arXiv 2410.20742.
- **FreeTalk: plug-and-play black-box defense against speech synthesis** — arXiv 2509.00561.

## 2. Purification attacks on voice protection (the threat we model)

- **De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks** —
  Fan et al., *ICML 2025*, arXiv 2507.02606. First systematic evaluation of protective
  perturbations under a threat model that includes purification; two-stage purification
  with phoneme-guided refinement. Targets **additive** protections.
  https://arxiv.org/abs/2507.02606
- **VocalBridge: Latent Diffusion-Bridge Purification for Defeating Perturbation-Based
  Voiceprint Defenses** — arXiv 2601.02444. Time-conditioned 1-D UNet in EnCodec latent
  space; notes that existing purifiers target ASR rather than speaker verification.
  https://arxiv.org/abs/2601.02444
- **DiffPure: Diffusion Models for Adversarial Purification** — Nie et al., *ICML 2022*,
  arXiv 2205.07460.

## 3. Adaptive attacks on diffusion purification (image domain; our crafting technique)

- **DiffAttack** — *NeurIPS 2023*, arXiv 2311.16124. Segment-wise forward-backward
  gradients through the DDPM loop within bounded memory.
- **DiffHammer** — *NeurIPS 2024*. Selective attacks + gradient grafting to avoid the
  gradient dilemma under stochastic purification.
- **Robust Evaluation of Diffusion-Based Adversarial Purification** — Lee & Kim,
  *ICCV 2023*, arXiv 2303.09051.
- **Unlocking the Potential of Adaptive Attacks on Diffusion-Based Purification /
  DiffBreak** — arXiv 2411.16598, *NeurIPS 2025*. Shows the adjoint method is a flawed
  adaptive attack; provides reliable differentiation through DBP.

### 3b. Purification-resistant protection (image domain — closest prior art to our method)

- **Towards Robust Defense against Customization via Protective Perturbation Resistant to
  Diffusion-based Purification (AntiPure)** — arXiv 2509.13922, Sep 2025. **Closest prior
  art to our purifier-aware crafting.** Instead of trying to preserve the perturbation
  through purification, it targets the purification model directly with additional
  guidance. Image / diffusion-customization domain; additive perturbations.
  Our contribution relative to it: the speech domain, a threshold-aware multi-encoder
  loss, and the per-frame loudness floor.
- **The Purification Paradox: Dissecting and Exploiting Generative Vulnerability Bands** —
  *CVPR 2026 Workshop (SAFE)*. Timesteps with high local sensitivity are not the most
  effective injection points, because later denoising steps absorb the perturbation;
  what matters is whether the injected signal survives the *remaining* reverse process.
  Also reports a quantitative relationship between residual perturbation scale and attack
  capability — **which our r = -0.86 appears to contradict across crafting methods.**
  Worth engaging directly in the discussion.
- **Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable
  Diffusion?** — arXiv 2312.00084.
- **Targeted Attack Improves Protection against Unauthorized Diffusion Customization** —
  arXiv 2310.04687.
- **Positive-Incentive Noise Predictor for Adversarial Purification in Speaker
  Verification** — arXiv 2607.00899. Purification *for* ASV robustness; relevant to the
  attacker's toolbox in speech.
- **Diffusion-based Adversarial Purification from the Perspective of the Frequency
  Domain** — arXiv 2505.01267.

## 4. Manifold geometry of purification (our theoretical spine)

- **How Do Diffusion Models Improve Adversarial Robustness? (a.k.a. Demystifying ...)** —
  arXiv 2505.22839. **CITATION CORRECTED 2026-09-16.** Previously cited here for an
  on-manifold/off-manifold decomposition. **It makes no such claim** — the words manifold,
  on-manifold, off-manifold and tangent do not occur in the paper. Its two mechanisms are
  (i) gradient masking induced by internal randomness, and (ii) *general* image-space
  compression when randomness is controlled. It is still a valid citation for one thing we
  use: purification **increases** the l2 distance to the clean sample, "rejecting the
  hypothesis that purification denoises perturbed images closer to the clean ones".
  Superseded text follows, retained struck through for the record: ~~Compression-rate analysis showing
  off-manifold perturbations are compressed while **on-manifold perturbations are largely
  preserved** ("Diffusion model purifies off-manifold while preserves on-manifold
  perturbations"). Also documents an *expansion* effect under sampling randomness
  (l2 radius 1.004 -> 3.282), which matches our observation that range-space directions
  are amplified ~4x.
  https://arxiv.org/html/2505.22839
- **Stanczuk et al., 2022** — diffusion models and manifold dimension; the score's null
  space at low noise spans the data-manifold tangent space. Basis for computing the
  pass-through subspace from the *correction* rather than the x0-prediction.
- **Adversarial Examples are Misaligned in Diffusion Model Manifolds** — arXiv 2401.06637.
- **Adversarial Purification with the Manifold Hypothesis** — *AAAI 2024*, arXiv 2210.14404.
- **Transcending Adversarial Perturbations: Manifold-Aided Adversarial Examples with
  Legitimate Semantics** — arXiv 2402.03095.
- **On-manifold adversarial attack based on latent space substitute model** —
  *Computers & Security*, 2022.
- **Origins of Low-Dimensional Adversarial Perturbations** — Dohmatob & Guo, arXiv 2203.13779.
- **Subspace Defense: Discarding Adversarial Perturbations by Learning a Subspace for
  Clean Signals** — *LREC-COLING 2024*, arXiv 2403.16176.
- **NPAT: Null-Space Projected Adversarial Training** — arXiv 2409.11754. Null-space
  projection used for *training*, not for protective perturbations.

## 5. Voice anonymisation (precedent for on-manifold voice transforms)

- **Speaker anonymisation using the McAdams coefficient** — Patino et al., arXiv 2011.01130.
  VoicePrivacy **B2 baseline**: LPC analysis, pole phases raised to power alpha ~ U(0.5, 0.9),
  shifting formants. Signal-processing only, no training data. Preserves ASR (WER barely
  degraded). *Goal is anonymisation, not anti-cloning, and purification is not considered.*
- **VoicePrivacy Challenge evaluation plans** — 2020 (arXiv 2205.07123), 2022
  (arXiv 2203.12468), 2024 (arXiv 2404.02677); **Third VoicePrivacy Challenge**
  (arXiv 2601.11846).
- **Vocal Tract Length Perturbation for Text-Dependent Speaker Verification** —
  arXiv 2011.12536. VTLP as *augmentation*.
- **Speaker anonymization by pitch shifting based on time-scale modification** (2022).

## 6. Physical / parametric audio attacks

- **Real-time, Universal, and Robust Adversarial Attacks Against Speaker Recognition
  Systems** — *ICASSP 2020*, arXiv 2003.02301.
- **Attack on practical speaker verification system using universal adversarial
  perturbations** — arXiv 2105.09022.
- **Robust Audio Adversarial Example for a Physical Attack** — arXiv 1810.11793.
  RIR convolution inside adversarial training for over-the-air robustness.
- **AS2T: Arbitrary Source-To-Target Adversarial Attack on Speaker Recognition** —
  arXiv 2206.03351.
- **Room Impulse Responses help attackers to evade Deep Fake Detection** — arXiv 2409.14712.

## 7. Models, data, metrics

- **WavLM** — Chen et al., 2022; `microsoft/wavlm-base-plus-sv` x-vector head.
- **ECAPA-TDNN** — Desplanques et al., 2020 (SpeechBrain `spkrec-ecapa-voxceleb`).
- **GE2E / Resemblyzer** d-vector — Wan et al., 2018.
- **LibriSpeech** — Panayotov et al., 2015. **LJSpeech** — Ito & Johnson, 2017.
- **PESQ** (ITU-T P.862), **STOI** — Taal et al., 2011.
- **Johnston**, 1988 — psychoacoustic masking threshold (our Bark-domain masker).
- Suggested additions for the naturalness claim: **UTMOS** / **NISQA** / **DNSMOS**
  (no-reference MOS prediction) and Whisper-based WER for content preservation.

## 8. Added 2026-09-16 after independent prior-art audit (were missing)

- **Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic
  Perspective** — *ICLR 2025*, arXiv 2404.14309. **DW-box protocol**; robust accuracy falls
  47.27 % -> 16.80 % once stochasticity is removed. The precedent for our purifier-validation idea.
- **A Study of the Removability of Speaker-Adversarial Perturbations** — arXiv 2510.09504.
  Speech-domain removability study across ignorant / semi-informed / well-informed attackers.
- **SyncBreaker** — arXiv 2604.08405. Coins **"pseudorobustness"**: attacks look robust when
  purification removes useful speech along with the noise. Precedent for our survival-vs-protection
  point.
- **RoVo** — arXiv 2505.12686. Embedding-level voice perturbation explicitly designed to resist
  removal by speech enhancement; closest "robust-by-design" voice defence.
- **Diffusion-Based Adversarial Purification for Speaker Verification** — arXiv 2310.14270.
  The speech DiffPure; cite as the purifier lineage we implement.
- **MDD** — arXiv 2508.19180. Speech purification reporting clean-data EER controls
  (4.0 % clean vs 18.0 % adversarial) — directly relevant to our channel-EER gate.
- **Silencer / Silence is Golden** — arXiv 2506.01591. Explicit "anti-purification loss".
- **Rethinking and Red-Teaming Protective Perturbation** — arXiv 2406.18944.
- **Proactive deepfake defence survey** — *ACM CSUR* 2026, 10.1145/3771296. Names
  purification-resilience and **metric mismatch** as open challenges.
- **Degrading Voice** survey — arXiv 2512.06304. CloneShield (2505.19119), Prosody Disruptor
  (IEEE Access 2026), VOICY (WWW 2026 Companion) as recent protection baselines.


## 9. Theory citations (added 2026-09-16 after theory audit)

The correct theoretical spine. Tweedie/Miyasawa gives
`J_m(x_t) = (I + (1-abar_t) * Hess log p_t(x_t)) / sqrt(abar_t)`, so a clean-signal
perturbation d enters the reverse process as `sqrt(abar_t) d` and is **passed through**
exactly when `Hess log p_t d = 0`, **erased** exactly when `Hess log p_t d = -d/(1-abar_t)`.
Survival and removal are the flat and steepest-curvature eigenspaces of ONE operator, the
log-density Hessian **at the noise level where purification happens**. That is what makes
"null space" an accurate word — of the score Jacobian, never of the denoiser network.

- **Batzolis, Stanczuk & Schonlieb — Your diffusion model secretly knows the dimension of the
  data manifold** — arXiv 2212.12611. Thm 5.1 / Cor 5.2. **Cite ONLY for the sigma -> 0 limit.**
  It concerns the span of score *vectors*, not the kernel of the score *Jacobian*, and its own
  Limitations section disclaims larger t. Their experiments use sigma ~ 0.01 on unit-scaled
  data; our t=200 has sigma_VE = 0.72 against a log-mel std of 2.8-4.8, i.e. 15-25x outside
  that regime. **Do not write "tangent space" next to a t=200 number.**
- **Kamkari et al. — A Geometric View of Data Complexity: LID Estimation with Diffusion
  Models (FLIPD)** — *NeurIPS 2024*, arXiv 2406.03537. Same "rank of the score Jacobian at
  small t" object.
- **Denoising: a powerful building block for imaging, inverse problems and machine
  learning** — arXiv 2409.06219. Survey containing the Tweedie/Miyasawa identity and the
  denoiser-Jacobian = posterior-covariance result that lets us *derive* amplification.
- **EigenScore: OOD detection using posterior covariance in diffusion models** —
  arXiv 2510.07206. Spectral use of the same identity.
- **The Information Dynamics of Generative Diffusion** — arXiv 2508.19897. DDIM is the
  probability-flow ODE; perturbations along unstable eigendirections are amplified
  exponentially along the trajectory. Explains why amplification compounds under DDIM.
- **Ambrogioni — Statistical Thermodynamics of Generative Diffusion Models** —
  arXiv 2310.17467. Symmetry-breaking at intermediate t, where amplification is strongest.

### Two honesty caveats our own code forces

1. **Our denoiser is time-unconditional** (`MelUNet2D.forward(self, x)` takes no timestep;
   `train_purifier_v2.py` samples t but never passes it). So Tweedie returns the score of a
   *mixture* marginal over noise levels, not `p_t` for a single t. State this.
2. **The residual parameterisation changes the condition.** With `x0 = x + U(x)`,
   `J_m = I + J_U`, so pass-through requires `J_U d = ((1-sqrt(abar))/sqrt(abar)) d ~= 0.232 d`
   at t=200 — an **eigenvalue** condition, not a kernel condition. This *derives* our own
   finding that the zero-noise regulariser is a no-op: it targets eigenvalue 0 when the
   required eigenvalue is 0.23.

---

# 10. NOVELTY AUDIT, 2026-09-16 — the theory section is not ours

A verified prior-art sweep (every citation below fetched and quoted, not inferred
from titles) found that **the theoretical framing we were about to lead with is
already published**, twice. This section records what must change. It is a
demotion, not a retraction: the paper is still publishable, but the contribution
sits somewhere other than where we had put it.

## 10.1 What we cannot claim

### The Tweedie/Hessian identity — PUBLISHED

> Kadkhodaie, Guth, Simoncelli, Mallat. *Generalization in diffusion models
> arises from geometry-adaptive harmonic representations.* **ICLR 2024 (oral).**
> arXiv:2310.02557

Prints the identity verbatim (VE parameterisation):
`grad f*(y) = Id + sigma^2 grad^2 log p_sigma(y) = sigma^-2 Cov[x|y]` — the same
object as our `J_m = (I + (1-ab_t) H_t)/sqrt(ab_t)`. It also eigendecomposes it
and gives the spectral reading we thought was ours: *"Small eigenvalues
lambda_k(y) reveal local invariances of the denoising function: small
perturbations in the noisy input along the corresponding eigenvectors e_k(y) do
not affect the denoised output."*

The second-order Tweedie identity itself is **Miyasawa (1961)**, *An empirical
Bayes estimator of the mean of a normal population*, Bull. Inst. Internat.
Statist. 38:181-188 (the issue number '(4)' seen in some citations is not
supported by any source we could reach; drop it). We cannot claim the identity.

### The tangent/normal spectral reading — PUBLISHED

> Saito, Matsubara. *Be Tangential to Manifold: Discovering Riemannian Metric for
> Diffusion Models.* arXiv:2510.05509 (2025)

Abstract, verbatim: *"the spectral structure of this Jacobian separates tangent
and normal directions of the data manifold."*

### Cross-model convergence — PUBLISHED, and it PREDICTS our E2

> Zhang, Zhou, Lu, Guo, Wang, Shen, Qu. *The Emergence of Reproducibility and
> Generalizability in Diffusion Models.* **ICML 2024.** arXiv:2310.05264

Abstract, verbatim: *"different diffusion models consistently reach the same data
distribution and scoring function regardless of diffusion model frameworks, model
architectures, or training procedures."* Kadkhodaie et al. show the same at the
denoiser level for **non-overlapping training sets**: *"two denoisers trained on
sufficiently large non-overlapping sets converge to essentially the same denoising
function."*

**Consequence for section 12 of RESULTS_ICASSP.md.** We wrote that the Tweedie
framing "predicts both transfer results in advance," presenting that as a strength
of our theory. It is a strength of *their* theory. Our rho = 0.9967 verifies a
published prediction in a new threat model; it does not discover cross-model
agreement. That sentence must be rewritten before it reaches a reviewer.

### Greedy orthogonal deflation (our E5) — PUBLISHED as a method

> Tramer, Papernot, Goodfellow, Boneh, McDaniel. *The Space of Transferable
> Adversarial Examples.* arXiv:1704.03453 (2017)

Iteratively searches for orthogonal adversarial directions to estimate subspace
dimensionality (~25 dims typical; 44 orthogonal directions on MNIST), and argues
that larger subspaces are likelier to intersect between two models — **which is
also the published explanation for transfer.** This is the precedent for both our
E5 method and our E2 logic. Must be cited for both.

## 10.2 The name collision — fix this in the abstract, not a footnote

> Li, Wenda; Zhang, Huijie; Qu, Qing. *Shallow Diffuse: Robust and Invisible
> Watermarking through Low-Dimensional Subspaces in Diffusion Models.*
> **NeurIPS 2025 (Spotlight).** arXiv:2410.21088

Embeds a watermark in the **null space of J** — *"a significant proportion of
lambda*dx lies in the null space of J, so that J(x_t)dx ~= 0"* — specifically so
it survives diffusion purification and regeneration attacks.

**This is the opposite end of the same spectrum from ours, and both are called
"null space."**

| | Shallow Diffuse | ours |
|---|---|---|
| subspace | null(J), eigenvalue lambda ~ 0 | lambda ~ 1 eigenspace of J, i.e. null(H) |
| what happens to a perturbation | **erased** by the denoiser | **passed through** intact |
| design goal | invisible to generation | invariant under purification |

A reviewer skimming "null space + diffusion purification + survives" will conclude
we were scooped. Two actions: **(a)** stop using "null space" for our object in
the paper — "pass-through subspace" throughout, with the distinction made
explicitly in the introduction; **(b)** the repo name `nullspace_voice_protection`
and the CSV label `null_t200` are now misnomers. Keep them in code for continuity
with existing results, but never let "null space" denote our object in the prose.

## 10.3 What survives as genuinely ours

1. **The connection of this known geometry to adversarial purification.** No paper
   links the denoiser-Jacobian spectrum to purification robustness. Searches for
   `"adversarial purification" AND "null space"` and `"diffusion purification" AND
   "subspace"` returned zero arXiv hits.
2. **A constructive spectral attack.** Every existing adaptive attack on DiffPure
   is gradient-based optimisation — DiffPure's own adjoint method (Nie et al.,
   ICML 2022, arXiv:2205.07460), Lee & Kim (ICCV 2023 oral, arXiv:2303.09051),
   DiffAttack (Kang, Song, Li, NeurIPS 2023, arXiv:2311.16124), DiffBreak
   (Kassis, Hengartner, Yu, NeurIPS 2025, arXiv:2411.16598). None derives
   directions from the purifier's spectrum.
3. **Everything in audio.** No purification-robust voice protection exists, and no
   geometric analysis of any audio diffusion model or purifier exists. This is
   our strongest ground.
4. **Spectrum-wide transfer.** Zhang et al. predict the score fields agree; they
   do not show that the *fine-grained ordering of optimised adversarial
   directions*, including the most-erased ones, is shared. rho = 0.9967 over 80
   directions spanning 2.5 orders of magnitude is a sharper statement than
   "the models agree."
5. **The quantified negative result** (see 10.4 for how it must be framed).

## 10.4 The negative result needs a conceptual antecedent

> Stutz, Hein, Schiele. *Disentangling Adversarial Robustness and Generalization.*
> **CVPR 2019.** arXiv:1812.00740

On-manifold adversarial examples exist but are **generalization errors**, a
different object from regular adversarial examples. This is the closest prior
statement of "on-manifold != adversarial," and it is the framing a reviewer will
reach for. Our result must be presented as a concrete, quantified instance of this
principle in the purification threat model — *not* as a new principle.

Counter-tension worth citing: arXiv:2210.00430 argues on-manifold examples attack
*better*, which is the disagreement our result sits inside.

## 10.5 The image-domain precedent for our application

> Yang, Cao, Duan, He. *Towards Robust Defense against Customization via
> Protective Perturbation Resistant to Diffusion-based Purification* (AntiPure).
> **ICCV 2025.** arXiv:2509.13922

*"we formalize the anti-purification task."* Mechanisms are Patch-wise Frequency
Guidance and Erroneous Timestep Guidance — **guidance/optimisation, not a spectral
construction.** Our claim narrows to: first in **audio/speech**, and by a
**spectral construction** rather than loss guidance. Expect "isn't this AntiPure
for audio?" and answer it in the related-work section pre-emptively.

## 10.6 Bibliographic corrections — errors currently in our reference list

| item | we had | verified |
|---|---|---|
| Attack-VC (Huang, Lin, Lee, Lee) | AAAI 2021 | **IEEE SLT 2021**, arXiv:2005.08781 |
| AudioPure (Wu, Wang, Ping, Nie, Xiao) | AAAI 2023 | **ICLR 2023**, arXiv:2303.01507 |
| VoiceGuard | one word | **"Voice Guard"**, IJCAI 2023, DOI 10.24963/ijcai.2023/535 |
| VoiceMe | listed as a protection | a **TTS personalization** paper (Interspeech 2022, arXiv:2203.15379) — not protection |
| **VoiceShield** | assumed to exist | **UNVERIFIABLE — do not cite.** Only a MobiSec 2025 program-book PDF; no authors, abstract or DOI |
| **DiffSpec** | assumed to exist | **NOT FOUND — do not cite.** Likely a confusion with DiffWave / DiffSpeech / DualPure |
| AntiFake | arXiv preprint assumed; short title | **no arXiv preprint exists**. Full title **"AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech Synthesis"**, Yu, Zhai, Zhang, ACM CCS 2023, **pp. 460-474**, DOI 10.1145/3576915.3623209 |
| De-AntiFake | "not diffusion-based" | **wrong — it is.** PhonePuRe stage 1 is a pretrained unconditional diffusion model (DiffWave, fine-tuned on LibriSpeech); stage 2 is phoneme-guided score-based diffusion on complex spectrograms |
| VocalBridge | — | unpublished preprint (Jan 2026), arXiv:2601.02444, **no venue** |

Two nearest-neighbour audio protections we were missing and reviewers will find:
**RoVo** (Kim et al., arXiv:2505.12686) and **SafeSpeech** (Zhang et al., USENIX
Security 2025, arXiv:2504.09839). Note that every existing robustness claim in
audio protection is against the *wrong* threat — SafeSpeech and POP claim
robustness to noise reduction and augmentation, RoVo to speech enhancement,
ClearMask to signal recovery. **None claims robustness to diffusion purification**,
which is exactly the gap.

## 10.7 Open — do not write these sentences until closed

* **The Purification Paradox** (Kamble, Sao, Ramachandra, CVPR 2026 SAFE
  workshop). CVF returned HTTP 403. Metadata suggests it distinguishes signal
  survival from attack success — **potentially the closest prior art to our
  negative result.** Being retrieved now.
* **arXiv:2505.22839** body claims. The abstract is verified; the sharper
  "on-manifold preserved / off-manifold compressed" sentences came from an HTML
  extraction and have burned us before (section 0b). Being re-verified character
  by character.
* The Q3 negative result rests on arXiv title/abstract metadata plus web search.
  arXiv has no full-text search, so a paper doing this in a body section could
  have been missed. A Google Scholar full-text sweep of the DiffPure citation
  graph would close it.

---

# 11. THE TWO OPEN ITEMS FROM SECTION 10.7, CLOSED (2026-09-16)

## 11.1 The Purification Paradox — NOT a threat, cite as complementary

> Kamble, Sao, Ramachandra. *The Purification Paradox: Dissecting and Exploiting
> Generative Vulnerability Bands in Diffusion Trajectories.* **CVPR 2026
> Workshops (SAFE), pp. 420-428.** IIT Bhilai / NTNU Gjovik.

CVF's HTTP 403 was a user-agent block, not a paywall. Retrieved.

It does distinguish survival from attack success, explicitly. Abstract, verbatim:
*"We analyse the reverse diffusion trajectory and identify a phenomenon that we
term the Purification Paradox: **timesteps with strong local sensitivity do not
necessarily yield effective attacks, because subsequent denoising steps can
suppress or absorb the perturbation.**"* Conclusion: *"effective attacks depend
not only on local sensitivity, but also on whether the injected signal survives
the remaining reverse process."*

**It makes the converse point on a different axis, and does not narrow our claim.**

| | Kamble et al. | ours |
|---|---|---|
| variable | **timestep** — where along the reverse trajectory to inject | **subspace** — which direction to perturb |
| finding | perturbation does **not** survive, so the attack fails | perturbation **does** survive, and the attack still fails |
| logical content | survival is **necessary** | survival is **not sufficient** |
| target | a DDPM *generator*, via noise-prediction space | a *purifier* defending speaker verification |
| transfer shown | across unseen **classifiers** | across independently trained **purifiers** |

Literal string counts in their full text: `tangent` 0, `subspace` 0, `eigen` 0,
`Jacobian` 0, `Hessian` 0, `null` 0. `manifold` appears 4 times, all in passing.
**No geometric or spectral analysis at all.**

Cite in one sentence as the temporal-axis analogue: *"Kamble et al. observe on the
temporal axis that local sensitivity does not imply attack success, because later
denoising absorbs the perturbation; we observe the complementary failure on the
spectral axis — a perturbation that survives purification intact still does not
translate into protection."*

## 11.2 arXiv:2505.22839 — THE PAPER WAS REWRITTEN, AND v2 IS NOW OUR BIGGEST THREAT

**Both of this project's earlier readings were correct. They were reading
different versions of the same paper.** Submission history from the abs page:
**[v1] 28 May 2025; [v2] 11 Aug 2026.** No v3. Section 0b of this ledger recorded
that the words manifold/on-manifold/off-manifold/tangent appear nowhere in it, and
treated that as *strengthening* our novelty. That was true of v1 and is false of
v2, which is what `/abs/` and `/html/` now serve by default.

Literal regex counts over tag-stripped full text of both versions:

| token | v1 (28 May 2025) | v2 (11 Aug 2026) |
|---|---|---|
| `manifold` | 1 | **63** |
| `on-manifold` | 0 | **27** |
| `off-manifold` | 0 | **29** |
| `eigen` | 0 | **22** |
| `subspace` | 0 | 2 |
| `Jacobian` | 4 | 7 |
| `tangent` | 0 | 0 |
| `Hessian` | 0 | 0 |
| `null space` / `nullspace` | 0 | **0** |
| doc length | 73.5k chars | 138.0k chars |

### What v2 now contains, verbatim

* **Our Tweedie -> Jacobian -> eigenvalue chain:** *"By Tweedie's formula, v_i is
  also an eigenvector of the Jacobian of the score function J_s(x_t), with the
  eigenvalue lambda~_i = -lambda_i/(1-ab_t)."*
* **Our subspace definition, under a different name:** *"Define the eigendirection
  with small absolute eigenvalue |lambda~_s| of the Jacobian of the score J_s(x)
  as the on-manifold direction, and the large absolute eigenvalue |lambda~_l|
  direction as off-manifold."*
* **Our Claim 4 mechanism, as a section heading:** *"Diffusion model purifies
  off-manifold while preserves on-manifold perturbations."*
* *"the CR begins at the off-manifold value and eventually converges to the
  on-manifold value, demonstrating that off-manifold perturbations are compressed
  while on-manifold components are preserved."*
* They even cite **GAAS** (Gradient Aligned Adversarial Subspaces, Tramer et al.)
  for counting orthogonal directions that fit in an adversarial region — adjacent
  to our E5.

**Revised verdict: this, not Kadkhodaie et al., is the single most dangerous paper
in the review.** It anticipates the substance of our theory framing and the
mechanism of our negative result, in the same mathematical language.

### What v2 still does not do — verified by string search

* **No attack construction.** `craft` 0, `restrict` 0. They observe the preserved
  subspace; they never build a perturbation inside it.
* **No cross-model transfer.** `independently trained` 0, `surrogate` 0,
  `different checkpoint` 0. All 13 uses of "transfer" concern EOT across
  *randomness configurations of the same model*.
* **No dimension measurement.** A scalar compression rate under isotropic noise
  plus an idealised two-level eigenvalue structure. `intrinsic dimension` appears
  as an assumption, never an estimate.
* **No audio.** `audio` 1 (passing), `speech` 0.
* **Never says "null space"** — 0 occurrences. Their word for our set is
  "on-manifold".

### Required narrowing

Claim 1 can no longer be stated as an identification; v2 identifies the same set.
Restate as:

> *"Yuezhang & Wei independently characterise the small-|eigenvalue|
> eigendirections of the score Jacobian as 'on-manifold' and show they are
> preserved under purification. We take the constructive step: we optimise
> perturbations into this subspace, measure its effective dimension, and test
> whether it transfers across independently trained purifiers — none of which they
> attempt."*

Surviving contributions: **constructive exploitation, dimension measurement,
cross-model transfer, and the entire audio domain.** With ICASSP closing 23 Sep
2026, v2 is unambiguously public prior art and must be cited as such.

**Process lesson worth keeping.** A prior-art check has a timestamp. We verified
this paper once, correctly, and the verification silently expired when the authors
posted a substantially expanded version. Any citation whose novelty argument
depends on what a paper does *not* say must be re-checked against the current
version before submission, and the version and date recorded next to the claim.

## 11.3 Naming — keep "pass-through subspace", and anchor it on first use

What each neighbouring paper calls the two ends of the spectrum:

| paper | matrix | preserved end (ours) | erased end |
|---|---|---|---|
| Kadkhodaie et al. 2310.02557 | denoiser Jacobian | "preserved subspace"; "tangent subspace" | "local invariances" |
| Saito & Matsubara 2510.05509 | score Jacobian | "tangent space, small eigenvalues" | "normal space, large eigenvalues" |
| LOCO Edit 2409.02374 | PMP Jacobian | "semantic subspace" | — (`nullspace` 32x, but for an unrelated editing-mask projector) |
| Shallow Diffuse 2410.21088 | PMP Jacobian | "low-dimensional subspace" | **"null space"** (J dx ~= 0) |
| Yuezhang & Wei v2 2505.22839 | score Jacobian | **"on-manifold"** | "off-manifold" |

Availability check — zero occurrences in any of the four papers of `pass-through`,
`passthrough`, `transparent subspace`, `unit eigenvalue`, `fixed-point subspace`.
**The term is free.**

Terms to avoid:

* **"null space"** — Shallow Diffuse owns it for the *opposite* (erased) end;
  LOCO Edit uses it for a third unrelated thing. **Drop "null space of the
  Hessian" from the abstract** even though it is mathematically correct: a
  reviewer who knows Shallow Diffuse will read it backwards. Write "near-zero
  eigenspace of the log-density Hessian" or "unit-eigenvalue eigenspace of the
  denoiser Jacobian".
* **"invariant subspace"** — misleading twice: Kadkhodaie uses "local invariances"
  for the *erased* end, and in linear algebra it means AV subset of V.
* **"tangent subspace"** — Saito's and Kadkhodaie's word; invites the one-line
  dismissal "this is just the tangent space, known since Stanczuk."
* **"on-manifold"** — now Yuezhang & Wei's term for precisely our set. Adopting it
  concedes priority in the name.

**Anchor sentence to use on first mention**, which defuses all four collisions at
once:

> *"We call this the **pass-through subspace**: the near-zero eigenspace of
> grad^2 log p_t, equivalently the unit-eigenvalue eigenspace of the denoiser
> Jacobian. It is the set Kadkhodaie et al. call the preserved subspace, that
> Saito & Matsubara identify with the manifold's tangent space, and that
> Yuezhang & Wei term the on-manifold directions. It is the complement of the null
> space of the denoiser Jacobian used by Shallow Diffuse for watermarking, which
> targets the erased directions instead."*

---

# 12. THE FOREIGN PURIFIER (E9) — setup, provenance, citations (2026-09-16)

## 12.1 Why

E2's transfer cell is v3 -> v4, two runs of one recipe: same architecture, corpus,
beta schedule and training script, differing only in seed, data order and
validation split. Section 10.1 records that Zhang et al. (ICML 2024) already
*predict* agreement between independently trained diffusion models, so a reviewer
may fairly read E2 as seed stability. To say anything about **speech diffusion
purifiers** rather than **our purifier**, the evaluation model has to be foreign
in architecture, weights, data and domain.

## 12.2 What we obtained

Two public unconditional DiffWave checkpoints, both verified loaded (408 tensors,
24.07 M parameters, zero missing and zero unexpected keys against the vendored
network definition):

| tag | weights from | training data | bytes | licence |
|---|---|---|---|---|
| `libri_deantifake` | De-AntiFake `purification.pkl` (HuggingFace `cyberrrange/De-AntiFake`) | **LibriSpeech**, 16 kHz | 288,831,938 | MIT (weights) |
| `sc09_philsyn` | `philsyn/DiffWave-unconditional` | Speech Commands digits (SC09), 16 kHz | 96,451,975 | MIT |

`libri_deantifake` is the important one: it is the purifier from the paper that
motivates this work, trained on our own corpus. It is foreign in every respect —
WaveNet on **raw waveform** rather than our 2-D UNet on mel.

`sc09_philsyn` is a second, narrower data point. Its prior is spoken digits in 1 s
clips, so **a null result against SC09 alone would be ambiguous** between "the
subspace is ours" and "the speech distribution is too different"; it is reported
as corroboration, never as the primary evidence.

## 12.3 Provenance and licensing — deliberate choices

* The network definition is philsyn's `WaveNet.py` + `util.py`, **MIT**, vendored
  under `models/diffwave/` with its LICENSE.
* **The purification loop is written by us** in `models/diffwave_purifier.py` from
  the DDPM equations and the published schedule (T = 200, beta linear
  1e-4 -> 0.02), *not* copied from `cychomatica/AudioPure`, which implements
  exactly this but **ships no LICENSE file**. The same applies to
  `cyberrrange/De-AntiFake`'s repo (its HuggingFace *weights* are MIT; the repo is
  unlicensed). Since this work is being published, we take the weights and the
  papers, and write the ~30 lines ourselves.

## 12.4 Calibrating t — by damage, not by integer

DiffWave's t is not our t. Its schedule is its own, and its noise is added in the
waveform domain. Mean relative mel-domain damage to **unprotected** audio,
5 held-out speakers:

| setting | damage |
|---|---|
| vocoder round trip only (floor) | 0.1972 |
| **ours v3, t = 200** | **0.2933** |
| **DiffWave-LibriSpeech, t = 1** | **0.3091** |
| DiffWave-SC09, t = 1 | 0.2919 |
| DiffWave-LibriSpeech, t = 3 | 0.4789 |
| DiffWave-SC09, t = 3 | 0.4446 |

So **t = 1** is the damage-matched setting, and t = 3 — AudioPure's own default —
is already substantially *harsher* than our strongest setting. Both are run.

*Caveat to state in the paper:* this calibration fed DiffWave the original
waveform while our purifier's figure includes the vocoder round trip, so the match
is approximate. Inside the experiment both purifiers receive identical vocoded
input, which is the comparison that carries the claim.

## 12.5 The measurement, and why the floor is not 1x

Our d lives in log-mel; a waveform purifier cannot be handed it directly. So
`w = vocode(mel + d)`, purify, return to mel. That round trip is lossy (E6:
alpha 0.57-0.68, injected noise 3-4x the surviving signal), so **an absolute alpha
from this pipeline is uninterpretable**. The interpretable quantity is the
advantage over random, in which the vocoder loss cancels.

But the vocoder floor is **not 1x**: E6 showed the round trip already favours the
pass-through direction (alpha 0.92 vs 0.68), because that direction is smoother
and survives mel inversion better. Crediting the purifier with that head start
overstates the transfer. The reported quantity is therefore the **excess over the
floor**, and `exp_foreign_purifier.py` prints all three cells every run
(vocoder-only / ours / foreign) so the floor can never be silently dropped.

Our own purifier is wrapped to the same waveform interface, so the reference cell
goes through the identical lossy pipeline. Without that the comparison would be
rigged in our favour.

**These numbers are not comparable to E2's 76x** — that was a mel-domain R_t ratio
with no vocoder in the path. Compare the foreign cell to the ours cell within this
table only. The script prints that warning in its own output.

## 12.6 Citations to add

```bibtex
@inproceedings{kong2021diffwave,
  title={DiffWave: A Versatile Diffusion Model for Audio Synthesis},
  author={Kong, Zhifeng and Ping, Wei and Huang, Jiaji and Zhao, Kexin and Catanzaro, Bryan},
  booktitle={ICLR}, year={2021}, eprint={2009.09761}}

@inproceedings{wu2023audiopure,
  title={Defending against Adversarial Audio via Diffusion Model},
  author={Wu, Shutong and Wang, Jiongxiao and Ping, Wei and Nie, Weili and Xiao, Chaowei},
  booktitle={ICLR}, year={2023}, eprint={2303.01507}}

@inproceedings{fan2025deantifake,
  title={De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks},
  author={Fan, Wei and Chen, Kejiang and Liu, Chang and Zhang, Weiming and Yu, Nenghai},
  booktitle={ICML}, year={2025}, eprint={2507.02606}}
```

Cite DiffWave for the architecture, AudioPure for the forward-to-t-then-reverse
recipe on audio (and for n* = 3 as the reference operating point), De-AntiFake for
the LibriSpeech weights *and* as the motivating threat.

**Unverified, check before submission:** the Kong et al. and Nie et al. entries
were reproduced from memory by the literature agent, not fetched. AudioPure's
ICLR 2023 venue is corroborated by the authors' own repo and arXiv, but
OpenReview and dblp both blocked automated access, so the canonical proceedings
entry is unconfirmed.

---

# 13. Novelty re-check, 2026-10-08 — and C2's delta is NARROWER than FRAME_v7 claimed

Run after `FRAME_v7` promoted §42h to a first-class contribution. §42h had
**never been novelty-checked**: it was filed as a setup table until the
2026-10-06 council asked for it to lead, so the check that every other claim
got was skipped for the one being promoted.

## 13.1 C1 — still open

No paper found that measures the survival of a **speaker encoder's own
identity direction** through a speech purifier, or compares that survival
across encoders. The near misses, and why each misses:

| work | what it does | why it is not C1 |
|---|---|---|
| DAP, arXiv 2310.14270 | diffusion purification in front of an ASV | defence side; no direction analysis |
| TDAP, Interspeech 2024 | text-guided purification, aims to retain the speaker | no geometry; no per-encoder comparison |
| PnP, arXiv 2607.00899 | positive-incentive noise predictor for ASV purification | defence side, benchmarks against DAP/AudioPure |
| MDD, arXiv 2508.19180 | mask-diffusion detector used as a purifier | detection, not direction survival |
| SEED, arXiv 2505.16798 | diffusion applied to speaker **embeddings**, 3 encoders | operates in embedding space, not on audio through a purifier; enhancement, not survival |
| Eta-WavLM, ACL Findings 2025 | linear removal of speaker identity from SSL features | removal by construction, not survival through a denoiser |
| TVTSyn, arXiv 2602.09389 | streaming VC / anonymisation | unrelated; checked and discarded |

**C1 remains open.** Re-check before submission.

## 13.2 C2.1 — the part that is NOT new, and must stop being written as if it were

**`FRAME_v7` §4 C2.1 overclaims, and this is the correction.** "The purifier
degrades the verifier, not just the perturbation" is **already known in the
literature, in EER**:

* **MDD (arXiv 2508.19180)** reports clean-trial ASV EER rising **1.4 % → 4.0 %**
  at its best-balanced masking ratio, and states plainly that heavier masking
  "degrades bona fide performance … due to excessive information loss during
  reconstruction". It explicitly measures "the impact of different purification
  methods on ASV performance using the Equal Error Rate".
* **DAP (arXiv 2310.14270)** scores with minDCF (p = 0.01, C_miss = C_fa = 1)
  and reports a clean ECAPA baseline at EER 1.069 %, minDCF 0.107.
* An older x-vector purification study reports clean minDCF 0.513 and claims
  its purifier "maintains the performance of genuine samples".
* **SVeritas (EMNLP Findings 2025)** lays out minDCF under codec and noise
  degradation — the table format we are using, for a different degradation.

So **we did not discover that purification costs clean verification accuracy.**
Any sentence implying that is wrong and a reviewer who knows this literature
will say so.

## 13.3 What is left, and it is still worth a contribution slot

Four things, none of which the above does:

1. **minDCF at a realistic operating point, not EER.** This is the whole
   point. On our channels EER moves 5.2 % → 9.5 % on WavLM-SV, which reads as
   a mild degradation; minDCF at p_target = 0.01 moves **0.77 → 0.91**, i.e.
   from poor to indistinguishable from the trivial system. **EER hides it and
   minDCF shows it**, and the protection literature reports EER.
2. **In the exact cells a protection paper scores in.** Prior work measures
   the clean-trial cost of *its own* purifier as a side effect of proposing a
   defence. Nobody reports the operating point of the verifier in the cells
   where a *protection* claim is being made, which is where it changes the
   reading.
3. **Per encoder, with the consequence drawn.** Three encoders, eight
   channels, and the finding that this **reorders which protection claims are
   trustworthy**: ECAPA is the only verifier worth believing at p_target =
   0.01, so "protection holds on WavLM-SV" is the weakest of our three claims
   and not the strongest. We know of no protection paper that ranks its own
   per-encoder claims by the quality of the instrument.
4. **Paired with §29's proxy-vs-end-to-end gap and §40c's clone degradation**,
   as three independent ways the same literature's instruments distort it.

## 13.4 Required narrowing — do this before the draft is submitted

Do **not** write: "we show that purification degrades the verifier"; "a
correction other people's papers inherit" without a citation to the papers that
already said the EER half.

**Write instead:** "Purification is known to cost clean verification accuracy
(MDD; DAP). We show that at a realistic operating point the cost is far larger
than EER suggests — WavLM-SV moves from minDCF 0.77 to 0.91, effectively the
trivial system — and that this reorders which per-encoder protection claims are
trustworthy."

Cite MDD and DAP at that sentence. Add SVeritas as the precedent for the table
format.

## 13.5 Citations to add

* MDD, arXiv 2508.19180 — purification's clean-EER cost, stated by its authors.
* SVeritas, EMNLP Findings 2025 — minDCF-under-degradation table format.
* TDAP, Interspeech 2024 — purification that explicitly targets retaining the
  speaker; the closest intent to our C2 framing.
* SEED, arXiv 2505.16798 — diffusion on embeddings; the "not us" boundary for
  C1, since a reader may think it is the same thing.

All four were retrieved from search excerpts, not full texts. **Verify the
numbers against the papers before quoting any of them** — the same rule §12.6
applied to the AudioPure entry.
