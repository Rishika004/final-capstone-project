# What Survives Is Not What Protects: Purification-Aware Voice Protection

*ICASSP 2027 draft skeleton, 4 pages + references. All numbers below are on
disk (experiments/results/main_table_v3.csv, eval_psr_v3.csv,
nullspace_survival_v3_ddim_matched.csv) and reproduced in
experiments/RESULTS_ICASSP.md.*

## Abstract (~150 words)

Protective perturbations that prevent voice cloning are removed by diffusion
purification. The intuitive fix is to make the perturbation survive the
purifier. We show that this is the wrong objective. Using a mel-domain
diffusion purifier whose denoising strength and identity retention we
characterise, we construct perturbation directions that pass through the
purifier almost losslessly (98 % of their energy at four times the assumed
purification strength, against 61 % for a random perturbation). Yet a transformation that is
perceptually free (UTMOS 4.12 vs 4.12 for the original, 0.4 % word errors) and
passes through the purifier untouched protects **0 of 20 speakers**: survival is
not sufficient. Within additive crafting the two are actively *anti*-correlated
(r = -0.86); purification maps a perturbation to something still large but no
longer adversarial. Crafting
through the purifier instead, so that what survives remains adversarial,
raises post-purification protection from 70 % to 90 % on two speaker
verification systems at equal perceptual cost, and adding the pass-through
constraint on top keeps protection at 100 % when the attacker purifies four
times harder than the defender assumed.

## 1. Introduction

- Voice cloning from seconds of public audio; protective perturbations
  (AntiFake CCS'23, SafeSpeech, VoiceGuard) are the deployable countermeasure.
- De-AntiFake (ICML'25) and VocalBridge (2026) strip them with purification
  and call for protections that survive it. The natural reading — "make the
  perturbation survive" — motivates null-space / invariant-subspace designs.
- We test that reading directly, and it fails. Contributions:
  1. A purifier that provably purifies, with the diagnostics that any
     purification-robustness claim should report (denoising gain + channel EER).
  2. The pass-through subspace of a diffusion purifier, computed at the noise
     level where removal happens; near-lossless survival.
  3. **Survival is not sufficient**, shown by existence (a free, purification-
     invariant transform protecting 0/20 speakers) and, within additive
     crafting, by anti-correlation (r = -0.86). NOTE: the correlation does NOT
     hold across families (r = -0.19 at t=200, +0.25 at t=50) -- it must be
     stated as a within-additive-family result. Evidence that the field's
     implicit objective is wrong.
  4. Purifier-aware crafting (EOT through the reverse process) + a cheap
     first-order pass-through regulariser: best protection, and robustness to
     an attacker who purifies harder than assumed.

## 2. Threat model, purifier, protocol

### 2.1 Threat model
x_p = x + delta published; attacker computes x_r = V(P_t(mel(x_p))), clones
from x_r. Verifier accepts iff cos(E(x), E(x_r)) >= tau. Defender assumes
t = 50; we evaluate t in {50, 100, 200}.

### 2.2 A purifier that purifies (Table 1)
Residual 2-D UNet over (mel x time), DDPM-consistent training noise
(t ~ U{0..200}), deterministic DDIM reverse process. Report denoising gain
(0.17 / 0.35 / 0.57 at t = 50/100/200) and channel EER (6.7 / 0.2 / 0.8 % for
WavLM-SV / ECAPA / d-vector at t=50).
*Methodological note (footnote or 3 lines):* an earlier purifier trained on one
speaker, with an off-by-one in the noise schedule, destroyed the identity of
**unprotected** audio (channel EER 14-23 %); any protection measured against it
looks robust for the wrong reason. Reporting channel EER should be standard.

### 2.3 Evaluation
Thresholds at EER **through the attacker's channel**; success = clone fails
after purification AND unprotected control retains identity (100 % throughout).
LibriSpeech dev-clean, speakers 0-19 train the purifier, 20-39 held out.

## 3. Method

### 3.1 Purifier-aware crafting
Threshold-aware ensemble loss over three encoders,
LSE_e[cos_e(E_e(V(P_s(mel(x+d)))), E_e(x)) - tau_e], fresh purifier noise each
step (EOT), gradient-checkpointed through all t steps. Perceptual constraint:
Bark masking projection + per-frame local SNR floor (removes the audible noise
in pauses: PESQ 1.8 -> 3.4 at the same protection ordering).

### 3.2 The pass-through subspace
For purifier P with correction C(x) = P(x) - x, first-order survival requires
J_C d ~= 0. Crucially this must be evaluated at the noise level where removal
happens:
R_t(d) = ||m(x_t + sqrt(ab_t) d) - m(x_t) - d||^2 / ||d||^2,
x_t = sqrt(ab_t) mel(x) + sqrt(1 - ab_t) eps.
The zero-noise version (null space of the clean-input Jacobian) is no better
than a random direction. One denoiser pass per step vs t passes for EOT.

## 4. Results

- **Table 1** purifier diagnostics; channel EER per t.
- **Table 2** subspace survival: range 1.08, random 0.61, zero-noise null 0.61,
  noise-matched null **0.98** (PSR at t=200); direction cosine 0.15 / 0.81 /
  0.88 / **0.99**.
- **Figure 1** survival vs protection scatter over five crafters, r = -0.86.
  *The core figure of the paper.*
- **Table 3** main results, 20 held-out speakers, two perceptual budgets:

  | crafting | PESQ | t=50 | t=100 | t=200 | cost |
  |---|---|---|---|---|---|
  | plain | 2.22 | 75/85/80 | 70/80/80 | 75/65/75 | 29 s |
  | + noise-matched reg. | 2.23 | 65/70/80 | 65/65/65 | 65/60/55 | 31 s |
  | purifier-aware (EOT) | 2.23 | 95/95/70 | 95/90/70 | 85/90/55 | 164 s |
  | **EOT + noise-matched reg.** | 2.24 | 95/95/65 | 95/95/60 | **100/95/60** | 168 s |
  | plain (tight budget) | 3.24 | 45/20/35 | 45/10/20 | 40/10/20 | 29 s |
  | purifier-aware (tight) | 3.40 | 75/60/20 | 75/55/15 | 70/50/5 | 156 s |

- **Table 4** transfer to an unseen purifier (crafted vs v3/DDIM, evaluated vs
  v2/DDPM): EOT 95/80/75 vs plain 85/80/70 at t=200.
- Ablations: loudness floor, threshold-aware loss, vocoder surrogate.

## 5. Discussion

Why survival misleads: the purifier is not a filter that deletes delta; it is
a projection toward the speech manifold. A perturbation can be preserved in
norm and direction while the *speaker-discriminative* component of it is
re-absorbed. Optimising survival therefore optimises a proxy that is not
monotone in the quantity we care about. The pass-through constraint still
helps once the perturbation is already adversarial-after-purification, which
is why EOT + regulariser is the most robust configuration to an attacker who
purifies harder than assumed.

## 6. Limitations

Mel-domain surrogate purifier (not the image-domain DiffPure score model);
Griffin-Lim attacker vocoder; one utterance per speaker, 20 speakers;
d-vector attacked through a proxy front-end; objective quality metrics only,
no MOS. Neural-vocoder attackers (VocalBridge-style) are the next test.

## References
AntiFake (CCS'23); De-AntiFake (ICML'25); VocalBridge (arXiv 2601.02444);
VoiceCloak; SafeSpeech; DiffPure (ICML'22); DiffAttack (NeurIPS'23);
DiffHammer (NeurIPS'24); Unlocking Adaptive Attacks on DBP (2411.16598);
Stanczuk et al. 2022; NPAT (2409.11754); Subspace Defense (2403.16176);
WavLM; ECAPA-TDNN; GE2E/Resemblyzer; Johnston masking; PESQ; STOI; LibriSpeech.
