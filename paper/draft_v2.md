# Purification-Aware Protective Perturbations Against Voice Cloning

*Draft skeleton for ICASSP 2027 (4 pages + references). Numbers marked
[v2] were measured against the weak v2 purifier and must be replaced by
`main_table_v3.csv` (queue 7). Numbers marked [v3] are final.*

## Abstract (150 words)

Protective perturbations that stop zero-shot voice cloning are removed by
diffusion-based purification (De-AntiFake, VocalBridge). We study protection
that is crafted *through* the purifier. We (i) build a multi-speaker
mel-domain diffusion purifier whose identity retention and denoising gain we
report, so that its removal of perturbations is meaningful; (ii) craft
perturbations by expectation-over-transformation through the full reverse
process, under a psychoacoustic mask with a per-frame loudness floor; and
(iii) analyse the purifier's pass-through subspace, the null space of its
correction Jacobian, which is the tangent space of the speech manifold, and
use it as a cheap first-order surrogate for purifier awareness. On 20 unseen
LibriSpeech speakers and three independent speaker-verification models with
thresholds calibrated through the attacker's channel, purifier-aware crafting
raises post-purification protection from [v2: 40 to 85 %] on WavLM-SV and
[v2: 20 to 45 %] on ECAPA at PESQ 3.4, and the gain grows with purification
strength. We release code.

## 1. Introduction

- Zero-shot TTS/VC clones a voice from seconds of audio; protective
  perturbations (AntiFake CCS'23, VoiceGuard, SafeSpeech) are the deployable
  countermeasure for people who must publish their voice.
- Two 2025-26 papers show the current generation is stripped by purification:
  De-AntiFake (ICML'25) and VocalBridge (Jan 2026). Both call for protection
  that survives purification. None exists yet.
- In images, DiffPure-style defenses were broken by attacks that
  differentiate through the purifier (DiffAttack, DiffHammer, Unlocking
  Adaptive Attacks). We turn that attacker's tool into the defender's tool.
- Contributions: (1) a validated purifier (identity retention, denoising gain,
  DDIM); (2) purifier-aware crafting with a threshold-aware ensemble loss and
  a per-frame loudness floor; (3) pass-through-subspace analysis and the
  null-space regulariser; (4) an honest evaluation protocol (channel-calibrated
  EER thresholds, unprotected control, three purification strengths).

## 2. Threat model and evaluation protocol

- Defender publishes x_p = x + delta. Attacker purifies:
  x_r = V(DiffPure_t(mel(x_p))) with V = Griffin-Lim, then clones from x_r.
  Attacker knows the purifier and t; defender knows a purifier but not t.
- Verifier: cosine between enrolment E(x) and E(x_r) on three encoders
  (WavLM-SV x-vector, ECAPA-TDNN, Resemblyzer d-vector); accept iff cos >= tau.
- tau is the EER threshold of same/different-speaker trials **through the
  attacker's channel** (clean enrolment vs purified test), so acceptance is the
  decision an ASV would make on purified audio. Table 1 reports EER per channel.
- Success on an utterance: clone fails after purification AND the unprotected
  control through the same channel keeps its identity.
- Data: LibriSpeech dev-clean; speakers 0-19 train the purifier, 20-39 are
  held out; calibration on utterances 0-2, protection on utterance 3 (4 s).

## 3. Method

### 3.1 A purifier that purifies
- Residual 2-D UNet over (mel x time), trained with the exact DDPM forward
  noise it will see (t ~ U{0..200}); deterministic DDIM reverse process.
- Report: clean reconstruction error, Gaussian denoising gain per t, and the
  removal of *adversarial* perturbations (mel PSR) — Table 1b.
- Lesson (as a footnote or short paragraph): an earlier 1-D UNet trained on
  one speaker, with an off-by-one in the noise schedule, scaled every log-mel
  by 0.55 and destroyed identity of unprotected audio (channel EER 14-23 %);
  every "survives purification" number measured against such a purifier is
  vacuous. We recommend reporting channel EER for any purifier used in a
  security evaluation.

### 3.2 Purifier-aware crafting
- Loss: LSE_e[(cos_e(E_e(V(P_s(mel(x+d)))), E_e(x)) - tau_e)] + lam *
  LSE_e[(cos_e(x+d) - tau_e)], fresh purifier noise seed s each step (EOT),
  gradient-checkpointed backprop through all t steps; V = fixed-phase
  differentiable mel inversion during crafting.
- Threshold-aware LSE: the encoder closest to *accepting* the clone gets the
  gradient, not the one with the largest raw cosine (raw-cosine loss gave
  WavLM-SV 100 % / ECAPA 0 %).
- Perceptual constraint: Bark masking threshold projection + per-frame local
  SNR floor (delta energy <= frame energy - f dB), which removes the audible
  noise in pauses (PESQ 1.8 -> 3.4 at equal protection ordering).

### 3.3 The pass-through subspace (null space)
- For a purifier P with correction C(x) = P(x) - x, first-order survival of
  delta requires J_C delta ~= 0: the null space of the correction Jacobian.
  For a score-based denoiser at low noise this is the tangent space of the
  data manifold (Stanczuk et al. 2022). (Contrast: the null space of an
  x0-predictor is what the purifier *erases*.)
- Empirical test (E1): optimised pass-through directions vs random vs range
  directions, survival through the real loop at t = 50..300: Table 2.
- Regulariser: R(d) = ||C(mel(x+d)) - C(mel(x))||^2 / ||d_mel||^2, one UNet
  pass per step (vs t passes for EOT). Variants: plain, +R, EOT, EOT+R.

## 4. Experiments

- Table 1: channel EER / thresholds (clean, GL only, purifier t=50/100/200).
- Table 1b: purifier strength (denoising gain; adversarial mel PSR).
- Table 2: E1 subspace survival (PSR, direction cosine) [v3].
- Table 3 (main): 20 speakers, two perceptual budgets (PESQ ~3.4 and ~2.3),
  four crafters, protection per encoder at t = 50 / 100 / 200, plus crafting
  time per utterance. [v3 pending; v2 numbers: plain 40/20/15 vs EOT 85/45/10
  at PESQ 3.2-3.4; plain 60/75/75 vs EOT 95/95/65 at PESQ 2.2-2.3]
- Table 4: transfer — perturbations crafted against purifier A evaluated
  against purifier B (v2 -> v3): EOT gap widens with purifier strength
  (ECAPA 80 vs 30 % at t=200) [v3 eval of v2-crafted files, done].
- Ablations: loudness floor on/off; threshold-aware loss on/off; vocoder
  surrogate (fixed phase vs differentiable Griffin-Lim).
- Listening samples on the project page.

## 5. Limitations and conclusion

- Surrogate mel-domain purifier trained by us, not the image-domain DiffPure
  score model; Griffin-Lim as the attacker's vocoder; one utterance per
  speaker; d-vector attacked through a proxy front-end; objective quality
  metrics only. Stronger neural-vocoder attackers (VocalBridge-style) are the
  obvious next test.
- Conclusion: purification robustness must be measured against a purifier
  that provably purifies; given one, crafting through it roughly doubles
  post-purification protection at equal perceptual cost, and the pass-through
  subspace explains which perturbation directions survive.

## References (to format)
AntiFake (Yu et al., CCS 2023); De-AntiFake (Fan et al., ICML 2025);
VocalBridge (2026, arXiv 2601.02444); VoiceCloak (2505.12332); SafeSpeech;
DiffPure (Nie et al., ICML 2022); DiffAttack (NeurIPS 2023); DiffHammer
(NeurIPS 2024); Unlocking Adaptive Attacks on DBP (2411.16598); Stanczuk et
al. 2022 (diffusion models and manifold dimension); NPAT (2409.11754);
Subspace Defense (2403.16176); WavLM (Chen et al. 2022); ECAPA-TDNN;
Resemblyzer / GE2E; Johnston psychoacoustic model; PESQ; STOI; LibriSpeech.
