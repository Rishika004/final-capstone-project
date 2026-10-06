# Results for the ICASSP 2027 submission (rebuilt 2026-09-11)

*Everything below was produced after two defects in the original pipeline were
found and fixed. Numbers from before 2026-09-11 (RESULTS_SUMMARY.md, end2end.csv,
all PRR figures) must not be used: they were measured against a broken purifier.*

## 1. What was wrong, and what changed

| Defect | Effect | Fix |
|---|---|---|
| DDPM schedule off-by-one in `models/real_diffpure.py` | Purifier output log-mel scaled by ~0.55 at every t (oracle-denoiser test: 1.75 nats error). Identity of *unprotected* audio destroyed (channel EER 14–23 %). | Standard indexing; oracle error now 0.003 nats. |
| Denoiser trained on LJSpeech only (1 speaker), fixed noise | 2–3x reconstruction error on unseen speakers; DDPM loop compounds it. | `training/train_purifier_v2.py`: residual UNet, LibriSpeech dev-clean speakers 0–19 + LJSpeech, DDPM-consistent noise (t ~ U{0..100}). Loop error on unseen speakers 1.40 -> 0.33 nats. |
| WavLM encoder had a random 768->256 head | Not a speaker-verification system. | `microsoft/wavlm-base-plus-sv` (trained x-vector head). |
| Old end-to-end script never used the purifier or null space in its loss | The failing table was plain PGD. | `experiments/exp_purifier_aware.py`: variants `plain` / `surrogate` / `full` (EOT through the full DDPM loop, gradient-checkpointed). |
| Masking cap floored at hearing threshold + margin | Perturbation 14 dB *above* signal in pauses -> PESQ ~1.8. | Per-frame local-SNR floor (`--snr-floor`, `core/psychoacoustic.py`). |

### 1b. Purifier strength (added 2026-09-11 evening)

A purifier that does not remove noise is not an attacker. Measured on 5
held-out speakers, Gaussian noise at the DDPM level of each t:

| purifier | single-pass denoising gain t=50 / 100 / 200 | loop gain (DDPM) t=50 / 100 / 200 | loop gain (DDIM) |
|---|---|---|---|
| v1 (LJSpeech 1-D UNet, fixed loop) | -5.0 / -1.9 / -0.4 | -9.2 / -4.3 / -1.4 | – |
| v2 (residual 1-D UNet, multi-speaker) | 0.06 / 0.12 / 0.14 | -0.95 / -0.54 / -0.34 | – |
| **v3 (residual 2-D UNet, multi-speaker, 30k steps)** | **0.17 / 0.35 / 0.57** | -0.42 / -0.05 / +0.21 | **-0.15 / +0.09 / +0.25** |

(gain = 1 - L1(P(x_t), x) / L1(x_t, x); negative = the purifier adds error.)
v2 was effectively an identity map, so its "identity retention" was trivial
and the main table in section 5 measures survival of the mel+Griffin-Lim
bottleneck more than survival of purification (eval_saved.csv: attacker
cosine barely changes between t=0 and t=200). v3 + DDIM (deterministic reverse
process, which does not re-inject noise) is the final attacker; all section-5
numbers are being regenerated against it (`main_table_v3.csv`, queue 7).

### 1c. Null-space idea, corrected

* Null space of an x0-predictor = directions the purifier erases. The
  surviving ("pass-through") subspace is null(J_C) of the correction
  C(x) = P(x) - x, i.e. the data-manifold tangent space (Stanczuk et al. 2022).
* `core/nullspace.compute_null_space_factored` returned the complement of the
  K random probe directions, not of J's row space: the old projection was ~identity.
* E1 (`exp_nullspace_survival.py`, v2 purifier): optimised pass-through
  directions survive the loop with PSR 0.96 / 0.92 / 0.79 / 0.59 / 0.20 at
  t=50/100/200/300/500 vs random 0.955 / 0.90 / 0.76 / 0.57 / 0.19; direction
  cosine 0.98 vs 0.93. Range directions are amplified 4x and scrambled
  (cos 0.2). The subspace is real but, for this purifier, random mel
  perturbations are already ~95 % pass-through, so the projection adds only a
  few percent. Re-run against v3/DDIM pending.
* New crafting option `--lam-null`: first-order purification-invariance
  regulariser ||C(mel(x+d)) - C(mel(x))||^2 / ||d_mel||^2 (one UNet pass per
  step instead of the 50-step loop). Compared against plain and EOT in the
  v3 main table (queue 7): protection, cost, transfer.

## 2. Protocol

* Data: LibriSpeech dev-clean. Speakers 0–19 (sorted ids) train the purifier;
  speakers 20–39 are held out for everything below. Calibration uses
  utterances 0–2 of each held-out speaker; protection uses utterance 3 (4-s crop).
* Attacker: `x_r = GriffinLim(DiffPure_t(mel(x_p)))`, real DDPM loop with
  purifier v2, three purification seeds, t in {50, 100}.
* Verifier: cosine between the clean enrolment embedding and the embedding of
  the attacker's purified audio, three independent encoders (WavLM-SV,
  ECAPA-TDNN, Resemblyzer d-vector). Thresholds are set at EER on same/different
  speaker trials **through the attacker's channel**, so the acceptance decision
  is the one an ASV deployed on purified audio would make.
* Success on a file ("defence valid"): cloning fails after purification AND the
  unprotected control through the same channel keeps its identity.

## 3. Threshold calibration (20 held-out speakers x 3 utterances)

| channel | WavLM-SV EER / thr | ECAPA EER / thr | d-vector EER / thr |
|---|---|---|---|
| clean | 3.8 % / 0.90 | 0.04 % / 0.40 | 0.04 % / 0.78 |
| Griffin-Lim only | 5.4 % / 0.89 | 0.0 % / 0.49 | 0.0 % / 0.75 |
| DiffPure v2 t=50 + GL | 7.5 % / 0.889 | 0.8 % / 0.324 | 1.7 % / 0.717 |
| DiffPure v2 t=100 + GL | 9.9 % / 0.883 | 0.8 % / 0.328 | 2.4 % / 0.703 |
| DiffPure v2 t=200 + GL | 15.1 % / 0.857 | 1.7 % / 0.292 | 4.8 % / 0.665 |
| *old v1 purifier, t=50 (fixed loop)* | *20.8 %* | *11.8 %* | *19.2 %* |

The realistic purifier keeps speaker identity (an attacker's precondition);
the old one did not.

## 4. Trial sweep (3 held-out speakers) — quality vs. protection

Protection = % of files where the defence is valid (per encoder / all three).
Evaluated at t=50 and t=100 (identical unless stated).

| crafting | margin / floor (dB) | PESQ | STOI | SNR dB | WavLM-SV | ECAPA | d-vec | all |
|---|---|---|---|---|---|---|---|---|
| plain | 30 / none | 1.86 | 0.898 | 16.6 | 100 | 100 | 100 | 100 |
| purifier-aware (EOT) | 30 / none | 1.73 | 0.869 | 14.4 | 100 | 100 | 100 | 100 |
| plain | 30 / 15 | 2.14 | 0.918 | 17.6 | 67 | 67 | 67 | 67 |
| purifier-aware (EOT) | 30 / 15 | 2.27 | 0.933 | 18.3 | 100 | 100 | 67 | 67 |
| plain | 20 / 20 | 3.20 | 0.955 | 20.7 | 67 | 67 | 67 | 67 |
| purifier-aware (EOT) | 20 / 20 | 3.39 | 0.969 | 22.0 | 100 | 0 | 33 | 0 |
| plain | 15 / 20 | 3.70 | 0.970 | 20.5 | 33 | 0 | 0 | 0 |
| purifier-aware (EOT) | 15 / 20 | 3.88 | 0.976 | 22.2 | 67 | 0 | 0 | 0 |
| plain | 10 / 20 | 3.90 | 0.982 | 20.3 | 33 | 0 | 0 | 0 |
| plain | 20 / 25 | 3.74 | 0.975 | 25.1 | 33 | 0 | 0 | 0 |
| plain | 10 / 25 | 4.26 | 0.990 | 25.2 | 0 | 0 | 0 | 0 |

Observation from the 20/20 and 15/20 rows: the raw-cosine ensemble loss
attacks the encoder with the largest cosine (WavLM-SV, threshold 0.89) and
under-attacks ECAPA (threshold 0.32) -> WavLM-SV 100 % but ECAPA 0 %.
Fixed by a threshold-aware loss, LSE over (cos_e - tau_e); the main table
(queue 4) uses it for BOTH crafting methods.

Reading: at an equal perceptual budget (30/15) purifier-aware crafting keeps
the WavLM-SV and ECAPA clones broken on every file where plain crafting fails
on a third of them. Section 5 reports the two operating points 20/20 and 30/15 on 20 speakers.

## 5. Main table — FINAL attacker (purifier v3 + DDIM), 20 held-out speakers

`experiments/results/main_table_v3.csv` (summary `main_table_v3_summary.md`).
Threshold-aware ensemble loss for all crafters; thresholds = EER through the
v3/DDIM channel at t=50 (0.897 / 0.346 / 0.735); 3 purification seeds.
Protection = % of speakers where the clone fails after purification AND the
unprotected control keeps its identity (control kept on 100 % everywhere).

Budget A: margin 30 dB, floor 15 dB (PESQ ~2.2)

| crafting | PESQ | t=50: WavLM-SV / ECAPA / d-vec | t=100 | t=200 |
|---|---|---|---|---|
| plain | 2.22 | 75 / 85 / 80 | 70 / 80 / 80 | 75 / 65 / 75 |
| plain + zero-noise null reg. | 2.22 | 70 / 85 / 75 | 75 / 75 / 75 | 70 / 70 / 70 |
| **purifier-aware (EOT)** | 2.23 | **95 / 95** / 70 | **95 / 90** / 70 | **85 / 90** / 55 |
| EOT + zero-noise null reg. | 2.25 | 90 / 90 / **80** | – | **90 / 90** / 60 |

Budget B: margin 20 dB, floor 20 dB (PESQ ~3.3)

| crafting | PESQ | t=50: WavLM-SV / ECAPA / d-vec | t=200 |
|---|---|---|---|
| plain | 3.24 | 45 / 20 / 35 | 40 / 10 / 20 |
| plain + zero-noise null reg. | 3.23 | 45 / 25 / 40 | 45 / 5 / 25 |
| **purifier-aware (EOT)** | 3.40 | **75 / 60** / 20 | **70 / 50** / 5 |

Crafting cost per 4-s utterance on an RTX 3050 laptop: plain ~30 s, +null
reg. ~32 s, EOT ~170 s.

Reading: against a purifier that demonstrably purifies, crafting through it
raises post-purification protection on the two real ASV encoders from
75/85 % to 95/95 % at PESQ 2.2 and from 45/20 % to 75/60 % at PESQ 3.3, and
the advantage persists when the attacker purifies four times harder than the
defender assumed. The zero-noise null-space regulariser is a no-op (as the
survival test predicted: that subspace does not describe the purifier at the
noise level where removal happens). d-vector, attacked only through a proxy
front-end, is the weak encoder for EOT at the tight budget.

### 5b. Transfer to an unseen purifier

Files crafted against purifier v2 (DDPM), evaluated against v3 (DDIM)
(`eval_saved_v3_ddim.csv`), protection % ECAPA / WavLM-SV:

| crafting (budget A) | t=0 (GL only) | t=50 | t=100 | t=200 |
|---|---|---|---|---|
| plain | 100 / 70 | 75 / 60 | 65 / 55 | 40 / 45 |
| purifier-aware (EOT, vs v2) | 100 / 85 | 95 / 95 | 90 / 85 | 85 / 80 |

EOT crafted against a *different, weaker* purifier still transfers, and the
gap widens as the unseen purifier gets stronger.

### 5c. Null space: the subspace is real, survival is the wrong objective, but the constraint helps on top of EOT

**(i) The pass-through subspace, computed at the right noise level, is almost
perfect** (`nullspace_survival_v3_ddim_matched.csv`, 5 speakers, v3/DDIM).
Directions optimised for R_t = ||m(x_t + sqrt(ab_t) d) - m(x_t) - d||^2/||d||^2:

| direction | linear gain | PSR t=50 | t=100 | t=200 | t=300 | dir-cos t=200 |
|---|---|---|---|---|---|---|
| range (worst case) | 0.601 | 1.48 | 1.40 | 1.08 | 0.60 | 0.15 |
| random | 0.059 | 0.83 | 0.75 | 0.61 | 0.48 | 0.81 |
| null, zero-noise | 0.375 | 0.90 | 0.78 | 0.61 | 0.48 | 0.88 |
| **null, noise-matched t=200** | **0.004** | **0.99** | **0.99** | **0.98** | **0.94** | **0.99** |

The zero-noise subspace (what the original code aimed at) is no better than a
random direction once purification is strong; the noise-matched one keeps 98 %
of its energy at t=200 where a random perturbation keeps 61 %.

**(ii) Survival does not imply protection — it anti-correlates with it.**
Measured on the delivered waveforms of five crafters, 20 speakers, t=200
(`eval_psr_v3.csv`):

| crafting | mel PSR (survival) | protection, real ASV encoders |
|---|---|---|
| purifier-aware (EOT) | 0.920 | **90 %** |
| EOT + zero-noise reg. | 0.928 | 90 % |
| plain | 0.939 | 70 % |
| plain + zero-noise reg. | 0.939 | 70 % |
| plain + noise-matched reg. | **0.974** | 62 % |

Pearson r = **-0.86**. The perturbation that survives purification best
protects worst. Purification does not simply erase the perturbation: it maps
it to something that is still large but no longer adversarial. Maximising
survival (the premise of null-space protection) optimises the wrong quantity;
what matters is that *what survives is still adversarial*, which is what
crafting through the purifier optimises directly.

**(iii) As a regulariser on top of EOT, the noise-matched constraint does pay,
and it pays exactly where it should: under stronger-than-expected
purification.** 20 speakers, budget A (PESQ ~2.2):

| crafting | t=50 | t=100 | t=200 |
|---|---|---|---|
| purifier-aware (EOT) | 95 / 95 / 70 | 95 / 90 / 70 | 85 / 90 / 55 |
| EOT + zero-noise reg. | 90 / 90 / 80 | 90 / 95 / 70 | 90 / 90 / 55 |
| **EOT + noise-matched reg.** | 95 / 95 / 65 | 95 / 95 / 60 | **100 / 95 / 60** |

The defender crafts assuming t=50; when the attacker purifies four times
harder, plain EOT loses 10-15 points on WavLM-SV while the regularised version
loses none (100 % at t=200).

The benefit has a sweet spot and needs perturbation budget to spend:

| configuration | PESQ | t=50 | t=200 |
|---|---|---|---|
| EOT, budget A | 2.23 | 95 / 95 / 70 | 85 / 90 / 55 |
| **EOT + reg., lambda=10, budget A** | 2.24 | 95 / 95 / 65 | **100 / 95 / 60** |
| EOT + reg., lambda=30, budget A | 2.32 | 100 / 80 / 50 | 100 / 80 / 50 |
| EOT, budget B (tight) | 3.40 | 75 / 60 / 20 | 70 / 50 / 5 |
| EOT + reg., lambda=10, budget B | 3.44 | 80 / 35 / 5 | 75 / 40 / 5 |

lambda=30 over-constrains (ECAPA 95 -> 80) and at the tight budget the
constraint costs more attack strength than the perturbation can spare
(ECAPA 60 -> 35). **Recommended configuration: EOT + noise-matched
regulariser, lambda=10, at the loose perceptual budget.**

**(iv) Transfer to an unseen purifier** (files crafted against v3/DDIM,
evaluated against v2/DDPM, `eval_transfer_v2.csv`, t=200, ECAPA / d-vec /
WavLM-SV): EOT 95 / 80 / 75, EOT+reg 95 / 85 / 75, plain 85 / 80 / 70,
noise-matched reg alone 75 / 75 / 65. The null-space constraint does not buy
transfer either; crafting through *a* purifier already transfers.

## 6. Honest limitations to state in the paper

* Purifier is a mel-domain DDPM surrogate (residual UNet) trained by us; not the
  image-domain DiffPure score model. Attacker resynthesises with Griffin-Lim.
* One utterance per speaker for protection; 20 speakers.
* Resemblyzer d-vector is attacked through a torch re-implementation of its mel
  front-end (proxy) and evaluated on the real path.
* Perceptual quality is PESQ/STOI/SNR only; no listening test.

## 7. On-manifold protection under purification (2026-09-15)

The open gap from the novelty check: ClearMask-style *noise-free* protection has
never been evaluated against diffusion purification. Run on the same 20 held-out
speakers, same attacker (purifier v3 + DDIM), same channel-calibrated thresholds.
Four transformation families swept over magnitude; each measured for survival,
protection and cost at once (`parametric_probe.csv`, `onmanifold_summary.md`,
`paper/fig2_onmanifold_vs_additive.png`).

### 7.1 Survival: the manifold prediction is confirmed, completely

Every on-manifold transformation passes through the purifier essentially intact.
Pass-through score R_t = 0.000-0.006 at t=50 (0.001-0.10 at t=200); mel PSR
0.88-0.99; direction cosine 0.95-0.99. For comparison a *random* mel perturbation
retains 0.61 at t=200 and crafted additive perturbations 0.92-0.97. So a purifier
genuinely cannot undo a speech-to-speech transformation, exactly as the tangent-
space argument predicts.

### 7.2 Protection: determined by WHICH direction, not by how much survives

Protection (% of 20 speakers, real ASV encoders WavLM-SV / ECAPA):

| family | magnitude range | survival PSR | protection | PESQ |
|---|---|---|---|---|
| spectral tilt | +-6 to +-18 dB | 0.92-0.98 | **0 % everywhere** | 4.3-4.6 |
| reverberation | RT60 50-600 ms | 0.88-0.98 | **0 % everywhere** | 2.5-4.5 |
| peaking EQ | 4-16 dB | 0.98-0.99 | 0 % (WavLM 10 % at 16 dB) | 2.9-4.5 |
| **formant warp** | 0.88-1.12 | 0.91-0.99 | **up to 95 % (WavLM-SV)** | 1.2-1.9 |

Channel-like directions survive perfectly and protect not at all, at any
magnitude tested. Only the identity-carrying direction protects.

**Mechanism.** Modern speaker encoders are trained with heavy channel and
reverberation augmentation, so they are invariant to precisely the on-manifold
directions that are perceptually cheap. Formant warping is the exception for a
structural reason: you cannot augment it away, because warping changes the
speaker label and would destroy the training targets. The one on-manifold
direction that is not covered by augmentation is the one that works.

### 7.3 The anti-correlation does NOT generalise -- correction to section 5c

Across both families under this attacker, Pearson r between survival and
protection is **-0.19 at t=200 and +0.25 at t=50** -- i.e. no relationship. The
r = -0.86 reported in section 5c holds *within the additive family only* and
must be stated that way. The defensible claim is the weaker, more useful one:

> Surviving purification is necessary but nowhere near sufficient. What survives
> is fixed by the manifold; what protects is fixed by which manifold coordinate
> is moved.

### 7.4 Where this leaves the comparison

Purifier-aware additive crafting reaches 90 % protection at PESQ 2.23; the best
on-manifold setting (formant warp 0.88) reaches 65 % at PESQ 1.18. On the
*measured* trade-off additive still wins -- but PESQ is reference-based and is
known to punish formant shifts that listeners accept, so this comparison is
unresolved until no-reference quality is measured. Next: UTMOS/NISQA/DNSMOS,
Whisper WER for content preservation, and a small listening test. Audio for
listening is in `experiments/results/parametric/`.

### 7.5 A prediction of ours that failed

Reverberation was predicted (2026-09-15, before running) to be a strong attack
because channel mismatch is the classic ASV nuisance factor. It protects on 0 %
of speakers at every RT60 up to 600 ms. The augmentation argument above explains
why, and the prediction is recorded here rather than quietly dropped.

## 8. Fair perceptual measurement, and the quality/protection frontier (2026-09-15)

PESQ is reference-based and scores any voice change as damage, so it cannot
adjudicate a formant shift. Re-measured with instruments that can
(`eval_quality_nr.py` -> `quality_nr.csv`, `quality_mcadams.csv`):

* **UTMOS** and **torchaudio SQUIM**: no-reference naturalness. "Does this sound
  like natural speech?", not "is it identical to the original?"
* **Whisper word error rate** against Whisper's transcript of the *original*
  4-s crop (ground-truth transcripts cover the whole utterance, not our crop).
* Still missing, and only humans can supply it: does the protected audio sound
  like the SAME PERSON. A speaker encoder cannot judge this, because
  disagreeing with the encoder is the method's entire purpose.

`t_warp` (STFT-magnitude warp, original phase) is artefact-prone, so the fair
test of formant shifting is `t_mcadams`: per-frame LPC, pole phases raised to
alpha, resynthesis through the modified filter -- the VoicePrivacy B2 baseline
(Patino et al., arXiv 2011.01130). Doing it properly gains ~0.7 UTMOS at
comparable protection (a=0.8: 2.41 vs crude warp 1.69).

### Pareto frontier, 20 held-out speakers, attacker at t=200

Protection = % of speakers on the two *trained* ASV systems (WavLM-SV, ECAPA).

| condition | UTMOS | WER | protection |
|---|---|---|---|
| original | 4.12 | 0.000 | – |
| spectral tilt, any magnitude | 4.03–4.12 | 0.004–0.013 | **0 %** |
| reverberation, any RT60 | 3.74–4.06 | 0.004–0.035 | **0 %** |
| McAdams a=1.05 | 3.40 | 0.015 | 12 % |
| McAdams a=0.95 | 3.29 | 0.027 | 18 % |
| McAdams a=0.85 | 2.57 | 0.058 | 38 % |
| McAdams a=0.80 | 2.41 | 0.206 | 58 % |
| additive, plain | 2.77 | 0.127 | 70 % |
| **additive, purifier-aware (EOT)** | **2.73** | 0.145 | **90 %** |

The frontier is: tilt (free, useless) -> McAdams (natural, partial) ->
purifier-aware additive (best protection, same naturalness as plain).

### What this settles

1. **Purifier-aware additive crafting dominates.** At UTMOS 2.73 it protects
   90 %; the best on-manifold setting reaches 58 % at UTMOS 2.41. There is no
   operating point where formant shifting is preferable *on trained ASV*.
2. **Existence proof against the field's intuition.** Spectral tilt survives
   purification essentially untouched (PSR 0.98, R_t 0.000), is perceptually
   free (UTMOS 4.12 vs original 4.12, WER 0.4 %), and protects **0 %** of
   speakers at up to 18 dB. Imperceptible + purification-invariant + useless.
3. **The two families fail differently.** Additive protection costs
   intelligibility (WER 13–15 %) and little naturalness; formant shifting costs
   naturalness and (except at a=0.80) little intelligibility. Reporting only one
   metric hides this.
4. **One place on-manifold wins:** the Resemblyzer d-vector, where McAdams
   a=0.80 protects 80–85 % against the additive method's 55 %. d-vector leans
   harder on the spectral envelope, which is exactly what McAdams moves.

### Caveat

These are *fixed-magnitude* transforms. ClearMask optimises its transformation
parameters against an encoder ensemble; an optimised on-manifold protection
could sit higher on this frontier. What is established here is the behaviour of
the transformation families and the fact that survival does not buy protection —
not a refutation of any tuned system.

## 9. CORRECTIONS after the Paper-3 review (2026-09-16)

### 9.1 The headline was at the wrong operating point

Paired McNemar on `main_table_v3.csv`, EOT vs plain, verified independently twice
and then by us:

| budget | encoder | t | plain -> EOT | better / worse | p |
|---|---|---|---|---|---|
| A (PESQ 2.2) | WavLM-SV | 50 | 15/20 -> 19/20 | 5 / 1 | 0.219 |
| A | WavLM-SV | 200 | 15/20 -> 17/20 | 2 / 0 | 0.500 |
| A | ECAPA | 200 | 13/20 -> 18/20 | 5 / 0 | 0.063 |
| **B (PESQ 3.2-3.4)** | **WavLM-SV** | **50** | **9/20 -> 15/20** | **6 / 0** | **0.031** |
| **B** | **WavLM-SV** | **200** | **8/20 -> 14/20** | **6 / 0** | **0.031** |
| **B** | **ECAPA** | **50** | **4/20 -> 12/20** | **8 / 0** | **0.008** |
| **B** | **ECAPA** | **200** | **2/20 -> 10/20** | **8 / 0** | **0.008** |

**Every significant cell is at budget B, and no speaker is ever worse.** The
90 %-vs-70 % figure quoted throughout sections 5 and 8 is budget A, which is the
operating point with NO significant effect. The defensible headline is budget B:
at the perceptually cheaper setting, purifier-aware crafting roughly doubles to
triples protection, consistently and significantly. Sections 5 and 8 must be
rewritten around budget B.

### 9.2 The pass-through score R_t must be cut as a "cheap predictor"

Measured on `parametric_probe.csv` + `parametric_mcadams.csv`:

* At t=50 it has no dynamic range at all (0.0000-0.0081 over 400 conditions);
  rank correlation with survival -0.12.
* Its pooled correlation with survival is largely *between*-family, not within:
  -0.65 warp, -0.52 reverb, -0.31 tilt, ~0 EQ and McAdams.
* **It correlates POSITIVELY with protection** (rho +0.39 / +0.34 on WavLM-SV at
  t=50 / 200, p < 1e-14). Higher R_t means less predicted survival yet MORE
  protection, because formant warp is both the only protective family and the
  high-R_t family. R_t is behaving as a family indicator.

So R_t would be a cheap predictor of a quantity this work argues is the wrong
one, pointing the wrong way for the quantity that matters. Retain it only as the
objective used to *construct* noise-matched directions, in two sentences.

### 9.3 The honest baseline for the survival result

Section 7.1 compares optimised pass-through directions (PSR 0.98) against random
mel directions (0.61). The fair comparison is a **spectral tilt**, which also
reaches PSR 0.98 with no purifier access, no optimisation, and one filter. State
this ourselves: an expensive per-utterance white-box optimisation reproduces what
a one-line filter gives away for free. That is the paper's strongest sentence and
it costs only a number we should not have been claiming.

### 9.4 The framing that makes Paper 3 a paper: a double dissociation

| construction | purifier access | cost | survival | protection |
|---|---|---|---|---|
| spectral tilt | none | one filter | R_t 0.0001, PSR 0.98 | **0 / 480 trials** |
| noise-matched null direction | full white-box | 200-step optimisation | R_t 0.004, PSR 0.98 | **reduces** protection |

Two constructions at opposite extremes of cost and knowledge converge on the same
near-perfect survival, and neither protects. One arm is an anecdote; both arms
are a refutation.

## 10. THE STATISTICS WERE THROWING AWAY THE EVIDENCE (2026-09-16)

Every comparison in sections 5, 7 and 9 was tested on the BINARY outcome
(protected / not protected) over 20 speakers. With at most 8 discordant pairs the
exact test bottoms out at p = 0.0078, so almost nothing could reach significance
no matter how large the true effect. `main_table_v3.csv` also records the
CONTINUOUS decision margin (`cos_after_purify - threshold`, negative = clone
fails). Wilcoxon signed-rank on the paired margin over the same 20 files:

### 10.1 Purifier-aware crafting works, at BOTH budgets

| budget | encoder | t | binary p | margin shift | Wilcoxon p |
|---|---|---|---|---|---|
| A (PESQ 2.2) | WavLM-SV | 50 | 0.219 | **-0.128** | **<0.0001** |
| A | WavLM-SV | 200 | 0.500 | **-0.115** | **<0.0001** |
| A | ECAPA | 50 | 0.625 | **-0.219** | **<0.0001** |
| A | ECAPA | 200 | 0.063 | **-0.196** | **<0.0001** |
| B (PESQ 3.2-3.4) | WavLM-SV | 50 | 0.031 | -0.051 | **0.0002** |
| B | ECAPA | 200 | 0.008 | -0.113 | **<0.0001** |

**This supersedes section 9.1.** The effect was never confined to budget B; the
binary test simply could not see it. Report the margin shift as the effect size
and the binary rate as a secondary, interpretable summary.

### 10.2 The null-space constraint alone significantly HURTS

Two-sided Wilcoxon, plain vs plain + noise-matched regulariser. Positive shift =
protection worse.

| encoder | t | margin shift | worse on | p |
|---|---|---|---|---|
| d-vector | 50 | +0.034 | 18/20 | **0.0003** |
| d-vector | 200 | +0.019 | 17/20 | **0.0027** |
| WavLM-SV | 50 | +0.025 | 16/20 | **0.0107** |
| WavLM-SV | 200 | +0.013 | 15/20 | 0.070 |
| ECAPA | 50 | +0.023 | 13/20 | 0.133 |

Consistently worse on 15-18 of 20 files on every encoder. This is now a
**statistically supported negative result**, not a directional hint.

### 10.3 The null-space constraint on top of EOT does nothing

Margin shifts +0.006 / -0.006 / +0.043 / +0.021, Wilcoxon p = 0.31 to 0.97.
**Delete the "100 % at t=200" claim entirely** (sections 5c(iii), 8). It was three
discordant pairs where the arithmetic floor is p = 0.25, and the margin test finds
nothing. Both reviewers independently flagged it as the single most dangerous
sentence in the draft.

### 10.4 Consequence for all three papers

Every headline is now testable and three are decided: purifier-aware crafting
works (p < 0.0001), the subspace constraint alone hurts (p = 0.0003 on d-vector),
and the constraint on top of EOT does nothing. Report margins everywhere; keep
binary rates only as a readable summary alongside.

---

## 11. E6 — the vocoder control, and a metric that was lying to us (2026-09-16)

Every survival number in this project is a mel-domain ratio measured across the
purifier, but the attacker's real chain is `mel -> purify -> vocoder -> wav`.
Before any survival claim can be attributed to the purifier, the vocoder round
trip has to be ruled out as the cause. `experiments/exp_vocoder_control.py`
measures three pipelines on the same directions: vocoder only, purifier only,
and the full chain.

### 11a. PSR is not a safe metric, and it nearly cost us the result

The first run reported "the mel + Griffin-Lim round trip alone preserves 211 %
of a perturbation" and concluded the vocoder was transparent. Both halves of
that sentence are wrong, and they are wrong for the same reason: **a norm ratio
cannot distinguish a preserved perturbation from a destroyed one that has been
replaced by uncorrelated noise of similar size.** Griffin-Lim is emphatically
the second case, so PSR > 1 was measuring injected junk, not survival.

The fix is to decompose the output shift along the input perturbation:

```
d_out = alpha * d_in + residual
alpha = <d_out, d_in> / ||d_in||^2        signal gain   (1 = fully survives)
nsr   = ||residual|| / ||alpha * d_in||   noise-to-signal injected
```

`alpha` and `nsr` answer different questions and PSR confounds them. **All
survival numbers in the paper are reported as `alpha`.** PSR is retained in the
CSVs only for continuity with earlier tables.

### 11b. Result — the subspace is not a vocoder artifact

5 held-out speakers, purifier v3, DDIM, t = 200. Signal gain `alpha`
(higher = survives better), `nsr` in parentheses:

| direction | vocoder only | purifier only | full chain |
|---|---|---|---|
| random | 0.675 (3.30) | 0.493 (0.70) | 0.362 (5.02) |
| null_zero (zero-noise criterion) | 0.687 (2.82) | 0.538 (0.52) | 0.395 (4.85) |
| **null_matched (noise-matched R_t)** | **0.918 (1.93)** | **0.971 (0.11)** | **0.871 (2.55)** |

Repeated with the stronger phase-preserving resynthesis (magnitude from the
purified mel, phase from the waveform the attacker already holds), which is what
a competent attacker would actually use:

| direction | vocoder only | purifier only | full chain |
|---|---|---|---|
| random | 0.574 (3.82) | 0.493 (0.70) | 0.399 (3.23) |
| null_matched | 1.472 (1.54) | 0.971 (0.11) | 1.504 (1.47) |

Three things to take from this.

1. **The vocoder is not transparent, and it is not the explanation.** The round
   trip alone passes only 57–68 % of a random perturbation while injecting error
   3–4x the size of what survives. It degrades every direction, but it degrades
   them roughly equally, so it cannot manufacture the null/random gap.
2. **The purifier is where the separation happens.** Across the purifier alone,
   the noise-matched direction comes through at `alpha = 0.971` with `nsr = 0.11`
   — essentially untouched — against `0.493 / 0.70` for a random direction of the
   same norm. That is the pass-through subspace, measured cleanly, with the
   vocoder removed from the loop.
3. **The effect survives the full attacker chain**, under both vocoders
   (0.871 with Griffin-Lim, 1.504 with phase-preserving inversion). The
   confound we were worried about is not there.

The `null_zero` row also earns its place: the zero-noise criterion barely beats
random (0.538 vs 0.493), while the noise-matched criterion nearly doubles it.
This is independent confirmation, at the level of the vocoder control, of the
correction recorded in section 9 — **the criterion has to be matched to the noise
level the purifier actually operates at.** It is the difference between an effect
and no effect.

### 11c. What this does and does not license

It licenses the survival half of the paper: the subspace is real, it is a
property of the purifier, and measuring it across the purifier is sound. It says
nothing about protection. Section 10 already established that constraining an
attack into this subspace does not buy protection, and E6 removes the most
attractive excuse for that failure — we cannot blame the vocoder.

---

## 12. E2 — THE PASS-THROUGH SUBSPACE TRANSFERS ACROSS PURIFIERS (2026-09-16)

This was the go/no-go experiment. Everything measured so far characterises one
network, `purifier_v3`. If the subspace were an idiosyncrasy of that checkpoint,
the paper would be a case study of a single model and not a statement about
diffusion purification. So: find directions against v3, then evaluate them
through `purifier_v4` — a separately trained network that the optimiser never
saw.

20 held-out speakers (offset 20, utterance 3), DDIM, seeds {2026, 7}, RMS 0.2.
`--eval-ckpt` reports the Jacobian-level quantities on **both** networks, not
just the loop PSR, because the transfer question has to be asked at the level
where the subspace is defined.

| direction | lin gain v3 | lin gain **v4** | R_t200 v3 | R_t200 **v4** | loop PSR **v4** t=200 | dir cos **v4** t=200 |
|---|---|---|---|---|---|---|
| random | 0.053 | 0.083 | 0.621 | 0.624 | 0.604 | 0.803 |
| null (zero-noise criterion) | 0.317 | 0.517 | 0.497 | 0.494 | 0.601 | 0.883 |
| **null_t200 (noise-matched)** | **0.003** | **0.005** | **0.006** | **0.008** | **0.980** | **0.992** |
| range (worst-case, maximised) | 0.508 | 0.812 | 2.071 | 2.147 | 1.195 | 0.158 |

**Advantage over a random direction of the same norm: 109.8x on the purifier the
directions were optimised against, 76.2x on the one they were not.**

Three observations, in order of how much they matter.

1. **The subspace transfers.** A direction optimised to pass through v3 passes
   through v4 essentially untouched — loop PSR 0.980 with direction cosine 0.992,
   against 0.604 / 0.803 for a random direction. Losing only 109.8x -> 76.2x when
   moving to an unseen network is a small penalty for a property that could
   plausibly have been checkpoint-specific.

2. **The worst directions transfer too.** `range`, optimised to be maximally
   *destroyed* by v3, is also maximally destroyed by v4 (R_t 2.071 -> 2.147).
   This is the stronger form of the claim, and it is the one to put in the paper:
   it is not that we found a lucky subspace that happens to survive two networks.
   **The entire ordering of directions — which survive and which are erased — is
   shared between independently trained purifiers.** A single privileged subspace
   could be coincidence; a preserved spectrum cannot.

3. **The noise-matched criterion is doing the work.** `null` (the zero-noise
   criterion, which is what the original repo implemented) transfers at R_t 0.494
   versus random 0.624 — a 1.26x advantage, essentially nothing, and its loop PSR
   (0.601) is indistinguishable from random (0.604). The noise-matched criterion
   gives 76x. Section 9's correction is not a technicality; it is the difference
   between the result and no result.

### 12a. What this licenses, and the honest limit

v3 and v4 share an architecture (2-D residual UNet, base 32), a beta schedule,
and a training corpus; they differ in initialisation, data ordering, and
validation split. So the demonstrated claim is **"the subspace is a property of
the training procedure, not of the individual checkpoint"** — which is what rules
out the case-study reading. It is *not yet* "a property of diffusion purifiers of
speech in general." A second cell against a purifier of different capacity is
running (`purifier_v6`, base 64, seed 7, different crops); the 1-D UNet that was
meant to fill this slot (v5) was abandoned after its denoising gain sat at -0.5
for 7k steps, the same failure mode as v2, and a purifier that does not purify
cannot be used to test anything.

The paper must state the shared-architecture limitation in the same paragraph as
the transfer claim, not in a limitations section at the end.

### 12b. The statistics of the transfer (paired, n = 20 held-out speakers)

Measured on `purifier_v4`, which the directions were never optimised against.
Wilcoxon signed-rank, paired by speaker; 1.91e-06 is the smallest attainable
p-value at n = 20.

| quantity on v4 | null_t200 | random | speakers better | Wilcoxon p |
|---|---|---|---|---|
| pass-through error R_t | 0.0082 | 0.6242 | **20 / 20** | 1.91e-06 |
| loop PSR (t = 200) | 0.9798 | 0.6039 | **20 / 20** | 1.91e-06 |
| direction cosine (t = 200) | 0.9920 | 0.8027 | **20 / 20** | 1.91e-06 |

**Spearman correlation between the pass-through error on v3 and on v4, over all
80 directions: rho = 0.9967 (p = 4.7e-87).**

That last number is the one to lead with. It is not a statement about a
privileged subspace surviving two networks — it says that **the pass-through
error of an arbitrary direction under one purifier predicts its pass-through
error under an independently trained purifier almost perfectly, across two and a
half orders of magnitude.** The two networks agree not just on which directions
pass through, but on the entire ordering, including the directions each one
destroys most violently.

This is what makes the object worth naming. A defender-chosen subspace that
happened to survive two checkpoints would be an engineering coincidence. A
rank correlation of 0.997 over the full spectrum means both networks have learned
the same operator, and the subspace is a property of that operator rather than of
either network.

**Attribution correction (2026-09-16).** An earlier version of this section said
the Tweedie framing "predicts this in advance," implying the prediction was ours.
It is not. That independently trained diffusion models converge to the same score
function is published: Zhang et al., *The Emergence of Reproducibility and
Generalizability in Diffusion Models*, ICML 2024 (arXiv:2310.05264) —
*"different diffusion models consistently reach the same data distribution and
scoring function regardless of diffusion model frameworks, model architectures, or
training procedures"* — and Kadkhodaie et al., ICLR 2024 (arXiv:2310.02557) show
it at the denoiser level for non-overlapping training sets. **Our result verifies
a published prediction in a new threat model; it does not discover cross-model
agreement.**

What is not implied by that prior work, and is ours: the agreement holds for
*optimised adversarial* directions rather than random or isotropic ones, and it
holds over the *fine-grained ordering of the whole spectrum* — including the
directions each network destroys most violently — not merely over the coarse score
field or the generated samples. See `paper/related_work.md` section 10 for the
full audit.

Figure: `paper/fig2_transfer.png`.

### 12c. Bug found while producing this table — worth recording

`pandas.read_csv` treats the bare string `null` as a missing value. One of our
direction families is named `null` (the zero-noise criterion), so reading any of
the `nullspace_*.csv` files back silently converted 20 of 80 rows to NaN and
dropped them from every `groupby`. The published numbers were never affected —
the experiment scripts summarise their in-memory DataFrame, not a re-read — but
any downstream analysis would have quietly lost precisely the prior-art
formulation whose weakness is one of our results. Fixed centrally in
`utils/csvio.py` and applied to all six scripts that read results back.

---

## 13. E4 — survival at n = 20, with the criterion comparison made properly (2026-09-16)

The survival table in §2 was five files. This replaces it: 20 held-out speakers
(offset 20), purifier v3, DDIM, seeds {2026, 7}, RMS 0.2, evaluated at four
purification strengths including t = 300, which is **6x the strength the defender
assumes** and 1.5x the strongest level the purifier was trained on.

`experiments/results/nullspace_survival_20spk_u3.csv`

| direction | lin gain | R_t100 | R_t200 | PSR t=50 | t=100 | t=200 | t=300 | cos t=300 |
|---|---|---|---|---|---|---|---|---|
| range (worst case) | 0.508 | 12.258 | 2.071 | 1.513 | 1.434 | 1.115 | 0.613 | 0.229 |
| random | 0.053 | 0.588 | 0.621 | 0.828 | 0.746 | 0.605 | 0.473 | 0.776 |
| null (zero-noise criterion) | 0.317 | 0.243 | 0.497 | 0.896 | 0.782 | 0.604 | 0.477 | 0.828 |
| **null_t100 (matched)** | 0.004 | 0.004 | 0.036 | 0.988 | 0.978 | 0.933 | **0.848** | 0.977 |
| **null_t200 (matched)** | 0.003 | 0.001 | 0.006 | 0.992 | 0.991 | 0.982 | **0.938** | 0.991 |

### 13a. Paired statistics (Wilcoxon signed-rank vs a random direction, n = 20)

1.91e-06 is the smallest p-value attainable at n = 20.

| criterion | t=50 | t=100 | t=200 | t=300 |
|---|---|---|---|---|
| null (zero-noise) | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 | **6/20, p=0.12** | **13/20, p=0.57** |
| null_t100 (matched) | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 |
| null_t200 (matched) | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 | 20/20, p=1.9e-06 |

**The zero-noise criterion — which is what the original repo implemented and what
"null space of the denoiser" naturally means — is statistically indistinguishable
from a random perturbation at exactly the purification strengths that matter.**
It wins at t = 50 only because at t = 50 almost anything survives (random itself
is at 0.828). By t = 200 it is at 0.604 against random's 0.605. The
noise-matched criterion is at 0.982.

This is the cleanest available statement of §9's correction, and it is now an
n = 20 result with a paired test rather than an observation on five files.

### 13b. The subspace is not tied to the noise level it was found at

`null_t100` was optimised at t = 100. At t = 300 — three times that noise level,
and beyond the purifier's training range (t_max = 200) — it still passes through
at PSR 0.848 with direction cosine 0.977, against random's 0.473 / 0.776.

This matters for the interpretation. If the subspace were a quirk of the
operator at one noise level, a direction found at t = 100 would have no reason to
survive at t = 300. That it does is consistent with the Tweedie reading: the
criterion is identifying directions in which the log-density is locally flat, and
that flatness is a property of the data distribution, which smoothing changes
slowly rather than reorganising.

The `range` row is the control that makes this argument non-vacuous: directions
optimised to be *destroyed* are destroyed at every t (PSR 1.5 -> 0.61 with the
direction cosine collapsing to 0.23, i.e. what comes out is no longer the input
perturbation in any meaningful sense). The operator is not simply gentle at high
t; it is selectively gentle.

### 13c. Replication on a second, disjoint utterance per speaker

Same 20 held-out speakers, utterance index 6 instead of 3 — different recordings,
different content, same speakers. `nullspace_survival_20spk_u6.csv`.

| direction | R_t(200) | PSR t=50 | t=100 | t=200 | t=300 | cos t=300 |
|---|---|---|---|---|---|---|
| range | 2.078 | 1.466 | 1.391 | 1.086 | 0.602 | 0.236 |
| random | 0.622 | 0.828 | 0.747 | 0.604 | 0.472 | 0.779 |
| null (zero-noise) | 0.499 | 0.894 | 0.779 | 0.600 | 0.473 | 0.828 |
| null_t100 (matched) | 0.037 | 0.986 | 0.976 | 0.930 | 0.845 | 0.976 |
| **null_t200 (matched)** | 0.006 | 0.991 | 0.989 | 0.980 | **0.936** | 0.990 |

Side by side with utterance 3, at the hardest setting (t = 300, six times the
strength the defender assumes):

| | utterance 3 | utterance 6 |
|---|---|---|
| null_t200 PSR | 0.9379 | 0.9355 |
| random PSR | 0.4733 | 0.4721 |
| speakers better | 20/20 | 20/20 |
| Wilcoxon p | 1.91e-06 | 1.91e-06 |

**Every row agrees to three decimal places.** The zero-noise criterion is again
indistinguishable from random at t = 200 (0.600 vs 0.604) and t = 300 (0.473 vs
0.472), reproducing §13a exactly. The variance across utterances is far smaller
than the effect, so the n = 20 result is not resting on a favourable draw of
recordings.

### 13d. E4b — the memorisation control

A purifier can be unreasonably gentle on audio it was trained on. If the
pass-through subspace were partly a memorisation effect, it would be wider or
cleaner on the purifier's own training speakers than on held-out ones, and the
entire result would be an artefact of overfitting rather than a property of the
operator. `nullspace_survival_trainspk.csv`: 16 of the purifier's **training**
speakers (offset 0), same settings as section 13.

| direction | R_t(200) training | R_t(200) held-out | PSR t=200 training | PSR t=200 held-out |
|---|---|---|---|---|
| random | 0.6286 | 0.6213 | 0.6102 | 0.6052 |
| null (zero-noise) | 0.4954 | 0.4965 | 0.6049 | 0.6040 |
| **null_t200 (matched)** | **0.0058** | **0.0057** | **0.9869** | **0.9822** |

**The subspace is the same on speakers the purifier memorised and speakers it has
never seen** — the pass-through error agrees to four decimal places, and the
zero-noise criterion is equally dead in both. The `range` row behaves identically
too (2.132 training vs 2.071 held-out).

This closes the third deflationary reading. The subspace is not specific to a
checkpoint (E2), not specific to a noise level (13b), not specific to an utterance
(13c), and not an artefact of the training set (here). What remains is the
reading the prior work already supports: it is a property of the score function
the purifier approximates, and therefore of the data.

---

## 14. E5 — how big is the pass-through subspace? We did not find its edge (2026-09-16)

Everything up to here characterises the subspace by one optimised direction. That
is an anecdote, not a geometry. E5 measures its effective dimension by greedy
orthogonal deflation (the method of Tramer et al. 2017, applied here to a
purifier rather than a classifier): find the best pass-through direction, then the
best direction orthogonal to it, and so on, re-projecting onto the orthogonal
complement at every optimiser step.

5 held-out speakers, purifier v3, DDIM, noise-matched criterion at t = 200,
perturbation RMS 0.2. Ambient dimension 80 mels x 251 frames = **20,080**.
`nullspace_dimension.csv`, figure `paper/fig3_dimension.png`.

| k (mutually orthogonal directions) | R_t | as % of random | loop PSR |
|---|---|---|---|
| random direction, same norm | 0.60930 | 100 % | 0.604 |
| 1 | 0.00880 | 1.4 % | 0.973 |
| 6 | 0.00884 | 1.5 % | 0.975 |
| 12 | 0.00851 | 1.4 % | 0.975 |
| 18 | 0.00845 | 1.4 % | 0.976 |
| 24 | 0.00907 | 1.5 % | 0.975 |
| 30 | 0.00860 | 1.4 % | 0.976 |
| 36 | 0.00896 | 1.5 % | 0.973 |
| 42 | 0.00876 | 1.4 % | 0.975 |
| **48** | **0.00820** | **1.3 %** | **0.977** |

**The curve is flat.** R_t at k = 48 is *lower* than at k = 1 (0.00820 vs
0.00880); the linear trend is +4.1e-06 per direction, which is indistinguishable
from zero. Loop PSR never leaves the band 0.973-0.977. Demanding 48 mutually
orthogonal pass-through directions costs nothing at all.

### 14a. State this honestly: we found a lower bound, not the dimension

We did **not** find the edge of the subspace. The sweep stopped at k = 48 because
that is what the compute budget allowed, not because the criterion began to
degrade. The defensible claim is:

> The pass-through subspace contains **at least 48 mutually orthogonal
> directions**, each surviving purification at 97 % signal with a pass-through
> error 1.4 % of a random direction's, and the deflation curve shows no sign of
> approaching a limit.

Extrapolating the fitted slope to the random baseline would put the boundary near
150,000 directions — beyond the ambient dimension of 20,080, i.e. the linear fit
is simply measuring noise. The sentence to avoid is "the subspace is
48-dimensional." What we can say is that it is **large**, and that 48 is a floor.

### 14b. This kills one of the two explanations for the negative result

E5 was designed so that either outcome would be publishable, and the two outcomes
carried opposite mechanisms:

* *If R_t had risen quickly with k*, the subspace would retain few degrees of
  freedom, and that would **mechanistically explain** why constraining an attack
  into it costs protection: the constraint would be discarding most of the
  attack's search space.
* *If R_t stayed flat*, that explanation is unavailable.

**R_t stayed flat, so the explanation is unavailable.** Confining a perturbation
to the pass-through subspace does not meaningfully restrict the attacker: there
are at least 48 orthogonal directions to work in, all of which survive, and the
perturbation is not being forced onto a knife edge.

This matters because it is the obvious defence of the null-space idea — "of course
it did not protect, you gave the optimiser no room" — and the data refuses it. The
constraint is cheap in degrees of freedom and still buys nothing. What remains is
the mechanism in section 7: ASV encoders are trained with channel, reverb and
codec augmentation, so they are *already invariant* to precisely the directions
that survive purification. Survival and adversariality are selecting for opposite
properties, and the failure is not a budget problem.

### 14c. Method attribution

Greedy orthogonal deflation to estimate an adversarial subspace's dimension is
**Tramer, Papernot, Goodfellow, Boneh, McDaniel, *The Space of Transferable
Adversarial Examples*, arXiv:1704.03453 (2017)** — they report ~25 orthogonal
directions typical and 44 on MNIST, and they link subspace dimension to
transferability, which is also the published explanation for our E2. The algorithm
is theirs and must not be presented as new. Ours is the application: the first
measurement of the effective dimension of a **diffusion purifier's pass-through
subspace**, as distinct from Stanczuk's score-vector SVD and Kamkari's trace-based
FLIPD, both of which estimate the data manifold's intrinsic dimension rather than
the subspace of perturbations that survive a particular map.

---

## 15. E7/E8 — the subspace is shared across utterances, and it is not a filter (2026-09-16)

E2 ruled out "idiosyncrasy of one checkpoint". E4b ruled out "memorised training
speakers". This closes the last deflationary reading: the subspace could still be
local to the utterance it was optimised on, making it a per-example curiosity.

6 held-out speakers, purifier v3, DDIM, noise-matched criterion at t = 200.
Directions optimised on one utterance, evaluated at every other.

| | R_t | vs random | loop PSR (t=200) | dir cosine |
|---|---|---|---|---|
| random direction | 0.62411 | 1.0x | — | — |
| optimised **on** this utterance | 0.00419 | **148.8x** | 0.983 | 0.993 |
| optimised on **another** utterance | 0.03652 | **17.1x** | 0.939 | 0.978 |

**Mean |cos| between two directions from different utterances: 0.016.**

The two numbers together are the result. The directions are **near-orthogonal to
each other** — 0.016, against 0.005 for independent random directions — and yet
each one passes through the *other* utterance 17x better than a random
perturbation, at 94 % signal. They are therefore not one shared vector reused; they
are different vectors drawn from one shared, high-dimensional subspace. That is
the much stronger form of the claim, and it is consistent with E5's finding that
the subspace admits at least 48 mutually orthogonal directions.

### 15a. A confound that inverted the reading, and was caught by a built-in control

The first run of E7 reported mean |cos| = **0.690** and triggered the
"largely the SAME vector" branch, which would have meant the construction was
reducible to a common component — plausibly a global spectral tilt, which we know
passes through at R_t ~ 1e-4 and protects 0/480. Had that been true the paper's
central construction would have been an expensive way to rediscover a
one-parameter filter.

**It was an artefact of the experiment, not a property of the subspace.** Every
speaker's optimisation was initialised from `torch.manual_seed(0)`, so all runs
started from the *same* random vector and inherited its correlations. The tell was
in the diagnostic's own baseline: random directions were reported with pairwise
cosine **1.000**, which is impossible for genuinely independent draws. Seeding per
speaker gives 0.016.

Two things worth keeping from this:

1. **The shared seed is correct within a speaker and wrong across speakers.** Within
   a speaker it makes `random` a controlled baseline against the other direction
   kinds, which is exactly what sections 12-14 need; only the cross-example
   comparison is affected. E2, E4, E4b, E5 and E6 are unaffected — verified by
   inspection of what each compares.
2. **The transfer number barely moved** (16.5x -> 17.1x). The confound inflated the
   *cosine*, not the transfer, so the qualitative conclusion was right for the
   wrong reason until it was fixed. A baseline that reports an impossible value is
   worth more than one that reports a plausible one.

### 15b. E8 — the directions are not a static filter

`exp_direction_structure.py` decomposes each optimised direction by energy. A
random direction over T frames carries ~1/T of its energy in any time-constant
component, so the random column fixes the scale.

| component | pass-through | random |
|---|---|---|
| time-constant (the whole static-EQ family) | **0.0275** | 0.0040 |
| single linear tilt ramp | **0.0010** | 0.0000 |
| quietest quartile of frames | 0.1979 | 0.2500 |

The static-EQ family accounts for **2.8 %** of the direction's energy and the tilt
ramp for **0.1 %**. The construction is not rediscovering an EQ curve. Nor is it
hiding in silence: it puts *less* energy in the quietest frames than a random
direction does (0.198 vs 0.250), so it is not exploiting pauses either.

This matters for the double dissociation in section 8. The spectral tilt and the
optimised pass-through direction reach comparable survival by **genuinely
different means** — the tilt is a one-parameter static filter, the optimised
direction is 97 % time-varying structure nearly orthogonal to it. Two
mechanistically unrelated constructions converging on the same near-perfect
survival, and neither protecting, is a much stronger argument than two variants of
one trick.

---

## 16. E9 — THE SUBSPACE TRANSFERS TO PURIFIERS WE DID NOT TRAIN (2026-09-16)

E2 established transfer from purifier v3 to v4. Both are runs of one recipe, so
that result supports "a property of the training procedure" and no more. E9 is the
experiment that decides whether the claim can be about **speech diffusion
purifiers**. Setup, provenance, licensing and t-calibration are in
`paper/related_work.md` section 12.

Two public unconditional DiffWave checkpoints, each foreign in architecture
(WaveNet on **raw waveform**, not our 2-D mel UNet), weights, training data and
domain:

* **DiffWave-LibriSpeech** — De-AntiFake's `purification.pkl` (Fan et al., ICML
  2025; weights MIT). This is the purifier from the paper whose call for
  purification-robust protection this work answers, trained on our own corpus.
* **DiffWave-SC09** — philsyn's unconditional checkpoint (MIT), trained on Speech
  Commands spoken digits. A deliberately narrow prior, reported as corroboration.

10 held-out speakers, directions optimised against **our** v3 in mel, rendered to
waveform, purified by the foreign model, measured back in mel. Signal gain alpha;
paired Wilcoxon against a random direction of equal norm.

| foreign purifier | t | alpha pass-through | alpha random | better | p | excess over floor | ours | retention |
|---|---|---|---|---|---|---|---|---|
| DiffWave-LibriSpeech | 1 | 0.713 | 0.357 | **10/10** | 2.0e-03 | 1.57x | 1.99x | **79 %** |
| DiffWave-LibriSpeech | 3 | 0.501 | 0.222 | **10/10** | 2.0e-03 | 1.78x | 2.01x | **89 %** |
| DiffWave-SC09 | 1 | 0.729 | 0.360 | **10/10** | 2.0e-03 | 1.59x | 2.02x | **78 %** |
| DiffWave-SC09 | 3 | 0.538 | 0.222 | **10/10** | 2.0e-03 | 1.91x | 2.00x | **95 %** |

`excess over floor` is the purifier's contribution after dividing out the vocoder
round trip, which is itself 1.27x in favour of the pass-through direction (E6:
that direction is smoother and survives mel inversion better). `ours` is our v3 at
t = 200 measured through the identical lossy pipeline. `retention` is the foreign
excess as a fraction of ours.

### 16a. What makes this convincing is the consistency, not any single number

> **SUPERSEDED (2026-09-17).** The table above is the n = 10 pilot. The n = 20
> rerun is on disk (`foreign_*_n20.csv`) and is what the draft quotes: retention
> **86 / 106 / 87 / 107 %**, all 20/20 speakers, p = 1.9e-06. The n = 10 numbers
> understated the effect. Kept for the record; cite the n = 20 figures.

Retention sits in a **78-95 % band** across two independently trained models, two
purification strengths, and a large corpus mismatch, with every one of the four
cells separating on all 10 of 10 speakers. Three things follow that a single cell
could not establish.

1. **It is not an artefact of how we chose t.** If transfer appeared only at the
   damage-matched setting (t = 1) it would look tuned. It *strengthens* at t = 3,
   where the foreign purifier does markedly more damage (mel damage 0.48 vs our
   0.29) than our own strongest setting: retention 79 % -> 89 % on LibriSpeech and
   78 % -> 95 % on SC09.
2. **It is not specific to the corpus the foreign model saw.** A model trained on
   *spoken digits in 1 s clips* transfers as well as one trained on LibriSpeech
   (78 % vs 79 % at matched damage). Whatever the subspace is, it is not an
   artefact of LibriSpeech statistics.
3. **It crosses a domain boundary.** The directions were optimised against a
   mel-domain 2-D UNet and evaluated through a WaveNet operating on raw waveform
   samples. The two models share no architecture, no parameterisation and no input
   representation.

### 16b. Scope: what this licenses, stated exactly

The claim that is now supported is that the pass-through subspace is a property of
**diffusion purifiers of speech**, not of our training procedure. Combined with
E4b (not memorisation), E7 (not the utterance) and E5 (at least 48 orthogonal
directions), the object is about as well established as this project can make it.

This is also, per section 10.1, exactly what the prior work *predicts*: Zhang et
al. (ICML 2024) show independently trained diffusion models converge to the same
score function, and Yuezhang & Wei (arXiv:2505.22839 v2) characterise the
small-|eigenvalue| eigendirections as preserved under purification. **We verify
their prediction across a domain and architecture boundary they did not test, for
optimised adversarial directions rather than isotropic noise.** The paper must
present it that way.

### 16c. Caveats that travel with these numbers

* **Do not compare these ratios to E2's 76x.** That was a mel-domain R_t ratio
  with no vocoder in the path. This pipeline is lossy by construction (mel -> wav
  -> mel, E6: alpha 0.57-0.68 with injected noise 3-4x the surviving signal) and
  compresses the dynamic range enormously. The only valid comparison is the
  foreign cell against the `ours` cell *within this table*, which is why every run
  prints all three pipelines and the script emits this warning itself.
* **The vocoder floor is 1.27x, not 1x**, and an earlier version of the verdict
  function credited the purifier with that head start, printing "TRANSFERS" off an
  uncorrected 1.91x. All figures above are excess over the measured floor.
* **t-calibration is approximate.** Damage was matched by feeding DiffWave the
  original waveform while our purifier's figure includes the vocoder round trip.
  Inside the experiment both receive identical vocoded input, which is the
  comparison that carries the claim; the calibration only chooses the bracket.
* **n = 10, not the n = 20 used elsewhere.** p = 2.0e-03 is the *smallest
  attainable* two-sided Wilcoxon p at n = 10, so it reports "consistent across
  every speaker", not "overwhelmingly powered". The substantive claim is the
  effect size (a 2x separation, 78-95 % retention). **DONE — rerun at n = 20 is on
  disk; see the superseded note above. Originally flagged: rerun before
  submission** for consistency with sections 12-15.
* **SC09 could not have carried a null.** Its prior is spoken digits, so a failure
  there would have been ambiguous between "the subspace is ours" and "the speech
  distribution is too different". It corroborates; LibriSpeech carries the claim.

---

## 17. E10 — PROTECTION THROUGH A FOREIGN PURIFIER, AND THE TRADE THE CONSTRAINT MAKES (2026-09-16)

Every protection number up to here was measured against **our own** purifier v3.
That is the single weakest point in the paper: a surrogate we built, evaluated
against itself. E9 gave us two public purifiers foreign in architecture, weights,
data and domain. E10 re-runs the protection experiment against them.

No re-crafting: the four variants that carry the argument were already saved for
all 20 held-out speakers, so this is a re-evaluation.

* `plain` baseline additive crafting
* `nsT` pass-through constraint **alone**
* `full` purifier-aware EOT crafting
* `fullnsT` EOT **plus** the constraint

**Thresholds were recalibrated through every channel.** EER thresholds from
"our v3 + Griffin-Lim" are meaningless against a different purifier; reusing them
would have invalidated every number. Channel EERs (WavLM / ECAPA / d-vector):

| channel | EER |
|---|---|
| clean (no purifier) | 3.33 / 0.03 / 0.09 % |
| DiffWave-LibriSpeech t=1 | 4.16 / 0.84 / 0.09 % |
| DiffWave-LibriSpeech t=3 | 8.33 / 1.67 / 0.90 % |
| DiffWave-SC09 t=1 | 3.98 / 0.84 / 0.10 % |
| DiffWave-SC09 t=3 | 7.46 / 1.86 / 0.84 % |
| *(our v3 + GL, t=50, for reference)* | *6.7 / 0.2 / 0.8 %* |

All foreign channels preserve unprotected identity as well as ours does, so
protection measured against them is meaningful. Note they are slightly **gentler**
on WavLM than our own channel, which makes protection *harder*, not easier.

### 17a. The negative result survives

Constraint **alone** versus plain, on the decision margin (positive = worse):

| channel | delta | better | p |
|---|---|---|---|
| our v3, t=50 | +0.0273 | 13/60 | <0.0001 |
| our v3, t=100 | +0.0237 | 17/60 | 0.0002 |
| our v3, t=200 | +0.0114 | 17/60 | 0.0050 |
| DiffWave-LibriSpeech t=1 | +0.0040 | 23/60 | 0.28 |
| DiffWave-LibriSpeech t=3 | +0.0109 | 26/60 | 0.035 |
| DiffWave-SC09 t=1 | +0.0007 | 24/60 | 0.58 |
| DiffWave-SC09 t=3 | +0.0035 | 26/60 | 0.50 |

**The constraint is never a protection mechanism in its own right**, against any
purifier. Worth stating precisely, because the p-values differ by two orders of
magnitude: it is significantly *harmful* against the purifier it was computed on
and merely neutral against foreign ones. A reviewer will notice that asymmetry, so
the paper should state it rather than round it to "it never helps".

### 17b. But on top of EOT it depends entirely on WHOSE purifier

`fullnsT` versus `full` (negative = the constraint protects better):

| attacker's purifier | WavLM-SV | ECAPA-TDNN | d-vector |
|---|---|---|---|
| **none at all** | **+0.1624, 0/20, p<0.0001** | **+0.2916, 0/20, p<0.0001** | +0.0251, 5/20, p=0.006 |
| ours v3, t=50 (matched) | +0.0207 pooled over encoders, 24/60, p=0.027 | | |
| ours v3, t=100 (matched) | +0.0166 pooled, 23/60, p=0.056 | | |
| ours v3, t=200 (matched) | +0.0065 pooled, 30/60, p=0.68 | | |
| DiffWave-LibriSpeech t=3 | -0.0248, 15/20, p=0.033 | -0.0372, 15/20, p=0.005 | +0.0001, 9/20, ns |
| DiffWave-SC09 t=3 | -0.0449, 18/20, p=0.001 | -0.0443, 17/20, p=0.002 | -0.0066, 14/20, ns |
| DiffWave-LibriSpeech t=1 | **-0.0815, 19/20, p<0.0001** | **-0.0876, 18/20, p=0.0003** | +0.0085, 7/20, ns |
| DiffWave-SC09 t=1 | **-0.0912, 19/20, p<0.0001** | **-0.0949, 17/20, p=0.0001** | +0.0069, 10/20, ns |

**8 of 8 tests on WavLM and ECAPA are significant and in the same direction**,
across two independently trained foreign purifiers and two strengths. Two
purifiers sharing nothing but architecture family give effect sizes within 10 % of
each other. d-vector is null in all four foreign cells, consistently.

### 17c. What the constraint actually does — a trade, not insurance

Read the rows in order and the pattern is monotonic in *how far the attacker's
purifier is from the defender's assumption*:

| channel | constraint |
|---|---|
| nothing purifies | **much worse** (0/20 files) |
| the purifier you crafted against | break-even to slightly worse |
| a foreign purifier, strong | better |
| a foreign purifier, matched strength | **clearly better** |

So the constraint is **not** generically robust — with no purifier in the loop it
loses on *every single file*, by a large margin. **It trades raw adversarial
strength for purification survival.** That trade is a loss when nothing purifies,
a wash against a known purifier (EOT's gradients already handle that one), and a
win when the attacker's purifier differs from the assumption.

This is exactly what E9 predicts and is the mechanism, not an analogy: EOT
gradients encode the specific model they were computed on, while the pass-through
subspace is shared across purifiers (E9: 86-107 % retention). Purifier-specific
information does not transfer; subspace membership does.

**And the trade is perceptually free.** Equal PESQ (2.239 vs 2.233, p=0.50), and
the constrained perturbation is *less* audible: +0.96 dB SNR (p<0.001) and +0.008
STOI (p=0.003). It cannot be dismissed as a larger perturbation.

### 17d. The verdict helper was confidently wrong, twice

`analyse_mismatch.py` branched on the p-value alone and announced the no-purifier
control as *"the constraint helps even with NO purifier -- GENERIC ROBUSTNESS"*.
The delta was **+0.16 with 0/20 files better**, i.e. a large, highly significant
**deficit** -- the opposite conclusion. Earlier the same day the vocoder-control
verdict credited the purifier with the vocoder's own 1.27x head start and printed
"TRANSFERS" off an uncorrected ratio.

Both were caught by reading the numbers rather than the conclusion line, and both
are now fixed with the reasoning recorded in the code. **These helpers are for
scanning, never for quoting.** A printed verdict is an assertion about the data,
and it needs the same scepticism as any other assertion -- more, because it looks
authoritative and sits right next to the evidence.

### 17e. Consequence for the paper

The thesis changes from *"survival is the wrong objective"* to something sharper
and more useful:

> Survival is the wrong objective **against the purifier you assumed**, and the
> right one **against the purifier you did not**. Optimising survival costs raw
> adversarial strength; that cost is wasted under a matched purifier and repaid
> under a mismatched one.

Since a defender publishing audio cannot know what an attacker will run,
**purifier mismatch is the realistic threat model**, and the pass-through
construction earns its place there. The paper keeps its negative result intact
(the constraint alone never protects) and gains a positive one with a mechanism
behind it.

---

## 18. E11 — VOICE CLONING: the protection works, and the mismatch finding does NOT transfer (2026-09-17)

Every protection number before this section is speaker verification. The threat is
cloning. E11 runs the real attack: **DiffVC** (Popov et al., ICLR 2022 Oral), the
voice-conversion model used by De-AntiFake, VoiceCloak and ClearMask -- more than
any other system -- and the one De-AntiFake chose for its own adaptive-protection
experiment.

Attacker chain: `victim publishes x+delta` -> `attacker purifies` -> `DiffVC(source
= attacker's speech, reference = the published clip)` -> verifier compares the clone
against a **third, held-back** utterance of the victim. Protection is applied to the
reference, never the source. AntiFake's pre-filter is applied per encoder: only
speakers whose UNPROTECTED audio cloned are scored.

### 18a. Q1/Q2 -- protection stops cloning, and survives purification ✅

Clone success rate, % (the attack WORKED; lower = better protection), n = 20 before
the pre-filter, budget m20_f20:

| attacker's purifier | arm | ECAPA | d-vector | WavLM |
|---|---|---|---|---|
| none | unprotected | 62 | 86 | 67 |
| none | protected (best arm) | **0** | **14** | **0** |
| ours (matched) | unprotected | 38 | 43 | 78 |
| ours | protected (best arm) | **15** | **21** | **11** |
| DiffWave (foreign) | unprotected | 50–92 | — | — |
| DiffWave | protected (best arm) | **0** | **6** | **0** |

At the louder m30_f15 budget protection is near-total (0-12 % against 58-81 %
unprotected).

**This closes the paper's largest gap.** The ASV result transfers to the real
threat: protection defeats a published voice-conversion model, and keeps defeating
it after the attacker purifies with a published diffusion purifier.

*Caveat worth stating:* De-AntiFake's section 5.4 tried adaptive protection with
**BPDA**+EOT and concluded it fails. We use **exact** gradients and it works. That
is a genuine methodological difference and a genuine reversal -- but we attack
DiffWave alone, not their full two-stage PhonePuRe (DiffWave + phoneme-guided
refinement), so it is **not** a direct refutation of their result and must not be
written as one.

### 18b. Q3 -- the purifier-mismatch finding does NOT transfer ❌

Section 17 found, in ASV, that the pass-through constraint added to EOT buys
nothing under a matched purifier and real protection under a mismatched one (8/8
tests significant). **That does not replicate against cloning.**

`fullnsT` vs `full`, budget m20_f20, negative delta = constraint protects better:

| attacker's purifier | WavLM-SV | ECAPA-TDNN | d-vector |
|---|---|---|---|
| none | +0.016, 4/9, p=0.36 | +0.004, 7/13, p=0.84 | -0.000, 6/14, p=0.81 |
| **ours (matched)** | +0.005, 5/9, p=0.82 | +0.003, 6/13, p=0.84 | -0.000, 5/14, p=0.67 |
| **DiffWave (mismatched)** | **+0.076, 2/9, p=0.039** | +0.037, 6/13, p=0.34 | -0.006, 7/14, p=0.54 |

Eight of nine cells show nothing. The ninth is **significant in the wrong
direction** -- under the mismatched purifier the constraint makes protection
*worse* on WavLM, the opposite of the ASV prediction. With nine tests at n = 9-14,
a single nominal p = 0.039 is about what chance produces, so we do not claim a
reversal either; we claim **no evidence of transfer**.

### 18c. Why this is a real negative and not a floor effect

The first cloning run used the louder **m30_f15** budget and also showed nothing --
but there it was **uninformative by construction**: every arm drove clone success
to ~0 % (ECAPA similarity 0.078-0.140 against a 0.376 threshold), so no arm could
be distinguished from any other. That is a saturated experiment, not a negative
result, and reporting it as one would have been wrong.

**m20_f20 was run specifically to obtain headroom**: ASV protection there spans
5-80 %, and cloning success spans 0-33 %, so a difference between arms could have
appeared. It did not. Both budgets are reported; m30's floor effect is stated as
the reason it cannot speak to Q3.

*On the choice of budget:* selecting the regime where an effect **can** be resolved
is legitimate when the alternative is uninformative by construction. It would be
cherry-picking only if both had been run and just the agreeable one reported --
hence both are here regardless of which supports the thesis.

### 18d. Consequence for the paper -- the headline must change

Section 17's mismatch result is now established as **ASV-scoped**. It is real in
speaker verification (8/8 tests, two foreign purifiers, two strengths) and does not
appear in voice cloning. The paper cannot lead with it as a general claim about
protection.

What the paper should lead with instead, in order of how well each is supported:

1. **The geometry (section 4).** Nine deflationary readings closed; transfers to
   three of our purifiers (rho = 0.994-0.997) and two foreign ones (86-107 %).
   The most thoroughly controlled result in the project.
2. **Cloning (18a).** Protection defeats a published VC model and survives a
   published purifier. New, comparable to baselines, in the real threat model.
3. **Survival is not sufficient.** The spectral tilt protects 0/480 despite perfect
   survival; the constraint alone never protects against any of five channels.
4. **The mismatch effect, scoped to ASV**, with 18b stated plainly as its limit.

A reviewer who finds the ASV/cloning divergence themselves would be entitled to
distrust the whole paper. Stated by us, with the saturation analysis showing we
looked for it properly, it is evidence of care.

---

## 19. CORRECTION to section 18 — the pre-filter was unioned across purifiers (2026-09-17)

Section 18's cloning numbers were computed with a pre-filter that did **not**
condition on the attacker's purifier. `prefilter()` built its keep-set from
`df[(df.variant=="unprotected") & (df.encoder==n)]`, pooling every purifier
condition. A speaker whose clean audio cloned only with **no** purifier in the loop
was therefore retained and scored under `ours` and `diffwave` — conditions where the
unprotected attack had already failed. Those speakers were free protection
successes, which is precisely what AntiFake's filter exists to exclude:

> *"AERR is defined as the rate at which DeepFake samples can no longer bypass the
> corresponding SV system, **given that all samples have already been filtered to
> bypass SV without AntiFake**."*

A code comment asserted the union was intentional. It was not. Fixed to filter per
`(encoder, purifier)`.

### 19a. Corrected results, budget m20_f20

Clone success %, i.e. the attack WORKED (lower = better protection). Unprotected is
100 % by construction after the filter — that is the point of it.

**No purifier** (n = 8 / 12 / 6 for ECAPA / d-vector / WavLM):

| arm | ECAPA | d-vector | WavLM |
|---|---|---|---|
| unprotected | 100 | 100 | 100 |
| plain | 12 | 8 | 17 |
| nsT | 0 | 17 | 0 |
| full | 0 | 25 | 0 |
| fullnsT | 12 | 25 | 0 |
| **mean over protected arms** | **6.0** | **18.8** | **4.3** |

**Attacker purifies with OUR purifier** (matched; n = 5 / 6 / 7):

| arm | ECAPA | d-vector | WavLM |
|---|---|---|---|
| unprotected | 100 | 100 | 100 |
| plain / nsT / full / fullnsT | 40 | 50 | 43 / 43 / 14 / 29 |
| **mean over protected arms** | **40.0** | **50.0** | **32.2** |

**Attacker purifies with DiffWave** (foreign): see `analyse_cloning.py` output;
same direction, intermediate magnitude.

### 19b. What changed, and what it costs the paper

| matched-purifier protected rate | as published in §18a | corrected |
|---|---|---|
| ECAPA-TDNN | 15 % | **40 %** |
| Resemblyzer d-vector | 21 % | **50 %** |
| WavLM-SV | 11 % | **14 %** (best arm) / 32 % (mean) |

**The claim "protection stops cloning" survives, but weakened and scoped.** Against
an unpurified attacker it holds strongly — 6 / 19 / 4 % mean across protected arms
against 100 % unprotected. Against an attacker who purifies with the model the
perturbation was crafted against, **the clone still succeeds 40–50 % of the time.**
That is not "protection stops cloning"; it is "protection substantially reduces
cloning, and a matched-purifier attacker recovers about half of it."

Two further reporting changes made at the same time:

* **"Best arm" is no longer quoted.** A minimum over four arms at n = 5–14 is a
  biased estimator. Every arm is printed, plus the mean across protected arms.
* **Per-cell n is now printed.** The filter leaves 5–7 speakers per cell in the
  matched condition, not the 13–14 the union produced. Any claim at that n needs a
  binomial interval; a 0 % at n = 5 has a 95 % upper bound near 45 %.

### 19c. How this was found

An adversarial statistics audit recomputed the filter both ways and reported the
discrepancy. The bug was in code this project wrote, with a comment defending the
behaviour — so neither review of the code nor reading the output would have caught
it. What caught it was recomputing a published number by an independent route.

---

## 20. F1 — the cloning result was white-box. Held-out-encoder ablation (2026-09-17)

### 20a. What was wrong

DiffVC's speaker conditioner is **the same file** as the Resemblyzer encoder in our
crafting ensemble: identical git blob SHA-1 `47e405a459e4b97a21325dc78ba6f8270498ac06`,
17,090,379 bytes, in both `huawei-noah/Speech-Backbones` and `resemble-ai/Resemblyzer`.
One third of the ensemble *was* the cloner's identity pathway. Every cloning number
in section 18/19 was therefore produced with white-box access to the network the
attacker reads identity with.

### 20b. The ablation

`exp_purifier_aware.py --craft-encoders no-dvec` re-crafts all four m20_f20 arms
against WavLM-SV and ECAPA only. Resemblyzer is still **evaluated**, so it becomes a
held-out encoder. Everything else is pinned to the published run: `dev-clean` (via
the new `--subset` flag, since `EVAL_SUBSET` had moved to `test-clean` the same day),
speakers 20-39, utterance 3, purifier v3/DDIM, the v3/DDIM t=50 thresholds, 200 steps.
The only difference is the crafting ensemble.

Runner `experiments/run_f1_nodvec.sh`, analysis `experiments/analyse_f1.py`,
results `main_table_f1.csv` and `cloning_protection_f1.csv`.

### 20c. ASV: the held-out encoder goes to zero

Protection rate (margin < 0), mean over the four arms, n = 20 per cell:

| encoder | published | held out | delta |
|---|---|---|---|
| WavLM-SV | 56-61 % | 35-40 % | -16 to -26 pts |
| ECAPA-TDNN | 31-38 % | 15-27 % | -11 to -16 pts |
| **Resemblyzer (held out)** | **13-21 %** | **0 %** | **-13 to -21 pts** |

The margin shift on Resemblyzer is significant at **p < 0.0001 in all 9 cells**
(3 arms x 3 purification strengths, paired Wilcoxon, n = 20). The two encoders that
remain in the ensemble also lose ground — they now carry alone what three gradients
shared — but they retain real protection.

### 20d. Cloning: the claim does not survive intact

Clone success %, i.e. the attack WORKED. Unprotected is 100 % by construction of the
AntiFake pre-filter. n = 5-12 per cell.

**Attacker does not purify:**

| arm | ECAPA pub -> held | d-vector pub -> held | WavLM pub -> held |
|---|---|---|---|
| plain | 12 -> **75** | 8 -> **58** | 17 -> **67** |
| nsT | 0 -> **62** | 17 -> **75** | 0 -> **50** |
| full | 0 -> **38** | 25 -> **58** | 0 -> **17** |
| fullnsT | 12 -> **25** | 25 -> **50** | 0 -> **17** |
| *mean over protected arms* | *6 -> **50*** | *19 -> **60*** | *4 -> **38*** |

Clone success rises in **9 of 9 (encoder x purifier) cells**, by 5 to 44 points.

### 20e. Two things this establishes, one of which was not anticipated

**1. The published cloning result was white-box.** "100 % -> 4-19 %" becomes
"100 % -> 38-60 %" on the mean, or "100 % -> 17-58 %" for the best EOT arm. The
protection is real but roughly half the size, and the headline cannot stand.

**2. It collapses on ALL THREE encoders, not only the held-out one.** WavLM-SV and
ECAPA are still in the crafting ensemble, yet their cloning protection falls by 29
and 44 points. The pre-registered prediction was that only the held-out column would
move.

The mechanism is specific to the cloning threat model and is the more interesting
finding. In ASV the three encoders are three independent verifiers, so removing one
gradient hurts that one. In cloning they are three *readouts of a single event* —
whether DiffVC's conditioning embedding was moved off the victim. That conditioning
network is the Resemblyzer encoder. Once the perturbation stops fooling it, the clone
keeps the victim's identity and **every** verifier accepts it.

**Quantified, because the first version of this paragraph overstated it.** The three
columns are not "one measurement reported three times" — pairwise Spearman
correlation of the margin is 0.39-0.59 with 73-82 % binary agreement (n = 300,
before the pre-filter). Dependent, not redundant. Fix S7 designates ECAPA-TDNN as
the primary encoder, labels the others secondary, prints the correlation table above
every analysis run, and applies no multiple-comparison correction across them.

That also means section 19's per-encoder cloning table is less independent than it
looks, and the same applies to published work: De-AntiFake scores with a Resemblyzer
d-vector while two of its cloning methods (SV2TTS, DiffVC) condition on that same
encoder, and RoVo crafts with SV2TTS's encoder while reporting Resemblyzer as an
evaluation system.

### 20f. What survives

The **ordering across arms is preserved**. Unpurified, held out, the EOT arms still
beat plain PGD on every encoder: `full` 38/58/17 % and `fullnsT` 25/50/17 % against
`plain` 75/58/67 %. Purifier-aware crafting is doing real work that does not depend
on encoder match. It is the *magnitude* of the cloning claim that was inflated, not
the existence of the effect.

Against the matched purifier, held out, protection is mostly gone (40-86 % clone
success). Stated plainly: **an attacker who both purifies with the defender's model
and clones with an encoder the defender did not craft against defeats this
protection.**

### 20g. Caveats that bound the reading

* n = 5-12 per (encoder, purifier) cell after the pre-filter. Individual cells carry
  binomial intervals of +/-20 points or worse. The claim rests on 9 of 9 cells moving
  the same way, not on any single number.
* One cloner, one budget (m20_f20). The m30_f15 budget saturates and was not re-run.
* WavLM-SV and ECAPA remain white-box for ASV. F1 fixes the cloning confound only.

---

## 21. Cross-session thresholds, and the corpus move (S1 + S1b, 2026-09-19)

### 21a. The 0.03 % EER was measuring the room, not the speaker

`per_speaker()` takes the first N utterances per speaker. In LibriSpeech those all
come from ONE chapter: one sitting, one room, one microphone. Every same-speaker
trial therefore shared a recording session, and the encoder was rewarded for
matching the channel as much as the voice.

`cross_session_split()` had been written for exactly this and was **called from
nowhere**. It is now wired into `calibrate_thresholds.py --cross-session`, which
draws the enrolment side from a different chapter than the test side.

Measured on `test-clean`, 29 of 40 speakers (the other 11 have a single chapter and
cannot support the protocol):

| encoder | same-session | cross-session | ratio | threshold |
|---|---|---|---|---|
| WavLM-SV | 3.33 % | **5.20 %** | 2x | 0.911 → 0.893 |
| ECAPA-TDNN | **0.03 %** | **0.62 %** | **21x** | 0.376 → 0.336 |
| Resemblyzer d-vector | 0.09 % | **3.45 %** | **38x** | 0.779 → 0.703 |

**The ECAPA figure is the one to quote.** Section 20's fix document predicted the
real value at roughly 1 %, on the grounds that VocalBridge reports 0.6 % under the
same same-session limitation. The corrected measurement is **0.62 %**. The
prediction was made before the experiment, which is the only kind of prediction
worth anything.

d-vector moving 38x is the larger surprise and is worth a sentence in the paper:
that encoder was the most flattered by same-session calibration, and it is also the
encoder the cloner uses.

### 21b. What this does to the results, predicted in advance

Every threshold moved **down**, so every encoder accepts more readily and the
defender's job gets harder. **Absolute protection rates will fall.** Paired
comparisons should not move, because the threshold cancels in a difference — the
same argument that made fix S2 change the rates while leaving the p-values
identical to four decimal places.

If a paired result DOES move materially once re-run, that is a finding in itself
and must be investigated rather than absorbed.

### 21c. Corpus and purifier

Running at the same time, and required before any other experiment:

* **Evaluation moves to `test-clean`**, which AntiFake and De-AntiFake both use and
  `dev-clean` no comparable paper uses. This also lifts the speaker ceiling: the
  old split had 40 speakers with the purifier consuming 20.
* **Purifier v7 trains on `train-clean-100`**, 60 speakers, disjoint from every
  evaluation speaker. The published purifier was trained on the same 40-speaker
  split it was evaluated on, which De-AntiFake explicitly guards against.

`run_testclean_pipeline.sh` enforces both gates from earlier lessons before any
number is measured against v7: denoising gain >= 0.15 (the v2 lesson, where an
identity-map purifier invalidated a whole table), and channel EER on UNPROTECTED
audio within 10 points of clean (the lesson where a purifier destroyed unprotected
identity and made protection look good for the wrong reason). Both abort the run
rather than warn.

---

## 22. P1.3 — the loud budget does NOT rescue the cloning number (2026-09-19)

### 22a. The question, and why it was asked first

The honest cloning number (section 20) was 38-60 % clone success. Two candidate
explanations called for completely different work:

* **Budget.** The loud budget `m30_f15` already gave 1-5 % clone success, but
  those runs crafted against all three encoders including DiffVC's own. If the
  loud budget stayed near zero with that encoder held out, the weak number was a
  budget artefact.
* **Transfer.** If it degraded anyway, encoder mismatch is the barrier and the fix
  is a larger crafting ensemble, not a louder perturbation.

This run settles it. **It is transfer.**

### 22b. Result

test-clean, 29 speakers, purifier v7 (trained on train-clean-100), cross-session
thresholds, DiffWave at t=3, Resemblyzer held out of crafting. Clone success %
after the AntiFake pre-filter; unprotected is 100 % by construction.

| attacker's purifier | arm | WavLM-SV* | ECAPA* | **d-vector (HELD OUT)** |
|---|---|---|---|---|
| none | plain | 43 % | 38 % | 92 % |
| none | EOT | **14 %** | **12 %** | 88 % |
| none | EOT+constraint | 21 % | 19 % | **75 %** |
| ours | EOT | 50 % | 21 % | 76 % |
| ours | EOT+constraint | 43 % | 21 % | 84 % |
| DiffWave | EOT | 60 % | 25 % | 91 % |
| DiffWave | EOT+constraint | 47 % | 25 % | **77 %** |

\* crafted against. d-vector is held out AND is the encoder DiffVC conditions on.

### 22c. What this establishes

**The loud budget works — on the encoders we craft against.** Unpurified, EOT
drives clone success to 12-14 % on WavLM-SV and ECAPA. That is close to the
near-zero figure the white-box runs reported, so the perturbation is not weak.

**It does essentially nothing for the held-out encoder.** d-vector sits at
75-92 % in every cell. Against the one encoder the cloner actually reads identity
with, protection at the loud budget is barely better than no protection.

**This is a clean, controlled contrast** and it is the most useful thing in this
section: same speakers, same perturbation files, same thresholds, same run. The
only difference between the 12 % column and the 88 % column is whether that
encoder was in the crafting ensemble. Nothing else varies.

So loudness protects what you optimised against and does not generalise. Ensemble
composition, not perturbation energy, is the binding constraint. **Roadmap P1.1
(enlarge the crafting ensemble) is therefore promoted from "highest expected
value" to "the only remaining lever on the cloning claim".**

### 22d. A second finding, smaller but consistent with the paper's thesis

The pass-through constraint helps the **held-out** encoder specifically, and only
under a purifier:

| condition | EOT | EOT+constraint |
|---|---|---|
| no purifier, d-vector | 88 % | 75 % |
| DiffWave, d-vector | 91 % | **77 %** |
| ours (matched), d-vector | 76 % | 84 % |

Better under the foreign purifier, worse under the matched one. That is the same
mismatch signature section 5 reports for ASV, now visible in the cloning setting
where section 20 found nothing. It is 22-25 speakers per cell and not corrected
for multiplicity, so it is a signal to test properly, not a result.

### 22e. Perceptual cost, which must be reported with these numbers

The loud budget is audibly worse: PESQ 2.2-2.3 against 3.44 at the tight budget,
SNR ~15.5 dB against 20.9. If any loud-budget number enters the paper, this table
goes beside it. AntiFake reports MOS 3.37-3.84, so we remain the stricter of the
two on perturbation audibility even here.

---

## 23. P0 — segment-level crafting. A large partial win (2026-09-20)

### 23a. The defect, and the prediction made before the run

Resemblyzer sat at 35-50 % clone success in the DEPLOYMENT setting *while being
crafted against*, where WavLM-SV and ECAPA reached 0 %. `diag_dvec_proxy.py`
traced it: the crafting objective was ~0.13 cosine away from the scoring function.

| comparison | cosine |
|---|---|
| pooled (what we optimised) vs real scoring path | 0.882 |
| pooled vs partial-averaged | **0.872** — the whole gap |
| partial-averaged vs real | 0.977 — preprocessing is NOT the cause |

`embed()` now returns the partial average (agreement 0.882 -> **0.977**), and
`--seg-partials` takes a smooth max over per-partial cosines so the optimiser
pushes the WORST partial. That is AntiFake §C2's weighted segment-based
deviation; the pooled formulation is one their paper explicitly identifies as
hitting local optima.

### 23b. Result. It helped a lot, and it did not fix what it was aimed at

Clone success %, test-clean, 29 speakers, purifier v7, cross-session thresholds,
loud budget, full-resolution crafting.

**DEPLOYMENT (all encoders crafted):**

| purifier | encoder | pooled | segments | change |
|---|---|---|---|---|
| none | WavLM-SV | 6 % | **0 %** | −6 |
| none | ECAPA | 0 % | **0 %** | 0 |
| none | **d-vector** | 35 % | **12 %** | **−23** |
| DiffWave | WavLM-SV | 20 % | **7 %** | −13 |
| DiffWave | ECAPA | 6 % | **0 %** | −6 |
| DiffWave | **d-vector** | 57 % | 68 % | **+11** |

**STRICT (cloner's encoder held out):**

| purifier | encoder | pooled | segments | change |
|---|---|---|---|---|
| DiffWave | WavLM-SV | 58 % | **13 %** | **−44** |
| DiffWave | ECAPA | 29 % | **0 %** | **−29** |
| DiffWave | **d-vector** | 86 % | 91 % | +5 |
| none | **d-vector** | 85 % | **62 %** | −23 |

### 23c. Reading it against the pre-registered criteria

The criteria were written into `analyse_p0.py` before the run.

**"d-vector falls a lot" — only with NO purifier.** −23 points in both settings
unpurified (35→12, 85→62). Under a purifier it does not fall, and under DiffWave
it gets worse (+11, +5). So the proxy gap was a real cause of the unpurified
failure and is **not** the cause of the purified one. P0 is a genuine fix that
did not close the case it was aimed at.

**"WavLM/ECAPA move a lot → investigate."** They did, so we did. Pre-purifier
cosines show the optimiser reallocated: it now attacks the d-vector harder
(+0.661 → +0.530) and the other two slightly less (WavLM +0.220 → +0.297, ECAPA
−0.394 → −0.283). Correcting one encoder's objective rebalanced the ensemble
loss, which is expected and mechanistic, not a bug. Note the direction: WavLM and
ECAPA are *less* fooled before purification yet clone *far less* successfully
after it — pre-purifier cosine is not a proxy for end-to-end protection.

### 23d. Where the chain stands now

The best cells are strong, and two of three encoders now behave the way the
project set out to achieve:

| condition | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| deployment, no purifier | **0 %** | **0 %** | 12 % |
| deployment, foreign purifier | **7 %** | **0 %** | 68 % |
| **strict, foreign purifier** | **13 %** | **0 %** | 91 % |

The third row is the paper's target condition — the attacker purifies with a model
we never trained AND clones with an encoder we never crafted against — and
protection holds on two encoders.

**The d-vector under purification is now the single remaining failure**, and it is
isolated: unpurified it is fixed, purified it is not. That is a narrower and more
tractable problem than "protection does not transfer", which is what it looked
like before P0.

---

## 24. Two negative results that redirected the work (2026-09-20)

Both were cheap pilots run instead of committing to a full experiment, and both
came back NO. Recorded because the reasoning is reusable, and because the
temptation in each case was to skip the check.

### 24a. More optimisation steps does NOT fix the d-vector margin

**Why it looked promising.** After fix P0 the d-vector cleared its accept
threshold by only −0.073 against −0.161 and −0.213 for the other two, and the
crafting log showed its cosine still falling at the final step (0.72, 0.68, 0.65,
0.62, 0.61, 0.59 over steps 100–200). It is also the binding constraint in the
threshold-aware loss, so its budget caps the whole run.

**Pilot: 4 speakers, 200 vs 500 steps, everything else fixed.**

| | 200 | 500 | change |
|---|---|---|---|
| d-vector margin | −0.085 | −0.111 | −0.026 |
| WavLM margin | −0.145 | −0.156 | −0.012 |
| ECAPA margin | −0.252 | −0.245 | **+0.007** |
| PESQ | 2.318 | **2.229** | worse |
| STOI | 0.941 | **0.933** | worse |
| SNR (dB) | 15.72 | **15.14** | louder |

**The gain was bought, not found.** 2.5x the compute for a 0.026 margin
improvement, with every perceptual column degrading. The projection caps loudness
per step, so what is happening is that more steps let the perturbation fill more
of the permitted budget. That is a legitimate mechanism and it is a COST.

**Conclusion.** The d-vector is genuinely hard to move, not budget-limited. Do not
raise `--steps`. The remaining lever is the ensemble.

### 24b. x-vector is a poor CRAFTING encoder, and would have flattered our tables

Added as the first ensemble extension: architecturally distinct (TDNN over
filterbanks) and directly relevant since De-AntiFake scores with an x-vector
system. Calibrated cross-session on test-clean: EER 4.96 %, threshold 0.935.

**Its cosine geometry is compressed, and that is disqualifying here:**

| encoder | threshold | genuine mean | impostor mean | separation |
|---|---|---|---|---|
| ECAPA | 0.336 | 0.623 | 0.077 | **0.546** |
| WavLM-SV | 0.893 | 0.941 | 0.677 | 0.264 |
| d-vector | 0.703 | 0.814 | 0.564 | 0.250 |
| **x-vector** | 0.935 | 0.957 | **0.895** | **0.062** |

x-vectors are meant to be scored with a trained backend (PLDA), not raw cosine.
The whole decision lives in a 0.06-wide band, so there is almost no gradient.

**Measured over 200 steps, 2 speakers, all four crafted:**

| encoder | start | end | moved |
|---|---|---|---|
| ECAPA | 1.00 | −0.16 | 1.16 |
| d-vector | 1.11 | 0.63 | 0.48 |
| WavLM-SV | 1.00 | 0.54 | 0.46 |
| **x-vector** | 1.00 | **0.92** | **0.08** |

**It barely moves and still reports "protected"** (margin −0.012). It would have
contributed a near-perfect column having been barely attacked — and at −0.012 that
column flips on any speaker with slightly different geometry.

**This is the 0.25-threshold error with the sign reversed.** RESULTS §18 records
applying the field's cosine-0.25 convention to encoders whose chance floor is 0.6,
where every clone trivially "succeeded". Here an encoder trivially "fails" to
clone because its threshold sits 0.02 below its genuine mean. Both are scale
mismatches; only the direction of the flattery differs.

All three existing encoders also ended up LESS fooled with x-vector in the loop
(WavLM 0.297→0.389, ECAPA −0.283→−0.140, d-vector 0.530→0.548). n=2, so
indicative only, but consistent with it consuming perceptual budget while
contributing no gradient.

**Decision.** x-vector is retained as an EVALUATION encoder (opt-in
`--with-xvector`, default off) where it is genuinely useful for comparability
with De-AntiFake. It is **not** added to crafting. A real fourth crafting member
needs an encoder scored natively by cosine — CAM++, ERes2Net or a ResNet
speaker model — none of whose libraries are installed here.

### 24c. The general lesson

Three checks this week (the directed-mode hypothesis, the steps pilot, x-vector)
each cost under two hours and each returned NO. The alternative was committing
5–10 GPU-hours per idea on plausible reasoning. Two of the three would also have
produced numbers that looked *better* while meaning less.

---

## 25. The mismatch claim did not survive the rebase, and the rebase was not clean (2026-09-21)

**Status: the paper's central contribution is under investigation. Do not quote
rows 4 and 5 of the headline table until section 26 resolves this.**

### What happened

`run_rebase_mismatch.sh` ran overnight 2026-09-20 to move the mismatch claim onto
the corrected base. Block (c), Holm over 12 cells:

| | old base (n=20) | rebased (n=29) |
|---|---|---|
| significant after Holm | **7 of 12** | **4 of 12** |
| sign of those cells | all negative — constraint **helps** | all positive — constraint **hurts** |
| which encoders | WavLM-SV, ECAPA | d-vector only |

WavLM deltas fell from −0.082, −0.025, −0.091, −0.045 to −0.017, −0.009, −0.019,
−0.011: same sign, four to five times smaller, none surviving correction. The
no-purifier control moved too — WavLM no longer shows the constraint being
significantly harmful there (delta +0.008, p=0.096), so for WavLM the pattern is
no longer "loses unpurified, wins mismatched" but "does very little either way".

This is not a power problem. n went **up**, 20 → 29, and the effect went down.

### Why the rebase cannot settle it: two uncontrolled changes

The run was specified to change four things. It changed six.

1. **The budget.** Every old foreign and control CSV contains only `*_m30_f15*`
   arms — the **loud** budget. The rebase ran `m20_f20`, the tight one. The run
   script justified this as "the loud budget saturates and cannot distinguish
   arms", which is true of the **binary rate** and irrelevant here: this claim has
   always been measured on the **continuous margin**, where saturation does not
   apply. The claim was rebased onto a budget it was never measured at.
2. **`--seg-partials`.** Only `DvecEnc` defines `embed_partials`, so the flag
   changes the d-vector term alone — but that term feeds the shared LSE loss, so
   it reallocates perturbation budget away from WavLM and ECAPA, the two encoders
   that carried the claim. That the d-vector is now the *only* encoder with a
   significant effect, in the opposite direction, is consistent with it having
   become the binding constraint and competing directly with the regulariser.

The budget is the more likely cause of the two. The constraint trades attack
strength for purification survival; at a tight budget there is less to trade.

### A third defect, affecting the rates but not the flip

The foreign-channel thresholds were **stale**. The script's `[ ! -f "$CAL" ]`
guard found the legacy `thresholds_diffwave_t{1,3}[_sc09].csv` — no
`cross_session` or `subset` column, d-vector EER **0.0009**, the exact
same-session optimism fix S1 exists to remove — and reused them instead of
calibrating on the corrected base.

Paired deltas and p-values are **immune**: the threshold cancels in a
within-speaker difference between two arms scored at the same value, which is
what fix S2 demonstrated. So the sign flip is real and not an artefact of this.
But every binary rate in block (c) of that run (`3/29`, `4/29`, …) is wrong and
is not quoted anywhere until recomputed.

### What is running

`run_rebase_mismatch_m30.sh`, launched 2026-09-21 14:06. It changes only the four
intended things: loud budget, pooled crafting, corrected base. It recalibrates the
four foreign channels cross-session on test-clean, then scores **both** budgets
against those same calibrations in one `eval_saved` pass, so the 2×2 is internally
comparable and the m20 rates are corrected at the same time.

Read it with `experiments/analyse_budget_2x2.py`, which prints both a
within-budget (m=12) and an across-budget (m=24) Holm correction, and counts
significant cells **by sign** — a significant cell with a positive delta is
evidence against the claim and was never distinguished from a hit in the old
"7 of 12" phrasing.

### The three outcomes, written before reading the result

* **m30 holds, m20 fails** — the effect is real and **budget-conditional**. The
  paper must say "at the loud budget" and report the tight budget beside it. That
  is a weaker claim than `draft_v5.md` currently makes.
* **Both fail** — the effect was a property of the old evaluation base. Rows 4
  and 5 of the headline table are dropped. The survival and foreign-purifier
  transfer results (§E9) are untouched by this and remain the strongest thing in
  the paper.
* **Both hold** — the m20 result was the `--seg-partials` reallocation, and that
  becomes the thing to investigate.

This is a post-hoc rescue of a failed confirmation and will be labelled as one in
the paper. Running a second budget after the first returned null is a multiplicity
problem, not a neutral act, which is why the m=24 correction is printed alongside.

---

## 26. The mismatch claim survives the rebase at the loud budget (2026-09-22)

**Resolves §25, outcome 1: the effect is real and budget-conditional.** Run:
`run_rebase_mismatch_m30.sh`, launched 2026-09-21 14:06, finished 22:21. Loud
budget (`m30_f15`), pooled crafting, corrected base (test-clean / v7 /
cross-session / n=29 / shared selector), foreign channels **recalibrated
cross-session** and both budgets scored against those same calibrations.

### 26a. Block (c), mismatched purifier — the positive claim

Paired on the decision margin, Holm over 12. Negative = EOT+constraint protects
better than EOT alone.

| cell | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| LibriSpeech DiffWave t=1 | **−0.058** 23/29 p_adj=.003 | **−0.049** 21/29 p_adj=.024 | **+0.074** 0/29 p_adj<.001 |
| LibriSpeech DiffWave t=3 | **−0.036** 24/29 p_adj=.001 | **−0.036** 22/29 p_adj=.003 | +0.012 8/29 p_adj=.057 |
| SC09 DiffWave t=1 | **−0.059** 25/29 p_adj=.001 | −0.041 20/29 p_adj=.057 | **+0.079** 0/29 p_adj<.001 |
| SC09 DiffWave t=3 | **−0.055** 25/29 p_adj<.001 | **−0.039** 23/29 p_adj=.001 | +0.012 10/29 p_adj=.057 |

**6 of 12 cells help after correction, and all 6 survive the across-budget
correction at m=24 too.** WavLM is 4 of 4, ECAPA 3 of 4 with the fourth at
p_adj=0.057. The old base reported 7 of 12; this is a replication with 45 % more
speakers and every base defect fixed.

**Two cells go significantly the wrong way, and they are not noise.** The
d-vector at t=1, on both foreign purifiers, 0 of 29 speakers improved,
+0.074 and +0.079. The constraint makes the d-vector *worse* under mild
purification. The old "7 of 12" phrasing never separated a hit from a
significant result in the opposite direction; the paper must.

### 26b. Block (a), no purifier — the control

| | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| m30, unpurified | −0.009 16/29 p=0.48 (null) | **+0.040** 8/29 p_adj=.023 (worse) | **+0.079** 0/29 (worse) |

The old base had the constraint clearly losing unpurified on *both* strong
encoders (0/20, +0.16, +0.29). On the corrected base that replicates on **ECAPA
only**. WavLM is null unpurified, not worse. So the sign *change* — the thing
that rules out "they just built a sturdier perturbation" — is clean on ECAPA
(+0.040 → −0.036…−0.049) and is a null→help pattern on WavLM. Both are
consistent with "the benefit is specific to purification"; only ECAPA shows the
*cost*. Row 5 of the headline table is reworded accordingly.

### 26c. The tight budget, and the confound I did not remove

At `m20_f20` **nothing helps**: 0 of 12 cells negative and significant; the
d-vector is significantly positive in all 4. The effect needs perturbation
strength to trade.

But the m20 arms were crafted with `--seg-partials` and the m30 arms without, so
these two rows are the **diagonal** of a 2×2, not the 2×2. `analyse_budget_2x2.py`
was named and labelled as if it were, and that was wrong. What §26a establishes
is that corpus / purifier / thresholds / n did not kill the claim. Whether
*segment crafting* does is separated by the run queued next:
`run_rebase_mismatch_m30_seg.sh`, all four arms at m30 with `--seg-partials`.
That is the crafting the paper will use for every cloning number, so the
mismatch claim has to be shown under it or the paper carries two recipes.

### 26d. Perceptual control at the loud budget, n=29

| arm | PESQ | STOI | SNR (dB) |
|---|---|---|---|
| EOT (full) | 2.207 | 0.911 | 15.28 |
| EOT + constraint (fullnsT) | **2.278** | **0.928** | **16.10** |

The constrained arm is **quieter and scores higher** on both perceptual metrics
while protecting better under mismatch. The gain is not bought with loudness.
But note the absolute level: PESQ ≈ 2.2 at this budget against 3.44 at m20. The
loud budget is audibly perturbed and the paper must report it as such; AntiFake's
own MOS sits at 3.37–3.84, so this is roughly the field's operating point, not
below it.

### 26e. What goes in the paper, as of now

* The mismatch trade is stated **at the loud budget** with the tight-budget null
  reported beside it. Not unconditionally.
* Six cells, not seven; and the two d-vector reversals reported in the same table.
* Row 5 becomes "the cost shows on ECAPA; WavLM is null unpurified".
* Pending §26c before any of it is typeset.


---

## 27. §26c resolved: the mismatch claim holds under segment crafting (2026-09-22)

**Outcome "both hold".** `run_rebase_mismatch_m30_seg.sh` (launched 00:47,
finished 14:27; one ~7 h GPU stall on speaker 7021) — the m30 script with exactly
one change, `--seg-partials` on. Same base, same 29 speakers, same seeds, same
recalibrated foreign thresholds. So the m20 null in §25 was the **budget**, not
segment crafting, and the paper ships **one recipe**: m30_f15 + segment crafting.

### 27a. Block (c) under the shipping recipe — ONE Holm family of 24 across both crafting modes

The statistics reviewer's demand: not two families of 12 read separately, one
pre-registered family across both m30 crafting modes. Under m=24:

| cell | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| LibriSpeech t=1 | **−0.076** 25/29 p_adj<.001 | −0.046 19/29 p_adj=.054 | **+0.071** 3/29 p_adj<.001 |
| LibriSpeech t=3 | **−0.026** 22/29 p_adj=.013 | −0.029 19/29 p_adj=.091 | +0.014 9/29 p_adj=.091 |
| SC09 t=1 | **−0.078** 23/29 p_adj<.001 | −0.038 18/29 p_adj=.098 | **+0.075** 3/29 p_adj<.001 |
| SC09 t=3 | **−0.046** 22/29 p_adj<.001 | **−0.034** 22/29 p_adj=.014 | +0.014 9/29 p_adj=.098 |

Segment: **5 help, 2 hurt, 7 significant of 12.** Pooled under the same m=24
family: 6 help, 2 hurt (ECAPA LibriSpeech t=1 drops to p_adj=.054 and SC09 t=1 to
.095 under the harsher correction, so pooled ECAPA is 2/4 at m=24, not 3/4).

**Reading by encoder.**
* **WavLM: robust.** 4/4 at m=24 under both recipes, and the t=1 effects are
  *larger* under segment crafting (−0.076/−0.078 vs −0.058/−0.059 pooled).
* **ECAPA: weakened.** All four cells keep the sign and 18–22/29 speakers, but
  only SC09 t=3 survives m=24; the other three sit at .054–.098. Segment crafting
  reallocates budget toward the d-vector (§23c) and ECAPA pays for it.
* **d-vector: reversed at t=1**, both purifiers, 3/29 improved (+0.071, +0.075).
  Consistent with pooled (0/29). This is a stable feature of the method, not noise.

### 27b. Bootstrap 95 % CIs of the mean paired delta, speaker as the unit (2000 resamples), segment recipe

**These are marginal, per-cell intervals with NO multiplicity control** (added
2026-09-25 after review). Do not read "all 12 exclude zero" as rescuing the
cells Holm did not accept (e.g. ECAPA LibriSpeech t=1: CI [−0.074, −0.016] but
p_adj=.054). If simultaneous intervals are wanted, they must be computed as
such; until then the Holm column is the inferential statement and the CIs are
effect-size context.

"6 of 12" reads as twelve pieces of evidence; it is 29 paired files. The effect
size with its uncertainty is the honest statement:

| cell | encoder | mean Δ | 95 % CI | |
|---|---|---|---|---|
| LibriSpeech t=1 | WavLM-SV | -0.076 | [-0.102, -0.050] | excludes 0 |
| LibriSpeech t=1 | ECAPA-TDNN | -0.046 | [-0.074, -0.016] | excludes 0 |
| LibriSpeech t=1 | d-vector | +0.071 | [+0.052, +0.090] | excludes 0 |
| LibriSpeech t=3 | WavLM-SV | -0.026 | [-0.040, -0.011] | excludes 0 |
| LibriSpeech t=3 | ECAPA-TDNN | -0.029 | [-0.050, -0.008] | excludes 0 |
| LibriSpeech t=3 | d-vector | +0.014 | [+0.002, +0.026] | excludes 0 |
| SC09 t=1 | WavLM-SV | -0.078 | [-0.106, -0.053] | excludes 0 |
| SC09 t=1 | ECAPA-TDNN | -0.038 | [-0.069, -0.009] | excludes 0 |
| SC09 t=1 | d-vector | +0.075 | [+0.057, +0.093] | excludes 0 |
| SC09 t=3 | WavLM-SV | -0.046 | [-0.064, -0.026] | excludes 0 |
| SC09 t=3 | ECAPA-TDNN | -0.034 | [-0.050, -0.017] | excludes 0 |
| SC09 t=3 | d-vector | +0.014 | [+0.001, +0.027] | excludes 0 |

### 27c. Block (a), no purifier, segment recipe

| | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| m30 segment | −0.002 17/29 p=.77 (null) | +0.022 13/29 p=.26 (**null**) | **+0.079** 0/29 (worse) |

Under the shipping recipe the constraint's unpurified *cost* is visible on the
**d-vector only**. ECAPA's cost (+0.040, p_adj=.023 under pooled) is gone. So for
the two strong encoders the pattern is **null unpurified → helps purified**, not
worse → better. That still says the benefit is specific to purification; it no
longer demonstrates a price paid on those encoders. Row 5 is reworded again.

### 27d. What the paper says, final

* Mechanism claim: **at the loud budget, under segment crafting, the constraint
  improves protection against unassumed purifiers on WavLM (4/4 cells, m=24) and
  directionally on ECAPA (4/4 sign, 1/4 significant), and worsens it on the
  d-vector at mild purification (2/4 significant reversals).** Null at the tight
  budget. One Holm family of 24; bootstrap CIs in 27b.
* This is the recipe every cloning number uses. No second recipe anywhere.
* The reversal and the ECAPA weakening are reported in the same table as the
  WavLM hits.

---

## 28. The stochastic-attacker test: DDPM purification does not break the DDIM-crafted perturbation (2026-09-22)

**Why this was run.** The council's security reviewer made a concrete prediction:
we craft with exact gradients through a *deterministic* DDIM reverse process, an
attacker is free to purify with the *stochastic* DDPM sampler using several
seeds, and "WavLM/ECAPA clone success climbs from 7–13 % toward 40–86 %" once
they do. That is the attack they would run, so we ran it first.
`run_matched_ddpm_attacker.sh`: purifier v7 recalibrated as a DDPM channel at
t=50 (cross-session, test-clean; EER WavLM 9.06 %, ECAPA 0.87 %, d-vector
5.43 %), then the two shipping-recipe arms pushed through v7/DDPM t=50 with
**five seeds**, scored at that calibration. Evaluation only; no crafting.

### 28a. Result — the prediction is falsified at the matched purifier

Protection rate (clone fails), same 29 speakers, DDIM row from
`main_table_tc_m30_seg.csv` re-thresholded to its own calibration:

| arm | encoder | DDIM (crafted against) | **DDPM, 5 seeds** | Δ margin | p |
|---|---|---|---|---|---|
| EOT | WavLM-SV | 90 % | **93 %** | +0.002 | .54 |
| EOT | ECAPA | 97 % | **97 %** | +0.008 | .027 |
| EOT | d-vector | 93 % | **93 %** | +0.010 | <.001 |
| EOT + constraint | WavLM-SV | 97 % | **97 %** | +0.001 | .93 |
| EOT + constraint | ECAPA | 97 % | **97 %** | +0.007 | .056 |
| EOT + constraint | d-vector | 69 % | **72 %** | +0.006 | .006 |

The stochastic attacker recovers **+0.001 to +0.010 cosine** more identity than
the deterministic one — statistically detectable on two encoders, practically
nothing, and protection rates move by at most three points, in both directions.
The perturbation does not depend on the sampler's path. This is consistent with
the mechanism: the pass-through subspace is a property of the denoiser's
Jacobian, not of the trajectory the sampler takes through it.

### 28b. What this does and does not settle

* **Settles:** "the defender picked the sampler" is not where the result lives.
  The paper can state that crafting through DDIM and attacking with DDPM gives
  the same protection, with this table.
* **Does not settle:** the attacker's *best-of-seeds* strategy. `eval_saved.py`
  averages cosine over the five seeds (line 159); a patient attacker keeps the
  seed that recovers the most identity. `exp_purifier_aware.py` already records
  `cos_after_max` / `fails_all_seeds` for its own evaluation; `eval_saved.py`
  should gain the same two columns before this table is typeset. Expect the
  best-of-5 rates to be a few points worse than the mean-of-5 above, not tens.
* **Does not settle:** PhonePuRe stage 2, which remains the attacker the
  "reverses De-AntiFake" sentence actually needs.

### 28c. The patient attacker: best of five seeds (added after the re-run, 2026-09-22)

`eval_saved.py` now records `cos_after_max` and `fails_all_seeds`. Protection when
the attacker keeps whichever of the five DDPM seeds recovered the most identity:

| arm | encoder | mean-of-5 | **best-of-5** | Δ cosine (best − mean) |
|---|---|---|---|---|
| EOT | WavLM-SV | 93 % | **90 %** | +0.031 |
| EOT | ECAPA | 97 % | **90 %** | +0.037 |
| EOT | d-vector | 93 % | **83 %** | +0.021 |
| EOT + constraint | WavLM-SV | 97 % | **93 %** | +0.035 |
| EOT + constraint | ECAPA | 97 % | **97 %** | +0.034 |
| EOT + constraint | d-vector | 72 % | **62 %** | +0.020 |

Read the best-of-5 column as the honest stochastic-attacker number; it is the
one the paper quotes for this test. The mean-of-5 column is what 28a reported.

---

## 29. De-AntiFake's full PhonePuRe (stage 1 + 2) recovers the speaker from our protected audio (2026-09-22)

**The council's top demand, run today. The answer is against us, and it has to be
stated first, not last.**

### 29a. Setup

Their code (`external/De-AntiFake`, patched only for Windows: dead `pypapi`
import, CUDA ops routed to the pure-PyTorch reference, `weights_only=False` on
their official checkpoint, `soundfile` I/O), their weights (`purification.pkl`,
byte-identical to our foreign DiffWave; `refinement.ckpt`, 555,946,052 bytes
from HF `cyberrrange/De-AntiFake`), their hyper-parameters
(`test_PhonePuRe.sh`: t=3, sample_step 1, N=30, snr 0.4). Phoneme timings from
MFA `english_us_arpa` alignments of the original utterances (HF
`gilkeyio/librispeech-alignments`), clipped to our 4-s crop, so MFA is bypassed
(`prep_phonepure_inputs.py`). Input: the 29 shipping-recipe files
(`fullnsT_tc_m30_seg`) and the 29 clean crops. Scored with our three encoders
against the clean originals. `phonepure_stage2_seg.csv`.

### 29b. Result, n=29, cosine to the original speaker

| encoder | protected | after **stage 1 only** (§27 files) | after **stage 1+2** | stage 2 adds | p (paired) |
|---|---|---|---|---|---|
| WavLM-SV | 0.296 | 0.840 | **0.900** | +0.604 vs protected | 3.7e-09 |
| ECAPA | −0.283 | 0.271 | **0.451** | +0.734 | 3.7e-09 |
| d-vector | 0.532 | 0.779 | **0.855** | +0.323 | 3.7e-09 |

29 of 29 files move toward the speaker on every encoder. Clean audio through the
same pipeline keeps identity 100 % (cosine 0.96 / 0.67 / 0.94), so this is not
the channel destroying everything — it is the channel restoring exactly what the
perturbation removed.

Provisional ASV acceptance of the purified *protected* audio (clean-channel
thresholds; the PhonePuRe channel is not yet calibrated — fix S2 — so read as
indicative): stage 1 only **21 / 45 / 86 %** → stage 1+2 **55 / 76 / 100 %**.

### 29c. What this kills, and what it leaves

* **Dead as written: "with exact gradients we reverse De-AntiFake's conclusion."**
  We reverse it against their *stage 1*, which is a DiffWave purifier at t=3 (§20,
  §23, §27). Their *stage 2* — the phoneme-guided score-based refiner trained on
  (purified-clean, clean) pairs — undoes most of the protection. The security and
  ML reviewers said precisely this would happen if we only ran half the attacker.
* **Dead: the P2 hypothesis** that our stage-1 survivor would leave stage 2 out of
  distribution. Their clean-pair training assumption holds well enough on our
  perturbation. This must be written as a negative result, not omitted.
* **Still standing, unchanged:** survival and transfer across diffusion
  *purifiers* (§E9, 29/29, p=3.7e-09 rebased); the mechanism claim under one recipe
  (§27); the stochastic-attacker result (§28). None of those were about stage 2.
* **Not yet measured:** DiffVC clone success on the stage-1+2 outputs (the
  ASV-level cosine above is a proxy), and the PhonePuRe-channel calibration.

### 29d. The only adaptive move left, and it is the one their paper concedes

Stage 2 is a differentiable NCSN++ run for 30 reverse steps. Their App. C states
that adaptive attacks "could potentially be adapted for speech tasks" and names
memory as the obstacle, with chunking and accelerated sampling as the fixes. We
already differentiate through a 50-step reverse process with checkpointing. So
the experiment that decides whether an adaptive protection against the *full*
PhonePuRe exists is: EOT through stage 1 **and** stage 2, in the loop, on this
hardware. Cost is the unknown — stage 2 is 54 s forward per file; backward with
checkpointing through 30 NCSN++ steps on 6 GB may not fit at all. That is the
next thing to find out, and until it is found out the paper's attacker claim is
"survives diffusion purification, including De-AntiFake's stage 1; does not
survive its learned refinement stage."

### 29e. End-to-end DiffVC cloning under the full PhonePuRe — better than the ASV proxy suggested (2026-09-22, later)

`exp_cloning_protection.py --purifiers none,diffwave,phonepure` (a `phonepure`
purifier = lookup of the pre-purified file; the clone is scored against the clean
enrolment at the clean cross-session calibration, as in every cloning table).
Shipping-recipe files, 29 speakers, AntiFake pre-filter, n per cell in brackets.
`cloning_tc_m30_seg_phonepure.csv`.

| attacker's purifier | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| none | 6 % (16) | **0 %** (11) | 29 % (24) |
| De-AntiFake stage 1 (DiffWave t=3) | 25 % (16) | **0 %** (9) | 57 % (23) |
| **full PhonePuRe, stage 1 + 2** | **38 %** (16) | **27 %** (11) | **71 %** (24) |

Clone success, lower is better for the defender. So under their *complete*
attacker, DiffVC clones still fail **62 % / 73 % / 29 %** of the time. That is a
real degradation from stage 1 (75 % / 100 % / 43 % fail) and it is not the
collapse the ASV-level acceptance in 29b implied (55 / 76 / 100 % accepted):
DiffVC does not carry all of the identity that stage 2 restores in the reference.

**How to write it.** Two numbers, both honest, side by side: "ASV acceptance of
the purified reference" (29b, the attacker's best case) and "clone success"
(29e, what the victim actually suffers). The paper's attacker claim is: against
De-AntiFake's full pipeline, protection is *reduced*, not removed — WavLM/ECAPA
clones fail roughly two-thirds of the time; the d-vector, DiffVC's own
conditioner, remains the weak column at 71 %. Against their stage 1 alone it is
near-complete on the two modern verifiers.

**Caveats carried forward.** One arm (`fullnsT_tc_m30_seg`); `none` here reads
6/0/29 % where §23d's `fullnsT_m30_seg_all` read 0/0/12 % — same recipe,
different run and pre-filter membership, and the spread between them is the
seed-variance S6 still owes. PhonePuRe-channel calibration for 29b is still to
run; 29e does not need it.

### 29f. Feasibility probe: exact gradients through stage 2 (2026-09-22, 17:54)

`external/probe_stage2_grad.py`: one 4-s file, their Refiner as configured in
`test_PhonePuRe.sh` (N=30, ald corrector, snr 0.4), `torch.no_grad` replaced by
a call-time switch so the sampler builds a graph, full backward with **no
checkpointing**.

* Forward alone is unchanged (54 s). The backward **OOMs at a recorded peak of
  11.3 GB** on the 6 GB card, with the crafting job holding 1.5 GB alongside.
* Read the number the useful way: 30 predictor-corrector steps, so roughly
  0.35 GB of activations per step. **Per-step checkpointing brings this to
  ~1 GB**, i.e. it fits, at the cost of a second forward per step
  (~2 × 54 s per EOT sample per optimisation step). That is the "chunking and
  accelerated sampling" De-AntiFake App. C says would make an adaptive attack
  tractable, and it is a sampler-loop edit, not a research question.
* Cost estimate if built: crafting through stage 1 + stage 2 at 200 steps and
  one EOT sample is ~6 h *per speaker* at 54 s forward — impractical at n=29.
  Reducing N (their own accelerated-sampling suggestion) or the step count is
  mandatory: at N=6 and 50 optimisation steps it is ~20 min per speaker, ~10 h
  for 29. So the experiment is feasible on this card only as a reduced-N
  adaptive attack, and must be labelled as such.

**Decision deferred to the user:** build the checkpointed, reduced-N EOT through
stage 1+2 (≈ 1 day code + ≈ 10 h GPU, uncertain outcome), or reframe now around
purifier transfer and the mechanism with §29 as the stated limit.

---

## 30. BPDA vs exact gradients: the estimator is NOT the delta (2026-09-22, evening)

**The council's ML reviewer asked for exactly this control, and it removes a
claimed novelty.** `run_bpda_headtohead.sh`: the shipping recipe with ONE change,
`--grad-estimator bpda` (real purifier forward, identity backward — the estimator
De-AntiFake §5.4 and VocalBridge §V-E used). Same base, budget, seg crafting,
seeds, EOT samples, steps, thresholds.

### 30a. Constrained arm (EOT + constraint), BPDA vs exact, paired by speaker, n=29

| condition | encoder | exact prot. | BPDA prot. | Δ margin (exact − BPDA) |
|---|---|---|---|---|
| matched v7 t=50 | WavLM / ECAPA / d-vec | 97 / 97 / 69 % | 93 / 93 / 69 % | −0.002 / −0.001 / −0.004 |
| matched v7 t=200 | | 93 / 97 / 66 % | 93 / 93 / 62 % | −0.003 / **−0.009** / −0.003 |
| DiffWave-Libri t=1 | | 100 / 97 / 62 % | 97 / 93 / 62 % | −0.012 / −0.007 / −0.002 |
| DiffWave-Libri t=3 | | 79 / 55 / 14 % | 83 / 55 / 17 % | −0.002 / +0.001 / +0.000 |
| DiffWave-SC09 t=1 | | 97 / 97 / 72 % | 97 / 93 / 76 % | −0.014 / −0.007 / −0.003 |
| DiffWave-SC09 t=3 | | 86 / 66 / 34 % | 86 / 62 / 34 % | −0.005 / −0.003 / −0.001 |
| no purifier | | 97 / 100 / 97 % | 93 / 100 / 93 % | +0.000 / −0.014 / +0.000 |

Holm over 21 cells: **1 significant** (matched ECAPA t=200, Δ = −0.009), zero in
the other direction. Protection rates agree within four points everywhere.
Perceptual cost identical (PESQ 2.283 vs 2.285, STOI 0.929 vs 0.929, SNR 16.11
vs 16.06 dB). **BPDA crafts in 139 s per speaker against 259 s for exact
gradients** — half the cost for the same protection.

### 30b. What this changes

* **Dead:** "our delta over the published negative results is the gradient
  estimator." It is not. On our purifier, BPDA and exact gradients produce the
  same protection. Whatever made De-AntiFake's and VocalBridge's adaptive
  attempts fail, it was not treating the purifier as the identity in the
  backward pass. Candidates, none yet tested: their attacker was the *full*
  pipeline (§29 shows stage 2 beats us too), their budget/step count, or their
  crafting objective. The paper must say this plainly; it is a cleaner claim
  than the one it replaces, and it is testable by others.
* **What the contribution actually is**, now stated without the estimator:
  the crafting recipe (purifier-in-the-loop EOT with a threshold-aware ensemble
  loss, segment crafting, at a stated budget), the purification-invariant
  constraint and its budget-conditional trade (§27), and transfer across
  purifiers (§E9). None of those depend on how the gradient is estimated.
* **Practical upshot for the paper's own runs:** BPDA is the default estimator
  going forward — same numbers, half the GPU time — with the exact-gradient
  equivalence reported once as this table.

### 30c. Process note — a self-inflicted crash

The `full` BPDA arm died at speaker 27 with a CUDA OOM at 17:54:21, the minute
`probe_stage2_grad.py` (§29f) allocated the entire GPU. That was my mistake:
running a memory probe beside a crafting job. The `fullnsT` arm completed
(n=29) and is what 30a reports. The `full` arm has been relaunched (script
re-crafts only the missing arm; partial evaluation files set aside as
`*_partial26.csv`); the `full` head-to-head and the BPDA-side mismatch effect
(fullnsT vs full under BPDA) follow when it lands. Rule added to memory: no
GPU probes while a crafting arm is running.

### 30d. EOT arm (no constraint), BPDA vs exact, n=29 — relaunched after the crash, clean

Holm over 21 cells: **2 significant, both on the matched purifier at t=200**
(WavLM Δ −0.025, ECAPA Δ −0.033, exact better), **none** on either foreign
purifier or the no-purifier control. Protection rates within 4 points in 19 of
21 cells; the largest gap is DiffWave-Libri t=1 WavLM (97 vs 86 %, p=.25, n.s.).
Perceptual identical (PESQ 2.22 vs 2.28, STOI 0.911 vs 0.911, SNR 15.27 vs
15.20 dB). Craft time **255 s vs 122 s** per speaker.

So exact gradients buy a small advantage **only against the purifier they were
computed through, at its strongest setting**, and none of it transfers. That is
exactly what a gradient through one specific model should buy, and it is not
what any published negative result was about. 30b stands.

### 30e. The mechanism claim replicates under a THIRD crafting condition

The mismatch trade (fullnsT vs full), BPDA-crafted arms, Holm m=12:

| | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| significant, helps | **4 / 4** (Δ −0.024 to −0.078, 20–25/29) | 1 / 4 (three at p_adj .057–.073, same sign, 18–23/29) | 0 |
| significant, hurts | 0 | 0 | **2 / 4** (t=1: +0.067, +0.074; 2–3/29) |

7 of 12 significant, 5 help, 2 hurt — the same shape as pooled/exact (§26) and
segment/exact (§27). No-purifier control: ECAPA worse (+0.052, p_adj=.013),
d-vector worse (0/29), WavLM null. **Three crafting conditions, one pattern —
but one sample.** The statistics reviewer's correction (2026-09-25) is right and
is adopted here: pooled/exact, segment/exact and segment/BPDA are three
analyses of the SAME 29 speakers, base, seeds and thresholds. Their agreement
shows the result is insensitive to the crafting/estimator choice; it is NOT
three independent replications and the paper must not call it that. Independent
replication needs different speakers or a second corpus.
The claim "the purification-invariant constraint improves protection against
unassumed purifiers on WavLM, directionally on ECAPA, and worsens it on the
d-vector at mild purification, at the loud budget" is now replicated across
pooled vs segment crafting and exact vs BPDA gradients, always on the corrected
base at n=29.

---

## 31. Fix S6 — seed variance of the shipping recipe (2026-10-01)

**The council's single owed analysis.** `run_s6_seeds.sh`: the two EOT arms of
the shipping recipe re-crafted at crafting seeds 7 and 99 (exact gradients, to
measure the spread of the files the paper ships), beside the existing seed-42
files. A `--craft-seed` flag was added for this: every run before 2026-09-30
used craft()'s default of 42, and a one-speaker dry run confirmed the flag
changes the perturbation before the 13-h run was committed. No stalls; GPU
throttled to ~457 MHz for most of it (14:24 -> 03:27).

### 31a. The optimisation is stable; the cloning rates are not

Mean cosine after a FOREIGN purifier, fullnsT arm, seeds 42 / 7 / 99
(from the evaluator's own summaries, `protect_foreign_seg_*` and `protect_foreign_s6_*`):

| cell | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| DiffWave-Libri t=1 | 0.696 / 0.693 / 0.690 | 0.103 / 0.098 / 0.105 | 0.630 / 0.625 / 0.632 |
| DiffWave-SC09 t=3 | 0.797 / 0.792 / 0.797 | 0.215 / 0.212 / 0.214 | 0.741 / 0.735 / 0.742

---

## 31. S6 — seed variance of the shipping recipe (2026-10-01)

The council's "single analysis still owed". `run_s6_seeds.sh`: the two EOT arms
of the shipping recipe re-crafted at crafting seeds **7** and **99** (exact
gradients, `--craft-seed`, added for this; every earlier run used craft()'s
default 42), then the four foreign cells, the no-purifier control, and DiffVC
cloning. Launched 14:24, finished 03:27, all arms exit 0, no stalls. A guard ran
first: seed 7 vs 42 on one speaker had to produce different perturbations or the
script refused to launch.

### 31a. Continuous quantities are essentially seed-invariant

Mean `cos_after_purify`, DiffWave-LibriSpeech t=1:

| arm | seed 42 | seed 7 | seed 99 | spread |
|---|---|---|---|---|
| EOT (ECAPA / d-vec / WavLM) | 0.149 / 0.559 / 0.772 | 0.150 / 0.565 / 0.772 | 0.151 / 0.568 / 0.768 | ≤0.009 |
| EOT+constraint | 0.103 / 0.630 / 0.696 | 0.098 / 0.625 / 0.693 | 0.105 / 0.632 / 0.690 | ≤0.007 |

Same picture at SC09 t=3 (≤0.007) and in `mel_psr`/`dircos` at libri t=3 and
SC09 t=1 (≤0.001). **The mismatch deltas the mechanism claim rests on (−0.026 to
−0.078, §27a) are 4–10× larger than this.** Seed noise does not explain them.

### 31b. Binary rates wobble by a few points; small-n cloning cells wobble a lot

Protection rate (clone fails), EOT+constraint:

| cell | encoder | 42 | 7 | 99 |
|---|---|---|---|---|
| Libri t=1 | WavLM / ECAPA / d-vec | 100 / 97 / 62 % | 100 / 90 / 62 % | 100 / 97 / 62 % |
| SC09 t=3 | | 86 / 66 / 34 % | 86 / 69 / 41 % | 86 / 66 / 38 % |

DiffVC clone success (pre-filtered, n = 8–26 per cell):

| purifier | encoder | 42 | 7 | 99 | **range** |
|---|---|---|---|---|---|
| none | WavLM / ECAPA / d-vec | 6 / 0 / 29 % | 0 / 0 / 19 % | 0 / 0 / 15 % | 6 / 0 / **14** |
| DiffWave t=3 | | 25 / 0 / 57 % | 23 / **33** / 61 % | 21 / 0 / 61 % | 4 / **33** / 4 |

**ECAPA under the foreign purifier swings 0 → 33 → 0 %** on a pre-filtered
denominator of ~9 speakers: one or two files. That is the §29e anomaly explained
— it is seed/denominator noise, not a real difference between runs.

### 31c. Consequences for the paper

* **Continuous, paired statistics (§26, §27, §28, §30) stand as reported.** Their
  effect sizes exceed seed variance by a wide margin.
* **No single-seed binary cloning rate may be quoted as a point estimate.** Every
  cloning table must either pool the three seeds or print the range. The honest
  deployment line becomes "d-vector 15–29 % unpurified, 57–61 % under the foreign
  purifier", not "29 %" and "57 %".
* **Cells with n < 12 should not be reported alone at all.** ECAPA's pre-filtered
  denominator is the problem, not ECAPA.
* Protection rates carry roughly ±3–7 points of seed noise; differences smaller
  than that between arms are not interpretable from one seed.

---

## 32. A2 — encoder geometry explains the d-vector reversal (2026-10-01)

Two measurements on the shipping recipe, 29 speakers, purifier v7 (DDIM).
Logs: `a2_directional_survival_full.log`, `a2_encoder_subspace.log`;
data: `encoder_subspace.csv`.

### 32a. The crafted perturbation's encoder-relevant component survives for ALL encoders

`diag_directional_survival.py`, keep_e = share of the crafted δ's component along
encoder e's direction that survives purification:

| t | WavLM-SV | ECAPA | d-vector | global α |
|---|---|---|---|---|
| 50 | 1.004 | 0.988 | 0.975 | 0.955 |
| 200 | 0.971 | 0.913 | 0.882 | 0.945 |

Pre-registered reading: all similar. The purifier does not absorb "the part of δ
the d-vector reads". Matches §24.

### 32b. But each encoder's OWN identity direction survives differently — and the d-vector's survives worst

`exp_encoder_subspace.py`: α of the raw direction that moves each encoder most
(no crafting involved), through the same purifier. Reference on this purifier:
optimised pass-through 0.867, random 0.713.

| direction | α t=50 | nsr t=50 | α t=200 | nsr t=200 |
|---|---|---|---|---|
| WavLM-SV | **0.573** | 0.63 | 0.375 | 0.99 |
| ECAPA | 0.490 | 0.79 | 0.320 | 1.16 |
| x-vector | 0.434 | 0.92 | 0.297 | 1.26 |
| d-vector | **0.406** | **1.01** | 0.280 | 1.39 |
| random | 0.713 | 0.54 | 0.463 | 0.76 |

Paired across speakers at t=50: WavLM > d-vector +0.166, 24/29, p=6.5e-06;
ECAPA > d-vector +0.084, 21/29, p=.011; ECAPA > x-vector +0.056, 19/29, p=.013;
WavLM > ECAPA +0.083, 25/29, p=3.6e-05; random > WavLM +0.140, 27/29, p=2e-07.

### 32c. Reading, against the pre-registered criteria

* **Every encoder's raw identity direction is purifier-hostile** — all below
  random. Directions that move a speaker encoder are exactly what a speech
  denoiser removes. This is why plain PGD dies under purification and why a
  survival constraint is needed at all.
* **The ordering is the hypothesised one, with p-values:** WavLM > ECAPA >
  x-vector > d-vector. The d-vector's direction has nsr > 1 — more noise than
  signal comes back. So for the d-vector, "move the encoder" and "survive the
  purifier" conflict most; the constraint buys survival by discarding the most
  attack strength there. That is the §27 reversal, predicted rather than
  observed. And it predicts the §27 strength ordering too: WavLM helps most
  (4/4), ECAPA less (1/4), d-vector reverses.
* **Retro-validation passes.** x-vector scores second-lowest *without being
  told* it failed as a crafting member (§24b). The criterion predicts the one
  case we already knew.
* 32a and 32b are consistent: after the constraint, what is left of δ along
  g_e survives for everyone (32a) — but reaching that state cost the d-vector
  far more (32b).

### 32d. What this buys

* **A mechanism for the whole encoder pattern** (§27, §30e, §24) from one
  measurement that involves no crafting.
* **A selection criterion for A1:** compute α(g_e) for each candidate encoder
  (CAM++, ERes2Net) on the purifier BEFORE crafting; add the one with the
  highest α. "Add more encoders" becomes "add the encoder whose identity lives
  in the invariant subspace".
* **Caveat:** measured on our mel-domain purifier; the significant reversal is
  on the foreign waveform purifier (DiffWave t=1). Same measurement through
  DiffWave needs a waveform-domain survival adapter — the natural next check,
  and purifier transfer (§E9) makes the same ordering likely.

---

## 33. Purifier transfer on the corrected base (2026-10-05)

The lead claim, finally measured on the base the paper actually reports.
`foreign_tc_libri_t{1,3}.csv`, `foreign_tc_sc09_t{1,3}.csv`,
`survival_tc_v7.csv`; all numbers regenerable with
`python experiments/analyse_transfer.py`.

Until now this result was quoted at n = 20 on the dev-clean base with purifier
v3 and same-session thresholds, while the corrected-base files sat on disk
unanalysed (council #2, "do first"). Both table generators also read the old
files, and their loader returned `None` for a missing one, so a repointed table
would have vanished silently instead of complaining. That is fixed: reading an
old-base file is now a hard error naming its replacement.

### 33a. A perturbation aimed at the pass-through subspace survives purifiers we did not train

α = signal gain of the direction through the foreign purifier (1.0 = returned
untouched); `random` is a white-noise direction of identical norm.

| Foreign purifier | t | α pass | α random | separates | retention | 95 % CI |
|---|---|---|---|---|---|---|
| DiffWave-LibriSpeech | 1 | 0.822 | 0.356 | 29/29 | 89 % | [81, 98] |
| DiffWave-LibriSpeech | 3 | 0.699 | 0.212 | 29/29 | 128 % | [111, 149] |
| DiffWave-SC09 | 1 | 0.838 | 0.359 | 29/29 | 91 % | [82, 100] |
| DiffWave-SC09 | 3 | 0.694 | 0.214 | 29/29 | 126 % | [110, 144] |

Both purifiers are independently published DiffWave checkpoints, neither
trained by us, one on LibriSpeech and one on SC09 — a different corpus and a
different task.

`retention` divides out the mel + Griffin-Lim round trip, which on its own
favours the smoother pass-through direction by **1.35×** (the old tables
hard-coded 1.33× from the dev-clean base), and expresses the foreign excess as
a fraction of our own purifier's 1.92×.

### 33b. What "128 % retention" does and does not mean

The two t = 3 cells exceed 100 % with intervals that exclude it, so the effect
is real rather than noise. **It is not amplification.** α pass itself *falls*
from 0.822 to 0.699 as the foreign purifier gets stronger: the perturbation
survives t = 3 *less* well in absolute terms. Retention rises because its
denominator falls faster — the random direction collapses from 0.356 to 0.212.
Retention is a ratio of separations, so it can exceed 1 while absolute survival
drops, and the only honest reading is:

> the *advantage* of pointing into the pass-through subspace over pointing
> anywhere is larger against stronger foreign purification, while the absolute
> amount that survives is smaller.

Do not write "survives better at t = 3", and do not write "exceeds 100 %" for
SC09 t = 1, whose interval [82, 100] includes it.

### 33c. The p-value here carries no information, and is not quoted as strength

All four cells separate on **every** speaker. A Wilcoxon or sign test then sits
at its floor, 2/2²⁹ = 3.7 × 10⁻⁹, which is a restatement of "29 of 29" and not
a measure of effect size. The old headline quoted p = 1.9 × 10⁻⁶, the same floor
at n = 20 (council #8). **Report the count.**

### 33d. Survival through our own purifier, corrected base

`survival_tc_v7.csv`, n = 29, purifier v7:

| direction | α t=50 | nsr t=50 | α t=200 | nsr t=200 |
|---|---|---|---|---|
| pass-through (optimised) | 0.867 | 0.22 | 0.525 | 0.53 |
| pass-through, noise-matched t=200 | 0.989 | 0.07 | 0.976 | 0.10 |
| random (white noise) | 0.713 | 0.54 | 0.466 | 0.77 |
| range (worst case) | 0.185 | 8.79 | 0.172 | 7.20 |

Pass-through beats white-noise random on 29/29 speakers at t = 50. The draft's
old figures for this table were 0.97 vs 0.49; on the corrected base the contrast
is 0.867 vs 0.713, a ratio of 1.22× rather than the 1.98× the old numbers imply.
**The corrected base weakens this contrast substantially, and the paper must
quote 0.867 / 0.713.**

### 33e. The caveat that outranks everything above

`random` here is **isotropic white noise**, and purifier v7 is a denoiser
trained to remove exactly that. So "the pass-through direction survives better
than random" may be no more than "it is smoother than white noise", especially
once §33d shows the margin is 1.22× and not 2×. This is council #3, and it is
the same defect that governs §32's per-encoder α ordering.

The matched controls now exist (`core/matched_random.py`: an envelope-matched
random direction and a clean-PCA-span random direction, both norm-matched) and
the §32 measurement is being re-run against them. **Until that lands, every
number in §33a and §33d is reported against a baseline that flatters it, and
no claim of the form "survives because of where it points" is settled.**
`exp_nullspace_survival.py` now takes the controls too; this table is owed a
re-run.

---

## 34. The empirical noise floor for a crafting-loop edit — and a correction to PLAN_V3 (2026-10-05)

PLAN_V3 2b's idea is right and nobody had used it: BPDA-vs-exact (§30) is a
*measured change to the crafting loop with no real effect*, on the same base,
speakers, seeds, budget and thresholds. Its paired per-speaker spread is an
**empirical null** — what a gate sees move when nothing has changed — and every
future gate should state itself as a multiple of it.

Regenerable with `python experiments/analyse_bpda_null.py`
(`main_table_tc_m30_seg.csv` vs `main_table_tc_m30_seg_bpda.csv`, 522 paired
rows, 29 speakers, 2 arms × 3 encoders × 3 `eval_t`).

### 34a. The floor is a table, not a constant

Paired per-speaker SD of Δmargin, all 18 cells:

| arm | encoder | SD t=50 | SD t=100 | SD t=200 |
|---|---|---|---|---|
| `full` (unconstrained) | WavLM-SV | 0.0433 | 0.0354 | 0.0390 |
| `full` | ECAPA-TDNN | 0.0546 | **0.0557** | 0.0435 |
| `full` | d-vector | 0.0251 | 0.0268 | 0.0198 |
| `fullnsT` (shipping) | WavLM-SV | 0.0254 | 0.0243 | 0.0202 |
| `fullnsT` | ECAPA-TDNN | 0.0202 | 0.0174 | 0.0134 |
| `fullnsT` | d-vector | **0.0096** | 0.0091 | **0.0086** |

Across the 18 cells: **min 0.0086, median 0.0247, max 0.0557.**

### 34b. PLAN_V3 quotes the smallest cell as if it were the floor

PLAN_V3 2b, and the mandatory gate wording in PLAN_V3 §4
("crafting-change component 0.0096 from the BPDA null"), quote **0.0096**. That
is the `fullnsT` / d-vector / `t=50` cell — the second-smallest of eighteen, and
the smallest arm×encoder combination there is.

**A gate calibrated on 0.0096 and then read in any other cell is calibrated
2.6× too tight at the median cell and 5.8× too tight at the worst**, and will
fire on noise. The constrained arm is quieter than the unconstrained one by a
factor of ~2–3 across the board, and the d-vector is the quietest encoder by
another factor of ~2, so 0.0096 is the product of two separate best cases.

PLAN_V3 2b's companion figure for the worst-encoder margin, **0.0102, does not
reproduce** under any definition tried. The worst-encoder-per-speaker margin has
SD **0.0192** on the shipping arm and **0.0429** unconstrained.

### 34c. The rule, corrected

> A gate takes its noise floor **from the cell it names**, using the table in
> §34a — never the minimum over cells. This is the same discipline PLAN_V3 §4
> already imposes on the endpoint ("name the cell, column, arm and estimator");
> it has to apply to the denominator too, or the gate is calibrated against a
> different cell than the one it tests.
>
> When a gate also spans crafting seeds, combine with the seed component
> (§31, ≈0.0127) in quadrature.

Worked example, the cell PLAN_V3 §4's template names (`fullnsT`, d-vector,
DiffWave t=1 — read here at the nearest measured `eval_t`=50): crafting-edit
component 0.0096, seed component 0.0127, combined **0.0159** per speaker. At
n=6 that is SE 0.0065, which is what §4 states — so §4's arithmetic is right
*for that one cell* and must not be carried to another.

### 34d. Why this is worth a section

Two of the three numbers in PLAN_V3's own noise-floor prescription were wrong in
the conservative-looking direction: they make gates easier to pass. The
plan-level lesson is the §25 lesson again — a statistic assembled from the most
favourable available row stops being the statistic it is named after. The fix is
mechanical: the floor is regenerated by a committed script, per cell, and the
gate quotes the cell.

---

## 35. The LSE aggregates incommensurable margins (2026-10-05)

PLAN_V3 2e, verified. `lse` in `exp_purifier_aware.py:328` smooth-maxes
`(c − thr)` across encoders:

```
logsumexp((c - thr_t) * w * tau_lse, dim=0) / tau_lse
```

Those are **raw cosine** margins. Each encoder has its own score spread, so
"whichever encoder is closest to its threshold" is being decided in an arbitrary
unit. Per-encoder impostor σ on the shipping channel
(`diffpure_gl_ddim_v7_xsess_tc`, t=50, via `utils.thresholds.encoder_scales`):

| encoder | impostor σ |
|---|---|
| ECAPA-TDNN | 0.0995 |
| WavLM-SV | 0.0896 |
| Resemblyzer d-vector | 0.0652 |

A **1.53× spread**. A d-vector margin of 0.03 is 0.46 σ; the same 0.03 on ECAPA
is 0.30 σ. The LSE treats them as equal.

### 35a. Which encoder binds is partly an artefact of scale

Binding encoder = the one with the largest margin (closest to acceptance), per
speaker, `eval_t=50`, raw cosine vs margins divided by that encoder's impostor σ:

| arm | binding encoder changes | raw counts | normalised counts |
|---|---|---|---|
| `fullnsT` (shipping) | **4 / 29** | d-vec 20, WavLM 6, ECAPA 3 | d-vec 18, WavLM 7, ECAPA 4 |
| `full` (unconstrained) | **8 / 29** | d-vec 17, WavLM 10, ECAPA 2 | **WavLM 16**, d-vec 9, ECAPA 4 |

The shipping-arm figure is exactly the 4/29 PLAN_V3 2e predicted. The
unconstrained arm is worse than predicted and qualitatively so: **in raw cosine
the d-vector appears to be the binding encoder for 17 of 29 speakers, but in
commensurable units WavLM binds for 16 of 29.** Which encoder the optimiser
concentrates its budget on was being decided by score scale rather than by
proximity to threshold.

### 35b. Consequence for `tau_lse`

Raising `tau_lse` sharpens the smooth-max toward the single binding encoder.
§35a says that encoder is partly chosen by an arbitrary unit, so sharpening
concentrates the budget on "whichever encoder happens to have the widest score
spread" — which is why PLAN_V3 §5 cuts `tau_lse` as a shipping change and keeps
it as an ablation only. §23c/§27a already ran this experiment by accident:
segment crafting reallocated budget toward the d-vector and ECAPA fell from 3/4
to 1/4 significant cells.

### 35c. Status

`utils.thresholds.encoder_scales` now returns the per-encoder divisor and says
whether it is `measured` (from the `nontarget_sd` column added by 2f) or
`inferred-from-eer` (the equal-variance Gaussian fallback, which is what the
project did implicitly all along). **Every threshold file on disk predates 2f,
so every scale above is `inferred-from-eer`**; the measured path activates on
the next calibration run.

The two-line change to `lse` itself is **not** applied to the shipping recipe.
It would alter v1, and PLAN_V3 freezes the base. It ships as an
implementation-notes disclosure beside the selection note in §36, and as an
ablation if there is GPU left.

---

## 36. The shipped iterate is selected on a single purification draw (2026-10-05)

PLAN_V3 2d, confirmed in the source and in the data. This is a defect in our own
method that works **in the paper's favour**, and it is a transferable
methodological point for any purifier-in-the-loop paper.

### 36a. What the code does

`exp_purifier_aware.py`, crafting loop:

```
score = score_acc                 # mean of c_after.max() over n_eot draws
...
if score < best_score:
    best_score, best_delta = score, delta.detach().clone()
```

`c_after` comes from `diffpure_grad`, which draws **fresh noise on every call**.
So `score` is a stochastic estimate of the iterate's quality, and `best_delta`
is the arg-min of that estimate over all optimisation steps.

Two facts make this a winner's curse rather than a rounding detail:

* **`n_eot` is 1.** Confirmed in the data, not just the default:
  `main_table_tc_m30_seg.csv` and its BPDA counterpart both carry `n_eot = 1`
  for every row. So each iterate is scored on **one** purification draw.
* **`steps` is 200.** The shipped delta is the luckiest of ~200 single draws.

The selection statistic is also `c_after.max()` — a **raw cosine**, not a
threshold margin (PLAN_V3 2.1/2.2) — and §35 shows raw cosines are not
commensurable across encoders, so the max is taken in an arbitrary unit too.

### 36b. The asymmetry, and why it is conservative for us

The defender selects its shipped iterate on **1** draw. The attacker, in §28c,
selects its best attempt from **5** draws, which was measured to be worth
**+0.020 to +0.037** of recovered cosine. So the two sides of the reported
contest are not optimised with equal care, and the inequality runs against us:

> Protection as reported is a **lower bound for this recipe**.

### 36c. The disclosure, for the paper

> Ship-iterate selection used raw cosine rather than threshold margin
> (`exp_purifier_aware.py:393`, `:412`) and selected on a single purification
> draw (`n_eot = 1`) over 200 optimisation steps, while the attacker in §28c
> selects best-of-5. Reported protection is therefore a lower bound for this
> recipe.

This pre-empts the objection and makes C3.1's coverage contrast **conservative
in our favour**: a reviewer cannot argue the 75–92 % held-out column is an
artefact of under-optimisation once we have said the optimiser was handicapped.

### 36d. Not fixed, deliberately

Three fixes exist (select on a running mean; ship the final delta; re-score the
top-k at `n_eot = 5`). **None is applied.** Any of them changes v1, and PLAN_V3
freezes the base: *the base never moves*. They are listed here, ship as the
§36c paragraph, and are candidates for the W6 `--n-eot 5` parity ablation
(PLAN_V3 #9) if GPU remains.

Note what the fix would *not* do: switching the selection statistic from cosine
to margin does not address selecting on noise. Those are two independent
defects that happen to live on adjacent lines.

---

## 37. Council #3 — C1 survives the matched controls (2026-10-05)

**The C1 gate. It clears.** Pre-registered in `core/matched_random.py` and
hashed into every output row (`prereg_sha=ac0c2ba6c97e`, `code_sha=464986d262ce`);
29 speakers, purifier v7 DDIM, `encoder_subspace_matched.csv`;
`python experiments/analyse_matched_control.py`.

### 37a. Why the gate existed

Every α in §32b was read against **isotropic white noise**, and purifier v7 is a
denoiser trained to remove exactly that (council #3, R1). So
"encoder identity directions live outside the purifier's invariant subspace"
could have been nothing more than "they are not smooth", especially once §33d
showed the pass-through margin is 1.22× and not the 2× the old draft implied.

Two norm-matched controls, one per alternative explanation:

* **`band_matched:<enc>`** — white noise carrying *that encoder direction's own*
  per-mel-band power profile. Matched on spectral envelope, random in direction.
* **`pca_span`** — every frame inside the top-20 principal components of clean
  log-mel frames (95.5 % of clean variance), basis built from **dev-clean**,
  disjoint from the test-clean speakers it controls for.

### 37b. The result: the spectral envelope explains none of the gap

α at t=50, n=29, and the paired difference against each encoder's own control:

| direction | α t=50 | α t=200 | Δ vs band-matched (t=50) | 95 % CI | below |
|---|---|---|---|---|---|
| WavLM-SV | 0.573 | 0.375 | **−0.134** | [−0.160, −0.100] | 26/29 |
| ECAPA-TDNN | 0.490 | 0.320 | **−0.225** | [−0.264, −0.184] | 28/29 |
| x-vector | 0.434 | 0.297 | **−0.273** | [−0.314, −0.230] | 29/29 |
| d-vector | 0.406 | 0.280 | **−0.295** | [−0.353, −0.238] | 27/29 |
| `band_matched` (mean of the four) | 0.707 | 0.465 | — | — | — |
| `pca_span` | **0.843** | 0.617 | — | — | — |
| `random` (white noise) | 0.713 | 0.463 | — | — | — |

All four encoders are **CONFIRMED** against both controls at both timesteps — 8
of 8 cells, every CI excluding zero, Holm-adjusted p ≤ 2 × 10⁻⁶. The gate's
calibration is on record: the CI alone false-confirms at 0.035 under a global
null at this n, the implemented conjunction at 0.030 family-wise and 0.007 per
encoder.

**The decisive number is in the control row, not the encoder rows.** The
envelope-matched controls land at **0.701–0.715**, statistically
indistinguishable from white noise at **0.713**. Giving a random direction an
encoder's exact spectral envelope buys it *nothing*. So the encoder-vs-random
gap is **not** a spectral-smoothness artefact, which is precisely what council
#3 asked.

### 37c. The ordering survives, which is what C1 actually claims

C1 is an *ordering* claim, not a sign claim. Ranking by raw α and by α minus
each encoder's own matched control gives the **same order**:

> WavLM-SV > ECAPA-TDNN > x-vector > d-vector

identical to §32b. The per-encoder differences are therefore not an artefact of
per-encoder spectral envelope either. **C1 holds as written** — the first of the
three pre-registered branches.

### 37d. What the controls newly teach: the invariant subspace is speech-shaped

`pca_span` survives at **0.843**, far above white noise (0.713) and above every
encoder direction. So "smooth, speech-like directions survive better" is a
**real effect** — it just does not explain the encoder ordering. The purifier's
invariant subspace is approximately *the subspace clean speech occupies*, and
every speaker encoder's identity direction points substantially out of it.

That sharpens C1's mechanism from "encoder directions are purifier-hostile" to:

> A speech denoiser preserves the subspace clean speech lives in. Speaker
> identity, as every encoder tested reads it, is largely **not** in that
> subspace — which is why protection and survival conflict, and why they
> conflict most for the d-vector.

**An open question this raises, flagged not answered:** §33d puts the optimised
pass-through direction at α = 0.867 and `pca_span` here reads 0.843. If those
are comparable, a *free* random direction in the clean-speech subspace survives
almost as well as a 300-step optimised one, and the optimisation buys ≈0.02.
**They are not yet comparable** — see §37e — so this is not claimed. Settling it
needs the `exp_nullspace_survival.py` re-run with controls (the script now takes
them), where both quantities would be paired per speaker in one file.

### 37e. A cohort discrepancy found while checking that comparison

`exp_encoder_subspace.py` selected `sorted(ref)[:29]`, while every other script
in the project takes the **cross-session-filtered** set. The two cohorts overlap
in **21 of 29**; 8 of the α speakers are not in the cohort §27 and §33 report.

* **The gate is unaffected.** Each encoder is compared to *its own* control on
  *the same* speakers, and α is pure geometry — no enrolment, no thresholds, no
  second session — so cross-session usability is irrelevant to its validity.
* **Cross-section pairing is affected.** §32/§37 rows may **not** be paired
  per-speaker with §27/§33 rows, and no text may say "the same 29 speakers" of
  both.

This is the §25 failure mode once more: a run that silently changed the speaker
set. `exp_encoder_subspace.py` now takes `--cross-session` and **prints the
overlap with the reported cohort on every run**, so it cannot hide again.
Re-running the α table on the reported cohort costs ~1.5 GPU-h and is the
cheapest way to make the two sections pairable.

---

## 38. The front-end pre-check: the windows do move, and the embedding does not care (2026-10-05)

PLAN_V3 W1's last zero-GPU item ("run 2.4's CPU pre-check and report it as part
of #10's answer"). `experiments/diag_dvec_proxy.py --frontend --n-files 29
--cross-session`; `dvec_frontend_precheck.csv`; CPU only, no crafting.

### 38a. What was being checked

`IMPROVE_DVECTOR_2026-10-05.md` T1.1 argued the residual d-vector front-end gap
is a **window-alignment** problem:

> VAD trimming of LibriSpeech removes 0.3-0.8 s — half to a full partial hop.
> So the ~4 windows whose worst case `--seg-partials` optimises are *not* the
> windows the evaluator scores.

and added that `normalize_volume(increase_only=True)` "rescales the whole
signal — perturbation included". PLAN_V3 cut the implementation (§23a put the
whole remaining preprocessing gap at cosine 0.977) but kept the pre-check,
because the claim is the premise of a 2.5 GPU-h item and costs nothing to test.

The measurement maps each partial boundary of the **trimmed** signal back
through the VAD keep-mask to original sample indices, so "how far did this
window move" is answered in the original timebase rather than inferred. The
torch mel front end is held fixed on both sides, so gain and trimming are the
only variables. The reimplemented mask is asserted bit-identical to
`preprocess_wav`'s output on every speaker before anything is read off it.

Cohort: the **cross-session 29**, i.e. the cohort §27 and §33 report — not
§32/§37's. This section is therefore pairable with the rest of the paper.

### 38b. Result

| quantity | mean | SD | range |
|---|---|---|---|
| audio removed by VAD | **0.453 s** | 0.283 | 0.12 – 1.44 |
| the same, in inter-partial hops (0.77 s) | **0.59** | 0.37 | 0.16 – 1.87 |
| volume gain applied | **+0.000 dB** | 0.000 | 0.00 – 0.00 |
| max window shift per speaker | 0.315 s | 0.154 | 0.09 – 0.69 |
| cos(partial-averaged untrimmed, trimmed) | **0.9849** | 0.0110 | 0.9400 – 0.9945 |
| cos, matched partial by partial | **0.9369** | 0.0318 | 0.8576 – 0.9768 |

* Partials: **4.00 to 3.62** per speaker; 11 of 116 are lost outright, and
  10 of 29 speakers lose one.
* **33 of 105** comparable windows move by more than half a hop; 10 of 29
  speakers have at least one such window.

### 38c. Reading, against the branches written before the run

**The mechanism is real; the lever is not.** T1.1's first half is confirmed and
then some — a third of the surviving windows move more than half a hop, and the
per-partial cosine of 0.937 says those windows genuinely see different audio.
But the quantity that is actually scored, the partial **average**, moves by
0.0151 (cos 0.9849). The averaging absorbs the misalignment, which is exactly
the pre-registered "premise fails" branch and is consistent with §23a's
independent 0.977 for the whole preprocessing step.

The effect is orderly rather than noisy: seconds cut predicts the embedding
cosine at **r = -0.912** (Spearman -0.870), and the ten speakers who lose a
partial sit at 0.9761 against 0.9896 for those who keep all four. The worst case
in the cohort is speaker 260 — 1.44 s cut, cos 0.940 — and even that is above
the gap §23a already called minor.

**One claim in IMPROVE_DVECTOR is simply false.** `normalize_volume` applies
**exactly 0.000 dB on all 29 speakers**: `increase_only=True` fires only below
-30 dBFS and `load_flac` already peak-normalises to 0.9, so the gain branch is
unreachable on this corpus. "It rescales the whole signal, perturbation
included" never happens here, and the T1.1 fix would have been half aimed at
nothing. Add it to the §0 correction table.

### 38d. What this decides

Keeping the frozen-VAD front end **cut** is correct, and #10's
evaluation-only re-score is the right substitute at a tenth of the cost: the
confound it tests for is worth ~0.015 cosine on the quantity that scores us, not
the ~0.13 that P0 fixed. This is the second time a d-vector front-end lever has
been measured and found small (§23a was the first), and the second time the
partial *average* turned out to be the robust part while the per-partial
structure moved a lot.

Carry into #10 as the prior: WavLM-SV and ECAPA have no VAD stage at all, so if
the re-score moves them it is the ±2 % resample, not the trimming.

---

## 39. The v1 selection defects are now implemented, switched off, and locked by a golden file (2026-10-05)

PLAN_V3 W1: "commit 2.1 / 2.2 / 2.3 / 2d / 2e as code fixes behind
`--legacy-selection`, so v1 stays reproducible from HEAD. Golden-file test
committed first." `tests/test_selection.py` (15 tests, all passing),
`experiments/exp_purifier_aware.py`.

### 39a. The golden file comes first, and it is numeric

`craft()`'s two decision expressions lived in closures and could not be tested.
They are now module-level `agg_lse` and `ship_score`, extracted
behaviour-preservingly, and pinned:

    lse(c) = logsumexp((c - thr_t) * w * tau_lse) / tau_lse
    score  = c.max().item()

against stored values on three fixed cosine vectors, using the thresholds the
m30_seg arms were actually crafted against — **0.8740 / 0.3067 / 0.6819** from
`thresholds_diffpure_gl_t50_ddim_v7_xsess_tc.csv`, not the clean-channel triple
0.8926 / 0.3358 / 0.7029 the *cloning* script scores with. A second test
recomputes the literal pre-refactor expression so the constants cannot drift
together with the code. If any future edit changes what the v1 objective
computes, this fails — the §25 failure mode caught one level up.

### 39b. The fixes, all off by default

`--no-legacy-selection` turns on three things together; `--legacy-selection`
(the default) is v1.

| fix | what it changes | evidence it matters |
|---|---|---|
| **2.1** | ship-iterate chosen by margin to threshold, not raw cosine | thresholds span 0.567 here, so the largest-cosine encoder need not be the one closest to accepting; the test exhibits two iterates whose ranking inverts |
| **2e** | margins divided by each encoder's own score scale before the LSE | §35: the binding encoder changes for 4 of 29 speakers; the test reproduces a flip from ECAPA to the d-vector on this channel's measured scales |
| **2d** | selection on a trailing mean of `--sel-window` steps, not one draw | §36: at `n_eot=1` each score is a single purification draw, so argmin over 200 steps ships the luckiest one |

Two further items are unconditional because neither can change a run that
already exists:

* **2.3** — `tau_lse` and `seg_tau` are now CLI arguments at their crafted
  defaults (4.0, 8.0). `seg_tau` was previously an inner closure's default and
  could not be set at all.
* **2.2** — `--thr-aware` without `--thresholds` now **aborts**. It used to fall
  back silently to the model-card placeholders in `multi_encoders.py`
  (0.86 / 0.25 / 0.75) against calibrated values of 0.874 / 0.307 / 0.682 —
  0.047 of false slack on the d-vector, twice any effect this paper reports.
  Every run script already passes `--thresholds`, so this refuses a mistake
  rather than changing a result.

### 39c. Why off, and what it buys

§36d's reasoning stands: applying any of the three changes v1, and the base
never moves. What changes today is that they are **written, tested and
switchable** rather than described in prose — so the §36c disclosure paragraph
now names code a reviewer can run, and the W6 `--n-eot 5` parity ablation has
its arm ready instead of needing to be implemented under deadline.

The disclosure is unchanged and still conservative in our favour:

> Ship-iterate selection used raw cosine rather than threshold margin, and
> selected on a single purification draw; reported protection is therefore a
> lower bound for this recipe.

---

## 40. Council #6 — the quality evidence, including on the attacker's output (2026-10-05)

`experiments/eval_clone_quality.py --n-files 29 --asr openai/whisper-large-v3`;
`clone_quality.csv`; 174 files = 29 speakers x 6 conditions;
`code_sha=470b702bb6ff`. Paired per speaker, BCa 95 % CI (2000 resamples),
Wilcoxon signed-rank, Holm within each family.

### 40a. What was wrong with the old evidence, and what changed

Four defects, all named by the council:

| | defect | now |
|---|---|---|
| 6a | no no-reference score on the arm the paper ships | `fullnsT_tc_m30_seg`, the cross-session 29 |
| 6b | no NISQA; SQUIM read as naturalness | NISQA v2 (vendored, MIT, commit `fe84f0f2`) beside UTMOS; SQUIM labelled as the PESQ/STOI predictor it is |
| 6c | WER referenced to Whisper-small's own transcript | LibriSpeech ground truth, MFA word timings clipped to the 4-s crop by midpoint, `EnglishTextNormalizer` both sides, whisper-large-v3 |
| 6d | **no quality evidence on what the ATTACKER produces** | 116 DiffVC clones scored |

One methodological point had to be fixed before any number was trustworthy.
**Level was confounded with condition.** Measured here: the protected files sit
at **−18.8 LUFS** against **−24.3** for the clean originals, a 5.4 dB gap, so
the shipping arm was systematically the loudest thing in the comparison. NISQA
reports a loudness dimension and UTMOS is not level-invariant, so that gap alone
could have appeared as a naturalness difference. Every file is matched to
−23 LUFS before scoring (`utils/loudness.py`, the BS.1770-4 meter written for
the listening test), in both passes and in the copies NISQA reads.
**3 of 174** files hit the peak ceiling and did not reach the target; they are
flagged `lufs_clamped` in the CSV rather than dropped.

A second trap, in 6d itself: a DiffVC clone speaks the **attacker's** words in
the victim's voice, so a clone's WER reference is the **source** speaker's
transcript, read from the cloning CSV's `source_speaker`. Scoring clones
against the victim's transcript would have returned ~100 % WER for every clone,
protected or not, and looked like a finding.

### 40b. What we publish: one UTMOS point, and no intelligibility cost

Protected minus clean original, paired, n = 29, Holm over the four metrics:

| metric | protected | original | Δ | 95 % CI | lower in | p_holm |
|---|---|---|---|---|---|---|
| UTMOS | 3.073 | 4.152 | **−1.079** | [−1.219, −0.950] | 29/29 | 1.5e−08 |
| NISQA MOS | 3.266 | 4.054 | **−0.787** | [−0.912, −0.647] | 28/29 | 2.2e−08 |
| **WER** | 0.088 | 0.079 | **+0.009** | [+0.000, +0.022] | 0/29 | **0.157** |
| SQUIM-PESQ | 2.635 | 3.532 | −0.898 | [−1.071, −0.733] | 28/29 | 2.2e−08 |

The shape of this is the useful part. Naturalness costs about **one point of
UTMOS and 0.79 of NISQA** — consistent, 28–29 of 29 speakers, not a tail — and
lands at 3.07 / 3.27, i.e. "fair", not broken. **Intelligibility costs nothing
measurable**: +0.9 points of WER with a CI touching zero and the only
non-significant cell in the table. So the protection degrades perceived quality
without damaging content, which is the trade the paper should claim, and it is
now claimable from a no-reference measure instead of from PESQ 2.2, which
§eval_quality_nr's docstring already explained cannot decide anything here.

This replaces the "**Not run**" rows in Table 7 for NISQA and per-arm WER. MOS
and PSD stay absent: these are MOS *predictors*, and `FRAME_v6` §4 now forbids
calling them MOS.

### 40c. 6d — the attacker's clone is modestly less natural, and equally intelligible

Protected clone minus unprotected clone, paired per speaker, n = 29, Holm
within each purifier:

| purifier | metric | Δ | 95 % CI | lower in | p_holm |
|---|---|---|---|---|---|
| none | UTMOS | **−0.323** | [−0.444, −0.194] | 23/29 | 1.2e−04 |
| none | NISQA | **−0.442** | [−0.618, −0.275] | 25/29 | 4.1e−05 |
| none | WER | −0.001 | [−0.091, +0.048] | 5/29 | 1.00 |
| DiffWave | UTMOS | **−0.145** | [−0.262, −0.033] | 20/29 | 0.039 |
| DiffWave | NISQA | **−0.203** | [−0.357, −0.056] | 20/29 | 0.039 |
| DiffWave | WER | −0.029 | [−0.069, +0.025] | 11/29 | 0.155 |

**The answer is "yes, a little", and it has to be reported.** Protection does
make the attacker's output measurably less natural: 0.32 UTMOS / 0.44 NISQA
unpurified. So part of what the similarity numbers call protection coincides
with a modest degradation of the clone. A reviewer would have asked, and
"we did not measure it" would have been the wrong answer.

Three things bound it:

* **It is small beside the identity effect.** 0.3 of 5 points, while the same
  files move clone success from 100 % to 0 / 30 / 6 % (§cloning_tc_m30_seg_qual).
* **Intelligibility is untouched** in both purifier conditions (WER CIs span
  zero). The clone is slightly less natural, not unusable — so "the protection
  merely breaks the audio" is not what is happening.
* **DiffVC already costs more than the protection adds.** The *unprotected*
  clone is itself −0.374 UTMOS [−0.515, −0.229] below the clean original; the
  protected clone is −0.698 [−0.831, −0.548]. Protection roughly doubles a
  penalty the cloner was already paying.

And one observation offered as suggestive only: the degradation **halves** under
the attacker's purifier (−0.323 → −0.145) in the same cells where protection
also weakens. If the mechanism of protection were clone damage, a purifier that
removes the damage should remove the protection with it; instead both shrink
but protection persists on two encoders. That is consistent with identity
displacement being the operative effect rather than damage, but it is one
sample of 29 and is not claimed as a test of it.

### 40d. The two predictors are not interchangeable

Across the six conditions UTMOS and NISQA agree well (Pearson r = 0.87,
Spearman rho = 0.77, n = 6). **Per file they agree only moderately**
(r = 0.53, n = 174), and they disagree on which condition is best: UTMOS puts
the clean original top (4.15 vs NISQA 4.05), NISQA puts both unprotected clones
there (4.18). So a single no-reference score should not be quoted as *the*
naturalness number; both are reported, and the listening test remains the only
thing that can adjudicate between them.

This is also why 40b's conclusion is stated as a *direction with a consistent
sign across 28–29 of 29 speakers* rather than as a MOS value.

### 40e. What is still missing

Human ratings. The stimuli, the blinded page and the screener are built and the
panel is an unpaid colleague group (`FRAME_v6` §6, decided today), which changes
what the 20 Jan gate can conclude: the venue decision reads off the **naive
subgroup alone**. Until then MOS and PSD remain "not run" in Table 7, and that
table now reads its own status off these files instead of asserting it.

---

## 41. Council #10 — the front-end confound is dead where the claims live, and real where they do not (2026-10-06)

`experiments/eval_frontend.py --n-files 29`, decided by
`experiments/analyse_frontend.py`. `frontend_rescore.csv`,
`frontend_summary.csv`; 29 cross-session speakers, 2088 rows,
`code_sha=d417ba226f85`. **Evaluation only — nothing re-crafted, the base does
not move.**

### 41a. The objection, and why §23a and §38 did not answer it

R1: WavLM-SV and ECAPA are scored through the exact function the optimiser
differentiated through, while the d-vector alone is scored through
`preprocess_wav` (volume normalise + VAD trim) and a partial average — and the
d-vector is the one that fails. On that evidence the per-encoder pattern could
be a property of the encoders or an artefact of who is scored through what.

§23a and §38 measured how far the **d-vector's own** embedding moves under its
front end (cos 0.977, 0.985). #10 is the complementary question about the other
two, and it needs the swap run in both directions:

* WavLM / ECAPA **gain** `preprocess_wav`;
* the d-vector **loses** it, scored instead through the torch-mel partial
  average its neighbours effectively use.

Both the clean reference and the arm pass through the same front end on every
row, because that is what a verifier does. A ±2 % resample round trip is
included as a third front end: the benign channel a real upload applies, which
nothing was crafted against.

### 41b. Result: the answer depends on the purifier, and that is the finding

Paired per speaker, n = 29, swapped minus native on the **shipping arm**, BCa
95 % CI, Wilcoxon, Holm within encoder:

| purifier | encoder | Δ swapped − native | 95 % CI | p_holm | protection effect | ratio |
|---|---|---|---|---|---|---|
| **DiffWave t=3** | WavLM-SV | −0.0020 | [−0.0064, +0.0014] | 1.00 | +0.134 | 66× |
| | ECAPA-TDNN | −0.0053 | [−0.0144, +0.0041] | 0.39 | +0.442 | 84× |
| | d-vector *(front end removed)* | −0.0041 | [−0.0120, +0.0134] | 0.43 | +0.166 | 41× |
| **none** | WavLM-SV | **+0.1207** | [+0.0843, +0.1610] | 1.8e−07 | +0.704 | 5.8× |
| | ECAPA-TDNN | **+0.0344** | [+0.0218, +0.0583] | 2.6e−07 | +1.283 | 37× |
| | d-vector *(front end removed)* | **−0.0272** | [−0.0397, −0.0029] | 6.2e−04 | +0.468 | 17× |

A **positive** Δ on a protected arm means the swap moved the arm back towards
the speaker, i.e. less protection. "Protection effect" is
`cos(clean, unprotected) − cos(clean, protected)` in the same cell, so "ratio"
is how many times larger the thing being explained is than the shift offered to
explain it.

**Under the attacker's purifier — the cell every claim in the paper lives in —
nothing moves.** All three shifts are ≤ 0.005, every CI spans zero, no cell
survives Holm, and the protection effect is 41–84× larger. That is the
pre-registered branch 1: **the confound is dead there**, and C1's per-encoder
ordering is a property of the encoders rather than of their evaluation paths.

**Unpurified, the front end matters.** WavLM loses 0.121 and ECAPA 0.034 when
given the d-vector's front end — branch 2, and it must be reported. So the
front ends are **not** interchangeable in general, and no sentence in the paper
may say they are.

### 41c. What the unpurified row actually shows: the constraint buys front-end robustness

The unpurified shift is much larger on the **unconstrained** arm than on the
shipping arm:

| encoder | plain PGD | shipping (`fullnsT`) | ratio |
|---|---|---|---|
| WavLM-SV | +0.4890 | +0.1207 | 4.1× |
| ECAPA-TDNN | +0.2520 | +0.0344 | 7.3× |

Plain PGD loses roughly half of its WavLM protection the moment the verifier
trims silence and renormalises, which is a thing any deployed verifier does.
The purification-invariance constraint cuts that sensitivity four- to
sevenfold. That is a **new and favourable property of the constraint** which no
experiment here was designed to find: a perturbation tuned to survive a
purifier also survives a front end it never saw. Stated as an observation on
one sample, not as a claim about invariance in general.

It also explains the purifier split without any extra assumption. Unpurified,
the fine time-frequency structure the front end disturbs is still present.
After DiffWave has run, that structure is largely gone for every arm, so there
is nothing left for the front end to disturb and all six cells collapse to
zero.

### 41d. Removing the d-vector's front end does not rescue it

This is the branch the whole objection rests on, and it fails:

* Unpurified: **−0.0272** — the d-vector protects *slightly better* without its
  front end. Directionally what the confound hypothesis predicts, and 17×
  too small to matter.
* Under DiffWave, **where the d-vector actually fails**: **−0.0041**, CI
  [−0.0120, +0.0134], p_holm 0.43. Nothing. The d-vector still sits at cos
  0.779 against 0.945 unprotected, with or without its own front end.

So the d-vector's collapse under purification is **not** attributable to its
evaluation path. §38 predicted this from the other side — the partial average
absorbed the window misalignment at cos 0.985 — and the prediction was on
record before this run.

### 41e. Resample robustness, the cheap half of the cut 2.4

A ±2 % resample round trip moves the shipping arm by at most 0.018:

| encoder | unpurified | DiffWave |
|---|---|---|
| WavLM-SV | +0.0055 | −0.0003 |
| ECAPA-TDNN | +0.0177 | +0.0059 |
| d-vector | −0.0014 | −0.0011 |

ECAPA is the only encoder with a consistent, Holm-significant sensitivity, and
even there it is a fortieth of the protection effect. The protection is
effectively resample-robust, which is worth one sentence in the setup section
because no prior work in this line reports it.

### 41f. What the paper must say, and must not

* **Say:** "under the attacker's purifier, re-scoring every encoder through the
  others' front end changes the paired margin by ≤0.005 with CIs spanning zero,
  so the per-encoder ordering is not an artefact of the evaluation path."
* **Say:** "unpurified, the front end does matter, and it matters four to seven
  times less for the constrained arm than for plain PGD."
* **Do not say:** "the front ends are interchangeable", or quote the purified
  null without the unpurified row beside it.
* **§27c consequence:** the unpurified null row ("no cost term on the new base")
  is measured on the raw front end, where WavLM is 0.121 front-end sensitive.
  That row now needs the qualifier; its *sign* is unaffected because both arms
  in the paired difference move together.

Council #10 is closed. The remaining front-end question — whether a matched
front end changes α — is the C1 robustness row scheduled beside it.

---

## 42. Council #19 — the operating points, with intervals, and what minDCF reveals that EER hid (2026-10-06)

`experiments/calibrate_thresholds.py --channel clean --cross-session --subset
test-clean --n-spk 29 --n-utt 4 --dump-scores --tag xsess_tc_v2`, then
`experiments/analyse_asv.py`. 464 target and 12 992 non-target trials over 29
speakers; `asv_operating_points.csv`, `det_clean_xsess_tc_v2_*.csv`.

### 42a. A free reproducibility check, taken first

The recalibration was written to a **new tag** so the frozen
`thresholds_clean_xsess_tc.csv` could not move. It reproduces it exactly:

| encoder | frozen τ | recomputed τ | Δ | EER then | EER now |
|---|---|---|---|---|---|
| WavLM-SV | 0.8926 | 0.8926 | 0.0000 | 5.20 % | 5.20 % |
| ECAPA-TDNN | 0.3358 | 0.3358 | 0.0000 | 0.62 % | 0.62 % |
| d-vector | 0.7029 | 0.7029 | 0.0000 | 3.45 % | 3.45 % |

So the operating point the whole paper is scored against is bit-reproducible
from HEAD, two weeks and one `recipe-v1` tag later. The new file additionally
carries the SD columns PLAN_V3 2f added (`target_sd`, `nontarget_sd`,
`pooled_sd`, `dprime`, `speaker_set`), which is why it was worth re-running at
all.

### 42b. The operating points

Cluster bootstrap over **speakers**, 2000 resamples, 95 %. minDCF is
normalised by the better trivial system, C_miss = C_fa = 1.

| encoder | EER % | minDCF p=0.01 | minDCF p=0.05 |
|---|---|---|---|
| ECAPA-TDNN | **0.62** [0.00, 0.84] | **0.02** [0.00, 0.04] | **0.02** [0.00, 0.04] |
| Resemblyzer d-vector | 3.45 [1.99, 4.74] | 0.39 [0.26, 0.49] | 0.27 [0.17, 0.36] |
| WavLM-SV | 5.20 [3.69, 6.73] | **0.77** [0.62, 0.85] | 0.51 [0.38, 0.61] |

### 42c. The finding: our three "encoders" are not three comparable verifiers

EER alone says WavLM-SV (5.2 %) and the d-vector (3.5 %) are merely a few
points worse than ECAPA (0.6 %). **minDCF says something far stronger.** At
p_target = 0.01, the operating point a verifier facing mostly impostors
actually sits at:

* ECAPA is an excellent verifier — **0.02**, i.e. it costs 2 % of what the
  trivial system costs.
* The d-vector is mediocre — **0.39**.
* **WavLM-SV is barely better than trivial — 0.77**, with the CI reaching 0.85.

That reframes a sentence the paper was going to write. "Protection holds on
WavLM-SV and ECAPA, and fails on the d-vector" is true, but the three columns
do not carry equal weight: the encoder where protection is strongest (ECAPA)
is also the only one whose accept/reject decisions are trustworthy at a
realistic operating point, and the encoder with the most comfortable-looking
protection margin (WavLM-SV) is the weakest verifier of the three. Both
directions of that have to be said, because a reviewer who computes minDCF
will find it in a minute.

It also explains an oddity already on record: §23b/§27 repeatedly show WavLM-SV
moving by large cosine amounts for small changes in behaviour. An encoder whose
target and impostor distributions overlap this much (d' = 2.48, against 6.16
for ECAPA) has a compressed, noisy decision scale, so large cosine excursions
there mean less than they appear to.

### 42d. The council's actual complaint, quantified

> "EER 0.62 % = ~3 errors with no CI."

Correct, and now measurable: ECAPA's EER interval is **[0.00, 0.84] %** — it
*includes zero*. The point estimate 0.62 % is three errors out of 464 target
trials, and the honest statement is "below about 0.9 %", not "0.62 %". Any
table quoting ECAPA's EER to two decimals without the interval is overclaiming
precision, and the project did that in five places.

### 42e. The bootstrap has to cluster on speakers, and here is the price

Every trial a speaker appears in shares that speaker's voice, so trials are not
independent and a trial-level bootstrap understates the spread. Measured on
this data, the speaker-clustered interval is **1.7× wider** than the
trial-level one, consistently across all three encoders:

| encoder | speaker-clustered width | trial-level width |
|---|---|---|
| ECAPA-TDNN | 0.84 pts | 0.50 |
| d-vector | 2.75 | 1.61 |
| WavLM-SV | 3.04 | 1.80 |

1.7×, not an order of magnitude — 29 speakers × 16 trials each is not severe
clustering. But it is the difference between ECAPA's interval containing zero
and not, so it is the one to report. (An earlier draft of this section said
"several times narrower"; that was a guess, and the measurement replaces it.)

### 42f. What this adds to the paper

* §2 gains the real operating-point table: EER **and** minDCF at two p_target
  values **and** speaker-bootstrap CIs, which is the convention
  `FRAME_v6` §3 lists against council #19.
* DET curves are on disk per encoder, so the low-false-alarm region can be
  plotted without re-running anything.
* The per-encoder weighting above is a limitation to state in §6, not a result
  to bury: with one strong verifier and two weak ones, a three-encoder
  "coverage" claim is less uniform than it looks.
* Still open from #19: the same treatment for the **purified** channels
  (diffpure t=50, DiffWave t=1/t=3). Only the clean channel has raw scores so
  far; the others need the same `--dump-scores` re-run, which is ~1 GPU-h and
  is the remaining piece of this item.

---

## 43. C1 robustness: α is unchanged through the d-vector's real front end, and the cohort discrepancy is closed (2026-10-06)

`experiments/exp_encoder_subspace.py --n-files 29 --cross-session
--dvec-frozen-vad --t 50,200 --seeds 2026,7`;
`encoder_subspace_xsess_fe.csv`. PLAN_V3 W2's C1 robustness row, which also
settles §37e.

### 43a. Two questions, one run

**The robustness question.** §32/§37 compute α(g_dvec) through `DvecEnc.embed`
— the torch-mel partial average with **no** `preprocess_wav`. The real
evaluation path trims silence first. So a reviewer can ask whether the
d-vector's low α is a property of the encoder or of the front end it was
measured without. That is council #10's objection aimed one level upstream,
at the *direction* rather than at the protection, and §41 does not answer it.

`preprocess_wav` is not differentiable: the VAD decision is a hard threshold.
Freezing that decision **on the clean signal** makes the rest differentiable —
the trim becomes a fixed index selection — and §38 measured the
volume-normalisation gain at exactly 0.000 dB across this cohort, so there is
nothing else in `preprocess_wav` to model. Freezing on the clean signal is
legitimate precisely because the quantity is a property of the clean utterance.
The twin is measured as an additional direction, so it is paired per speaker
with the native one and gets its own matched control.

**The cohort question.** §37e found that the α table was computed on
`sorted(ref)[:29]` while the rest of the paper uses the cross-session-filtered
set: 21 of 29 shared, 8 not. α is pure geometry so it was valid on its own
terms, but §32/§37 could not be paired per speaker with §27/§33. This run uses
`--cross-session`, i.e. the reported cohort.

### 43b. The front end does not move α

Paired per speaker, n = 29, frozen-VAD minus native:

| t | native α | through the real front end | Δ | 95 % CI | p |
|---|---|---|---|---|---|
| 50 | 0.3880 | 0.3907 | **+0.0027** | [−0.0176, +0.0182] | 0.51 |
| 200 | 0.2622 | 0.2653 | **+0.0031** | [−0.0106, +0.0134] | 0.37 |

Nothing. Both intervals straddle zero, and the point estimates are a hundredth
of the gap C1 is about (α 0.568 against 0.388 across encoders). **α(g_dvec) is
not an artefact of measuring it without `preprocess_wav`.**

The ordering is untouched, with the twin landing on top of the native d-vector
at both timesteps:

> t = 50: WavLM-SV 0.568 > ECAPA 0.463 > d-vector (frozen VAD) 0.391 ≈ d-vector 0.388
> t = 200: WavLM-SV 0.369 > ECAPA 0.302 > d-vector (frozen VAD) 0.265 ≈ d-vector 0.262

### 43c. The council #3 gate still clears, on the reported cohort

All four directions, including the twin, against their own band-matched
controls, Holm over four:

| direction | α | matched control | Δ | 95 % CI | below | p_holm |
|---|---|---|---|---|---|---|
| WavLM-SV | 0.568 | 0.708 | −0.1398 | [−0.1625, −0.1111] | 27/29 | 2.6e−08 |
| ECAPA-TDNN | 0.463 | 0.711 | −0.2483 | [−0.2878, −0.2033] | 28/29 | 2.2e−08 |
| d-vector | 0.388 | 0.702 | −0.3139 | [−0.3653, −0.2603] | 28/29 | 2.2e−08 |
| d-vector *(frozen VAD)* | 0.391 | 0.699 | −0.3087 | [−0.3576, −0.2578] | **29/29** | 1.5e−08 |

`pca_span` 0.844 and `random` 0.714 reproduce §37d: the invariant subspace is
speech-shaped, and every encoder direction points substantially out of it.

### 43d. The cohort correction, and what it costs

| encoder | §37 cohort | reported cohort | shift |
|---|---|---|---|
| WavLM-SV | 0.573 | 0.568 | −0.005 |
| ECAPA-TDNN | 0.490 | 0.463 | −0.028 |
| d-vector | 0.406 | 0.388 | −0.019 |

Overlap is 21 of 29 exactly as §37e said; the eight swapped speakers are
listed in the CSV. The shifts are small, the ordering is unchanged, and the
gate verdict is unchanged. **So §37e cost the paper nothing except the right to
say "the same 29 speakers" — and that right is now bought back.** §32/§37 may
be paired per speaker with §27/§33 from here, using
`encoder_subspace_xsess_fe.csv`; the older `encoder_subspace_matched.csv`
stays on disk as the record of what §37 reported and should not be mixed with
it.

**Which file the paper quotes:** this one. Fig. 1 is built from
`encoder_subspace_xsess_fe.csv`, and §32/§37's numbers are restated at the
values above. The differences are within a fiftieth everywhere, so no
conclusion moves, but two tables quoting α to three decimals from two cohorts
would be the §25 failure mode in miniature.

### 43e. What is now closed, and what C1 still owes

Closed: #3 (matched controls, §37), the front-end objection at the protection
level (#10, §41) and at the direction level (this section), and the cohort
discrepancy (§37e).

Still open for C1, both scheduled: **#13c**, the same α measurement through
DiffWave, so the explanation and the phenomenon live on the same purifier; and
**#13b**, the prospective test on CAM++ and ERes2Net that turns C1 from an
explanation into a prediction.

---

## 44. Council #8, the C1 half: all ten α pairs under one Holm family (2026-10-06)

`experiments/analyse_alpha_pairs.py` on `encoder_subspace_matched.csv` (the
file §32/§37 report). Zero GPU. `alpha_pairs.csv`.

### 44a. Why the five p-values in §32b were not enough

§32b quotes five pairwise α comparisons with unadjusted p-values. With five
directions there are **ten** pairs, so five of them were selected after the
means were visible, and the multiplicity was unaccounted. `FRAME_v6` §2 already
hedges against this by restricting C1 to "a two-group separation plus d-vector
nsr > 1, never a four-way ranking" — a sensible guess, made without the
arithmetic. This section does the arithmetic: all ten pairs, one Holm family
per timestep, paired per speaker, BCa CI plus Wilcoxon, with a Friedman omnibus
first.

### 44b. t = 50 — nine of ten pairs separate

Friedman χ² = 71.5, p = 1.1e−14, so the five directions are not interchangeable.

| A − B | Δ | 95 % CI | A>B | p_holm |
|---|---|---|---|---|
| d-vector − random | −0.3065 | [−0.3644, −0.2452] | 0/29 | 3.7e−08 |
| random − x-vector | +0.2791 | [+0.2318, +0.3268] | 29/29 | 3.7e−08 |
| ECAPA − random | −0.2227 | [−0.2637, −0.1786] | 2/29 | 2.1e−07 |
| d-vector − WavLM | −0.1665 | [−0.2202, −0.1104] | 5/29 | 3.2e−05 |
| WavLM − random | −0.1400 | [−0.1657, −0.1042] | 2/29 | 1.4e−06 |
| WavLM − x-vector | +0.1391 | [+0.1012, +0.1789] | 25/29 | 2.5e−06 |
| ECAPA − d-vector | +0.0838 | [+0.0277, +0.1429] | 21/29 | **3.2e−02** |
| ECAPA − WavLM | −0.0827 | [−0.1196, −0.0506] | 4/29 | 1.3e−04 |
| ECAPA − x-vector | +0.0564 | [+0.0148, +0.0976] | 19/29 | **3.2e−02** |
| d-vector − x-vector | −0.0274 | [−0.0926, +0.0371] | 14/29 | 0.54 |

Structure at t = 50, stated at the resolution the Holm column supports:

> **random > WavLM-SV > ECAPA-TDNN > { x-vector ≈ d-vector }**

### 44c. t = 200 — seven of ten, and the middle collapses

| not separated at t = 200 | Δ | p_holm |
|---|---|---|
| ECAPA − d-vector | +0.0400 | 0.15 |
| ECAPA − x-vector | +0.0227 | 0.23 |
| d-vector − x-vector | −0.0174 | 0.47 |

Structure at t = 200:

> **random > WavLM-SV > { ECAPA ≈ x-vector ≈ d-vector }**

### 44d. What C1 may now claim, and it is more than FRAME_v6 allowed

`FRAME_v6` §2 and §4 currently say "WavLM separates from the other three" and
forbid any ordering. That is **exactly right at t = 200 and too conservative at
t = 50**, where ECAPA separates from both low encoders with Holm-adjusted
p = 0.032 on each side.

The claim the data support, with the timestep attached:

* **At t = 50: a three-group separation** — WavLM-SV, then ECAPA, then
  x-vector and the d-vector together, every step surviving Holm over all ten
  pairs.
* **At t = 200: a two-group separation** — WavLM-SV against the rest; the
  middle is no longer resolvable.
* **At both: x-vector and the d-vector are never distinguishable** (p = 0.54,
  0.47). The pair at the bottom is a tie, and any sentence implying an order
  within it is unsupported.
* **At both: every encoder direction sits below random**, which is C1's actual
  content and the most robust thing in the table.

The two timesteps disagreeing is itself informative rather than awkward:
stronger purification compresses the differences between encoders, which is
what §37d's "the invariant subspace is the subspace clean speech occupies"
predicts — push harder and everything that is not clean speech is removed
alike.

### 44e. Consequence for the claims ledger

`FRAME_v6` §4's row should be timestep-specific rather than blanket:

| write this | not this |
|---|---|
| "at t = 50, WavLM-SV > ECAPA > {x-vector, d-vector}, all ten pairs under one Holm family" | "WavLM-SV > ECAPA > x-vector > d-vector" |
| "at t = 200 only WavLM-SV separates" | quoting the t = 50 ordering without its timestep |
| "x-vector and the d-vector are not distinguishable at either timestep" | any order within that pair |

Note the x-vector rows are available only on the §37 cohort: the
cross-session re-run (§43) was done without `--with-xvector`. The pairs above
are therefore reported from `encoder_subspace_matched.csv`, and §43's cohort
caveat applies — they may not be paired per speaker with §27/§33. Re-running
the pairs on the cross-session cohort with the x-vector would cost ~1.5 GPU-h
and is **not** scheduled; the ordering is unchanged between cohorts (§43d), so
the only thing it would buy is pairability for a table that does not need it.

**Still open in #8:** the sign-test reframing of floor p-values, log-ratio plus
BCa for retention, TOST for §30, the mixed model, and the missing
`analyse_mismatch_final.py`. Those are W8 items and are untouched here.

---

## 45. #5 — C2 is directionally reproduced out of sample, and underpowered at n = 31 (2026-10-06)

The pre-registered out-of-sample replication. `run_devclean_replication.sh`
executing `PREREG_05_devclean_replication.md`; 31 cross-session dev-clean
speakers, recipe v1 unchanged, BPDA, thresholds calibrated on test-clean;
`main_table_dc_m30_seg_bpda.csv`, `dc5_foreign_*.csv`, `dc5_cells.csv`.
`prereg_sha=7b2e245ae936`.

**VERDICT: OUTCOME B.** Written as the pre-registration requires:
*directionally reproduced out of sample, underpowered at n = 31.* **Not
"replicated".**

### 45a. The 12 cells

Paired per speaker, (fullnsT − full) margin. **Negative = the constraint
improves protection.** `pred` is the test-clean point estimate from §27a, fixed
as the prediction before the run. Holm over all 12.

| # | purifier | t | encoder | pred | Δ | 95 % CI | k/n | p_holm | |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DW-Libri | 1 | WavLM-SV | −0.076 | −0.0553 | [−0.0848, −0.0208] | 23/31 | .023 | * |
| 2 | DW-Libri | 1 | ECAPA | −0.046 | −0.0620 | [−0.0952, −0.0252] | 21/31 | .023 | * |
| 3 | DW-Libri | 1 | d-vector | **+0.071** | **+0.0596** | [+0.0330, +0.0892] | 4/31 | 4.9e−04 | * |
| 4 | DW-Libri | 3 | WavLM-SV | −0.026 | −0.0232 | [−0.0419, −0.0007] | 22/31 | .055 | ~ |
| 5 | DW-Libri | 3 | ECAPA | −0.029 | −0.0133 | [−0.0322, +0.0086] | 17/31 | .67 | |
| 6 | DW-Libri | 3 | d-vector | +0.014 | +0.0127 | [−0.0013, +0.0325] | 14/31 | .67 | ~ |
| 7 | DW-SC09 | 1 | WavLM-SV | −0.078 | −0.0596 | [−0.0958, −0.0273] | 24/31 | .014 | * |
| 8 | DW-SC09 | 1 | ECAPA | −0.038 | −0.0605 | [−0.0944, −0.0249] | 22/31 | .023 | * |
| 9 | DW-SC09 | 1 | d-vector | **+0.075** | **+0.0664** | [+0.0400, +0.0965] | 3/31 | 2.4e−04 | * |
| 10 | DW-SC09 | 3 | WavLM-SV | −0.046 | −0.0300 | [−0.0551, −0.0078] | 19/31 | .074 | |
| 11 | DW-SC09 | 3 | ECAPA | −0.034 | −0.0249 | [−0.0457, −0.0035] | 21/31 | .17 | |
| 12 | DW-SC09 | 3 | d-vector | +0.014 | +0.0102 | [−0.0075, +0.0308] | 12/31 | .67 | ~ |

`*` = CI excludes 0 **and** Holm p < .05. `~` = pre-declared underpowered in
§5, so non-significance there is uninformative by prior agreement.

### 45b. The three predictions

| | | result |
|---|---|---|
| **P1** | WavLM negative in all four cells | **4/4 negative**, 2/4 significant |
| **P2** | d-vector positive at t = 1, both purifiers | **2/2 positive, 2/2 significant** |
| **P3** | ECAPA keeps the sign in all four cells | **4/4 negative** |

Outcome A required P1 at ≥3 of 4 significant. It is 2 of 4. **That single
shortfall is the whole reason this is B rather than A**, and the pre-registered
action follows without discretion.

### 45c. How closely the pattern reproduced

Every cell has the predicted sign — **12 of 12** — which under a coin-flip null
is p = 4.9e−04. Across the 12 cells, the dev-clean estimate against the
test-clean prediction:

| | |
|---|---|
| Pearson r | **0.961** (p = 6.8e−07) |
| Spearman ρ | 0.902 |
| regression slope, observed on predicted | **0.850** |
| mean \|observed\| / mean \|predicted\| | 0.873 |

So the *shape* of C2 transferred almost exactly to speakers never crafted
against, at about 85 % of the magnitude. That is a stronger statement than the
verdict label carries, and it is reportable — but it is a descriptive
agreement, not a significance claim, and it does not upgrade B to A.

### 45d. Where the power went: entirely to t = 3

| timestep | significant | mean \|Δ\| |
|---|---|---|
| t = 1 | **6 of 6** | 0.0606 |
| t = 3 | **0 of 6** | 0.0190 |

At the mild purifier setting the replication is complete: all six cells, both
purifiers, all three encoders, every sign right and every cell clearing Holm.
At t = 3 the effects are a third the size and nothing clears.

Three of the six t = 3 cells (4, 6, 12) were pre-declared underpowered in §5,
so their silence was anticipated. **Three were not** — cells 5, 10 and 11, at
−0.013, −0.030 and −0.025 against a pre-stated detectable effect of
0.030–0.045. Cell 10 sits exactly on that boundary; cells 5 and 11 are below
it. That is the honest accounting: the run resolved what it predicted it could,
and the t = 3 WavLM cells were on the edge rather than comfortably inside.

The t = 1 / t = 3 split is a design factor of the fixed 12-cell grid, not a
subgroup chosen after the fact, and §5 already flagged t = 3 as the weak half.

### 45e. The reversal — the prediction C1 exists to explain — replicated outright

| purifier | Δ | 95 % CI | speakers improved | p_holm |
|---|---|---|---|---|
| DiffWave-LibriSpeech t=1 | +0.0596 | [+0.0330, +0.0892] | **4 of 31** | 4.9e−04 |
| DiffWave-SC09 t=1 | +0.0664 | [+0.0400, +0.0965] | **3 of 31** | 2.4e−04 |

On test-clean this was +0.071 and +0.075 with 3 of 29 improved. On 31 speakers
never crafted against, scored at an operating point set on a different cohort,
it is +0.060 and +0.066 with 4 and 3 of 31. **This is the single most
transferable thing in the paper**, and it is the cell α(g_dvec) was measured to
account for (§32, §43).

### 45f. The control, which is NOT one of the 12 cells, and does not match test-clean

Unpurified, mean paired (fullnsT − full), n = 31:

| encoder | dev-clean | test-clean (§27c) |
|---|---|---|
| WavLM-SV | +0.0174 | null (p = .77) |
| ECAPA-TDNN | **+0.0418** | null (p = .26) |
| d-vector | +0.0765 | +0.079 |

The d-vector reproduces almost exactly. **ECAPA does not**: test-clean found no
unpurified cost, dev-clean finds +0.042. So `FRAME_v6`'s ledger row — "there is
no cost term on the new base (§27c)" — is **test-clean-specific and must not be
stated as a general property.** On dev-clean the constraint does cost something
unpurified on all three encoders. This is a control rather than an endpoint, so
it carries no Holm correction and no verdict, but it is the kind of thing a
reviewer finds and it is cheaper to say first.

### 45g. The sensitivity check was vacuous, and that is a defect in the pre-registration

§2.2 fixed a check: re-score the same 12 cells against thresholds calibrated on
dev-clean itself; if the verdict flips, call it "threshold-dependent" and report
neither scoring. It was run (`run_devclean_sensitivity.sh`) and the verdict is
unchanged — **but not because the result is robust to the threshold choice.
Because it cannot depend on it.**

The endpoint is a *paired difference of margins*:

> (cos_nsT − τ) − (cos_full − τ) = cos_nsT − cos_full

τ cancels algebraically. Verified rather than argued: across all 12 cells the
two scorings agree to **0.0 exactly** — identical Δ, identical CIs, identical
Holm p — while the thresholds genuinely differ (WavLM 0.8892 vs 0.8769, ECAPA
0.3253 vs 0.2774, d-vector 0.7016 vs 0.6753).

So **§2.2 provides no evidence either way, and I wrote it believing it could.**
It would bite on any *absolute* quantity — per-arm margins, binary protection
rates — and the mean per-arm margin does move materially with the calibration
(libri t=1 ECAPA: −0.216 against −0.168). The pre-registration excluded exactly
those as endpoints (§6), which is why the check it specified could not fire.
The check was unnecessary, not reassuring, and the correct statement is that
**the primary endpoint is threshold-invariant by construction**.

**A real finding fell out of it anyway.** dev-clean's own calibration is far
looser than test-clean's at the same protocol:

| channel | WavLM-SV | ECAPA | d-vector |
|---|---|---|---|
| test-clean EER | 5.85 % | **0.86 %** | 3.84 % |
| dev-clean EER | 8.67 % | **4.84 %** | 6.05 % |

dev-clean is a harder verification cohort — ECAPA's EER is 5.6× worse. So while
the *paired deltas* are comparable across the two samples, any **absolute**
protection number is not, and the two corpora must never be pooled or compared
on rates. That reinforces §6's "do not pool them" for a reason the
pre-registration did not anticipate.

### 45h. What C2 may now claim

* **Write:** "the per-encoder pattern reproduces directionally on 31 speakers
  never crafted against, scored at an operating point calibrated on a disjoint
  cohort: 12 of 12 cells with the predicted sign (r = 0.96 against the
  first sample), all six mild-purifier cells significant under Holm, and the
  d-vector reversal significant on both purifiers. At the stronger purifier
  setting the effects are a third the size and none of the six cells resolves
  at n = 31."
* **Do not write:** "replicated". C1 keeps the lead. C2 remains a supported
  hypothesis with two samples behind it rather than one.
* **Limitations must say:** the confirmatory evidence is 29 + 31 speakers from
  one corpus family, read out as two samples and never pooled; the t = 3 half
  is unresolved at this n; and the unpurified control disagrees with test-clean
  on ECAPA.
* **No re-run** at another budget, timestep, purifier or speaker subset. §6
  forbids it and nothing here invites it.

### 42g. The purified channels, and the verifier the attacker leaves behind (2026-10-06)

The #5 sensitivity run calibrated the four foreign channels on dev-clean with
`--dump-scores`, so they get the same treatment at no extra GPU cost. n = 31,
496 target / 14 880 non-target trials, speaker-cluster bootstrap.

| channel | ECAPA EER / minDCF₀.₀₁ | d-vector | WavLM-SV |
|---|---|---|---|
| clean (test-clean, n=29) | 0.62 % / **0.02** | 3.45 % / 0.39 | 5.20 % / 0.77 |
| DiffWave-Libri t=1 | 4.84 % / 0.17 | 6.05 % / 0.40 | 8.67 % / 0.74 |
| DiffWave-SC09 t=1 | 4.64 % / 0.17 | 5.65 % / 0.38 | 8.42 % / 0.74 |
| DiffWave-Libri t=3 | 5.42 % / **0.36** | 6.65 % / 0.49 | 10.90 % / **0.86** |
| DiffWave-SC09 t=3 | 4.64 % / **0.38** | 5.88 % / 0.44 | 10.48 % / **0.88** |

**The attacker's purifier degrades the verifier, not just the perturbation.**
ECAPA goes from an excellent operating point (minDCF 0.02) to a mediocre one
(0.36–0.38 at t = 3); WavLM-SV goes from 0.77 to 0.86–0.88, i.e. from barely
better than trivial to effectively not better at all.

That has to be said beside every protection number measured through these
channels. A clone "failing" a verifier whose minDCF is 0.88 is a weaker
statement than the same failure against ECAPA on the clean channel, and the
paper should not let the three columns read as equally authoritative.

**Caveat, and why #19 is not yet closed.** These calibrations are on
**dev-clean**, because that is the cohort the sensitivity run needed. The
main table's cells are scored against **test-clean** foreign calibrations,
which still have no raw trial scores. Closing #19 for the channels the paper
actually reports needs the same `--dump-scores` run on test-clean, ~1 GPU-h,
and §45g's finding that dev-clean is the harder cohort means these numbers are
**not** a substitute for it.

### 42h. #19 CLOSED — every channel the paper reports (2026-10-06)

`run_asv_conventions.sh` recalibrated all eight channels on **test-clean** with
`--dump-scores`, to `_v2` tags so nothing frozen moved.
`analyse_asv.py --tag _tc_v2`; 464 target / 12 992 non-target trials over 29
speakers per channel; 24 DET curves on disk.

**minDCF at p_target = 0.01** (1.0 = no better than the trivial system):

| channel | ECAPA | d-vector | WavLM-SV |
|---|---|---|---|
| clean | **0.02** | 0.39 | 0.77 |
| DiffWave-Libri t=1 | 0.11 | 0.44 | 0.81 |
| DiffWave-SC09 t=1 | 0.12 | 0.43 | 0.83 |
| DiffWave-Libri t=3 | 0.30 | 0.51 | **0.90** |
| DiffWave-SC09 t=3 | 0.31 | 0.48 | **0.91** |
| ours v7 t=50 | 0.10 | 0.54 | **0.91** |
| ours v7 t=100 | 0.11 | 0.54 | **0.91** |
| ours v7 t=200 | 0.10 | 0.57 | **0.90** |

**The headline is the WavLM-SV column.** In *every cell the paper reports*, the
encoder sits at minDCF 0.77–0.91 — at the matched purifier and at DiffWave
t = 3 it is effectively **not better than the trivial system** at a realistic
operating point, despite EERs of 5–9 % that read as respectable. The d-vector
is mediocre throughout (0.39–0.57). **ECAPA is the only genuinely strong
verifier**, and even it degrades threefold under DiffWave t = 3 (0.02 → 0.30).

This reframes how the three columns may be read, and it cuts both ways:

* "Protection holds on WavLM-SV" is the **weakest** of the three claims, not
  the strongest, because the verifier making that judgement is barely
  discriminating in those cells. Any sentence pairing it with ECAPA as equal
  evidence is wrong.
* "Protection holds on ECAPA" is the **strongest**, because ECAPA is the only
  encoder whose accept/reject decision is trustworthy at p_target = 0.01.
* The d-vector failure — the paper's central negative result — is measured on
  a mediocre verifier, which **weakens** it symmetrically. §45e's reversal is a
  paired *margin* effect and does not depend on the operating point (§45g), so
  that result stands; but any **absolute** d-vector clone rate inherits this.

It also explains a long-standing oddity: WavLM-SV moves by large cosine amounts
for small behavioural changes (§23b, §27). Its target and impostor
distributions overlap heavily — d′ = 2.48 against ECAPA's 6.16 — so its cosine
scale is compressed and noisy, and a large excursion there means less than the
same excursion on ECAPA.

**Reproducibility, as a by-product.** All **24** (channel, encoder) operating
points regenerate **exactly** — threshold and EER identical to four decimals
against the frozen files (`check_calibration_drift.py`). The paper's entire
operating point is therefore regenerable from HEAD, two weeks and one
`recipe-v1` tag later.

Council #19 is closed: EER, minDCF at two p_target values, DET curves and
speaker-cluster bootstrap CIs, for every channel in the paper.

---

## 46. #13b — the prospective test: every sign right, every effect null (2026-10-06)

The pre-registered prospective test of C1 on two encoders the optimiser has
never seen. `PREREG_13b_prospective_encoders.md`; α from
`encoder_subspace_13b.csv`, outcome from `extra_encoder_protection.csv`,
decided by `analyse_extra_encoders.py`. 29 cross-session test-clean speakers,
evaluation only.

**VERDICT: OUTCOME B — directional only.** The predicted sign is right in
**8 of 8** cells and **not one effect is distinguishable from zero**. C1 keeps
the word "explanation"; `FRAME_v6` §4's ban on "predicted" stands.

### 46a. The design, and the one thing it got right

C1 was generated after observing the behaviour it explains. The only way to
turn that into a prediction is to commit one before the outcome is visible, so
the item was run in three separately committed stages:

1. the rule — α* = 0.4255, helps above, reverses below — committed **before any
   α was measured**;
2. α measured on CAM++ and ERes2Net, and the derived per-encoder prediction
   committed as §7 **before any crafted file was scored through either**;
3. the outcome.

The decider refuses to run at all if §7 is absent, because a prospective test
whose prediction is written afterwards is not one.

### 46b. α: C1's core clause holds on two unseen architectures

| direction | α (t=50) | α (t=200) |
|---|---|---|
| `pca_span` | 0.8439 | 0.6201 |
| `random` | 0.7142 | 0.4653 |
| band-matched controls | 0.699–0.711 | 0.453–0.465 |
| **CAM++** | **0.6308** | 0.3855 |
| **ERes2Net** | **0.5760** | 0.3547 |
| WavLM-SV | 0.5678 | 0.3693 |
| ECAPA-TDNN | 0.4627 | 0.3016 |
| d-vector | 0.3880 | 0.2622 |

Both new encoders sit below `random` and below their **own** band-matched
control: CAM++ −0.074 [−0.101, −0.050] with 24/29 speakers below, ERes2Net
−0.133 [−0.155, −0.101] with 28/29. So *every speaker encoder's identity
direction is purifier-hostile* now holds across **five encoders from four
architecture families**, two of them never seen by the optimiser, by any
analysis or by any calibration. That is the most robust statement in the paper
and it got more robust today.

### 46c. The architectural prior was wrong, and is on record as wrong

§3.1 predicted both would land between the d-vector (0.388) and WavLM-SV
(0.568), in the ECAPA/x-vector band, on the reasoning that both are fbank
TDNN/CNN encoders. **Both are above that band.** CAM++ at 0.631 is above every
encoder previously measured. The architectural reasoning simply did not hold.

It was marked in advance as worth little and it was worth less than that. It
constrains nothing in the real rule, which keys only on which side of α* a
value falls — but it is reported as a failed sub-prediction rather than quietly
dropped, and it is a caution against reading α off an architecture diagram.

### 46d. The outcome: right direction, no magnitude

Paired Δ = cos(clean, fullnsT) − cos(clean, full), n = 29, Holm over 8 cells.
**Both encoders were predicted to HELP (Δ < 0).**

| encoder | purifier | t | Δ | 95 % CI | k/29 | p_holm | sign |
|---|---|---|---|---|---|---|---|
| CAM++ | DW-Libri | 1 | −0.0066 | [−0.0275, +0.0185] | 16 | .93 | ✓ |
| CAM++ | DW-Libri | 3 | −0.0102 | [−0.0268, +0.0065] | 16 | .93 | ✓ |
| CAM++ | DW-SC09 | 1 | −0.0167 | [−0.0379, +0.0049] | 19 | .93 | ✓ |
| CAM++ | DW-SC09 | 3 | −0.0178 | [−0.0328, −0.0017] | 22 | .12 | ✓ |
| ERes2Net | DW-Libri | 1 | −0.0145 | [−0.0420, +0.0147] | 17 | .93 | ✓ |
| ERes2Net | DW-Libri | 3 | −0.0095 | [−0.0273, +0.0100] | 21 | .93 | ✓ |
| ERes2Net | DW-SC09 | 1 | −0.0163 | [−0.0419, +0.0131] | 18 | .93 | ✓ |
| ERes2Net | DW-SC09 | 3 | −0.0190 | [−0.0372, +0.0009] | 21 | .27 | ✓ |

**0 of 8 cells survive Holm.** Seven of eight CIs span zero. Per-speaker
agreement runs 16–22 of 29 against a chance baseline of ~14.5.

The effects are also **4.4× smaller** than the same quantity on the
crafted-against encoders at t = 1 (mean |Δ| 0.0138 against 0.0606) and 1.4×
smaller than at t = 3. That is expected — these encoders were never optimised
against, so the constraint has no direct purchase on them — but it is the
reason the cells cannot resolve at n = 29.

### 46e. Why this is outcome B and not outcome A

The script's first pass printed **outcome A**, because §5's outcome A keys on
sign agreement alone and the signs are perfect. That is wrong, and the
pre-registration already said so: **§7.4, committed with the prediction and
before the outcome, states "a null — Δ indistinguishable from zero — is NOT a
confirmation and is read under §5's outcome B."**

Every cell here is consistent with zero. So the stricter, earlier-committed
reading governs and the verdict is B. `analyse_extra_encoders.py` now
implements §7.4 rather than §5 alone, so the flattering reading is not
available next time either.

The sign agreement is still worth reporting, as evidence of direction and
nothing more: 8 of 8 is p = 0.0078 under a coin flip — **but the eight cells
are two purifiers × two timesteps × two encoders over the *same* 29 speakers,
so they are nowhere near independent and that p is optimistic.** It is quoted
as suggestive, never as a test.

### 46f. The weakness this test had from the start, stated in advance

§7.4 recorded it before the outcome: both encoders landed on the same side of
α*, so **both predictions pointed the same way**. The rule could therefore not
be falsified by the two encoders disagreeing — only by a positive Δ where
"helps" was predicted. A sharper test needed an encoder predicted to *reverse*,
and neither available one qualified.

So even an unambiguous confirmation here would have been weaker evidence than
it looked. Reporting that is the difference between a prospective test and a
decorated one.

### 46g. What the paper says

* **Write:** "on two encoders never used in crafting, chosen and committed in
  advance, the constraint moved protection in the predicted direction in all
  eight cells, with no cell distinguishable from zero at n = 29."
* **Write:** "every encoder's identity direction is purifier-hostile — now
  five encoders across four architecture families, two of them unseen."
* **Do not write:** "predicted", "confirmed", or "C1 generalises". `FRAME_v6`
  §4's ban stands.
* **No re-run** at a different α*, timestep or third encoder. §5 forbids it and
  the result does not invite it.
* C1's status is unchanged: the strongest claim remains the α ordering against
  matched controls (§37, §43, §44), not its predictive power.
