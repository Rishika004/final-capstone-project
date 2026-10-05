# Paper frame v6 — the reframe

Written 2026-10-04 from `REVIEW_COUNCIL_2026-10-04.md` (all five reviewers agreed
on this). Supersedes the framing of `draft_v5.md` and of
`PLAN_TO_CLOSE_THE_CHAIN.md`.

**BASE RULE: new base only.** Every number in the paper comes from RESULTS
§21–32 — LibriSpeech test-clean, purifier v7, cross-session thresholds, n = 29
speakers, budget m30_f15, `--seg-partials`. Nothing from §1–20 may appear as a
number, including in the abstract, intro, tables or figures. Where a §21–32
number does not yet exist, the slot is marked **[PENDING #n]** against the
council item that produces it, and stays empty until then.

---

## 1. Why the frame changes

Two of our own experiments on the new base removed both novelty claims of the
defence framing:

* **§29** — De-AntiFake's full PhonePuRe recovers the speaker (WavLM cosine
  0.296 → 0.900, 29/29). We beat their stage 1 only.
* **§30** — BPDA and exact gradients give the same protection (1 of 21 Holm
  cells, Δ −0.009), at half the craft time. The gradient estimator is not why
  earlier adaptive attempts failed.

What survived is not a method but a measurement with an explanation: why a
perturbation that survives purification still fools some speaker encoders and
not others. That is an analysis paper, and the two defeats become findings
inside it.

---

## 2. Title, thesis, contributions

### Title
**What a Speech Purifier Erases Is What a Speaker Encoder Reads:
Encoder Geometry and the Limits of Purification-Robust Voice Protection**

### Thesis
A direction that moves a speaker encoder is one a speech purifier removes more
readily than a random direction, and how strongly that holds differs by encoder;
this single crafting-free measurement accounts for which encoders a
purification-invariance constraint helps, which it hurts, and where the
protective-perturbation paradigm stops working.

### C1 — The measurement (§32)
Through purifier v7, each encoder's own identity direction survives worse than a
norm-matched random direction. α at t = 50: WavLM-SV 0.573, ECAPA 0.490,
x-vector 0.434, d-vector 0.406, random 0.713 (random > WavLM +0.140, 27/29).
WavLM separates from the other three; ECAPA > d-vector only at t = 50
(p_adj .032); x-vector vs d-vector is **not** separable (p = .54 / .47). The
d-vector's noise-to-signal ratio exceeds 1.

Claim it at the resolution §44 measured, with the timestep attached: **at
t = 50 a three-group separation** (WavLM-SV > ECAPA > {x-vector ≈ d-vector},
all ten pairs in one Holm family, both middle steps p_adj = .032); **at
t = 200 a two-group separation** (WavLM-SV against the rest). x-vector and the
d-vector are **never** distinguishable (p = .54, .47), so no order inside that
pair. Plus d-vector nsr > 1. Never a four-way ranking.
**#3 DONE (§37, §43)** — spectrally-matched and clean-PCA-span controls, 8 of
8 cells confirmed; the envelope-matched controls land at 0.701–0.715 against
white noise 0.713, so the gap is not a spectral artefact. **Front-end
robustness DONE (§43)** — α through the d-vector's real front end moves
+0.003, CI straddling zero, and §41 closes the same objection at the
protection level. **Cohort DONE (§43)** — the α table is now on the
cross-session cohort and is pairable with §27/§33; quote
`encoder_subspace_xsess_fe.csv`, and restate §32's α as 0.568 / 0.463 / 0.388.
**#8 (C1 half) DONE (§44)** — all ten pairs under one Holm family per
timestep, with a Friedman omnibus.
**[PENDING #13c the same measurement through DiffWave]**

### C2 — What it explains: the constraint's encoder-dependent effect (§27, §30e)
Under purifier mismatch at the loud budget the invariance constraint improves
the paired margin on WavLM-SV in all four foreign cells (Δ −0.026 to −0.078),
is same-sign but mostly non-significant on ECAPA (1/4 at m = 24), and **worsens**
the d-vector at mild purification (+0.071, +0.075; 3/29). Unpurified it is null
on WavLM and ECAPA (p = .77, .26) and worse on the d-vector (+0.079, 0/29).

Stated as a **hypothesis supported by two samples**, 29 test-clean and 31
dev-clean, read out separately and **never pooled**. §45: directionally
reproduced out of sample — 12 of 12 cells with the predicted sign, r = 0.96
against the first sample, all six mild-purifier cells significant under Holm,
and the d-vector reversal significant on both purifiers — but **underpowered at
n = 31**, with none of the six t = 3 cells resolving. **Do not write
"replicated".** §27/§30e remain analysis-choice insensitivity, not replication.
C1 is the explanation, generated after observing the reversal, not a prediction.
**#5 DONE (§45) — OUTCOME B of the three pre-registered branches: directionally reproduced out of sample, underpowered at n = 31. The pre-registration's own sensitivity check turned out to be vacuous (the paired endpoint is threshold-invariant by construction, §45g).**
**[PENDING #13b CAM++ prospective test makes C1 a prediction]**

### C3 — The paradigm's measured ceiling (§22b, §30, §29)
1. **Coverage, not generalisation** (§22b). Same files, same thresholds, one
   variable: clone success 12–14 % on crafted-against encoders vs 75–92 % on the
   held-out one.
2. **The gradient estimator is not the lever** (§30). BPDA matches exact within
   4 points everywhere; exact gradients help only against the purifier they were
   computed through, at its strongest setting (§30d). Report as a TOST
   equivalence, not as a failure to reject. **[PENDING #8]**
3. **A learned refinement stage recovers most of what purification leaves**
   (§29). Full PhonePuRe restores identity on 29/29 files; end-to-end clone
   success rises from 6/0/29 % to 38/27/71 % — reduced, not removed — and the
   collapse concentrates on the d-vector, C1's lowest-α encoder. ASV acceptance
   of the purified reference overstates what DiffVC actually clones, which is a
   methodological warning for proxy-scored protection papers.
   **#4 DONE (W1)** — the pre-filter is keyed per (encoder, purifier); the
   union made protection look better everywhere it mattered. The rates are
   still single-seed on small n, so §31c still forbids shipping them as point
   estimates: quote ranges pooled over seeds, with n.
   **Quality of the attacker's output DONE (§40)** — the protected clone is
   0.32 UTMOS / 0.44 NISQA less natural unpurified, halving under the
   purifier, with WER untouched. Report it: part of what the similarity
   numbers call protection coincides with modest damage to the clone.

### Not a contribution
The crafting recipe (purifier-in-the-loop optimisation at EOT size 1,
threshold-aware ensemble loss, segment crafting after AntiFake §C2, m30_f15)
moves to §2 as "the protection we analyse".

---

## 3. Four-page budget (Interspeech: 4 + 1 references)

| § | Content | Pages | Source | Gate |
|---|---|---|---|---|
| 1 | Intro: purification defeats protection; our question; C1–C3 | 0.6 | related_work §0, §0b | — |
| 2 | Setup: rewritten threat model, corpus, disjoint calibration, encoders, purifiers with denoise gain and conditioning disclosed, the protection analysed, metrics, operating point | 0.6 | §21, §23, §26d, §40, §42 | ~~#2~~ ~~#6~~ ~~#7~~ #11 #19 (clean channel done, purified channels open) |
| 3 | **C1.** Fig. 1: α per encoder vs matched random at t = 50/200 | 0.9 | §32, §37, §43, `encoder_subspace_xsess_fe.csv` | ~~#3~~ #8 #13c |
| 4 | **C2.** Table 1: 4 cells × 3 encoders, paired Δ with CI and Holm, plus the unpurified row | 0.7 | §27a–c, §30e | #5 #8 |
| 5 | **C3.** Table 2: coverage contrast and the PhonePuRe two-number row; BPDA in one sentence | 0.8 | §22b, §29, §30, §40 | ~~#4~~ #8 |
| 6 | Limitations and conclusion | 0.4 | council §B | — |

Purifier transfer (the new-base figures from `foreign_tc_*.csv`) is supporting
context in §2–3, not a headline: it is two checkpoints of one architecture, and
the one architecturally different purifier (§29) breaks it. **#2 DONE
(§33).**

---

## 4. Claims ledger

| Write this | Not this |
|---|---|
| "purification-invariant subspace"; "null space" once, defined against Shallow Diffuse | "null space" as the name of our object |
| "at t = 50, WavLM-SV > ECAPA > {x-vector, d-vector}, all ten pairs in one Holm family"; "at t = 200 only WavLM-SV separates" (§44) | "WavLM > ECAPA > x-vector > d-vector"; the t = 50 ordering without its timestep; any order inside {x-vector, d-vector} |
| "an explanation generated after observing the reversal" | "retro-validation", "predicted", "without being told" |
| "two samples, 29 + 31 speakers, read out separately and never pooled"; "directionally reproduced out of sample, underpowered at n = 31" (§45) | "replicated"; "a hypothesis supported by one sample"; any pooling of the two corpora, or any comparison of ABSOLUTE rates between them (§45g: dev-clean EER is up to 5.6x worse) |
| "reduced, not removed, by full PhonePuRe" | "reverses De-AntiFake"; "survives purification" without "stage 1" |
| "no advantage over BPDA (equivalence tested)" | "exact gradients are why prior work failed" |
| "single-sample stochastic gradient (EOT size 1)" | bare "EOT" beside baselines' 1/5/10/15 sweeps |
| "iterated mel denoiser in a DiffPure-style loop, denoise gain 0.19" | "diffusion purifier" for v7 |
| "two checkpoints of one architecture" | "a property of speech diffusion purifiers" |
| "matched-encoder (upper bound)" / "held-out (realistic)" | "deployment" for the matched-encoder setting |
| "k of n speakers; sign test at its attainable minimum" | floor p-values quoted as strength |
| ranges pooled over three seeds, n stated, n ≥ 12 | any single-seed binary rate as a point estimate |
| "null unpurified on WavLM and ECAPA **on test-clean**" (§27c); "on dev-clean the constraint costs +0.017/+0.042/+0.077 unpurified" (§45f) | "there is no cost term", stated as a general property — it is test-clean-specific |
| "n unpaid colleagues, k of them naive to the method", with the recruitment route in the same sentence | "a listening test with n listeners"; any bare MOS from this panel |
| "UTMOS / NISQA are MOS *predictors*" | "MOS" for an automated score |

---

## 5. Positioning (novelty check run 2026-10-04)

| Prior work | What it has | Our delta |
|---|---|---|
| arXiv 2505.22839 v2 (Aug 2026) | purification preserves on-manifold directions, in images | we do not claim it; we measure the target encoders' own directions, in speech |
| ClearMask (ASIA CCS 2025) | on-manifold voice protection, no purification test | not proposed here; cited only |
| De-AntiFake (ICML 2025), VocalBridge (2026) | adaptive protection fails against their purifiers | §30 removes the estimator as the explanation; §29 runs their full pipeline against an invariance-constrained perturbation |
| AntiFake (CCS 2023) | ensemble coverage, segment loss | §22b measures the coverage limit with one variable changed; C1 gives a criterion for choosing members |
| arXiv 2510.09504 | removability of speaker-adversarial perturbations by attacker knowledge | no per-encoder or identity-direction analysis (checked today) |
| arXiv 2607.00899 (PnP) | purification defence for ASV, four backbones | defence side; no direction analysis (checked today) |
| arXiv 2310.14270 (DAP) | diffusion purification for ASV | defence side |

C1 is open as of today: no paper found that measures the survival of
speaker-encoder identity directions through a speech purifier, or compares it
across encoders. Re-check before submission.

---

## 6. Decided in advance, so it is not decided under deadline

* **#3 null** (identity direction survives no worse than a spectrally-matched
  random direction): C1 becomes a spectral statement — encoder identity lives in
  the bands a purifier removes — still per-encoder, still explains C2. Retitle
  around bandwidth, and report the null.
* **#3 null and the per-encoder gaps vanish**: C1 is dead. C3 leads, C2 becomes
  an observation, venue moves to SPSC.
* **#5 replication fails**: report that C2 does not replicate; C1 and C3 carry
  the paper.
* **#13b contradicts C1**: report it. A failed prospective test is a result; a
  skipped one is a hole.
* **The listening panel is unpaid colleagues, not 20–25 paid naive listeners**
  (decided 2026-10-05). PLAN_V3's 20 Jan gate reads "no human listening data →
  SPSC". A convenience panel is not the same as satisfying that gate, so the
  gate is sharpened here before the data exists, not after:
  * The decision is read off the **naive subgroup alone**. Non-naive ratings
    are reported beside it and never substituted for it.
  * Interspeech needs the naive subgroup to carry the two claims the test
    exists for — the shipping arm is not catastrophically unnatural, and it is
    still heard as the same speaker — with the panel size stated in the
    sentence that states the number.
  * If the naive subgroup is too small to support those, the result is reported
    as a pilot and the venue moves, exactly as a null would. "We ran a
    listening test" with five non-naive colleagues is weaker than reporting no
    panel and saying why, because the first invites a reviewer to discover the
    composition and the second does not.

---

## 7. Retired

* `draft_v5.md` — history only. Do not edit it and do not quote its numbers; it
  is written entirely on the old base.
* `PLAN_TO_CLOSE_THE_CHAIN.md` — served the defence framing. Its P5 (second
  cloner) and P6 (perceptual evidence) survive as council items #16 and #6.
* Cut for this submission: the reduced-N adaptive attack through PhonePuRe
  stage 2; more optimisation steps (§24a); further gradient-estimator work
  (§30); additional same-family purifiers.
