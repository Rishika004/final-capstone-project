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
norm-matched random direction. α at t = 50 **on the reported cross-session
cohort** (`encoder_subspace_13b.csv`, §43/§46): WavLM-SV 0.568, ECAPA 0.463,
d-vector 0.388, random 0.714, plus the two eval-only encoders CAM++ 0.631 and
ERes2Net 0.576. x-vector 0.434 is available only on the §37 cohort
(`encoder_subspace_matched.csv`, §44e) and must be quoted with that caveat.
*(The older 0.573 / 0.490 / 0.406 triple is the §37 cohort and is superseded —
two α tables in two files is the §25 failure mode.)*
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

**SUPERSEDED BY §47 (2026-10-08).** Control A held the d-vector OUT of the
crafting ensemble, one variable changed, and the reversal **disappears and
flips sign**: +0.0736/+0.0790 matched against −0.0084/−0.0120 held out, with
non-overlapping CIs. The reversal is a property of optimising against the
d-vector, not of the constraint. C2's reversal is demoted to a
**matched-encoder observation**, and §45e may no longer be called the most
transferable result. One crafting arm (`full_m30_seg_nodvec`, ~1 GPU-h) is now
mandatory before C2 is written either way.

**NOW TESTED AS AN INTERACTION (§52, 2026-10-08).** The 12-cell grid fits
twelve simple effects and compares signs by eye; C2's claim is a
`recipe × encoder` interaction and the grid tests it nowhere. The mixed model
`margin ~ recipe*encoder*purifier + (1|speaker)` does:

| cohort | condition | interaction LRT |
|---|---|---|
| test-clean (§27) | matched | χ² 76.9, df 8, **p = 2.1e−13** |
| dev-clean (§45) | matched | χ² 50.2, df 8, **p = 3.7e−8** |
| test-clean (§47) | **held-out** | χ² 11.2, df 8, **p = 0.19** |

Two analyses with different assumptions agree with §47. And **the whole
interaction is the d-vector**: ECAPA vs WavLM is non-significant in all three
cohorts (p = .26, .82, .80), so **C2 is a two-group claim**,
{WavLM-SV, ECAPA} against {d-vector}. The "same-sign but mostly
non-significant on ECAPA" phrasing above describes a difference in
*significance* that the model says is not a difference in *effect* — ECAPA's
cells are noisier, not smaller. Rewrite it that way.

Power on our own null (§52g): the held-out cohort's MDE on that contrast is
0.0823 and the matched effect 0.1463 is **1.78×** it, so a matched-sized
interaction would have been found; an interaction **half** that size would not.
Write "the matched-sized interaction is absent", never "the interaction
vanishes".

Stated as a **hypothesis supported by two samples**, 29 test-clean and 31
dev-clean, read out separately and **never pooled**. §45: directionally
reproduced out of sample — 12 of 12 cells with the predicted sign, r = 0.96
against the first sample (**descriptive only; the p-value on r and on the
12-of-12 count are withdrawn, §48c**), all six mild-purifier cells significant
under Holm, and the d-vector reversal significant on both purifiers — but
**underpowered at n = 31**. **§48b: no t = 3 cell could have resolved** — the
pre-registration named three cells underpowered and the arithmetic says six, so
"none of the six t = 3 cells resolving" is not an outcome. **Do not write
"replicated".** §27/§30e remain analysis-choice insensitivity, not replication.

**§49 (budget × crafting attribution).** The m20 null was a **budget** effect,
not a segment-crafting effect: crafting mode moves these cells by 0.0072 on
average with zero sign flips, budget by 0.0281 with two. So write "at the loud
budget" and drop the segment-crafting hedge. The d-vector reversal is
**budget-insensitive** (+0.069…+0.079 in all three recipes) while the
WavLM/ECAPA benefit is **budget-dependent** — two different mechanisms.
C1 is the explanation, generated after observing the reversal, not a prediction.
**#5 DONE (§45) — OUTCOME B of the three pre-registered branches: directionally reproduced out of sample, underpowered at n = 31. The pre-registration's own sensitivity check turned out to be vacuous (the paired endpoint is threshold-invariant by construction, §45g).**
**#13b DONE (§46) — OUTCOME B, and RELABELLED INCONCLUSIVE by §48a.** The
prospective test ran on CAM++ and ERes2Net with the rule and the per-encoder
prediction each committed before the next measurement. All 8 cells have the
predicted sign. **C1 keeps "explanation" and the ban on "predicted" stands.**

But §48a measured what that test could detect: **8 of 8 cells sit below their
own 80 %-power detectable effect, median power 21 %** (range 8.6–57.5 %). So
"none is distinguishable from zero" is true and **uninformative**, and the word
**null must not be used** — §46 is *inconclusive*. And on the per-speaker sign,
which is the more powerful test here, **one cell survives Holm** (CAM++ /
SC09 / t = 3, 22/29, p_holm .0325) with two more just outside. That cell is
**not** promoted: `PREREG_13b` §7.4 fixed the rule on the mean paired
difference before scoring, and switching endpoint to harvest it is the §25
failure mode.

**Consequence: §46 becomes two sentences in Limitations, not a results
subsection** (R3's position, §C.2, now settled by arithmetic). What the test
genuinely bought moves into C1 and is **not** underpowered: C1's core clause
holds on **five encoders across four architecture families**, two never seen by
the optimiser (§46b). That is a level claim and it is the part worth keeping.

### C3 — The paradigm's measured ceiling (§22b, §30, §29)
1. **Coverage, not generalisation** (§22b). Same files, same thresholds, one
   variable: clone success 12–14 % on crafted-against encoders vs 75–92 % on the
   held-out one.
2. **The gradient estimator is a small lever, not the lever** (§30, §51).
   **#8 TOST DONE (§51), and the pre-specified test FAILED.** Equivalence at
   the `PLAN_V3`-pre-specified margin D = 0.01 holds in only **7 of 24 cells**;
   the data support equivalence at **D = 0.032**. And the sign is systematic:
   exact gradients protect *better* in 10 of 12 cells, **mean −0.005**, which
   §30's per-cell Holm family was too coarse to resolve.

   So: **never write "equivalent"**, and never "no significant difference,
   therefore the same". Write "exact gradients protect better by 0.005 on
   average, bounded above by 0.032 in the worst cell and 0.007 at the median".

   The contribution **survives and improves**: 0.005 margin units cannot
   explain why De-AntiFake's and VocalBridge's adaptive attacks failed when the
   effects at stake are 0.076. A measured bound with a measured direction beats
   a null. Two riders: the 0.032 bound is **larger** than §49's budget contrast
   (0.028), the d-vector t = 3 cells (0.014) and §47's held-out cells (0.012),
   so those were never cross-estimator claims and must not be read as such;
   and BPDA remains the default at half the GPU cost, with that cost now
   **stated** as ≈0.005 of protection rather than assumed to be zero.
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
| 3 | **C1.** Fig. 1: α per encoder vs matched random at t = 50/200 | 0.9 | §32, §37, §43, §44, `encoder_subspace_xsess_fe.csv` | ~~#3~~ ~~#8~~ #13c |
| 4 | **C2.** Table 1: 4 cells × 3 encoders, paired Δ with CI and Holm; the unpurified row; **the matched-vs-held-out pair (§47, §53) and the interaction LRT (§52)** | 0.7 | §27a–c, §30e, §47, §49, §52 | ~~#5~~ ~~#8~~ §47e arm |
| 5 | **C3.** Table 2: coverage contrast and the PhonePuRe two-number row; the BPDA **bound** in one sentence | 0.8 | §22b, §29, §30, §40, §51 | ~~#4~~ ~~#8~~ |
| 6 | Limitations and conclusion — **now carries §46 (two sentences, inconclusive), §48b's pre-registration defect, §50's rerun term, and THREAT_MODEL §6b's four unmeasured capabilities** | 0.4 | council §B, §46, §48, §50 | — |

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
| "exact gradients protect better by 0.005 on average (10 of 12 cells); the difference is bounded by 0.032 in the worst cell, 0.007 at the median" (§51) | "no advantage over BPDA"; "equivalent"; "equivalence tested" — the pre-specified D = 0.01 TOST **failed** in 17 of 24 cells; "exact gradients are why prior work failed" |
| "§46 is inconclusive: 8 of 8 cells below their 80 %-power detectable effect, median power 21 %" (§48a) | "every effect null"; "a reported null"; §46 as a results subsection |
| per-speaker agreement k/n with an exact sign test, Holm within section (§48c) | "12 of 12 cells, p = …"; "8 of 8 cells, p = …"; the r = 0.961 **p-value** |
| "all six t = 1 cells informative and significant; no t = 3 cell could have resolved" (§48b) | reading any t = 3 non-significance as a failed replication |
| "C2 is an interaction, significant in the matched condition (p = 2e−13) and not detectable held out (p = 0.19, with power for a matched-sized effect)" (§52, §52g) | "the interaction vanishes held out"; treating p = 0.19 as proof of no interaction |
| "{WavLM-SV, ECAPA} against {d-vector}" — a two-group claim (§52c) | three distinct per-encoder constraint effects; "mostly non-significant on ECAPA" as if ECAPA's effect were smaller rather than noisier |
| "at the loud budget" (§49) | hedging the claim on segment crafting — crafting mode moves it 0.0072, budget 0.0281 |
| "the minimum over an eight-configuration, three-architecture measured set" (THREAT_MODEL §6a) | "min over purifiers"; "robust to purification" |
| "matched-encoder (upper bound)" on every verdict drawn from §27, §29e, §40, §41, §45 (THREAT_MODEL §3) | "held-out (realistic)" — "realistic" is withdrawn |
| "crafting is stochastic at fixed seed; two identical runs differ by up to 0.03 on one utterance" (§50) | within-run CIs presented as the whole uncertainty of a crafting comparison |
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

---

## 8. Status after the 2026-10-08 statistics pass (§48–§52)

`PLAN_V3` W8 is **brought forward and largely closed**, at zero GPU. What moved:

| item | status |
|---|---|
| #8 TOST for §30 | **DONE (§51).** Pre-specified margin **failed**; bound is 0.032, direction systematic. |
| #8 mixed model | **DONE (§52).** C2 is an interaction: p = 2e−13 matched, p = 0.19 held out. |
| #8 sign-test reframing of floor p-values | **DONE (§48c).** Across-cell p-values withdrawn; per-speaker exact sign test, Holm within section. |
| #8 all ten α pairs under Holm | **DONE (§44).** |
| #8 log-ratio + BCa for retention | **DONE (§33).** |
| `analyse_mismatch_final.py` | **DONE (§49).** The budget × crafting attribution: it is the budget. |
| power/MDE on every null cell | **DONE (§48a, §48b, §52g).** |
| council B1 / B8 / B9 / B10 | **DONE** in `THREAT_MODEL` §3, §6, §6a, §6b. |
| council #9 EOT parity | **CUT** (council v4 §C.3, §E.10): 8 GPU-h defending a declared non-contribution. |
| §47e mandatory held-out seg arm | **RUNNING** — `run_heldout_seg.sh`, ~6 min/speaker; read by `analyse_heldout_seg.py`, prediction pre-committed in §49e. |
| Control B (the "speaker" control) | **BUILT AND QUEUED** — `run_control_b.sh`, 3 untrained inits; premise verified (+0.020/+0.006 separation against trained +0.614). **It can falsify the title.** |
| B7 protected-arm operating point | **BUILT AND QUEUED** — `eval_protected_asv.py`. 29 target trials, not 464; the caption must say so. |
| #13c α through DiffWave | **BUILT AND QUEUED** — `run_13c.sh` + `band_matched_wav`; both checkpoints, both timesteps. |
| Fig. 1, Tables 1–3, draft v6 | **DONE** — `make_fig1.py`, `make_tables_v7.py`, `draft_v6.md` (two [PENDING] slots marked). |
| claims lint | **DONE** — `check_claims.py`; it found a real B1 violation in `THREAT_MODEL` §6. |
| novelty re-check | **DONE** — `related_work.md` §13. C1 open; **C2.1 narrowed** after verifying MDD against the paper. |
| CAM++/ERes2Net as calibrated verifiers | **not built** (council §E.9). |
| Listening panel | **launched, zero responses.** The 20 Jan gate reads off the naive subgroup (§6). |

The GPU chain runs unattended: `run_queue_w7.sh` waits for the arm, then
Control B, then #13c; `run_queue_w7b.sh` waits for that and runs B7. One job
at a time, liveness judged by log staleness, each stage skipped if its output
already exists.

**The three sentences the paper has gained, none of which it had on 2026-10-07:**

1. C2's encoder-dependence is a **tested interaction** in the matched condition
   and **not detectable** held out — two analyses, one conclusion (§47, §52).
2. The m20 null was a **budget** effect; segment crafting is nearly inert
   (0.0072 against 0.0281), so the claim is "at the loud budget" (§49).
3. The gradient estimator costs about **0.005**, bounded by 0.032 — a measured
   quantity where there was a null (§51).

**And the three it has lost:**

1. "BPDA ≡ exact" as an equivalence (§51).
2. §46 as a reported null, and as a results subsection (§48a).
3. The across-cell sign p-values and the r = 0.961 p-value (§48c).

Every loss was found by our own arithmetic before a reviewer's, which is the
only version of this that is worth anything.
