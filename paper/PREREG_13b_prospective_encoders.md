# Pre-registration — #13b, the prospective test that turns C1 into a prediction

**Written 2026-10-06, before α(g_CAM++) or α(g_ERes2Net) has been computed and
before either encoder has scored a single crafted file.** PLAN_V3 W6. The
content hash of this file lands in every output row as `prereg_sha`.

---

## 1. Why this exists

C1 says each speaker encoder's own identity direction survives purification
worse than a norm-matched random direction, by an amount that differs per
encoder, and that this accounts for C2's per-encoder pattern — the invariance
constraint helps WavLM-SV and ECAPA under purifier mismatch and **reverses** on
the d-vector.

That explanation was **generated after observing the behaviour it explains**.
`FRAME_v6` §4 requires it to be written that way and forbids the word
"predicted". §32's retro-validation on x-vector is suggestive but was run
knowing x-vector had already failed as a crafting member (§24b).

A prediction is different from an explanation in exactly one way: it is
committed before the outcome is visible. This document does that.

---

## 2. The two encoders, and why them

Neither has ever been in the crafting ensemble, in any analysis, or in any
threshold calibration. The optimiser has never seen them.

| | family | why it is a fair test |
|---|---|---|
| **CAM++** | context-aware-masking TDNN over 80-dim fbank (3D-Speaker) | architecturally nearest ECAPA/x-vector, so C1 should place it in that band |
| **ERes2Net** | Res2Net 2-D CNN with attentional feature fusion, 80-dim fbank | a fourth architecture family — not a transformer, not a channel-attentive 1-D CNN, not an LSTM |

Both are VoxCeleb-trained English models, matching the corpus. Measured
separation on 6 cross-session test-clean speakers before any of this was
written: CAM++ same 0.740 / different 0.126, ERes2Net 0.726 / 0.074. Both are
**strong** verifiers, stronger in raw separation than the three in the paper.

They are **eval-only**. `build_encoders` is not touched; nothing here can
change a crafted file. §24b's lesson, and the reason this costs ~2 GPU-h rather
than the ~12 that adding a crafting encoder would.

---

## 3. Two predictions, at different strengths

### 3.1 SECONDARY (architectural prior) — where α will land

Stated now, before measuring, and worth little on its own because it leans on
architecture rather than on C1:

> **α(g_CAM++) and α(g_ERes2Net) will both fall between the d-vector (0.388)
> and WavLM-SV (0.568) at t = 50**, i.e. in the ECAPA (0.463) / x-vector
> (0.434) band, and **both will be below `random` (0.714)**.

The second clause is the one C1 actually requires: *every* encoder's identity
direction should be purifier-hostile. An α **above** random on either encoder
contradicts C1 directly and is recorded as such in §5.

### 3.2 PRIMARY — α predicts the sign of the constraint effect

This is the real test, and it is a **rule fixed now**, not a number chosen
later. After α is measured, each new encoder is assigned a predicted sign by
one rule, applied mechanically:

> Let α_e be the encoder's survival coefficient at t = 50, and let
> α* = 0.4255 be the midpoint between the d-vector (0.388), the only encoder
> that reverses, and ECAPA (0.463), the lowest that does not.
>
> **Predict the constraint HELPS (Δ < 0) if α_e > α*, and REVERSES (Δ > 0) if
> α_e < α*.**
>
> Δ is the paired per-speaker difference `fullnsT − full` in cosine to the
> clean original, evaluated through the same foreign purifier cells the paper
> reports. **Negative = the constraint improves protection.**

α* is fixed here at 0.4255 and may not move. The rule is deliberately crude —
a single threshold on one measured number — because a rule with free parameters
chosen after the fact would predict anything.

The prediction derived from the measured α values will be appended to this file
as §7 **before any crafted file is scored through either encoder**, with its own
commit.

---

## 4. Design, fixed

| | |
|---|---|
| α measurement | `exp_encoder_subspace.py`, 29 cross-session test-clean speakers, purifier v7 DDIM, t = 50 and 200, seeds 2026,7, `--pca-subset dev-clean`, matched `band_matched` + `pca_span` controls as in §37 |
| protection measurement | `eval_frontend.py`-style eval-only scoring of the **existing** `full_tc_m30_seg` and `fullnsT_tc_m30_seg` wavs — nothing re-crafted |
| cells | 2 purifiers (DiffWave-LibriSpeech, DiffWave-SC09) × t ∈ {1, 3} × 2 encoders = **8** |
| endpoint | mean paired per-speaker Δ = cos(clean, fullnsT) − cos(clean, full), n = 29 |
| estimator | mean paired difference, BCa 95 % CI (2000 resamples), Wilcoxon signed-rank, Holm over the 8 cells |
| thresholds | **none needed** — the endpoint is a paired difference of cosines to the clean original, so no operating point enters it (and §45g showed a margin difference cancels τ anyway) |

The reference values the prediction will be judged against, already on record
for the three crafted-against encoders (§27a / §45a), are Δ ≈ −0.055 to −0.062
on WavLM and ECAPA at t = 1, and +0.060 to +0.066 on the d-vector at t = 1.

---

## 5. Outcome → action, written before the result

### Outcome A — C1 predicts out of sample
**Both encoders' sign matches the rule in §3.2 in ≥3 of their 4 cells, and the
t = 1 cells agree.**

→ **C1 becomes a prediction, not merely an explanation.** `FRAME_v6` §4's ban
on the word "predicted" is lifted *for this specific claim*, with the rule and
this document cited. C2 gains independent support that does not come from the
same sample.

### Outcome B — mixed
**One encoder's sign matches and the other's does not, or the agreement is
2 of 4.**

→ **Report both.** C1 keeps "explanation", and the paper states that a
prospective test on two unseen encoders agreed on one and not the other, naming
which. No post-hoc adjustment of α*.

### Outcome C — C1 does not predict
**Neither encoder's sign matches, or an α lands above `random`.**

→ **Report it as a failed prospective test.** `FRAME_v6` §6 pre-holds this
branch for #13b ("a failed prospective test is a result; a skipped one is a
hole"). C1 is demoted to a description of the three encoders it was measured
on, the ordering claim stays, and the mechanism language goes. **No re-run with
a different α*, a different timestep, or a third encoder.**

### Binding
* The 8 cells and α* are fixed. Neither may change after results are seen.
* α is measured **once**, at the stated settings. The t = 200 column is
  reported but the rule uses t = 50, as stated in §3.2.
* If an encoder fails to load or produces a degenerate embedding, that cell is
  **missing**, not a null, and the run is inconclusive for it.

---

## 6. What would invalidate the run

1. `prereg_sha` in the output rows does not match this file at the commit the
   rows were produced under (checked the way `assemble_devclean.py` checks it:
   historical hash plus append-only).
2. Either encoder appears in `build_encoders`, or any crafted file changes.
3. α* is altered, or a sign rule other than §3.2 is applied.
4. The protection cells are scored on anything but the existing
   `full_tc_m30_seg` / `fullnsT_tc_m30_seg` wavs.
5. Any commit lands between the α measurement and the §7 prediction commit
   that touches the crafting or scoring path (the §10 lesson from PREREG_05).
