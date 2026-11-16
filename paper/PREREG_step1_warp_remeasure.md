# Pre-registration — Step 1, fixed-magnitude warp on the current base

**Written 2026-10-09, before the run.** `PLAN_V4` §4 reads the result as
GATE 2. Runs in parallel with Step 0 and does not wait on it.

---

## 1. Why, and what is actually new

The September probe (`experiments/results/parametric_probe.csv`,
`parametric_mcadams.csv`) measured on-manifold transforms on **20 speakers of
the superseded base** with a **purifier v3 + DDIM** attacker. `FRAME_v6`'s base
rule forbids quoting any of it as a number. So the three figures that motivate
this whole phase — 0.55 held-out protection at warp 0.88, survival loss 0.019,
the 58 % / UTMOS 2.41 frontier point — are **not currently quotable**, and
`NEXT_STEPS` §3 quotes them anyway. That is the first thing this step fixes.

Three conditions the probe never had:

1. **The current evaluation base.** test-clean, 29 cross-session speakers,
   purifier v7 DDIM, the thresholds in `THREAT_MODEL` §1 (WavLM-SV 0.8926,
   ECAPA 0.3358, d-vector 0.7029), calibrated through each channel.
2. **The held-out-encoder condition, named as such.** `THREAT_MODEL` §3 defines
   matched and held-out and the probe occupies neither explicitly. For a
   *transform* there is no crafting ensemble, so "held-out" has a different and
   weaker meaning that §2.3 below states precisely rather than borrowing the
   additive system's label.
3. **PhonePuRe stage 1 + 2**, the strongest attacker in 𝓡₇ and the one that
   defeated the additive method on the d-vector (§29: 38 / 27 / 71 % clone
   success). A phoneme-conditioned restorer cannot know the original formants,
   so this is the genuinely open question and the reason this step is worth
   12–17 GPU-h even if Step 0 fails.

**What is not new, and is not claimed as new:** formant warping as a privacy
transform is McAdams/VoicePrivacy B2 (Patino et al., arXiv 2011.01130) and
ClearMask. The contribution of this step is the **evaluation under
restoration**, which ClearMask does not do (verified: zero mentions), on a base
with calibrated cross-session thresholds.

---

## 2. What is run

Eval-only; **no crafting**. Transform, then score. One GPU job at a time.

### 2.1 Arms

`t_mcadams` at α ∈ {1.00, 0.95, 0.90, 0.85, 0.80}, where **α = 1.00 is the LPC
analysis–resynthesis control** — the same addition Step 0's prereg makes, for
the same reason: without it, protection at α = 0.80 cannot be separated from
damage done by the vocoder round trip. `t_tilt` at −6 dB is carried as the
0 %-protection reference (§8.2's existence proof).

Reference arms, already on disk, re-scored through the identical channels so
every comparison is paired per speaker: `plain_m30_f15` and `full_m30_f15`
(additive, and the EOT arm `THREAT_MODEL` §3 labels the matched-encoder upper
bound).

### 2.2 Channels — the eight configurations of `THREAT_MODEL` §6a

| # | channel |
|---|---|
| 1 | identity (no purifier) |
| 2 | purifier v7, DDIM, t = 50 |
| 3 | purifier v7, DDIM, t = 200 |
| 4 | DiffWave-LibriSpeech (De-AntiFake ckpt), t = 1 |
| 5 | DiffWave-LibriSpeech, t = 3 |
| 6 | DiffWave-SC09, t = 1 |
| 7 | DiffWave-SC09, t = 3 |
| 8 | **PhonePuRe stage 1 + 2** |

Channel 1 matters more here than for the additive system: a transform's
protection should be largely channel-independent if the hypothesis is right,
and channel 1 is where that is cleanest.

### 2.3 "Held-out" for a transform — the honest definition

A transform is not optimised against anything, so it has no held-out encoder in
the additive sense. Two separate things are therefore reported and never
conflated:

* **Transfer across encoders.** Protection on each of the three encoders
  independently. The hypothesis predicts this is roughly flat; the additive
  system's defining failure is that it is not (12–14 % against 75–92 %, §22b).
  **This is the primary comparison** and it is a comparison of *spreads*, not
  of single cells.
* **The cloner's own conditioner.** DiffVC conditions on a GE2E network, the
  d-vector's family. So a d-vector result is simultaneously the most relevant
  cell for end-to-end cloning and, per §42h, the **least trustworthy verifier**
  (minDCF 0.39–0.57). Both facts are printed in the same table header. A
  protection win on the d-vector alone is weakened by the same instrument
  argument that weakens the WavLM-SV claims — the held-out condition and the
  trustworthy-verifier condition point at different encoders and no column
  satisfies both.

### 2.4 Quality

`eval_quality_nr.py`: UTMOS, NISQA, Whisper WER against Whisper's transcript of
the same 4 s crop. Reference-based PESQ/STOI are recorded but **no decision
reads them** — §8 established they score any voice change as damage.

### 2.5 The instrument gate (§58's rule, non-negotiable)

`t_mcadams` through the v7 channel at 29 speakers is a new measurement path.
Before any verdict is read:

* The path is gated through `experiments/check_direction_sanity.py`, per
  speaker, and `gate_passed` is recorded on every row.
* **Known cohort hazard:** speakers **121** and **260** carry digital silence
  (8637 and 8794 consecutive zero samples), which §58b showed puts a
  Kaldi-fbank front end on a knife-edge. LPC analysis on a frame of exact
  zeros is a second instance of the same hazard — the autocorrelation matrix is
  singular. The implementation must handle it explicitly and the handling is
  reported, not patched silently.
* A positive control that must pass before a verdict: the α = 1.00 arm must
  reproduce the clean embeddings to within the measured vocoder round-trip
  cost, and the `tilt` arm must return ~0 % protection. If `tilt` protects,
  the harness is wrong.

---

## 3. Endpoint and decision rule

**Primary endpoint.** Per (α, encoder, channel) cell: mean paired per-speaker
protection margin **τ − cos(E(G(R(x_p))), E(x_enrol))**, n = 29, BCa 95 % CI
(4000 resamples), Wilcoxon signed-rank, **Holm within channel** across the
α grid. Positive = protected. The continuous margin is primary and binary rates
are secondary, per `THREAT_MODEL` §6 — binary rates discard most of the
evidence and move by tens of points between identical runs (§31b).

**The comparison that decides the branch.** Against the additive EOT arm on the
same 29 speakers and the same channel, paired per speaker:

* **Spread across encoders**, max − min protection rate, for warp against
  additive. The hypothesis predicts warp's spread is materially smaller.
* **The d-vector cell under PhonePuRe**, where the additive system's clone
  success is 71 % (§29).

**Pre-specified thresholds.** "Materially exceeds" = the paired difference's
95 % CI excludes zero **and** the point estimate is ≥ 0.05 in margin units.
"Collapses under PhonePuRe" = the margin's CI includes zero in the d-vector
cell at channel 8.

### Outcome → action

* **2-A — transfer holds and survives the refiner.** Warp's encoder spread is
  materially smaller than additive's, **and** the d-vector margin at channel 8
  does not collapse.
  → The strongest input to GATE 3. Step 2 becomes fundable, subject to Step 0
  returning 1-A, measured headroom, and the GPU arithmetic in `PLAN_V4` §3.
* **2-B — transfer holds, the refiner defeats it.** Spread is smaller but the
  channel-8 d-vector margin collapses.
  → The transform fails against the strongest attacker in 𝓡₇ — the same place
  the additive method fails (§29). **Step 2 is not funded.** This is a
  reportable second measured ceiling and joins the fallback paper: a learned
  refiner defeats both an additive perturbation and an on-manifold transform,
  which is a stronger statement about the paradigm than either alone.
* **2-C — transfer does not reproduce.** Warp's encoder spread is not materially
  smaller than additive's on the current base.
  → Report as a **failed replication of our own probe**, with the base
  difference named (20 → 29 speakers, v3 → v7, pooled → cross-session
  thresholds). The base rule exists because this can happen. Step 2 is not
  funded. §45 is the precedent for how to write a directionally-reproduced,
  underpowered result without overclaiming.
* **2-D — the instrument gate fails.** The path is rejected by
  `check_direction_sanity.py`, or the α = 1.00 / `tilt` positive controls fail.
  → Fix the instrument, re-measure **once**, label it a fixed-instrument
  re-measurement (§57's terminology). This is the only sanctioned re-run in
  `PLAN_V4`.

---

## 4. Binding constraints

* **No re-run to obtain a different answer.** The α grid, the channel set and
  the thresholds above are fixed. A cell may not be dropped after the fact and
  no channel may be preferred over another — picking the agreeing one is the
  §25 failure mode.
* **Every reported claim names its condition**: which encoder, which of the
  eight channels, which α, and whether the encoder is the cloner's own
  conditioner. A sentence without those is wrong by construction.
* **No number from §1–§20 or from the September probe appears in the paper.**
  Base rule. The probe's figures appear in this document as motivation and
  nowhere else.
* **minDCF in every table header**, per `FRAME_v7` §6 and council §A.3, and
  WavLM-SV is called an embedding probe rather than a verifier wherever it
  carries an identity claim.
* **The quality floors are external and can fail**, exactly as
  `THREAT_MODEL` §5 rewrote them: WER ≤ +3 points over clean, UTMOS ≥ 3.0,
  NISQA ≥ 3.0. §8 measured McAdams α = 0.80 at UTMOS 2.41, which **fails the
  UTMOS floor by 0.59**. If that reproduces, the protective end of the warp grid
  does not meet the project's own stated quality floor, and that must be written
  in the result rather than discovered at GATE 4.
* Step 0's listening frontier, when it arrives, is attached to this step's
  tables as the perceptual condition. Until then no cell here is described as
  acceptable to publish.

---

## 5. Cost

12–17 GPU-h, split as `PLAN_V4` §3 allocates: 8–10 for the warp grid across
channels 1–7, 3–5 for the PhonePuRe rows, 1–2 for quality. One job at a time.
Must report by **2026-11-13**.
