# Pre-registration — the two controls council v5 says decide the paper

**Written 2026-10-07, before either control is run.** Both were identified
independently by two reviewers, both are cheap, and **both can falsify a
headline**. Their readings are fixed here because deciding after seeing the
output is the §25 failure mode the project keeps catching itself in.

---

## Control A — the held-out C2 grid

### A.1 Why

Every headline number in the paper is crafted with `--craft-encoders all`
(`run_rebase_mismatch_m30_seg.sh:70`), which includes the GE2E/Resemblyzer
network that **is DiffVC's own speaker conditioner**. `THREAT_MODEL` §3 defines
a second condition, "held-out (realistic) — stronger than anything published",
and no reported result occupies it. So the d-vector reversal — C2's single
most transferable finding (§45e) — has never been measured in the condition
where the d-vector is *not* in the crafting ensemble.

This is not a relabelling problem. The open question is whether the reversal is
a property of the constraint or an artefact of optimising against the very
encoder that then fails.

### A.2 What is run

**No crafting.** `eval_saved.py` re-scores wavs already on disk:
`full_m30_f15_nodvec` and `fullnsT_m30_f15_nodvec`, 29 speakers each, verified
present, through the four foreign channels (DiffWave LibriSpeech and SC09,
t ∈ {1,3}) against the existing test-clean calibrations.

**Declared limitation, stated now:** these are the **pooled-partials m30_f15**
arms, not the `--seg-partials` shipping recipe. `full_m30_seg_nodvec` does not
exist and would cost one crafting arm. So this control answers "does the
reversal survive holding the encoder out, at the non-seg budget", not "…at the
shipping budget". If the control contradicts the matched result, the seg
version becomes mandatory; if it agrees, the non-seg evidence is sufficient and
the seg arm is not worth a crafting run.

### A.3 Endpoint

Identical in form to §45: mean paired per-speaker Δ = (fullnsT − full) in
`cos_after_purify − threshold`, n = 29, BCa 95 % CI, Wilcoxon, **Holm over the
12 cells of this control** (3 encoders × 2 purifiers × 2 timesteps).
**Negative = the constraint helps. Positive = the reversal.**

### A.4 Outcome → action

* **A-1 — the reversal survives held-out.** The d-vector cells at t = 1 are
  **positive** on both purifiers.
  → C2's reversal is a property of the constraint, not of encoder overlap.
  Report the held-out cells as the **primary** C2 result and the matched grid
  as the upper bound. This strengthens the paper and removes B1 as a defect.
* **A-2 — the reversal disappears or inverts held-out.** Either t = 1
  d-vector cell is non-positive.
  → **This is the paper's most important sentence.** The reversal is then
  entangled with crafting against the cloner's own conditioner, C2 is demoted
  to a matched-encoder observation, and §45e may no longer be called "the most
  transferable thing in the paper". No re-run at another budget to recover it.
* **A-3 — underpowered.** Signs match A-1 but no cell resolves.
  → Report as directional, exactly as §45 was, and say the held-out condition
  is noisier at the non-seg budget.

---

## Control B — the non-identity task control

### B.1 Why

C1's claim is that a **speaker encoder's identity direction** is
purifier-hostile. Every existing control (band-matched, `pca_span`, white
noise) tests a *spectral* alternative. None tests the **task** alternative:

> "A speech denoiser removes the directions a differentiable speech network is
> most sensitive to" is a generic fact about neural Jacobians and smooth-prior
> denoisers. "…that a *speaker* encoder reads" is the paper's claim.

They are currently indistinguishable, and the word "speaker" in the title rests
on the difference. Two reviewers found this independently.

### B.2 What is run

α for the top gradient direction of networks with **no speaker identity to
read**, on the same purifier, same 29 speakers, same `--rms`, same seeds,
measured by the same `survival()` path:

1. **CAM++ (untrained)** — identical architecture, randomly initialised, no
   trained weights. Verified before this document: speaker separation
   **+0.038** (against +0.614 trained), so it carries essentially no identity,
   while its gradient is finite and non-zero.
2. **A content/ASR direction** if one can be wired cheaply on the existing
   path; recorded as not-run if it cannot, rather than silently dropped.

### B.3 Outcome → action

Let α_untrained be the untrained encoder's mean α at t = 50, against trained
encoders at 0.388–0.631 and `random` at 0.714.

* **B-1 — α_untrained lands at or below the trained band (≲ 0.63).**
  → **C1's title is wrong.** The effect is about gradients of deep speech
  networks through a denoiser, not about speaker identity. "Speaker" comes out
  of the title and the claim is restated as a property of neural Jacobians.
  C1 is demoted; the paper leads with the measurement-distortion contribution.
* **B-2 — α_untrained sits near or above `random` (≳ 0.70).**
  → C1 survives and is **strengthened**: an untrained network of the same
  architecture is *not* purifier-hostile, so the effect tracks learned identity
  rather than architecture or gradient roughness. This is the control that
  earns the word "speaker", and it should be reported in the C1 figure.
* **B-3 — intermediate (0.63–0.70).**
  → Report the number and weaken the claim to "learned speaker encoders are
  more purifier-hostile than an untrained network of the same architecture, by
  X", with no mechanism language.

### B.4 Binding on both controls

* Neither control may be re-run at other settings to obtain a different
  verdict.
* Control B is **not** a test of C1's level clause against matched random —
  that already cleared (§37, §43, §46b). It tests whether the word "speaker" is
  doing work.
* An untrained network is a weak proxy for "no identity": it also has untrained
  *features*. If B-1 fires, the honest statement is the disjunction "identity
  or architecture", not "architecture".
