# Making the evaluation black-box

*Written 2026-09-17, after the council review. The finding that forces this:
`checkpoints/diffvc/spk_encoder/pretrained.pt` and Resemblyzer's `pretrained.pt`
have the same MD5 (`a2e2a0bf4cc3ac59053d3fe4d67b0c4b`, 17,090,379 bytes). **DiffVC's
speaker encoder is not "the same family as" one of our crafting encoders. It is
the same file.** Our cloning result is therefore a white-box attack on DiffVC's
conditioning pathway, not a transfer result.*

## What "black-box" has to mean here

Three separations. We currently have one.

| | separation | status |
|---|---|---|
| **A** | the encoder that MEASURES protection was not used to CRAFT it | ❌ none — `build_encoders()` returns the same three everywhere |
| **B** | the encoder the CLONER conditions on was not used to craft | ❌ worse than absent — it is byte-identical to one of ours |
| **C** | the PURIFIER was not trained by us | ✅ done — two public DiffWave checkpoints |

Full query-only black-box (no gradients at all) is a different paper. The
achievable and conventional target is **transfer / gray-box**: craft on a surrogate
ensemble, evaluate on targets never used in crafting. That is what AntiFake does
with four encoders, and it is the standard this work should meet.

---

## The change that does the most for the least

**Drop Resemblyzer from the CRAFTING ensemble.**

Because DiffVC conditions on exactly that checkpoint, removing it from crafting
converts the existing DiffVC experiment from white-box to a genuine held-out
transfer test **with no new cloning model required**. The same saved audio, the
same pipeline, one changed flag — and the result becomes the thing we wanted to
claim all along.

It also gives Level A for free: Resemblyzer becomes an evaluation-only encoder.

### Implementation note — do NOT do this with `--weights 1,1,0`

`craft()` computes `logsumexp((c - thr) * w * tau_lse) / tau_lse`. With `w = 0`
that encoder's term becomes `exp(0) = 1`, which does not remove it — it adds a
constant floor to the logsumexp that dominates whenever the other terms go
negative (i.e. exactly when protection is succeeding). **A zero weight silently
changes the objective instead of dropping the encoder.**

Add an `--encoders` flag that subsets the list passed to `craft()`, and make
`build_encoders()` take a selection. Keep `build_encoders()` as the single source
of truth so crafting and evaluation cannot drift apart — but give it two named
sets, `CRAFT_SET` and `EVAL_SET`, and assert they are disjoint where the
experiment claims transfer.

---

## Changes, in order

### 1. Split crafting and evaluation encoders ⭐ highest value
* `build_encoders(which=...)`; `CRAFT = {WavLM-SV, ECAPA}`, `EVAL = {Resemblyzer,
  x-vector}`.
* Add an assertion in the transfer experiments that the two sets are disjoint, so
  this cannot silently regress.
* **Cost:** ~1 h coding, then re-craft 20 speakers x 4 variants (~2 h GPU).

### 2. Add x-vector as a second held-out encoder
* SpeechBrain `spkrec-xvect-voxceleb` — a TDNN, architecturally unlike both
  crafting encoders.
* **Bonus: it is what De-AntiFake (xSVA) and VocalBridge use**, so it buys
  comparability as well as independence.
* **Cost:** ~30 min.

### 3. Re-run the cloning experiment — now a real transfer test
DiffVC is unchanged; what changes is that its encoder is no longer ours. Report
the white-box and transfer numbers **side by side** — the contrast is itself a
result, and it is the AntiFake ablation the reviewers said we would be asked for.
* **Cost:** ~1 h GPU.

### 4. A cloner that does not use GE2E at all
Even with (1), DiffVC is a single target. Ranked by independence:
* **kNN-VC** — no speaker encoder at all (WavLM feature matching). Maximum
  independence; appears in no baseline paper, so report it as a robustness check
  rather than a comparability number.
* **OpenVoice V2** — its own tone-colour extractor; used by De-AntiFake. **Note the
  official S3 bucket is dead; use the HuggingFace mirrors.**
* **YourTTS** — H/ASP ResNet encoder; used by AntiFake and De-AntiFake. A working
  but unvalidated path already exists in `cloning_attack.py` mode 2.
* **Cost:** 2-3 h each.

### 5. Fixes the council found that are independent of black-boxness
* **Pre-filter must condition on purifier.** Verified: the union filter reports the
  matched-purifier protected rate as 15/21/11 %; per-channel it is **40/50/14 %**.
  One line in `exp_cloning_protection.py`.
* **Stop reporting "best arm."** A min over four arms at n=5-14 is a biased
  estimator. Report the mean across arms, or each arm.
* **Run DiffWave at t=3 and t=10, not t=1.** At t=1 it is nearly a no-op —
  unprotected clone similarity 0.327 with it vs 0.346 without. "Survives a
  published purifier" currently means "survives one step of 200." We already ran
  t=3 for the ASV arm.
* **Recalibrate thresholds cross-session.** `per_speaker` takes the first four
  files per speaker, which for held-out speakers all come from the same LibriSpeech
  chapter — one session, one channel. The resulting EERs (ECAPA 0.03 %) are not
  credible operating points, and the threshold lands almost exactly on the
  unprotected clone's mean. Draw enrolment from a different chapter.
* **Lead with cos > 0.25 on ECAPA.** It is the field convention, our unprotected
  rate at it is 95 % (a genuinely strong attack, which helps us), and it removes
  the "you picked a threshold that made protection easy" objection.

---

## What each level would let us claim

| after | claim |
|---|---|
| nothing (today) | "protection defeats a cloner conditioned on an encoder we attacked directly" |
| **1 + 3** | "protection transfers to a cloner whose speaker encoder we never touched" |
| 1 + 3 + 4 | "…and to a second cloner with a different encoder architecture" |
| + 5 | numbers that survive a methods reviewer |

**Step 1+3 is the one that changes the paper.** Everything else is corroboration.

---

## The risk worth naming before starting

AntiFake's Table 4 is the prior here: protection crafted against one encoder
achieves **99.6-100 %** against the synthesizer built on that encoder and
**0-20.6 %** against the other four. Their GE2E row is our situation exactly --
100 % on the synthesizer that uses GE2E, 1.2-20.6 % on those that do not. It is
entirely possible that dropping Resemblyzer collapses our cloning result.

*(Corrected 2026-09-17. This document previously said "99.6 % matched, 0 %
mismatched". Both halves were wrong: 99.6 % is one specific cell (AdaIN->AdaptVC)
while the other three diagonals are 100 %, and the off-diagonal floor is 0-20.6 %,
not 0. The axis is also synthesizers, not encoders. Verified against the published
PDF, cybersecurity.seas.wustl.edu/paper/AntiFakeCCS23.pdf.)*

That outcome is still publishable, and arguably more useful than the current one:
*protection against diffusion-purified cloning is encoder-matched, and the field
should stop reporting single-encoder transfer numbers as if they generalise.* It
would also explain why our purifier-mismatch effect lived only in ASV.

What is not acceptable is shipping the white-box number described as transfer.
That is the one outcome the checksum rules out.
