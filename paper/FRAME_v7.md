# Paper frame v7 — the measurement-and-limits reframe

Written 2026-10-08. Supersedes `FRAME_v6`'s **contribution structure**;
everything else in `FRAME_v6` (base rule, claims ledger, pre-decided branches)
**still binds** and is extended, not replaced.

Sources: `REVIEW_COUNCIL_2026-10-06.md` §A (unanimous) and §D (proposed v7),
plus RESULTS §47–§52, which landed after that council and changed C2 enough
that the slot had to be refilled anyway.

**BASE RULE unchanged: new base only.** §21 onward. Nothing from §1–20 as a
number.

---

## 1. Why the structure changes

Three things forced it, in order of weight.

1. **§47 demoted the old C2.** The d-vector reversal — FRAME_v6's C2 and
   "the most transferable thing in the paper" — is an artefact of crafting
   against the d-vector, not a property of the constraint. §52 confirmed it
   from a second direction: the `recipe × encoder` interaction is p = 2e−13
   matched and p = 0.19 held out. A contribution slot opened.
2. **All five reviewers independently wanted §42h promoted** (council §A.1).
   R1 ranks it #2, R2 calls it "C0", R4 calls it #2, the AC makes it C2. It is
   measured, costs zero GPU, and is currently filed as a setup table.
3. **R4 would not sign a "protection paper".** Council §A.5: measurement and
   limits, or reject. The AC's estimate moves 30 % → 42 % on the reframe.

---

## 2. Title — PROVISIONAL, and one experiment can still break it

Council §D proposes:

> **Every Speaker Encoder's Identity Direction Is Purifier-Hostile: Encoder
> Geometry and the Measurement Limits of Purification-Robust Voice
> Protection**

**This title is not yet earned.** `PREREG_council5_controls.md` Control B tests
whether the word **speaker** is doing any work, by measuring α for an
**untrained** CAM++ — same architecture, no learned identity. Its branches were
fixed before the control was run:

* **B-2** (α_untrained ≳ 0.70, near `random`) → the effect tracks *learned
  identity*. The title is earned and C1 is strengthened.
* **B-1** (α_untrained ≲ 0.63, inside the trained band) → **the title is
  wrong.** The effect is about gradients of deep speech networks through a
  denoiser, not about speaker identity. "Speaker" comes out, C1 is demoted, and
  the paper leads with the measurement contribution.
* **B-3** (0.63–0.70) → weaken to "learned speaker encoders are more
  purifier-hostile than an untrained network of the same architecture, by X",
  no mechanism language.

**Control B is not yet run.** Until it is, the working title is used and no
abstract sentence may assert that the effect is about *speaker* identity rather
than about trained networks. This is written here so the decision is not made
under deadline.

Fallback title if B-1 fires:
*What a Speech Purifier Erases, and What That Does to the Instruments:
Measurement Limits of Purification-Robust Voice Protection.*

---

## 3. Thesis

For five speaker encoders across four architecture families, each encoder's own
identity direction survives a speech purifier **worse than a random direction
matched to it in norm and per-mel-band envelope**; and the same purification
degrades the verifiers used to *score* protection to near-trivial operating
points. So part of what this paradigm reports as its limit is a property of its
instruments, and the geometry says which encoders the instruments will mislead
you about.

---

## 4. Contributions

### C1 — the geometry (§32, §37, §43, §44, §46b; Fig. 1)

**Lead with the LEVEL, not the ordering** (council §A.2, non-negotiable).

* α below each encoder's **own band-matched** control in **10 of 10 cells**
  (5 encoders × 2 timesteps), Holm within timestep, p_holm ≤ 4.0e−6, 24–28 of
  29 speakers (`fig1_alpha.csv`).
* **Five encoders, four architecture families** — transformer, TDNN, D-TDNN,
  ResNet, GE2E LSTM — **two never in the crafting ensemble** (§46b).
* The band-matched controls land at **0.699–0.711 against isotropic random
  0.714**: giving a random direction an encoder's exact per-mel-band envelope
  buys it **nothing**, so the gap is **not spectral** (§37b). This is the
  single most load-bearing control in the paper.
* `pca_span` survives **best** (0.844): the purifier preserves the subspace
  clean speech occupies, and speaker identity points **out** of it (§37d).

**The ordering is the fragile half and is quoted only at the resolution §44
measured** — at t = 50 a three-group separation, at t = 200 only WavLM-SV
separates, x-vector and d-vector never distinguishable. Never a four-way
ranking.

**Inside C1, as its explanandum**, the d-vector reversal — with §47's
correction attached in the same breath, never later. **§53 closed §47e**: at
the shipping budget the held-out d-vector cells are −0.0075 / −0.0113, A-2 is
confirmed at both budgets, and §49e's pre-registered prediction held (mean
change 0.0019, zero sign flips). Nine of twelve held-out cells are negative
and significant; none is positive.

**[PENDING #13c α through DiffWave]** — the one structural hole: α is measured
on the mel purifier while the phenomenon lives on DiffWave.
**[PENDING Control B]** — see §2.

### C2 — the measurement is not neutral (§42h, §29, §40c) **[PROMOTED]**

Three independent ways the field's proxy scoring distorts itself, all already
measured, all zero GPU.

1. **The purifier degrades the verifier, and EER hides how much** (§42h).

   **NARROWED 2026-10-08 by the novelty check (`related_work.md` §13), which
   §42h had never had — it was a setup table until the council asked it to
   lead.** That purification costs clean verification accuracy is **already
   published, and far more dramatically than we report it**: MDD (arXiv
   2508.19180, APSIPA ASC 2025) Table 3 puts **DAP at 31.2 % bona-fide EER
   against 1.4 % unpurified**, verified against the full text. Ours moves
   5.2 % → 9.5 %. **We did not discover this, we are not the ones who measured
   it most dramatically, and we must say neither.**

   What is ours, and still worth the slot:
   * **The operating point, not the EER.** On our channels WavLM-SV's EER
     moves 5.2 % → 9.5 %, which reads as mild; minDCF at p_target = 0.01 moves
     **0.77 → 0.91**, i.e. poor to indistinguishable from the trivial system.
     EER hides it; the protection literature reports EER.
   * **In the cells a protection claim is actually made**, not as a side
     effect of proposing a defence.
   * **Per encoder, with the consequence drawn**: ECAPA is the only verifier
     worth believing at this operating point, so "protection holds on
     WavLM-SV" is the **weakest** of our three claims, not the strongest. We
     know of no protection paper that ranks its own per-encoder claims by the
     quality of the instrument.

   Required wording and the citations that must sit on it: `related_work.md`
   §13.4.
2. **ASV acceptance overstates what the cloner actually clones** (§29).
   Full PhonePuRe restores the speaker on 29/29 files by ASV, while end-to-end
   DiffVC clone success rises to 38/27/71 % — reduced, not removed. A paper
   scoring only the proxy would have reported a total defeat.
3. **Part of "protection" is damage to the clone** (§40c). The protected
   clone is 0.32 UTMOS / 0.44 NISQA less natural, halving under the purifier,
   with WER untouched.

**Why this is a contribution and not a caveat:** every one of these is a
correction other people's papers inherit, and none of them requires our
method to be good.

### C3 — the measured ceiling (§22b, §51, §29)

1. **Coverage, not generalisation** (§22b). One variable: clone success
   12–14 % on crafted-against encoders against 75–92 % on the held-out one.
2. **The gradient estimator is a small lever, not the lever** (§51). The
   pre-specified TOST at D = 0.01 **failed**; the data support equivalence only
   at **D = 0.032**, and exact gradients protect better by **0.005** in 10 of
   12 cells. Still far too small to explain prior adaptive failures, where the
   effects at stake are 0.076. Report the **bound**, never "equivalent".
3. **A learned refinement stage recovers most of what purification leaves**
   (§29), concentrated on the d-vector — C1's lowest-α encoder.

### Not a contribution

The crafting recipe. It moves to Setup as "the protection we analyse", and
**as the matched-encoder upper bound** (council B1, `THREAT_MODEL` §3).

### Cut outright

"C1 gives a criterion for choosing ensemble members" — §46 killed it
(council §A.4). Do not reintroduce it in the discussion.

---

## 5. Page budget (Interspeech, 4 + 1)

| § | content | pp | gate |
|---|---|---|---|
| 1 | Intro: purification defeats protection; our question; C1–C3 | 0.55 | — |
| 2 | Setup: threat model, corpus, disjoint calibration, encoders **with minDCF in the header**, purifiers enumerated (§6a), the protection analysed **as the matched-encoder upper bound** | 0.60 | — |
| 3 | **C1** + Fig. 1 | 0.95 | #13c, Control B |
| 4 | **C2** (promoted) + Table 1: minDCF × channel | 0.70 | — |
| 5 | **C3** + Table 2: coverage contrast, PhonePuRe two-number row, the BPDA bound in one sentence | 0.70 | — |
| 6 | Limitations and conclusion | 0.50 | — |

**Limitations has four named occupants** and they are not optional: §46
(inconclusive, two sentences, §48a), §48b's pre-registration defect, §50's
rerun term, and `THREAT_MODEL` §6b's four unmeasured capabilities.

---

## 6. Rules carried forward from FRAME_v6, unchanged

* §4's claims ledger, **plus the ten rows added 2026-10-08**.
* §6's pre-decided branches, **plus Control B's three above**.
* "Two samples, 29 + 31, read out separately and never pooled."
* "Directionally reproduced", never "replicated".
* R2's mechanism, now mandatory: **print minDCF in the column header of every
  protection table**, and call WavLM-SV an *embedding probe* rather than a
  verifier wherever it carries an identity claim (council §A.3).
