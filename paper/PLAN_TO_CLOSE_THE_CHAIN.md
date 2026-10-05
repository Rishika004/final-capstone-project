> **RETIRED 2026-10-04.** This plan serves the defence framing that RESULTS §29
> (full PhonePuRe recovers the speaker) and §30 (BPDA ≡ exact) removed. The live
> framing is `FRAME_v6.md`; the live gap list is `REVIEW_COUNCIL_2026-10-04.md`.
> Surviving items: P5 second cloner → council #16; P6 perceptual → council #6.

# Plan to close the chain

**Written 2026-09-19, after the deployment-setting result.** Supersedes the
ordering in `ROADMAP_TO_PUBLICATION.md`; that file's individual entries remain
valid. Every citation here was verified against the primary source in this
session — where a claim rests on a paper saying something, the section is named.

---

## The chain, and where it currently breaks

> audio → our protection → **purifier fails** → **transfers to a different
> encoder** → a De-AntiFake-class attacker clones → **cloning fails completely**

| # | Link | Status | Evidence |
|---|---|---|---|
| 1 | protection layer | ✅ | — |
| 2 | purifier fails (protection survives) | ✅ **strong** | α 0.87 vs 0.74; 86–107 % retention on 2 foreign purifiers, 20/20, p=1.9e-06 |
| 3 | transfers to a different encoder | ⛔ **the break** | see below — and it is TWO separate problems, not one |
| 4 | De-AntiFake-class attacker | ⚠️ partial | DiffVC is one of their 6 VC models; our DiffWave t=3 ≈ AudioPure, their strongest baseline. **PhonePuRe itself never run.** |
| 5 | cloning fails completely | ⚠️ 2 of 3 encoders | deployment + foreign purifier: **ECAPA 0 %, WavLM 13 %**, d-vector 50 % |

### Link 3 is two problems that look like one

| | deployment (all encoders crafted) | strict (cloner's encoder held out) |
|---|---|---|
| WavLM-SV | **0 %** | 21 % |
| ECAPA | **0 %** | 19 % |
| Resemblyzer | **35 %** | 75 % |

* **3a — the proxy gap.** Resemblyzer is at 35 % *while being crafted against*.
  That is not encoder transfer failing. We attack an approximation of it.
* **3b — encoder transfer.** The strict column. A genuinely unseen encoder is
  barely protected.

Fix 3a and link 5 closes in the deployment setting. Fix 3b and it closes in the
strict setting too. **3a is much cheaper and must go first.**

---

## P0 — Attack the real Resemblyzer path, not a rebuilt one ⛔ HIGHEST PRIORITY
**Effort:** ~4 h code, ~5 h GPU · **Closes:** link 3a, and therefore link 5 in deployment

### The defect
`multi_encoders.py:DvecEnc.embed()` feeds a torch mel front-end straight into
`self.dvec(m)` — one forward pass over the whole clip, producing a pooled
embedding. The **evaluation** path, `real_embed_dvec()`, calls Resemblyzer's own
`preprocess_wav()` (normalisation + voice-activity trimming) and then
`embed_utterance()`, which **splits the clip into overlapping partials, embeds
each, and averages them**.

So we optimise against a different function than the one that scores us. Hence
35 % where the other two crafted encoders reach 0 %.

### The fix, and it is the one the baseline already prescribes
AntiFake §C2 identified exactly this and fixed it. Verbatim from the paper:

> *"directly deviating the final embedding is susceptible to local optima, as the
> impact of a partial embedding not close to the target is mitigated by other
> partials that may have attained better optimal points."*

Their solution is a **weighted segment-based raw embedding deviation**. Adopting
it is not a novelty claim — it is implementing the baseline correctly.

### Concretely
1. Add a differentiable partial-splitting wrapper in `multi_encoders.py` that
   mirrors `embed_utterance`'s windowing (Resemblyzer's `compute_partial_slices`).
2. In `craft()`'s `ens_cos()`, score **every partial** and combine with a weighted
   loss that pushes the *worst* partial rather than the mean, so a good segment
   cannot mask a bad one.
3. Do the same for ECAPA and WavLM-SV — both also pool over time, so both are
   subject to the same local-optimum argument even though they currently do
   better.
4. Re-run the deployment arm and check the Resemblyzer column moves toward 0.

**Citation to carry:** AntiFake (Yu et al., CCS 2023) §C2. Frame as *"we adopt the
segment-level formulation of [AntiFake], which the pooled formulation we started
from is known to under-optimise."*

---

## P1 — Enlarge the crafting ensemble ⛔
**Effort:** ~2 h code, ~5 h GPU · **Closes:** link 3b (the strict column)

Add x-vector (SpeechBrain `spkrec-xvect-voxceleb`), CAM++, ERes2Net, ResNet-34.
Resemblyzer stays held out in the strict arm, so the strict number remains a
genuine transfer measurement.

**Evidence it works, verified:** AntiFake Table 4. Crafting against **one**
encoder gives 0–20.6 % protection on synthesizers built around a *different*
encoder; crafting against **all four** gives 97–100 %, including on ElevenLabs,
which §6.1 calls *"a completely black-box system"*.

**Temper the expectation with their own concession.** AntiFake §7 states they
*"did not identify any voice feature that was generalizable across all encoders"*.
So the mechanism is **coverage, not generalisation** — you are raising the chance
that an unknown encoder resembles one in your set, not finding a universal
feature. Say it that way in the paper and the claim is defensible; say
"generalises across encoders" and it is not.

**Cheap diagnostic first (~1 h):** add a GE2E model with Resemblyzer's
architecture but different weights. Transfers to it but not Resemblyzer → the
barrier is weights. Transfers to neither → the barrier is architecture. That one
result tells you whether to buy ensemble *size* or ensemble *diversity*.

---

## P2 — Build the PhonePuRe attacker, and test the assumption it rests on ⛔
**Effort:** ~1–2 days · **Closes:** link 4 · **This is the paper's biggest opportunity**

### Why this is not just "a stronger baseline"
De-AntiFake's stage 2 (score-based refinement, NCSN++ over complex STFT) is
**trained only on (clean, purified-clean) pairs. It never sees a protected sample
during training.** Their stated justification is that after stage 1 the
purified-protected and purified-clean distributions have collapsed together, so a
mapping learned on one transfers to the other.

**Our entire contribution is a perturbation that survives stage 1.** If it
survives, those distributions do *not* collapse, and their refinement operates
out of distribution. That is a falsifiable prediction our geometry makes about
their method, and it is the strongest claim available to this paper.

### Build order (cheapest informative step first)
1. **Stage 1 only** — we effectively have this. DiffWave, T_pur = 3. Their Table 2
   ablation shows "w/o Refinement" = 0.597/0.646, numerically identical to
   AudioPure. So **our current foreign purifier already equals their strongest
   baseline.** Say so in the paper; it is a free strengthening of the existing
   robustness claim.
2. **Stage 2** — NCSN++ + OUSDE. Their published hyperparameters: γ=1.5,
   σ_min=0.05, σ_max=0.5, t_ε=0.03, T=1, N=30 reverse steps, one Langevin
   corrector, SNR r=0.4; STFT window 510, hop 128, sqrt-Hann. Code at
   `github.com/cyberrrange/De-AntiFake`.
3. **The measurement that matters:** after stage 1, compare the distance between
   purified-clean and purified-*ours* against purified-clean and
   purified-*plain-PGD*. Their assumption predicts both collapse. Ours predicts
   only the plain one does.

**Do NOT target the phoneme conditioner.** Their own ablation puts it at ~1.6
xSVA / 3.2 dSVA points, while the refinement stage existing at all is worth
~9.8/14.0. Attack the clean-pair training assumption, not the phoneme lookup.

**They concede the opening.** App. C states that adaptive attacks from the vision
literature *"could potentially be adapted for speech tasks, enabling the
construction of adaptive protection methods robust against our purification"*,
and that the only obstacle is memory — naming chunking and accelerated sampling as
the workarounds. **We already compute exact gradients through the full reverse
process with gradient checkpointing.** The blocker they name is one we solved.

---

## P3 — Re-run the verification claims on the corrected base ⛔
**Effort:** ~1 day GPU, no new code

Links 1–2 (survival, transfer, the mismatch trade) are still on `dev-clean` +
purifier v3 + same-session thresholds. The cloning links are on `test-clean` +
v7 + cross-session. **A paper cannot mix two evaluation bases**, and the
inconsistency is visible by comparing sections.

Everything now routes through one shared selector, so this is time, not work.

---

## P4 — Three sampling seeds per cell
**Effort:** ~1 h code, ~2 h GPU

DiffVC is stochastic and every cloning cell is one draw. Demonstrated this
session: regenerating one clone flipped a cell from reject to accept (0.792 vs a
0.779 threshold). Any borderline cell is currently a coin toss.

---

## P5 — A second cloning system
**Effort:** ~4 h code, ~4 h GPU

One cloner is thin against baselines that use five and six. Pick from
De-AntiFake's set so the comparison is direct: **OpenVoice V2**, **YourTTS**
(a TTS rather than a converter, uses the H/ASP encoder), or **SeedVC** — their
hardest case, where AttackVC and VoiceGuard both *failed*.

---

## P6 — Perceptual evidence
**Effort:** ~1 h (NISQA) then ~4 h (listening test)

Run **NISQA** first: it is exactly what De-AntiFake reports (objective MOS
3.36 ± 0.58), so it is a directly comparable number for one hour of work. The
loud budget sits at PESQ 2.2, which is audible degradation, and we currently have
no human evidence at all. AntiFake reports MOS 3.37–3.84.

---

## P7 — Resolve the Shallow Diffuse comparison
**Effort:** ~2 h reading

Our defence against the closest prior work rests on a sign convention we could not
verify from their text. Settle it before writing, not in rebuttal. Re-read v5.

---

## P8 — Missing citations
**Effort:** ~1 h

Tramèr et al., ICML 2020 (arXiv:2002.04599) — *"defenses against
sensitivity-based attacks actively harm a model's accuracy on invariance-based
attacks"*, a closer antecedent to our trade than Stutz. And arXiv:2607.00899 on
defender-aware adaptive attacks with gradients through the whole pipeline.

---

## Two openings neither baseline has taken

Worth knowing because they are cheap novelty, not just gap-filling.

**Multi-clip aggregation is untested by both papers.** AntiFake §7 asserts that
multiple protected samples give the attacker only limited advantage, and runs no
experiment. De-AntiFake purifies and clones each clip independently and never
considers aggregation. An attacker averaging several protected clips of one
speaker to cancel per-clip perturbation is unevaluated in this literature. It is
an attack we could run in ~3 h, and reporting it is a contribution whether it
succeeds or fails.

**PhonePuRe needs a transcript.** Its refinement stage uses Montreal Forced
Aligner, and every experiment in the paper uses LibriSpeech ground-truth text, not
ASR output. The dependency is real and unevaluated. Their load-bearing assumption
is that perturbations disturb speaker identity but not content — untested against
a content-aware protection.

---

## Order of work

| Order | Item | Cost | Why here |
|---|---|---|---|
| 1 | **P0** real Resemblyzer path | ~9 h | The only thing between us and the chain closing in deployment. Cheapest link in the chain. |
| 2 | **P1** enlarge ensemble | ~7 h | Closes the strict column; evidence it works is published |
| 3 | **P3** re-run verification on the new base | ~1 day | Must happen before writing; everything else lands on this base anyway |
| 4 | **P2** PhonePuRe + the distribution test | ~2 days | The strongest claim, but only worth building once links 3 and 5 hold |
| 5 | **P4** three seeds | ~3 h | Cheap, and we have live evidence it matters |
| 6 | **P5** second cloner | ~8 h | Required for the venue |
| 7 | **P6** NISQA, then listening test | ~5 h | Cheapest comparability win left |
| 8 | **P7 / P8** citations | ~3 h | No GPU; do while runs are going |

**The single most important sentence in this plan:** link 3 looked like one
problem and is two, and the cheaper half (P0) is an implementation defect we
inherited by not following the baseline's own prescription. Fixing it may close
the chain in the deployment setting on its own.
