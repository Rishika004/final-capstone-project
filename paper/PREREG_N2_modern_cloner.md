# Pre-registration — N2, the modern-cloner column

**Written 2026-11-12, before the run.** `paper/DEFENSE_BRIEF.md` §I.3 schedules
this as N2 and reads the result as the third stated limit or as an abstract
sentence. Evaluation only: no crafting, no re-optimisation, no new protection.

---

## 1. The question, and why it is the one a reviewer asks first

The narrow claim (DEFENSE_BRIEF §I.1) is scoped to **DiffVC**, a 2021
diffusion voice-conversion model that clones by conditioning on a separate
speaker encoder (Resemblyzer d-vector). Our protection is optimised against
speaker encoders, so a cloner built on one is the favourable case by
construction.

Zero-shot TTS systems in current use do not work that way. **F5-TTS**
(flow matching, Chen et al. 2024) has **no speaker-encoder module at all**: the
reference clip is fed to the model as an in-context audio prompt and the voice
is inferred from it directly. There is therefore no embedding for our
perturbation to have been optimised against, even indirectly.

So this is a genuine out-of-distribution test of the protection, and it is the
one question a reviewer will raise against a DiffVC-only defence paper. Its
answer is useful in **both** directions, which is why it is worth 4–8 GPU-h
before any writing starts.

## 2. What is run

### 2.1 Arms (the reference audio the attacker publishes)

Files already on disk in `experiments/results/purifier_aware/`, 29 speakers
each, current base:

| arm | meaning |
|---|---|
| `unprotected` | the clean 4-s crop; **the sanity gate, see §4** |
| `plain_tc_m30_seg` | crafting WITHOUT the purifier in the loop, same budget |
| `fullnsT_tc_m30_seg` | **the shipping recipe** |

No other arm is scored. Adding arms after seeing the result is the failure mode
§48b records.

### 2.2 Channels (what the attacker does to the reference first)

`none`, `ours` (mel DiffPure v7 = `checkpoints/purifier_v7.pt`, DDIM, t = 200),
`diffwave` (`--dw-t 3`) — exactly the three channels of
`cloning_tc_m30_nodvec.csv`, built by the same `build_purifiers()` call with
the same `--ours-ckpt`. PhonePuRe is **not** in this run; it needs its own
pre-purified inputs per arm and is deferred to N4.

3 arms × 3 channels × 29 speakers = **261 generations**.

### 2.3 Protocol, held identical to the DiffVC experiment

* **Speakers:** the 29 `CROSS_SESSION_SPEAKERS`, via the same shared selector
  (`reference_items`, `cross_session_split`) with `--cross-session`,
  `--ref-utt 0 --enroll-utt 0 --spk-offset 0 --n-files 29`, i.e. byte-identical
  arguments to the runs that produced `cloning_tc_m30_*.csv`.
* **Reference** = the victim's published clip, purified by the channel. This is
  where protection lives; it is never applied to the content.
* **Enrolment** = a clean utterance from a **different chapter**, as in
  `exp_cloning_protection.py`.
* **Thresholds:** the cross-session clean-channel EER thresholds already in
  use — WavLM-SV 0.8926, ECAPA 0.3358, d-vector 0.7029 — plus the ECAPA 0.25
  field convention reported alongside. Rationale for not varying τ with the
  attacker's purifier is unchanged (`exp_cloning_protection.py` docstring).
* **Encoders:** the same three. The d-vector is no longer the cloner's own
  conditioner here, and that is the point of the experiment; it stays in the
  table because dropping it would be outcome selection.

### 2.4 The one protocol difference, stated because it cannot be avoided

DiffVC is voice **conversion**: content comes from the attacker's own waveform.
F5-TTS is **TTS**: content comes from text. The substitution is

    DiffVC:  clone = VC(source_wav   , Purify(reference))
    F5-TTS:  clone = TTS(gen_text    , Purify(reference), ref_text)

* `gen_text` = Whisper's transcript of the **source speaker's** clean 4-s crop,
  i.e. the same source-speaker rotation (`spks[(i+1) % n]`) supplies the words.
  Held fixed across arms and channels for a given victim.
* `ref_text` = Whisper's transcript of the victim's **clean** crop, held fixed
  across arms. Giving the attacker the true words of a protected clip is the
  **attacker-favourable** choice: it removes any protection that works only by
  confusing the attacker's ASR. We are not entitled to that effect and do not
  want it in the number.
* Transcripts are cached once (`experiments/results/f5_prompts.csv`) so every
  arm sees byte-identical text.

### 2.5 Sampling noise

F5-TTS is a flow-matching sampler and draws noise. Seeded per
(speaker, arm, channel) by the same CRC mixing `--clone-seed` uses, so each
generation is reproducible on its own and independent of loop order. One seed,
and §31's rule therefore binds: **the primary endpoint is continuous; binary
rates are reported with their n and never as a point estimate in a thin cell.**

## 3. Primary endpoint and analysis, fixed here

**Primary:** the paired per-speaker **margin** `cos(E(clone), E(enrolment)) − τ`,
shipping arm **minus** unprotected, within channel and encoder, n = 29,
paired t on the differences with BCa bootstrap CI. Nine primary cells
(3 encoders × 3 channels), **Holm within the family of nine**.

**Secondary, pre-specified:** the same contrast for `plain_tc_m30_seg` (this is
the N1 contrast seen through a second cloner); clone success rates at the EER
thresholds and at ECAPA 0.25, with n stated; the unprotected margin itself,
which is §4's gate.

**Direction:** a protection that works makes the margin **more negative**. A
positive difference means the protection helped the attacker and is reported as
such.

## 4. The gate — and it fires before any verdict is read

F5-TTS must be a **working cloner on unprotected audio** in this harness, or
the protected cells carry no information (the failure mode
`exp_cloning_protection.py`'s pre-filter exists to prevent).

> **GATE:** on the `unprotected` arm, `none` channel, clone success must hold on
> **≥ 15 of 29 speakers (≥ 50 %) on at least one encoder**, and the mean
> unprotected margin must be positive on that encoder.

If the gate fails, the run is reported as **"the harness did not establish a
working modern cloner"** and no protection verdict is read from it. Not a
licence to read point estimates — §48 is the precedent. A failed gate is
debugged as a harness problem (prompt length, sample rate, ref_text) and re-run
**once**; a second failure ends N2 and the paper keeps DiffVC-scoped wording.

## 5. Outcome branches, fixed before the data

* **N2-A — protection transfers.** The shipping arm's margin is significantly
  below unprotected (Holm-corrected) on **ECAPA and at least one other
  encoder**, in the two purifier channels.
  → The abstract may say the protection transfers to a modern zero-shot cloner,
  with F5-TTS named and the encoder-free architecture noted. Strongest
  available outcome.
* **N2-B — partial transfer.** Significant on some cells but not on ECAPA, or
  only in the `none` channel.
  → Reported as a per-encoder result in the limits section. The claim keeps
  DiffVC-scoped wording; the abstract does not generalise.
* **N2-C — no transfer.** No significant reduction in any purifier channel.
  → **The third stated limit** (DEFENSE_BRIEF §I.4): the protection is
  specific to cloners that read identity through a speaker encoder. This is a
  real and publishable limit, it does not contradict the narrow claim, and it
  is the honest answer to the reviewer's question. It also becomes the single
  strongest motivation for the on-manifold route in future work.
* **N2-D — protection helps the attacker.** Margin significantly *higher* than
  unprotected anywhere.
  → Reported. No mechanism is asserted without a follow-up.
* **N2-E — gate failure.** §4.

**Pre-committed prediction (so it can be scored).** I expect **N2-C or N2-B**,
around 60 / 25, with N2-A at ~15 %. Reasoning: the perturbation was optimised
through three speaker encoders' gradients, and F5-TTS has none; the mechanism
that gave 12–14 % on crafted-against encoders has no counterpart here. The
~15 % on N2-A allows for the possibility that the perturbation damages the
acoustic prompt in a way any voice-prompted model inherits — which, if it
happens, is a stronger result than the DiffVC column and must then be checked
against the quality numbers before being believed.

## 6. What is NOT claimed from this run

* Not that F5-TTS is *the* modern attacker — it is one, chosen for being
  encoder-free and openly available.
* Not anything about intelligibility or naturalness of the clones; no quality
  metric is scored here (that is N3's and §40c's territory).
* Not a comparison to published F5-TTS cloning rates: our thresholds, crop
  length and enrolment protocol are ours.
* Not a replication of the DiffVC numbers. Different cloner, different content
  pathway; the columns sit side by side and are never averaged.

## 7. Files

| what | path |
|---|---|
| prompts + transcript cache | `experiments/prep_f5_prompts.py` → `results/f5_prompts.csv` |
| purified references on disk | `results/f5_refs/<name>__<arm>__<channel>.wav` |
| generation (isolated venv) | `experiments/f5_clone.py`, `external/f5tts_venv` |
| scoring | `experiments/score_f5_clones.py` → `results/cloning_f5tts.csv` |
| analysis | `experiments/analyse_f5.py` |

`cloning_f5tts.csv` carries the same schema as `cloning_tc_m30_seg_s7.csv` plus
a `cloner` column, so the two cloners can be put in one table without a join
that could silently mix them.
