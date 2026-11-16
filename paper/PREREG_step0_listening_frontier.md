# Pre-registration — Step 0, the listening frontier

**Written 2026-10-09, before a single response is collected.**
`listening_test/responses/` is empty today; that is verifiable in git.

This is the gate `THREAT_MODEL` §5 has carried as **open** since 2026-10-05:
the only success criterion in this project that no instrument can answer.
`PLAN_V4` §4 reads its result as GATE 1. Nothing in Step 2 is funded until it
reports.

---

## 0. Why this is a frontier test and not a point test

`NEXT_STEPS` §5 asks for "a same/different task on warped versus original
audio". Run at a single strong warp that is almost certainly the wrong design,
for a reason that costs nothing to fix.

The probe's protective setting is warp 0.88, where reference-based quality is
PESQ 1.18 and UTMOS ≈ 2.4 (§7.2, §8). A single-point test there has two
outcomes: listeners accept it, or they do not. If they do not — which I
estimate at better than even — the test has established that **one** point
fails and has said nothing about whether an acceptable point exists lower down.
We would then need a second round to find out, after recruiting again.

So the design sweeps magnitude and reads the **frontier**: the strongest
setting at which a naive listener majority still hears the same person. Same
listeners, same session, same cost. A "no" at the protective end still returns
a usable number.

This is a change of design, not of endpoint, and it is made before any data
exists.

---

## 1. Stimuli

Built with the existing code, unchanged in substance:
`experiments/make_listening_stimuli.py` (blinded filenames, loudness matching
via `utils/loudness.py`), `experiments/make_listening_test.py`,
`experiments/make_listening_page.py`, `experiments/serve_listening_test.py`,
analysed by `experiments/analyse_listening.py`.

`listening_test/design.json` already specifies 6 speakers and 8 conditions and
is **superseded by this document** for the condition list and the trial counts.
The speaker set (3000, 3081, 3170, 3536, 3576, 3752) is retained.

### 1.1 The transform under test is `t_mcadams`, not `t_warp`

§8 established that `t_warp` (STFT-magnitude warp with original phase) is
artefact-prone and that `t_mcadams` (per-frame LPC, pole phases raised to α,
resynthesis — the VoicePrivacy B2 baseline, Patino et al., arXiv 2011.01130)
gains ≈ 0.7 UTMOS at comparable protection. A listening test on the artefact-prone
implementation would measure the artefact, not the formant shift.

**Consequence, stated now:** this test licenses conclusions about **McAdams
formant shifting**. If Step 2 later optimises a different transform family, the
frontier does not automatically transfer and a second gate is required.

### 1.2 Conditions

Six transform levels plus four reference arms. α is the McAdams coefficient;
α = 1.0 is identity.

| condition | α | why it is here | §8 UTMOS | §8 protection (trained ASV) |
|---|---|---|---|---|
| `original` | — | **high anchor.** Must be heard as the same speaker or the listener is excluded. | 4.12 | — |
| `anchor_diff` | — | **low anchor.** A *different* speaker, same sex, same corpus. Must be heard as different or the listener is excluded. | — | — |
| `tilt_-6db` | — | the existence proof: free, purification-invariant, 0 % protective. Calibrates "a listener notices a channel change but keeps the speaker". | 4.12 | **0 %** |
| `mcadams_100` | 1.00 | **the processing control.** Full LPC analysis–resynthesis at α = 1, i.e. the vocoder round trip with no formant shift. Separates "warping changed the speaker" from "resynthesis damaged the audio". | not measured | ~0 % expected |
| `mcadams_095` | 0.95 | frontier rung | 3.29 | 18 % |
| `mcadams_090` | 0.90 | frontier rung (interpolated; not in §8's grid) | ~2.9 | ~28 % interpolated |
| `mcadams_085` | 0.85 | frontier rung | 2.57 | 38 % |
| `mcadams_080` | 0.80 | **the protective end** | 2.41 | 58 % |
| `additive_eot` | — | the incumbent, for a direct comparison on the same listeners | 2.73 | 90 % |

`mcadams_100` is the one condition not in `design.json` and it is the most
important addition. Without it, a rejection at α = 0.80 cannot be attributed to
formant shifting rather than to LPC resynthesis, and the whole frontier is
uninterpretable. The §8 column is quoted here as **design motivation only** — it
is 20 speakers on the superseded base and may not appear in the paper (base
rule, `FRAME_v6` §4).

### 1.3 Blinding and presentation

* Filenames randomised by `make_listening_stimuli.py`; the key stays in
  `listening_test/stimuli_blind_key.txt` and is not read until analysis.
* All stimuli loudness-matched (`utils/loudness.py`) so level is not a cue.
* Trial order randomised per listener; condition order counterbalanced.
* Each trial pairs a **reference** utterance (clean, one session) with a
  **test** utterance (a different session, then transformed), so a listener
  judging "same speaker" is doing the task the threat model's verifier does:
  cross-session identity, not clip matching.

---

## 2. Task and response

Per trial: two clips, A then B, replayable. One question.

> **Are these two recordings the same person?**
> Definitely same / probably same / probably different / definitely different

Binarised at the midpoint for the primary endpoint (**same** = definitely or
probably same). The four-point scale is retained because confidence is free to
collect and informative if the primary lands near the threshold.

Secondary question, same trial: *"Does B sound like a normal recording?"*
(yes / no / unsure). This is naturalness as a listener reports it, against the
MOS predictors used in §40 and §8. It is secondary and no gate reads it.

---

## 3. Sample size, fixed before collection

**n = 24 listeners**, each judging **one trial per condition per speaker-pair**,
across 6 speakers × 10 conditions = **60 trials per listener**, plus 6 anchor
repeats for the exclusion check, so **66 trials**, ~12 minutes.

Power basis for the primary endpoint. The decision rule in §4 is a per-condition
test of whether the same-speaker rate exceeds 50 %. With 24 listeners × 6
speakers = 144 judgements per condition, clustered by listener, I assume an
intra-listener correlation of 0.2, giving a design effect of about
1 + (6 − 1) × 0.2 = 2.0 and an effective n per condition of ≈ 72. A one-sided
binomial test at α = 0.05 with n_eff = 72 detects a true rate of 0.65 against
0.50 with power ≈ 0.80. So the test is powered to distinguish "a clear majority
hears the same speaker" from "a coin flip", which is exactly the resolution the
gate needs — and no finer.

**It is not powered** to distinguish 0.55 from 0.50, and no branch in §4 depends
on that distinction. Recording this now because §48 found the project reading
verdicts from cells whose MDE it had never computed.

**Minimum to read any verdict: 18 listeners** after exclusions. Below that the
result is **1-D, underpowered**, and the point estimates may not be read — the
§48 precedent is explicit that an underpowered cell is inconclusive, not null.

### 3.1 Exclusions, fixed in advance

A listener is excluded if either anchor fails:
* `original` judged **different** on ≥ 2 of 6 trials, or
* `anchor_diff` judged **same** on ≥ 2 of 6 trials.

Exclusions are reported with the result. No other exclusion is permitted.

### 3.2 Recruitment, and the honest risk

Listeners are naive (not the author, not anyone who has heard the stimuli
before). The recruitment channel is not yet solved and that is the largest
risk to this gate — `PLAN_V4` §7 puts 1-D at 10 % for exactly this reason. If
n = 18 is not reached by **2026-10-30**, the gate reports **1-D** and
`PLAN_V4`'s GATE 3 treats it as a failure to fund Step 2, not as a licence to
proceed on point estimates.

---

## 4. Endpoint, decision rule and outcomes

**Primary endpoint.** Per condition, the proportion of trials judged
**same speaker**, with a 95 % CI by cluster bootstrap over listeners (2000
resamples), and a one-sided test of H₀: p ≤ 0.50. Holm-corrected across the
**six** McAdams conditions. Anchors and `tilt` are not in the correction family;
they are validity checks.

**The frontier** is defined as: the **lowest α** (strongest warp) whose
Holm-adjusted lower CI bound exceeds 0.50. If no McAdams condition clears,
the frontier is **empty**.

**Validity preconditions.** No verdict is read unless all three hold:
1. `original` same-speaker rate ≥ 0.90;
2. `anchor_diff` same-speaker rate ≤ 0.10;
3. ≥ 18 listeners after exclusions.

Failing any one → **1-D**.

### Outcome → action

The protection figures below are from §8's 20-speaker superseded base and are
used **only** to classify the frontier into a branch; Step 1 re-measures them
on the current base and `PLAN_V4`'s GATE 2 reads that separately.

* **1-A — the frontier is at α ≤ 0.85**, i.e. listeners keep the speaker at a
  setting the probe associates with ≥ ~38 % protection.
  → The on-manifold route proceeds. **The frontier α becomes the naturalness
  budget Step 2 optimises under**, as a hard constraint, not a penalty term.
  `THREAT_MODEL` §5's open row closes with a number. This is the branch that
  funds Step 2 at GATE 3.
* **1-B — the frontier is at α ≥ 0.90**, i.e. listeners keep the speaker only
  where protection is ≤ ~28 %.
  → **The trilemma is confirmed on human listeners**, which is a result and not
  a null: inaudible, survives purification, actually protects — pick two, with
  the perceptual axis now measured on people rather than on a MOS predictor.
  The on-manifold route is dropped **as a method**. Step 2 is not funded. The
  frontier is reported as a contribution in the fallback paper
  (`PLAN_V4` §5, item 7) alongside `tilt`'s existence proof at the other end.
* **1-C — the frontier is empty.** No McAdams condition clears 0.50, including
  α = 0.95.
  → Either formant shifting at any useful magnitude is heard as a different
  person, **or** LPC resynthesis itself breaks identity — and `mcadams_100`
  decides which. If `mcadams_100` also fails, the finding is about the vocoder,
  not about warping, and must be written that way. On a genuine 1-C the route
  is **anonymisation, not protection**; write it up and go to the fallback paper
  immediately (`PLAN_V4` GATE 4 is pulled forward).
* **1-D — underpowered or invalid.** As defined above. No verdict, no point
  estimates quoted, and the test is reported as not run.

### 4.1 Secondary readings, which no gate consumes

* The four-point confidence distribution per condition.
* The "normal recording" rate per condition, against §40's UTMOS and NISQA.
* `additive_eot` versus the McAdams rungs **on the same listeners** — the first
  time the incumbent's perceptual cost is measured on humans. This is reportable
  whichever branch fires and is one reason the test is worth running even on a
  1-C.
* `mcadams_100` versus `original`: the cost of the vocoder round trip alone.

---

## 5. Binding constraints

* **This test may not be re-run at other settings to obtain a different
  frontier.** One collection, one verdict. If a design defect is found *before*
  responses are collected, this document is amended and the amendment dated; if
  it is found after, the defect is reported with the result.
* **The decision rule above may not be changed after seeing the data.** Not the
  0.50 threshold, not the Holm family, not the α ≤ 0.85 / α ≥ 0.90 branch cut.
* **Naturalness is secondary and stays secondary.** A branch may not be argued
  from the "normal recording" rate.
* **The §8 protection figures are motivation, not evidence.** They are 20
  speakers on a superseded base. No paper sentence pairs this test's frontier
  with those numbers; it pairs it with Step 1's.
* The transform is `t_mcadams`. A frontier measured here does not transfer to
  another transform family without a new gate (§1.1).
