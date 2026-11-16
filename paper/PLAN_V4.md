# PLAN_V4 — calendar, compute budget and abandon rules to 9 Feb 2027

**Written 2026-10-09. Supersedes `PLAN_V3_2026-10-05.md` in full.** PLAN_V3 was
built around the additive-crafting and α-geometry programme that RESULTS
§54–§58 closed: Control B removed the word *speaker* from the title (§54), #13c
was withdrawn as void (§55, §57), and the second audit showed the α contrast is
reproduced by a front end with no network in it (§58d). PLAN_V3's week
structure therefore plans work that no longer has a claim attached. It stays in
git history and is not quoted.

`NEXT_STEPS.md` says **what** to do next. This document says **when, with how
much compute, and when to stop.** Where the two disagree, this one governs
dates and budget; `NEXT_STEPS.md` governs scientific content.

Companion pre-registrations: `PREREG_step0_listening_frontier.md` (Step 0) and
`PREREG_step1_warp_remeasure.md` (Step 1). Step 2 has no pre-registration
because it is **unfunded** until both report.

---

## 1. The constraint, stated first

| | |
|---|---|
| Primary target | **Interspeech 2027, Sao Paulo — papers 9 Feb 2027, update 16 Feb 2027** |
| Today | 2026-10-09 |
| Calendar remaining | **17.5 weeks** to submission |
| Measured GPU capacity | **4–6 h/week**, one job at a time, RTX 3050 6 GB laptop |
| Hard compute stop | **2027-01-15** — after this nothing new is run, only writing |
| Compute weeks to hard stop | 14 weeks → **56–84 GPU-h nominal** |
| Planning figure (20 % reserve for failures and re-runs) | **~56 GPU-h usable** |

The 20 % reserve is not pessimism. §50 measured that the crafting loop does not
reproduce itself at fixed seed, §56 lost a fifth of an arm's n to a silent
filename mismatch, and §58 found two cohort speakers whose clips break a front
end. Every one of those cost a re-run.

**The budget does not fit all three steps.** Step 1 + Step 2 + Step 3 is
47–59 GPU-h at the low estimates and more if anything fails. That is the reason
the gates in §4 exist: they are not caution, they are arithmetic.

---

## 2. Calendar

Dates are end-of-day. "Report" means written into `RESULTS_ICASSP.md` as a
numbered section, not merely run.

### Phase A — the gates (2026-10-09 → 2026-11-20)

| by | what must be finished | GPU-h | blocking? |
|---|---|---|---|
| **10-14** | Step 0 stimuli built, blinded, loudness-matched; decision rule and n frozen in the prereg; page served | 0 | yes — nothing else waits on it |
| **10-16** | Step 1 pre-registration final; new measurement path gated through `check_direction_sanity.py` (§58 rule) | 1 | yes |
| **10-30** | **Step 0 reports.** Listener data collected and analysed (`analyse_listening.py`) | 0 | **GATE 1** |
| **11-13** | **Step 1 reports.** Fixed-magnitude warp re-measured on test-clean, 29 cross-session speakers, purifier v7 + the three foreign channels + PhonePuRe, matched and held-out | 12–16 | **GATE 2** |
| **11-20** | **Step 2 fund-or-drop decision** recorded in this file with the numbers that triggered it | 0 | **GATE 3** |

Step 0 and Step 1 run **in parallel**. Step 0 needs listeners and no GPU; Step 1
needs GPU and no listeners. Neither blocks the other, and if Step 0 fails while
Step 1 is mid-run, Step 1 still finishes — its re-measurement is a reportable
result either way (§5, fallback item 6).

### Phase B — build or fall back (2026-11-20 → 2026-12-04)

| by | if Step 2 FUNDED | if Step 2 DROPPED |
|---|---|---|
| 11-27 | optimiser implemented, smoke-tested on 3 speakers, naturalness budget enforced as a hard constraint | fallback paper outline written against the §5 contribution list |
| **12-04** | Step 2 pilot reports: does the optimiser beat fixed-magnitude warp at equal naturalness on ≥ 3 speakers? | **fallback paper first full draft** |
| **12-04** | **COMMIT DATE — see §4, GATE 4.** After this the paper's contribution list is frozen. | same |

### Phase C — the paper (2026-12-04 → 2027-02-09)

| by | what |
|---|---|
| 12-18 | Step 3 evaluation complete if Step 2 survived 12-04; otherwise all fallback tables and figures final |
| **2027-01-15** | **HARD COMPUTE STOP.** No new runs. Anything unmeasured is a limitation. |
| 01-22 | full draft, every number traced to a CSV, every claim carrying its condition |
| 01-29 | internal review council pass; `check_claims.py` green |
| 02-05 | final draft, references, 4 + 1 pages within the Interspeech limit |
| **02-09** | **submit** |
| 02-16 | update deadline — reserved for fixes only, not new content |

### Slack, named

There is **one** slack week, 2026-12-25 → 2027-01-01, deliberately left empty.
It is the only buffer. Spending it on anything other than overrun means the
hard stop moves into the writing period, which is how §25's failure mode starts.

---

## 3. Compute budget

Allocation against the ~56 GPU-h usable figure. "Measured" means a comparable
run exists in the record; "estimated" means it does not.

| item | GPU-h | basis |
|---|---|---|
| Step 1 — warp re-measure, 29 speakers, 5 warp strengths, 4 purifier channels, 3 encoders, eval-only | **8–10** | measured: §53's held-out grid was ~1 GPU-h for 12 cells eval-only; this is ~60 cells plus transform cost |
| Step 1 — PhonePuRe rows (external venv, stage 1 + 2) | **3–5** | measured: §29 ran 29 files through the full refiner |
| Step 1 — quality (UTMOS, NISQA, Whisper WER) on the warp arms | **1–2** | measured: `eval_quality_nr.py` on 29 files |
| Step 1 subtotal | **12–17** | |
| Step 2 — optimiser pilot, 3 speakers, CMA-ES or grid with purifier in the loop | **4–6** | estimated |
| Step 2 — full optimiser run, 29 speakers | **18–24** | estimated; dominant cost, and the reason for GATE 3 |
| Step 3 — evaluation of the optimised arm on held-out encoders × 4 purifier settings + PhonePuRe + quality | **10–14** | measured by analogy with §53 + §29 |
| Step 2 + 3 subtotal | **32–44** | |
| Fallback paper (§5) | **0–2** | §42h, §29, §40c, §22b, §47–§53, §58 are all **already measured**; only figure regeneration |
| Reserve already deducted | — | 20 % of nominal |

**Reading of the table.** Step 1 alone fits comfortably. Step 1 + Step 2 + Step 3
is 44–61 GPU-h against ~56 usable, i.e. it fits only at the optimistic end and
only if nothing is re-run. The fallback costs essentially nothing because it is
built from work already done.

**Standing rules.** One GPU job at a time (`run_queue*.sh` discipline, §memory).
Every crafting or optimiser arm ships with a skip-guard. Every new measurement
path passes `check_direction_sanity.py` before a verdict is read from it — §58's
rule, and it is now the first thing a reviewer of this project would check.

---

## 4. Decision gates

Each gate names the result that triggers it. The branches are fixed here,
before the data, because deciding afterwards is the failure mode this project
keeps catching in itself (§25, §48b, §57f).

### GATE 1 — Step 0, the listening frontier (by 2026-10-30)

Endpoint and decision rule are fixed in `PREREG_step0_listening_frontier.md`.
The gate reads its **frontier**: the strongest warp, if any, at which a naive
listener majority still hears the same speaker.

* **1-A — a usable operating point exists at or below the protective range.**
  Majority same-speaker holds at a warp strength that the probe associates with
  non-trivial held-out protection.
  → On-manifold route proceeds. Step 0's frontier becomes the **naturalness
  budget** Step 2 optimises under, and `THREAT_MODEL` §5's open row closes.
* **1-B — a usable point exists only where protection is negligible.**
  Majority same-speaker holds only at warp strengths the probe associates with
  ~0 protection.
  → **This is the trilemma confirmed on humans**, and it is a reportable
  finding, not a null. The on-manifold route is dropped as a *method*; the
  frontier becomes a result in the fallback paper. Step 2 is not funded.
* **1-C — no warp strength keeps the speaker.**
  → The route is **anonymisation, not protection** (`NEXT_STEPS` §4). Write it
  up, stop the phase, go to the fallback paper immediately rather than at
  GATE 4.
* **1-D — underpowered.** Fewer than the pre-registered n, or the two anchors
  fail their sanity check.
  → No verdict. Collect to n or report the test as not run. **Not** a licence to
  read the point estimates; §48 is the precedent.

### GATE 2 — Step 1, the re-measurement (by 2026-11-13)

Endpoint and decision rule are fixed in `PREREG_step1_warp_remeasure.md`. Two
things are being asked at once and they are scored separately.

* **2-A — transfer holds and survives the refiner.** Held-out-encoder
  protection from fixed-magnitude warp materially exceeds the additive system's
  held-out figure, **and** does not collapse under PhonePuRe.
  → The strongest possible input to GATE 3. Step 2 is fundable.
* **2-B — transfer holds, the refiner defeats it.**
  → The transform is not a defence against the strongest attacker in 𝓡₇, which
  is the same place the additive method fails (§29). Step 2 is **not** funded;
  the result is reportable and joins the fallback paper as a second
  measured ceiling.
* **2-C — transfer does not reproduce on the current base.**
  → The September probe's 20-speaker figure was base-specific. Report it as a
  failed replication of our own probe — the base rule exists precisely because
  this can happen. Step 2 is not funded.
* **2-D — the warp arm fails its instrument gate.** `check_direction_sanity.py`
  or the cohort-level checks reject the path.
  → Fix the instrument, re-measure once, and label it a fixed-instrument
  re-measurement. This is the **only** sanctioned re-run in this plan.

### GATE 3 — fund or drop Step 2 (2026-11-20, recorded in this file)

Step 2 is funded **only if all four hold**:

1. GATE 1 returned **1-A**.
2. GATE 2 returned **2-A**.
3. There is measured headroom: a gap between the fixed-magnitude frontier and
   the Step 0 naturalness budget large enough for optimisation to occupy. If
   fixed-magnitude warp already sits at the budget, an optimiser has nothing to
   win — it can only trade one axis for the other.
4. Remaining GPU-h at 11-20 ≥ **32**, the low estimate for Step 2 + Step 3.

Any one failing → **dropped**, and the fallback paper becomes the plan of
record at that moment rather than at GATE 4.

### GATE 4 — commit to a paper (2026-12-04, immovable)

Whatever exists on 12-04 is the paper. If the on-manifold route has not by then
produced a measured result that beats the additive system's held-out condition
under at least one purifier, **the fallback paper is the submission** and the
on-manifold work becomes one section of it.

This date is immovable because Phase C needs nine weeks and the hard compute
stop is 01-15. Moving 12-04 means submitting an unreviewed draft.

---

## 5. The fallback paper

It is not a consolation prize. Every component is **already measured**, costs
0–2 GPU-h, and survived the two audits.

Working title: *What a Speech Purifier Erases, and What That Does to the
Instruments: Measurement Limits of Purification-Robust Voice Protection*
(FRAME_v7 §2, the title Control B forced).

| # | content | source | status |
|---|---|---|---|
| 1 | The purifier degrades the **verifier**, and EER hides how much: WavLM-SV EER 5.2 % → 9.5 % while minDCF at p_target = 0.01 moves 0.77 → 0.91 | §42h | measured; narrowed by `related_work.md` §13 — MDD already published the EER effect more dramatically, and we must say so |
| 2 | ASV acceptance **overstates** what the cloner clones: PhonePuRe restores the speaker on 29/29 by ASV while end-to-end clone success is 38 / 27 / 71 % | §29 | measured |
| 3 | Part of "protection" is **damage to the clone**: 0.32 UTMOS / 0.44 NISQA less natural, halving under the purifier, WER untouched | §40c | measured |
| 4 | **Coverage, not generalisation**: clone success 12–14 % on crafted-against encoders against 75–92 % held out, one variable changed | §22b | measured |
| 5 | The d-vector reversal is an **artefact of encoder overlap**, confirmed twice: held-out grid at both budgets, and the interaction p = 2e−13 matched against p = 0.19 held out | §47, §49, §52, §53 | measured |
| 6 | **The instrument audits**: a measurement path that evaluated a gradient at the maximum of its own objective returned residue; the α contrast is reproduced by a random projection of the front end with no network in it; α tracks spike height at ρ = −0.75 and is sign-asymmetric | §57, §58 | measured |
| 7 | Step 0's listening frontier, whichever branch fired | Step 0 | pending GATE 1 |
| 8 | Step 1's warp re-measurement, whichever branch fired | Step 1 | pending GATE 2 |

Items 7 and 8 are why Step 0 and Step 1 are worth running **even if the route
dies**: both are contributions to the fallback paper. Item 6 is the one a
reviewer will find hardest to dismiss, because it is a correction other papers
in this paradigm inherit and it does not require our method to be good.

**Commit date for the fallback: 2026-12-04** (GATE 4), or immediately on a
1-C at GATE 1.

---

## 6. Venue

| venue | deadline | decision |
|---|---|---|
| **Interspeech 2027**, Sao Paulo | **9 Feb 2027** (update 16 Feb) | **primary target.** Both the on-manifold paper and the fallback fit its 4 + 1 format and its audience. |
| ISCA **SPSC Symposium 2027** | ~May–Jun 2027, co-located with Interspeech | **the workshop fallback.** Exactly our community, high acceptance, and a natural home for a measurement-and-limits paper. |
| ACM **IH&MMSec 2027** | ~Feb 2027 | parallel option if the paper reads as watermarking/forensics rather than speech |
| **ASRU 2027** | ~May 2027 | later shot if Interspeech rejects |

**When the workshop becomes the target instead of the main conference.** Any
one of these flips it, and the flip is a decision to make on the date, not a
drift:

* **2026-12-04 (GATE 4)** — the fallback paper is the submission *and* items 1–6
  have not been assembled into a draft with a clear lead contribution. A
  measurement paper with a contested lead is a workshop paper; Interspeech
  rejects it at 30 %.
* **2027-01-15 (hard stop)** — a headline number is still unmeasured. Submitting
  a paper whose lead claim rests on a pending run is how §55 happened.
* **2027-01-29** — the internal council pass does not clear, or `check_claims.py`
  is still failing.

Note the asymmetry that makes this cheap: Interspeech notification leaves time
to redirect to SPSC, whose deadline is months later. Submitting to Interspeech
and losing costs the submission effort only. **So the default is: aim at
Interspeech, hold SPSC as the catch**, and only pre-emptively switch if one of
the three triggers above fires.

---

## 7. Odds, per branch, honestly

These are my estimates, written before the data, so they can be scored
afterwards. The project's record of pre-registered predictions is mixed
(§7.5 failed, §49e held), which is the reason for writing them down.

| branch | probability | reasoning |
|---|---|---|
| **GATE 1 → 1-A** (usable point exists where protection is real) | **30 %** | Formant structure is the strongest perceptual identity cue humans use. The reason warping transfers across encoders — it cannot be augmented away without destroying the speaker label (§7.2) — is the same reason listeners notice it. The mechanism that makes the method work is the mechanism that makes it audible. |
| GATE 1 → 1-B (usable only where protection is negligible) | 40 % | The most likely single outcome, and the one the trilemma predicts. Spectral tilt is the existence proof at one end (§8.2: free, invariant, useless). |
| GATE 1 → 1-C (no strength keeps the speaker) | 20 % | At warp 0.88 the probe measures PESQ 1.18 and UTMOS ~2.4; a listener may well reject the whole range. |
| GATE 1 → 1-D (underpowered) | 10 % | `listening_test/responses` is empty today and recruitment is unsolved. This is a real risk and it is a project-management risk, not a scientific one. |
| **GATE 2 → 2-A** (transfer holds *and* survives PhonePuRe) | **35 %** | Transfer itself I put at ~65 %: the mechanism argument is sound and two independent cells agree (§7.2's d-vector 0.55; §8's point 4, McAdams 0.80 protecting 80–85 % on the d-vector against additive's 55 %). Surviving the refiner I put at ~55 % and it is the genuinely unknown half — a phoneme-conditioned restorer cannot know the original formants, but it is trained to produce natural speech and a warped vowel is not where its prior sits. |
| GATE 2 → 2-B (transfer holds, refiner wins) | 30 % | |
| GATE 2 → 2-C (transfer does not reproduce) | 25 % | 20 speakers, superseded base, one warp family. §45 is the precedent for a result that was directionally reproduced and underpowered. |
| GATE 2 → 2-D (instrument gate fails) | 10 % | The warp path is new code. §58's two audits found a fatal bug in one new path and a front-end knife-edge in another. |
| **GATE 3 → Step 2 funded** | **~15 %** | Requires 1-A **and** 2-A **and** measured headroom **and** 32 GPU-h left. The conjunction is the point. |
| **Submission is the fallback paper** | **~70 %** | And it is a real paper. |
| Interspeech acceptance, fallback paper | 30–40 % | The AC's estimate on the v7 reframe was 42 %; item 1's narrowing by the MDD citation argues for the lower end. |
| Interspeech acceptance, on-manifold paper if Step 2 lands | 45 % | Optimised on-manifold parameters evaluated under restoration is a genuinely unoccupied cell. |

**The single most likely history**, at roughly one chance in three: Step 0
returns 1-B, Step 1 returns 2-A or 2-C, Step 2 is dropped at GATE 3, and we
submit the measurement paper to Interspeech on 9 Feb with the listening
frontier and the warp re-measurement as two new sections. That is a worse
outcome than the on-manifold method working and a much better one than
discovering in January that it does not.

---

## 8. What would make me wrong

Recorded so it can be checked later, in the spirit of §7.5.

* If Step 0 returns 1-A at a warp strength where held-out protection is ≥ 50 %,
  my 30 % was too low and the perceptual-cue argument above is wrong.
* If PhonePuRe leaves warp untouched, the "restoration has nothing to subtract"
  half of the hypothesis is stronger than I credit, and Step 2 deserves
  funding even on a weaker GATE 1.
* If the per-speaker warp direction idea (warping toward each speaker's own
  distribution edge rather than a global α) buys transfer at materially lower
  perceptual cost, the trilemma as stated is about *global* warps only, and
  the frontier has to be re-drawn. This is untested, cheap, and not in
  `NEXT_STEPS`; it is the one addition I would make to Step 2 if it is funded.
