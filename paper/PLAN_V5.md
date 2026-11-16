# PLAN_V5 — from here to Interspeech 2027

**Written 2026-11-16. Supersedes `PLAN_V4_2026-10-09.md` in full.** PLAN_V4 was
built around the on-manifold / formant-warp phase, which was never chosen
(`DEFENSE_BRIEF` Part II) and is not scheduled. Its Step 0 and Step 1
pre-registrations stay in git history and are not quoted.

The idea is fixed (`DEFENSE_BRIEF` §I.1) and this plan does not revisit it:

> An AntiPure for voice — a protective perturbation that survives diffusion
> purification and stops at least some diffusion-based voice cloning.

---

## 1. The constraint

| | |
|---|---|
| Target | **Interspeech 2027, São Paulo — papers 9 Feb 2027**, update 16 Feb |
| Today | 2026-11-16 |
| Calendar remaining | **12 weeks** |
| GPU remaining need | **4–6 hours total** — one week's budget at the measured 4–6 h/week |
| Hard compute stop | **2027-01-15** |

**Compute has stopped being the constraint.** §59–§64 closed the experimental
chain; what is left is one small batch of quality and calibration runs. The
binding constraints are now **writing** and **listener recruitment**, and only
one of those is mine.

---

## 2. What is already done and needs nothing

Do not re-run or re-open these.

| evidence | status |
|---|---|
| C1 — defence vs DiffVC, 9/9 cells, ECAPA 48.3 % → 0.0 % | §61a |
| C1 — defence vs F5-TTS, 9/9 cells, ECAPA 86.2 % → 10.3 % | §60b |
| C1 — transfer to a purifier never crafted against, 89–128 % retention | §33 |
| C1 — stochastic best-of-5 attacker, 90–97 % protection | §28c |
| C2 — attribution, cloner-varying route, budget matched | §60d, §61c |
| C2 — attribution, ensemble-varying route, energy matched exactly | §64 |
| C2 — decomposition: EOT vs the regulariser | §62 |
| C3 — held-out encoder, 72.4 % → 72.4 % under a foreign purifier | §63a |
| C3 — PhonePuRe reduces protection, 38/14/59 % unfiltered | §29e |
| quality of the shipping arm: UTMOS 3.073 vs 4.152, WER untouched | §40b |
| Figures 1 and 2 | `fig_n1_coverage.png`, `fig_cloners.png` |

---

## 3. Work remaining, in priority order

### P0 — commit (today, 10 minutes, me)
Six GPU-hours, nine scripts and ~600 lines of `RESULTS_ICASSP.md` exist only
as working-tree changes. Nothing else on this list is at risk of being lost;
this is. Tag it so the paper's numbers have a referenceable commit.

### P1 — start listener recruitment (today, you, 0 GPU)
**The longest lead item and the only one I cannot do.** Stimuli, blinding and
loudness matching are built (`listening_test/`, 48 stimuli at −23.00 LUFS).
Target **15 listeners**; 10 is usable.

It gates the sentence "and it still sounds like the same person", which a
defence paper wants. It is **not** an absolute blocker: §40b already gives
UTMOS 3.073 vs 4.152 with WER untouched, which is defensible objective
evidence. See GATE 1 for the fallback.

### P2 — quality on the F5-TTS clones (1–2 GPU-h, me) — NEW, call it N8
§40c established that part of what we call protection is **damage to the
clone** rather than a change of identity. That is measured for DiffVC and
**not** for F5-TTS, which is now a headline cloner in C1. A reviewer will ask
"is this protection or just a broken clone?" and C1 should answer it.
`eval_quality_nr.py` over `results/f5_clones/`, UTMOS + NISQA + WER,
loudness-matched (§40a).

### P3 — N4, PhonePuRe-channel threshold calibration (1–2 GPU-h, me)
§29's ASV-acceptance figures are provisional and they sit in C3, the limits
section — the second thing a reviewer reads.

### P4 — N6, quality on the plain and purifier-aware arms (1–2 GPU-h, me)
§40 measured the shipping arm only. C2 is a plain-vs-ours comparison, so both
arms need quality. Partly pre-empted: §60d already has PESQ for the seg pair
(2.091 vs 2.283) and §63 has SNR for all four. N6 adds UTMOS/NISQA/WER.

### P5 — the draft (weeks, me + you)
`draft_v6` predates §59–§64 and cannot be revised into this paper; write
fresh against `DEFENSE_BRIEF` §I.6. Four pages plus references.

### P6 — verification before submission
`check_claims.py` green; every number re-read from its CSV; references
verified against §10.6's correction table (AntiFake's DOI, AudioPure's venue,
and the two entries marked **do not cite**: VoiceShield, DiffSpec).

### Explicitly NOT doing
More speakers (§52/§64's power limit — needs ~2× the cohort and a full re-run;
state as a limitation). The on-manifold hybrid. Anything with α.

---

## 4. Calendar

| by | what | whose |
|---|---|---|
| **11-16** | P0 commit + tag | me |
| **11-17** | P1 recruitment started, page live | **you** |
| 11-23 | P2 (F5-TTS clone quality) reported as a RESULTS section | me |
| 11-30 | P3 + P4 done. **EVIDENCE FREEZE** — contribution list fixed | me |
| **12-14** | **First full draft**, every number traced to a CSV | me |
| 12-24 | Internal review pass (council), figures final | me |
| 01-01 | *slack week — the only one* | — |
| 01-08 | Revision complete | me |
| **01-15** | **HARD COMPUTE STOP.** Anything unmeasured is a limitation | — |
| **01-22** | **Listening data closes.** Whatever arrived is what is reported | you |
| 01-29 | `check_claims.py` green; references verified | me |
| 02-05 | Final read | both |
| **02-09** | **Submit** | — |

---

## 5. Gates

### GATE 1 — listening data (2027-01-22)
* **≥ 10 listeners** → report the same/different result in C1's quality
  paragraph, whichever way it goes. A negative result is reported, not hidden.
* **< 10** → the paper states plainly that no listening test was collected and
  rests quality on UTMOS 3.073 / NISQA / WER (§40b). Costs roughly half a
  point of review score; does not block submission.

### GATE 2 — evidence freeze (2026-11-30)
Whatever is measured on 11-30 is the paper. After this date, new experiments
are out of scope regardless of how cheap they look. This is the date that
protects the writing period, and PLAN_V4's §25 failure mode is what happens
without it.

### GATE 3 — draft exists (2026-12-14)
If there is no full draft on 12-14, the submission target moves to **ISCA SPSC
2027** (~May–Jun, co-located, our exact community) and Interspeech is dropped
rather than rushed. Deciding on the date, not drifting.

---

## 6. Risks, named

| risk | mitigation |
|---|---|
| **Recruitment never happens** (zero responses since September) | GATE 1's fallback; the paper survives without it |
| **C2 is thin** — one significant cell of nine energy-matched | It is not the headline (§I.1); reported with its interval and labelled underpowered, never null |
| **"Isn't this AntiPure for audio?"** | Conceded in the first related-work paragraph, in one sentence (§I.5); the audio gap and C2/C3 are the answer |
| **Writing slips** | GATE 3 moves the venue instead of submitting a rushed paper |
| **A number does not reproduce** | §62b's rule: every cloning comparison from one run, `--clone-seed` on all new runs |

---

## 7. Honest odds

| | |
|---|---|
| Submission happens on 9 Feb | **80 %** — the evidence is done; only writing stands between here and there |
| Listening data arrives (≥ 10) | 40 % — unsolved since September and outside my control |
| Interspeech acceptance | **40 %**, or ~45 % with listening data |
| SPSC acceptance if Interspeech rejects | high; it is the natural home for this paper |

The single most likely failure is not scientific. It is that the draft starts
in January.
