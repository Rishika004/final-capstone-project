# Reviewer council v4 — 2026-10-06

Five reviewers re-read RESULTS §37–46, `FRAME_v6`, `THREAT_MODEL`, both
pre-registrations and the CSVs, after the fortnight that closed #3, #10, #19,
#7, #8(C1 half) and returned **OUTCOME B** on both #5 and #13b.

| Reviewer | Score as-is | Score if followed | Vote |
|---|---|---|---|
| R1 adversarial ML | 4.5/10 (was 4) | 6.5 | weak accept if level-led + §42h promoted + #13c |
| R2 speech / ASV | 5.5/10 (was 4) | 7.0 | weak accept; demands a protected-arm operating point |
| R3 statistics | trust 7.5/10 (was 6.5) | 8.5 | signs off §45h; will **not** sign §46 as sectioned |
| R4 security | 5/10 (was 4) | 6.5 | reject as a protection paper; accept as measurement/limits |
| AC | — | — | **Interspeech, 42 %** under reframe vs 30 % as FRAME_v6 |

Every reviewer moved **up**. Nobody moved up much. The fortnight's two
pre-registered upgrade paths both returned B, and the reviewers agree the
compensation is a finding nobody was looking for (§42h).

---

## A. Unanimous

1. **Promote §42h to a first-class contribution.** R1 ranks it #2, R2 calls it
   "C0", R4 calls it #2, the AC makes it C2. *The purifier degrades the
   verifier, not just the perturbation* — WavLM-SV at minDCF 0.77–0.91 in every
   reported cell, ECAPA 0.02 → 0.30 under DiffWave t=3 — is a correction other
   people's papers inherit, it is already measured, and it costs zero GPU. It
   is currently filed as a setup table.
2. **Lead C1 with the LEVEL result, not the ordering.** α below norm- **and**
   band-matched random for five encoders across four architecture families, two
   never seen by the optimiser, p ≤ 2.6e−8, 24–29/29 speakers. The ordering is
   the fragile half (both t=50 middle steps at p_adj = .032; the middle
   collapses at t=200; x-vector ≈ d-vector never separates).
3. **Never pair WavLM-SV with ECAPA as equal evidence.** Demote, do not drop —
   it is C1's highest-α encoder. R2's mechanism: print minDCF in the column
   header of every protection table, and call it an *embedding probe* rather
   than a verifier wherever it carries an identity claim.
4. **Cut "C1 gives a criterion for choosing ensemble members."** §46 killed it.
5. **Measurement-and-limits paper, not a protection paper.** R4 would not let
   "protection method" appear in the title, abstract or contributions.
6. **Keep "directionally reproduced"; never "replicated."**

---

## B. New defects found, all verified against the files

| # | finding | status |
|---|---|---|
| **B1** | **The shipping arm is the matched-encoder upper bound.** `run_rebase_mismatch_m30_seg.sh:70` and the dev-clean runner both pass `--craft-encoders all`, which includes the GE2E/Resemblyzer network that *is* DiffVC's conditioner. So §29e, §40, §41, §45 all live in the condition `THREAT_MODEL` §3 itself labels an upper bound, while "held-out (realistic)" is defined and then **not used**. (R4 §1e) | **CONFIRMED** |
| **B2** | **`alpha_pairs.csv` held only the ten t=200 rows.** The t=50 table in §44b — which carries the *new, stronger* three-group claim — was unreproducible from committed results, because the analyser was run twice with single timesteps and the second overwrote the first. (R3 §3.4) | **CONFIRMED, FIXED** — now 20 rows, both timesteps |
| **B3** | **PREREG_13b has no power analysis.** PREREG_05 §5 has one. From the delivered CIs, the minimum detectable effect was 0.028–0.052 against observed \|Δ\| 0.0066–0.0190: **~10 % power**. The test could not have resolved anything. (R3 §1) | **CONFIRMED** |
| **B4** | **§46 is mislabelled "null"; it is INCONCLUSIVE.** Accepting a null from a ~10 %-power test is the error this project criticises elsewhere. (R3 §1) | **CONFIRMED** |
| **B5** | **§46e understates the §7.4 override.** §5 unamended returns **Outcome A** on this result. §7.4 changed the rule — legitimately (committed before scoring, strictly tighter, a general principle) but it must be written as *an amendment closing a gap in §5's outcome space*, not as "the pre-registration already said so". (R3 §4) | **CONFIRMED** |
| **B6** | **The p-values on 12/12 and 8/8 signs are not defensible.** Cells are 2–3 factors over the *same* speakers; effective independent facts ≈ 2–4, not 8–12. §46e flags this for its 8 cells and §45c does not for its 12 — an internal inconsistency on the same day. Replace with speaker-level agreement (16–22 of 29 against 14.5; four of twelve dev-clean cells are coin flips at 17–19/31). Drop the r = 0.961 **p-value**. (R3 §1, §2) | **CONFIRMED** |
| **B7** | **No operating point for the protected arm.** All eight channels are *purification* channels; there is no DET/EER/minDCF for protected audio, so the paper still cannot state what protection does in ASV currency. #19 is closed for channels, not for arms. (R2 §5.1) | **CONFIRMED** — R2's top ask |
| **B8** | **The success criterion's quality conjunct cannot fail.** §6 requires "the budget in (a) is met", and (a) *is* the crafting constraint — satisfied by construction for every file the optimiser emits. Withdrawing PESQ > 3.0 deleted the only part that could fail. (R4 §1a) | **CONFIRMED** |
| **B9** | **𝓡 is never enumerated.** "min over purifier R ∈ 𝓡" quantifies over an undefined set; under Kerckhoffs with permanently published audio it is unbounded. Label the verdict "minimum over a six-element measured set". (R4 §1b) | **CONFIRMED** |
| **B10** | **Codec / channel transport still unmeasured and now also unlisted** in the capability list — R4's first question for two councils running, and δ is bandlimited to 6 kHz, exactly where a codec lives. Also missing: multi-clip aggregation, cropping, speech-enhancement preprocessing. (R4 §1c) | **CONFIRMED** |
| **B11** | **PhonePuRe is load-bearing in the verdict but has no calibrated operating point.** §42h closed eight channels; PhonePuRe is not among them. (R4 §1g) | **CONFIRMED** |
| **B12** | Possible contamination: is the foreign DiffWave trained on dev-clean? | **CHECKED — NO.** De-AntiFake's `data_preprocess/configs/libri_get_spec.json` points at `train-clean-100`, disjoint from both evaluation cohorts. Evidence is their config, not an explicit statement; record it as indicative. |

---

## C. Where the reviewers disagree

1. **How much credit the sign agreement earns.** R1 and the AC treat 12/12 and
   8/8 as meaningful corroboration; **R3 says the quoted p-values are
   indefensible** and the honest unit is per-speaker agreement. **R3 is right**
   — the cells are not independent, and §46e already concedes exactly this for
   its own eight while §45c does not for its twelve. Adopt R3's framing in both.
2. **What §46 is worth.** R1 and R4 want it reported as a reported null; **R3
   wants it demoted to a limitations paragraph and relabelled inconclusive**,
   on the grounds that prominence invites the misreading the author is trying
   to avoid. R3's reading is better supported: a ~10 %-power test deserves two
   sentences, not a results subsection.
3. **EOT parity (#9).** PLAN_V3 schedules it for W6 at ~4 GPU-h (R3 and R2 are
   silent). **The AC says cut it outright**: it is 8 GPU-h defending a crafting
   recipe that `FRAME_v6` §2 explicitly calls "Not a contribution". The AC has
   the better argument and it frees most of the remaining budget.
4. **What the listening panel is for.** PLAN_V3's gate makes it decide the
   venue. **The AC says that gate was written for a defence paper** and no
   longer fits a measurement paper; run the panel as a declared pilot with a
   pairwise same/different design, and let the 20 January decision read off the
   draft instead. R2 and R4 still treat absent human data as a reject-on.

---

## D. Proposed FRAME_v7

**Title.** *Every Speaker Encoder's Identity Direction Is Purifier-Hostile:
Encoder Geometry and the Measurement Limits of Purification-Robust Voice
Protection*

**Thesis.** For five speaker encoders across four architecture families, each
encoder's own identity direction survives a speech purifier *worse* than a
norm- and band-matched random direction, and how far below it sits differs by
encoder; this one crafting-free measurement accounts for which encoders a
purification-invariance constraint helps and which it reverses — while the same
purification degrades the verifiers used to score protection to near-trivial
operating points, so part of what this paradigm reports as its limit is a
property of its instruments.

**C1 — the geometry, and what it explains.** α below matched-random on five
encoders, four families, two unseen; front-end-invariant; ordering
timestep-qualified, two-group claim at both timesteps, three-group at t=50 only
after the Holm family scope is declared. The d-vector reversal (§45e) sits
*inside* C1 as the explanandum, with its own figure.

**C2 — the measurement is not neutral.** §42h's verifier weakness, §29's
proxy-vs-end-to-end gap, §40c's clone degradation. Three independent ways the
field's proxy scoring distorts itself, all measured, zero GPU.

**C3 — the measured ceiling.** Coverage ≠ generalisation; BPDA ≡ exact as a
TOST; PhonePuRe reduces-not-removes.

**Setup.** The recipe, declared as the artefact analysed and **as the
matched-encoder upper bound** (B1).

---

## E. What to do, in order

**Zero GPU, and all of it required:**

1. **B1 relabelling** — every verdict sentence marked matched-encoder, or
   "realistic" dropped from the threat model. One line plus table headers.
   *Highest value-per-minute item in the project.*
2. **TOST on §30** (R3: highest-value statistic). The confirmatory run's use of
   BPDA rests on an equivalence currently asserted from a non-rejection.
3. **Power/MDE statement on every null cell in §45 and §46**; relabel §46
   inconclusive (B3, B4); fix the §46e sentence (B5); drop the sign p-values
   and the r p-value (B6).
4. **Mixed model** `margin ~ recipe×encoder×purifier + (1|speaker)` as a
   **pre-specified secondary** analysis — R3 calls it the paper's missing
   primary analysis, since C2 *is* an interaction claim and the 12-cell grid
   tests it nowhere.
5. Commit `analyse_mismatch_final.py`; declare the §44 Holm family scope.
6. Threat model: external quality floor (B8), enumerate 𝓡 (B9), list the
   unmeasured capabilities (B10).

**GPU, in order:**

7. **#13c — α through DiffWave (~3 GPU-h).** Unanimous: C1's α is measured on
   the mel purifier while the phenomenon lives on DiffWave. The only structural
   hole left in the lead contribution.
8. **Protected-arm operating point (B7, ~1 GPU-h)** — R2's top ask.
9. **Calibrate CAM++ and ERes2Net as verifiers (R1, near-zero GPU)** — replaces
   "one strong verifier and two weak ones" with a credible modern set.
10. **CUT #9 EOT parity.** 8 GPU-h for a declared non-contribution.

**Not before 20 January:** second cloner, codec study, PhonePuRe calibration.
