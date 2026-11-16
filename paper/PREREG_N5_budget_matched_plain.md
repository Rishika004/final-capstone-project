# Pre-registration — N5, the budget-matched plain arm

**Written 2026-11-12, before the run.** Settles the one open confound in the
defence paper's lead claim (RESULTS §59f). `DEFENSE_BRIEF` §I.3 lists it as the
highest-value remaining run.

---

## AMENDMENT, 2026-11-12, before any N5 result was seen

While checking the budget of the **seg-crafted** arms for §60 (N2's write-up),
the confound turned out not to exist there:

| seg arm | realised SNR | PESQ |
|---|---|---|
| `plain_tc_m30_seg` | 16.206 dB | 2.091 |
| `fullnsT_tc_m30_seg` | 16.113 dB | **2.283** |

Paired, purifier-aware minus plain: SNR **−0.093 dB, p = 0.408** (not
significant) and PESQ **+0.192, p = 1.6e-06** — the purifier-aware arm is at
the same perturbation energy and *better* perceptual quality. Segment crafting
makes the plain arm spend its budget (plain goes 19.02 dB non-seg → 16.21 dB
seg), which is why §59f's 2.89 dB gap is specific to the `m30_f15` arms.

**So the budget-matched comparison already exists on disk and needs no
rescaling and no re-crafting** — it needs the DiffVC cloning evaluation, which
was never run for `plain_tc_m30_seg` (§59's "not measured" note). That is
strictly better than N5a and N5b: it uses the **actual shipping recipe**, the
budgets match **by measurement rather than by construction**, and it stays
inside the psychoacoustic constraint instead of a rescale that breaks it.

**N5 is therefore redefined, before seeing its outcome, as:**

> **N5′ — DiffVC cloning of `plain_tc_m30_seg` and `fullnsT_tc_m30_seg`**,
> 29 cross-session test-clean speakers, channels `none` / `ours` / `diffwave`
> (`--dw-t 3`), evaluation only, no crafting.
> Output: `results/cloning_tc_m30_seg_n5.csv`.

Everything below still governs: the primary endpoint (§4), the matching check
(§4, now satisfied by measurement at 0.093 dB, well inside the 0.5 dB
tolerance), and the branches (§5, read on ECAPA at `ours`). The prediction in
§5 stands unchanged and is scored against N5′.

N5a (rescale) and N5b (re-craft) are **not run**: both existed only to
manufacture a budget match that the seg arms already have. `make_budget_matched_plain.py`
is kept for the record, unused. If N5′ returns 5-C, N5a becomes worth running
as a cross-check on the non-seg arms, and only then.

---

## 1. The confound, stated exactly

§59c says purifier-in-the-loop crafting makes protection **generalise**: held
out, ECAPA clone success 27.6 % → 10.3 % through the purifier, 7 of 9 cells
Holm-significant.

§59f found the two arms do not spend the same budget. Held out, the
purifier-aware arm uses **2.89 dB more** (SNR 19.02 → 16.13 dB, p = 1.4e-11)
and 0.232 PESQ. "Protects better" and "perturbs more" predict the same margin.

Two tests already argue against magnitude — the louder `full_` arm gains *less*
than the quieter `fullnsT_`, and the within-speaker correlation between extra
dB and protection gained is r = −0.024 (p = 0.90). Both are indirect. This
pre-registration fixes the direct test.

## 2. Why the obvious version does not work, and what we do instead

The obvious fix is "re-craft plain at a bigger budget until it lands at
16.1 dB". A per-frame diagnostic (measured 2026-11-12, before this document)
says that is not straightforward:

| arm | median local SNR | frames within 1 dB of the floor |
|---|---|---|
| `fullnsT_m30_f15_nodvec` | 15.13 dB | **84.0 %** |
| `plain_m30_f15_nodvec` | 16.12 dB | **63.0 %** |

The purifier-aware arm is **budget-saturated**: the `--snr-floor 15` per-frame
gate is what binds, in 84 % of non-silent frames. The plain held-out arm is
only 63 % saturated, so in roughly a third of its frames the limit is either
the masking cap or **the optimiser converging** — its objective is already
satisfied and it stops. Raising `--margin-db` cannot force a converged
optimiser to spend more.

So N5 runs **two** arms, and they answer the question from both sides.

### N5a — plain, rescaled to equal energy (no crafting)

Per speaker, scale the plain delta by a scalar so its waveform SNR against the
original equals that speaker's `fullnsT_m30_f15_nodvec` SNR, then re-run the
cloning evaluation unchanged.

* It is **exact** on the quantity in question: equal perturbation energy.
* It is **free of crafting noise** (§50: the loop does not reproduce itself at
  fixed seed, worst 0.0328) because it reuses the same delta.
* It is **generous to plain**: rescaling pushes some frames above the masking
  cap, so N5a's plain is *more* perturbed-sounding than any legitimate plain
  arm at that SNR. If it still loses, the magnitude explanation is dead.
* Its cost is the cloning evaluation only.

Quality (SNR, PESQ, and the per-frame floor violation rate) is reported for the
rescaled arm, because it is no longer psychoacoustically constrained and the
paper must say so.

### N5b — plain, re-crafted at a relaxed budget

Pilot on **dev-clean** (never the confirmatory 29, §RULES) over
`--snr-floor ∈ {15, 13, 12}` × `--margin-db ∈ {30, 33}`, 6 speakers, plain
variant, measuring realised SNR. Choose the single setting whose mean realised
SNR is nearest 16.13 dB, then craft that setting on the 29 test-clean speakers
and run the cloning evaluation.

* It is the **legitimate** comparison: a plain arm that genuinely occupies the
  same budget, still masking-constrained.
* If no setting reaches 16.13 dB because plain converges first, **that is the
  result** — "plain cannot spend the budget, because its objective is already
  satisfied" — and it is reported as such, not retried.

## 3. What is held fixed

Everything else matches the §59 held-out run byte for byte: `--craft-encoders
no-dvec`, 29 cross-session test-clean speakers, `--spk-offset 0 --utt-index 0`,
`--steps 200`, purifier v7 DDIM, `--thr-aware --weights 1,1,1`, cloning through
DiffVC against a different-chapter clean enrolment at the cross-session EER
thresholds, channels `none` / `ours` / `diffwave` (`--dw-t 3`).

## 4. Primary endpoint and analysis

**Primary:** the paired per-speaker margin, `fullnsT_m30_f15_nodvec` **minus**
the budget-matched plain arm, ECAPA, channel `ours`, n = 29 — the single cell
that carries §59c's headline. Paired t with a BCa interval.

**Secondary, pre-specified:** the same contrast on the other two encoders and
the other two channels (9 cells, Holm within the nine, reported for N5a and
N5b separately and never pooled); clone success rates with n; and the realised
SNR/PESQ of every arm, so the reader can check that the matching worked.

**The matching check comes first.** If the budget-matched arm's mean SNR is
not within **0.5 dB** of 16.13, the matching failed and the arm is reported as
a failed match, not as a comparison.

## 5. Outcome branches, fixed before the data

Read on the primary cell (ECAPA, `ours`), for each of N5a and N5b.

* **5-A — the effect survives at equal energy.** The purifier-aware arm is
  still significantly below budget-matched plain.
  → §59c's confound is settled. The paper states the effect with the matched
  comparison as the headline and §59f as a methods note.
* **5-B — the effect shrinks but survives in direction.** Point estimate still
  negative, no longer significant.
  → Report the attenuated effect and the matched numbers. The claim weakens to
  "part of the advantage is budget"; the abstract says so.
* **5-C — the effect disappears.** Budget-matched plain matches or beats the
  purifier-aware arm.
  → **§59c is a budget effect, not a method effect.** The lead claim is
  withdrawn, §59 is amended, and the defence paper falls back to §59a (the
  protection works) plus the measurement contributions. This is a real
  possibility and the reason the run is worth doing.
* **5-D — the matching failed** (§4's 0.5 dB check), or N5b's pilot cannot
  reach 16.13 dB.
  → For N5b that is itself reportable ("plain converges before it can spend
  the budget"). For N5a a matching failure is a bug: fix and re-run once.

**Where N5a and N5b disagree,** N5a is the strict upper bound on plain and N5b
is the legitimate arm. A 5-A on N5a with a 5-C on N5b would be incoherent and
would mean the rescaling is doing something other than changing energy; in that
case report both and claim neither.

**Pre-committed prediction, so it can be scored.** I expect **5-A**, at about
70 %, with 5-B at 20 % and 5-C at 10 %. Reasoning: the two indirect tests in
§59f both point away from magnitude, and the across-arm ordering is actually
inverted (the louder arm gains less). The 10 % on 5-C is not negligible —
2.89 dB is a large difference and no direct test has been run.

## 6. Files

| what | path |
|---|---|
| rescaling + quality check | `experiments/make_budget_matched_plain.py` |
| pilot | `experiments/run_n5_pilot.sh` → `results/n5_pilot.csv` |
| crafting + cloning | `experiments/run_n5.sh` |
| analysis | `experiments/analyse_n5.py` → `results/n5_analysis.txt` |

Output CSVs carry the arm tags `plain_m30_f15_nodvec_rescaled` (N5a) and
`plain_n5b_nodvec` (N5b) so neither can be confused with the §59 arms.
