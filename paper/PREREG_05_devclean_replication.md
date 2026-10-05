# Pre-registration — #5, out-of-sample replication of C2 on dev-clean

**Written 2026-10-05, before any dev-clean file is crafted.** PLAN_V3 W3–W5.
This document is the precondition for that run: PLAN_V3 requires "the 12 cells
and the outcome→action table committed before the first file is crafted", in the
§25 format — the outcomes written before the result is read.

The content hash of this file is recorded in every output row as `prereg_sha`
(`utils/provenance.py`, W0). A row whose `prereg_sha` does not match this file
was not produced under this pre-registration.

---

## 1. The question

C2 is the claim that **the purification-invariance constraint improves
protection specifically under purifier mismatch**, with a per-encoder pattern:
it helps WavLM-SV robustly, helps ECAPA directionally, and **reverses on the
Resemblyzer d-vector at mild purification**. RESULTS §27 establishes this on
test-clean, and §30e shows it is insensitive to three analysis choices of the
*same* sample.

It is currently **a hypothesis supported by one sample**. Every number comes
from the same 29 test-clean speakers, scored against thresholds calibrated on
those same 29 speakers (council #5). §30e is analysis-choice insensitivity, not
replication — RESULTS §31 is explicit that it must never be written as
"replicated three times".

This run asks one question: **does the §27 pattern appear in speakers we have
never crafted against, scored against thresholds we did not calibrate on them?**

---

## 2. Two corrections to PLAN_V3's description of this experiment

Found while writing this document, before any GPU time was spent.

**2.1 It is 31 speakers, not 40.** PLAN_V3 W3–W5 says "dev-clean (40 speakers,
unused)". dev-clean has 40 speakers with a ≥4 s reference utterance, but the
shipping protocol is **cross-session** (`--cross-session`, the shared reference
selector), and only **31** dev-clean speakers have usable cross-session
material. Every reported number in this paper uses cross-session, so this run
must too. **n = 31.** Any later text saying 40 is wrong.

**2.2 The disjoint calibration cohort is test-clean, not a dev-clean split.**
PLAN_V3 asks for "disjoint threshold calibration". Three options existed:

| option | cost |
|---|---|
| split dev-clean, e.g. 21 eval / 10 calibration | throws away a third of the sample; 10 speakers is too few to calibrate 4 foreign channels |
| calibrate on train-clean-100 | purifier v7 was **trained** on train-clean-100, so the channel would be calibrated on speakers the purifier has seen |
| **calibrate on test-clean (chosen)** | reuses the previously evaluated cohort, but the speaker sets are fully disjoint and neither is purifier-seen |

test-other would have been the cleanest third cohort and **is not on disk**
(only dev-clean, test-clean and train-clean-100 are extracted). The defect being
fixed is specifically "thresholds calibrated on the same speakers they score";
calibrating on test-clean's 29 and scoring dev-clean's 31 removes exactly that,
loses no sample, and keeps both cohorts unseen by purifier v7.

**Sensitivity check, pre-specified:** the same 12 cells re-scored against
thresholds calibrated on dev-clean itself. If the verdict changes between the
two scorings, **the verdict is "threshold-dependent" and neither scoring is
reported as the result.** This check is cheap (eval-only, no re-crafting).

**2.3 The clean-PCA control basis must not be dev-clean.** `core/matched_random.py`
defaults `--pca-subset dev-clean`, which was correct for a test-clean evaluation.
Any α or control measurement on **dev-clean** must pass `--pca-subset test-clean`,
or the control basis is built from the speakers it controls for. The script
raises if the two subsets are equal, but it cannot catch dev-clean-vs-dev-clean
across two separate invocations, so this is a standing instruction.

---

## 3. The design, fixed

| | |
|---|---|
| corpus | LibriSpeech **dev-clean**, cross-session, shared reference selector |
| speakers | **all 31** cross-session-usable, `--spk-offset 0`, no selection of any kind |
| recipe | **v1, unchanged** — `recipe-v1` tag, m30_f15 budget, `--seg-partials`, `--thr-aware`, `--weights 1,1,1`, `--craft-encoders all` |
| arms | **`full` and `fullnsT` only** (the contrast C2 is about). No `plain`, no `nsT`. |
| gradient | **BPDA** (`--grad-estimator bpda`). RESULTS §30: equivalent to exact on the constrained arm, and ~2× faster. |
| purifier | v7 for crafting; evaluation through the **foreign** DiffWave checkpoints |
| cells | 3 encoders × 2 foreign purifiers × 2 timesteps = **12** |
| thresholds | calibrated on **test-clean**, per channel, `*_xsess_tc` keys |
| seeds | crafting `--craft-seed 42`; evaluation seeds `2026,7` |
| `--prereg` | this file |

Nothing above may change after the first file is crafted. If a stall or crash
forces a restart, the restart uses identical flags; the skip-guard
(`experiments/skipguard.sh`, W0) refuses to merge rows from a different
`code_sha`.

---

## 4. Primary endpoint and estimator

> **Primary endpoint:** the mean paired per-speaker difference in
> `cos_after_purify − threshold` between the `fullnsT` and `full` arms
> (fullnsT − full), evaluated through the stated foreign purifier at the stated
> timestep, same speakers, same thresholds, same `--craft-seed`.
> **Unit:** speaker. **n = 31.**
> **Sign convention:** cosine to the victim is what the attacker wants, so
> **negative Δ = the constraint improves protection**; positive Δ = the
> reversal.
> **Estimator:** mean paired difference with a BCa bootstrap 95 % CI (2000
> resamples) and a Wilcoxon signed-rank p.
> **Multiplicity:** Holm over the **12 cells of this run** — one family, fixed
> here. Not m = 24; this run has a single crafting mode.
> **Cell-level significance requires both** (i) the 95 % CI excluding 0 and
> (ii) Holm-adjusted p < 0.05.

Per-cell intervals are marginal and carry no simultaneous guarantee; the Holm
column is the inferential statement, exactly as RESULTS §27b insists.

---

## 5. The 12 cells, with the test-clean point estimates as the predictions

From RESULTS §27a (segment recipe, m = 24 family). **These are the predictions.**

| # | purifier | t | encoder | test-clean Δ | predicted sign | §27 verdict there |
|---|---|---|---|---|---|---|
| 1 | DiffWave-LibriSpeech | 1 | WavLM-SV | −0.076 | **negative** | 25/29, p_adj<.001 |
| 2 | DiffWave-LibriSpeech | 1 | ECAPA-TDNN | −0.046 | negative | 19/29, p_adj=.054 |
| 3 | DiffWave-LibriSpeech | 1 | d-vector | **+0.071** | **positive (reversal)** | 3/29, p_adj<.001 |
| 4 | DiffWave-LibriSpeech | 3 | WavLM-SV | −0.026 | negative | 22/29, p_adj=.013 |
| 5 | DiffWave-LibriSpeech | 3 | ECAPA-TDNN | −0.029 | negative | 19/29, p_adj=.091 |
| 6 | DiffWave-LibriSpeech | 3 | d-vector | +0.014 | positive | 9/29, p_adj=.091 |
| 7 | DiffWave-SC09 | 1 | WavLM-SV | −0.078 | **negative** | 23/29, p_adj<.001 |
| 8 | DiffWave-SC09 | 1 | ECAPA-TDNN | −0.038 | negative | 18/29, p_adj=.098 |
| 9 | DiffWave-SC09 | 1 | d-vector | **+0.075** | **positive (reversal)** | 3/29, p_adj<.001 |
| 10 | DiffWave-SC09 | 3 | WavLM-SV | −0.046 | negative | 22/29, p_adj<.001 |
| 11 | DiffWave-SC09 | 3 | ECAPA-TDNN | −0.034 | **negative** | 22/29, p_adj=.014 |
| 12 | DiffWave-SC09 | 3 | d-vector | +0.014 | positive | 9/29, p_adj=.098 |

**The three load-bearing predictions**, stated separately so the verdict cannot
be reshaped afterwards:

* **P1 — WavLM helps in all four purifier×t cells** (cells 1, 4, 7, 10 negative).
* **P2 — the d-vector reverses at t = 1 on both purifiers** (cells 3 and 9
  positive). This is the prediction §32's α ordering explains and the one that
  makes C2 a mechanism rather than a curiosity.
* **P3 — ECAPA keeps the sign in all four cells** (2, 5, 8, 11 negative),
  without a significance requirement: §27 already found it survives Holm in only
  1 of 4.

**Power, stated in advance.** At n = 31 with the per-speaker SDs implied by
§27b's intervals (roughly 0.035–0.075), the detectable effect at 80 % power and
Holm m = 12 is about 0.030–0.045. **Cells 6 and 12 (+0.014) and cell 4 (−0.026)
are below that.** Their failure to reach significance is therefore
**uninformative** and must not be read as a failed replication. The verdict
below depends only on cells the run can actually resolve.

---

## 6. Outcome → action, written before the result

Exactly three outcomes. Each has one action, fixed now.

### Outcome A — replicated
**P1 holds (4/4 WavLM negative, ≥3 of 4 significant under Holm m=12), P2 holds
(both t=1 d-vector cells positive and significant), and P3 holds in ≥3 of 4.**

→ **C2 becomes a finding, replicated out of sample on disjoint speakers with
disjoint thresholds.** Write it as such. Report both samples side by side with
their CIs; do not pool them. This also retires council #5's "thresholds
calibrated on the same 29 speakers" for C2.

### Outcome B — directional only
**Signs are preserved where the run has power (P1 and P2 directionally, ≥3/4 and
2/2) but significance is substantially weaker than §27 — e.g. WavLM ≤2 of 4
significant.**

→ **C2 is "directionally reproduced out of sample, underpowered at n = 31".**
Report both samples and the per-cell CIs; state the power calculation from §5
above. **Do not write "replicated".** C2 stays a supported hypothesis, C1 keeps
the lead, and the limitation section says the confirmatory sample is 29 + 31
speakers from one corpus.

### Outcome C — not replicated
**P1 fails (≥2 WavLM cells non-negative), or P2 fails (either t=1 d-vector cell
non-positive).**

→ **C2 fails out of sample, and that is the reported result.** FRAME_v6 §6
pre-holds this branch. C2 is demoted to a single-sample observation about
test-clean, the paper leads with **C1 + C3**, and §27 is reported with an
explicit non-replication note. **No re-run at a different budget, timestep,
purifier or speaker subset, and no search for a cell where it does clear.** A
null here is a publishable finding about the fragility of mismatch effects, and
it is cheaper to report than to launder.

### Binding on all three outcomes
* The 12 cells are fixed. No cell may be added, dropped or swapped after
  results are seen.
* The primary endpoint is the margin. The binary clone rate is **not** an
  endpoint of this run (PLAN_V3 2a, council #4).
* If the sensitivity check in §2.2 flips the verdict, the verdict is
  "threshold-dependent" and neither scoring is reported as the result.
* A crash that leaves fewer than 28 of 31 speakers makes the run
  **inconclusive**, not a null.

---

## 7. What would invalidate the run

Each of these means the output is discarded, not reinterpreted.

1. `prereg_sha` in the output rows does not match this file's hash.
2. Any flag in §3 differs from what is recorded in the rows' `argv`.
3. The threshold file is not the test-clean cross-session calibration
   (`load_thresholds` must fail on a mismatched base — PLAN_V3 W2).
4. A dev-clean speaker appears in the crafting set that is not among the 31
   cross-session-usable speakers.
5. Rows from two different `code_sha` values land in one output file. The
   skip-guard refuses this; `SKIPGUARD_FORCE_SKIP=1` must **not** be used for
   this run's crafting stages.
6. Any GPU probe runs beside the crafting job (the OOM failure mode of
   2026-10-01).

---

## 8. Cost

~9 GPU-h at the PLAN_V3 estimate: 2 arms × 31 speakers, BPDA, plus eval-only
scoring through 4 foreign channels and the threshold calibration already on disk
for test-clean. Budget check before launch: this must fit before the **25 Nov
base freeze** with the W6 and W7 items still ahead of it.

---

## 9. Execution amendment — chunked crafting (added 2026-10-06, before any dev-clean file was crafted)

`exp_purifier_aware.py` writes its CSV once, at the end of a run, and has no
resume. Measured from the test-clean BPDA run (`main_table_tc_m30_seg_bpda.csv`)
the mean crafting time is **122 s** per speaker on `full` and **139 s** on
`fullnsT`, so 31 speakers × 2 arms is roughly **2.5–3.5 GPU-h including
per-file evaluation** — longer than a single uninterrupted execution window is
guaranteed to be here. A kill at 2 h would lose an entire arm.

**Amendment.** Each arm is crafted in four consecutive chunks by speaker
offset, `--spk-offset {0, 8, 16, 24}` with `--n-files {8, 8, 8, 7}`, each
writing its own CSV, concatenated afterwards.

**Why this changes nothing that matters.**

* `list_files` applies `--spk-offset` to the **cross-session-filtered, sorted**
  speaker list (`keep[args.spk_offset:]`, then the first `--n-files`), so the
  four chunks tile the 31 speakers exactly once, in order, with no gap and no
  overlap. Verified against the speaker list before launch.
* Crafting is **per-file deterministic and independent of loop position**:
  `craft()` re-seeds on entry (`torch.manual_seed(seed)`, `gen` at `seed+1000`,
  `gl_gen` at `seed+2000`) with the same `--craft-seed 42` for every file, so a
  file's δ does not depend on which chunk it was crafted in, or on how many
  files preceded it.
* All four chunks run the same `code_sha`, which every row records, so §7.5
  still applies: the concatenation is checked to carry exactly one `code_sha`
  and one `prereg_sha`, and is rejected otherwise.

**What this does change, declared.** §3 records `--spk-offset 0`; the rows will
record offsets 0/8/16/24 in their `argv`. §7.2 ("any flag in §3 differs from
what is recorded in `argv`") is therefore waived **for `--spk-offset` and
`--n-files` only**, and only because their union is the full, unselected set of
31. Every other flag in §3 is unchanged and still binding. No speaker is
selected, dropped or reordered by this, and the chunk boundaries were fixed
here — before any result was seen — rather than chosen afterwards.

**Union check, run before the concatenation is accepted:** the 31 speakers in
the combined CSV must equal the 31 cross-session-usable dev-clean speakers
exactly. A mismatch means the run is discarded under §7.4.

---

## 10. Incident: two `code_sha` values in one run (2026-10-06, before any cell was scored)

**What happened.** Crafting completed all eight chunks cleanly (31 speakers × 2
arms, 02:19 wall clock). The assembler then **refused**, under §7.5, because the
rows carry two `code_sha` values: `2fd23a02ea02` on the `full` arm's first chunk
(8 speakers) and `79aae4857cc3` on everything else.

**Why.** Five commits landed between chunk 1 and chunk 2 — while the GPU was
busy, analysis scripts and write-ups were being committed. `code_sha` is
computed over `utils/provenance.CODE_DIRS`, which includes the whole of
`experiments/`, so committing an *analysis* script, or even a results CSV,
moves it. That is an own goal: §7.6 forbids a second GPU job beside the
crafting, and the spirit of §7.5 is "do not merge arms computed by two
versions", but nothing said "do not commit during the run". **It does now.**

**What was checked, rather than assumed.** The diff between the two shas is:

```
experiments/RESULTS_ICASSP.md                     (prose)
experiments/analyse_alpha_pairs.py                (new, analysis only)
experiments/analyse_devclean.py                   (analysis only)
experiments/results/alpha_pairs.csv               (output)
experiments/results/encoder_subspace_xsess_fe.csv (output)
experiments/run_devclean_sensitivity.sh           (new, runs after this)
paper/FRAME_v6.md                                 (prose)
```

Nothing on the crafting import path. That path was enumerated empirically — by
importing `experiments/exp_purifier_aware.py` and listing every repo module
that ended up in `sys.modules` — and it is 21 files under `core/`, `models/`,
`utils/`, `evaluation/`, plus `exp_purifier_aware.py` and `multi_encoders.py`.
None appears in the diff. The crafted δ for any file therefore cannot depend on
which of the two shas produced it, and §3's recipe is untouched.

**How it was resolved.** Not by a manual override. `assemble_devclean.py` gained
`--verify-multi-sha`, which **diffs every pair of shas and refuses unless every
changed path is classifiable as unable to affect crafting** — a path the
classifier does not recognise is a refusal, so it fails closed. The assembly
was accepted under that check and nowhere else. Data integrity was verified
independently: 558 rows = 31 × 2 × 3 encoders × 3 `eval_t`, both arms covering
all 31 speakers, one `prereg_sha` matching this file.

**What this does and does not license.** It licenses this assembly, on the
evidence above. It does not weaken §7.5 for anything else: a diff that touches
`core/`, `models/`, `utils/`, `evaluation/`, `exp_purifier_aware.py` or
`multi_encoders.py` still discards the run, and `SKIPGUARD_FORCE_SKIP` was not
used and remains forbidden for the crafting stages.

**Standing rule added, for the rest of this project.** No commits while a
crafting job is running. Analysis written during a run is held in the working
tree and committed after the job's last chunk, so the sha cannot move
underneath it.

**No cell had been scored when this was written.** The four foreign-purifier
blocks and the control had not yet run, so nothing in this resolution could
have been shaped by a result.
